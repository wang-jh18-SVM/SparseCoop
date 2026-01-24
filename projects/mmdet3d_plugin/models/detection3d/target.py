import torch
import numpy as np
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from mmdet.core.bbox.builder import BBOX_SAMPLERS

from projects.mmdet3d_plugin.core.box3d import *
from ..base_target import BaseTargetWithDenoising


__all__ = ["SparseBox3DTarget"]


@BBOX_SAMPLERS.register_module()
class SparseBox3DTarget(BaseTargetWithDenoising):
    def __init__(
        self,
        cls_weight=2.0,
        alpha=0.25,
        gamma=2,
        eps=1e-12,
        box_weight=0.25,
        reg_weights=None,
        cls_wise_reg_weights=None,
        num_dn_groups=0,
        dn_noise_scale=0.5,
        max_dn_gt=32,
        add_neg_dn=True,
        num_temp_dn_groups=0,
    ):
        super(SparseBox3DTarget, self).__init__(num_dn_groups, num_temp_dn_groups)
        self.cls_weight = cls_weight
        self.box_weight = box_weight
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.reg_weights = reg_weights
        if self.reg_weights is None:
            self.reg_weights = [1.0] * 8 + [0.0] * 2
        self.cls_wise_reg_weights = cls_wise_reg_weights
        self.dn_noise_scale = dn_noise_scale
        self.max_dn_gt = max_dn_gt
        self.add_neg_dn = add_neg_dn

    def encode_reg_target(self, box_target, device=None):
        """Encode ground truth 3D boxes to match the model's prediction format.

        This method converts ground truth boxes from their raw annotation format
        to the encoded format used by the model for regression. The encoding helps
        with training stability and numerical properties.

        Args:
            box_target (List[torch.Tensor]): List of raw ground truth boxes per batch
                Each tensor has shape [N_gt_i, box_dim] with format:
                [x, y, z, l, w, h, yaw, vx, vy, ...] (raw world coordinates)
            device (torch.device, optional): Target device for the output tensors

        Returns:
            List[torch.Tensor]: List of encoded ground truth boxes per batch
                Each tensor has shape [N_gt_i, encoded_dim] with format:
                [x, y, z, log(l), log(w), log(h), sin(yaw), cos(yaw), vx, vy, ...]
        """
        outputs = []
        for box in box_target:
            # ========== BOX PARAMETER ENCODING ==========
            # Transform box parameters to improve training stability and convergence

            output = torch.cat(
                [
                    # **Position coordinates**: Keep as-is (x, y, z)
                    # These represent the center position of the box in world coordinates
                    # No encoding needed as they are already in a reasonable range
                    box[..., [X, Y, Z]],  # [N_gt, 3]
                    # **Size parameters**: Apply log transform (l, w, h) → (log(l), log(w), log(h))
                    # This encoding provides several benefits:
                    # 1. Numerical stability: prevents very small/large values
                    # 2. Scale invariance: log(2*size) = log(2) + log(size)
                    # 3. Better gradient flow: log derivative scales inversely with magnitude
                    # 4. Symmetric error: log(pred/gt) = log(pred) - log(gt)
                    box[..., [L, W, H]].log(),  # [N_gt, 3]
                    # **Orientation**: Convert angle to (sin, cos) representation
                    # This encoding addresses the circular nature of angles:
                    # 1. Avoids discontinuity at 2π/0 boundary (359° vs 1° are similar)
                    # 2. Provides smooth gradients everywhere
                    # 3. sin/cos are bounded in [-1, 1] for numerical stability
                    # 4. Model learns both components to fully represent orientation
                    torch.sin(box[..., YAW]).unsqueeze(-1),  # [N_gt, 1] - sin(yaw)
                    torch.cos(box[..., YAW]).unsqueeze(-1),  # [N_gt, 1] - cos(yaw)
                    # **Velocity and other parameters**: Keep as-is (vx, vy, ...)
                    # Velocity is already in a reasonable range for most applications
                    # Additional parameters (if any) are passed through unchanged
                    box[..., YAW + 1 :],  # [N_gt, remaining_dims]
                ],
                dim=-1,
            )  # [N_gt, encoded_dim]

            # Move to target device if specified
            if device is not None:
                output = output.to(device=device)
            outputs.append(output)

        return outputs

    def sample(self, cls_pred, box_pred, cls_target, box_target):
        """Hungarian matching-based target assignment for 3D object detection.

        This method implements the core target assignment strategy for DETR-style
        object detection, where we need to assign ground truth objects to prediction
        queries in an optimal way. The key challenge is that we have a set of N
        prediction queries and M ground truth objects, and we need to find the best
        1-to-1 matching between them.

        Args:
            cls_pred (torch.Tensor): [B, N_query, N_cls] - Classification predictions from model
            box_pred (torch.Tensor): [B, N_query, box_dim] - Box regression predictions from model
            cls_target (List[torch.Tensor]): List of [N_gt_i] - Ground truth class labels per batch
            box_target (List[torch.Tensor]): List of [N_gt_i, box_dim=9] - Ground truth boxes per batch

        Returns:
            Tuple containing:
                - output_cls_target: [B, N_query] - Assigned class targets (num_cls=background)
                - output_box_target: [B, N_query, box_dim] - Assigned regression targets
                - output_reg_weights: [B, N_query, box_dim] - Per-parameter regression weights
        """
        bs, num_pred, num_cls = cls_pred.shape

        # ========== STEP 1: COMPUTE CLASSIFICATION COST MATRIX ==========
        # Calculate the cost of assigning each prediction to each ground truth based on classification
        # Returns: List[torch.Tensor] of shape [N_query, N_gt_i] for each batch item
        cls_cost = self._cls_cost(cls_pred, cls_target)

        # ========== STEP 2: ENCODE GROUND TRUTH BOXES ==========
        # Convert ground truth boxes from raw format to the same encoding as predictions
        # Raw format: [x, y, z, l, w, h, yaw, vx, vy, vz] (world coordinates)
        # Encoded format: [x, y, z, log(l), log(w), log(h), sin(yaw), cos(yaw), vx, vy]
        box_target = self.encode_reg_target(box_target, box_pred.device)

        # ========== STEP 3: COMPUTE REGRESSION WEIGHTS ==========
        # Generate per-parameter weights to handle invalid/missing ground truth data
        instance_reg_weights = []
        for i in range(len(box_target)):
            # Base weights: 1.0 for valid parameters, 0.0 for NaN parameters
            # This handles cases where some ground truth annotations are incomplete
            # (e.g., missing velocity information for static objects)
            weights = torch.logical_not(box_target[i].isnan()).to(
                dtype=box_target[i].dtype
            )  # [N_gt_i, box_dim]

            # Apply class-wise regression weights if specified
            # Some object classes might need different regression emphasis
            # (e.g., traffic cones have zero weight on orientation)
            if self.cls_wise_reg_weights is not None:
                for cls, weight in self.cls_wise_reg_weights.items():
                    weights = torch.where(
                        (cls_target[i] == cls)[:, None],  # Broadcast class mask
                        weights.new_tensor(weight),  # Apply class-specific weight
                        weights,  # Keep original weight
                    )
            instance_reg_weights.append(weights)

        # ========== STEP 4: COMPUTE BOX REGRESSION COST MATRIX ==========
        # Calculate the cost of assigning each prediction to each ground truth based on box regression
        # Returns: List[torch.Tensor] of shape [N_query, N_gt_i] for each batch item
        box_cost = self._box_cost(box_pred, box_target, instance_reg_weights)

        # ========== STEP 5: SOLVE HUNGARIAN MATCHING ==========
        # Combine classification and regression costs, then solve optimal assignment
        indices = []
        for i in range(bs):
            if cls_cost[i] is not None and box_cost[i] is not None:
                # Combine both cost components into total cost matrix
                cost = (
                    (cls_cost[i] + box_cost[i]).detach().cpu().numpy()
                )  # [N_query, N_gt_i]

                # Handle numerical issues (inf/nan values) by replacing with large cost
                cost = np.where(np.isneginf(cost) | np.isnan(cost), 1e8, cost)

                # Apply Hungarian algorithm to find optimal assignment
                # Returns: (row_indices, col_indices) representing the optimal matching
                # - row_indices: which predictions are matched
                # - col_indices: which ground truths they are matched to
                assign = linear_sum_assignment(cost)
                indices.append(
                    [cls_pred.new_tensor(x, dtype=torch.int64) for x in assign]
                )
            else:
                # No ground truth objects in this batch item
                indices.append([None, None])

        # ========== STEP 6: GENERATE ASSIGNMENT TARGETS ==========
        # Create final target tensors based on the Hungarian matching results

        # Initialize all predictions as background (class = num_cls)
        output_cls_target = (
            cls_target[0].new_ones([bs, num_pred], dtype=torch.long) * num_cls
        )  # [B, N_query] - initially all background

        # Initialize regression targets and weights as zeros
        output_box_target = box_pred.new_zeros(box_pred.shape)  # [B, N_query, box_dim]
        output_reg_weights = box_pred.new_zeros(box_pred.shape)  # [B, N_query, box_dim]

        # Apply the Hungarian matching results
        for i, (pred_idx, target_idx) in enumerate(indices):
            if len(cls_target[i]) == 0:
                continue  # No ground truth objects in this batch item

            # For matched predictions, assign the corresponding ground truth
            # pred_idx: indices of matched prediction queries
            # target_idx: indices of matched ground truth objects
            output_cls_target[i, pred_idx] = cls_target[i][
                target_idx
            ]  # Assign GT class
            output_box_target[i, pred_idx] = box_target[i][target_idx]  # Assign GT box
            output_reg_weights[i, pred_idx] = instance_reg_weights[i][
                target_idx
            ]  # Assign weights

        # Final output interpretation:
        # - Matched queries (positive samples): have ground truth class and meaningful regression targets
        # - Unmatched queries (background samples): have background class and zero regression targets (only contribute to classification loss)
        return output_cls_target, output_box_target, output_reg_weights

    def _cls_cost(self, cls_pred, cls_target):
        """Compute classification cost matrix for Hungarian matching.

        This method calculates the cost of assigning each prediction query to each
        ground truth object based on classification predictions. The cost is inspired
        by focal loss formulation to handle class imbalance.

        Args:
            cls_pred (torch.Tensor): [B, N_query, N_cls] - Raw classification logits
            cls_target (List[torch.Tensor]): List of [N_gt_i] - Ground truth class labels per batch

        Returns:
            List[torch.Tensor]: Classification cost matrices [N_query, N_gt_i] for each batch item
        """
        bs = cls_pred.shape[0]
        cls_pred = cls_pred.sigmoid()  # Convert logits to probabilities [0, 1]
        cost = []

        for i in range(bs):
            if len(cls_target[i]) > 0:
                # ========== FOCAL LOSS-BASED COST CALCULATION ==========
                # We compute the focal loss cost for assigning each query to each ground truth

                # **Negative cost**: Cost of predicting this class as background
                # This represents the loss if we DON'T assign this query to this ground truth
                # Formula: -log(1-p) * (1-alpha) * p^gamma
                neg_cost = (
                    -(1 - cls_pred[i] + self.eps).log()
                    * (1 - self.alpha)
                    * cls_pred[i].pow(self.gamma)
                )  # [N_query, N_cls]

                # **Positive cost**: Cost of predicting this class as the ground truth class
                # This represents the loss if we DO assign this query to this ground truth
                # Formula: -log(p) * alpha * (1-p)^gamma
                pos_cost = (
                    -(cls_pred[i] + self.eps).log()
                    * self.alpha
                    * (1 - cls_pred[i]).pow(self.gamma)
                )  # [N_query, N_cls]

                # **Final cost**: Difference between positive and negative costs
                # Cost = pos_cost - neg_cost (for the target class only)
                # - NEGATIVE cost (pos_cost < neg_cost): FAVOR assignment
                # - POSITIVE cost (pos_cost > neg_cost): AVOID assignment

                # Select costs only for the ground truth classes
                cost.append(
                    (
                        pos_cost[:, cls_target[i]] - neg_cost[:, cls_target[i]]
                    )  # [N_query, N_gt_i]
                    * self.cls_weight  # Scale by classification weight hyperparameter
                )
            else:
                # No ground truth objects in this batch item
                cost.append(None)

        return cost

    def _box_cost(self, box_pred, box_target, instance_reg_weights):
        """Compute box regression cost matrix for Hungarian matching.

        This method calculates the cost of assigning each prediction query to each
        ground truth object based on box regression predictions. The cost represents
        how well each predicted box aligns with each ground truth box.

        Args:
            box_pred (torch.Tensor): [B, N_query, box_dim] - Predicted box parameters
            box_target (List[torch.Tensor]): List of [N_gt_i, box_dim] - Encoded ground truth boxes per batch
            instance_reg_weights (List[torch.Tensor]): List of [N_gt_i, box_dim] - Per-parameter weights per batch

        Returns:
            List[torch.Tensor]: Box regression cost matrices [N_query, N_gt_i] for each batch item
        """
        bs = box_pred.shape[0]
        cost = []

        for i in range(bs):
            if len(box_target[i]) > 0:
                # ========== L1 DISTANCE-BASED COST CALCULATION ==========
                # Compute L1 distance between each prediction and each ground truth

                # **L1 distance computation**:
                # abs(pred - gt) gives element-wise absolute differences
                # box_pred[i]: [N_query, box_dim] → [N_query, 1, box_dim] (add GT dimension)
                # box_target[i]: [N_gt_i, box_dim] → [1, N_gt_i, box_dim] (add query dimension)
                l1_distance = torch.abs(
                    box_pred[i, :, None] - box_target[i][None]
                )  # [N_query, N_gt_i, box_dim]

                # **Apply per-instance regression weights**:
                # instance_reg_weights[i]: [N_gt_i, box_dim] → [1, N_gt_i, box_dim] (add query dimension)
                # These weights handle invalid/NaN ground truth parameters and class-wise weighting
                # - 1.0 for valid parameters, 0.0 for invalid/NaN parameters
                # - Can have class-specific scaling
                weighted_distance = (
                    l1_distance * instance_reg_weights[i][None]
                )  # [N_query, N_gt_i, box_dim]

                # **Apply global parameter weights**:
                # self.reg_weights: [box_dim] - global weights for each box parameter
                # This allows selectively weighting different aspects of the box
                global_weighted_distance = weighted_distance * box_pred.new_tensor(
                    self.reg_weights
                )  # [N_query, N_gt_i, box_dim]

                # **Sum across box parameters and apply box weight**:
                # Sum over the box_dim to get total cost per (query, gt) pair
                total_cost = torch.sum(
                    global_weighted_distance, dim=-1
                )  # [N_query, N_gt_i]

                cost.append(
                    total_cost * self.box_weight  # Scale by box weight hyperparameter
                )
            else:
                # No ground truth objects in this batch item
                cost.append(None)

        return cost

    def get_dn_anchors(self, cls_target, box_target, gt_instance_id=None):
        """Generate denoising anchors for robust training.

        Denoising training is a technique to improve model robustness and convergence
        by training the model to "denoise" corrupted object queries back to ground truth.
        This method creates noisy versions of ground truth objects and sets up the
        training targets for the model to learn to correct these noisy inputs.

        The key idea is:
        1. Take ground truth objects and add controlled noise to them
        2. Train the model to predict the original ground truth from these noisy inputs
        3. This helps the model become more robust and converge faster

        Args:
            cls_target (List[torch.Tensor]): Ground truth class labels per batch
            box_target (List[torch.Tensor]): Ground truth boxes per batch
            gt_instance_id (List[torch.Tensor], optional): Ground truth instance IDs for temporal tracking

        Returns:
            Tuple containing denoising training data:
                - dn_anchor: [B, N_dn, box_dim] - Noisy anchor boxes
                - dn_box_target: [B, N_dn, box_dim] - Regression targets to denoise back to GT
                - dn_cls_target: [B, N_dn] - Classification targets for denoising
                - attn_mask: [N_dn, N_dn] - Attention mask to prevent information leakage
                - valid_mask: [B, N_dn] - Mask for valid denoising samples (not padding)
                - dn_id_target: [B, N_dn] - Instance IDs for temporal denoising (if enabled)
        """
        # Early exit if denoising is disabled
        if self.num_dn_groups <= 0:
            return None
        # Disable instance tracking if temporal denoising is not used
        if self.num_temp_dn_groups <= 0:
            gt_instance_id = None

        # ========== STEP 1: LIMIT NUMBER OF GROUND TRUTH OBJECTS ==========
        # Prevent memory explosion when scenes have too many objects
        if self.max_dn_gt > 0:
            cls_target = [x[: self.max_dn_gt] for x in cls_target]
            box_target = [x[: self.max_dn_gt] for x in box_target]
            if gt_instance_id is not None:
                gt_instance_id = [x[: self.max_dn_gt] for x in gt_instance_id]

        # ========== STEP 2: PAD TO CONSISTENT BATCH SIZE ==========
        # Find maximum number of objects across the batch
        max_dn_gt = max([len(x) for x in cls_target])
        if max_dn_gt == 0:
            return None  # No objects to create denoising anchors from

        # Pad all batch items to have the same number of objects
        # Pad class with -1
        cls_target = torch.stack(
            [F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1) for x in cls_target]
        )  # [B, max_dn_gt]

        # Encode and pad box targets
        box_target = self.encode_reg_target(box_target, cls_target.device)
        box_target = torch.stack(
            [F.pad(x, (0, 0, 0, max_dn_gt - x.shape[0])) for x in box_target]
        )  # [B, max_dn_gt, box_dim]
        # Set padded box regions to zero (where class_target == -1)
        box_target = torch.where(
            cls_target[..., None] == -1, box_target.new_tensor(0), box_target
        )

        # Pad instance IDs with -1 if temporal denoising is enabled
        if gt_instance_id is not None:
            gt_instance_id = torch.stack(
                [
                    F.pad(x, (0, max_dn_gt - x.shape[0]), value=-1)
                    for x in gt_instance_id
                ]
            )  # [B, max_dn_gt]

        # ========== STEP 3: REPLICATE FOR MULTIPLE DENOISING GROUPS ==========
        # Create multiple copies of the same objects to later add different noise
        # Each denoising group gets independently sampled noise from the same distribution
        # This increases training diversity and robustness
        bs, num_dn_per_group, state_dims = box_target.shape
        if self.num_dn_groups > 1:
            cls_target = cls_target.tile(
                self.num_dn_groups, 1
            )  # [B*num_dn_groups, max_dn_gt]
            box_target = box_target.tile(
                self.num_dn_groups, 1, 1
            )  # [B*num_dn_groups, max_dn_gt, box_dim]
            if gt_instance_id is not None:
                gt_instance_id = gt_instance_id.tile(
                    self.num_dn_groups, 1
                )  # [B*num_dn_groups, max_dn_gt]

        # ========== STEP 4: ADD NOISE TO CREATE DENOISING ANCHORS ==========
        # Generate random noise in range [-1, 1] and scale by noise factor
        noise = torch.rand_like(box_target) * 2 - 1  # [-1, 1] uniform random
        noise *= box_target.new_tensor(
            self.dn_noise_scale
        )  # Scale by noise factor (e.g., 0.5)
        dn_anchor = box_target + noise  # [B*num_dn_groups, max_dn_gt, box_dim]

        # ========== STEP 5: ADD HARD NEGATIVE DENOISING SAMPLES (OPTIONAL) ==========
        # Create additional noisy samples that are farther from ground truth
        # This helps the model learn to distinguish between good and bad predictions
        if self.add_neg_dn:
            # Generate stronger noise for hard negative denoising samples (range [1, 2] or [-2, -1])
            noise_neg = torch.rand_like(box_target) + 1  # [1, 2] range
            flag = torch.where(
                torch.rand_like(box_target) > 0.5,
                noise_neg.new_tensor(1),
                noise_neg.new_tensor(-1),
            )  # Randomly flip sign
            noise_neg *= flag  # [-2, -1] or [1, 2]
            noise_neg *= box_target.new_tensor(
                self.dn_noise_scale
            )  # Scale by noise factor (e.g., 0.5)

            # Concatenate positive and hard negative denoising samples
            dn_anchor = torch.cat([dn_anchor, box_target + noise_neg], dim=1)
            num_dn_per_group *= 2  # Double the number of denoising samples

        # ========== STEP 6: ASSIGN DENOISING TARGETS VIA HUNGARIAN MATCHING ==========
        # Use box cost to optimally assign noisy anchors back to ground truth
        # This ensures each noisy sample has a clear target to learn towards
        box_cost = self._box_cost(dn_anchor, box_target, torch.ones_like(box_target))

        # Initialize denoising targets
        dn_box_target = torch.zeros_like(dn_anchor)  # Regression targets
        dn_cls_target = (
            -torch.ones_like(cls_target) * 3
        )  # Set -3 for all samples initially
        if gt_instance_id is not None:
            dn_id_target = -torch.ones_like(gt_instance_id)  # Instance ID targets

        # Double the targets if hard negative denoising is used
        if self.add_neg_dn:
            dn_cls_target = torch.cat([dn_cls_target, dn_cls_target], dim=1)
            if gt_instance_id is not None:
                dn_id_target = torch.cat([dn_id_target, dn_id_target], dim=1)

        # Apply Hungarian matching to assign each noisy anchor to a ground truth
        for i in range(dn_anchor.shape[0]):
            cost = box_cost[i].cpu().numpy()  # [N_noisy, N_gt]
            anchor_idx, gt_idx = linear_sum_assignment(cost)  # Optimal assignment
            anchor_idx = dn_anchor.new_tensor(anchor_idx, dtype=torch.int64)
            gt_idx = dn_anchor.new_tensor(gt_idx, dtype=torch.int64)

            # Assign targets based on matching
            dn_box_target[i, anchor_idx] = box_target[i, gt_idx]  # Regression target
            dn_cls_target[i, anchor_idx] = cls_target[
                i, gt_idx
            ]  # Classification target
            if gt_instance_id is not None:
                dn_id_target[i, anchor_idx] = gt_instance_id[
                    i, gt_idx
                ]  # Instance ID target

        # ========== STEP 7: RESHAPE FOR FINAL OUTPUT ==========
        # Reshape from (B*num_dn_groups, num_dn_per_group, ...) to (B, num_dn_groups*num_dn_per_group, ...)
        dn_anchor = (
            dn_anchor.reshape(self.num_dn_groups, bs, num_dn_per_group, state_dims)
            .permute(1, 0, 2, 3)  # [B, num_dn_groups, num_dn_per_group, state_dims]
            .flatten(1, 2)  # [B, num_dn_groups*num_dn_per_group, state_dims]
        )
        dn_box_target = (
            dn_box_target.reshape(self.num_dn_groups, bs, num_dn_per_group, state_dims)
            .permute(1, 0, 2, 3)
            .flatten(1, 2)
        )
        dn_cls_target = (
            dn_cls_target.reshape(self.num_dn_groups, bs, num_dn_per_group)
            .permute(1, 0, 2)  # [B, num_dn_groups, num_dn_per_group]
            .flatten(1)  # [B, num_dn_groups*num_dn_per_group]
        )
        if gt_instance_id is not None:
            dn_id_target = (
                dn_id_target.reshape(self.num_dn_groups, bs, num_dn_per_group)
                .permute(1, 0, 2)
                .flatten(1)
            )
        else:
            dn_id_target = None

        # ========== STEP 8: CREATE VALIDITY MASK AND ATTENTION MASK ==========

        # **Create validity mask**: Identifies which denoising samples are valid (not padded)
        valid_mask = dn_cls_target >= 0  # Valid samples have non-negative class targets

        # **Handle hard negative denoising samples**: Update validity mask for hard negative denoising
        if self.add_neg_dn:
            # Reconstruct original class targets for hard negative denoising validation
            cls_target = (
                torch.cat(
                    [cls_target, cls_target], dim=1
                )  # Duplicate for pos + hard neg DN
                .reshape(self.num_dn_groups, bs, num_dn_per_group)
                .permute(1, 0, 2)
                .flatten(1)
            )

            # A denoising sample is valid (not from padding) if:
            # 1. It has a positive class target (dn_cls_target >= 0), OR
            # 2. It's a hard negative denoising sample from a real object (cls_target >= 0 & dn_cls_target == -3)
            valid_mask = torch.logical_or(
                valid_mask, ((cls_target >= 0) & (dn_cls_target == -3))
            )

        # **Create attention mask**: Prevents information leakage between denoising groups
        # Key principle: Each denoising group should operate independently

        # Initialize mask as all-ones (block all attention by default)
        attn_mask = dn_box_target.new_ones(
            num_dn_per_group * self.num_dn_groups, num_dn_per_group * self.num_dn_groups
        )  # [total_dn_samples, total_dn_samples]

        # Allow attention within each denoising group (set to 0 = allow attention)
        for i in range(self.num_dn_groups):
            start = num_dn_per_group * i  # Start index of group i
            end = start + num_dn_per_group  # End index of group i
            attn_mask[start:end, start:end] = 0  # Allow intra-group attention

        # Convert to boolean mask: True = block attention, False = allow attention
        attn_mask = attn_mask == 1

        # Final structure of attention mask:
        # - Block size: [num_dn_per_group, num_dn_per_group] for each group
        # - Diagonal blocks (within groups): False (allow attention)
        # - Off-diagonal blocks (between groups): True (block attention)
        #
        # Example with 2 groups, 3 objects per group:
        # [False, False, False,  True,  True,  True]  # Group 0 can attend to Group 0
        # [False, False, False,  True,  True,  True]
        # [False, False, False,  True,  True,  True]
        # [ True,  True,  True, False, False, False]  # Group 1 can attend to Group 1
        # [ True,  True,  True, False, False, False]
        # [ True,  True,  True, False, False, False]

        # Ensure class targets are integers for loss computation
        dn_cls_target = dn_cls_target.long()

        return (
            dn_anchor,  # [B, N_dn, box_dim] - Noisy anchor boxes for denoising training
            dn_box_target,  # [B, N_dn, box_dim] - Regression targets to denoise back to GT
            dn_cls_target,  # [B, N_dn] - Classification targets for denoising samples
            attn_mask,  # [N_dn, N_dn] - Attention mask preventing inter-group leakage
            valid_mask,  # [B, N_dn] - Validity mask for non-padding samples
            dn_id_target,  # [B, N_dn] - Instance IDs for tracking objects temporally (optional)
        )

    def update_dn(
        self,
        instance_feature,
        anchor,
        dn_reg_target,
        dn_cls_target,
        valid_mask,
        dn_id_target,
        num_normal_anchor,
        temporal_valid_mask,
    ):
        """
        Updates denoising (DN) targets across frames to maintain temporal consistency.
        1. **Temporal Instance Matching**: Uses instance IDs to match denoising samples
           between consecutive frames, ensuring the same real object maintains consistent
           denoising targets over time.

        2. **Target Propagation**: Updates current frame's denoising targets based on
           previous frame's targets for matched instances, creating temporal coherence.

        3. **Adaptive Integration**: Handles varying numbers of objects between frames
           and selectively applies temporal updates based on validity masks.

        Args:
            instance_feature (torch.Tensor): [B, N_total, C] - Combined instance features
                (normal queries + denoising queries)
            anchor (torch.Tensor): [B, N_total, 11] - Combined anchor boxes
                (normal anchors + denoising anchors)
            dn_reg_target (torch.Tensor): [B, N_dn, box_dim] - Current frame DN regression targets
            dn_cls_target (torch.Tensor): [B, N_dn] - Current frame DN classification targets
            valid_mask (torch.Tensor): [B, N_dn] - Mask for valid DN samples (not padding)
            dn_id_target (torch.Tensor): [B, N_dn] - Instance IDs for DN samples
            num_normal_anchor (int): Number of normal (non-DN) queries
            temporal_valid_mask (torch.Tensor): [B] - Mask indicating which batch items
                have valid temporal information from previous frame

        Returns:
            Tuple of updated tensors:
                - instance_feature: [B, N_total, C] - Updated combined instance features
                - anchor: [B, N_total, 11] - Updated combined anchor boxes
                - dn_reg_target: [B, N_dn, box_dim] - Updated DN regression targets
                - dn_cls_target: [B, N_dn] - Updated DN classification targets
                - valid_mask: [B, N_dn] - Updated validity mask
                - dn_id_target: [B, N_dn] - Updated instance IDs
        """

        # Get batch size and total number of anchors (normal + denoising)
        bs, num_anchor = instance_feature.shape[:2]

        # ========== EARLY EXIT CONDITIONS ==========
        # Skip temporal denoising update if:
        # 1. No temporal validity information available
        # 2. No cached denoising metadata from previous frame
        # 3. No denoising queries present (all queries are normal)

        if temporal_valid_mask is None:
            # No temporal information available - clear cached DN metadata
            self.dn_metas = None

        if self.dn_metas is None or num_normal_anchor >= num_anchor:
            # Either no previous DN cache OR no DN queries in current frame
            # Return original inputs unchanged
            return (
                instance_feature,
                anchor,
                dn_reg_target,
                dn_cls_target,
                valid_mask,
                dn_id_target,
            )

        # ========== STEP 1: SPLIT CURRENT FRAME DATA ==========
        # Separate normal detection queries from denoising queries

        num_dn = num_anchor - num_normal_anchor  # Number of denoising queries

        # Extract denoising parts (last num_dn queries)
        dn_instance_feature = instance_feature[:, -num_dn:]  # [B, N_dn, C]
        dn_anchor = anchor[:, -num_dn:]  # [B, N_dn, 11]

        # Extract normal parts (first num_normal_anchor queries)
        instance_feature = instance_feature[:, :num_normal_anchor]  # [B, N_normal, C]
        anchor = anchor[:, :num_normal_anchor]  # [B, N_normal, 11]

        # ========== STEP 2: RESHAPE INTO DENOISING GROUPS ==========
        # Denoising queries are organized into groups for training diversity
        # Reshape: [B, total_dn] → [B, num_groups, samples_per_group]

        num_dn_groups = self.num_dn_groups  # Number of denoising groups
        num_dn = num_dn // num_dn_groups  # Samples per group

        # Reshape all current frame denoising tensors into group structure
        dn_feat = dn_instance_feature.reshape(bs, num_dn_groups, num_dn, -1)
        dn_anchor = dn_anchor.reshape(bs, num_dn_groups, num_dn, -1)
        dn_reg_target = dn_reg_target.reshape(bs, num_dn_groups, num_dn, -1)
        dn_cls_target = dn_cls_target.reshape(bs, num_dn_groups, num_dn)
        valid_mask = valid_mask.reshape(bs, num_dn_groups, num_dn)

        if dn_id_target is not None:
            dn_id = dn_id_target.reshape(bs, num_dn_groups, num_dn)

        # ========== STEP 3: EXTRACT PREVIOUS FRAME DENOISING DATA ==========
        # Get cached denoising information from previous frame
        # This was stored during the previous frame's processing

        temp_dn_feat = self.dn_metas[
            "dn_instance_feature"
        ]  # [B, temp_groups, temp_dn, C]
        _, num_temp_dn_groups, num_temp_dn = temp_dn_feat.shape[:3]
        temp_dn_id = self.dn_metas["dn_id_target"]  # [B, temp_groups, temp_dn]

        # ========== STEP 4: MATCH INSTANCES ACROSS FRAMES ==========
        # Use instance IDs to find correspondences between current and previous frame
        # This is the core of temporal consistency - matching the same real objects

        # Create matching matrix: True where instance IDs match
        # Shape: [B, temp_groups, temp_dn, current_dn]
        # match[b,g,i,j] = True if temp_dn_id[b,g,i] == dn_id[b,g,j]
        match = temp_dn_id[..., None] == dn_id[:, :num_temp_dn_groups, None]

        # ========== STEP 5: PROPAGATE TARGETS FROM CURRENT TO PREVIOUS ==========
        # Update regression targets by matching instances between frames
        # Use current frame's GT targets (not previous frame's) to ensure up-to-date information
        #
        # Typical case: each previous DN sample matches exactly one current DN sample (same instance ID)
        # No match case: sum gives zero vector, and will be ignored in the loss calculation
        temp_reg_target = (
            match[..., None] * dn_reg_target[:, :num_temp_dn_groups, None]
        ).sum(
            dim=3
        )  # [B, temp_groups, temp_dn, box_dim]

        # Update classification targets
        # If no match found for a previous DN sample, set class target to -1 (ignore)
        # If match found, keep the original class target from previous frame
        temp_cls_target = torch.where(
            torch.all(
                torch.logical_not(match), dim=-1
            ),  # No match across all current samples
            self.dn_metas["dn_cls_target"].new_tensor(-1),  # Set to ignore (-1)
            self.dn_metas["dn_cls_target"],  # Keep original target
        )

        # Extract other cached temporal metadata
        temp_valid_mask = self.dn_metas["valid_mask"]  # Previous frame validity mask
        temp_dn_anchor = self.dn_metas["dn_anchor"]  # Previous frame anchors

        # ========== STEP 6: HANDLE DIMENSION ALIGNMENT AND MERGE ==========
        # Previous and current frames may have different numbers of DN samples
        # due to varying numbers of objects. We need to align dimensions.

        # Create lists of temporal (previous) and current metadata
        temp_dn_metas = [
            temp_dn_feat,  # [B, temp_groups, temp_dn, C] - Previous frame features
            temp_dn_anchor,  # [B, temp_groups, temp_dn, box_dim] - Previous frame anchors
            temp_reg_target,  # [B, temp_groups, temp_dn, box_dim] - Updated regression targets
            temp_cls_target,  # [B, temp_groups, temp_dn] - Updated classification targets
            temp_valid_mask,  # [B, temp_groups, temp_dn] - Previous frame validity mask
            temp_dn_id,  # [B, temp_groups, temp_dn] - Previous frame instance IDs
        ]
        dn_metas = [
            dn_feat,  # [B, curr_groups, curr_dn, C] - Current frame features
            dn_anchor,  # [B, curr_groups, curr_dn, box_dim] - Current frame anchors
            dn_reg_target,  # [B, curr_groups, curr_dn, box_dim] - Current regression targets
            dn_cls_target,  # [B, curr_groups, curr_dn] - Current classification targets
            valid_mask,  # [B, curr_groups, curr_dn] - Current validity mask
            dn_id,  # [B, curr_groups, curr_dn] - Current instance IDs
        ]

        output = []

        # Process each type of metadata (features, anchors, targets, etc.)
        for i, (temp_meta, meta) in enumerate(zip(temp_dn_metas, dn_metas)):

            # ===== DIMENSION ALIGNMENT =====
            # Align the number of DN samples per group between frames

            if num_temp_dn < num_dn:
                # Previous frame has fewer DN samples → pad with zeros
                pad = (0, num_dn - num_temp_dn)  # Pad last dimension
                if temp_meta.dim() == 4:
                    # For 4D tensors (features, anchors), pad the 3rd dimension
                    pad = (0, 0) + pad  # (0, 0, 0, num_dn - num_temp_dn)
                else:
                    # For 3D tensors (targets, masks), pad the 2nd dimension
                    assert temp_meta.dim() == 3
                temp_meta = F.pad(temp_meta, pad, value=0)
            else:
                # Previous frame has more DN samples → truncate to current size
                temp_meta = temp_meta[:, :, :num_dn]

            # ===== SELECTIVE TEMPORAL UPDATE =====
            # Apply temporal data only where temporal_valid_mask is True
            # This handles cases where some batch items don't have valid temporal info

            # Create mask for batch items with valid temporal information
            mask = temporal_valid_mask[:, None, None]  # [B, 1, 1]
            if meta.dim() == 4:
                mask = mask.unsqueeze(dim=-1)  # [B, 1, 1, 1] for 4D tensors

            # Where temporal_valid_mask is True: use temporal data
            # Where temporal_valid_mask is False: use current data (for initial groups only)
            temp_meta = torch.where(
                mask,
                temp_meta,  # Use temporal data
                meta[
                    :, :num_temp_dn_groups
                ],  # Use current data (limited to temp groups)
            )

            # ===== COMBINE TEMPORAL AND CURRENT DATA =====
            # Concatenate temporal groups with remaining current groups
            # Final structure: [temporal_groups, remaining_current_groups]
            meta = torch.cat(
                [
                    temp_meta,  # Temporal groups (updated)
                    meta[
                        :, num_temp_dn_groups:
                    ],  # Remaining current groups (unchanged)
                ],
                dim=1,
            )

            # Flatten group and sample dimensions: [B, groups, samples, ...] → [B, total_samples, ...]
            meta = meta.flatten(1, 2)
            output.append(meta)

        # ========== STEP 7: COMBINE WITH NORMAL INSTANCE DATA ==========
        # Merge updated denoising data back with normal detection queries
        # Final layout: [normal_queries, updated_denoising_queries]

        output[0] = torch.cat([instance_feature, output[0]], dim=1)  # Combine features
        output[1] = torch.cat([anchor, output[1]], dim=1)  # Combine anchors

        # Return updated tensors
        # output[0]: instance_feature [B, N_total, C]
        # output[1]: anchor [B, N_total, 11]
        # output[2]: dn_reg_target [B, N_dn, box_dim]
        # output[3]: dn_cls_target [B, N_dn]
        # output[4]: valid_mask [B, N_dn]
        # output[5]: dn_id_target [B, N_dn]
        return output

    def cache_dn(
        self, dn_instance_feature, dn_anchor, dn_cls_target, valid_mask, dn_id_target
    ):
        if self.num_temp_dn_groups < 0:
            return
        num_dn_groups = self.num_dn_groups
        bs, num_dn = dn_instance_feature.shape[:2]
        num_temp_dn = num_dn // num_dn_groups
        temp_group_mask = torch.randperm(num_dn_groups) < self.num_temp_dn_groups
        temp_group_mask = temp_group_mask.to(device=dn_anchor.device)
        dn_instance_feature = dn_instance_feature.detach().reshape(
            bs, num_dn_groups, num_temp_dn, -1
        )[:, temp_group_mask]
        dn_anchor = dn_anchor.detach().reshape(bs, num_dn_groups, num_temp_dn, -1)[
            :, temp_group_mask
        ]
        dn_cls_target = dn_cls_target.reshape(bs, num_dn_groups, num_temp_dn)[
            :, temp_group_mask
        ]
        valid_mask = valid_mask.reshape(bs, num_dn_groups, num_temp_dn)[
            :, temp_group_mask
        ]
        if dn_id_target is not None:
            dn_id_target = dn_id_target.reshape(bs, num_dn_groups, num_temp_dn)[
                :, temp_group_mask
            ]
        self.dn_metas = dict(
            dn_instance_feature=dn_instance_feature,
            dn_anchor=dn_anchor,
            dn_cls_target=dn_cls_target,
            valid_mask=valid_mask,
            dn_id_target=dn_id_target,
        )
