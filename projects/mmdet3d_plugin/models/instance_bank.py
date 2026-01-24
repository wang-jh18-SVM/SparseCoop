import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from projects.mmdet3d_plugin.core.box3d import *
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS

__all__ = ["InstanceBank"]


def topk_custom(confidence, k, *inputs):
    """Helper function to select top-k elements based on confidence scores.

    Args:
        confidence (torch.Tensor): Confidence scores with shape [batch_size, N]
        k (int): Number of top elements to select
        *inputs: Variable number of tensors with shape [batch_size, N, ...]

    Returns:
        tuple: (top_k_confidence, [top_k_input1, top_k_input2, ...])
    """
    bs, N = confidence.shape[:2]
    # Get top-k indices
    confidence, indices = torch.topk(confidence, k, dim=1)
    # Convert to flattened indices
    indices = (indices + torch.arange(bs, device=indices.device)[:, None] * N).reshape(
        -1
    )
    outputs = []
    # Select top-k elements for each input tensor
    for input in inputs:
        outputs.append(input.flatten(end_dim=1)[indices].reshape(bs, k, -1))
    return confidence, outputs


@PLUGIN_LAYERS.register_module()
class InstanceBank(nn.Module):
    """Instance Bank for SparseCoop.

    This module manages object instances across frames for 4D detection and tracking.
    It maintains:
    1. A set of learnable instance queries and anchors
    2. A cache of instances from previous frames
    3. Tracking of instance IDs for object association

    The instance bank is a core component for temporal modeling, enabling the
    4D detection system to track objects across time.
    """

    def __init__(
        self,
        num_anchor,
        embed_dims,
        anchor,
        anchor_handler=None,
        num_temp_instances=0,
        default_time_interval=0.5,
        confidence_decay=0.6,
        anchor_grad=True,
        feat_grad=True,
        max_time_interval=2,
        use_2d_proposals=False,
        proposal_2d_config={},
        default_coop_time_interval=0.0,
        cross_agent_interaction=None,
        v2x_side="vehicle-side",
        only_pos_dn=False,
        coop_dn_noise_trans_std=-1,
        coop_dn_noise_rot_std=-1,
        # Ablation Flags
        coop_anchor_wo_latency_compensation=False,
    ):
        """Initialize the InstanceBank.

        Args:
            num_anchor (int): Number of learnable instance queries/anchors.
            embed_dims (int): Dimension of instance features.
            anchor (str, list, np.ndarray): Initial anchor values, can be loaded from a file.
            anchor_handler (dict, optional): Configuration for anchor projection/transformation handler.
            num_temp_instances (int): Number of temporal instances to cache from previous frames.
                If 0, no temporal modeling is performed.
            default_time_interval (float): Default time interval between frames.
            confidence_decay (float): Decay factor for confidence scores when propagating instances.
            anchor_grad (bool): Whether anchors are learnable (require gradients).
            feat_grad (bool): Whether instance features are learnable (require gradients).
            max_time_interval (float): Maximum allowed time interval for valid temporal modeling.
            use_2d_proposals (bool): Whether to use 2D detection proposals to initialize instances.
            proposal_2d_config (dict, optional): Configuration for 2D proposals initialization.
        """
        super(InstanceBank, self).__init__()
        self.embed_dims = embed_dims
        self.num_temp_instances = num_temp_instances
        self.default_time_interval = default_time_interval
        self.confidence_decay = confidence_decay
        self.max_time_interval = max_time_interval
        self.use_2d_proposals = use_2d_proposals
        self.proposal_2d_config = proposal_2d_config
        self.default_coop_time_interval = default_coop_time_interval
        # Build anchor handler if provided (for anchor transformations)
        if anchor_handler is not None:
            anchor_handler = build_from_cfg(anchor_handler, PLUGIN_LAYERS)
            assert hasattr(anchor_handler, "anchor_projection")
        self.anchor_handler = anchor_handler
        self.v2x_side = v2x_side
        self.coop_dn_noise_trans_std = coop_dn_noise_trans_std
        self.coop_dn_noise_rot_std = coop_dn_noise_rot_std

        # Load initial anchors from file, list, or numpy array
        if isinstance(anchor, str):
            anchor = np.load(anchor)
        elif isinstance(anchor, (list, tuple)):
            anchor = np.array(anchor)

        # Limit number of anchors and create learnable parameter
        self.num_anchor = min(len(anchor), num_anchor)
        anchor = anchor[:num_anchor]
        self.anchor = nn.Parameter(
            torch.tensor(anchor, dtype=torch.float32), requires_grad=anchor_grad
        )
        self.anchor_init = anchor

        # Create learnable instance features
        self.instance_feature = nn.Parameter(
            torch.zeros([self.anchor.shape[0], self.embed_dims]),
            requires_grad=feat_grad,
        )

        # Initialize cross-agent interaction module
        if self.v2x_side == "cooperative":
            self.cross_agent_interaction = build_from_cfg(
                cross_agent_interaction, PLUGIN_LAYERS
            )
        else:
            self.cross_agent_interaction = None

        # Debug Flags
        self.only_pos_dn = only_pos_dn
        self.coop_anchor_wo_latency_compensation = coop_anchor_wo_latency_compensation

        # Initialize cache variables
        self.reset()
        self.reset_coop()

    def init_weight(self):
        """Initialize weights for anchors and instance features."""
        # Reset anchors to initial values
        self.anchor.data = self.anchor.data.new_tensor(self.anchor_init)
        # Initialize instance features with Xavier uniform if they are learnable
        if self.instance_feature.requires_grad:
            torch.nn.init.xavier_uniform_(self.instance_feature.data, gain=1)

    def reset(self):
        """Reset all cache variables for temporal modeling."""
        self.cached_feature = None  # Cached instance features from previous frame
        self.cached_anchor = None  # Cached instance anchors from previous frame
        self.metas = None  # Metadata from previous frame
        self.mask = None  # Mask for valid temporal instances (time interval check)
        self.confidence = None  # Confidence scores
        self.temp_confidence = None  # Temporal confidence scores
        self.instance_id = None  # Instance IDs for tracking
        self.prev_id = 0  # Counter for assigning new instance IDs
        # # For 2D proposals
        # self.dynamic_instances_count = (
        #     0  # Count of dynamic instances (from 2D proposals)
        # )
        # For cross-agent ID mapping
        if self.cross_agent_interaction is not None:
            self.cross_agent_interaction.reset_id_mappings()

    def reset_coop(self):
        """Reset cooperative anchors and instance features."""
        self.coop_instance_feature = None
        self.coop_anchor = None
        self.coop_confidence = None
        self.coop_valid_mask = None
        self.coop_dn_id_target = None
        self.coop_dn_cls_target = None
        self.coop_dn_valid_mask = None

    def _convert_bin_depth_to_specific(self, pred_indices, mode="LID", inverse=False):
        """Convert between depth bins and specific depth values.

        This method handles the conversion between:
        1. Depth bins to specific depth values (inverse=False)
        2. Specific depth values to nearest bin (inverse=True)

        The real depth value vs bin index is non-linear, which is a common practice because:
        - Objects closer to the camera need finer depth resolution
        - Objects further away can have coarser depth resolution

        Args:
            pred_indices (Tensor): Input indices or depth values
            mode (str): Conversion mode, currently only "LID" supported
            inverse (bool): Whether to convert from depth to bin

        Returns:
            Tensor: Converted depth values or bin indices
        """
        depth_min, depth_max, num_bins = [
            self.proposal_2d_config["depthnet_config"].get(key)
            for key in ["depth_min", "depth_max", "num_depth_bins"]
        ]
        if mode == "LID":
            bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
            if not inverse:  # bin -> depth
                depth = depth_min + bin_size / 8 * (
                    torch.square(pred_indices / 0.5 + 1) - 1
                )
                return depth
            else:  # depth -> nearest bin
                # the param "pred_indices" is actually the predicted depth value
                indices = -0.5 + 0.5 * torch.sqrt(
                    1 + 8 * (pred_indices - depth_min) / bin_size
                )
                indices = indices.type(torch.int64)
                return indices

    @torch.no_grad()
    def build_query2d_proposal(
        self,
        context2d_feat,
        pred_bbox_list,
        bbox2d_scores,
        pred_depth,
        img2lidar_RTs,
        bn,
        imgHW,
        sample_idx_list=None,  # Only for debug
    ):
        """Build 3D query proposals from 2D detections and depth estimates.

        This method converts 2D bounding box proposals into 3D query points by:
        1. Processing 2D bounding boxes and their corresponding depth estimates
        2. Optionally generating multiple depth proposals for each 2D box
        3. Projecting 2D+Depth points to 3D space using camera parameters
        4. Normalizing 3D coordinates to the point cloud range

        Args:
            context2d_feat (Tensor): 2D context features of shape (M, C)
            pred_bbox_list (list): List of 2D bounding box predictions from each camera view.
                Each element is a tensor of shape (Mi, 4) containing [cx, cy, h, w] coordinates.
            bbox2d_scores (Tensor): Confidence scores for 2D proposals of shape (M, 1)
            pred_depth (Tensor): Predicted depth maps of shape (B*N, H, W, 1) or (B*N, H, W, D)
                if using depth logits. Contains depth estimates for each pixel.
            img2lidar_RTs (Tensor): Camera to LiDAR transformation matrices of shape (B, N, 4, 4)
            bn (tuple): Shape tuple (batch_size, num_cams)
            imgHW (tuple): Padded image dimensions (height, width)

        Returns:
            tuple:
                - new_reference_points (Tensor): 3D reference points of shape (B, M, 3)
                - context2d_feat (Tensor): Context features for the 3D queries
        """
        B, N = bn  # (3, 4)
        img_h, img_w = imgHW  # (256, 704)
        eps = 1e-5  # small value to avoid division by zero
        assert (
            img_h % pred_depth.shape[1] == 0
        ), f"The img_h={img_h} must be divisible by pred_depth.shape[1]={pred_depth.shape[1]}"
        depth_downsample = int(img_h / pred_depth.shape[1])  # 256/64=4

        # Convert list of 2D boxes to tensor and get number of boxes per view
        bbox_nums = [len(bbox) for bbox in pred_bbox_list]  # BN*[Mi]
        bboxes = torch.cat(
            pred_bbox_list, dim=0
        ).float()  # [sum(Mi),4], gather boxes together

        # For each 2D box, get its corresponding depth estimate
        depth_list = []
        h_max, w_max = pred_depth.shape[1:3]  # 64, 176
        for ith, pred_bbox in enumerate(pred_bbox_list):
            if bbox_nums[ith] != 0:
                # Get depth map for current view
                cur_depthmap = pred_depth[ith].flatten(0, 1)  # shape (HW, D)

                # Get center points of 2D boxes and convert to integer coordinates
                cur_center2d = (
                    (pred_bbox[:, :2] / depth_downsample).round().long()
                )  # first w then h
                cur_center2d[cur_center2d < 0] = 0
                cur_center2d[:, 0][cur_center2d[:, 0] >= w_max] = w_max - 1
                cur_center2d[:, 1][cur_center2d[:, 1] >= h_max] = h_max - 1

                # Convert to (h,w) format, then flatten for depth lookup
                cur_center2d = cur_center2d.flip(dims=(-1,))  # [Mi,2]
                cur_center2d_ = (
                    cur_center2d[:, 0] * (img_w / depth_downsample) + cur_center2d[:, 1]
                )  # [Mi]

                # Get depth values at box centers
                # torch.gather() selects specific rows from cur_depthmap based on the indices in cur_center2d_
                cur_depth = torch.gather(
                    cur_depthmap,  # [HW, 51]
                    0,
                    cur_center2d_.long().unsqueeze(1).repeat(1, cur_depthmap.shape[1]),
                )  # (Mi, D)
                depth_list.append(cur_depth)

        # Combine depth estimates from all views
        depths = torch.cat(depth_list, dim=0)  # (M, D)
        assert depths.shape[0] == sum(bbox_nums)

        # Remove the farthest depth bin which means infinite depth
        depths = depths[:, :-1]  # (M, D-1)

        # Optionally generate multiple depth proposals for each 2D box by selecting top-k depth values
        topk = self.proposal_2d_config['multi_depth_config'].get("topk", 1)
        assert topk > 0

        # Get top-k depth values and their indices
        topk_values, topk_bins = torch.topk(depths, topk, dim=1)  # (M, K)

        # Filter out depth proposals that are too close to the camera
        range_min = self.proposal_2d_config['multi_depth_config'].get("range_min", -1)
        if range_min != -1:
            range_min_bin = self._convert_bin_depth_to_specific(
                torch.tensor([range_min]), inverse=True
            ).item()
            valid_indices = topk_bins[:, 0] >= range_min_bin  # (M) Bool
            # todo: why not topk_indices[:, 1:]
        else:
            valid_indices = torch.ones_like(topk_bins[:, 0], dtype=torch.bool)

        # Duplicate boxes for each depth proposal
        bboxes_extra = bboxes.repeat(topk - 1, 1)
        bboxes = torch.cat(
            [bboxes, bboxes_extra[valid_indices.repeat(topk - 1)]], dim=0
        )  # (M+vM*(topk-1), 4), vM=sum(valid_indices)

        # Get additional depth proposals
        depths_extra_bins = topk_bins[:, 1:][valid_indices]  # (vM, topk-1)
        depths_extra_bins = (
            depths_extra_bins.transpose(1, 0).flatten().unsqueeze(-1)
        )  # (vM*(topk-1), 1)
        depth_bins = torch.cat(
            [topk_bins[:, 0:1], depths_extra_bins], dim=0
        )  # (M+vM*(topk-1), 1)

        # Expand context features for additional proposals
        context2d_feat_extra = context2d_feat.repeat(topk - 1, 1)
        context2d_feat = torch.cat(
            [
                context2d_feat,
                context2d_feat_extra[valid_indices.repeat(topk - 1)],
            ],
            dim=0,
        )  # (M+vM*(topk-1), C)

        assert len(valid_indices) == sum(bbox_nums)
        assert (
            bboxes.shape[0]
            == depth_bins.shape[0]
            == context2d_feat.shape[0]
            == sum(bbox_nums) + sum(valid_indices) * (topk - 1)
        )

        # Optionally weight proposals by 2D confidence scores
        if bbox2d_scores is not None:  # True
            thr = torch.tensor([0.1]).to(bbox2d_scores.device)  # score threshold

            # convert 2d scores(probabilities) to log_odds centered at threshold
            # log_odds = log(p/(1-p))
            log_odds = torch.log(bbox2d_scores / (1 - bbox2d_scores)) - torch.log(
                thr / (1 - thr)
            )  # (M, 1)

            # Weight depth proposals by their relative confidence
            topk_values = topk_values / topk_values[:, 0:1]  # rescale, (M, topk)
            dscores_extra = (
                topk_values[:, 1:][valid_indices]
                .transpose(1, 0)
                .flatten()
                .unsqueeze(-1)
            )  # (vM*(topk-1), 1)
            dscores = torch.cat(
                [topk_values[:, 0:1], dscores_extra], dim=0
            )  # (M+vM*(topk-1), 1)
            log_odds = torch.cat(
                [log_odds, log_odds[valid_indices].repeat(topk - 1, 1)], dim=0
            )
            log_odds = log_odds * dscores

            # Reweight original features using log odds instead of concatenating
            # Apply sigmoid to log_odds to get a 0-1 weight and add 1 to keep original feature scale
            context2d_feat = context2d_feat * (torch.sigmoid(log_odds) + 1.0)

        # # Filter out proposals of the farthest depth bin
        # valid_indices = valid_indices & (
        #     topk_bins[:, 0]
        #     < self.proposal_2d_config["depthnet_config"]["num_depth_bins"]
        # )

        # Convert depth bins to actual depth values (M', 1) M'=M+vM*(topk-1)
        depths = self._convert_bin_depth_to_specific(depth_bins)

        # Project 2D+Depth points to 3D camera coordinates
        # (u,v), d -> (ud,vd,d,1) in homogeneous coordinates
        coords = torch.cat(
            [bboxes[:, :2], depths], dim=1
        )  # (M', 3), order is (w, h, d)
        coords = torch.cat((coords, torch.ones_like(coords[..., :1])), -1)  # (M', 4)
        coords[..., :2] = coords[..., :2] * torch.maximum(
            coords[..., 2:3], eps * torch.ones_like(coords[..., 2:3])
        )
        coords = coords.unsqueeze(-1)  # (M', 4, 1)

        # Get camera to LiDAR transformation matrices
        img2lidars = img2lidar_RTs.view(B * N, 1, 4, 4)  # (BN, 1, 4, 4)

        # Repeat transformation matrices for each proposal
        img2lidars_ = torch.cat(
            [img2lidars[kth].repeat(num, 1, 1) for kth, num in enumerate(bbox_nums)],
            dim=0,
        )  # (M, 4, 4)

        # Handle multiple depth proposals
        img2lidars_extra = img2lidars_.repeat(topk - 1, 1, 1)
        img2lidars_extra = img2lidars_extra[valid_indices.repeat(topk - 1)]
        img2lidars_ = torch.cat([img2lidars_, img2lidars_extra], dim=0)  # (M', 4, 4)

        # Project points to 3D LiDAR coordinates
        coords3d = torch.matmul(img2lidars_, coords).squeeze(-1)[..., :3]  # (M', 3)

        # Reorganize coordinates and features as list
        # each list contains Mi+vMi*(topk-1) coordinates and features
        # i means the i-th sample from (batch_size x num_views)
        valid_indices = torch.cat(
            [torch.ones_like(valid_indices), valid_indices.repeat(topk - 1)], dim=0
        )  # [M*K]
        assert sum(valid_indices) == coords3d.shape[0] == context2d_feat.shape[0]

        # Collect valid box indices
        total_idx = 0
        valid_idx = 0
        sample_indices = [[] for _ in range(B * N)]
        for _ in range(topk):
            for i, box_num in enumerate(bbox_nums):
                for _ in range(box_num):
                    if valid_indices[total_idx]:
                        sample_indices[i].append(valid_idx)
                        valid_idx += 1
                    total_idx += 1

        # Get tensors for each sample
        list_coords3d = []
        list_context2d_feat = []
        for i, indices in enumerate(sample_indices):
            if len(indices) > 0:
                # Non-empty sample: stack tensors
                list_coords3d.append(coords3d[indices])
                list_context2d_feat.append(context2d_feat[indices])
            else:
                # Empty sample: create empty tensor with correct feature dimensions
                list_coords3d.append(
                    torch.zeros(
                        (0, coords3d.shape[1]),
                        device=coords3d.device,
                        dtype=coords3d.dtype,
                    )
                )
                list_context2d_feat.append(
                    torch.zeros(
                        (0, context2d_feat.shape[1]),
                        device=context2d_feat.device,
                        dtype=context2d_feat.dtype,
                    )
                )

        assert (
            sum(valid_indices)
            == sum([tensor.shape[0] for tensor in list_coords3d])
            == sum([tensor.shape[0] for tensor in list_context2d_feat])
        )
        return list_coords3d, list_context2d_feat

    def generate_ref_pts_from_2d_proposals(self, feature_maps, metas):
        """Generate 3D reference points from 2D detection proposals.

        Args:
            feature_maps (torch.Tensor): Feature maps from backbone
            metas (dict): Meta information including camera parameters

        Returns:
            tuple: (anchors, features)
                - anchors: 3D reference points generated from 2D detections B*[N*M', 3]
                - features: Features for these reference points B*[N*M', C]
        """
        batch_size, num_views, _ = metas['image_wh'].shape
        img_W, img_H = metas['image_wh'][0][0]

        # Get camera projection matrices
        lidar2img_RTs = np.stack(
            [meta['lidar2img'] for meta in metas['img_metas']], axis=0
        )  # (B, N, 4, 4)
        img2lidar_RTs = (
            torch.from_numpy(np.linalg.inv(lidar2img_RTs)).cuda().float()
        )  # (B, N, 4, 4)

        # Get 2D detection boxes and scores
        outs_roi = metas["2d_proposals"]
        pred_bbox_list = [it.detach() for it in outs_roi['bbox_list']]  # BN*(Mi, 4)
        bbox2d_scores = outs_roi['bbox2d_scores'].detach()

        # Get predicted depths
        pred_depth = outs_roi['pred_depth'].detach()
        depth_input = pred_depth.permute(0, 2, 3, 1)  # (B*N, H, W, D)

        # Get 2D context features
        _dim = feature_maps[0].shape[-1]
        valid_indices = outs_roi["valid_indices"]
        context_feat = feature_maps[0][
            valid_indices.reshape(  # (B*N, sum(Hi*Wi), 1)
                batch_size, -1, 1
            ).repeat(  # (B,N*sum(Hi*Wi), 1)
                1, 1, _dim
            )  # (B,N*sum(Hi*Wi), C)
        ].reshape(
            -1, _dim
        )  # (sum(Mi), C)
        context2d_feat = context_feat.detach()  # (sum(Mi), C)

        # If there are no 2D proposals, return empty lists
        if context2d_feat.shape[0] == 0:
            return [], [], []

        list_coords3d, list_context2d_feat = self.build_query2d_proposal(
            context2d_feat,  # [M, 256], M: sum of Mi
            pred_bbox_list,  # BN*(Mi, 4)
            bbox2d_scores,  # [M, 1]
            depth_input,  # [BN, H, W, 51]
            img2lidar_RTs,  # (B, N, 4, 4)
            (batch_size, num_views),  # (3, 4)
            (img_H, img_W),  # (256, 704)
            [i["sample_idx"] for i in metas["img_metas"]],  # Only for debug
        )  # B*N*[M', 3], B*N*[M', C]

        # Reorganize to B*[N*M', 3], B*[N*M', C]
        list_coords3d = [
            torch.cat(list_coords3d[i : i + num_views], dim=0)
            for i in range(0, len(list_coords3d), num_views)
        ]
        list_context2d_feat = [
            torch.cat(list_context2d_feat[i : i + num_views], dim=0)
            for i in range(0, len(list_context2d_feat), num_views)
        ]

        # Only keep proposals under specific height # TODO: hard code
        height_threshold = 2.0 if self.v2x_side != "drone-side" else -15.0
        height_mask = [coords3d[:, 2] < height_threshold for coords3d in list_coords3d]
        list_coords3d = [
            coords3d[mask] for coords3d, mask in zip(list_coords3d, height_mask)
        ]
        list_context2d_feat = [
            context2d_feat[mask]
            for context2d_feat, mask in zip(list_context2d_feat, height_mask)
        ]

        valid_nums = [len(l) for l in list_coords3d]

        return list_coords3d, list_context2d_feat, valid_nums

    def get(
        self,
        batch_size,
        metas=None,
        dn_metas=None,
        feature_maps=None,
        is_cooperative=False,
    ):
        """Get instance features and anchors for the current frame.

        This function:
        1. Replicates learnable queries/anchors for each batch item
        2. If temporal instances exist, transforms them to the current frame
        3. Calculates the time interval between frames
        4. If using 2D proposals, generates anchors from 2D detections

        Args:
            batch_size (int): Current batch size.
            metas (dict): Metadata for the current frame.
            dn_metas (dict, optional): Denoising metadata.
            feature_maps (torch.Tensor, optional): Feature maps for 2D proposal generation.
            outs_roi (dict, optional): Outputs from ROI head for 2D proposal generation.

        Returns:
            tuple: (instance_feature, anchor, cached_feature, cached_anchor, time_interval)
                - instance_feature: Current frame instance features [B, N, C]
                - anchor: Current frame anchors [B, N, 11]
                - cached_feature: Temporal features from previous frame (or None)
                - cached_anchor: Temporal anchors from previous frame (or None)
                - time_interval: Time difference between frames
        """
        # Static query initialization
        instance_feature = torch.tile(self.instance_feature[None], (batch_size, 1, 1))
        anchor = torch.tile(self.anchor[None], (batch_size, 1, 1))

        # Use 2D proposals to add dynamic queries if specified
        if self.use_2d_proposals:
            assert (
                feature_maps is not None
            ), "feature_maps is required for 2D proposal generation"
            dynamic_pts, dynamic_features, dynamic_nums = (
                self.generate_ref_pts_from_2d_proposals(feature_maps, metas)
            )

            if sum(dynamic_nums) > 0:
                # Randomly select static anchors and replace them with dynamic ones
                # Keeping the total number of anchors unchanged as self.num_anchor
                for b in range(batch_size):
                    replace_idx = np.random.choice(
                        range(self.num_anchor), dynamic_nums[b], replace=False
                    )
                    anchor[b, replace_idx, :3] = dynamic_pts[b]
                    instance_feature[b, replace_idx] = dynamic_features[b]

                # # Concatenate dynamically generated anchors and features with static ones
                # instance_feature = torch.cat(
                #     [instance_feature, dynamic_features], dim=1
                # )
                # anchor = torch.cat([anchor, dynamic_anchors], dim=1)
                # self.num_anchor = instance_feature.shape[1]

        # If we have cached instances from previous frame
        if self.cached_anchor is not None and batch_size == self.cached_anchor.shape[0]:
            # Calculate time interval between current and previous frames
            history_time = self.metas["timestamp"]
            time_interval = metas["timestamp"] - history_time
            time_interval = time_interval.to(dtype=instance_feature.dtype)

            # Create mask for valid temporal instances (time interval not too large)
            self.mask = torch.abs(time_interval) <= self.max_time_interval

            # If anchor handler exists, transform cached anchors to current frame
            if self.anchor_handler is not None:
                # Calculate transformation matrix from previous to current frame
                T_temp2cur = self.cached_anchor.new_tensor(
                    np.stack(
                        [
                            x["T_global_inv"] @ self.metas["img_metas"][i]["T_global"]
                            for i, x in enumerate(metas["img_metas"])
                        ]
                    )
                )

                # Project cached anchors to current frame using transformation
                self.cached_anchor = self.anchor_handler.anchor_projection(
                    self.cached_anchor, [T_temp2cur], time_intervals=[-time_interval]
                )[0]

            # If denoising is active, also transform denoising anchors
            if (
                self.anchor_handler is not None
                and dn_metas is not None
                and batch_size == dn_metas["dn_anchor"].shape[0]
            ):
                num_dn_group, num_dn = dn_metas["dn_anchor"].shape[1:3]
                dn_anchor = self.anchor_handler.anchor_projection(
                    dn_metas["dn_anchor"].flatten(1, 2),
                    [T_temp2cur],
                    time_intervals=[-time_interval],
                )[0]
                dn_metas["dn_anchor"] = dn_anchor.reshape(
                    batch_size, num_dn_group, num_dn, -1
                )

            # Set default time interval where invalid or zero
            time_interval = torch.where(
                torch.logical_and(time_interval != 0, self.mask),
                time_interval,
                time_interval.new_tensor(self.default_time_interval),
            )
        else:
            # If no cached instances or batch size mismatch, reset cache
            self.reset()
            time_interval = instance_feature.new_tensor(
                [self.default_time_interval] * batch_size
            )

        # if we have cooperative anchors from other agents
        if is_cooperative:
            self.coop_anchor_unfiltered = []
            coop_instance_feature_filtered = []
            coop_anchor_filtered = []
            coop_confidence_filtered = []
            coop_instance_id_filtered = []
            valid_counts = []
            self.inf_direct_masks = []

            if "coop_dn_instance_feature" in metas:
                coop_dn_instance_feature_filtered = []
                coop_dn_anchor_filtered = []
                coop_dn_id_target_filtered = []
                coop_dn_cls_target_filtered = []
                coop_dn_valid_counts = []

            # # For certain GPU and CUDA version, inverse() by cuda is not working
            # self.coop_inf2veh_rt = (
            #     metas["veh2inf_rt"]
            #     .cpu()
            #     .inverse()
            #     .to(device=instance_feature.device, dtype=instance_feature.dtype)
            # )
            self.coop_inf2veh_rt = (
                metas["veh2inf_rt"].inverse().to(dtype=instance_feature.dtype)
            )
            for b in range(batch_size):
                coop_time_interval = metas["timestamp"][b] - metas["coop_timestamp"][b]
                coop_time_interval = coop_time_interval.to(dtype=instance_feature.dtype)
                if self.coop_anchor_wo_latency_compensation:
                    coop_time_interval = 0.0 * coop_time_interval
                coop_instance_feature = metas["coop_instance_feature"][b]
                coop_anchor = metas["coop_anchor"][b]

                coop_anchor = self.anchor_handler.anchor_projection(
                    coop_anchor,
                    [self.coop_inf2veh_rt[b]],
                    time_intervals=[-coop_time_interval],
                )[0]
                self.coop_anchor_unfiltered.append(coop_anchor)

                # Filter cooperative instances with low confidence, far distance, and low height
                coop_pos = coop_anchor[..., [X, Y, Z]]
                inf_interaction_mask, inf_direct_mask = (
                    self.cross_agent_interaction.inf_filter(
                        coop_pos,  # (Nc, 3)
                        lidar2img_rt=metas["projection_mat"][b],  # (N, 4, 4)
                        image_wh=metas["image_wh"][b],  # (N, 2)
                    )
                )

                coop_instance_feature_filtered.append(
                    coop_instance_feature[inf_interaction_mask]
                )
                coop_anchor_filtered.append(coop_anchor[inf_interaction_mask])
                coop_confidence_filtered.append(
                    metas["coop_score"][b][inf_interaction_mask]
                )
                coop_instance_id_filtered.append(
                    metas["coop_instance_ids"][b][inf_interaction_mask]
                )

                valid_counts.append(inf_interaction_mask.sum().item())
                self.inf_direct_masks.append(inf_direct_mask)

                if "coop_dn_instance_feature" in metas:
                    self.coop_dn_group_num, coop_dn_per_group = metas[
                        "coop_dn_instance_feature"
                    ][b].shape[:2]
                    ##### Create validity mask for every group: True if dn ID exists in gt_instance_id_inf #####
                    gt_instance_id_inf = metas["instance_id_inf"][b]
                    gt_instance_id_inf = gt_instance_id_inf[gt_instance_id_inf != -1]
                    coop_dn_valid_mask = []
                    for g in range(self.coop_dn_group_num):
                        coop_dn_valid_mask_group = torch.isin(
                            metas["coop_dn_id_target"][b][g].squeeze(-1),
                            gt_instance_id_inf,
                        )
                        if self.only_pos_dn:
                            # Set False for negative samples
                            coop_dn_neg_mask_group = (
                                metas["coop_dn_cls_target"][b][g].squeeze(-1) == -3
                            )
                            assert coop_dn_neg_mask_group.sum() * 2 == coop_dn_per_group
                            coop_dn_valid_mask_group[coop_dn_neg_mask_group] = False

                        if g == 0:
                            coop_dn_per_group_filtered = coop_dn_valid_mask_group.sum()
                        else:
                            # Each group should have the same number of valid samples
                            assert (
                                coop_dn_valid_mask_group.sum()
                                == coop_dn_per_group_filtered
                            )

                        coop_dn_valid_mask.append(coop_dn_valid_mask_group)

                    coop_dn_valid_mask = torch.stack(coop_dn_valid_mask, dim=0)

                    coop_dn_id_target = metas["coop_dn_id_target"][b][
                        coop_dn_valid_mask
                    ]
                    coop_dn_instance_feature = metas["coop_dn_instance_feature"][b][
                        coop_dn_valid_mask
                    ]
                    coop_dn_anchor = metas["coop_dn_anchor"][b][coop_dn_valid_mask]
                    coop_dn_cls_target = metas["coop_dn_cls_target"][b][
                        coop_dn_valid_mask
                    ]
                    # coop_dn_reg_target = metas["coop_dn_reg_target"][b][
                    #     coop_dn_valid_mask
                    # ]

                    if coop_dn_valid_mask.sum() > 0:
                        # ID Alignment: Transform DN inf ID to cooperative ID
                        coop_dn_id_target_coop = -torch.ones_like(coop_dn_id_target)
                        for dn_idx, id in enumerate(coop_dn_id_target.squeeze(-1)):
                            idx_inf = metas["instance_id_inf"][b] == id
                            assert idx_inf.sum() == 1
                            id_coop = metas["instance_id"][b][idx_inf]
                            coop_dn_id_target_coop[dn_idx] = id_coop
                        coop_dn_id_target = coop_dn_id_target_coop

                        # Anchor Alignment
                        ideal_inf2veh_rt = self.coop_inf2veh_rt[b]
                        noise_rt = torch.eye(4).to(ideal_inf2veh_rt.device)

                        # Add translation noise
                        if self.coop_dn_noise_trans_std > 0:
                            # Generate random translation noise (3D vector)
                            trans_noise = (
                                torch.randn(3, device=ideal_inf2veh_rt.device)
                                * self.coop_dn_noise_trans_std
                            )
                            noise_rt[:3, 3] = trans_noise

                        # Add rotation noise
                        if self.coop_dn_noise_rot_std > 0:
                            # Generate small random rotation angles in degrees (roll, pitch, yaw)
                            rot_angles = np.random.randn(3) * self.coop_dn_noise_rot_std
                            R_noise = R.from_euler("xyz", rot_angles).as_matrix()
                            R_noise = torch.tensor(
                                R_noise,
                                device=ideal_inf2veh_rt.device,
                                dtype=ideal_inf2veh_rt.dtype,
                            )
                            noise_rt[:3, :3] = R_noise

                        inf2veh_rt = noise_rt @ ideal_inf2veh_rt

                        coop_dn_anchor = self.anchor_handler.anchor_projection(
                            coop_dn_anchor,
                            [inf2veh_rt],
                            time_intervals=[-coop_time_interval],
                        )[0]

                    # Flatten the reshaped cooperative denoising parameters back to 1D for subsequent processing
                    coop_dn_instance_feature_filtered.append(
                        coop_dn_instance_feature.reshape(
                            self.coop_dn_group_num,
                            coop_dn_per_group_filtered,
                            coop_dn_instance_feature.shape[-1],
                        )
                    )
                    coop_dn_anchor_filtered.append(
                        coop_dn_anchor.reshape(
                            self.coop_dn_group_num,
                            coop_dn_per_group_filtered,
                            coop_dn_anchor.shape[-1],
                        )
                    )
                    coop_dn_id_target_filtered.append(
                        coop_dn_id_target.reshape(
                            self.coop_dn_group_num,
                            coop_dn_per_group_filtered,
                            coop_dn_id_target.shape[-1],
                        )
                    )
                    coop_dn_cls_target_filtered.append(
                        coop_dn_cls_target.reshape(
                            self.coop_dn_group_num,
                            coop_dn_per_group_filtered,
                            coop_dn_cls_target.shape[-1],
                        )
                    )
                    coop_dn_valid_counts.append(coop_dn_valid_mask.sum().item())

            # Pad to the maximum number of valid instances
            max_valid_count = max(valid_counts)
            # assert max_valid_count > 0, "no valid inf instances"

            coop_valid_mask = []
            for b in range(batch_size):
                pad_len = max_valid_count - valid_counts[b]
                if pad_len > 0:
                    coop_instance_feature_filtered[b] = torch.cat(
                        [
                            coop_instance_feature_filtered[b],
                            torch.zeros(
                                (pad_len, coop_instance_feature.shape[-1]),
                                device=coop_instance_feature.device,
                                dtype=coop_instance_feature.dtype,
                            ),
                        ],
                        dim=0,
                    )
                    coop_anchor_filtered[b] = torch.cat(
                        [
                            coop_anchor_filtered[b],
                            torch.zeros(
                                (pad_len, coop_anchor.shape[-1]),
                                device=coop_anchor.device,
                                dtype=coop_anchor.dtype,
                            ),
                        ],
                        dim=0,
                    )
                    # Pad confidence and instance id with -1
                    coop_confidence_filtered[b] = torch.cat(
                        [
                            coop_confidence_filtered[b],
                            -torch.ones(
                                (pad_len),
                                device=coop_instance_feature.device,
                                dtype=coop_instance_feature.dtype,
                            ),
                        ],
                        dim=0,
                    )
                    coop_instance_id_filtered[b] = torch.cat(
                        [
                            coop_instance_id_filtered[b],
                            -torch.ones(
                                (pad_len),
                                device=coop_instance_feature.device,
                                dtype=coop_instance_feature.dtype,
                            ),
                        ],
                        dim=0,
                    )

                # Create attention mask (1 for valid, 0 for padding)
                batch_mask = torch.ones(
                    max_valid_count, device=coop_anchor.device, dtype=torch.bool
                )
                batch_mask[valid_counts[b] :] = False
                coop_valid_mask.append(batch_mask)

            coop_instance_feature = torch.stack(coop_instance_feature_filtered, dim=0)
            self.coop_anchor = torch.stack(coop_anchor_filtered, dim=0)
            self.coop_confidence = torch.stack(coop_confidence_filtered, dim=0)
            self.coop_instance_id = torch.stack(coop_instance_id_filtered, dim=0)
            self.coop_valid_mask = torch.stack(coop_valid_mask, dim=0)

            if "coop_dn_instance_feature" in metas:
                max_dn_valid = max(coop_dn_valid_counts)
                max_dn_valid_per_group = max_dn_valid // self.coop_dn_group_num

                coop_dn_instance_feature = torch.stack(
                    [
                        F.pad(
                            x, (0, 0, 0, max_dn_valid_per_group - x.shape[1]), value=0
                        )
                        for x in coop_dn_instance_feature_filtered
                    ],
                    dim=0,
                ).reshape(batch_size, -1, coop_dn_instance_feature.shape[-1])
                coop_dn_anchor = torch.stack(
                    [
                        F.pad(
                            x, (0, 0, 0, max_dn_valid_per_group - x.shape[1]), value=0
                        )
                        for x in coop_dn_anchor_filtered
                    ],
                    dim=0,
                ).reshape(batch_size, -1, coop_dn_anchor.shape[-1])
                self.coop_dn_id_target = torch.stack(
                    [
                        F.pad(
                            x, (0, 0, 0, max_dn_valid_per_group - x.shape[1]), value=-1
                        )
                        for x in coop_dn_id_target_filtered
                    ],
                    dim=0,
                ).reshape(batch_size, -1, coop_dn_id_target.shape[-1])
                self.coop_dn_cls_target = torch.stack(
                    [
                        F.pad(
                            x, (0, 0, 0, max_dn_valid_per_group - x.shape[1]), value=-1
                        )
                        for x in coop_dn_cls_target_filtered
                    ],
                    dim=0,
                ).reshape(batch_size, -1, coop_dn_cls_target.shape[-1])

                # Vectorize valid mask for DN cross-attention, false for padded instances
                coop_dn_valid_mask = coop_dn_instance_feature.new_zeros(
                    batch_size,
                    self.coop_dn_group_num * max_dn_valid_per_group,
                )
                for b in range(batch_size):
                    for g in range(self.coop_dn_group_num):
                        start = g * max_dn_valid_per_group
                        end = start + coop_dn_valid_counts[b] // self.coop_dn_group_num
                        coop_dn_valid_mask[b, start:end] = 1
                self.coop_dn_valid_mask = coop_dn_valid_mask.bool()

                # Concatenate cooperative and denoising instances
                coop_instance_feature = torch.cat(
                    [coop_instance_feature, coop_dn_instance_feature], dim=1
                )
                self.coop_anchor = torch.cat([self.coop_anchor, coop_dn_anchor], dim=1)
            else:
                self.coop_dn_id_target = None
                self.coop_dn_cls_target = None
                self.coop_dn_valid_mask = None

            if max_valid_count > 0:
                # Implictly align cooperative features
                inf2veh_r_flattten = (
                    self.coop_inf2veh_rt[:, :3, :3]
                    .reshape(batch_size, 1, 9)
                    .repeat(
                        1,
                        max_valid_count
                        + (max_dn_valid if "coop_dn_instance_feature" in metas else 0),
                        1,
                    )
                )
                self.coop_instance_feature = (
                    self.cross_agent_interaction.cross_agent_align(
                        torch.cat([coop_instance_feature, inf2veh_r_flattten], dim=-1)
                    )
                )
            else:
                # dummy path to include the alignment layer
                dummy = self.cross_agent_interaction.cross_agent_align(
                    torch.zeros(
                        (
                            coop_instance_feature.shape[0],
                            coop_instance_feature.shape[2] + 9,
                        )
                    ).to(coop_instance_feature.device)
                )
                self.coop_instance_feature = coop_instance_feature + 0 * dummy.sum()

            # dummy path to include the alignment layer
            instance_feature = instance_feature + 0 * self.coop_instance_feature.sum()
        else:
            self.reset_coop()

        return (
            instance_feature,
            anchor,
            self.cached_feature,
            self.cached_anchor,
            time_interval,
            self.coop_instance_feature,
            self.coop_anchor,
            self.coop_valid_mask,
            self.coop_dn_valid_mask,
        )

    def update(self, instance_feature, anchor, confidence):
        """Update instance features and anchors using cached temporal instances.

        This function:
        1. Separates regular and denoising instances if present
        2. Selects top-N instances based on confidence
        3. Combines with cached temporal instances

        Args:
            instance_feature (torch.Tensor): Instance features [B, N, C]
            anchor (torch.Tensor): Instance anchors [B, N, 11]
            confidence (torch.Tensor): Confidence scores [B, N, num_classes]

        Returns:
            tuple: (updated_instance_feature, updated_anchor)
        """
        # If no cached instances, no update needed
        if self.cached_feature is None:
            return instance_feature, anchor

        # Handle denoising instances if present (separate them)
        num_dn = 0
        if instance_feature.shape[1] > self.num_anchor:
            num_dn = instance_feature.shape[1] - self.num_anchor
            dn_instance_feature = instance_feature[:, -num_dn:]
            dn_anchor = anchor[:, -num_dn:]
            instance_feature = instance_feature[:, : self.num_anchor]
            anchor = anchor[:, : self.num_anchor]
            confidence = confidence[:, : self.num_anchor]

        # Calculate how many new instances to select (total - temporal)
        N = self.num_anchor - self.num_temp_instances
        # Get max confidence per instance
        confidence = confidence.max(dim=-1).values
        # Select top-N instances based on confidence
        _, (selected_feature, selected_anchor) = topk_custom(
            confidence, N, instance_feature, anchor
        )

        # Combine cached temporal instances with selected new instances
        selected_feature = torch.cat([self.cached_feature, selected_feature], dim=1)
        selected_anchor = torch.cat([self.cached_anchor, selected_anchor], dim=1)

        # Only apply update for valid temporal instances (based on time interval mask)
        # Keeps the temporal instances (selected_feature) where the mask is True,
        # and keeps the original instances where the mask is False.
        # Each sample in the batch can independently decide whether to use temporal information
        # based on whether the time interval is within the threshold.
        instance_feature = torch.where(
            self.mask[:, None, None], selected_feature, instance_feature
        )
        anchor = torch.where(self.mask[:, None, None], selected_anchor, anchor)

        # Update instance IDs if tracking is active
        if self.instance_id is not None:
            # Set id of all invalid temporal instances to -1
            self.instance_id = torch.where(
                self.mask[:, None], self.instance_id, self.instance_id.new_tensor(-1)
            )

        # Reattach denoising instances if they were present
        if num_dn > 0:
            instance_feature = torch.cat([instance_feature, dn_instance_feature], dim=1)
            anchor = torch.cat([anchor, dn_anchor], dim=1)

        return instance_feature, anchor

    def coop_update(
        self,
        instance_feature,
        anchor,
        confidence,
        coop_interaction_inds,
        dn_id_target,
        timestamps,
    ):
        """Update instance features using cooperative sensing information.

        This function performs query matching, fusion, and complementation
        between vehicle and infrastructure instances.

        Args:
            instance_feature (torch.Tensor): Vehicle instance features [B, N, C]
            anchor (torch.Tensor): Vehicle instance anchors [B, N, 11]
            confidence (torch.Tensor): Vehicle instance class scores [B, N, num_classes]

        Returns:
            tuple: (updated_instance_feature, updated_anchor)
        """
        batch_size = instance_feature.shape[0]
        updated_features = []
        updated_anchors = []
        for ind in coop_interaction_inds:
            ego_start = ind["ego_start"]
            ego_end = ind["ego_end"]
            coop_start = ind["coop_start"]
            coop_end = ind["coop_end"]

            ego_instance_feature = instance_feature[:, ego_start:ego_end]
            ego_anchor = anchor[:, ego_start:ego_end]
            ego_confidence = confidence[:, ego_start:ego_end]

            if ind["ego_type"] == "normal" or ind["coop_type"] == "normal":
                ego_instance_id = (
                    self.instance_id[:, ego_start:ego_end]
                    if self.instance_id is not None
                    else None
                )
                ego_instance_id = (
                    None
                    if ego_instance_id is not None and ego_instance_id.shape[0] == 0
                    else ego_instance_id
                )

                coop_instance_id = self.coop_instance_id[:, coop_start:coop_end]
                coop_instance_id = (
                    None if coop_instance_id.shape[1] == 0 else coop_instance_id
                )
            else:
                # For ego and coop denoising instances, use DN ID target as instance ID
                assert dn_id_target is not None and self.coop_dn_id_target is not None
                ego_instance_id = dn_id_target[
                    :,
                    ego_start
                    - coop_interaction_inds[0]["ego_end"] : ego_end
                    - coop_interaction_inds[0]["ego_end"],
                ]
                coop_instance_id = self.coop_dn_id_target[
                    :,
                    coop_start
                    - coop_interaction_inds[0]["coop_end"] : coop_end
                    - coop_interaction_inds[0]["coop_end"],
                ].squeeze(-1)
            coop_instance_feature = self.coop_instance_feature[:, coop_start:coop_end]
            coop_anchor = self.coop_anchor[:, coop_start:coop_end]
            coop_confidence = self.coop_confidence[:, coop_start:coop_end]
            coop_confidence = None if coop_confidence.shape[1] == 0 else coop_confidence
            if self.coop_dn_valid_mask is not None:
                coop_valid_mask = torch.cat(
                    [self.coop_valid_mask, self.coop_dn_valid_mask], dim=1
                )[:, coop_start:coop_end]
            else:
                coop_valid_mask = self.coop_valid_mask[:, coop_start:coop_end]

            # Fusion and Complementation for current vehicle instances
            ego_confidence = ego_confidence.max(dim=-1).values.sigmoid()  # (B, Nv)
            updated_features_group = []
            updated_anchors_group = []
            for b in range(batch_size):
                updated_feature, updated_anchor = self.cross_agent_interaction(
                    veh_features=ego_instance_feature[b],
                    veh_anchors=ego_anchor[b],
                    veh_confidence=ego_confidence[b],
                    veh_instance_id=(
                        ego_instance_id[b] if ego_instance_id is not None else None
                    ),
                    veh_type=ind["ego_type"],
                    inf_features=coop_instance_feature[b][coop_valid_mask[b]],
                    inf_anchors=coop_anchor[b][coop_valid_mask[b]],
                    inf_confidence=(
                        coop_confidence[b][coop_valid_mask[b]]
                        if coop_confidence is not None
                        else None
                    ),
                    inf_instance_id=(
                        coop_instance_id[b][coop_valid_mask[b]]
                        if coop_instance_id is not None
                        else None
                    ),
                    inf_type=ind["coop_type"],
                    timestamp=timestamps[b],
                )
                updated_features_group.append(updated_feature)
                updated_anchors_group.append(updated_anchor)

            # Stack results back to batch dimension
            updated_features.append(torch.stack(updated_features_group, dim=0))
            updated_anchors.append(torch.stack(updated_anchors_group, dim=0))

        # # Reattach temporal instances
        # if self.cached_feature is not None:
        #     updated_features = torch.cat(
        #         [temporal_instance_feature, updated_features], dim=1
        #     )
        #     updated_anchors = torch.cat([temporal_anchor, updated_anchors], dim=1)

        # # Reattach denoising instances if they were present
        # if num_dn > 0:
        #     updated_features = torch.cat([updated_features, dn_instance_feature], dim=1)
        #     updated_anchors = torch.cat([updated_anchors, dn_anchor], dim=1)

        updated_features = torch.cat(updated_features, dim=1)
        updated_anchors = torch.cat(updated_anchors, dim=1)

        return updated_features, updated_anchors

    def cache(
        self, instance_feature, anchor, confidence, metas=None, feature_maps=None
    ):
        """Cache current frame instances for temporal modeling in the next frame.

        This function:
        1. Selects top-k instances based on confidence scores
        2. Stores their features, anchors, and metadata

        Args:
            instance_feature (torch.Tensor): Instance features [B, N, C]
            anchor (torch.Tensor): Instance anchors [B, N, 11]
            confidence (torch.Tensor): Confidence scores [B, N, num_classes]
            metas (dict): Frame metadata
            feature_maps (torch.Tensor, optional): Feature maps (unused)
        """
        # Skip if temporal modeling is disabled
        if self.num_temp_instances <= 0:
            return

        # Detach tensors to avoid gradients propagating through time
        instance_feature = instance_feature.detach()
        anchor = anchor.detach()
        confidence = confidence.detach()

        # Store metadata
        self.metas = metas

        # Calculate confidence scores from current frame observation(sigmoid of max class score)
        confidence = confidence.max(dim=-1).values.sigmoid()

        # Apply confidence decay and retention for previously cached instances
        if self.confidence is not None:
            # Use maximum of decayed previous confidence and current confidence
            # This helps maintain instance identity even with temporary occlusions
            confidence[:, : self.num_temp_instances] = torch.maximum(
                self.confidence * self.confidence_decay,
                confidence[:, : self.num_temp_instances],
            )
        # Store confidence for rematching instance ids
        self.temp_confidence = confidence

        # Select top-k instances to cache based on confidence
        (self.confidence, (self.cached_feature, self.cached_anchor)) = topk_custom(
            confidence, self.num_temp_instances, instance_feature, anchor
        )

    def get_instance_id(self, confidence, anchor=None, threshold=None):
        """Generate or update instance IDs for tracking.

        This function:
        1. Assigns IDs to previously unidentified high-confidence instances
        2. Maintains IDs across frames for consistent tracking

        Args:
            confidence (torch.Tensor): Instance confidence scores [B, N, num_classes]
            anchor (torch.Tensor, optional): Instance anchors (unused)
            threshold (float, optional): Confidence threshold for assigning new IDs

        Returns:
            torch.Tensor: Instance IDs for each instance [B, N]
        """
        # Calculate confidence scores
        confidence = confidence.max(dim=-1).values.sigmoid()
        # Initialize all instances with no ID (-1)
        instance_id = confidence.new_full(confidence.shape, -1).long()

        # If we have previous instance IDs, copy them to the current frame
        if (
            self.instance_id is not None
            and self.instance_id.shape[0] == instance_id.shape[0]
        ):
            instance_id[:, : self.instance_id.shape[1]] = self.instance_id

        # Identify instances that need new IDs (don't have an ID yet)
        mask = instance_id < 0
        # If threshold is provided, only assign IDs to high-confidence instances
        if threshold is not None:
            mask = mask & (confidence >= threshold)

        # Count how many new instances need IDs
        num_new_instance = mask.sum()
        # Create new IDs starting from previous highest ID
        new_ids = torch.arange(num_new_instance).to(instance_id) + self.prev_id
        # Assign new IDs to instances that need them
        instance_id[torch.where(mask)] = new_ids
        # Update counter for next frame
        self.prev_id += num_new_instance

        # Update instance IDs in the cache for temporal consistency
        if self.num_temp_instances > 0:
            self.update_instance_id(instance_id, confidence)

        return instance_id

    def update_instance_id(self, instance_id=None, confidence=None):
        """Update cached instance IDs based on current frame instances.

        This ensures ID consistency when instances are propagated to the next frame.

        Args:
            instance_id (torch.Tensor, optional): Current instance IDs
            confidence (torch.Tensor, optional): Instance confidence scores
        """
        # Get confidence scores from appropriate source
        if self.temp_confidence is None:
            if confidence.dim() == 3:  # bs, num_anchor, num_cls
                temp_conf = confidence.max(dim=-1).values
            else:  # bs, num_anchor
                temp_conf = confidence
        else:
            temp_conf = self.temp_confidence

        # Select top-k instance IDs based on confidence scores
        instance_id = topk_custom(temp_conf, self.num_temp_instances, instance_id)[1][0]
        instance_id = instance_id.squeeze(dim=-1)

        # Pad instance IDs to fill all slots, marking unused ones with -1
        self.instance_id = F.pad(
            instance_id, (0, self.num_anchor - self.num_temp_instances), value=-1
        )
        # self.instance_id = instance_id
