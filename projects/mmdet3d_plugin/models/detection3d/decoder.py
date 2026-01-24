from typing import Optional

import torch

from mmdet.core.bbox.builder import BBOX_CODERS

from projects.mmdet3d_plugin.core.box3d import *


@BBOX_CODERS.register_module()
class SparseBox3DDecoder(object):
    def __init__(
        self,
        num_output: int = 300,
        score_threshold: Optional[float] = None,
        sorted: bool = True,
    ):
        super(SparseBox3DDecoder, self).__init__()
        self.num_output = num_output
        self.score_threshold = score_threshold
        self.sorted = sorted

    def decode_box(self, box):
        yaw = torch.atan2(box[:, SIN_YAW], box[:, COS_YAW])
        box = torch.cat(
            [box[:, [X, Y, Z]], box[:, [L, W, H]].exp(), yaw[:, None], box[:, VX:]],
            dim=-1,
        )
        return box

    def decode(
        self,
        cls_scores,
        box_preds,
        instance_id=None,
        quality=None,
        instance_feature=None,  # Only for cooperative saving
        output_idx=-1,
    ):
        """
        Decode 3D detection results from model predictions.

        Args:
            cls_scores: Raw classification scores [batch_size, num_predictions, num_classes]
            box_preds: 3D box predictions [batch_size, num_predictions, box_dim]
            instance_id: Instance IDs for tracking (optional)
            quality: Quality scores containing centerness and yawness [batch_size, num_predictions, 2]
            instance_feature: Instance features for cooperative detection (optional)
            output_idx: Index of the output layer to decode (-1 for last layer)
        """
        # Apply sigmoid activation to convert logits to probabilities
        cls_scores = cls_scores[output_idx].sigmoid()

        # Determine if we need to squeeze class dimension (when instance_id is provided)
        squeeze_cls = instance_id is not None
        if squeeze_cls:
            # Take max class score and corresponding class ID
            cls_scores, cls_ids = cls_scores.max(dim=-1)
            cls_scores = cls_scores.unsqueeze(dim=-1)

        # Get box predictions for the specified output layer
        box_preds = box_preds[output_idx]
        bs, num_pred, num_cls = cls_scores.shape

        # Select top-k predictions based on class scores
        # This reduces the number of predictions to process
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            self.num_output, dim=1, sorted=self.sorted
        )

        # Extract class IDs from the flattened indices
        if not squeeze_cls:
            cls_ids = indices % num_cls

        # Apply score threshold filtering if specified
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold

        # QUALITY SCORE INTEGRATION: Combine class scores with quality scores
        if quality is not None:
            # Extract centerness scores from quality predictions
            # CNS = 0 represents centerness index in quality tensor
            centerness = quality[output_idx][..., CNS]

            # Gather centerness scores corresponding to the top-k predictions
            centerness = torch.gather(centerness, 1, indices // num_cls)

            # Store original class scores before quality integration
            cls_scores_origin = cls_scores.clone()

            # FINAL SCORE CALCULATION:
            # Final score = Class Score × Quality Score (centerness)
            # This ensures predictions with both high class confidence AND good localization get higher scores
            cls_scores *= centerness.sigmoid()

            # Re-sort predictions based on the combined scores
            cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True)

            # Update all related tensors to maintain consistency with new sorting
            if not squeeze_cls:
                cls_ids = torch.gather(cls_ids, 1, idx)
            if self.score_threshold is not None:
                mask = torch.gather(mask, 1, idx)
            indices = torch.gather(indices, 1, idx)

        # Process each batch item separately
        output = []
        for i in range(bs):
            # Extract category IDs and scores for current batch item
            category_ids = cls_ids[i]
            if squeeze_cls:
                category_ids = category_ids[indices[i]]
            scores = cls_scores[i]
            box = box_preds[i, indices[i] // num_cls]

            # Apply score threshold filtering if specified
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]

            # Store original class scores (before quality integration) if quality was used
            if quality is not None:
                scores_origin = cls_scores_origin[i]
                if self.score_threshold is not None:
                    scores_origin = scores_origin[mask[i]]

            # Decode 3D boxes from predictions
            decoded_box = self.decode_box(box)

            # Create output dictionary for current batch item
            output.append(
                {
                    "boxes_3d": decoded_box.cpu(),
                    "scores_3d": scores.cpu(),  # Final scores (class × quality)
                    "labels_3d": category_ids.cpu(),
                    "anchor": box.cpu(),
                }
            )

            # Add original class scores if quality was used
            if quality is not None:
                output[-1]["cls_scores"] = scores_origin.cpu()

            # Add instance IDs if provided
            if instance_id is not None:
                ids = instance_id[i, indices[i]]
                if self.score_threshold is not None:
                    ids = ids[mask[i]]
                output[-1]["instance_ids"] = ids.cpu()

            # Add instance features if provided (only for cooperative fusion)
            if instance_feature is not None:
                instance_feature = instance_feature[output_idx]
                instance_feature = instance_feature[i, indices[i] // num_cls]
                if self.score_threshold is not None:
                    instance_feature = instance_feature[mask[i]]
                output[-1]["instance_feature"] = instance_feature.cpu()
        return output
