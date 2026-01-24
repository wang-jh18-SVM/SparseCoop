import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.registry import PLUGIN_LAYERS
from projects.mmdet3d_plugin.core.box3d import *

from scipy.optimize import linear_sum_assignment
from typing import Dict, Optional, Set
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from enum import Enum


class MappingStatus(Enum):
    """Status of ID mapping entries."""

    ACTIVE = "active"
    STALE = "stale"
    EXPIRED = "expired"


@dataclass
class MappingEntry:
    """Individual mapping entry with metadata."""

    veh_id: int
    confidence: float
    last_updated: float
    creation_time: float
    update_count: int = 1
    status: MappingStatus = MappingStatus.ACTIVE


class CrossAgentIDMappingManager:
    """Enhanced ID mapping manager for cross-agent object tracking.

    Key improvements over simple dictionary:
    - Temporal decay with automatic cleanup
    - Conflict resolution based on confidence
    - Memory management with size limits
    - Comprehensive statistics and monitoring
    """

    def __init__(
        self,
        data_frequency: float = 10.0,  # Dataset frequency in Hz
        max_entries: int = 1000,  # Maximum number of ID mappings to store
        stale_threshold: float = 5.0,  # Mark mappings stale after N seconds
        expire_threshold: float = 30.0,  # Remove mappings after N seconds
        confidence_threshold: float = 0.3,  # Minimum confidence to keep mappings
        cleanup_interval: int = 50,  # Clean up every N updates
    ):
        # Configuration
        self.data_frequency = data_frequency
        self.max_entries = max_entries
        self.stale_threshold = stale_threshold
        self.expire_threshold = expire_threshold
        self.confidence_threshold = confidence_threshold
        self.cleanup_interval = cleanup_interval

        # Core storage - using OrderedDict for LRU-like behavior
        self.inf2veh_mappings: OrderedDict[int, MappingEntry] = OrderedDict()
        self.veh2inf_mappings: Dict[int, Set[int]] = defaultdict(set)

        # Counters
        self.update_counter = 0
        self.conflict_count = 0

        # Statistics
        self.stats = {
            'total_mappings_created': 0,
            'conflicts_resolved': 0,
            'cleanups_performed': 0,
        }

    def add_mapping(
        self,
        inf_id: int,
        veh_id: int,
        timestamp: float,
        confidence: float = 0.5,
        force_update: bool = False,
    ) -> bool:
        """Add or update an ID mapping."""
        # Check if mapping already exists
        if inf_id in self.inf2veh_mappings:
            existing_entry = self.inf2veh_mappings[inf_id]

            # Handle conflicts
            if existing_entry.veh_id != veh_id:
                if not force_update and not self._should_replace_mapping(
                    existing_entry, confidence, timestamp
                ):
                    self.conflict_count += 1
                    return False

                # Remove old reverse mapping
                self.veh2inf_mappings[existing_entry.veh_id].discard(inf_id)
                self.stats['conflicts_resolved'] += 1

            # Update existing mapping
            existing_entry.veh_id = veh_id
            existing_entry.confidence = max(existing_entry.confidence, confidence)
            existing_entry.last_updated = timestamp
            existing_entry.update_count += 1
            existing_entry.status = MappingStatus.ACTIVE

            # Move to end for LRU behavior
            self.inf2veh_mappings.move_to_end(inf_id)

        else:
            # Create new mapping
            new_entry = MappingEntry(
                veh_id=veh_id,
                confidence=confidence,
                last_updated=timestamp,
                creation_time=timestamp,
                update_count=1,
                status=MappingStatus.ACTIVE,
            )
            self.inf2veh_mappings[inf_id] = new_entry
            self.stats['total_mappings_created'] += 1

        # Update reverse mapping
        self.veh2inf_mappings[veh_id].add(inf_id)

        # Periodic cleanup
        self.update_counter += 1
        if self.update_counter % self.cleanup_interval == 0:
            self._cleanup_mappings(timestamp)

        return True

    def get_vehicle_id(self, inf_id: int, timestamp: float) -> Optional[int]:
        """Get vehicle ID for given infrastructure ID."""
        if inf_id not in self.inf2veh_mappings:
            return None

        entry = self.inf2veh_mappings[inf_id]

        # Check if mapping is still valid
        if self._is_mapping_valid(entry, timestamp):
            # Update access time and move to end
            entry.last_updated = timestamp
            self.inf2veh_mappings.move_to_end(inf_id)
            return entry.veh_id
        else:
            # Mark as expired
            entry.status = MappingStatus.EXPIRED
            return None

    def has_mapping(self, inf_id: int, timestamp: float) -> bool:
        """Check if infrastructure ID has an active mapping."""
        if inf_id not in self.inf2veh_mappings:
            return False

        entry = self.inf2veh_mappings[inf_id]
        return self._is_mapping_valid(entry, timestamp)

    # def get_inf_ids(self, veh_id: int, timestamp: float) -> Set[int]:
    #     """Get all infrastructure IDs mapped to a vehicle ID."""
    #     if veh_id not in self.veh2inf_mappings:
    #         return set()

    #     # Filter for active mappings only
    #     active_inf_ids = set()
    #     for inf_id in self.veh2inf_mappings[veh_id]:
    #         if inf_id in self.inf2veh_mappings:
    #             entry = self.inf2veh_mappings[inf_id]
    #             if self._is_mapping_valid(entry, timestamp):
    #                 active_inf_ids.add(inf_id)

    #     return active_inf_ids

    def remove_mapping(self, inf_id: int) -> bool:
        """Remove a specific mapping."""
        if inf_id not in self.inf2veh_mappings:
            return False

        entry = self.inf2veh_mappings[inf_id]
        veh_id = entry.veh_id

        # Remove from both mappings
        del self.inf2veh_mappings[inf_id]
        self.veh2inf_mappings[veh_id].discard(inf_id)

        # Clean up empty reverse mappings
        if not self.veh2inf_mappings[veh_id]:
            del self.veh2inf_mappings[veh_id]

        return True

    def update_confidence(
        self, inf_id: int, confidence: float, timestamp: float
    ) -> bool:
        """Update confidence for an existing mapping."""
        if inf_id not in self.inf2veh_mappings:
            return False

        entry = self.inf2veh_mappings[inf_id]
        entry.confidence = max(entry.confidence, confidence)
        entry.last_updated = timestamp
        entry.update_count += 1

        # Reactivate if was stale
        if entry.status == MappingStatus.STALE:
            entry.status = MappingStatus.ACTIVE

        # Move to end for LRU behavior
        self.inf2veh_mappings.move_to_end(inf_id)

        return True

    def reset_mappings(self) -> None:
        """Reset all mappings."""
        self.inf2veh_mappings.clear()
        self.veh2inf_mappings.clear()
        self.update_counter = 0
        self.conflict_count = 0

    # def get_statistics(self, timestamp: float) -> Dict:
    #     """Get comprehensive statistics about mappings."""

    #     active_count = 0
    #     stale_count = 0
    #     expired_count = 0

    #     for entry in self.inf2veh_mappings.values():
    #         time_since_update = timestamp - entry.last_updated

    #         if time_since_update > self.expire_threshold:
    #             expired_count += 1
    #         elif time_since_update > self.stale_threshold:
    #             stale_count += 1
    #         else:
    #             active_count += 1

    #     return {
    #         'total_entries': len(self.inf2veh_mappings),
    #         'active_mappings': active_count,
    #         'stale_mappings': stale_count,
    #         'expired_mappings': expired_count,
    #         'unique_vehicles': len(self.veh2inf_mappings),
    #         'conflicts_resolved': self.stats['conflicts_resolved'],
    #         'cleanups_performed': self.stats['cleanups_performed'],
    #         'update_counter': self.update_counter,
    #         'conflict_count': self.conflict_count,
    #     }

    def _should_replace_mapping(
        self, existing_entry: MappingEntry, new_confidence: float, current_time: float
    ) -> bool:
        """Determine if an existing mapping should be replaced."""
        # Replace if existing mapping is expired
        if current_time - existing_entry.last_updated > self.expire_threshold:
            return True

        # Replace if new mapping has significantly higher confidence
        if new_confidence > existing_entry.confidence + 0.15:
            return True

        # Replace if existing mapping is stale and new has decent confidence
        if (
            current_time - existing_entry.last_updated > self.stale_threshold
            and new_confidence > self.confidence_threshold
        ):
            return True

        return False

    def _is_mapping_valid(self, entry: MappingEntry, timestamp: float) -> bool:
        """Check if a mapping entry is still valid."""
        # # Check if expired
        # if timestamp - entry.last_updated > self.expire_threshold:
        #     return False

        # Check if confidence is too low
        if entry.confidence < self.confidence_threshold:
            return False

        # Use adaptive aging for expiration check
        aging_factor = self._adaptive_aging_factor(entry, timestamp)
        effective_expire_threshold = self.expire_threshold / aging_factor

        # Check if expired with adaptive threshold
        if timestamp - entry.last_updated > effective_expire_threshold:
            return False

        return True

    def _adaptive_aging_factor(self, entry: MappingEntry, current_time: float) -> float:
        """
        Calculate adaptive aging factor based on update frequency and confidence.
        Uses dynamic frequency calculation based on actual timestamps.

        Args:
            entry: Mapping entry to evaluate
            current_time: Current timestamp

        Returns:
            float: Aging factor (1.0 = normal aging, > 1.0 = faster aging, < 1.0 = slower aging)
        """
        time_since_creation = current_time - entry.creation_time

        # Avoid division by zero
        if time_since_creation <= 0:
            return 1.0

        # Calculate update frequency (updates per second)
        update_frequency = entry.update_count / time_since_creation

        frequency_ratio = update_frequency / self.data_frequency

        # High confidence + high frequency = slower aging (extend lifetime)
        confidence_factor = min(entry.confidence / 0.5, 1.0)  # Normalize to [0, 1]
        frequency_factor = min(frequency_ratio, 1.0)  # Cap at 1.0

        # Combined factor: higher values mean slower aging
        stability_factor = (confidence_factor + frequency_factor) / 2.0

        # Convert to aging factor: stable objects age slower
        aging_factor = 2.0 - stability_factor  # Range [1.0, 2.0]

        return aging_factor

    def _cleanup_mappings(self, timestamp: float) -> None:
        """Clean up stale and expired mappings with adaptive aging."""
        # Find expired mappings
        expired_inf_ids = []
        for inf_id, entry in self.inf2veh_mappings.items():
            time_since_update = timestamp - entry.last_updated

            # if time_since_update > self.expire_threshold:
            #     expired_inf_ids.append(inf_id)
            #     entry.status = MappingStatus.EXPIRED
            # elif time_since_update > self.stale_threshold:
            #     entry.status = MappingStatus.STALE

            # Use adaptive aging for both stale and expire thresholds
            aging_factor = self._adaptive_aging_factor(entry, timestamp)
            effective_expire_threshold = self.expire_threshold / aging_factor
            effective_stale_threshold = self.stale_threshold / aging_factor

            if time_since_update > effective_expire_threshold:
                expired_inf_ids.append(inf_id)
                entry.status = MappingStatus.EXPIRED
            elif time_since_update > effective_stale_threshold:
                entry.status = MappingStatus.STALE

        # Remove expired mappings
        for inf_id in expired_inf_ids:
            self.remove_mapping(inf_id)

        # Enforce max entries limit (LRU eviction)
        while len(self.inf2veh_mappings) > self.max_entries:
            # Remove oldest (least recently used) entry
            oldest_inf_id = next(iter(self.inf2veh_mappings))
            self.remove_mapping(oldest_inf_id)

        self.stats['cleanups_performed'] += 1

    # def export_mappings(self, timestamp: float) -> Dict:
    #     """Export mappings for debugging or persistence."""
    #     export_data = {
    #         'mappings': {},
    #         'statistics': self.get_statistics(timestamp),
    #         'metadata': {
    #             'export_time': timestamp,
    #             'total_entries': len(self.inf2veh_mappings),
    #         },
    #     }

    #     for inf_id, entry in self.inf2veh_mappings.items():
    #         export_data['mappings'][inf_id] = {
    #             'veh_id': entry.veh_id,
    #             'confidence': entry.confidence,
    #             'last_updated': entry.last_updated,
    #             'creation_time': entry.creation_time,
    #             'update_count': entry.update_count,
    #             'status': entry.status.value,
    #         }

    #     return export_data


@PLUGIN_LAYERS.register_module()
class CrossAgentSparseInteraction(nn.Module):

    def __init__(
        self,
        embed_dims=256,
        remove_ego_from_inf=False,
        coop_distance_threshold=60,
        coop_height_threshold=-3,
        coop_vis_image_bound=False,
        coop_vis_dist_threshold=-1,
        box_weights=[2.0] * 3 + [0.5] * 3 + [0.0] * 5,
        id_mapping_config=None,
        # Options for matching
        match_veh_conf_threshold=-1,
        match_dim_threshold=-1,
        match_feat_weight=0.0,
        # Ablation Flags
        coop_wo_coarse_fusion=False,
    ):
        super(CrossAgentSparseInteraction, self).__init__()

        self.embed_dims = embed_dims
        self.match_veh_conf_threshold = match_veh_conf_threshold
        self.remove_ego_from_inf = remove_ego_from_inf
        self.coop_distance_threshold = coop_distance_threshold
        self.coop_height_threshold = coop_height_threshold
        self.coop_vis_image_bound = coop_vis_image_bound
        self.coop_vis_dist_threshold = coop_vis_dist_threshold
        # self.coop_confidence_topk = coop_confidence_topk
        self.match_dim_threshold = match_dim_threshold
        self.match_feat_weight = match_feat_weight

        # cross-agent feature alignment
        self.cross_agent_align = nn.Linear(self.embed_dims + 9, self.embed_dims)
        # self.cross_agent_align = nn.Sequential(
        #     nn.Linear(self.embed_dims+9, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, self.embed_dims),
        # )

        # cross-agent feature fusion
        if not coop_wo_coarse_fusion:
            self.cross_agent_fusion = nn.Linear(self.embed_dims, self.embed_dims)
            # self.cross_agent_fusion = nn.Linear(self.embed_dims*2, self.embed_dims)
            # self.cross_agent_fusion = nn.Sequential(
            #     nn.Linear(self.embed_dims*2, self.embed_dims),
            #     nn.ReLU(),
            #     nn.Linear(self.embed_dims, self.embed_dims),
            # )
        else:
            self.cross_agent_fusion = nn.Identity()

        # self.veh_low_score_replace = nn.Linear(2 * self.embed_dims, self.embed_dims)

        # Enhanced ID mapping manager
        if id_mapping_config is None:
            id_mapping_config = {
                'data_frequency': 10.0,
                'max_entries': 1000,
                'stale_threshold': 5.0,
                'expire_threshold': 30.0,
                'confidence_threshold': 0.3,
                'cleanup_interval': 50,
            }
        self.id_mapping_manager = CrossAgentIDMappingManager(**id_mapping_config)

        # For box cost
        self.box_weights = box_weights

        for p in self.cross_agent_align.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        # Initialize fusion as zeros to avoid negative influence at beginning
        for p in self.cross_agent_fusion.parameters():
            if p.dim() > 1:
                nn.init.zeros_(p)

    def _box_cost(self, veh_anchors, inf_anchors):
        """
        Calculate box-based cost matrix using L1 distance between anchors with weighted costs
        """
        # Expand anchors for pairwise comparison
        veh_anchors_expanded = veh_anchors.unsqueeze(1).expand(
            -1, inf_anchors.shape[0], -1
        )  # (veh_nums, inf_nums, anchor_dims)
        inf_anchors_expanded = inf_anchors.unsqueeze(0).expand(
            veh_anchors.shape[0], -1, -1
        )  # (veh_nums, inf_nums, anchor_dims)

        # Calculate L1 distance for each dimension
        l1_distances = torch.abs(
            veh_anchors_expanded - inf_anchors_expanded
        )  # (veh_nums, inf_nums, anchor_dims)

        # Apply box weights to get weighted cost matrix
        # Ensure box_weights tensor is on the same device
        box_weights = torch.tensor(
            self.box_weights, device=veh_anchors.device, dtype=veh_anchors.dtype
        )

        # Calculate weighted sum across anchor dimensions
        cost_matrix = torch.sum(
            l1_distances * box_weights.unsqueeze(0).unsqueeze(0), dim=-1
        )  # (veh_nums, inf_nums)

        return cost_matrix

    def _feat_cost(self, veh_features, inf_features):
        """
        Calculate feature-based cost matrix using cosine similarity between features
        """
        cos_sim = torch.nn.functional.cosine_similarity(
            veh_features.unsqueeze(1), inf_features.unsqueeze(0), dim=-1
        )
        return 1 - cos_sim

    def _query_matching(
        self,
        veh_features,
        veh_anchors,
        veh_confidence,
        inf_features,
        inf_anchors,
    ):
        # Calculate box-based cost matrix using L1 distance with weights
        cost_matrix = self._box_cost(veh_anchors, inf_anchors)

        # Add feature-based cost matrix (optional)
        if self.match_feat_weight > 0:
            feat_cost_matrix = self._feat_cost(veh_features, inf_features)
            cost_matrix = cost_matrix + self.match_feat_weight * feat_cost_matrix

        # Filter by vehicle confidence (optional)
        if self.match_veh_conf_threshold > 0:
            veh_conf_mask = veh_confidence >= self.match_veh_conf_threshold
            cost_matrix[~veh_conf_mask, :] = 1e6

        # Filter by relative dimensions (optional)
        if self.match_dim_threshold > 0:
            veh_nums = veh_anchors.shape[0]
            inf_nums = inf_anchors.shape[0]

            # Extract position and dimensions from anchors and decode to meters
            veh_pos = veh_anchors[:, [X, Y, Z]]
            inf_pos = inf_anchors[:, [X, Y, Z]]
            veh_dims = veh_anchors[:, [L]].exp()

            # Expand position and dimensions for pairwise comparison
            veh_pos_expanded = veh_pos.unsqueeze(1).expand(-1, inf_nums, -1)
            inf_pos_expanded = inf_pos.unsqueeze(0).expand(veh_nums, -1, -1)
            veh_dims_expanded = veh_dims.unsqueeze(1).expand(-1, inf_nums, -1)

            # Calculate relative distance between position and dimensions
            diff = torch.abs(veh_pos_expanded - inf_pos_expanded) / veh_dims_expanded

            # Filter by relative dimensions
            filter_mask = (diff[..., 0] <= self.match_dim_threshold) & (
                diff[..., 1] <= self.match_dim_threshold
            )

            # Set infinite cost for unmatched pairs
            cost_matrix[~filter_mask] = 1e6

        # Hungarian matching to find optimal assignment
        cost_matrix_np = cost_matrix.detach().cpu().numpy()
        paired_veh_idx, paired_inf_idx = linear_sum_assignment(cost_matrix_np)

        return (
            torch.tensor(paired_veh_idx).to(device=cost_matrix.device),
            torch.tensor(paired_inf_idx).to(device=cost_matrix.device),
            cost_matrix,
        )

    def _query_matching_id(self, veh_instance_id, inf_instance_id):
        """
        Query matching using instance ID. Find the same id in both veh and inf to be a pair, set others costs as 1e6.

        veh_instance_id: [Nv]
        inf_instance_id: [Ni]

        Returns:
            paired_veh_idx: [Np]
            paired_inf_idx: [Np]
            cost_matrix: [Nv, Ni]
        """
        veh_nums = veh_instance_id.shape[0]
        inf_nums = inf_instance_id.shape[0]

        # Initialize cost matrix with high cost
        cost_matrix = torch.full(
            (veh_nums, inf_nums),
            1e6,
            device=veh_instance_id.device,
            dtype=torch.float32,
        )

        # Create expanded tensors for vectorized comparison
        veh_id_expanded = veh_instance_id.unsqueeze(1).expand(-1, inf_nums)  # [Nv, Ni]
        inf_id_expanded = inf_instance_id.unsqueeze(0).expand(veh_nums, -1)  # [Nv, Ni]

        # Find matching IDs (excluding -1 which means untracked)
        matching_mask = (veh_id_expanded == inf_id_expanded) & (veh_id_expanded != -1)

        # Set low cost for matching pairs
        cost_matrix[matching_mask] = 0.0

        # Directly find matching pairs without Hungarian assignment
        paired_veh_idx, paired_inf_idx = torch.where(matching_mask)

        # Verify that matched pairs have the same IDs
        if len(paired_veh_idx) > 0:
            assert torch.all(
                veh_instance_id[paired_veh_idx] == inf_instance_id[paired_inf_idx]
            ), (
                f"veh_instance_id: {veh_instance_id[paired_veh_idx]}, "
                f"inf_instance_id: {inf_instance_id[paired_inf_idx]}"
            )
        return (
            paired_veh_idx,
            paired_inf_idx,
            cost_matrix,
        )

    def _calculate_3d_iou(self, boxes1, boxes2):
        """
        Calculate 3D IoU between two sets of 3D bounding boxes using mmcv's optimized rotated IoU.

        Args:
            boxes1: (N, 11) tensor with format [x, y, z, l, w, h, sin_yaw, cos_yaw, vx, vy, vz]
            boxes2: (M, 11) tensor with format [x, y, z, l, w, h, sin_yaw, cos_yaw, vx, vy, vz]

        Returns:
            iou_matrix: (N, M) tensor of 3D IoU values
        """
        from mmcv.ops import boxes_iou3d

        N, M = boxes1.shape[0], boxes2.shape[0]

        if N == 0 or M == 0:
            return torch.zeros((N, M), device=boxes1.device, dtype=boxes1.dtype)

        # Convert from internal format to mmcv format [x, y, z, dx, dy, dz, heading]
        def convert_to_mmcv_format(boxes):
            # Extract centers
            centers = boxes[:, [X, Y, Z]]  # (N, 3)

            # Decode dimensions from log space
            dims = boxes[:, [L, W, H]].exp()  # (N, 3)

            # Calculate yaw angles from sin/cos encoding
            yaw = torch.atan2(boxes[:, SIN_YAW], boxes[:, COS_YAW])  # (N,)

            # Combine into mmcv format: [x, y, z, dx, dy, dz, heading]
            mmcv_boxes = torch.cat(
                [
                    centers,  # x, y, z
                    dims,  # dx (length), dy (width), dz (height)
                    yaw.unsqueeze(-1),  # heading
                ],
                dim=-1,
            )

            return mmcv_boxes

        # Convert both sets of boxes
        mmcv_boxes1 = convert_to_mmcv_format(boxes1)  # (N, 7)
        mmcv_boxes2 = convert_to_mmcv_format(boxes2)  # (M, 7)

        # Use mmcv's 3D IoU calculation
        iou_matrix = boxes_iou3d(mmcv_boxes1, mmcv_boxes2)  # (N, M)

        return iou_matrix

    def _query_matching_iou(
        self,
        veh_anchors,
        veh_confidence,
        inf_anchors,
    ):
        """
        Query matching using 3D IoU as cost metric.
        Higher IoU means lower cost, zero IoU means infinite cost.
        """
        # Create confidence mask for vehicle instances (filter low confidence)
        veh_conf_mask = veh_confidence >= self.match_veh_conf_threshold

        # Calculate 3D IoU matrix
        iou_matrix = self._calculate_3d_iou(
            veh_anchors, inf_anchors
        )  # (veh_nums, inf_nums)

        # Convert IoU to cost: cost = 1 - IoU (higher IoU -> lower cost)
        # Zero IoU will result in cost = 1, very low IoU will have high cost
        cost_matrix = 1.0 - iou_matrix

        # Set infinite cost for zero IoU (optional, since 1-0=1 is already high cost)
        zero_iou_mask = iou_matrix <= 1e-8
        cost_matrix[zero_iou_mask] = 1e6

        # Apply confidence mask to cost matrix
        cost_matrix[~veh_conf_mask, :] = 1e6

        # Hungarian matching to find optimal assignment
        cost_matrix_np = cost_matrix.detach().cpu().numpy()
        paired_veh_idx, paired_inf_idx = linear_sum_assignment(cost_matrix_np)

        return (
            torch.tensor(paired_veh_idx).to(device=cost_matrix.device),
            torch.tensor(paired_inf_idx).to(device=cost_matrix.device),
            cost_matrix,
        )

    def _query_fusion(
        self, veh_features, inf_features, paired_veh_idx, paired_inf_idx, cost_matrix
    ):
        """
        Query fusion:
            replacement for scores, ref_pts and pos_embed according to confidence_score
            fusion for features via MLP

        inf: Instance from infrastructure
        veh: Instance from vehicle
        inf_idx: matched idxs for inf side
        veh_idx: matched idxs for veh side
        cost_matrix
        """
        accept_veh_idx = []
        accept_inf_idx = []
        mask = cost_matrix[paired_veh_idx, paired_inf_idx] < 1e5

        accept_veh_idx = paired_veh_idx[mask]
        accept_inf_idx = paired_inf_idx[mask]

        # We have done both explicit and implicit alignment in instance bank
        # Here we just do simple linear fusion
        fused_features = veh_features[accept_veh_idx] + self.cross_agent_fusion(
            inf_features[accept_inf_idx]
        )

        # Get all indices
        all_veh_idx = torch.arange(veh_features.shape[0], device=veh_features.device)
        all_inf_idx = torch.arange(inf_features.shape[0], device=inf_features.device)

        # Find unmatched indices by excluding accepted ones
        unmatched_veh_idx = all_veh_idx[~torch.isin(all_veh_idx, accept_veh_idx)]
        unmatched_inf_idx = all_inf_idx[~torch.isin(all_inf_idx, accept_inf_idx)]
        assert len(unmatched_veh_idx) + len(accept_veh_idx) == veh_features.shape[0]
        assert len(unmatched_inf_idx) + len(accept_inf_idx) == inf_features.shape[0]

        return (
            fused_features,
            accept_veh_idx,
            accept_inf_idx,
            unmatched_veh_idx,
            unmatched_inf_idx,
        )

    def _query_replacement(
        self,
        unmatched_veh_confidence,
        unmatched_inf_confidence,
        replace_nums,
    ):
        """
        Query replacement: replace low-confidence vehicle-side query with unmatched inf-side query
        """
        # Sort unmatched vehicle instances(negative confidence)
        _, select_veh_idx = torch.topk(
            -unmatched_veh_confidence,
            replace_nums,
        )

        # Sort unmatched infrastructure instances(positive confidence)
        _, select_inf_idx = torch.topk(
            unmatched_inf_confidence,
            replace_nums,
        )

        return select_veh_idx, select_inf_idx

    def inf_filter(
        self,
        inf_pos,
        lidar2img_rt=None,
        image_wh=None,
    ):
        """
        Filter out ego vehicle and distant objects in infrastructure views
        """
        # Remove distant objects in infrastructure views
        inf_distance = torch.norm(inf_pos, dim=-1)
        inf_valid_mask = (inf_distance <= self.coop_distance_threshold) & (
            inf_pos[..., 2] > self.coop_height_threshold
        )

        # Remove ego vehicle in infrastructure views
        if inf_valid_mask.sum() > 0 and self.remove_ego_from_inf:
            H_B, H_F = -2.04, 2.04  # H = 4.084
            W_L, W_R = -0.92, 0.92  # W = 1.85

            inf_ego_mask = (
                (inf_pos[..., 0] >= H_B)
                & (inf_pos[..., 0] <= H_F)
                & (inf_pos[..., 1] >= W_L)
                & (inf_pos[..., 1] <= W_R)
            )

            inf_valid_mask = torch.logical_and(inf_valid_mask, ~inf_ego_mask)

        # Visible in any image bound
        if self.coop_vis_image_bound:
            assert lidar2img_rt is not None and image_wh is not None
            inf_pos_extend = torch.cat(
                [inf_pos, torch.ones_like(inf_pos[..., :1])], dim=-1
            )  # (N_inf, 4)

            inf_pos_img = torch.matmul(
                lidar2img_rt, inf_pos_extend[None, :, :].transpose(1, 2)
            ).transpose(
                1, 2
            )  # (N_cam, N_inf, 4)

            inf_pos_img = inf_pos_img[..., :2] / torch.clamp(
                inf_pos_img[..., 2:3], min=1e-5
            )

            # Check if points fall within image boundaries
            in_img_mask = (
                (inf_pos_img[..., 0] >= 0)
                & (inf_pos_img[..., 0] < image_wh[:, 0].unsqueeze(1))
                & (inf_pos_img[..., 1] >= 0)
                & (inf_pos_img[..., 1] < image_wh[:, 1].unsqueeze(1))
            )

            # Object is visible if it appears in any camera view
            visible_in_any_cam = in_img_mask.any(dim=0)
        else:
            visible_in_any_cam = torch.ones_like(inf_valid_mask, dtype=torch.bool)

        # Visible in distance threshold
        if self.coop_vis_dist_threshold > 0:
            visible_distance_mask = inf_distance <= self.coop_vis_dist_threshold
        else:
            visible_distance_mask = torch.ones_like(inf_valid_mask, dtype=torch.bool)

        visible_mask = visible_in_any_cam & visible_distance_mask

        # Instances valid(within distance threshold) and visible are involved in cooperative interaction
        coop_interaction_mask = inf_valid_mask & visible_mask
        # Instances valid(within distance threshold) but not visible are directly included in post-processing
        coop_direct_mask = inf_valid_mask & ~visible_mask

        # # if no valid objects, set the most near object as valid
        # if inf_valid_mask.sum() == 0:
        #     closest_idx = torch.argmin(inf_distance)
        #     inf_valid_mask[closest_idx] = True

        return coop_interaction_mask, coop_direct_mask

    def forward(
        self,
        veh_features,
        veh_anchors,
        veh_confidence,
        veh_instance_id,
        veh_type,
        inf_features,
        inf_anchors,
        inf_confidence,
        inf_instance_id,
        inf_type,
        timestamp: float,
    ):
        """
        Query-based cross-agent interaction
        """
        if inf_anchors.shape[0] == 0:
            # dummy path to include the fusion layer
            dummy = self.cross_agent_fusion(torch.zeros_like(inf_features[0:1]))
            veh_features = veh_features + 0 * dummy.sum()
            return veh_features, veh_anchors

        ##### Query matching between vehicle and infrastructure #####
        if veh_type == "denoising" and inf_type == "denoising":
            paired_veh_idx, paired_inf_idx, cost_matrix = self._query_matching_id(
                veh_instance_id, inf_instance_id
            )
        else:
            paired_veh_idx, paired_inf_idx, cost_matrix = self._query_matching(
                veh_features,
                veh_anchors,
                veh_confidence,
                inf_features,
                inf_anchors,
            )

        ##### Query fusion for matched pairs #####
        if len(paired_inf_idx) > 0:
            (
                fused_features,
                accept_veh_idx,
                accept_inf_idx,
                unmatched_veh_idx,
                unmatched_inf_idx,
            ) = self._query_fusion(
                veh_features, inf_features, paired_veh_idx, paired_inf_idx, cost_matrix
            )
        if len(paired_inf_idx) == 0 or len(accept_inf_idx) == 0:
            # dummy path to include the fusion layer
            dummy = self.cross_agent_fusion(torch.zeros_like(inf_features[0:1]))
            veh_features = veh_features + 0 * dummy.sum()
            return veh_features, veh_anchors

        # Update ID mapping
        if (
            veh_instance_id is not None
            and inf_instance_id is not None
            and veh_type == "normal"
            and inf_type == "normal"
        ):
            for i, inf_idx in enumerate(accept_inf_idx):
                veh_tracked = veh_instance_id[accept_veh_idx[i]] != -1
                inf_tracked = self.id_mapping_manager.has_mapping(
                    inf_instance_id[inf_idx].item(), timestamp
                )

                # Calculate confidence for this pair based on matching quality
                pair_confidence = float(veh_confidence[accept_veh_idx[i]])
                if inf_confidence is not None:
                    pair_confidence = max(
                        pair_confidence, float(inf_confidence[inf_idx])
                    )

                if veh_tracked:
                    # Add/update mapping from infrastructure to vehicle
                    self.id_mapping_manager.add_mapping(
                        inf_instance_id[inf_idx].item(),
                        veh_instance_id[accept_veh_idx[i]].item(),
                        timestamp=timestamp,
                        confidence=pair_confidence,
                    )
                elif inf_tracked:
                    # Use existing mapping
                    mapped_veh_id = self.id_mapping_manager.get_vehicle_id(
                        inf_instance_id[inf_idx].item(), timestamp
                    )
                    if mapped_veh_id is not None:
                        veh_instance_id[accept_veh_idx[i]] = mapped_veh_id
                        # Update confidence for this mapping
                        self.id_mapping_manager.update_confidence(
                            inf_instance_id[inf_idx].item(),
                            pair_confidence,
                            timestamp,
                        )

        # Initialize cooperative features and anchors with original vehicle data
        cooperative_features = veh_features.clone()
        cooperative_anchors = veh_anchors.clone()

        # Update matched pairs with fused features
        cooperative_features[accept_veh_idx] = fused_features

        if inf_confidence is not None and veh_type == "normal" and inf_type == "normal":
            ##### Query complementation #####
            # Find unmatched infrastructure instances
            unmatched_inf_features = inf_features[unmatched_inf_idx]
            unmatched_inf_anchors = inf_anchors[unmatched_inf_idx]
            unmatched_inf_confidence = inf_confidence[unmatched_inf_idx]
            unmatched_inf_instance_id = inf_instance_id[unmatched_inf_idx]

            # Find unmatched and untracked vehicle instances
            if veh_instance_id is not None:
                unmatched_veh_idx = unmatched_veh_idx[
                    veh_instance_id[unmatched_veh_idx] == -1
                ]
            else:
                # Only replace vehicle instances within last 1/3 indices(means not temporal)
                veh_nums = veh_features.shape[0]
                unmatched_veh_idx = unmatched_veh_idx[
                    unmatched_veh_idx >= veh_nums * 2 / 3
                ]

            # Identify low-confidence vehicle instances to be replaced
            replace_nums = min(len(unmatched_inf_idx), len(unmatched_veh_idx))
            if replace_nums > 0:
                unmatched_veh_confidence = veh_confidence[unmatched_veh_idx]
                select_veh_idx, select_inf_idx = self._query_replacement(
                    unmatched_veh_confidence,
                    unmatched_inf_confidence,
                    replace_nums=min(
                        len(unmatched_inf_idx),
                        len(unmatched_veh_idx),
                    ),
                )

                # Convert indices to the original vehicle indices
                orig_veh_idx = unmatched_veh_idx[select_veh_idx]

                # Replace low-confidence vehicle instances with high-confidence infrastructure instances
                cooperative_features[orig_veh_idx] = unmatched_inf_features[
                    select_inf_idx
                ]
                cooperative_anchors[orig_veh_idx] = unmatched_inf_anchors[
                    select_inf_idx
                ]

                # Update replaced instances ID with inf ID
                if veh_instance_id is not None:
                    for i, veh_idx in enumerate(orig_veh_idx):
                        inf_id = unmatched_inf_instance_id[select_inf_idx[i]].item()

                        # Check if infrastructure ID already has a mapping
                        mapped_veh_id = self.id_mapping_manager.get_vehicle_id(
                            inf_id, timestamp
                        )

                        if mapped_veh_id is not None:
                            # Use existing mapping
                            veh_instance_id[veh_idx] = mapped_veh_id
                            # Update confidence for this mapping
                            replacement_confidence = float(
                                unmatched_inf_confidence[select_inf_idx[i]]
                            )
                            self.id_mapping_manager.update_confidence(
                                inf_id,
                                replacement_confidence,
                                timestamp,
                            )
                        else:
                            # No existing mapping - infrastructure ID will be handled at higher level
                            # Keep the original vehicle ID for now
                            pass

        return cooperative_features, cooperative_anchors

    def reset_id_mappings(self):
        """Reset all ID mappings."""
        self.id_mapping_manager.reset_mappings()
