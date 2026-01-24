from typing import List, Optional, Tuple, Union
import warnings

import numpy as np
import torch
import torch.nn as nn

from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmcv.runner import BaseModule, force_fp32
from mmcv.utils import build_from_cfg
from mmdet.core.bbox.builder import BBOX_SAMPLERS
from mmdet.core.bbox.builder import BBOX_CODERS
from mmdet.models import HEADS, LOSSES
from mmdet.core import reduce_mean

from .blocks import DeformableFeatureAggregation as DFG

__all__ = ["SparseCoopHead"]


@HEADS.register_module()
class SparseCoopHead(BaseModule):
    """Detection Head for SparseCoop.

    This head processes multi-view image features and outputs 3D bounding box
    predictions. It utilizes instance queries, temporal modeling, graph neural
    networks (GNNs), deformable attention, and a denoising strategy during
    training.
    """

    def __init__(
        self,
        instance_bank: dict,
        anchor_encoder: dict,
        graph_model: dict,
        norm_layer: dict,
        ffn: dict,
        deformable_model: dict,
        refine_layer: dict,
        num_decoder: int = 6,
        num_single_frame_decoder: int = -1,
        num_single_agent_decoder: int = -1,
        coop_graph_model: dict = None,
        temp_graph_model: dict = None,
        loss_cls: dict = None,
        loss_reg: dict = None,
        decoder: dict = None,
        sampler: dict = None,
        gt_cls_key: str = "gt_labels_3d",
        gt_reg_key: str = "gt_bboxes_3d",
        reg_weights: List = None,
        operation_order: Optional[List[str]] = None,
        cls_threshold_to_reg: float = -1,
        dn_loss_weight: float = 5.0,
        decouple_attn: bool = True,
        init_cfg: dict = None,
        num_heads: int = 8,
        is_cooperative: bool = False,
        freeze_ego_agent: bool = False,
        use_temp_gnn_for_coop: bool = False,
        # Coop
        save_dn_for_coop: bool = False,
        # Ablation Flags
        coop_cross_attn_wo_dn: bool = False,
        coop_wo_coarse_fusion: bool = False,
        **kwargs,
    ):
        """Initializes the SparseCoopHead.

        Args:
            instance_bank (dict): Configuration for the instance bank module.
            anchor_encoder (dict): Configuration for the anchor encoder.
            graph_model (dict): Configuration for the spatial GNN module.
            norm_layer (dict): Configuration for normalization layers.
            ffn (dict): Configuration for the feed-forward network.
            deformable_model (dict): Configuration for the deformable attention module.
            refine_layer (dict): Configuration for the refinement layer (predicts box deltas).
            num_decoder (int): Total number of decoder layers (transformer blocks). Defaults to 6.
            num_single_frame_decoder (int): Number of decoder layers used for single-frame processing
                before temporal instance update. Defaults to -1 (no single-frame update).
            temp_graph_model (dict, optional): Configuration for the temporal GNN module. Defaults to None.
            loss_cls (dict, optional): Configuration for the classification loss. Defaults to None.
            loss_reg (dict, optional): Configuration for the regression loss. Defaults to None.
            decoder (dict, optional): Configuration for the bounding box decoder. Defaults to None.
            sampler (dict, optional): Configuration for the sampler (assigns targets and handles denoising).
                Defaults to None.
            gt_cls_key (str): Key for ground truth classification labels in the data dictionary.
            gt_reg_key (str): Key for ground truth regression targets (bboxes) in the data dictionary.
            reg_weights (List, optional): Weights for different components of the regression target.
                Defaults to None (all weights are 1.0).
            operation_order (Optional[List[str]]): Specifies the order of operations within the decoder layers.
                If None, a default order is used.
            cls_threshold_to_reg (float): Classification score threshold to apply regression loss.
                Defaults to -1 (apply to all positive samples).
            dn_loss_weight (float): Weight for the denoising loss. Defaults to 5.0.
            decouple_attn (bool): Whether to decouple query and key features in GNN attention. Defaults to True.
            init_cfg (dict, optional): Initialization configuration. Defaults to None.
            num_heads (int): Number of attention heads. Defaults to 8.
        """
        super(SparseCoopHead, self).__init__(init_cfg)
        self.num_decoder = num_decoder
        self.num_single_frame_decoder = num_single_frame_decoder
        self.num_single_agent_decoder = num_single_agent_decoder
        self.gt_cls_key = gt_cls_key
        self.gt_reg_key = gt_reg_key
        self.cls_threshold_to_reg = cls_threshold_to_reg
        self.dn_loss_weight = dn_loss_weight
        self.decouple_attn = decouple_attn
        self.save_dn_for_coop = save_dn_for_coop

        if reg_weights is None:
            self.reg_weights = [1.0] * 10
        else:
            self.reg_weights = reg_weights

        if operation_order is None:
            operation_order = [
                "coop_gnn",
                "temp_gnn",
                "gnn",
                "norm",
                "deformable",
                "norm",
                "ffn",
                "norm",
                "refine",
            ] * num_decoder
            # delete the 'gnn' and 'norm' layers in the first transformer blocks
            operation_order = operation_order[3:]
        self.operation_order = operation_order

        self.use_temp_gnn_for_coop = use_temp_gnn_for_coop
        if self.use_temp_gnn_for_coop:
            assert (
                "coop_gnn" not in operation_order
            ), "coop_gnn should not be built explicitly if use_temp_gnn_for_coop is True"

        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.instance_bank = build(instance_bank, PLUGIN_LAYERS)
        self.anchor_encoder = build(anchor_encoder, POSITIONAL_ENCODING)
        self.sampler = build(sampler, BBOX_SAMPLERS)
        self.decoder = build(decoder, BBOX_CODERS)
        self.loss_cls = build(loss_cls, LOSSES)
        self.loss_reg = build(loss_reg, LOSSES)
        self.op_config_map = {
            "coop_gnn": [coop_graph_model, ATTENTION],
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "deformable": [deformable_model, ATTENTION],
            "refine": [refine_layer, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        self.num_heads = num_heads
        self.is_cooperative = is_cooperative
        if freeze_ego_agent:
            assert num_single_frame_decoder <= num_single_agent_decoder <= num_decoder
            # Calculate based on the actual operation_order structure:
            # single_frame_decoder blocks: 6 ops each
            # single_agent_decoder blocks (after single_frame): 7 ops each
            # Plus account for [2:] slice that removes first 2 operations
            single_agent_layers = (6 * num_single_frame_decoder - 2) + 7 * (
                num_single_agent_decoder - num_single_frame_decoder
            )

            # Freeze only the single-agent layers
            for i, op in enumerate(operation_order[:single_agent_layers]):
                if self.layers[i] is not None:
                    self.layers[i].eval()
                    for param in self.layers[i].parameters():
                        param.requires_grad = False

        self.embed_dims = self.instance_bank.embed_dims
        if self.decouple_attn:
            self.fc_before = nn.Linear(self.embed_dims, self.embed_dims * 2, bias=False)
            self.fc_after = nn.Linear(self.embed_dims * 2, self.embed_dims, bias=False)
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        # Ablation Flags
        self.coop_cross_attn_wo_dn = coop_cross_attn_wo_dn
        self.coop_wo_coarse_fusion = coop_wo_coarse_fusion

    def init_weights(self):
        """Initializes the weights of the head modules."""
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)
        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def graph_model(
        self, index, query, key=None, value=None, query_pos=None, key_pos=None, **kwargs
    ):
        """Applies a GNN (spatial or temporal) layer with optional decoupling.

        Args:
            index (int): Index of the layer in `self.layers`.
            query (torch.Tensor): Query features.
            key (torch.Tensor, optional): Key features. Defaults to None.
            value (torch.Tensor, optional): Value features. Defaults to None.
            query_pos (torch.Tensor, optional): Positional embeddings for queries. Defaults to None.
            key_pos (torch.Tensor, optional): Positional embeddings for keys. Defaults to None.
            **kwargs: Additional arguments passed to the GNN layer.

        Returns:
            torch.Tensor: Output features from the GNN layer.
        """
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query, key, value, query_pos=query_pos, key_pos=key_pos, **kwargs
            )
        )

    def forward(self, feature_maps: Union[torch.Tensor, List], metas: dict):
        """Forward pass of the SparseCoopHead.

        Args:
            feature_maps (Union[torch.Tensor, List]): Input feature maps from the backbone/neck,
                typically a list of features from different FPN levels.
            metas (dict): Meta information for the batch, including image metas, ground truths, etc.

        Returns:
            dict: A dictionary containing predictions and intermediate results:
                - 'classification': List of classification scores from each refine layer.
                - 'prediction': List of bounding box predictions from each refine layer.
                - 'quality': List of predicted quality scores (e.g., IoU) from each refine layer.
                - 'dn_classification' (training only): List of denoising classification scores.
                - 'dn_prediction' (training only): List of denoising bounding box predictions.
                - 'dn_reg_target' (training only): Denoising regression targets.
                - 'dn_cls_target' (training only): Denoising classification targets.
                - 'dn_valid_mask' (training only): Mask indicating valid denoising samples.
                - 'temp_dn_reg_target' (training only, if applicable): Temporal denoising targets.
                - 'temp_dn_cls_target' (training only, if applicable): Temporal denoising targets.
                - 'temp_dn_valid_mask' (training only, if applicable): Temporal denoising mask.
                - 'dn_id_target' (training only, if applicable): IDs for temporal denoising matching.
                - 'instance_id' (inference only): Predicted instance IDs for tracking.
        """
        if isinstance(feature_maps, torch.Tensor):
            feature_maps = [feature_maps]
        batch_size = feature_maps[0].shape[0]

        # ========= Retrieve instance bank data ============
        # The instance bank maintains a collection of learnable instance queries and anchors,
        # as well as temporal instances cached from previous frames.
        # This is crucial for tracking objects across time and handling 4D processing.

        # Reset denoising metas if batch size changed
        if (
            self.sampler.dn_metas is not None
            and self.sampler.dn_metas["dn_anchor"].shape[0] != batch_size
        ):
            self.sampler.dn_metas = None

        # Get instance data from the instance bank:
        # - instance_feature: Current frame learnable instance queries [B, num_queries, embed_dims]
        # - anchor: Current frame anchor boxes [B, num_queries, 11] (x,y,z,l,w,h,sin,cos,vx,vy,vz)
        # - temp_instance_feature: Features from previous frame (cached) [B, num_temp, embed_dims]
        # - temp_anchor: Anchors from previous frame (cached) [B, num_temp, 11]
        # - time_interval: Time difference between current and previous frames
        # - coop_instance_feature: Cooperative features from previous frame (cached) [B, num_coop, embed_dims]
        # - coop_anchor: Cooperative anchors from previous frame (cached) [B, num_coop, 11]
        # - coop_valid_mask: Valid mask for cooperative features, false means invalid padded zeros
        (
            instance_feature,
            anchor,
            temp_instance_feature,
            temp_anchor,
            time_interval,
            coop_instance_feature,
            coop_anchor,
            coop_valid_mask,
            coop_dn_valid_mask,
        ) = self.instance_bank.get(
            batch_size,
            metas,
            dn_metas=self.sampler.dn_metas,
            feature_maps=feature_maps,
            is_cooperative=self.is_cooperative,
        )

        # ========= prepare for denosing training ============
        # 1. get dn metas: noisy-anchors and corresponding GT
        # 2. concat learnable instances and noisy instances
        # 3. get attention mask
        attn_mask = None
        dn_metas = None
        temp_dn_reg_target = None  # Placeholder for temporal denoising targets

        # Generate denoising training data if in training mode
        if (self.training or self.save_dn_for_coop) and hasattr(
            self.sampler, "get_dn_anchors"
        ):
            # Check if instance IDs are available for temporal denoising matching
            if "instance_id" in metas["img_metas"][0]:
                gt_instance_id = [
                    x["instance_id"].data.cuda() for x in metas["img_metas"]
                ]
            else:
                gt_instance_id = None

            # Get noisy anchors and corresponding targets for denoising training
            # This simulates both box jittering and label noise to improve robustness
            dn_metas = self.sampler.get_dn_anchors(
                metas[self.gt_cls_key], metas[self.gt_reg_key], gt_instance_id
            )

        # Process denoising data if available
        if dn_metas is not None:
            (
                dn_anchor,  # Noisy anchor boxes
                dn_reg_target,  # Regression targets to denoise back to GT
                dn_cls_target,  # Classification targets for denoising
                dn_attn_mask,  # Attention mask preventing inter-group leakage
                valid_mask,  # Mask for valid denoising anchors
                dn_id_target,  # Instance IDs for tracking objects temporally
            ) = dn_metas
            num_dn_anchor = dn_anchor.shape[1]

            # Pad dn_anchor if needed to match dimension of normal anchors
            if dn_anchor.shape[-1] != anchor.shape[-1]:
                remain_state_dims = anchor.shape[-1] - dn_anchor.shape[-1]
                dn_anchor = torch.cat(
                    [
                        dn_anchor,
                        dn_anchor.new_zeros(
                            batch_size, num_dn_anchor, remain_state_dims
                        ),
                    ],
                    dim=-1,
                )

            # Concatenate learnable instances with denoising instances
            anchor = torch.cat([anchor, dn_anchor], dim=1)

            # Initialize denoising instance features as zero
            instance_feature = torch.cat(
                [
                    instance_feature,
                    instance_feature.new_zeros(
                        batch_size, num_dn_anchor, instance_feature.shape[-1]
                    ),
                ],
                dim=1,
            )
            num_instance = instance_feature.shape[1]
            num_free_instance = num_instance - num_dn_anchor

            # Create attention mask to control information flow:
            # - False means attention is allowed between queries
            # - True means attention is blocked (masked)
            attn_mask = anchor.new_ones((num_instance, num_instance), dtype=torch.bool)
            # Learnable queries can attend to each other
            attn_mask[:num_free_instance, :num_free_instance] = False
            # Denoising queries follow their specific attention pattern
            attn_mask[num_free_instance:, num_free_instance:] = dn_attn_mask

        # Encode anchor boxes to positional embeddings for transformer inputs
        anchor_embed = self.anchor_encoder(anchor)  # [B, N_query (+ N_dn), C]

        # Also encode temporal anchors if available
        if temp_anchor is not None:
            temp_anchor_embed = self.anchor_encoder(temp_anchor)
        else:
            temp_anchor_embed = None

        if self.is_cooperative and coop_anchor.shape[1] > 0:
            coop_instance_num = coop_anchor.shape[1]
            # assert (
            #     coop_instance_num > 0
            # ), "cooperative anchor should have at least one valid object"

            coop_anchor_embed = self.anchor_encoder(coop_anchor)

            num_free_instance = (
                instance_feature.shape[1] if dn_metas is None else num_free_instance
            )
            ego_group_len = [num_free_instance]
            if dn_metas is not None:
                ego_num_dn_groups = self.sampler.num_dn_groups
                assert num_dn_anchor % ego_num_dn_groups == 0
                ego_num_dn_per_group = num_dn_anchor // ego_num_dn_groups
                for g in range(ego_num_dn_groups):
                    ego_group_len.append(ego_num_dn_per_group)
            ego_group_start = [0] + [
                sum(ego_group_len[: i + 1]) for i in range(len(ego_group_len) - 1)
            ]

            coop_num_free_instance = coop_valid_mask.shape[1]
            coop_group_len = [coop_num_free_instance]
            if "coop_dn_instance_feature" in metas:
                coop_valid_mask = torch.cat(
                    [coop_valid_mask, coop_dn_valid_mask], dim=1
                )

                coop_num_dn_groups = self.instance_bank.coop_dn_group_num
                coop_num_dn = coop_dn_valid_mask.shape[1]
                assert coop_num_dn % coop_num_dn_groups == 0
                coop_num_dn_per_group = coop_num_dn // coop_num_dn_groups
                for g in range(coop_num_dn_groups):
                    coop_group_len.append(coop_num_dn_per_group)
            coop_group_start = [0] + [
                sum(coop_group_len[: i + 1]) for i in range(len(coop_group_len) - 1)
            ]

            coop_attn_mask = anchor.new_ones(
                batch_size * self.num_heads,
                sum(ego_group_len),
                sum(coop_group_len),
            )
            assert coop_attn_mask.shape[1] == anchor.shape[1]
            assert coop_attn_mask.shape[2] == coop_anchor.shape[1]
            assert len(ego_group_len) >= len(coop_group_len)
            coop_interaction_inds = []
            for eg in range(len(ego_group_len)):
                cg = eg % len(coop_group_len)
                coop_ind = {
                    "ego_start": ego_group_start[eg],
                    "ego_end": ego_group_start[eg] + ego_group_len[eg],
                    "ego_type": "normal" if eg == 0 else "denoising",
                    "coop_start": coop_group_start[cg],
                    "coop_end": coop_group_start[cg] + coop_group_len[cg],
                    "coop_type": "normal" if cg == 0 else "denoising",
                }
                if not self.coop_cross_attn_wo_dn:
                    coop_valid_mask_group = (
                        coop_valid_mask[
                            :, coop_ind["coop_start"] : coop_ind["coop_end"]
                        ]
                        .unsqueeze(1)
                        .expand(-1, ego_group_len[eg], -1)
                        .unsqueeze(1)
                        .expand(-1, self.num_heads, -1, -1)
                        .reshape(batch_size * self.num_heads, ego_group_len[eg], -1)
                    )
                    coop_attn_mask[
                        :,
                        coop_ind["ego_start"] : coop_ind["ego_end"],
                        coop_ind["coop_start"] : coop_ind["coop_end"],
                    ] = ~coop_valid_mask_group
                else:
                    coop_start = coop_group_start[0]
                    coop_end = coop_group_start[0] + coop_group_len[0]
                    coop_valid_mask_group = (
                        coop_valid_mask[:, coop_start:coop_end]
                        .unsqueeze(1)
                        .expand(-1, ego_group_len[eg], -1)
                        .unsqueeze(1)
                        .expand(-1, self.num_heads, -1, -1)
                        .reshape(batch_size * self.num_heads, ego_group_len[eg], -1)
                    )
                    coop_attn_mask[
                        :,
                        coop_ind["ego_start"] : coop_ind["ego_end"],
                        coop_start:coop_end,
                    ] = ~coop_valid_mask_group

                coop_interaction_inds.append(coop_ind)
        else:
            coop_instance_num = 0
            coop_anchor_embed = None
            coop_attn_mask = None
        # =================== forward the layers ====================
        # Process features through the transformer layers defined in operation_order
        prediction = []  # Stores box predictions from each refine layer
        classification = []  # Stores class scores from each refine layer
        quality = []  # Stores quality predictions from each refine layer
        instance_feature_list = []  # Stores instance features from each refine layer

        # Iterate through all operations in order
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op == "coop_gnn":
                # Cooperative Graph Neural Network: attends ego features to cooperative features
                # This allows information flow from cooperative agents to ego agent
                # If coop_instance_num is 0, it will turn into self-attention with ego instances
                instance_feature = self.graph_model(
                    i,
                    instance_feature,  # Current frame queries
                    (
                        coop_instance_feature
                        if coop_instance_num > 0
                        else instance_feature
                    ),  # cooperative features
                    (
                        coop_instance_feature
                        if coop_instance_num > 0
                        else instance_feature
                    ),  # cooperative features as values
                    query_pos=anchor_embed,  # Current frame position embeddings
                    key_pos=(
                        coop_anchor_embed if coop_instance_num > 0 else anchor_embed
                    ),  # cooperative position embeddings
                    attn_mask=(coop_attn_mask if coop_instance_num > 0 else attn_mask),
                )
            elif op == "temp_gnn":
                # Temporal Graph Neural Network: attends current features to temporal features
                # This allows information flow from previous frame to current frame
                instance_feature = self.graph_model(
                    i,
                    instance_feature,  # Current frame queries
                    temp_instance_feature,  # Previous frame features
                    temp_instance_feature,  # Previous frame features as values
                    query_pos=anchor_embed,  # Current frame position embeddings
                    key_pos=temp_anchor_embed,  # Previous frame position embeddings
                    attn_mask=attn_mask if temp_instance_feature is None else None,
                )

                # Try using temp_gnn for cooperative features
                if (
                    self.use_temp_gnn_for_coop
                    and len(prediction) >= self.num_single_agent_decoder
                ):
                    instance_feature = self.graph_model(
                        i,
                        instance_feature,  # Current frame queries
                        (
                            coop_instance_feature
                            if coop_instance_num > 0
                            else instance_feature
                        ),  # cooperative features
                        (
                            coop_instance_feature
                            if coop_instance_num > 0
                            else instance_feature
                        ),  # cooperative features as values
                        query_pos=anchor_embed,  # Current frame position embeddings
                        key_pos=(
                            coop_anchor_embed if coop_instance_num > 0 else anchor_embed
                        ),  # cooperative position embeddings
                        attn_mask=(
                            coop_attn_mask if coop_instance_num > 0 else attn_mask
                        ),
                    )

            elif op == "gnn":
                # Spatial Graph Neural Network: self-attention within current frame
                # This allows information exchange between different instances
                instance_feature = self.graph_model(
                    i,
                    instance_feature,  # Queries
                    value=instance_feature,  # Keys and values (self-attention)
                    query_pos=anchor_embed,  # Position embeddings
                    attn_mask=attn_mask,  # Use denoising attention mask
                )  # [bs, num_anchor, 2*embed_dims]
            elif op == "norm" or op == "ffn":
                # Apply normalization or feed-forward network
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                # Deformable Attention: cross-attention from queries to image features
                # This extracts relevant information from the feature maps
                instance_feature = self.layers[i](
                    instance_feature, anchor, anchor_embed, feature_maps, metas
                )
            elif op == "refine":
                # Refinement Layer: predicts box refinements, class scores, and quality
                # This is where actual detection outputs are generated
                anchor, cls, qt = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        self.training
                        or len(prediction) == self.num_single_frame_decoder - 1
                        or len(prediction) == self.num_single_agent_decoder - 1
                        or i == len(self.operation_order) - 1
                        or self.save_dn_for_coop
                    ),
                )
                prediction.append(anchor)
                classification.append(cls)
                quality.append(qt)
                instance_feature_list.append(instance_feature)

                # --- Single-Frame Instance Update ---
                # After num_single_frame_decoder layers, update the instance bank
                if len(prediction) == self.num_single_frame_decoder:
                    # Update instance features and anchors with more refined ones
                    instance_feature, anchor = self.instance_bank.update(
                        instance_feature, anchor, cls
                    )

                    # --- Temporal Denoising Update ---
                    # If temporal denoising is active, update DN targets based on matching
                    if (
                        dn_metas is not None
                        and self.sampler.num_temp_dn_groups > 0
                        and dn_id_target is not None
                    ):
                        # Update denoising targets for temporal modeling
                        # This connects denoising training across frames
                        (
                            instance_feature,
                            anchor,
                            temp_dn_reg_target,  # Updated regression targets
                            temp_dn_cls_target,  # Updated classification targets
                            temp_valid_mask,  # Updated validity mask
                            dn_id_target,  # Updated instance IDs
                        ) = self.sampler.update_dn(
                            instance_feature,
                            anchor,
                            dn_reg_target,
                            dn_cls_target,
                            valid_mask,
                            dn_id_target,
                            self.instance_bank.num_anchor,  # Number of learnable queries
                            self.instance_bank.mask,  # Temporal validity mask
                        )

                # --- Cooperative Update ---
                # After num_single_agent_decoder layers, update the instance bank
                if (
                    self.is_cooperative
                    and len(prediction) == self.num_single_agent_decoder
                ):
                    if coop_instance_num > 0 and not self.coop_wo_coarse_fusion:
                        instance_feature, anchor = self.instance_bank.coop_update(
                            instance_feature,
                            anchor,
                            cls,
                            coop_interaction_inds,
                            dn_id_target if dn_metas is not None else None,
                            metas["timestamp"],
                        )
                    else:
                        # dummy path to include the fusion layer
                        dummy = self.instance_bank.cross_agent_interaction.cross_agent_fusion(
                            torch.zeros(
                                (
                                    instance_feature.shape[0],
                                    instance_feature.shape[2],
                                )
                            ).to(instance_feature.device)
                        )
                        instance_feature = instance_feature + 0 * dummy.sum()

                # Re-encode anchors after refinement for next layers
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)

                # Update temporal anchor embeddings if needed after single-frame stage
                if (
                    len(prediction) > self.num_single_frame_decoder
                    and temp_anchor_embed is not None
                ):
                    # Only update embeddings for cached temporal instances
                    temp_anchor_embed = anchor_embed[
                        :, : self.instance_bank.num_temp_instances
                    ]
            else:
                raise NotImplementedError(f"{op} is not supported.")

        output = {}

        # Split predictions into learnable queries and denoising queries
        if dn_metas is not None:
            # Recalculate boundary between normal and DN instances
            num_free_instance = anchor.shape[1] - num_dn_anchor

            # Split classification scores
            dn_classification = [x[:, num_free_instance:] for x in classification]
            classification = [x[:, :num_free_instance] for x in classification]

            # Split box predictions
            dn_prediction = [x[:, num_free_instance:] for x in prediction]
            prediction = [x[:, :num_free_instance] for x in prediction]

            # Split quality scores
            quality = [
                x[:, :num_free_instance] if x is not None else None for x in quality
            ]

            # Store denoising outputs for loss calculation
            output.update(
                {
                    "dn_prediction": dn_prediction,
                    "dn_classification": dn_classification,
                    "dn_reg_target": dn_reg_target,
                    "dn_cls_target": dn_cls_target,
                    "dn_valid_mask": valid_mask,
                }
            )

            # Add temporal denoising outputs if available
            if temp_dn_reg_target is not None:
                output.update(
                    {
                        "temp_dn_reg_target": temp_dn_reg_target,
                        "temp_dn_cls_target": temp_dn_cls_target,
                        "temp_dn_valid_mask": temp_valid_mask,
                        "dn_id_target": dn_id_target,
                    }
                )
                # Update targets to use temporal ones for later loss calculation
                dn_cls_target = temp_dn_cls_target
                valid_mask = temp_valid_mask

            # Split features and anchors for learnable and DN queries
            dn_instance_feature = instance_feature[:, num_free_instance:]
            dn_anchor = anchor[:, num_free_instance:]
            instance_feature = instance_feature[:, :num_free_instance]
            anchor = anchor[:, :num_free_instance]
            cls = cls[:, :num_free_instance]

            # Cache denoising instances for future temporal denoising
            self.sampler.cache_dn(
                dn_instance_feature, dn_anchor, dn_cls_target, valid_mask, dn_id_target
            )

            # Save denoising features and anchors for cooperation
            if self.save_dn_for_coop:
                # Apply valid_mask to each batch separately
                coop_dn_instance_feature = []
                coop_dn_anchor = []
                coop_dn_id_target = []
                coop_dn_cls_target = []
                coop_dn_reg_target = []

                for b in range(batch_size):
                    # Read mask and cls target from output because they have been temp-updated
                    batch_valid_mask = output["dn_valid_mask"][b]
                    num_valid_samples = batch_valid_mask.sum().item()
                    assert num_valid_samples % (self.sampler.num_dn_groups * 2) == 0
                    num_dn_per_group = num_valid_samples // self.sampler.num_dn_groups

                    batch_dn_instance_feature = dn_instance_feature[b][
                        batch_valid_mask
                    ].reshape(self.sampler.num_dn_groups, num_dn_per_group, -1)
                    batch_dn_anchor = dn_anchor[b][batch_valid_mask].reshape(
                        self.sampler.num_dn_groups, num_dn_per_group, -1
                    )
                    batch_dn_id_target = dn_id_target[b][batch_valid_mask].reshape(
                        self.sampler.num_dn_groups, num_dn_per_group, -1
                    )
                    batch_dn_cls_target = output["dn_cls_target"][b][
                        batch_valid_mask
                    ].reshape(self.sampler.num_dn_groups, num_dn_per_group, -1)
                    batch_dn_reg_target = dn_reg_target[b][batch_valid_mask].reshape(
                        self.sampler.num_dn_groups, num_dn_per_group, -1
                    )

                    coop_dn_instance_feature.append(
                        batch_dn_instance_feature[self.sampler.num_temp_dn_groups :]
                    )
                    coop_dn_anchor.append(
                        batch_dn_anchor[self.sampler.num_temp_dn_groups :]
                    )
                    coop_dn_id_target.append(
                        batch_dn_id_target[self.sampler.num_temp_dn_groups :]
                    )
                    coop_dn_cls_target.append(
                        batch_dn_cls_target[self.sampler.num_temp_dn_groups :]
                    )
                    coop_dn_reg_target.append(
                        batch_dn_reg_target[self.sampler.num_temp_dn_groups :]
                    )

                output["coop_dn_instance_feature"] = coop_dn_instance_feature
                output["coop_dn_anchor"] = coop_dn_anchor
                output["coop_dn_id_target"] = coop_dn_id_target
                output["coop_dn_cls_target"] = coop_dn_cls_target
                output["coop_dn_reg_target"] = coop_dn_reg_target

        # Add regular detection outputs
        output.update(
            {
                "classification": classification,
                "prediction": prediction,
                "quality": quality,
                "instance_feature": instance_feature_list,
            }
        )

        # Cache current frame's instances for the next frame's temporal modeling
        # This is the key to enabling 4D detection across time
        self.instance_bank.cache(instance_feature, anchor, cls, metas, feature_maps)

        # Generate instance IDs for tracking during inference or cooperative (ID is used for selecting replaced instances)
        if not self.training or self.is_cooperative:
            instance_id = self.instance_bank.get_instance_id(
                cls,  # (B=1, num_queries, C=3)
                anchor,  # (B=1, num_queries, C=11)
                self.decoder.score_threshold,  # Default: None
            )
            output["instance_id"] = instance_id
        return output

    @force_fp32(apply_to=("model_outs"))
    def loss(self, model_outs, data, feature_maps=None):
        """Calculates the loss for SparseCoopHead predictions.

        The loss function consists of two main components:
        1. Standard detection losses (classification + regression)
        2. Denoising losses (for training robustness via noisy query denoising)

        Each component is calculated across all decoder layers to enable deep supervision.

        Args:
            model_outs (dict): The output dictionary from the forward pass.
            data (dict): The input data dictionary containing ground truths.
            feature_maps (Optional): Feature maps (unused in loss calculation but kept for API consistency).

        Returns:
            dict: Dictionary of loss components:
                - 'loss_cls_{i}': Classification loss for decoder layer i
                - 'loss_reg_{i}': Regression loss for decoder layer i
                - 'loss_cls_dn_{i}': Denoising classification loss for decoder layer i
                - 'loss_reg_dn_{i}': Denoising regression loss for decoder layer i
        """

        # ===================== STANDARD DETECTION LOSSES ======================
        # These losses train the model to detect objects from learnable instance queries

        # Extract predictions from each decoder layer for deep supervision
        cls_scores = model_outs["classification"]  # List[Tensor[B, N_query, N_cls]]
        reg_preds = model_outs["prediction"]  # List[Tensor[B, N_query, box_dim]]
        quality = model_outs["quality"]  # List[Tensor[B, N_query, 1]]
        output = {}

        # Calculate loss for each decoder layer (deep supervision)
        for decoder_idx, (cls, reg, qt) in enumerate(
            zip(cls_scores, reg_preds, quality)
        ):
            # --------- Step 1: Prepare predictions and targets ---------
            # Truncate regression predictions to match expected dimensions
            # reg_weights defines which box parameters to use and how to weight them
            reg = reg[..., : len(self.reg_weights)]  # [B, N_query, len(reg_weights)]

            # Use sampler (SparseBox3DTarget) to assign ground truth targets to predictions
            # This implements Hungarian matching algorithm considering class cost and box cost
            cls_target, reg_target, reg_weights = self.sampler.sample(
                cls, reg, data[self.gt_cls_key], data[self.gt_reg_key]
            )
            # cls_target: [B, N_query] - assigned class labels (0~num_cls-1=object class, num_cls=background)
            # reg_target: [B, N_query, box_dim] - assigned regression targets
            # reg_weights: [B, N_query, box_dim] - per-sample regression weights

            reg_target = reg_target[..., : len(self.reg_weights)]

            # --------- Step 2: Identify positive samples ---------
            # Create mask for positive samples (queries successfully matched to ground truth)
            # - Positive samples: queries matched to real objects (cls_target < num_cls)
            # - Background samples: unmatched queries assigned to background class (cls_target == num_cls)
            #
            # Here we identify positive samples by checking regression targets:
            # - Positive samples have non-zero regression targets (actual object parameters)
            # - Background samples have all-zero regression targets (default padding)
            mask = torch.logical_not(torch.all(reg_target == 0, dim=-1))  # [B, N_query]
            mask_valid = mask.clone()  # Keep original mask for later use

            # --------- Step 3: Calculate normalization factor ---------
            # Count total positive samples across all batches for loss normalization
            # This ensures loss magnitude is stable regardless of number of objects
            num_pos = max(reduce_mean(torch.sum(mask).to(dtype=reg.dtype)), 1.0)

            # --------- Step 4: Apply classification confidence thresholding (optional) ---------
            # Only apply regression loss to samples with high classification confidence
            # This prevents training regression on low-confidence detections
            if self.cls_threshold_to_reg > 0:
                threshold = self.cls_threshold_to_reg
                # Get max class probability for each query and apply sigmoid
                cls_conf = cls.max(dim=-1).values.sigmoid()  # [B, N_query]
                # Update mask to only include confident positive samples
                mask = torch.logical_and(mask, cls_conf > threshold)

            # --------- Step 5: Calculate classification loss ---------
            # Flatten batch and query dimensions for loss calculation
            cls = cls.flatten(end_dim=1)  # [B*N_query, N_cls]
            cls_target = cls_target.flatten(end_dim=1)  # [B*N_query]

            # Apply classification loss (FocalLoss)
            # Normalized by number of positive samples to prevent class imbalance issues
            cls_loss = self.loss_cls(cls, cls_target, avg_factor=num_pos)

            # --------- Step 6: Calculate regression loss ---------
            # Only calculate regression loss on positive samples (identified by mask)
            mask = mask.reshape(-1)  # [B*N_query] - flatten mask

            # Apply regression weights: combine sampler weights with predefined weights
            #
            # **Two-level weighting system**:
            # 1. **Per-sample weights from sampler** (reg_weights): [B, N_query, box_dim]
            #    - Generated by SparseBox3DTarget.sample() during Hungarian matching
            #    - Handles invalid/NaN ground truth parameters
            #    - Can apply class-wise weights
            #
            # 2. **Predefined parameter weights** (self.reg_weights): [box_dim]
            #    - Global weights for different box parameters: [x,y,z,log_l,log_w,log_h,sin_yaw,cos_yaw,vx,vy,vz]
            #    - Allows selectively enabling/disabling certain regression targets
            #
            # **Final weight = per_sample_weight × predefined_weight**
            reg_weights = reg_weights * reg.new_tensor(self.reg_weights)

            # Extract only positive samples for regression loss calculation
            reg_target = reg_target.flatten(end_dim=1)[mask]  # [N_pos, box_dim]
            reg = reg.flatten(end_dim=1)[mask]  # [N_pos, box_dim]
            reg_weights = reg_weights.flatten(end_dim=1)[mask]  # [N_pos, box_dim]

            # Handle NaN values in targets (replace with 0)
            # This can happen with incomplete ground truth annotations
            reg_target = torch.where(
                reg_target.isnan(), reg.new_tensor(0.0), reg_target
            )

            # Extract classification targets for positive samples (needed by some loss functions)
            cls_target = cls_target[mask]  # [N_pos]

            # Extract quality scores for positive samples if available
            if qt is not None:
                qt = qt.flatten(end_dim=1)[mask]  # [N_pos, 1]

            # Apply regression loss (typically L1 loss, IoU loss, or combination)
            # Quality and cls_target can be used for quality-aware or class-aware regression
            reg_loss = self.loss_reg(
                reg,  # Predicted regression values
                reg_target,  # Ground truth regression targets
                weight=reg_weights,  # Per-parameter weights
                avg_factor=num_pos,  # Normalization factor
                suffix=f"_{decoder_idx}",  # Suffix for loss name
                quality=qt,  # Quality scores (optional)
                cls_target=cls_target,  # Class targets (optional)
            )

            # --------- Step 7: Store losses for this decoder layer ---------
            output[f"loss_cls_{decoder_idx}"] = cls_loss
            output.update(reg_loss)  # reg_loss is a dict of multiple loss components

        # Early return if no denoising training is used
        if "dn_prediction" not in model_outs:
            return output

        # ===================== DENOISING LOSSES ======================
        # Denoising training improves model robustness by training on noisy queries
        # The model learns to "denoise" corrupted object queries back to ground truth

        # Extract denoising predictions from each decoder layer
        dn_cls_scores = model_outs["dn_classification"]  # List[Tensor[B, N_dn, N_cls]]
        dn_reg_preds = model_outs["dn_prediction"]  # List[Tensor[B, N_dn, box_dim]]

        # --------- Prepare denoising targets and masks ---------
        # Extract all necessary tensors for denoising loss calculation
        (
            dn_valid_mask,  # [B*N_dn] - mask for valid denoising samples
            dn_cls_target,  # [N_valid_dn] - class targets for valid samples
            dn_reg_target,  # [N_pos_dn, box_dim] - regression targets for positive samples
            dn_pos_mask,  # [N_valid_dn] - mask for positive samples within valid ones
            reg_weights,  # [N_pos_dn, box_dim] - regression weights
            num_dn_pos,  # Scalar - number of positive denoising samples for normalization
        ) = self.prepare_for_dn_loss(model_outs)

        # Calculate denoising loss for each decoder layer
        for decoder_idx, (cls, reg) in enumerate(zip(dn_cls_scores, dn_reg_preds)):
            # --------- Handle temporal denoising updates (if applicable) ---------
            # After single-frame processing, we may have updated denoising targets
            # based on temporal matching between current and previous frames
            if (
                "temp_dn_valid_mask" in model_outs
                and decoder_idx == self.num_single_frame_decoder
            ):
                # Use updated temporal denoising targets instead of initial ones
                # This enables denoising training across temporal boundaries
                (
                    dn_valid_mask,
                    dn_cls_target,
                    dn_reg_target,
                    dn_pos_mask,
                    reg_weights,
                    num_dn_pos,
                ) = self.prepare_for_dn_loss(model_outs, prefix="temp_")

            # --------- Calculate denoising classification loss ---------
            # Only use valid denoising samples (some may be padding)
            dn_cls_flat = cls.flatten(end_dim=1)[dn_valid_mask]  # [N_valid_dn, N_cls]

            cls_loss = self.loss_cls(
                dn_cls_flat,  # Denoising class predictions
                dn_cls_target,  # Denoising class targets
                avg_factor=num_dn_pos,  # Normalize by number of positive denoising samples
            )

            # --------- Calculate denoising regression loss ---------
            # Apply multiple levels of masking:
            # 1. dn_valid_mask: select valid denoising samples
            # 2. dn_pos_mask: select positive samples among valid ones
            # 3. [..., :len(self.reg_weights)]: select relevant box parameters
            dn_reg_flat = reg.flatten(end_dim=1)[dn_valid_mask][dn_pos_mask][
                ..., : len(self.reg_weights)
            ]  # [N_pos_dn, len(reg_weights)]

            reg_loss = self.loss_reg(
                dn_reg_flat,  # Denoising regression predictions
                dn_reg_target,  # Denoising regression targets
                avg_factor=num_dn_pos,  # Normalize by number of positive denoising samples
                weight=reg_weights,  # Per-parameter regression weights
                suffix=f"_dn_{decoder_idx}",  # Suffix for loss name
            )

            # --------- Store denoising losses for this decoder layer ---------
            output[f"loss_cls_dn_{decoder_idx}"] = cls_loss
            output.update(reg_loss)  # reg_loss is a dict of multiple loss components

        return output

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        """Prepares tensors needed for calculating the denoising loss.

        Extracts targets, masks, and calculates the normalization factor
        either from the initial DN outputs or the temporal DN outputs.

        Args:
            model_outs (dict): The output dictionary from the forward pass containing:
                - '{prefix}dn_valid_mask': [B, N_dn] - mask for valid denoising samples
                - '{prefix}dn_cls_target': [B, N_dn] - class targets for denoising samples
                - '{prefix}dn_reg_target': [B, N_dn, box_dim] - regression targets for denoising
            prefix (str): Prefix to access the correct keys in model_outs:
                - "" for initial denoising targets (generated at start of forward pass)
                - "temp_" for temporal denoising targets (updated after single-frame processing)
                Defaults to "".

        Returns:
            Tuple containing processed tensors for denoising loss calculation:
                - dn_valid_mask (torch.Tensor): [B*N_dn] - flattened mask for valid DN samples
                - dn_cls_target (torch.Tensor): [N_valid_dn] - class targets for valid DN samples only
                - dn_reg_target (torch.Tensor): [N_pos_dn, len(reg_weights)] - regression targets for positive DN samples only
                - dn_pos_mask (torch.Tensor): [N_valid_dn] - mask indicating positive samples within valid ones
                - reg_weights (torch.Tensor): [N_pos_dn, len(reg_weights)] - regression weights for positive DN samples
                - num_dn_pos (float): Number of positive DN samples (for loss normalization)
        """

        # --------- Step 1: Extract and flatten validity mask ---------
        # dn_valid_mask indicates which denoising samples are valid (not padding)
        # Some denoising samples may be padding to maintain consistent batch size
        dn_valid_mask = model_outs[f"{prefix}dn_valid_mask"].flatten(
            end_dim=1
        )  # [B*N_dn]

        # --------- Step 2: Extract classification targets for valid samples ---------
        # Only keep classification targets for valid denoising samples
        # Classification targets: -3 = hard negative denoising sample, >= 0 = positive sample with class ID
        dn_cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(end_dim=1)[
            dn_valid_mask
        ]  # [N_valid_dn]

        # --------- Step 3: Extract and process regression targets ---------
        # Get regression targets for valid samples and truncate to expected dimensions
        dn_reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(end_dim=1)[
            dn_valid_mask
        ][
            ..., : len(self.reg_weights)
        ]  # [N_valid_dn, len(reg_weights)]

        # --------- Step 4: Identify positive samples within valid samples ---------
        # Positive samples have class_target >= 0 (hard negative denoising samples have class_target = -3)
        # Only positive samples contribute to regression loss
        dn_pos_mask = dn_cls_target >= 0  # [N_valid_dn]

        # --------- Step 5: Extract regression targets for positive samples only ---------
        # Regression loss is only calculated on positive samples
        dn_reg_target = dn_reg_target[dn_pos_mask]  # [N_pos_dn, len(reg_weights)]

        # --------- Step 6: Create regression weights ---------
        # Apply predefined regression weights to each box parameter
        # self.reg_weights: [len(reg_weights)] - predefined weights for each parameter
        # Tile to match the number of positive denoising samples
        reg_weights = dn_reg_target.new_tensor(self.reg_weights)[None].tile(
            dn_reg_target.shape[0], 1
        )  # [N_pos_dn, len(reg_weights)]

        # --------- Step 7: Calculate normalization factor ---------
        # Count total number of valid denoising samples across all batches
        # This is used to normalize the loss magnitude
        # Use at least 1.0 to prevent division by zero
        num_dn_pos = max(
            reduce_mean(torch.sum(dn_valid_mask).to(dtype=reg_weights.dtype)), 1.0
        )

        return (
            dn_valid_mask,  # [B*N_dn] - flattened validity mask
            dn_cls_target,  # [N_valid_dn] - class targets for valid samples
            dn_reg_target,  # [N_pos_dn, len(reg_weights)] - regression targets for positive samples
            dn_pos_mask,  # [N_valid_dn] - positive sample mask within valid samples
            reg_weights,  # [N_pos_dn, len(reg_weights)] - regression weights for positive samples
            num_dn_pos,  # Scalar - normalization factor (number of valid samples)
        )

    @force_fp32(apply_to=("model_outs"))
    def post_process(self, model_outs, output_idx=-1, metas=None):
        """Post-processes the model outputs to generate final detections.

        Typically involves applying NMS and thresholding based on scores.

        Args:
            model_outs (dict): The output dictionary from the forward pass.
            output_idx (int): Index of the decoder layer predictions to use
                              (-1 means the last layer). Defaults to -1.

        Returns:
            List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
                A list (one element per batch item) containing tuples of
                (bboxes, scores, labels).
        """
        ego_results = self.decoder.decode(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs.get("instance_id"),
            model_outs.get("quality"),
            model_outs.get("instance_feature"),
            output_idx=output_idx,
        )

        if self.save_dn_for_coop:
            for b in range(len(ego_results)):
                ego_results[b]["coop_dn_instance_feature"] = model_outs[
                    "coop_dn_instance_feature"
                ][b].cpu()
                ego_results[b]["coop_dn_anchor"] = model_outs["coop_dn_anchor"][b].cpu()
                ego_results[b]["coop_dn_id_target"] = model_outs["coop_dn_id_target"][
                    b
                ].cpu()
                ego_results[b]["coop_dn_cls_target"] = model_outs["coop_dn_cls_target"][
                    b
                ].cpu()
                ego_results[b]["coop_dn_reg_target"] = model_outs["coop_dn_reg_target"][
                    b
                ].cpu()
        if self.is_cooperative:
            for b in range(len(ego_results)):
                inf_direct_mask = self.instance_bank.inf_direct_masks[b]

                if (inf_direct_mask).sum() > 0:
                    # Get invisible infrastructure results directly from loaded data
                    inf_anchor = self.instance_bank.coop_anchor_unfiltered[b][
                        inf_direct_mask
                    ]
                    inf_scores = metas['coop_score'][b][inf_direct_mask]
                    inf_cls_scores = metas['coop_cls_score'][b][inf_direct_mask]
                    inf_label = metas['coop_label'][b][inf_direct_mask]
                    inf_instance_id = metas['coop_instance_ids'][b][inf_direct_mask]
                    for i in range(len(inf_instance_id)):
                        # Check if infrastructure ID already has a mapping
                        mapped_veh_id = self.instance_bank.cross_agent_interaction.id_mapping_manager.get_vehicle_id(
                            inf_instance_id[i].item(), metas["timestamp"][b]
                        )
                        if mapped_veh_id is not None:
                            inf_instance_id[i] = mapped_veh_id
                        else:
                            # Create new mapping with current prev_id
                            new_veh_id = self.instance_bank.prev_id.clone()
                            self.instance_bank.cross_agent_interaction.id_mapping_manager.add_mapping(
                                inf_instance_id[i].item(),
                                new_veh_id.item(),
                                timestamp=metas["timestamp"][b],
                                confidence=inf_scores[i].item(),
                            )
                            inf_instance_id[i] = new_veh_id
                            self.instance_bank.prev_id += 1

                    # Concat ego results and invisible inf results
                    ego_results[b]["boxes_3d"] = torch.cat(
                        [
                            ego_results[b]["boxes_3d"],
                            self.decoder.decode_box(inf_anchor).cpu(),
                        ],
                        dim=0,
                    )
                    ego_results[b]["scores_3d"] = torch.cat(
                        [
                            ego_results[b]["scores_3d"],
                            inf_scores.cpu(),
                        ],
                        dim=0,
                    )
                    ego_results[b]["cls_scores"] = torch.cat(
                        [
                            ego_results[b]["cls_scores"],
                            inf_cls_scores.cpu(),
                        ],
                        dim=0,
                    )
                    ego_results[b]["labels_3d"] = torch.cat(
                        [
                            ego_results[b]["labels_3d"],
                            inf_label.cpu(),
                        ],
                        dim=0,
                    )
                    ego_results[b]["instance_ids"] = torch.cat(
                        [
                            ego_results[b]["instance_ids"],
                            inf_instance_id.cpu(),
                        ],
                        dim=0,
                    )

                    # Sort by scores
                    scores = ego_results[b]["scores_3d"]
                    scores, indices = torch.sort(scores, descending=True)
                    ego_results[b]["boxes_3d"] = ego_results[b]["boxes_3d"][indices]
                    ego_results[b]["scores_3d"] = ego_results[b]["scores_3d"][indices]
                    ego_results[b]["labels_3d"] = ego_results[b]["labels_3d"][indices]
                    ego_results[b]["instance_ids"] = ego_results[b]["instance_ids"][
                        indices
                    ]
        return ego_results
