from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp.autocast_mode import autocast

from mmcv.cnn import Linear, build_activation_layer, build_norm_layer
from mmcv.runner.base_module import Sequential, BaseModule
from mmcv.cnn.bricks.transformer import FFN
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.drop import build_dropout
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import ATTENTION, PLUGIN_LAYERS, FEEDFORWARD_NETWORK

# Import deformable aggregation function if available
try:
    from ..ops import deformable_aggregation_function as DAF
except:
    DAF = None

__all__ = ["DeformableFeatureAggregation", "DenseDepthNet", "AsymmetricFFN"]


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    """
    Creates a sequence of linear layers with ReLU activation and LayerNorm.

    Args:
        embed_dims (int): Dimension of the embedding space
        in_loops (int): Number of linear+ReLU layers in each inner loop
        out_loops (int): Number of outer loops (each ending with LayerNorm)
        input_dims (int, optional): Input dimension. Defaults to embed_dims.

    Returns:
        list: List of layers forming the network
    """
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


@ATTENTION.register_module()
class DeformableFeatureAggregation(BaseModule):
    """
    Deformable Feature Aggregation module for multi-view feature fusion.

    This module aggregates features from multiple camera views using deformable sampling
    and attention mechanisms. It can optionally use temporal information and camera embeddings.

    Args:
        embed_dims (int): Feature dimension
        num_groups (int): Number of attention groups
        num_levels (int): Number of feature pyramid levels
        num_cams (int): Number of camera views
        proj_drop (float): Dropout rate for projection
        attn_drop (float): Dropout rate for attention weights
        kps_generator (dict): Configuration for keypoint generator
        temporal_fusion_module (dict, optional): Configuration for temporal fusion
        use_temporal_anchor_embed (bool): Whether to use temporal anchor embedding
        use_deformable_func (bool): Whether to use deformable aggregation function
        use_camera_embed (bool): Whether to use camera embeddings
        residual_mode (str): Residual connection mode ('add' or 'cat')
    """
    def __init__(
        self,
        embed_dims: int = 256,
        num_groups: int = 8,
        num_levels: int = 4,
        num_cams: int = 6,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        kps_generator: dict = None,
        temporal_fusion_module=None,
        use_temporal_anchor_embed=True,
        use_deformable_func=False,
        use_camera_embed=False,
        residual_mode="add",
    ):
        super(DeformableFeatureAggregation, self).__init__()
        # Validate input dimensions
        if embed_dims % num_groups != 0:
            raise ValueError(
                f"embed_dims must be divisible by num_groups, "
                f"but got {embed_dims} and {num_groups}"
            )

        # Initialize basic parameters
        self.group_dims = int(embed_dims / num_groups)
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.use_temporal_anchor_embed = use_temporal_anchor_embed

        # Validate deformable function availability
        if use_deformable_func:
            assert DAF is not None, "deformable_aggregation needs to be set up."
        self.use_deformable_func = use_deformable_func
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)

        # Build keypoint generator
        kps_generator["embed_dims"] = embed_dims
        self.kps_generator = build_from_cfg(kps_generator, PLUGIN_LAYERS)
        self.num_pts = self.kps_generator.num_pts

        # Build temporal fusion module if specified
        if temporal_fusion_module is not None:
            if "embed_dims" not in temporal_fusion_module:
                temporal_fusion_module["embed_dims"] = embed_dims
            self.temp_module = build_from_cfg(temporal_fusion_module, PLUGIN_LAYERS)
        else:
            self.temp_module = None
        self.output_proj = Linear(embed_dims, embed_dims)

        if use_camera_embed:
            self.camera_encoder = Sequential(*linear_relu_ln(embed_dims, 1, 2, 12))
            self.weights_fc = Linear(embed_dims, num_groups * num_levels * self.num_pts)
        else:
            self.camera_encoder = None
            self.weights_fc = Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts
            )

    def init_weight(self):
        """Initialize weights for the attention and projection layers."""
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution="uniform", bias=0.0)

    def forward(
        self,
        instance_feature: torch.Tensor,
        anchor: torch.Tensor,
        anchor_embed: torch.Tensor,
        feature_maps: List[torch.Tensor],
        metas: dict,
        **kwargs: dict,
    ):
        """
        Forward pass for feature aggregation.

        Args:
            instance_feature: Instance features [bs, num_anchor, embed_dims]
            anchor: Anchor points [bs, num_anchor, 3]
            anchor_embed: Anchor embeddings [bs, num_anchor, embed_dims]
            feature_maps: List of feature maps from different levels
            metas: Dictionary containing metadata like projection matrices

        Returns:
            Aggregated features with residual connection
        """
        bs, num_anchor = instance_feature.shape[:2]

        # Generate keypoints for feature sampling
        key_points = self.kps_generator(
            anchor, instance_feature
        )  # [bs, num_anchor, 13, 3]

        # Compute attention weights
        weights = self._get_weights(
            instance_feature, anchor_embed, metas
        )  # [bs, num_anchor, 6, 4, 13, 8]

        if self.use_deformable_func:
            # Project 3D points to 2D image coordinates
            points_2d = (
                self.project_points(
                    key_points, metas["projection_mat"], metas.get("image_wh")
                )
                .permute(0, 2, 3, 1, 4)
                .reshape(bs, num_anchor, self.num_pts, self.num_cams, 2)
            )  # [bs, num_anchor, 13, 6, 2]

            # Reshape weights for deformable aggregation
            weights = (
                weights.permute(0, 1, 4, 2, 3, 5)
                .contiguous()
                .reshape(
                    bs,
                    num_anchor,
                    self.num_pts,
                    self.num_cams,
                    self.num_levels,
                    self.num_groups,
                )
            )  # [bs, num_anchor, 13, 6, 4, 8]

            # Use deformable aggregation function
            features = DAF(*feature_maps, points_2d, weights).reshape(
                bs, num_anchor, self.embed_dims
            )  # [bs, num_anchor, embed_dims]
        else:
            # Standard feature sampling and fusion
            features = self.feature_sampling(
                feature_maps, key_points, metas["projection_mat"], metas.get("image_wh")
            )
            features = self.multi_view_level_fusion(features, weights)
            features = features.sum(dim=2)  # fuse multi-point features

        # Project and apply residual connection
        output = self.proj_drop(
            self.output_proj(features)
        )  # [bs, num_anchor, embed_dims]

        # # If the anchor is not included in any views, it will get features as zeros, and its output should be ignored
        # valid_mask = features.abs().any(dim=-1).unsqueeze(-1)
        # output = output * valid_mask  # Apply mask to set invalid outputs to zeros

        if self.residual_mode == "add":
            output = output + instance_feature
        elif self.residual_mode == "cat":
            output = torch.cat([output, instance_feature], dim=-1)
        return output  # [bs, num_anchor, 2* embed_dims]

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        """
        Compute attention weights for feature aggregation.

        Args:
            instance_feature: Instance features
            anchor_embed: Anchor embeddings
            metas: Metadata dictionary

        Returns:
            Attention weights for multi-view and multi-level feature fusion
        """
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed

        # Add camera embeddings if enabled
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                metas["projection_mat"][:, :, :3].reshape(bs, self.num_cams, -1)
            )
            feature = feature[:, :, None] + camera_embed[:, None]

        # Compute and normalize attention weights
        weights = (
            self.weights_fc(feature)
            .reshape(bs, num_anchor, -1, self.num_groups)
            .softmax(dim=-2)
            .reshape(
                bs,
                num_anchor,
                self.num_cams,
                self.num_levels,
                self.num_pts,
                self.num_groups,
            )
        )

        # Apply attention dropout during training
        if self.training and self.attn_drop > 0:
            mask = torch.rand(bs, num_anchor, self.num_cams, 1, self.num_pts, 1)
            mask = mask.to(device=weights.device, dtype=weights.dtype)
            weights = ((mask > self.attn_drop) * weights) / (1 - self.attn_drop)
        return weights

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        """
        Project 3D points to 2D image coordinates.

        Args:
            key_points: 3D keypoints [bs, num_anchor, num_pts, 3]
            projection_mat: Projection matrices [bs, num_cams, 3, 4]
            image_wh: Image width and height for normalization

        Returns:
            2D projected points
        """
        bs, num_anchor, num_pts = key_points.shape[:3]

        # Add homogeneous coordinate
        pts_extend = torch.cat(
            [key_points, torch.ones_like(key_points[..., :1])], dim=-1
        )

        # Project points
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]
        ).squeeze(
            -1
        )  # [bs, num_cams, num_anchor, num_pts, 3]

        # # Check if points are behind the camera
        # behind_camera = points_2d[..., 2:3] < 0
        # # For points behind camera, set their coordinates to be far outside the image
        # points_2d = torch.where(
        #     behind_camera,
        #     torch.tensor([-100.0, -100.0, 1.0], device=points_2d.device),
        #     points_2d,
        # )

        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)

        # Normalize coordinates if image dimensions provided
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        return points_2d

    @staticmethod
    def feature_sampling(
        feature_maps: List[torch.Tensor],
        key_points: torch.Tensor,
        projection_mat: torch.Tensor,
        image_wh: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample features from multiple feature maps using projected keypoints.

        Args:
            feature_maps: List of feature maps from different levels
            key_points: 3D keypoints
            projection_mat: Projection matrices
            image_wh: Image dimensions for normalization

        Returns:
            Sampled features from all views and levels
        """
        num_levels = len(feature_maps)
        num_cams = feature_maps[0].shape[1]
        bs, num_anchor, num_pts = key_points.shape[:3]

        # Project points to 2D
        points_2d = DeformableFeatureAggregation.project_points(
            key_points, projection_mat, image_wh
        )
        points_2d = points_2d * 2 - 1  # Scale to [-1, 1] for grid_sample
        points_2d = points_2d.flatten(end_dim=1)

        # Sample features from each level
        features = []
        for fm in feature_maps:
            features.append(
                torch.nn.functional.grid_sample(fm.flatten(end_dim=1), points_2d)
            )
        features = torch.stack(features, dim=1)

        # Reshape to [bs, num_anchor, num_cams, num_levels, num_pts, embed_dims]
        features = features.reshape(
            bs, num_cams, num_levels, -1, num_anchor, num_pts
        ).permute(0, 4, 1, 2, 5, 3)

        return features

    def multi_view_level_fusion(self, features: torch.Tensor, weights: torch.Tensor):
        """
        Fuse features from multiple views and levels using attention weights.

        Args:
            features: Sampled features
            weights: Attention weights

        Returns:
            Fused features
        """
        bs, num_anchor = weights.shape[:2]
        features = weights[..., None] * features.reshape(
            features.shape[:-1] + (self.num_groups, self.group_dims)
        )
        features = features.sum(dim=2).sum(dim=2)
        features = features.reshape(bs, num_anchor, self.num_pts, self.embed_dims)
        return features


@PLUGIN_LAYERS.register_module()
class DenseDepthNet(BaseModule):
    """
    Dense depth prediction network.

    Predicts depth maps from feature maps using a series of convolutional layers.

    Args:
        embed_dims (int): Feature dimension
        num_depth_layers (int): Number of depth prediction layers
        equal_focal (int): Default focal length
        max_depth (int): Maximum depth value
        loss_weight (float): Weight for depth loss
    """
    def __init__(
        self,
        embed_dims=256,
        num_depth_layers=1,
        equal_focal=100,
        max_depth=60,
        loss_weight=1.0,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.equal_focal = equal_focal
        self.num_depth_layers = num_depth_layers
        self.max_depth = max_depth
        self.loss_weight = loss_weight

        # Create depth prediction layers
        self.depth_layers = nn.ModuleList()
        for i in range(num_depth_layers):
            self.depth_layers.append(
                nn.Conv2d(embed_dims, 1, kernel_size=1, stride=1, padding=0)
            )

    def forward(self, feature_maps, focal=None, gt_depths=None):
        """
        Forward pass for depth prediction.

        Args:
            feature_maps: Input feature maps
            focal: Focal length (optional)
            gt_depths: Ground truth depths for loss computation

        Returns:
            Predicted depth maps or loss if training
        """
        if focal is None:
            focal = self.equal_focal
        else:
            focal = focal.reshape(-1)

        # Predict depth for each feature level
        depths = []
        for i, feat in enumerate(feature_maps[: self.num_depth_layers]):
            depth = self.depth_layers[i](feat.flatten(end_dim=1).float()).exp()
            depth = depth.transpose(0, -1) * focal / self.equal_focal
            depth = depth.transpose(0, -1)
            depths.append(depth)

        # Compute loss if ground truth provided
        if gt_depths is not None and self.training:
            loss = self.loss(depths, gt_depths)
            return loss
        return depths

    def loss(self, depth_preds, gt_depths):
        """
        Compute depth prediction loss.

        Args:
            depth_preds: Predicted depth maps
            gt_depths: Ground truth depth maps

        Returns:
            Depth prediction loss
        """
        loss = 0.0
        for pred, gt in zip(depth_preds, gt_depths):
            pred = pred.permute(0, 2, 3, 1).contiguous().reshape(-1)
            gt = gt.reshape(-1)

            # Only compute loss on valid depth values
            fg_mask = torch.logical_and(gt > 0.0, torch.logical_not(torch.isnan(pred)))
            gt = gt[fg_mask]
            pred = pred[fg_mask]
            pred = torch.clip(pred, 0.0, self.max_depth)

            # Compute L1 loss
            with autocast(enabled=False):
                error = torch.abs(pred - gt).sum()
                _loss = error / max(1.0, len(gt) * len(depth_preds)) * self.loss_weight
            loss = loss + _loss
        return loss


@FEEDFORWARD_NETWORK.register_module()
class AsymmetricFFN(BaseModule):
    """
    Asymmetric Feed-Forward Network with optional pre-normalization.

    Args:
        in_channels (int, optional): Input channels
        pre_norm (dict, optional): Pre-normalization configuration
        embed_dims (int): Embedding dimension
        feedforward_channels (int): Hidden dimension of feed-forward network
        num_fcs (int): Number of fully connected layers
        act_cfg (dict): Activation function configuration
        ffn_drop (float): Dropout rate
        dropout_layer (dict, optional): Dropout layer configuration
        add_identity (bool): Whether to add identity connection
        init_cfg (dict, optional): Initialization configuration
    """
    def __init__(
        self,
        in_channels=None,
        pre_norm=None,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        act_cfg=dict(type="ReLU", inplace=True),
        ffn_drop=0.0,
        dropout_layer=None,
        add_identity=True,
        init_cfg=None,
        **kwargs,
    ):
        super(AsymmetricFFN, self).__init__(init_cfg)
        assert num_fcs >= 2, "num_fcs should be no less " f"than 2. got {num_fcs}."

        # Initialize parameters
        self.in_channels = in_channels
        self.pre_norm = pre_norm
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.activate = build_activation_layer(act_cfg)

        # Build network layers
        layers = []
        if in_channels is None:
            in_channels = embed_dims
        if pre_norm is not None:
            self.pre_norm = build_norm_layer(pre_norm, in_channels)[1]

        # Add fully connected layers with activation and dropout
        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels),
                    self.activate,
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)

        # Setup dropout and identity connection
        self.dropout_layer = (
            build_dropout(dropout_layer) if dropout_layer else torch.nn.Identity()
        )
        self.add_identity = add_identity
        if self.add_identity:
            self.identity_fc = (
                torch.nn.Identity()
                if in_channels == embed_dims
                else Linear(self.in_channels, embed_dims)
            )

    def forward(self, x, identity=None):
        """
        Forward pass through the network.

        Args:
            x: Input tensor
            identity: Identity tensor for residual connection

        Returns:
            Output tensor with optional residual connection
        """
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)

        if not self.add_identity:
            return self.dropout_layer(out)

        if identity is None:
            identity = x
        identity = self.identity_fc(identity)
        return identity + self.dropout_layer(out)
