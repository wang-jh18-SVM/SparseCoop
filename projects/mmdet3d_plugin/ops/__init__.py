import torch

from .deformable_aggregation import DeformableAggregationFunction


def deformable_aggregation_function(
    feature_maps, spatial_shape, scale_start_index, sampling_location, weights
):
    """Wrapper function for DeformableAggregationFunction.

    This function aggregates features from multi-camera, multi-scale feature maps based on sampling locations and weights using deformable attention mechanism.

    Args:
        feature_maps: Formatted feature maps from multiple cameras
        spatial_shape: Shape information of each feature level
        scale_start_index: Starting indices for each scale level
        sampling_location: Coordinates where features should be sampled
        weights: Importance weights for each sampled feature

    Returns:
        Aggregated features after deformable sampling
    """
    return DeformableAggregationFunction.apply(
        feature_maps, spatial_shape, scale_start_index, sampling_location, weights
    )


def feature_maps_format(feature_maps, inverse=False):
    """Format conversion for multi-camera, multi-scale feature maps.

    This function handles two operations:
    1. Forward (inverse=False): Converts standard feature maps to a flattened format suitable for deformable aggregation operations
    2. Inverse (inverse=True): Converts the flattened format back to standard multi-camera multi-scale feature maps

    Args:
        feature_maps:
            - When inverse=False: List of feature maps from multiple cameras and scales
              with shape [batch_size, num_cameras, channels, height, width]
            - When inverse=True: Tuple of [col_feats, spatial_shape, scale_start_index]
        inverse: Whether to perform inverse conversion

    Returns:
        - When inverse=False: List containing [col_feats, spatial_shape, scale_start_index]
        - When inverse=True: Original format multi-camera multi-scale feature maps
    """
    if inverse:
        ############# Inverse conversion: Convert from flattened format back to multi-camera multi-scale format ############
        col_feats, spatial_shape, scale_start_index = feature_maps
        num_cams, num_levels = spatial_shape.shape[:2]

        # Calculate feature map sizes at each level for splitting
        split_size = spatial_shape[..., 0] * spatial_shape[..., 1]
        split_size = split_size.cpu().numpy().tolist()

        # Group cameras with same feature map shapes
        idx = 0
        cam_split = [1]  # Count of cameras in each group
        cam_split_size = [
            sum(split_size[0])
        ]  # Total features to split for each camera group
        for i in range(num_cams - 1):
            if not torch.all(spatial_shape[i] == spatial_shape[i + 1]):
                # When encountering different spatial shapes, create a new camera group
                cam_split.append(0)
                cam_split_size.append(0)
            cam_split[-1] += 1
            cam_split_size[-1] += sum(split_size[i + 1])

        # Split features by camera groups
        mc_feat = [
            x.unflatten(1, (cam_split[i], -1))
            for i, x in enumerate(col_feats.split(cam_split_size, dim=1))
        ]

        # Convert spatial shape to list for easier processing
        spatial_shape = spatial_shape.cpu().numpy().tolist()
        mc_ms_feat = []  # Multi-camera multi-scale features
        shape_index = 0

        # Reconstruct original feature map shapes for each camera group
        for i, feat in enumerate(mc_feat):
            # Split features by levels within each camera group
            feat = list(feat.split(split_size[shape_index], dim=2))
            for j, f in enumerate(feat):
                # Reshape flattened features back to spatial dimensions
                feat[j] = f.unflatten(2, spatial_shape[shape_index][j])
                # Rearrange dimensions to [batch, camera, channel, height, width]
                feat[j] = feat[j].permute(0, 1, 4, 2, 3)
            mc_ms_feat.append(feat)
            shape_index += cam_split[i]
        return mc_ms_feat

    # Handle nested feature maps (e.g., from multiple groups)
    if isinstance(feature_maps[0], (list, tuple)):
        # Recursively format each feature map group
        formated = [feature_maps_format(x) for x in feature_maps]
        # Concatenate results from all groups
        col_feats = torch.cat([x[0] for x in formated], dim=1)
        spatial_shape = torch.cat([x[1] for x in formated], dim=0)
        scale_start_index = torch.cat([x[2] for x in formated], dim=0)
        return [col_feats, spatial_shape, scale_start_index]

    ############ Forward conversion: Standard feature maps to flattened format ############
    bs, num_cams = feature_maps[0].shape[:2]
    spatial_shape = []

    # Process each feature level
    col_feats = []
    for i, feat in enumerate(feature_maps):
        # Record spatial dimensions (height, width) for each level
        spatial_shape.append(feat.shape[-2:])
        # Flatten spatial dimensions while keeping batch, camera, and channel dims
        col_feats.append(torch.reshape(feat, (bs, num_cams, feat.shape[2], -1)))
        # [[4, 4, 256, 11264], [4, 4, 256, 2816], [4, 4, 256, 704], [4, 4, 256, 176]]

    # Concatenate all levels and rearrange dimensions
    col_feats = torch.cat(col_feats, dim=-1).permute(0, 1, 3, 2).flatten(1, 2)
    # [bs, num_cams*sum(Hi*Wi), C] [4, 59840, 256]

    # Create tensor of spatial shapes for all cameras (assuming same shape across cameras)
    spatial_shape = [spatial_shape] * num_cams
    spatial_shape = torch.tensor(
        spatial_shape, dtype=torch.int64, device=col_feats.device
    )

    # Calculate starting index for each scale level based on spatial dimensions
    # This helps locate features from different levels in the flattened tensor
    scale_start_index = spatial_shape[..., 0] * spatial_shape[..., 1]
    scale_start_index = scale_start_index.flatten().cumsum(dim=0)
    scale_start_index = torch.cat(
        [torch.tensor([0]).to(scale_start_index), scale_start_index[:-1]]
    )
    scale_start_index = scale_start_index.reshape(num_cams, -1)

    # Return the flattened features, spatial shapes, and scale indices
    feature_maps = [col_feats, spatial_shape, scale_start_index]
    return feature_maps
