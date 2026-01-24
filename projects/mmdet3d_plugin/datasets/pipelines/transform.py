import numpy as np
import mmcv
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
import torch


@PIPELINES.register_module()
class MultiScaleDepthMapGenerator(object):
    """Generate multi-scale depth maps from LiDAR points projected onto camera images."""
    def __init__(self, downsample=1, max_depth=60):
        if not isinstance(downsample, (list, tuple)):
            downsample = [downsample]
        self.downsample = downsample
        self.max_depth = max_depth

    def __call__(self, input_dict):
        """Process input data to generate multi-scale depth maps.

        Only the pixels where LiDAR points project get updated with depth values(0.1, max_depth=60),
        other pixels remain -1.

        Args:
            input_dict (dict): Input data dictionary containing:
                - points: LiDAR points with shape (N, 3)
                - lidar2img: List of transformation matrices from LiDAR to image coordinates
                - img_shape: List of image shapes (height, width, channels)

        Returns:
            dict: Updated input dictionary with added 'gt_depth' key containing
                multi-scale depth maps for each camera view.
        """
        # Extract 3D points and add a dimension for matrix multiplication
        points = input_dict["points"][..., :3, None]
        gt_depth = []

        # Process each camera view
        for i, lidar2img in enumerate(input_dict["lidar2img"]):
            H, W = input_dict["img_shape"][i][:2]

            # Project 3D points to 2D image coordinates
            pts_2d = np.squeeze(lidar2img[:3, :3] @ points, axis=-1) + lidar2img[:3, 3]
            pts_2d[:, :2] /= pts_2d[:, 2:3]  # Perspective division
            U = np.round(pts_2d[:, 0]).astype(np.int32)  # X coordinates
            V = np.round(pts_2d[:, 1]).astype(np.int32)  # Y coordinates
            depths = pts_2d[:, 2]  # Z coordinates (depth values)

            # Filter points that are within image boundaries and have valid depth
            mask = np.logical_and.reduce(
                [
                    V >= 0,
                    V < H,
                    U >= 0,
                    U < W,
                    depths >= 0.1,
                    # depths <= self.max_depth,  # Commented out to preserve points beyond max_depth, but will be clipped to max_depth soon
                ]
            )
            V, U, depths = V[mask], U[mask], depths[mask]

            # Sort points by depth (farthest first) to handle occlusions
            sort_idx = np.argsort(depths)[::-1]
            V, U, depths = V[sort_idx], U[sort_idx], depths[sort_idx]

            # Clip depth values to the specified maximum (No points are removed, just capped at the boundaries)
            depths = np.clip(depths, 0.1, self.max_depth)

            # Generate depth maps at different scales
            for j, downsample in enumerate(self.downsample):
                if len(gt_depth) < j + 1:
                    gt_depth.append([])

                # Calculate dimensions for downsampled depth map
                h, w = (int(H / downsample), int(W / downsample))
                u = np.floor(U / downsample).astype(np.int32)
                v = np.floor(V / downsample).astype(np.int32)

                # Initialize depth map with -1 (no depth information)
                depth_map = np.ones([h, w], dtype=np.float32) * -1

                # Assign depth values to corresponding pixel positions
                depth_map[v, u] = depths
                gt_depth[j].append(depth_map)

        # Stack depth maps for each scale and add to input dictionary
        input_dict["gt_depth"] = [np.stack(x) for x in gt_depth]
        return input_dict


@PIPELINES.register_module()
class NuScenesSparse4DAdaptor(object):
    """A data transformation pipeline for NuScenes dataset to adapt it for Sparse4D model."""
    def __init(self):
        pass

    def __call__(self, input_dict):
        # Convert lidar2img transformation matrices to float32 and stack them
        input_dict["projection_mat"] = np.float32(np.stack(input_dict["lidar2img"]))

        # Extract and format image dimensions (width, height) from img_shape
        # Note: [:, :2] takes first two elements (height, width), [:, ::-1] reverses to (width, height)
        input_dict["image_wh"] = np.ascontiguousarray(
            np.array(input_dict["img_shape"], dtype=np.float32)[:, :2][:, ::-1]
        )

        # Compute inverse of global transformation matrix for later use
        input_dict["T_global_inv"] = np.linalg.inv(input_dict["lidar2global"])
        input_dict["T_global"] = input_dict["lidar2global"]

        # Process camera intrinsic parameters if available
        if "cam_intrinsic" in input_dict:
            # Stack camera intrinsic matrices and convert to float32
            input_dict["cam_intrinsic"] = np.float32(
                np.stack(input_dict["cam_intrinsic"])
            )
            # Extract focal length from camera intrinsic matrix
            input_dict["focal"] = input_dict["cam_intrinsic"][..., 0, 0]

            # Alternative focal length calculation
            # input_dict["focal"] = np.sqrt(
            #     np.abs(np.linalg.det(input_dict["cam_intrinsic"][:, :2, :2]))
            # )

        # Rename instance indices if present
        if "instance_inds" in input_dict:
            input_dict["instance_id"] = input_dict["instance_inds"]

        # Process 3D bounding boxes if present
        if "gt_bboxes_3d" in input_dict:
            # Normalize rotation angles to be within [0, 2π]
            input_dict["gt_bboxes_3d"][:, 6] = self.limit_period(
                input_dict["gt_bboxes_3d"][:, 6], offset=0.5, period=2 * np.pi
            )
            # Convert to tensor and wrap in DataContainer
            input_dict["gt_bboxes_3d"] = DC(
                to_tensor(input_dict["gt_bboxes_3d"]).float()
            )

        # Process 3D labels if present
        if "gt_labels_3d" in input_dict:
            # Convert to tensor and wrap in DataContainer
            input_dict["gt_labels_3d"] = DC(
                to_tensor(input_dict["gt_labels_3d"]).long()
            )

        # Process 2D proposals if present
        if "gt_bboxes_2d" in input_dict:
            input_dict["gt_bboxes_2d"] = DC(
                [to_tensor(it).float() for it in input_dict["gt_bboxes_2d"]]
            )
        if "gt_labels_2d" in input_dict:
            input_dict["gt_labels_2d"] = DC(
                [to_tensor(it).long() for it in input_dict["gt_labels_2d"]]
            )
        if "centers_2d" in input_dict:
            input_dict["centers_2d"] = DC(
                [to_tensor(it).float() for it in input_dict["centers_2d"]]
            )
        if "depths" in input_dict:
            input_dict["depths"] = DC(
                [to_tensor(it).float() for it in input_dict["depths"]]
            )

        # Process cooperative instance information if present
        if "coop_instance_feature" in input_dict:
            # input_dict["coop_instance_feature"] = DC(
            #     [to_tensor(it).float() for it in input_dict["coop_instance_feature"]]
            # )
            input_dict["coop_instance_feature"] = DC(
                to_tensor(input_dict["coop_instance_feature"]).float()
            )
        if "coop_instance_ids" in input_dict:
            # input_dict["coop_instance_ids"] = DC(
            #     [to_tensor(it).long() for it in input_dict["coop_instance_ids"]]
            # )
            input_dict["coop_instance_ids"] = DC(
                to_tensor(input_dict["coop_instance_ids"]).long()
            )
        if "coop_anchor" in input_dict:
            # input_dict["coop_anchor"] = DC(
            #     [to_tensor(it).float() for it in input_dict["coop_anchor"]]
            # )
            input_dict["coop_anchor"] = DC(to_tensor(input_dict["coop_anchor"]).float())

        if "coop_label" in input_dict:
            # input_dict["coop_label"] = DC(
            #     [to_tensor(it).long() for it in input_dict["coop_label"]]
            # )
            input_dict["coop_label"] = DC(to_tensor(input_dict["coop_label"]).long())
        if "coop_score" in input_dict:
            # input_dict["coop_score"] = DC(
            #     [to_tensor(it).float() for it in input_dict["coop_score"]]
            # )
            input_dict["coop_score"] = DC(to_tensor(input_dict["coop_score"]).float())
        if "coop_cls_score" in input_dict:
            # input_dict["coop_cls_score"] = DC(
            #     [to_tensor(it).float() for it in input_dict["coop_cls_score"]]
            # )
            input_dict["coop_cls_score"] = DC(
                to_tensor(input_dict["coop_cls_score"]).float()
            )
        if "coop_dn_instance_feature" in input_dict:
            input_dict["coop_dn_instance_feature"] = DC(
                to_tensor(input_dict["coop_dn_instance_feature"]).float()
            )
        if "coop_dn_anchor" in input_dict:
            input_dict["coop_dn_anchor"] = DC(
                to_tensor(input_dict["coop_dn_anchor"]).float()
            )
        if "coop_dn_id_target" in input_dict:
            input_dict["coop_dn_id_target"] = DC(
                to_tensor(input_dict["coop_dn_id_target"]).long()
            )
        if "coop_dn_cls_target" in input_dict:
            input_dict["coop_dn_cls_target"] = DC(
                to_tensor(input_dict["coop_dn_cls_target"]).long()
            )
        if "coop_dn_reg_target" in input_dict:
            input_dict["coop_dn_reg_target"] = DC(
                to_tensor(input_dict["coop_dn_reg_target"]).float()
            )
        if "instance_id_inf" in input_dict:
            input_dict["instance_id_inf"] = DC(
                to_tensor(input_dict["instance_id_inf"]).long()
            )
        if "instance_id_veh" in input_dict:
            input_dict["instance_id_veh"] = DC(
                to_tensor(input_dict["instance_id_veh"]).long()
            )
        if "instance_id" in input_dict:
            input_dict["instance_id"] = DC(to_tensor(input_dict["instance_id"]).long())

        # Process image data
        # Convert images from (H, W, C) to (C, H, W) format and stack them
        imgs = [img.transpose(2, 0, 1) for img in input_dict["img"]]
        imgs = np.ascontiguousarray(np.stack(imgs, axis=0))
        # Convert to tensor and wrap in DataContainer
        input_dict["img"] = DC(to_tensor(imgs), stack=True)
        return input_dict

    def limit_period(
        self, val: np.ndarray, offset: float = 0.5, period: float = np.pi
    ) -> np.ndarray:
        """Normalize values to be within a specified period.

        Args:
            val: Input array of values to be normalized
            offset: Offset value for the normalization
            period: The period to normalize values within

        Returns:
            Normalized values within the specified period
        """
        limited_val = val - np.floor(val / period + offset) * period
        return limited_val


@PIPELINES.register_module()
class InstanceNameFilter(object):
    """Filter GT objects by their names.

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        gt_labels_3d = input_dict["gt_labels_3d"]
        gt_bboxes_mask = np.array(
            [n in self.labels for n in gt_labels_3d], dtype=np.bool_
        )
        input_dict["gt_bboxes_3d"] = input_dict["gt_bboxes_3d"][gt_bboxes_mask]
        input_dict["gt_labels_3d"] = input_dict["gt_labels_3d"][gt_bboxes_mask]
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][gt_bboxes_mask]
        if "instance_id_veh" in input_dict:
            input_dict["instance_id_veh"] = input_dict["instance_id_veh"][
                gt_bboxes_mask
            ]
        if "instance_id_inf" in input_dict:
            input_dict["instance_id_inf"] = input_dict["instance_id_inf"][
                gt_bboxes_mask
            ]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(classes={self.classes})"
        return repr_str


@PIPELINES.register_module()
class CircleObjectRangeFilter(object):
    """Filter objects based on their distance from ego vehicle."""

    def __init__(self, class_dist_thred=[52.5] * 5 + [31.5] + [42] * 3 + [31.5]):
        self.class_dist_thred = class_dist_thred

    def __call__(self, input_dict):
        gt_bboxes_3d = input_dict["gt_bboxes_3d"]
        gt_labels_3d = input_dict["gt_labels_3d"]
        dist = np.sqrt(np.sum(gt_bboxes_3d[:, :2] ** 2, axis=-1))
        mask = np.array([False] * len(dist))
        for label_idx, dist_thred in enumerate(self.class_dist_thred):
            mask = np.logical_or(
                mask, np.logical_and(gt_labels_3d == label_idx, dist <= dist_thred)
            )

        gt_bboxes_3d = gt_bboxes_3d[mask]
        gt_labels_3d = gt_labels_3d[mask]

        input_dict["gt_bboxes_3d"] = gt_bboxes_3d
        input_dict["gt_labels_3d"] = gt_labels_3d
        if "instance_inds" in input_dict:
            input_dict["instance_inds"] = input_dict["instance_inds"][mask]
        if "instance_id_veh" in input_dict:
            input_dict["instance_id_veh"] = input_dict["instance_id_veh"][mask]
        if "instance_id_inf" in input_dict:
            input_dict["instance_id_inf"] = input_dict["instance_id_inf"][mask]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f"(class_dist_thred={self.class_dist_thred})"
        return repr_str


@PIPELINES.register_module()
class NormalizeMultiviewImage(object):
    """Normalize the image.
    Added key is "img_norm_cfg".
    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        results["img"] = [
            mmcv.imnormalize(img, self.mean, self.std, self.to_rgb)
            for img in results["img"]
        ]
        results["img_norm_cfg"] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})"
        return repr_str

@PIPELINES.register_module()
class DepthMapGenerator(object):
    """Generate depth maps."""
    def __init__(
        self,
        depth_type=0,
        hidden_dim=256,
        num_depth_bins=50,
        depth_min=1e-1,
        depth_max=80,
        stride=4,
        input_shape=(800, 448),
    ):
        self.num_depth_bins = num_depth_bins
        self.depth_min = depth_min
        self.depth_max = depth_max
        self.stride = stride
        self.input_shape = input_shape

    def __call__(self, input_dict):
        gt_bboxes_2d = input_dict["gt_bboxes_2d"]
        depths = input_dict["depths"]

        W, H = self.input_shape
        assert (
            H % self.stride == 0 and W % self.stride == 0
        ), "input_shape must be divisible by stride"
        H, W = round(H / self.stride), round(W / self.stride)
        depth_maps, fg_mask = self.build_target_depth_from_3dcenter_argo(
            gt_bboxes_2d, depths, (H, W)
        )
        depth_target = self.bin_depths(
            depth_maps,
            "LID",
            self.depth_min,
            self.depth_max,
            self.num_depth_bins,
            target=True,
        )
        input_dict.update(
            dict(
                ins_depthmap=depth_target,
                ins_depthmap_mask=fg_mask,
            )
        )
        return input_dict

    def build_target_depth_from_3dcenter_argo(self, gt_boxes2d, gt_center_depth, HW):
        H, W = HW
        B = len(gt_boxes2d) # B is N indeed
        depth_maps = torch.zeros((B, H, W), dtype=torch.float)
        fg_mask = torch.zeros_like(depth_maps).bool()

        for b in range(B):
            center_depth_per_batch = torch.from_numpy(gt_center_depth[b])
            # Set box corners
            if gt_boxes2d[b].shape[0] > 0:
                gt_boxes_per_batch = torch.from_numpy(gt_boxes2d[b])
                # Transform raw shape to input shape
                gt_boxes_per_batch = gt_boxes_per_batch / self.stride
                gt_boxes_per_batch[:, :2] = torch.floor(gt_boxes_per_batch[:, :2])
                gt_boxes_per_batch[:, 2:] = torch.ceil(gt_boxes_per_batch[:, 2:])
                gt_boxes_per_batch = gt_boxes_per_batch.long()

                for n in range(gt_boxes_per_batch.shape[0]):
                    u1, v1, u2, v2 = gt_boxes_per_batch[n]
                    depth_maps[b, v1:v2, u1:u2] = center_depth_per_batch[n]
                    fg_mask[b, v1:v2, u1:u2] = True

        return depth_maps, fg_mask

    def bin_depths(self, depth_map, mode="LID", depth_min=1e-3, depth_max=60, num_bins=80, target=False):
        """
        Converts depth map into bin indices
        Args:
            depth_map [torch.Tensor(H, W)]: Depth Map
            mode [string]: Discretiziation mode (See https://arxiv.org/pdf/2005.13423.pdf for more details)
                UD: Uniform discretiziation
                LID: Linear increasing discretiziation
                SID: Spacing increasing discretiziation
            depth_min [float]: Minimum depth value
            depth_max [float]: Maximum depth value
            num_bins [int]: Number of depth bins
            target [bool]: Whether the depth bins indices will be used for a target tensor in loss comparison
        Returns:
            indices [torch.Tensor(H, W)]: Depth bin indices
        """
        if mode == "UD":
            bin_size = (depth_max - depth_min) / num_bins
            indices = ((depth_map - depth_min) / bin_size)
        elif mode == "LID":
            bin_size = 2 * (depth_max - depth_min) / (num_bins * (1 + num_bins))
            indices = -0.5 + 0.5 * torch.sqrt(1 + 8 * (depth_map - depth_min) / bin_size)
        # elif mode == "SID":
        #     indices = num_bins * (torch.log(1 + depth_map) - math.log(1 + depth_min)) / \
        #               (math.log(1 + depth_max) - math.log(1 + depth_min))
        else:
            raise NotImplementedError

        if target:
            # Remove indicies outside of bounds
            mask = (indices < 0) | (indices > num_bins) | (~torch.isfinite(indices))
            indices[mask] = num_bins

            # Convert to integer
            indices = indices.type(torch.int64)

        return indices
