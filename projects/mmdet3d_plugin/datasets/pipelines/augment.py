import torch

import numpy as np
from numpy import random
import mmcv
from mmdet.datasets.builder import PIPELINES
from PIL import Image
from scipy.spatial.transform import Rotation as R

@PIPELINES.register_module()
class ResizeCropFlipImage(object):
    def __call__(self, results):
        aug_config = results.get("aug_config")
        if aug_config is None:
            return results
        imgs = results["img"]
        N = len(imgs)
        new_imgs = []
        for i in range(N):
            img, mat = self._img_transform(np.uint8(imgs[i]), aug_config)
            new_imgs.append(np.array(img).astype(np.float32))
            results["lidar2img"][i] = mat @ results["lidar2img"][i]
            if "cam_intrinsic" in results:
                results["cam_intrinsic"][i][:3, :3] *= aug_config["resize"]
                # results["cam_intrinsic"][i][:3, :3] = (
                #     mat[:3, :3] @ results["cam_intrinsic"][i][:3, :3]
                # )
            if "gt_bboxes_2d" in results:
                results["gt_bboxes_2d"][i] = self._2d_box_transform(
                    results["gt_bboxes_2d"][i], aug_config
                )
            if "centers_2d" in results:
                results["centers_2d"][i] = self._2d_point_transform(
                    results["centers_2d"][i], aug_config
                )
        results["img"] = new_imgs
        results["img_shape"] = [x.shape[:2] for x in new_imgs]
        return results

    def _img_transform(self, img, aug_configs):
        H, W = img.shape[:2]
        resize = aug_configs.get("resize", 1)
        resize_dims = (int(W * resize), int(H * resize))
        crop = aug_configs.get("crop", [0, 0, *resize_dims])
        flip = aug_configs.get("flip", False)
        rotate = aug_configs.get("rotate", 0)

        origin_dtype = img.dtype
        if origin_dtype != np.uint8:
            min_value = img.min()
            max_vaule = img.max()
            scale = 255 / (max_vaule - min_value)
            img = (img - min_value) * scale
            img = np.uint8(img)
        img = Image.fromarray(img)
        img = img.resize(resize_dims).crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        img = np.array(img).astype(np.float32)
        if origin_dtype != np.uint8:
            img = img.astype(np.float32)
            img = img / scale + min_value

        transform_matrix = np.eye(3)
        transform_matrix[:2, :2] *= resize
        transform_matrix[:2, 2] -= np.array(crop[:2])
        if flip:
            flip_matrix = np.array([[-1, 0, crop[2] - crop[0]], [0, 1, 0], [0, 0, 1]])
            transform_matrix = flip_matrix @ transform_matrix
        rotate = rotate / 180 * np.pi
        rot_matrix = np.array(
            [
                [np.cos(rotate), np.sin(rotate), 0],
                [-np.sin(rotate), np.cos(rotate), 0],
                [0, 0, 1],
            ]
        )
        rot_center = np.array([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        rot_matrix[:2, 2] = -rot_matrix[:2, :2] @ rot_center + rot_center
        transform_matrix = rot_matrix @ transform_matrix
        extend_matrix = np.eye(4)
        extend_matrix[:3, :3] = transform_matrix
        return img, extend_matrix

    def _2d_box_transform(self, boxes, aug_configs):
        if len(boxes) == 0:
            return boxes

        resize = aug_configs.get("resize", 1)
        crop = aug_configs.get("crop", [0, 0, 0, 0])
        flip = aug_configs.get("flip", False)
        rotate = aug_configs.get("rotate", 0)

        # Format of boxes: [u_min, v_min, u_max, v_max]
        boxes = boxes.copy()

        # Step 1: Apply resize
        boxes *= resize

        # Step 2: Apply crop offset
        boxes[:, 0] -= crop[0]  # u_min
        boxes[:, 2] -= crop[0]  # u_max
        boxes[:, 1] -= crop[1]  # v_min
        boxes[:, 3] -= crop[1]  # v_max

        # Step 3: Apply flip
        if flip:
            crop_width = crop[2] - crop[0]
            flipped_x_min = crop_width - boxes[:, 2]
            flipped_x_max = crop_width - boxes[:, 0]
            boxes[:, 0] = flipped_x_min
            boxes[:, 2] = flipped_x_max

        # Step 4: Apply rotation
        if rotate != 0:
            rotate_rad = rotate / 180 * np.pi
            rot_matrix = np.array(
                [
                    [np.cos(rotate_rad), np.sin(rotate_rad)],
                    [-np.sin(rotate_rad), np.cos(rotate_rad)],
                ]
            )

            # Get center of crop
            crop_width = crop[2] - crop[0]
            crop_height = crop[3] - crop[1]
            rot_center = np.array([crop_width, crop_height]) / 2

            # For each box, convert corners, rotate, then get new bounds
            new_boxes = []
            for box in boxes:
                corners = np.array(
                    [
                        [box[0], box[1]],  # top-left
                        [box[2], box[1]],  # top-right
                        [box[0], box[3]],  # bottom-left
                        [box[2], box[3]],  # bottom-right
                    ]
                )

                # Center corners around rotation center
                corners -= rot_center

                # Rotate corners
                corners = corners @ rot_matrix.T

                # Move corners back
                corners += rot_center

                # Get new bounds
                rotated_box = [
                    np.min(corners[:, 0]),  # new u_min
                    np.min(corners[:, 1]),  # new v_min
                    np.max(corners[:, 0]),  # new u_max
                    np.max(corners[:, 1]),  # new v_max
                ]
                new_boxes.append(rotated_box)

            boxes = np.array(new_boxes)

        return boxes

    def _2d_point_transform(self, points, aug_configs):
        if len(points) == 0:
            return points

        resize = aug_configs.get("resize", 1)
        crop = aug_configs.get("crop", [0, 0, 0, 0])
        flip = aug_configs.get("flip", False)
        rotate = aug_configs.get("rotate", 0)

        # Format of points: [center_u, center_v]
        points = points.copy()

        # Step 1: Apply resize
        points *= resize

        # Step 2: Apply crop offset
        points[:, 0] -= crop[0]  # u
        points[:, 1] -= crop[1]  # v

        # Step 3: Apply flip
        if flip:
            crop_width = crop[2] - crop[0]
            points[:, 0] = crop_width - points[:, 0]

        # Step 4: Apply rotation
        if rotate != 0:
            rotate_rad = rotate / 180 * np.pi
            rot_matrix = np.array(
                [
                    [np.cos(rotate_rad), np.sin(rotate_rad)],
                    [-np.sin(rotate_rad), np.cos(rotate_rad)],
                ]
            )

            # Get center of crop
            crop_width = crop[2] - crop[0]
            crop_height = crop[3] - crop[1]
            rot_center = np.array([crop_width, crop_height]) / 2

            # Center points around rotation center
            points -= rot_center

            # Rotate points
            points = points @ rot_matrix.T

            # Move points back
            points += rot_center

        return points


@PIPELINES.register_module()
class BBoxRotation(object):
    def __call__(self, results):
        angle = results["aug_config"]["rotate_3d"]
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)

        rot_mat = np.array(
            [
                [rot_cos, -rot_sin, 0, 0],
                [rot_sin, rot_cos, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        rot_mat_inv = np.linalg.inv(rot_mat)

        num_view = len(results["lidar2img"])
        for view in range(num_view):
            results["lidar2img"][view] = results["lidar2img"][view] @ rot_mat_inv
        if "lidar2global" in results:
            results["lidar2global"] = results["lidar2global"] @ rot_mat_inv
        if "gt_bboxes_3d" in results:
            results["gt_bboxes_3d"] = self.box_rotate(results["gt_bboxes_3d"], angle)
        return results

    @staticmethod
    def box_rotate(bbox_3d, angle):
        rot_cos = np.cos(angle)
        rot_sin = np.sin(angle)
        rot_mat_T = np.array([[rot_cos, rot_sin, 0], [-rot_sin, rot_cos, 0], [0, 0, 1]])
        bbox_3d[:, :3] = bbox_3d[:, :3] @ rot_mat_T
        bbox_3d[:, 6] += angle
        if bbox_3d.shape[-1] > 7:
            vel_dims = bbox_3d[:, 7:].shape[-1]
            bbox_3d[:, 7:] = bbox_3d[:, 7:] @ rot_mat_T[:vel_dims, :vel_dims]
        return bbox_3d

@PIPELINES.register_module()
class DropOutInf(object):
    """Dropout the drone file lines/images with a certain probability.
    Args:
        drop_prob (float): Probability of dropping out a certain line/image.
    """

    def __init__(self, drop_prob=0.01):
        self.drop_prob = drop_prob

    def __call__(self, results):
        """Call function to dropout data.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict with dropped out data.
        """
        num_rows = results['coop_instance_feature'].size(0)
        mask = torch.rand(num_rows) < self.drop_prob
        if mask.sum() > 0:
            results['coop_instance_feature'] = results['coop_instance_feature'][~mask,:]
            results['coop_instance_ids'] = results['coop_instance_ids'][~mask]
            results['coop_anchor'] = results['coop_anchor'][~mask,:]
            results['coop_label'] = results['coop_label'][~mask]
            results['coop_score'] = results['coop_score'][~mask]
            results['coop_cls_score'] = results['coop_cls_score'][~mask]
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(drop_prob={self.drop_prob})'
        return repr_str


@PIPELINES.register_module()
class AddNoiseInf(object):
    """Add noise to the position and pose of the vehicle relative to the drone/cam.
    Args:
        loc_noise_std (float): Standard deviation of the noise added to the
            position of the vehicle.d
        ori_noise_std (float): Standard deviation of the noise added to the
            pose of the vehicle.
    """

    def __init__(self, loc_noise_std=0.2, ori_noise_std=0.2):
        self.loc_noise_std = loc_noise_std
        self.ori_noise_std = ori_noise_std

    def __call__(self, results):
        """Call function to add noise to positions.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict with noisy positions.
        """
        random_angles = np.random.normal(0, self.ori_noise_std, 3)
        random_locs = np.random.normal(0, self.loc_noise_std, 3)
        R_matrix = R.from_euler('xyz', random_angles, degrees=True).as_matrix()
        results['veh2inf_rt'][:3, :3] = R_matrix @ results['veh2inf_rt'][:3, :3]
        results['veh2inf_rt'][:3, 3] += random_locs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(loc_noise_std={self.loc_noise_std}, ori_noise_std={self.ori_noise_std})'
        return repr_str


@PIPELINES.register_module()
class PhotoMetricDistortionMultiViewImage:
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.
    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels
    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(
        self,
        brightness_delta=32,
        contrast_range=(0.5, 1.5),
        saturation_range=(0.5, 1.5),
        hue_delta=18,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def __call__(self, results):
        """Call function to perform photometric distortion on images.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images distorted.
        """
        imgs = results["img"]
        new_imgs = []
        for img in imgs:
            assert img.dtype == np.float32, (
                "PhotoMetricDistortion needs the input image of dtype np.float32,"
                ' please set "to_float32=True" in "LoadImageFromFile" pipeline'
            )
            # random brightness
            if random.randint(2):
                delta = random.uniform(-self.brightness_delta, self.brightness_delta)
                img += delta

            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # convert color from BGR to HSV
            img = mmcv.bgr2hsv(img)

            # random saturation
            if random.randint(2):
                img[..., 1] *= random.uniform(
                    self.saturation_lower, self.saturation_upper
                )

            # random hue
            if random.randint(2):
                img[..., 0] += random.uniform(-self.hue_delta, self.hue_delta)
                img[..., 0][img[..., 0] > 360] -= 360
                img[..., 0][img[..., 0] < 0] += 360

            # convert color from HSV to BGR
            img = mmcv.hsv2bgr(img)

            # random contrast
            if mode == 0:
                if random.randint(2):
                    alpha = random.uniform(self.contrast_lower, self.contrast_upper)
                    img *= alpha

            # randomly swap channels
            if random.randint(2):
                img = img[..., random.permutation(3)]
            new_imgs.append(img)
        results["img"] = new_imgs
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f"(\nbrightness_delta={self.brightness_delta},\n"
        repr_str += "contrast_range="
        repr_str += f"{(self.contrast_lower, self.contrast_upper)},\n"
        repr_str += "saturation_range="
        repr_str += f"{(self.saturation_lower, self.saturation_upper)},\n"
        repr_str += f"hue_delta={self.hue_delta})"
        return repr_str
