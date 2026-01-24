import math
import os
import numpy as np
from torch.utils.data import Dataset
from .eval_utils.nuscenes_eval import NuScenesEval_custom, TrackingEval_custom
from nuscenes.utils.data_classes import Box as NuScenesBox
from nuscenes.eval.detection.config import config_factory as det_configs
from nuscenes.eval.common.config import config_factory as track_configs
import mmcv
from mmdet.datasets import DATASETS
from mmdet.datasets.pipelines import Compose

from os import path as osp
from nuscenes.eval.common.utils import Quaternion
import tempfile

from nuscenes import NuScenes
import wandb


@DATASETS.register_module()
class GriffinDataset(Dataset):
    DefaultAttribute = {
        "car": "vehicle.parked",
        "pedestrian": "pedestrian.moving",
        "trailer": "vehicle.parked",
        "truck": "vehicle.parked",
        "bus": "vehicle.moving",
        "motorcycle": "cycle.without_rider",
        "construction_vehicle": "vehicle.parked",
        "bicycle": "cycle.without_rider",
        "barrier": "",
        "traffic_cone": "",
    }
    ErrNameMapping = {
        "trans_err": "mATE",
        "scale_err": "mASE",
        "orient_err": "mAOE",
        "vel_err": "mAVE",
        "attr_err": "mAAE",
    }
    CLASSES = (
        "car",
        "truck",
        "trailer",
        "bus",
        "construction_vehicle",
        "bicycle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "barrier",
    )
    ID_COLOR_MAP = [
        (59, 59, 238),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 255),
        (0, 127, 255),
        (71, 130, 255),
        (127, 127, 0),
    ]

    def __init__(
        self,
        ann_file,
        pipeline=None,
        data_root=None,
        classes=None,
        load_interval=1,
        with_velocity=True,
        modality=None,
        test_mode=False,
        det3d_eval_version="detection_cvpr_2019",
        track3d_eval_version="tracking_nips_2019",
        version="v1.0-trainval",
        use_valid_flag=False,
        vis_score_threshold=0.25,
        data_aug_conf=None,
        sequences_split_num=1,
        with_seq_flag=False,
        keep_consistent_seq_aug=True,
        tracking=False,
        tracking_threshold=0.2,
        # inf_keys=[],
        splits_data_file="",
        v2x_side="",
        save_dn_for_coop=False,
        eval_mod=['det', 'track'],
    ):
        self.version = version
        self.load_interval = load_interval
        self.use_valid_flag = use_valid_flag
        super().__init__()
        self.data_root = data_root
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.modality = modality
        self.box_mode_3d = 0

        if classes is not None:
            self.CLASSES = classes
        self.cat2id = {name: i for i, name in enumerate(self.CLASSES)}
        self.data_infos = self.load_annotations(self.ann_file)

        if pipeline is not None:
            self.pipeline = Compose(pipeline)

        self.with_velocity = with_velocity
        self.det3d_eval_version = det3d_eval_version
        self.det3d_eval_configs = det_configs(self.det3d_eval_version)
        self.track3d_eval_version = track3d_eval_version
        self.track3d_eval_configs = track_configs(self.track3d_eval_version)

        self.vis_score_threshold = vis_score_threshold

        self.data_aug_conf = data_aug_conf
        self.tracking = tracking
        self.tracking_threshold = tracking_threshold
        self.sequences_split_num = sequences_split_num
        self.keep_consistent_seq_aug = keep_consistent_seq_aug
        self.current_aug = None
        self.last_id = None
        if with_seq_flag:
            self._set_sequence_group_flag()

        self.splits_data_file = splits_data_file
        self.v2x_side = v2x_side
        self.save_dn_for_coop = save_dn_for_coop
        if self.v2x_side not in [
            "vehicle-side",
            "drone-side",
            "cooperative",
            "early-fusion",
        ]:
            raise Exception("v2x_side is not correct with {}".format(self.v2x_side))

        self.det3d_eval_configs.class_names = self.CLASSES
        self.track3d_eval_configs.class_names = self.CLASSES

        self.eval_mod = eval_mod

    def __len__(self):
        return len(self.data_infos)

    def _set_sequence_group_flag(self):
        """
        Sets a flag for each sample indicating which sequence it belongs to.
        """
        # Initialize list to store sequence IDs for each sample
        res = []

        # Start with sequence ID 0
        curr_sequence = 0
        for idx in range(len(self.data_infos)):
            # In Griffin dataset, a new sequence begins when:
            # 1. It's not the first frame (idx != 0)
            # 2. It has a different scene_token
            if (
                idx != 0
                and self.data_infos[idx]["scene_token"]
                != self.data_infos[idx - 1]["scene_token"]
            ):
                # When these conditions are met, increment the sequence counter
                curr_sequence += 1
            # Assign current sequence ID to this sample
            res.append(curr_sequence)

        # Convert list to numpy array for efficient processing
        self.flag = np.array(res, dtype=np.int64)

        # Optional: Further split sequences into sub-sequences if requested
        if self.sequences_split_num != 1:
            if self.sequences_split_num == "all":
                # Special case: treat each sample as its own sequence
                self.flag = np.array(range(len(self.data_infos)), dtype=np.int64)
            else:
                # Count how many samples are in each sequence
                bin_counts = np.bincount(self.flag)
                new_flags = []
                curr_new_flag = 0

                # Process each original sequence
                for curr_flag in range(len(bin_counts)):
                    # Calculate split points for this sequence
                    # Creates an array of indices where each sub-sequence should start,
                    # plus the end index of the sequence
                    curr_sequence_length = np.array(
                        list(
                            range(
                                0,
                                bin_counts[curr_flag],
                                math.ceil(
                                    bin_counts[curr_flag] / self.sequences_split_num
                                ),
                            )
                        )
                        + [bin_counts[curr_flag]]
                    )

                    # For each sub-sequence segment
                    for sub_seq_idx in (
                        curr_sequence_length[1:] - curr_sequence_length[:-1]
                    ):
                        # Assign the new sequence ID to all samples in this sub-sequence
                        for _ in range(sub_seq_idx):
                            new_flags.append(curr_new_flag)
                        # Increment sub-sequence counter
                        curr_new_flag += 1

                # Validation: ensure we have the same number of flags as original samples
                assert len(new_flags) == len(self.flag)
                # Validation: ensure we have the expected number of unique sequence IDs
                assert (
                    len(np.bincount(new_flags))
                    == len(np.bincount(self.flag)) * self.sequences_split_num
                )
                # Update flags with new sub-sequence assignments
                self.flag = np.array(new_flags, dtype=np.int64)

    def get_augmentation(self):
        if self.data_aug_conf is None:
            return None
        H, W = self.data_aug_conf["H"], self.data_aug_conf["W"]
        fH, fW = self.data_aug_conf["final_dim"]
        if not self.test_mode:
            resize = np.random.uniform(*self.data_aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.random.uniform(*self.data_aug_conf["bot_pct_lim"])) * newH)
                - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.data_aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf["rot_lim"])
            rotate_3d = np.random.uniform(*self.data_aug_conf["rot3d_range"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate = 0
            rotate_3d = 0
        aug_config = {
            "resize": resize,
            "resize_dims": resize_dims,
            "crop": crop,
            "flip": flip,
            "rotate": rotate,
            "rotate_3d": rotate_3d,
        }
        return aug_config

    def __getitem__(self, idx):
        if isinstance(idx, dict):
            aug_config = idx["aug_config"]
            idx = idx["idx"]
        else:
            aug_config = self.get_augmentation()
        data = self.get_data_info(idx)
        data["aug_config"] = aug_config
        data = self.pipeline(data)
        return data

    def get_cat_ids(self, idx):
        info = self.data_infos[idx]
        if self.use_valid_flag:
            mask = info["valid_flag"]
            gt_names = set(info["gt_names"][mask])
        else:
            gt_names = set(info["gt_names"])

        cat_ids = []
        for name in gt_names:
            if name in self.CLASSES:
                cat_ids.append(self.cat2id[name])
        return cat_ids

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.
        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations sorted by timestamps.
        """
        data = mmcv.load(ann_file, file_format="pkl")
        data_infos = list(sorted(data["infos"], key=lambda e: e["timestamp"]))
        data_infos = data_infos[:: self.load_interval]
        self.metadata = data["metadata"]
        self.version = self.metadata["version"]

        return data_infos

    def get_data_info(self, index):
        """Get data info according to the given index.

        Args:
            index (int): Index of the sample data to get.
        Returns:
            dict: Data information that will be passed to the data preprocessing pipelines.
        """
        info = self.data_infos[index]

        input_dict = dict(
            sample_idx=info["token"],
            # pts_filename=info["lidar_path"],
            # sweeps=info["sweeps"],
            timestamp=info["timestamp"] / 1e6,
            lidar2ego_translation=info["lidar2ego_translation"],
            lidar2ego_rotation=info["lidar2ego_rotation"],
            ego2global_translation=info["ego2global_translation"],
            ego2global_rotation=info["ego2global_rotation"],
        )
        lidar2ego = np.eye(4)
        lidar2ego[:3, :3] = Quaternion(info["lidar2ego_rotation"]).rotation_matrix

        lidar2ego[:3, 3] = np.array(info["lidar2ego_translation"])

        ego2global = np.eye(4)
        ego2global[:3, :3] = Quaternion(info["ego2global_rotation"]).rotation_matrix
        ego2global[:3, 3] = np.array(info["ego2global_translation"])
        input_dict["lidar2global"] = ego2global @ lidar2ego

        if self.v2x_side == "cooperative":
            input_dict.update(
                dict(
                    veh2inf_rt=np.array(info["vehLidar2airLidar_rt"]),
                )
            )
        if "air_sample_token" in info:
            input_dict.update(dict(sample_idx_inf=info["air_sample_token"]))

        if self.modality["use_camera"]:
            image_paths = []
            lidar2img_rts = []
            lidar2cam_rts = []
            cam_intrinsics = []
            for cam_type, cam_info in info["cams"].items():
                image_paths.append(os.path.join(self.data_root, cam_info["data_path"]))

                # obtain lidar to image transformation matrix
                lidar2cam_r = Quaternion(cam_info["lidar2cam_rotation"]).rotation_matrix
                lidar2cam_t = cam_info["lidar2cam_translation"]
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r
                lidar2cam_rt[:3, 3] = lidar2cam_t

                intrinsic = np.array(cam_info["cam_intrinsic"])
                viewpad = np.eye(4)
                viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
                lidar2img_rt = viewpad @ lidar2cam_rt

                lidar2img_rts.append(lidar2img_rt)
                cam_intrinsics.append(viewpad)
                lidar2cam_rts.append(lidar2cam_rt)

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                )
            )

        if (not self.test_mode) or (self.save_dn_for_coop):
            annos = self.get_ann_info(index)
            input_dict.update(annos)
        return input_dict

    def get_ann_info(self, index):
        """Get annotation info according to the given index.

        Args:
            index (int): Index of the annotation data to get.

        Returns:
            dict: Annotation information consists of the following keys:

                - gt_bboxes_3d (:obj:`LiDARInstance3DBoxes`): \
                    3D ground truth bboxes
                - gt_labels_3d (np.ndarray): Labels of ground truths.
                - gt_names (list[str]): Class names of ground truths.
                - instance_inds (np.ndarray): Instance ids of ground truths.
        """
        info = self.data_infos[index]
        # filter out bbox containing no points
        if self.use_valid_flag:
            mask = info["valid_flag"]
        else:
            mask = info["num_lidar_pts"] > 0

        gt_bboxes_3d = info["gt_boxes"][mask]
        # cx, cy, cz, l, w, h, yaw(0: x, pi/2: y)

        gt_names_3d = info["gt_names"][mask]
        instance_inds = info["gt_inds"][mask]

        # sample = self.nusc.get("sample", info["token"])
        # ann_tokens = np.array(sample["anns"])[mask]
        # assert ann_tokens.shape[0] == gt_bboxes_3d.shape[0]

        gt_labels_3d = []
        for cat in gt_names_3d:
            if cat in self.CLASSES:
                gt_labels_3d.append(self.CLASSES.index(cat))
            else:
                gt_labels_3d.append(-1)
        gt_labels_3d = np.array(gt_labels_3d)

        if self.with_velocity:
            gt_velocity = info["gt_velocity"][mask]
            nan_mask = np.isnan(gt_velocity[:, 0])
            gt_velocity[nan_mask] = [0.0, 0.0, 0.0]
            gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

        # gt_bboxes_3d = LiDARInstance3DBoxes(
        #     gt_bboxes_3d, box_dim=gt_bboxes_3d.shape[-1], origin=(0.5, 0.5, 0.5)
        # ).convert_to(self.box_mode_3d)

        anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            instance_inds=instance_inds,
        )

        # Get raw track ids for cooperative denoise
        if self.v2x_side == "cooperative":
            anns_results.update(
                dict(
                    instance_id_veh=instance_inds,
                    instance_id_inf=instance_inds,
                )
            )

        # Get 2D annotations for dynamic anchor generation
        if "bboxes2d" in info:
            gt_bboxes_2d = info["bboxes2d"]
            centers_2d = info["centers2d"]
            depths = info["depths"]
            gt_labels_2d = []
            for cam_label in info["labels2d"]:
                gt_labels_2d_cam = []
                for cat in cam_label:
                    if cat in self.CLASSES:
                        gt_labels_2d_cam.append(self.CLASSES.index(cat))
                    else:
                        gt_labels_2d_cam.append(-1)
                gt_labels_2d.append(np.array(gt_labels_2d_cam))

            anns_results.update(
                dict(
                    gt_bboxes_2d=gt_bboxes_2d,
                    gt_labels_2d=gt_labels_2d,
                    centers_2d=centers_2d,
                    depths=depths,
                )
            )

        return anns_results

    def _format_bbox(self, results, jsonfile_prefix=None, tracking=False):
        """Convert the results to the standard format.
        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.
        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}
        mapped_class_names = self.CLASSES

        print(f"Start to convert {'tracking' if tracking else 'detection'} format...")
        for sample_id, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_nusc_box(
                det, threshold=self.tracking_threshold if tracking else None
            )
            sample_token = self.data_infos[sample_id]["token"]
            boxes = lidar_nusc_box_to_global(
                self.data_infos[sample_id],
                boxes,
                mapped_class_names,
                self.det3d_eval_configs,
                self.det3d_eval_version,
            )
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if tracking and name in [
                    "barrier",
                    "traffic_cone",
                    "construction_vehicle",
                ]:
                    continue
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = GriffinDataset.DefaultAttribute[name]
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = GriffinDataset.DefaultAttribute[name]

                # # center_ = box.center.tolist()
                # # change from ground height to center height
                # # center_[2] = center_[2] + (box.wlh.tolist()[2] / 2.0)
                # if name not in [
                #     "car",
                #     "truck",
                #     "bus",
                #     "trailer",
                #     "motorcycle",
                #     "bicycle",
                #     "pedestrian",
                # ]:
                #     continue

                # box_ego = boxes_ego[keep_idx[i]]
                # trans = box_ego.center
                # if "traj" in det:
                #     traj_local = det["traj"][keep_idx[i]].numpy()[..., :2]
                #     traj_scores = det["traj_scores"][keep_idx[i]].numpy()
                # else:
                #     traj_local = np.zeros((0,))
                #     traj_scores = np.zeros((0,))
                # traj_ego = np.zeros_like(traj_local)
                # rot = Quaternion(axis=np.array([0, 0.0, 1.0]), angle=np.pi / 2)
                # for kk in range(traj_ego.shape[0]):
                #     traj_ego[kk] = convert_local_coords_to_global(
                #         traj_local[kk], trans, rot
                #     )

                nusc_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    velocity=box.velocity[:2].tolist(),
                )
                if not tracking:
                    nusc_anno.update(
                        dict(
                            detection_name=name,
                            detection_score=box.score,
                            attribute_name=attr,
                        )
                    )
                else:
                    nusc_anno.update(
                        dict(
                            tracking_name=name,
                            tracking_score=box.score,
                            tracking_id=box.token,
                        )
                    )

                annos.append(nusc_anno)
            nusc_annos[sample_token] = annos
        nusc_submissions = {
            "meta": self.modality,
            "results": nusc_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, "results_nusc.json")
        print("Writing results to", res_path)
        res_path = osp.join(
            jsonfile_prefix,
            "results_nusc_tracking.json" if tracking else "results_nusc_detection.json",
        )
        mmcv.dump(nusc_submissions, res_path)
        return res_path

    def _evaluate_single(
        self, result_path, logger=None, result_name="img_bbox", tracking=False
    ):
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str | None): Logger used for printing
                    related information during evaluation. Default: None.
            result_name (str): Result name in the metric prefix.
                Default: "img_bbox".

        Returns:
            dict: Dictionary of evaluation details.
        """

        output_dir = osp.join(*osp.split(result_path)[:-1])
        output_dir_det = osp.join(output_dir, 'det')
        output_dir_track = osp.join(output_dir, 'track')
        mmcv.mkdir_or_exist(output_dir_det)
        mmcv.mkdir_or_exist(output_dir_track)

        nusc = NuScenes(version=self.version, dataroot=self.data_root, verbose=False)
        eval_set_map = {
            'v1.0-mini': 'mini_val',
            'v1.0-trainval': 'val',
        }
        detail = dict()
        metric_prefix = f'{result_name}'

        splits = self.create_splits_griffin()

        if not tracking:
            self.nusc_eval = NuScenesEval_custom(
                nusc,
                config=self.det3d_eval_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir_det,
                verbose=True,
                overlap_test=False,
                data_infos=self.data_infos,
                splits=splits,
                category_to_type_name=self.category_to_type_name,
            )
            self.nusc_eval.main(plot_examples=0, render_curves=False)

            metrics = mmcv.load(osp.join(output_dir_det, 'metrics_summary.json'))
            for name in self.CLASSES:
                for k, v in metrics['label_aps'][name].items():
                    val = float('{:.4f}'.format(v))
                    detail['{}/{}_AP_dist_{}'.format(metric_prefix, name, k)] = val
                for k, v in metrics['label_tp_errors'][name].items():
                    val = float('{:.4f}'.format(v))
                    detail['{}/{}_{}'.format(metric_prefix, name, k)] = val
                for k, v in metrics['tp_errors'].items():
                    val = float('{:.4f}'.format(v))
                    detail['{}/{}'.format(metric_prefix, self.ErrNameMapping[k])] = val
            detail['{}/NDS'.format(metric_prefix)] = metrics['nd_score']
            detail['{}/mAP'.format(metric_prefix)] = metrics['mean_ap']

        else:
            self.nusc_eval_track = TrackingEval_custom(
                nusc,
                config=self.track3d_eval_configs,
                result_path=result_path,
                eval_set=eval_set_map[self.version],
                output_dir=output_dir_track,
                verbose=True,
                splits=splits,
                category_to_type_name=self.category_to_type_name,
            )
            try:
                self.nusc_eval_track.main(render_curves=False)

                metrics = mmcv.load(osp.join(output_dir_track, 'metrics_summary.json'))
                keys = [
                    'amota',
                    'amotp',
                    'recall',
                    'motar',
                    'gt',
                    'mota',
                    'motp',
                    'mt',
                    'ml',
                    'faf',
                    'tp',
                    'fp',
                    'fn',
                    'ids',
                    'frag',
                    'tid',
                    'lgd',
                ]
                for key in keys:
                    detail['{}/{}'.format(metric_prefix, key)] = metrics[key]
            except AssertionError as e:
                if "len(scores) / gt_box_count" in str(e):
                    print(f"Warning: Assertion bypassed in nusc_eval_track.main - {e}")
                    # Continue without tracking evaluation
                else:
                    raise e

        # Log metrics to wandb
        if wandb.run is not None:
            wandb.log(detail)

        return detail

    def format_results(self, results, jsonfile_prefix=None, tracking=False):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: Returns (result_files, tmp_dir), where `result_files` is a \
                dict containing the json filepaths, `tmp_dir` is the temporal \
                directory created for saving json files when \
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), "results must be a list"
        assert len(results) == len(
            self
        ), f"The length of results is not equal to the dataset len: {len(results)} != {len(self)}"

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, "results")
        else:
            tmp_dir = None

        if not ("pts_bbox" in results[0] or "img_bbox" in results[0]):
            result_files = self._format_bbox(
                results, jsonfile_prefix, tracking=tracking
            )
        else:
            result_files = dict()
            for name in results[0]:
                print(f"\nFormating bboxes of {name}")
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_files.update(
                    {name: self._format_bbox(results_, tmp_file_, tracking=tracking)}
                )
        return result_files, tmp_dir

    def evaluate(
        self,
        results,
        metric=None,
        logger=None,
        jsonfile_prefix=None,
        result_names=["img_bbox"],
        show=False,
        out_dir=None,
        pipeline=None,
    ):
        """Evaluation in nuScenes protocol.
        Args:
            results (list[dict]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            show (bool): Whether to visualize.
                Default: False.
            out_dir (str): Path to save the visualization results.
                Default: None.
            pipeline (list[dict], optional): raw data loading for showing.
                Default: None.
        Returns:
            dict[str, float]: Results of each evaluation metric.
        """
        results_dict = dict()
        for metric in self.eval_mod:
            tracking = metric == "track"
            if tracking and not self.tracking:
                continue
            # if isinstance(results, dict):
            #     results = results["bbox_results"]
            result_files, tmp_dir = self.format_results(
                results, jsonfile_prefix, tracking
            )

            if isinstance(result_files, dict):
                for name in result_names:
                    print("Evaluating bboxes of {}".format(name))
                    ret_dict = self._evaluate_single(
                        result_files[name], tracking=tracking
                    )
                results_dict.update(ret_dict)
            elif isinstance(result_files, str):
                ret_dict = self._evaluate_single(result_files, tracking=tracking)
                results_dict.update(ret_dict)

            if tmp_dir is not None:
                tmp_dir.cleanup()

        return results_dict

    def show(self, results, save_dir=None, show=False, pipeline=None):
        assert False, "Not implemented, please use tools/visual_griffin.py instead"

    def create_splits_griffin(self):
        split_data = mmcv.load(self.splits_data_file)

        return split_data['batch_split']

    def category_to_type_name(self, category_name: str):
        if category_name in self.CLASSES:
            return category_name
        else:
            return None


def output_to_nusc_box(detection, threshold=None):
    """Convert the output to the box class in the nuScenes.
    Args:
        detection (dict): Detection results.
            - boxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.
    Returns:
        list[:obj:`NuScenesBox`]: List of standard NuScenesBoxes.
    """
    box3d = detection["boxes_3d"]
    scores = detection["scores_3d"].numpy()
    labels = detection["labels_3d"].numpy()
    if "instance_ids" in detection:
        ids = detection["instance_ids"].cpu().numpy()
    # else:
    #     ids = np.ones_like(labels)
    if threshold is not None:
        if "cls_scores" in detection:
            mask = detection["cls_scores"].numpy() >= threshold
        else:
            mask = scores >= threshold
        box3d = box3d[mask]
        scores = scores[mask]
        labels = labels[mask]
        ids = ids[mask]

    if hasattr(box3d, "gravity_center"):
        box_gravity_center = box3d.gravity_center.numpy()
        box_dims = box3d.dims.numpy()
        nus_box_dims = box_dims[:, [1, 0, 2]]
        box_yaw = box3d.yaw.numpy()
    else:
        box3d = box3d.numpy()
        box_gravity_center = box3d[..., :3].copy()
        box_dims = box3d[..., 3:6].copy()
        nus_box_dims = box_dims[..., [1, 0, 2]]
        box_yaw = box3d[..., 6].copy()

    box_list = []
    for i in range(len(box3d)):
        quat = Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        if hasattr(box3d, "gravity_center"):
            velocity = (*box3d.tensor[i, 7:9], 0.0)
        else:
            velocity = (*box3d[i, 7:9], 0.0)

        if np.any(np.isnan(box_gravity_center[i])):
            print(f"Warning: box_gravity_center[i] is nan: {box_gravity_center[i]}")
            continue

        box = NuScenesBox(
            box_gravity_center[i],
            nus_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i],
            velocity=velocity,
        )
        if "instance_ids" in detection:
            box.token = ids[i]
        box_list.append(box)
    return box_list


def lidar_nusc_box_to_global(
    info, boxes, classes, eval_configs, eval_version="detection_cvpr_2019"
):
    """Convert the box from ego to global coordinate.
    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (list[str]): Mapped classes in the evaluation.
        eval_configs (object): Evaluation configuration object.
        eval_version (str, optional): Evaluation version.
            Default: "detection_cvpr_2019"
    Returns:
        list: List of standard NuScenesBoxes in the global
            coordinate.
    """
    box_list = []
    for i, box in enumerate(boxes):
        # Move box to ego vehicle coord system
        box.rotate(Quaternion(info["lidar2ego_rotation"]))
        box.translate(np.array(info["lidar2ego_translation"]))
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(Quaternion(info["ego2global_rotation"]))
        box.translate(np.array(info["ego2global_translation"]))
        box_list.append(box)
    return box_list
