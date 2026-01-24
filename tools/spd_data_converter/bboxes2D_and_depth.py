import argparse
from nuscenes.nuscenes import NuScenes
from nuscenes.eval.common.utils import Quaternion
import copy
import mmcv
import numpy as np
import cv2
import os


def boxes_to_corners_3d_lidar(boxes):
    """
    Convert 3D gt boxes (cx, cy, cz, l, w, h, yaw) to 8 top point,
    and return corners (N, 8, 3),in lidar coordinate
    """
    N = boxes.shape[0]

    x_corners = np.array([0.5, 0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5])
    y_corners = np.array([0.5, -0.5, -0.5, 0.5, 0.5, -0.5, -0.5, 0.5])
    z_corners = np.array([-0.5, -0.5, -0.5, -0.5, 0.5, 0.5, 0.5, 0.5])

    corners_3d = np.zeros((N, 8, 3))
    for i in range(N):
        x, y, z, l, w, h, yaw = boxes[i]

        corner = np.stack(
            [l * x_corners, w * y_corners, h * z_corners], axis=1
        )  # shape (8, 3)

        rot_mat = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )

        corners_3d[i] = np.dot(corner, rot_mat.T) + np.array([x, y, z])

    return corners_3d  # shape (N, 8, 3)


def add_2D_and_Depth_infos(nusc_infos, filter_3d_gt=False):

    modify_infos = copy.deepcopy(nusc_infos)

    for info in mmcv.track_iter_progress(modify_infos):

        corners_3d = boxes_to_corners_3d_lidar(info['gt_boxes'])  # lidar cooridnate
        N = corners_3d.shape[0]
        pts_lidar = info['gt_boxes'][:, :3]

        gt_2dbboxes_cams = []
        centers2d_cams = []
        gt_2dlabels_cams = []
        depths_cams = []

        # Track which 3D objects have valid 2D projections in any camera
        valid_mask = np.zeros(N, dtype=bool)

        for cam, cam_info in info['cams'].items():

            gt_2dbboxes = []
            centers2d = []
            gt_2dlabels = []
            depths = []

            rotation_matrix = cam_info['lidar2cam_rotation']
            lidar2cam_rt = np.eye(4)
            lidar2cam_rt[:3, :3] = rotation_matrix
            lidar2cam_rt[:3, 3] = cam_info['lidar2cam_translation'][:3, 0]
            cam2lidar_rt = np.linalg.inv(lidar2cam_rt)
            cam2lidar_t = cam2lidar_rt[:3, 3]
            intrinsic = np.array(cam_info['cam_intrinsic'])
            # img = cv2.imread(cam_info['data_path'])
            # H, W = img.shape[:2]
            H, W = 1080, 1920

            for i in range(N):

                pts = corners_3d[i]
                pts_h = np.concatenate(
                    [pts, np.ones((8, 1))], axis=1
                )  # homogeneous coordinate
                pts_cam = (lidar2cam_rt @ pts_h.T).T[:, :3]
                cam_x, cam_y, cam_z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]

                # # filter out objects that are too far away
                # # No need for real-collected data, especially for infrastructure side
                # if cam_z.max() > 110:
                #     continue

                u = (intrinsic[0, 0] * cam_x + intrinsic[0, 2] * cam_z) / cam_z
                v = (intrinsic[1, 1] * cam_y + intrinsic[1, 2] * cam_z) / cam_z
                uv = np.stack([u, v], axis=-1)
                valid_pts = (cam_z > 0) & (u >= 0) & (v >= 0) & (u < W) & (v < H)
                if valid_pts.sum() >= 4:
                    u_min = max(0, np.min(u))
                    v_min = max(0, np.min(v))
                    u_max = min(W, np.max(u))
                    v_max = min(H, np.max(v))

                    h = v_max - v_min
                    w = u_max - u_min
                    # Filter out boxes with width or height larger than 1/3 of the image height or width
                    # Parameters based on tools/visual_tools/statics_2d_gt.py
                    if h > 0.6 * H or w > 0.6 * W:
                        continue

                    center_u = w / 2 + u_min
                    center_v = h / 2 + v_min
                    bboxes = [u_min, v_min, u_max, v_max]
                    diff = pts_lidar[i] - cam2lidar_t
                    depth = np.linalg.norm(diff)

                    gt_2dbboxes.append(bboxes)
                    centers2d.append([center_u, center_v])
                    gt_2dlabels.append(info['gt_names'][i])
                    depths.append(depth)

                    # Mark this 3D object as having a valid 2D projection
                    valid_mask[i] = True

            gt_2dbboxes_cams.append(gt_2dbboxes)
            centers2d_cams.append(centers2d)
            gt_2dlabels_cams.append(gt_2dlabels)
            depths_cams.append(depths)

        # Filter 3D gt boxes and names to only keep those with valid 2D projections
        if filter_3d_gt and np.sum(valid_mask) < N:
            # print(
            #     f"\nFilter {N - np.sum(valid_mask)} 3D gt from sample {info['token']}, {np.sum(valid_mask)} left"
            # )
            if np.sum(valid_mask) == 0:
                print(f"No more valid 3D gt from sample {info['token']}")

            # Update the info with filtered 3D ground truth
            for key in [
                'gt_boxes',
                'gt_names',
                'gt_ins_tokens',
                'gt_inds',
                'anno_tokens',
                'valid_flag',
                'num_lidar_pts',
                'visibility_tokens',
                'gt_velocity',
                'prev_anno_tokens',
                'next_anno_tokens',
                'gt_veh_track_ids',
                'gt_inf_track_ids',
                'gt_veh_tokens',
                'gt_inf_tokens',
            ]:
                assert len(info[key]) == N
                info[key] = info[key][valid_mask]

        info.update(
            bboxes2d=[np.array(gt_2dbboxes) for gt_2dbboxes in gt_2dbboxes_cams],
            labels2d=gt_2dlabels_cams,
            centers2d=[np.array(centers2d) for centers2d in centers2d_cams],
            depths=[np.array(depths) for depths in depths_cams],
        )

    return modify_infos


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="2Dbox and Depth generator")
    parser.add_argument(
        "--info_path",
        type=str,
        default="./data/infos/V2X-Seq-SPD-2hz-for-SparseCoop",
    )
    parser.add_argument(
        "--side",
        type=str,
        choices=["vehicle-side", "infrastructure-side", "cooperative"],
    )
    parser.add_argument(
        "--filter_3d_gt",
        action="store_true",
        help="Filter 3D ground truth to only keep objects with valid 2D projections",
        default=False,
    )
    args = parser.parse_args()

    train_data = mmcv.load(
        os.path.join(args.info_path, args.side, "spd_infos_temporal_train.pkl")
    )
    val_data = mmcv.load(
        os.path.join(args.info_path, args.side, "spd_infos_temporal_val.pkl")
    )

    modify_train_infos = add_2D_and_Depth_infos(train_data['infos'], args.filter_3d_gt)
    modify_val_infos = add_2D_and_Depth_infos(val_data['infos'], args.filter_3d_gt)

    modify_train_data = dict(infos=modify_train_infos, metadata=train_data['metadata'])
    modify_val_data = dict(infos=modify_val_infos, metadata=val_data['metadata'])
    info_train_path = os.path.join(
        args.info_path,
        args.side,
        (
            "modify_infos_train.pkl"
            if not args.filter_3d_gt
            else "modify_infos_train_filter_3d_gt.pkl"
        ),
    )
    info_val_path = os.path.join(
        args.info_path,
        args.side,
        (
            "modify_infos_val.pkl"
            if not args.filter_3d_gt
            else "modify_infos_val_filter_3d_gt.pkl"
        ),
    )
    mmcv.dump(modify_train_data, info_train_path)
    mmcv.dump(modify_val_data, info_val_path)
