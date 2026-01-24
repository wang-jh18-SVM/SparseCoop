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

    x_corners = np.array([0.5,  0.5, -0.5, -0.5,  0.5,  0.5, -0.5, -0.5])
    y_corners = np.array([0.5, -0.5, -0.5,  0.5,  0.5, -0.5, -0.5,  0.5])
    z_corners = np.array([-0.5, -0.5, -0.5, -0.5,  0.5,  0.5,  0.5,  0.5])

    corners_3d = np.zeros((N, 8, 3))
    for i in range(N):
        x, y, z, l, w, h, yaw = boxes[i]

        corner = np.stack([
            l * x_corners,
            w * y_corners,
            h * z_corners
        ], axis=1)  # shape (8, 3)

        rot_mat = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw),  np.cos(yaw), 0],
            [0,            0,           1]
        ])

        corners_3d[i] = np.dot(corner, rot_mat.T) + np.array([x, y, z])

    return corners_3d  # shape (N, 8, 3)


def add_2D_and_Depth_infos(nusc_infos):

    modify_infos = copy.deepcopy(nusc_infos)

    for info in mmcv.track_iter_progress(modify_infos):

        corners_3d = boxes_to_corners_3d_lidar(info['gt_boxes'])  #lidar cooridnate
        N = corners_3d.shape[0]
        pts_lidar = info['gt_boxes'][:, :3]

        gt_2dbboxes_cams = []
        centers2d_cams = []
        gt_2dlabels_cams = []
        depths_cams = []

        for cam, cam_info in info['cams'].items():

            gt_2dbboxes = []
            centers2d = []
            gt_2dlabels = []
            depths = []

            rotation_matrix = Quaternion(cam_info['lidar2cam_rotation']).rotation_matrix
            lidar2cam = np.eye(4)
            lidar2cam[:3, :3] = rotation_matrix
            lidar2cam[:3, 3] = cam_info['lidar2cam_translation']
            cam2lidar = np.linalg.inv(lidar2cam)
            cam_in_lidar = cam2lidar[:3, 3]
            intrinsic = np.array(cam_info['cam_intrinsic'])
            # img = cv2.imread(cam_info['data_path'])
            # H, W = img.shape[:2]
            H, W = 1080, 1920

            for i in range(N):

                pts = corners_3d[i]
                pts_h = np.concatenate([pts, np.ones((8,1))], axis=1)     #homogeneous coordinate
                pts_cam = (lidar2cam @ pts_h.T).T[:, :3]
                cam_x, cam_y, cam_z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]

                u = (intrinsic[0, 0] * cam_x + intrinsic[0, 2] * cam_z) / cam_z
                v = (intrinsic[1, 1] * cam_y + intrinsic[1, 2] * cam_z) / cam_z
                uv = np.stack([u, v], axis=-1) 
                valid_pts = (cam_z > 0) & (u >= 0) & (v >= 0) & (u < W) & (v < H)
                if valid_pts.sum() >= 6:
                    u_min = max(0,np.min(u))
                    v_min = max(0,np.min(v))
                    u_max = min(W,np.max(u))
                    v_max = min(H,np.max(v))
                    center_u = (u_max - u_min)/2 + u_min
                    center_v = (v_max - v_min)/2 + v_min
                    bboxes = [u_min, v_min, u_max, v_max]
                    diff = pts_lidar[i] - cam_in_lidar
                    depth = np.linalg.norm(diff)

                    gt_2dbboxes.append(bboxes)
                    centers2d.append([center_u,center_v])
                    gt_2dlabels.append(info['gt_names'][i])
                    depths.append(depth)

            gt_2dbboxes_cams.append(gt_2dbboxes)
            centers2d_cams.append(centers2d)
            gt_2dlabels_cams.append(gt_2dlabels)
            depths_cams.append(depths)

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
        "--root_path",
        type=str,
        default="./data/infos/griffin_50scenes_25m/vehicle-side",
    )
    parser.add_argument(
        "--out_path", type=str, default="./data/infos/griffin_50scenes_25m/vehicle-side"
    )
    parser.add_argument("--info_prefix", type=str, default="griffin")
    args = parser.parse_args()

    train_data = mmcv.load(os.path.join(args.root_path,"griffin_infos_train.pkl"))
    val_data = mmcv.load(os.path.join(args.root_path,"griffin_infos_val.pkl"))

    modify_train_infos = add_2D_and_Depth_infos(train_data['infos'])
    modify_val_infos = add_2D_and_Depth_infos(val_data['infos'])

    modify_train_data = dict(infos=modify_train_infos, metadata=train_data['metadata'])
    modify_val_data = dict(infos=modify_val_infos, metadata=val_data['metadata'])
    info_train_path = os.path.join(args.out_path,"modify_griffin_infos_train.pkl")
    info_val_path = os.path.join(args.out_path,"modify_griffin_infos_val.pkl")
    mmcv.dump(modify_train_data, info_train_path)
    mmcv.dump(modify_val_data, info_val_path)
