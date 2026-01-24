import os
import cv2
import mmcv
import numpy as np
import argparse
from pyquaternion import Quaternion

def project_3d_box_to_image(corners_3d, lidar2cam, intrinsic):
    corners_3d_h = np.concatenate([corners_3d, np.ones((8, 1))], axis=1)
    pts_cam = (lidar2cam @ corners_3d_h.T).T[:, :3]
    x, y, z = pts_cam[:, 0], pts_cam[:, 1], pts_cam[:, 2]
    u = (intrinsic[0, 0] * x + intrinsic[0, 2] * z) / z
    v = (intrinsic[1, 1] * y + intrinsic[1, 2] * z) / z
    return np.stack([u, v], axis=1), z

def boxes_to_corners_3d_lidar(box):
    x, y, z, l, w, h, yaw = box
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
    y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
    z_corners = [-h/2]*4 + [h/2]*4
    corners = np.stack([x_corners, y_corners, z_corners], axis=1)
    R = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0, 0, 1]
    ])
    return (R @ corners.T).T + np.array([x, y, z])

def visualize_frame(info, root_path, output_dir="output_vis"):
    cnt_cam = 0 #map 2D information to camera
    for cam_name in info['cams'].keys():
        cam = info['cams'][cam_name]
        img_path = os.path.join(root_path, cam['data_path'])
        img = cv2.imread(img_path)
        if img is None:
            print(f"Fail to access the img: {img_path}")
            return

        intrinsic = np.array(cam['cam_intrinsic'])
        rotation = Quaternion(cam['lidar2cam_rotation']).rotation_matrix
        translation = np.array(cam['lidar2cam_translation'])
        lidar2cam = np.eye(4)
        lidar2cam[:3, :3] = rotation
        lidar2cam[:3, 3] = translation

        for box in info['gt_boxes']:
            corners_3d = boxes_to_corners_3d_lidar(box)
            uv, depth = project_3d_box_to_image(corners_3d, lidar2cam, intrinsic)
            if (depth > 0).sum() >= 6:
                for i in range(4):
                    pt1, pt2 = tuple(uv[i].astype(int)), tuple(uv[(i+1)%4].astype(int))
                    cv2.line(img, pt1, pt2, (0, 255, 0), 2)
                    pt1, pt2 = tuple(uv[i+4].astype(int)), tuple(uv[(i+1)%4+4].astype(int))
                    cv2.line(img, pt1, pt2, (0, 255, 0), 2)
                    cv2.line(img, tuple(uv[i].astype(int)), tuple(uv[i+4].astype(int)), (0, 255, 0), 2)

        if 'bboxes2d' in info and 'depths' in info:
            for i, bbox in enumerate(info['bboxes2d'][cnt_cam]):
                x1, y1, x2, y2 = map(int, bbox)
                depth = info['depths'][cnt_cam][i]
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, f"{depth:.2f}m", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)

        # Save img
        frame_name = os.path.splitext(os.path.basename(cam['data_path']))[0]
        save_path = os.path.join(output_dir, f"{frame_name}_{cam_name}.jpg")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)
        print(f"Saved to {save_path}")
        cnt_cam+=1

def main():
    parser = argparse.ArgumentParser(
        description="Visualize 3D bounding boxes projected to 2D images from Griffin dataset"
    )
    parser.add_argument(
        "--pkl_path",
        type=str,
        default="./data/infos/griffin_50scenes_25m/vehicle-side/modify_griffin_infos_train.pkl",
        help="Path to the pickle file containing dataset information",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./datasets/griffin_50scenes_25m/griffin-nuscenes/vehicle-side",
        help="Path to the nuscenes-formatted dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./result_vis/griffin_50scenes_25m/vehicle-side/pkl_vis",
        help="Directory to save visualization results",
    )
    parser.add_argument(
        "--num_frames",
        type=int,
        default=2,
        help="Number of frames to visualize (use 0 for all frames)",
    )
    args = parser.parse_args()

    data = mmcv.load(args.pkl_path)
    visual_frame_num = (
        min(args.num_frames, len(data['infos']))
        if args.num_frames > 0
        else len(data['infos'])
    )
    for i in range(visual_frame_num):
        visualize_frame(data['infos'][i], args.dataset_path, args.output_dir)
        print(f"Processed frame {i+1}/{visual_frame_num}")

if __name__ == "__main__":
    main()
