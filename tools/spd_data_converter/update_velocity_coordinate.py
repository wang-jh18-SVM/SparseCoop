import pickle
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
import numpy as np
import os
import argparse

# Original velocity from info is from a dynamic coordinate fixed to lidar
# We transform it to a static coordinate fixed to ground, but overlapped with lidar coordinate in certain timestamp

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data-root",
    type=str,
    default="datasets/V2X-Seq-SPD-2hz-for-SparseCoop/vehicle-side",
)
parser.add_argument(
    "--info-root",
    type=str,
    default="data/infos/V2X-Seq-SPD-2hz-for-SparseCoop/vehicle-side",
)
args = parser.parse_args()

spd_nusc = NuScenes(version='v1.0-trainval', dataroot=args.data_root, verbose=True)
pkl_files = [f for f in os.listdir(args.info_root) if f.endswith('.pkl')]
for pkl_file in pkl_files:
    pkl_path = os.path.join(args.info_root, pkl_file)
    with open(pkl_path, 'rb') as f:
        spd_infos = pickle.load(f)

    for info in spd_infos['infos']:
        anno_tokens = info['anno_tokens']
        info_velocity = info['gt_velocity']
        nusc_velocity = np.array(
            [spd_nusc.box_velocity(anno_token) for anno_token in anno_tokens]
        )
        nusc_velocity[np.isnan(nusc_velocity)] = 0.0
        assert info_velocity.shape == nusc_velocity.shape

        # Convert velocity from global to lidar coordinate
        l2e_r_mat = Quaternion(info["lidar2ego_rotation"]).rotation_matrix
        e2g_r_mat = Quaternion(info["ego2global_rotation"]).rotation_matrix
        g2l_r_mat = np.linalg.inv(e2g_r_mat @ l2e_r_mat)
        nusc_velocity_to_lidar = (g2l_r_mat @ nusc_velocity.T).T

        info['gt_velocity'] = nusc_velocity_to_lidar

    # Save new pkl
    with open(pkl_path, 'wb') as f:
        pickle.dump(spd_infos, f)
