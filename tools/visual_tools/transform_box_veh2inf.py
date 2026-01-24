import mmcv
import json
import os.path as osp
import numpy as np
from scipy.linalg import polar
import argparse
from pyquaternion import Quaternion

def iterative_closest_point(A, num_iterations=100):
    R = A.copy()

    for _ in range(num_iterations):
        U, _ = polar(R)
        R = U

    return R

def read_json(path_json):
    with open(path_json, 'r') as load_f:
        data_json = json.load(load_f)
    return data_json

def veh2inf_convert(boxes, spd_data_root, veh2inf, token):
    veh_id = token
    inf_id = veh2inf[veh_id]

    inf_virtuallidar2world_path = osp.join(spd_data_root, 'infrastructure-side/calib/virtuallidar_to_world', inf_id+'.json')
    inf_virtuallidar2world = read_json(inf_virtuallidar2world_path)
    inf_virtuallidar2world_rotation = inf_virtuallidar2world['rotation']
    inf_virtuallidar2world_translation = inf_virtuallidar2world['translation']

    veh_ego2world_path = osp.join(spd_data_root, 'vehicle-side/calib/novatel_to_world', veh_id+'.json')
    veh_ego2world = read_json(veh_ego2world_path)
    veh_ego2world_rotation = veh_ego2world['rotation']
    veh_ego2world_translation = veh_ego2world['translation']

    veh_lidar2ego_path = osp.join(spd_data_root, 'vehicle-side/calib/lidar_to_novatel', veh_id+'.json')
    veh_lidar2ego = read_json(veh_lidar2ego_path)
    veh_lidar2ego_rotation = veh_lidar2ego['transform']['rotation']
    veh_lidar2ego_translation = veh_lidar2ego['transform']['translation']

    err_offset = veh2inf[veh_id+'offset']

    veh_l2e_r = np.array(veh_lidar2ego_rotation)
    veh_l2e_t = np.array(veh_lidar2ego_translation).reshape(3)
    veh_e2g_r = np.array(veh_ego2world_rotation)
    veh_e2g_t = np.array(veh_ego2world_translation).reshape(3)

    inf_e2g_r = np.array(inf_virtuallidar2world_rotation)
    inf_e2g_t = np.array(inf_virtuallidar2world_translation).reshape(3)
    inf_l2e_r = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    inf_l2e_t = np.array([0, 0, 0]).reshape(3)

    err_offset = np.array([err_offset['delta_x'], err_offset['delta_y'], 0])
    t = -err_offset @ veh_l2e_r.T @ veh_e2g_r.T
    
    # r = ((veh_l2e_r.T @ veh_e2g_r.T) @ (np.linalg.inv(inf_e2g_r).T @ np.linalg.inv(inf_l2e_r).T)).T
    # t = (-err_offset @ veh_l2e_r.T @ veh_e2g_r.T + veh_l2e_t @ veh_e2g_r.T + veh_e2g_t) @ (np.linalg.inv(inf_e2g_r).T @ np.linalg.inv(inf_l2e_r).T)
    # t -= inf_e2g_t @ (np.linalg.inv(inf_e2g_r).T @ np.linalg.inv(inf_l2e_r).T) +\
    #         inf_l2e_t @ (np.linalg.inv(inf_l2e_r).T)
    # vehlidar2inflidar_rotation, vehlidar2inflidar_translation = r, t
    # appro_vehlidar2inflidar = iterative_closest_point(vehlidar2inflidar_rotation)
    # vehlidar2inflidar_rotation = Quaternion(matrix=appro_vehlidar2inflidar.T)
    
    # inflidar2vehlidar_rotation = np.linalg.inv(appro_vehlidar2inflidar)
    # inflidar2vehlidar_translation = - np.dot(inflidar2vehlidar_rotation, vehlidar2inflidar_translation)
    box_list = []
    for box in boxes:
        # box.rotate(vehlidar2inflidar_rotation)
        box.translate(t)
        box_list.append(box)

    return box_list


# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--result_path', default='./output/1001_stage1_inf_0_100.pkl', help='path to results.pkl')
#     parser.add_argument('--data_root', default='datasets/V2X-Seq-SPD-Batch-95-2-3134-rm-ego-car-new', help='data root')
#     parser.add_argument('--output_result_path', default='./output/1001_stage1_inf_0_100_inf2veh.pkl', help='path to converted results.pkl')
#     args = parser.parse_args()

#     result_path = args.result_path
#     spd_data_root = args.data_root
#     pair_info_file = osp.join(spd_data_root, 'cooperative/data_info.json')
#     output_result_path = args.output_result_path

#     inf2veh_convert(result_path, spd_data_root, pair_info_file, output_result_path)
