import mmcv
import numpy as np
from tqdm import tqdm
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.core.bbox.structures.box_3d_mode import Box3DMode
CLASSES = [
    "car",
    "bicycle",
    "pedestrian",
]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

def get_bboxes(info):
    mask = info['valid_flag']
    gt_bboxes_3d = info['gt_boxes'][mask]
    gt_names_3d = info['gt_names'][mask]
    gt_inds = info['gt_inds'][mask]

    gt_labels_3d = []
    for cat in gt_names_3d:
        if cat in CLASSES:
            gt_labels_3d.append(CLASSES.index(cat))
        else:
            gt_labels_3d.append(-1)
    gt_labels_3d = np.array(gt_labels_3d)

    gt_velocity = info['gt_velocity'][mask]
    nan_mask = np.isnan(gt_velocity[:, 0])
    gt_velocity[nan_mask] = [0.0, 0.0]
    gt_bboxes_3d = np.concatenate([gt_bboxes_3d, gt_velocity], axis=-1)

    # the nuscenes box center is [0.5, 0.5, 0.5], we change it to be
    # the same as KITTI (0.5, 0.5, 0)
    gt_bboxes_3d = LiDARInstance3DBoxes(
        gt_bboxes_3d,
        box_dim=gt_bboxes_3d.shape[-1],
        origin=(0.5, 0.5, 0.5)).convert_to(Box3DMode.LIDAR)
    anns_results = dict(
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            gt_names=gt_names_3d,
            gt_inds=gt_inds,
        )
    return anns_results

def filter_bboxes(bboxes_dict, point_cloud_range):
    # range filter
    pcd_range = np.array(point_cloud_range, dtype=np.float32)
    bev_range = pcd_range[[0, 1, 3, 4]]
    gt_bboxes_3d = bboxes_dict['gt_bboxes_3d']
    gt_labels_3d = bboxes_dict['gt_labels_3d']
    gt_inds = bboxes_dict['gt_inds']
    mask = gt_bboxes_3d.in_range_bev(bev_range)
    gt_bboxes_3d = gt_bboxes_3d[mask]
    mask = mask.numpy().astype(np.bool)
    gt_labels_3d = gt_labels_3d[mask]
    gt_inds = gt_inds[mask]
    gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)

    # name filter
    labels = list(range(len(CLASSES)))
    gt_bboxes_mask = np.array([n in labels for n in gt_labels_3d],
                                  dtype=np.bool_)
    gt_bboxes_3d = gt_bboxes_3d[gt_bboxes_mask]
    gt_labels_3d = gt_labels_3d[gt_bboxes_mask]
    gt_inds = gt_inds[gt_bboxes_mask]

    return gt_bboxes_3d

input_dict = mmcv.load('data/infos/V2X-Seq-SPD-Batch-65-10-10761-forecasting/cooperative/spd_infos_temporal_train.pkl')
print(input_dict['metadata'])
infos = input_dict['infos']
print(len(infos))
import pdb;pdb.set_trace()
invalid_frames = []
invalid_scenes = []
for f in tqdm(infos):
    bboxes = get_bboxes(f)
    bboxes_in_range = filter_bboxes(bboxes, point_cloud_range)
    print(len(bboxes_in_range))
    if len(bboxes_in_range) == 0:
        invalid_frames.append(f['token'])
        if f['scene_token'] not in invalid_scenes:
            invalid_scenes.append(f['scene_token'])
print(invalid_frames)
print(invalid_scenes)

scene_to_sample = {}
for sample in infos:
    scene = sample['scene_token']
    sample_idx = sample['token']
    if scene not in scene_to_sample:
        scene_to_sample[scene] = []
    scene_to_sample[scene].append(sample_idx)
# print(scene_to_sample)

for scene in invalid_scenes:
    print(scene_to_sample[scene])

# coop need to remove
['003211', '003212', '003213', '003214', '003215', '003216', '003217', '003218', '003219', '003220', '003221', '003222', '003223', '003224', '003225', '003226', '003227', '003228', '003229', '003230', 
 '004394', '004395', '004396', '004397', '004398', '004399', '004400', '004401', '004402', '004403', '004404', '004405', '004406', '004407', '004408', '004409', '004410', '004411', '004412', '004413', 
 '007457', '007458', '007459', '007460', '007461', '007462', '007463', '007464', '007465', '007466', '007467', '007468', '007469', '007470', '007471', '007472', '007473', '007474', '007475', '007476', '007477', '007478', '007479', '007480', '007481', '007482', 
 '013006', '013083', '013084', '013085', '013086', '013087', '013088', '013089', '013090', '013091', '013092', '013093', '013094', '013206', '013207', '013208', '013209']