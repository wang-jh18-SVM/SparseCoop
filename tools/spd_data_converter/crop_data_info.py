import mmcv
import random
input_dict = mmcv.load('data/infos/V2X-Seq-SPD-Batch-65-10-10761/vehicle-side/spd_stream_infos_temporal_train.pkl')
print(input_dict['metadata'])
infos = input_dict['infos']
save_path = 'data/infos/V2X-Seq-SPD-Batch-65-10-10761/vehicle-side/spd_small_infos_temporal_train.pkl'
scene_to_sample = {}
scene_list = []
for sample in infos:
    scene = sample['scene_token']
    sample_idx = sample['token']
    if scene not in scene_to_sample:
        scene_list.append(scene)
        scene_to_sample[scene] = []
    scene_to_sample[scene].append(sample_idx)
# print(scene_to_sample)

# selected_scene_list = random.sample(scene_list, 1)
selected_scene_list = ['0001', '0005']
selected_sample_list = []
for scene in selected_scene_list:
    print(len(scene_to_sample[scene]))
    selected_sample_list.extend(scene_to_sample[scene])

selected_infos = []
for sample in infos:
    if sample['token'] in selected_sample_list:
        selected_infos.append(sample)

print(len(infos))
print(len(selected_infos))
metadata = dict(version='v1.0-trainval')
save_dict = dict(infos=selected_infos, metadata=metadata)
mmcv.dump(save_dict, save_path)