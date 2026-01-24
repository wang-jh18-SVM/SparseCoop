import json
import os
import matplotlib.pyplot as plt

def load_json(path):
    with open(path, mode="r") as f:
        data = json.load(f)

    return data

data_root = 'datasets/V2X-Seq-SPD-Batch-65-10-10761'
v2x_side = 'infrastructure-side'
root_path = os.path.join(data_root, v2x_side)
data_info_path = os.path.join(root_path, 'data_info.json')
split_path = 'data/split_datas/cooperative-split-data-spd.json'

data_infos = load_json(data_info_path)
split_data = load_json(split_path)
if v2x_side in ['vehicle-side', 'infrastructure-side']:
    train_scenes = split_data['batch_split']['train']
    val_scenes = split_data['batch_split']['val']
else:
    train_scenes = split_data['cooperative_split']['train']
    val_scenes = split_data['cooperative_split']['val']

train_infos = {
    'Car': 0,
    'Truck': 0,
    'Van': 0,
    'Bus': 0,
    'Pedestrian': 0,
    'Cyclist': 0,
    'Tricyclist': 0,
    'Motorcyclist': 0,
    'Barrowlist': 0,
    'Trafficcone': 0,
}
val_infos = {
    'Car': 0,
    'Truck': 0,
    'Van': 0,
    'Bus': 0,
    'Pedestrian': 0,
    'Cyclist': 0,
    'Tricyclist': 0,
    'Motorcyclist': 0,
    'Barrowlist': 0,
    'Trafficcone': 0,
}

for item in data_infos:
    if v2x_side in ['vehicle-side', 'infrastructure-side']:
        anno_path = os.path.join(root_path, item['label_lidar_std_path'])
        seq_id = item['sequence_id']
    else:
        sample_token = item['vehicle_frame']
        anno_path = os.path.join(root_path, 'label', sample_token+'.json')
        seq_id = sample_token
    annos = load_json(anno_path)
    
    for b in annos:
        if seq_id in train_scenes:
            train_infos[b['type']] += 1
        else:
            val_infos[b['type']] += 1


print(train_infos)
print(val_infos)
keys = list(train_infos.keys())
values1 = list(train_infos.values())
values2 = list(val_infos.values())

fig, ax = plt.subplots(figsize=(10, 6))
bar_width = 0.35
index = range(len(keys))

# Plotting each dictionary
bar1 = ax.bar(index, values1, bar_width, label='train-set')
bar2 = ax.bar([i + bar_width for i in index], values2, bar_width, label='val-set')

# Adding labels and title
ax.set_xlabel('Categories')
ax.set_ylabel('Number')
ax.set_title('Statistics of Number across different categories')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(keys)
ax.legend()

# Rotating the x-axis labels for better readability
plt.xticks(rotation=45)

# Saving the plot
plt.savefig(f"workspace/statistic_class/{v2x_side}.png")
