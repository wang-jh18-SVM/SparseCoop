# Dataset Format Conversion: From SPD To nuScenes

## Function 
Convert the SPD dataset into the nuScenes format.

## Dataset Conversion and Generation

Using the generation of a 2Hz V2X-Seq-SPD-Example dataset as an example:

### Generate `datasets/V2X-Seq-SPD-Example`

#### Run

```bash
cd ${UniV2X_ROOT}
python tools/spd_data_converter/gen_example_data.py \
    --input {Path_to_V2X-Seq-SPD} \
    --output ./datasets/V2X-Seq-SPD-Example \
    --sequences 0010 0016 0018 0022 0023 0025 0029 0030 0032 0033 0034 0035 0014 0015 0017 0020 0021 \
    --update-label \
    --freq 2
```

### Generate `data/infos/V2X-Seq-SPD-Example`

**Method 1:**

#### Run

```bash
cd ${UniV2X_ROOT}
sh tools/spd_example_converter.sh V2X-Seq-SPD-Example vehicle-side
sh tools/spd_example_converter.sh V2X-Seq-SPD-Example infrastructure-side
sh tools/spd_example_converter.sh V2X-Seq-SPD-Example cooperative
```

**Method 2:**

#### Run

```bash
cd ${UniV2X_ROOT}

# vehicle-side
# Convert SPD to UniAD
python tools/spd_data_converter/spd_to_uniad.py \
    --data-root './datasets/V2X-Seq-SPD-Example' \
    --save-root './datasets/V2X-Seq-SPD-Example' \
    --v2x-side 'vehicle-side'
# Convert SPD to nuScenes
python tools/spd_data_converter/spd_to_nuscenes.py \
    --data-root './datasets/V2X-Seq-SPD-Example' \
    --save-root './datasets/V2X-Seq-SPD-Example' \
    --v2x-side 'vehicle-side'
# Map from SPD to nuScenes
python tools/spd_data_converter/map_spd_to_nuscenes.py
    --save-root './datasets/V2X-Seq-SPD-Example'
    --v2x-side 'vehicle-side`

# infrastructure-side
# Convert SPD to UniAD
python tools/spd_data_converter/spd_to_uniad.py \
    --data-root './datasets/V2X-Seq-SPD-Example' \
    --save-root './datasets/V2X-Seq-SPD-Example' \
    --v2x-side 'infrastructure-side'
# Convert SPD to nuScenes
python tools/spd_data_converter/spd_to_nuscenes.py \
    --data-root './datasets/V2X-Seq-SPD-Example' \
    --save-root './datasets/V2X-Seq-SPD-Example' \
    --v2x-side 'infrastructure-side'
# Map from SPD to nuScenes
python tools/spd_data_converter/map_spd_to_nuscenes.py
    --save-root './datasets/V2X-Seq-SPD-Example'
    --v2x-side 'infrastructure-side`

# cooperative
# Convert SPD to UniAD
python tools/spd_data_converter/spd_to_uniad.py \
    --data-root './datasets/V2X-Seq-SPD-Example' \
    --save-root './datasets/V2X-Seq-SPD-Example' \
    --v2x-side 'cooperative'
# Convert SPD to nuScenes
python tools/spd_data_converter/spd_to_nuscenes.py \
    --data-root './datasets/V2X-Seq-SPD-Example' \
    --save-root './datasets/V2X-Seq-SPD-Example' \
    --v2x-side 'cooperative'
# Map from SPD to nuScenes
python tools/spd_data_converter/map_spd_to_nuscenes.py
    --save-root './datasets/V2X-Seq-SPD-Example'
    --v2x-side 'cooperative`
```

### Generate Annotations for SPD
#### Run
```bash
python spd_generate_annotation.py
```

#### Description
* To better describe temporal information, the dataset needs to construct a data structure based on objects. 
* This data structure is a dictionary where each key is a globally unique token corresponding to a bounding box (bbox) in the original label file. The value is a dictionary that records the information corresponding to that bbox.
* Each bbox dictionary has a next field to record the globally unique token of the bbox corresponding to this object in the next frame's collaborative annotation.
* Thus, by iterating over the next tokens, you can sequentially access the temporal annotation information of the same object, forming a linked-list-like data structure.

#### Structure
* Original label example:

```json
{
    "token": "46870de4-7673-3fee-acbc-ab4b35f70834",
    "type": "Car",
    "track_id": "002706",
    "truncated_state": 0,
    "occluded_state": 2,
    "alpha": -1.364216,
    "2d_box": {"xmin": 865.511414, "ymin": 567.811646, "xmax": 928.400513, "ymax": 615.924256},
    "3d_dimensions": {"l": 4.251301, "w": 1.8234, "h": 1.570357},
    "3d_location": {"x": 27.07514, "y": 4.712578, "z": -0.922006},
    "rotation": -0.034323,
    "from_side": "veh",
    "veh_pointcloud_timestamp": "1626155123881356",
    "inf_pointcloud_timestamp": "1626155123944230",
    "veh_frame_id": "000009",
    "inf_frame_id": "000000",
    "veh_track_id": "009525",
    "inf_track_id": "-1",
    "veh_token": "46870de4-7673-3fee-acbc-ab4b35f70834",
    "inf_token": "-1"
}
```
* Describing the data structure using the above label as an example:

```json
{
    "46870de4-7673-3fee-acbc-ab4b35f70834": {
        # Original label information
        "type": "Car",
        "track_id": "002706",
        "truncated_state": 0,
        "occluded_state": 2,
        "alpha": -1.364216,
        "2d_box": {"xmin": 865.511414, "ymin": 567.811646, "xmax": 928.400513, "ymax": 615.924256},
        "3d_dimensions": {"l": 4.251301, "w": 1.8234, "h": 1.570357},
        "3d_location": {"x": 27.07514, "y": 4.712578, "z": -0.922006},
        "rotation": -0.034323,
        "from_side": "veh",
        "veh_pointcloud_timestamp": "1626155123881356",
        "inf_pointcloud_timestamp": "1626155123944230",
        "veh_frame_id": "000009",
        "inf_frame_id": "000000",
        "veh_track_id": "009525",
        "inf_track_id": "-1",
        "veh_token": "46870de4-7673-3fee-acbc-ab4b35f70834",
        "inf_token": "-1",
        # Transformation matrix from the label coordinate system to the global coordinate system, obtained from the original data calibration
        "calib_e2g": {"rotation": "xxx", "translation": "xxx"},
        # Temporal information of the object. If there is a next frame, the value is the globally unique token of the corresponding bbox in the next frame's collaborative annotation. If there is no next frame, indicating the current frame is the last in the sequence, the value is ''.
        "next": "xxx"
    }
}
```