dataset_name=$1
v2x_side=$2
export PYTHONPATH=$PYTHONPATH:./

echo "Converting ${v2x_side} of ${dataset_name} to pkl infos..."
python tools/spd_data_converter/spd_to_uniad.py \
    --data-root ./datasets/${dataset_name} \
    --save-root ./data/infos/${dataset_name} \
    --v2x-side ${v2x_side} \
    # --forecasting 

echo "Converting ${v2x_side} of ${dataset_name} to nuscenes format..."
python tools/spd_data_converter/spd_to_nuscenes.py \
    --data-root ./datasets/${dataset_name} \
    --save-root ./datasets/${dataset_name} \
    --v2x-side ${v2x_side}

echo "Updating velocity coordinate of ${v2x_side} of ${dataset_name}..."
python tools/spd_data_converter/update_velocity_coordinate.py \
    --data-root ./datasets/${dataset_name}/${v2x_side} \
    --info-root ./data/infos/${dataset_name}/${v2x_side}

echo "Generating anchors for ${v2x_side} of ${dataset_name}..."
if [ "${v2x_side}" == "infrastructure-side" ]; then
    # infrastructure-side has double detection range
    python tools/anchor_generator.py --ann_file data/infos/${dataset_name}/${v2x_side}/spd_infos_temporal_train.pkl --detection_range 110
else
    python tools/anchor_generator.py --ann_file data/infos/${dataset_name}/${v2x_side}/spd_infos_temporal_train.pkl
fi

# echo "Converting 2D bboxes and depth GT of ${v2x_side} of ${dataset_name}..."
# if [ "${v2x_side}" == "cooperative" ]; then
#     python tools/spd_data_converter/bboxes2D_and_depth.py --info_path ./data/infos/${dataset_name} --side ${v2x_side} --filter_3d_gt
# else
#     python tools/spd_data_converter/bboxes2D_and_depth.py --info_path ./data/infos/${dataset_name} --side ${v2x_side}
# fi