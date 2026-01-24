export PORT=28651
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export PYTHONPATH=$PYTHONPATH:./
# export TORCH_DISTRIBUTED_DEBUG=DETAIL

gpus=(${CUDA_VISIBLE_DEVICES//,/ })
gpu_num=${#gpus[@]}
echo "number of gpus: "${gpu_num}

config=$1

work_dir_suffix=$(echo ${config} | sed 's|^projects/configs/||')
work_dir="work_dirs/${work_dir_suffix%.*}/"
if [ ! -d ${work_dir}logs ]; then
    mkdir -p ${work_dir}logs
fi

T=`date +%m%d%H%M`
config_name=$(basename ${config} .py)
log_file="${work_dir}logs/train_${T}_${config_name}.log"
echo "log file: ${log_file}"

# Create log file with config info header
echo "===== Training Log =====" > ${log_file}
echo "Config file: ${config}" >> ${log_file}
echo "Timestamp: $(date)" >> ${log_file}
echo "Work directory: ${work_dir}" >> ${log_file}
echo "=========================" >> ${log_file}
echo "" >> ${log_file}

if [ ${gpu_num} -gt 1 ]
then
    bash ./tools/dist_train.sh \
        ${config} \
        ${gpu_num} \
        --work-dir=${work_dir} \
        2>&1 | tee -a ${log_file}
else
    python ./tools/train.py \
        ${config} \
        --work-dir=${work_dir} \
        2>&1 | tee -a ${log_file}
fi

echo "===== Training Done =====" >> ${log_file}
echo "Timestamp: $(date)" >> ${log_file}
echo "=========================" >> ${log_file}
echo "" >> ${log_file}

echo "Find Log file at: ${log_file}"
