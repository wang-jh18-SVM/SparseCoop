export PYTHONPATH=$PYTHONPATH:./
export CUDA_VISIBLE_DEVICES=1

config=$1
checkpoint=$2

echo "config file: "${config}
echo "checkpoint: "${checkpoint}

# Extract directory structure from checkpoint path
checkpoint_dir=$(dirname ${checkpoint})
if [[ ${checkpoint_dir} == work_dirs/* ]]; then
    work_dir="${checkpoint_dir}/"
else
    # Fallback to config-based work_dir
    work_dir_suffix=$(echo ${config} | sed 's|^projects/configs/||')
    work_dir="work_dirs/${work_dir_suffix%.*}/"
fi

if [ ! -d ${work_dir}logs ]; then
    mkdir -p ${work_dir}logs
fi

T=`date +%m%d%H%M`
checkpoint_name=$(basename ${checkpoint} .pth)
config_name=$(basename ${config} .py)
log_file="${work_dir}logs/test_${T}_${checkpoint_name}_${config_name}.log"
echo "log file: ${log_file}"

# Create log file with config info header
echo "===== Testing Log =====" > ${log_file}
echo "Config file: ${config}" >> ${log_file}
echo "Checkpoint: ${checkpoint}" >> ${log_file}
echo "Timestamp: $(date)" >> ${log_file}
echo "Work directory: ${work_dir}" >> ${log_file}
echo "========================" >> ${log_file}
echo "" >> ${log_file}

python ./tools/test.py \
    ${config} \
    ${checkpoint} \
    --eval bbox \
    --out \
    2>&1 | tee -a ${log_file}

echo "===== Testing Done =====" >> ${log_file}
echo "Timestamp: $(date)" >> ${log_file}
echo "=========================" >> ${log_file}
echo "" >> ${log_file}

echo "Find Log file at: ${log_file}"
