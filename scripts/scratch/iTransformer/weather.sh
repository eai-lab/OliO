#!/bin/bash

# Auto-detect available GPUs
TARGET_GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' '))
JOBS_PER_GPU=1
MAX_JOBS=6

if [ -e "./venv/bin/activate" ]; then
    source ./venv/bin/activate
fi

export GPU_LIST_STR="${TARGET_GPUS[*]}"
NUM_GPUS=${#TARGET_GPUS[@]}

# Calculate actual TOTAL_JOBS as min(JOBS_PER_GPU * NUM_GPUS, MAX_JOBS)
TOTAL_JOBS=$((JOBS_PER_GPU * NUM_GPUS))
if [ $TOTAL_JOBS -gt $MAX_JOBS ]; then
    TOTAL_JOBS=$MAX_JOBS
fi

echo "Detected ${NUM_GPUS} GPUs: ${TARGET_GPUS[*]}"
echo "Running ${TOTAL_JOBS} parallel jobs (max: ${MAX_JOBS})"

LOG_DIR=logs/iTransformer/scratch/weather
mkdir -p $LOG_DIR
export EXTRA_ARGS="$@"


parallel  --line-buffer --jobs $TOTAL_JOBS --bar \
    --delay 5 \
    --results $LOG_DIR/scratch/seed-{1}/pred-{2} \
'
    # --------------------------------------------------------
    # [GPU 매핑 로직]
    # --------------------------------------------------------
    gpu_array=($GPU_LIST_STR)
    num_gpus=${#gpu_array[@]}
    slot_idx=$(( {%} - 1 ))
    array_idx=$(( slot_idx % num_gpus ))
    real_gpu_id=${gpu_array[$array_idx]}

    export CUDA_VISIBLE_DEVICES=$real_gpu_id
    
    random_seed={1}
    pred_len={2}

    echo "--------------------------------------------------------"
    echo " Data         : Weather"
    echo " Pred Len     : $pred_len"
    echo " Seq Len      : 96"
    echo " Random Seed  : $random_seed"
    echo "--------------------------------------------------------"

    python run_finetune.py \
        --model iTransformer \
        --data Weather \
        --root_path ./dataset \
        --data_path weather.csv \
        --features M \
        --seq_len 96 \
        --pred_len $pred_len \
        --enc_in 21 \
        --e_layers 3 \
        --n_heads 8 \
        --d_model 512 \
        --d_ff 512 \
        --dropout 0.1 \
        --des scratch \
        --is_scratch \
        --train_epochs 10 \
        --batch_size 32 \
        --learning_rate 0.0001 \
        --lradj type1 \
        --patience 3 \
        --random_seed $random_seed \
        '"$@"'
' ::: 2025 2024 2023 ::: 96 192 336 720