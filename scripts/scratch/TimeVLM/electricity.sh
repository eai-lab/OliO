#!/bin/bash

SCRIPT_PATH=$(realpath "$0")
SUB_PATH=${SCRIPT_PATH#*scripts/}

# Auto-detect available GPUs
TARGET_GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' '))
JOBS_PER_GPU=2
if [ ${#TARGET_GPUS[@]} -gt 4 ]; then
    TARGET_GPUS=(0 1 2 3 5 6)
    MAX_JOBS=12
else
    MAX_JOBS=6
fi

if [ -e "./venv/bin/activate" ]; then
    source ./venv/bin/activate
fi
echo "Detected ${NUM_GPUS} GPUs: ${TARGET_GPUS[*]}"
echo "Running ${TOTAL_JOBS} parallel jobs (max: ${MAX_JOBS})"

LOG_DIR=logs/$SUB_PATH
mkdir -p $LOG_DIR

export EXTRA_ARGS="$@"
export GPU_LIST_STR="${TARGET_GPUS[*]}"
NUM_GPUS=${#TARGET_GPUS[@]}

# Calculate actual TOTAL_JOBS as min(JOBS_PER_GPU * NUM_GPUS, MAX_JOBS)
TOTAL_JOBS=$((JOBS_PER_GPU * NUM_GPUS))
if [ $TOTAL_JOBS -gt $MAX_JOBS ]; then
    TOTAL_JOBS=$MAX_JOBS
fi


parallel  --line-buffer --jobs $TOTAL_JOBS --bar \
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
    echo " Data         : Electricity"
    echo " Pred Len     : $pred_len"
    echo " Seq Len      : 512"
    echo " Random Seed  : $random_seed"
    echo "--------------------------------------------------------"

    python run_finetune.py \
        --model TimeVLM \
        --data Electricity \
        --root_path ./dataset \
        --data_path electricity.csv \
        --features M \
        --seq_len 512 \
        --pred_len $pred_len \
        --enc_in 321 \
        --e_layers 2 \
        --n_heads 8 \
        --d_model 128 \
        --dropout 0.1 \
        --patch_len 16 \
        --stride 8 \
        --des scratch \
        --is_scratch \
        --train_epochs 20 \
        --batch_size 16 \
        --learning_rate 0.001 \
        --lradj type1 \
        --patience 5 \
        --random_seed $random_seed \
        --image_size 56 \
        --norm_const 0.4 \
        --use_mem_gate \
        --periodicity 24 \
        --three_channel_image \
        --learnable_image \
        --use_cross_attention \
        '"$@"'
' ::: 2025 2024 2023 ::: 96 192 336 720