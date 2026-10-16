#!/bin/bash

# venv 활성화 (User Rule)
if [ -e "./venv/bin/activate" ]; then
    source ./venv/bin/activate
fi

# 기본 설정
MODEL_NAME="iTransformer"
DATA_NAME="ETTh2"
DATA_PATH="ETTh2.csv"
ROOT_PATH="./dataset"

# 로그 디렉토리 설정
LOG_DIR="logs/finetune/$MODEL_NAME/etth2"
mkdir -p "$LOG_DIR"

# Pre-trained 모델 정보
PRETRAIN_EXP="iTransformer_ETTh2_sl672_pl12_tau8_128_sg-28"
EXP_NUM="28"

echo "=========================================================="
echo "[$DATA_NAME Finetune Start] Pretrain Exp: $PRETRAIN_EXP"
echo "=========================================================="

count=0

# 하이퍼파라미터 검색 (Finetuning)
for learning_rate in 0.001 0.0005 0.0001 0.00001; do
for head_dropout in 0.0 0.1 0.2 0.3; do
for random_seed in 2025 2024 2023; do
for pred_len in 96 192 336 720; do

    count=$((count + 1))
    
    # des 설정
    if [[ $learning_rate == "0.001" && $head_dropout == "0.0" ]]; then
        des="finetune-v1-triangular-causal-channel-individual-$EXP_NUM"
    else
        des="finetune-v1-triangular-causal-channel-individual-$EXP_NUM-lr$learning_rate-do$head_dropout"
    fi

    echo "----------------------------------------------------------"
    echo "[Task $count] pred_len:$pred_len, seed:$random_seed, lr:$learning_rate, dropout:$head_dropout"
    echo "----------------------------------------------------------"

    python run_finetune.py \
        --model "$MODEL_NAME" \
        --data "$DATA_NAME" \
        --root_path "$ROOT_PATH" \
        --data_path "$DATA_PATH" \
        --features M \
        --seq_len 96 \
        --pred_len "$pred_len" \
        --enc_in 7 \
        --e_layers 2 \
        --n_heads 8 \
        --d_model 256 \
        --d_ff 256 \
        --dropout "$head_dropout" \
        --des "$des" \
        --pretrain_exp "$PRETRAIN_EXP" \
        --train_epochs 10 \
        --batch_size 32 \
        --learning_rate "$learning_rate" \
        --lradj type1 \
        --patience 3 \
        --random_seed "$random_seed" \
        --pretrain_base_path ./results/pretrain_v1-triangular-causal-channel-individual \
        --finetune_result_path ./results/full-finetune_v1-triangular-causal-channel-individual \
        "$@"
done
done
done
done
