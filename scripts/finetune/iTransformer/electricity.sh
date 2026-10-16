#!/bin/bash

# venv 활성화 (User Rule)
if [ -e "./venv/bin/activate" ]; then
    source ./venv/bin/activate
fi

# 기본 설정
MODEL_NAME="iTransformer"
DATA_NAME="Electricity"
DATA_PATH="electricity.csv"
ROOT_PATH="./dataset"

# 로그 디렉토리 설정
LOG_DIR="logs/finetune/$MODEL_NAME/electricity"
mkdir -p "$LOG_DIR"

# Pre-trained 모델 정보 (전용 폴더)
PRETRAIN_EXP="iTransformer_Electricity_sl192_pl12_tau8_256_sg-39"
EXP_NUM="39"

echo "=========================================================="
echo "[$DATA_NAME Finetune Start] Pretrain Exp: $PRETRAIN_EXP"
echo "=========================================================="

count=0

# 하이퍼파라미터 검색 (Finetuning)
for learning_rate in 0.001 0.0005 0.0001 0.00001; do
for head_dropout in 0.0 0.1 0.2 0.3; do
for pred_len in 96 192 336 720; do
for random_seed in 2025 2024 2023; do

    count=$((count + 1))
    
    # des 설정
    if [[ $learning_rate == "0.001" && $head_dropout == "0.0" ]]; then
        des="finetune-$EXP_NUM"
    else
        des="finetune-$EXP_NUM-lr$learning_rate-do$head_dropout"
    fi

    echo "----------------------------------------------------------"
    echo "[Task $count] pred_len:$pred_len, seed:$random_seed, lr:$learning_rate, dropout:$head_dropout"
    echo "----------------------------------------------------------"

    bash scripts/check_and_run_finetune.sh \
        --model "$MODEL_NAME" \
        --data "$DATA_NAME" \
        --root_path "$ROOT_PATH" \
        --data_path "$DATA_PATH" \
        --features M \
        --seq_len 96 \
        --pred_len "$pred_len" \
        --enc_in 321 \
        --e_layers 3 \
        --n_heads 8 \
        --d_model 512 \
        --d_ff 512 \
        --dropout "$head_dropout" \
        --des "$des" \
        --pretrain_exp "$PRETRAIN_EXP" \
        --train_epochs 10 \
        --batch_size 32 \
        --learning_rate "$learning_rate" \
        --lradj type1 \
        --patience 5 \
        --random_seed "$random_seed" \
        "$@"
done
done
done
done