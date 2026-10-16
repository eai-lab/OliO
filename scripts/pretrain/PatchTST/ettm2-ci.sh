#!/bin/bash

# venv 활성화 (User Rule)
if [ -e "./venv/bin/activate" ]; then
    source ./venv/bin/activate
fi

# 기본 설정
MODEL_NAME="PatchTST"
DATA_NAME="ETTm2"
DATA_PATH="ETTm2.csv"
ROOT_PATH="./dataset"

# 로그 디렉토리 설정
LOG_DIR="logs/pretrain/$MODEL_NAME/$(basename "anonymous-repo/Anonymous.OliO/scripts/pretrain/PatchTST/ettm2-ci.sh" .sh)"
mkdir -p "$LOG_DIR"

echo "=========================================================="
echo "[$DATA_NAME CI Pretrain Start] Model: $MODEL_NAME"
echo "=========================================================="

count=0

# 하이퍼파라미터 검색 (Grid Search Space)
for seq_len in 336 672; do
for projection_dim in 128 256; do
for batch_size in 32; do
for max_tau in 8 32 64; do
for learning_rate in 0.001 0.0007; do
for train_epochs in 10 20; do

    count=$((count + 1))

    echo "----------------------------------------------------------"
    echo "[Task $count] seq:$seq_len, proj:$projection_dim, tau:$max_tau, lr:$learning_rate, epoch:$train_epochs"
    echo "----------------------------------------------------------"

    python -u run_pretrain.py \
        --model "$MODEL_NAME" \
        --data "$DATA_NAME" \
        --root_path "$ROOT_PATH" \
        --data_path "$DATA_PATH" \
        --features M \
        --seq_len "$seq_len" \
        --max_tau "$max_tau" \
        --enc_in 7 \
        --e_layers 3 \
        --n_heads 16 \
        --d_model 128 \
        --d_ff 512 \
        --dropout 0.3 \
        --fc_dropout 0.3 \
        --head_dropout 0.0 \
        --patch_len 12 \
        --stride 12 \
        --des sg-"$count" \
        --train_epochs "$train_epochs" \
        --batch_size "$batch_size" \
        --learning_rate "$learning_rate" \
        --module_version v1-triangular-causal-channel-individual \
        --projection_dim "$projection_dim" \
        "$@"
done
done
done
done
done
done
