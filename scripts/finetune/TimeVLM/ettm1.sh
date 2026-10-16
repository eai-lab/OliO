#!/bin/bash
if [ -e venv/bin/activate ]; then
    source ./venv/bin/activate
fi

MODEL="TimeVLM"
DATA="ETTm1"
ROOT_PATH="./dataset"
DATA_PATH="ETTm1.csv"
SEQ_LEN=512
ENC_IN=7
E_LAYERS=2
N_HEADS=8
D_MODEL=64
PATCH_LEN=16
STRIDE=8
BATCH_SIZE=32
EPOCHS=20
LRADJ="type1"
PATIENCE=5
IMAGE_SIZE=56
NORM_CONST=0.4
PERIODICITY=96

# Search Space
LRS=(0.001 0.0005 0.0001 0.00001)
DOS=(0.0 0.1 0.2 0.3)
PRED_LENS=(96 192 336 720)
SEEDS=(2025 2024 2023)

# Specific Pretrain (from ettm1-best.sh)
TARGET_PRETRAIN="TimeVLM_ETTm1_sl512_pl16_tau64_128_sg-9"
PRETRAIN_BASE="./results/pretrain_v1-triangular-causal-channel-individual"
FINETUNE_BASE="./results/full-finetune_v1-triangular-causal-channel-individual"

for lr in "${LRS[@]}"; do
for do in "${DOS[@]}"; do
for pl in "${PRED_LENS[@]}"; do
for seed in "${SEEDS[@]}"; do
    
    if [[ $lr == 0.001 && $do == 0.0 ]]; then
        des="finetune-v1-triangular-causal-channel-individual-9"
    else
        des="finetune-v1-triangular-causal-channel-individual-9-lr$lr-do$do"
    fi

    echo ">>> Running $MODEL on $DATA | Pretrain: $TARGET_PRETRAIN | Pred: $pl | Seed: $seed | LR: $lr | DO: $do"

    bash scripts/check_and_run_finetune.sh \
        --model $MODEL \
        --data $DATA \
        --root_path $ROOT_PATH \
        --data_path $DATA_PATH \
        --features M \
        --seq_len $SEQ_LEN \
        --pred_len $pl \
        --enc_in $ENC_IN \
        --e_layers $E_LAYERS \
        --n_heads $N_HEADS \
        --d_model $D_MODEL \
        --dropout $do \
        --patch_len $PATCH_LEN \
        --stride $STRIDE \
        --des "$des" \
        --pretrain_exp "$TARGET_PRETRAIN" \
        --train_epochs $EPOCHS \
        --batch_size $BATCH_SIZE \
        --learning_rate $lr \
        --lradj $LRADJ \
        --patience $PATIENCE \
        --random_seed $seed \
        --image_size $IMAGE_SIZE \
        --norm_const $NORM_CONST \
        --use_mem_gate \
        --periodicity $PERIODICITY \
        --three_channel_image \
        --learnable_image \
        --use_cross_attention \
        --pretrain_base_path "$PRETRAIN_BASE" \
        --finetune_result_path "$FINETUNE_BASE" \
        --gpu_preload \
        $EXTRA_ARGS
done
done
done
done