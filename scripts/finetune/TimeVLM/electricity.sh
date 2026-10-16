#!/bin/bash
if [ -e venv/bin/activate ]; then
    source ./venv/bin/activate
fi

MODEL="TimeVLM"
DATA="Electricity"
ROOT_PATH="./dataset"
DATA_PATH="electricity.csv"
SEQ_LEN=512
ENC_IN=321
E_LAYERS=2
N_HEADS=8
D_MODEL=128
PATCH_LEN=16
STRIDE=8
BATCH_SIZE=16
EPOCHS=20
LRADJ="type1"
PATIENCE=5
IMAGE_SIZE=56
NORM_CONST=0.4
PERIODICITY=24

# Search Space
LRS=(0.001 0.0005 0.0001 0.00001)
DOS=(0.0 0.1 0.2 0.3)
PRED_LENS=(96 192 336)
SEEDS=(2025 2024 2023)

# Filter for specific pretrain (from best.sh)
TARGET_PRETRAIN="sg-41"

for pretrain_folder in results/pretrain/TimeVLM/*/*; do
    if [ ! -d "$pretrain_folder" ]; then continue; fi
    pretrain_exp=$(basename "$pretrain_folder")
    
    # Validation
    if [[ $pretrain_exp != *"$DATA"* ]]; then continue; fi
    if [[ $pretrain_exp != *"$TARGET_PRETRAIN"* ]]; then continue; fi

    exp_num=${pretrain_exp##*_}

    for lr in "${LRS[@]}"; do
    for do in "${DOS[@]}"; do
    for pl in "${PRED_LENS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        
        if [[ $lr == 0.001 && $do == 0.0 ]]; then
            des="finetune-$exp_num"
        else
            des="finetune-$exp_num-lr$lr-do$do"
        fi

        echo ">>> Running $MODEL on $DATA | Pretrain: $pretrain_exp | Pred: $pl | Seed: $seed | LR: $lr | DO: $do"

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
            --pretrain_exp $pretrain_exp \
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
            $EXTRA_ARGS
    done
    done
    done
    done
done