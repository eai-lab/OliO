#!/bin/bash
if [ -e venv/bin/activate ]; then
    source ./venv/bin/activate
fi

MODEL="PatchTST"
DATA="ETTm2"
ROOT_PATH="./dataset"
DATA_PATH="ETTm2.csv"
SEQ_LEN=336
ENC_IN=7
E_LAYERS=3
N_HEADS=16
D_MODEL=128
D_FF=256
DROPOUT=0.3
FC_DROPOUT=0.3
PATCH_LEN=12
STRIDE=12
BATCH_SIZE=128
EPOCHS=20
LRADJ="TST"
PATIENCE=5

# Search Space
LRS=(0.001 0.0005 0.0001 0.00001)
DOS=(0.0 0.1 0.2 0.3)
PRED_LENS=(96 192 336 720)
SEEDS=(2025 2024 2023)

# Specific Pretrain (from best.sh)
TARGET_PRETRAIN="PatchTST_ETTm2_sl336_pl12_tau8_128_sg-2"

for lr in "${LRS[@]}"; do
for do in "${DOS[@]}"; do
for pl in "${PRED_LENS[@]}"; do
for seed in "${SEEDS[@]}"; do
    
    if [[ $lr == 0.001 && $do == 0.0 ]]; then
        des="finetune-2"
    else
        des="finetune-2-lr$lr-do$do"
    fi

    echo ">>> Running $MODEL on $DATA | Pretrain: $TARGET_PRETRAIN | Pred: $pl | Seed: $seed | LR: $lr | DO: $do"

    python run_finetune.py \
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
        --d_ff $D_FF \
        --dropout $DROPOUT \
        --fc_dropout $FC_DROPOUT \
        --head_dropout $do \
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
        $EXTRA_ARGS
done
done
done
done
