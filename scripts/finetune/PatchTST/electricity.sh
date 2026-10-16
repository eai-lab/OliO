#!/bin/bash
if [ -e venv/bin/activate ]; then
    source ./venv/bin/activate
fi

MODEL="PatchTST"
DATA="Electricity"
ROOT_PATH="./dataset"
DATA_PATH="electricity.csv"
SEQ_LEN=336
ENC_IN=321
E_LAYERS=3
N_HEADS=16
D_MODEL=128
D_FF=256
DROPOUT=0.2
FC_DROPOUT=0.2
PATCH_LEN=12
STRIDE=12
BATCH_SIZE=32
EPOCHS=20
LRADJ="TST"
PATIENCE=5
PCT_START=0.2

# Search Space
LRS=(0.001 0.0005 0.0001)
DOS=(0.0 0.1 0.2)
PRED_LENS=(96 192 336 720)
SEEDS=(2025 2024 2023)

# Filter for specific pretrain (from best.sh)
PRETRAIN_FILTER="sg-25"

for pretrain_folder in results/pretrain/PatchTST/*/*; do
    if [ ! -d "$pretrain_folder" ]; then continue; fi
    pretrain_exp=$(basename "$pretrain_folder")
    
    # Validation
    if [[ $pretrain_exp != *"$DATA"* ]]; then continue; fi
    if [[ $pretrain_exp != *"$PRETRAIN_FILTER"* ]]; then continue; fi

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
            --d_ff $D_FF \
            --dropout $DROPOUT \
            --fc_dropout $FC_DROPOUT \
            --head_dropout $do \
            --patch_len $PATCH_LEN \
            --stride $STRIDE \
            --des "$des" \
            --train_epochs $EPOCHS \
            --batch_size $BATCH_SIZE \
            --learning_rate $lr \
            --lradj $LRADJ \
            --patience $PATIENCE \
            --pct_start $PCT_START \
            --random_seed $seed \
            --pretrain_exp "$pretrain_exp" \
            $EXTRA_ARGS
    done
    done
    done
    done
done
