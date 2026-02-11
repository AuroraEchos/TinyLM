#!/bin/bash

# ===============================
# TinyLM Pretraining Launcher
# Single GPU
# ===============================

export CUDA_VISIBLE_DEVICES=0

PROJECT="TinyLM"
EXP_NAME="tinylm_pretrain_512d_8L_4e-4_lr"
DATA_PATH="data/pretrain_data.bin"

EPOCHS=4
BATCH_SIZE=8
GRAD_ACCUM=8
LR=5e-4
SEQ_LEN=512
WARMUP_STEPS=3000

python -m scripts.dataset_pretrain \
    --data_path ${DATA_PATH} \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --grad_accum_steps ${GRAD_ACCUM} \
    --lr ${LR} \
    --warmup_steps ${WARMUP_STEPS} \
    --max_seq_len ${SEQ_LEN} \
    --use_wandb \
    --wandb_project ${PROJECT} \
    --exp_name ${EXP_NAME}
