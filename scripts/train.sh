#!/bin/bash

model_name="google/long-t5-tglobal-base"
dataset_name="data/par3"

# CUDA_VISIBLE_DEVICES=0,1 python 
accelerate launch --multi_gpu train.py \
    --output_dir output/$model_name \
    --dataset_name $dataset_name \
    --model_name_or_path $model_name \
    --max_seq_length 512 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear \
    --num_warmup_steps 500 \
    --gradient_accumulation_steps 8