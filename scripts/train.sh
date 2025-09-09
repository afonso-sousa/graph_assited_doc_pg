#!/bin/bash

model_name="google/bigbird-pegasus-large-arxiv"
dataset_name="data/par3"
block_size=8

# CUDA_VISIBLE_DEVICES=0,1 python 
# accelerate launch --multi_gpu 
CUDA_VISIBLE_DEVICES=0 python train.py \
    --output_dir output/${model_name}_${block_size} \
    --dataset_name $dataset_name \
    --model_name_or_path $model_name \
    --max_seq_length 512 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-5 \
    --lr_scheduler_type linear \
    --num_warmup_steps 500 \
    --gradient_accumulation_steps 16 \
    --block_size $block_size