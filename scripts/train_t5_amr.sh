#!/bin/bash

datasets_dir="processed-data"
output_dir="output"
model_name="t5-large"
lr="1e-4"
dataset="qqppos"
data_type="amr" # "dep_tree" # "graph"
output_file=$model_name-$dataset-lr$lr-$data_type
file_suffix="with_$data_type"

# CUDA_VISIBLE_DEVICES=0 python
accelerate launch train.py \
    --model_name_or_path $model_name \
    --with_graph \
    --graph_type amr \
    --output_dir $output_dir/$output_file \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 16 \
    --dataset_name $datasets_dir/$dataset \
    --splits_suffix $file_suffix \
    --num_warmup_steps 100 \
    --learning_rate $lr \
    --num_train_epochs 4 \
    --max_eval_samples 10 \
    --evaluation_interval 2 \
    --max_source_length 256 \
    --max_target_length 64 \
