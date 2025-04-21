#!/bin/bash

datasets_dir="processed-data"
model_dir="output"
output_dir=$model_dir
model_name="facebook/bart-base"
lr="1e-4"
dataset="paws"
name=$model_name-$dataset-lr$lr-standard
checkpoint="checkpoint-2480"
file_suffix="_to_transform"

for split in "train" "validation" "test"
do
    CUDA_VISIBLE_DEVICES=0 python construct_by_predicting.py \
        --dataset_name $datasets_dir/$dataset \
        --train_file ${split}${file_suffix}.jsonl \
        --model_name_or_path $model_dir/${name}/${checkpoint} \
        --output_dir $output_dir/$name \
        --source_column source \
        --target_column target \
        --per_device_eval_batch_size 2
done
