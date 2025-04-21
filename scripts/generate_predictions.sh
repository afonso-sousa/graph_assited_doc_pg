#!/bin/bash

datasets_dir="processed-data"
model_dir="output"
output_dir=$model_dir
model_name="facebook/bart-base"
lr="1e-4"
dataset="paws"
name=$model_name-$dataset-lr$lr-standard
checkpoint="checkpoint-5460"
file_suffix="_with_graph"

python generate_predictions.py \
    --dataset_name $datasets_dir/$dataset \
    --test_file test${file_suffix}.jsonl \
    --model_name_or_path $model_dir/${name}/${checkpoint} \
    --output_dir $output_dir/$name \
    --source_column source \
    --target_column target \
    --per_device_eval_batch_size 2 \
    # --penalty_alpha 0.6 \
    # --top_k 4
    # --num_return_sequences 4 \
    # --beam_width 2 \
    # --num_beam_groups 2 \
    # --repetition_penalty 1.2 \
    # --diversity_penalty 0.3 \
    # --early_stopping
