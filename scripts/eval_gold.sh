#!/bin/bash

seed=1
predictions_dir="output"
output_dir=$predictions_dir
model_name="facebook/bart-base"
dataset="paws"
lr="1e-4"
name=$model_name-$dataset-lr$lr-standard
metrics="my_metric"
output_file="$output_dir/$name/$gold_metrics.csv"

if [ ! -f "$output_file" ]; then
    job="CUDA_VISIBLE_DEVICES=1 python compute_metrics.py \
            --input_path $predictions_dir/$name/eval_generations.csv \
            --source_column source \
            --target_column target \
            --predictions_column source \
            --metric metrics/$metrics \
            --output_path $output_file \
        "
    eval $job
fi            
