#!/bin/bash

base_path="../semantic_para_gen"
predictions_dir="output"
output_dir=$predictions_dir
model_name="facebook/bart-base"
dataset="paws"
lr="1e-4"
conditions="50-30"
name=$model_name-$dataset-lr$lr-vector
metrics="my_metric"
output_file="$output_dir/$name/$metrics-$conditions.csv"
input_file="diversity_generations.csv"
# "generated_predictions-$conditions.csv"

if [ ! -f "$output_file" ]; then
    job="CUDA_VISIBLE_DEVICES=1 python compute_metrics.py \
            --input_path ${base_path}/$predictions_dir/$name/$input_file \
            --source_column source \
            --target_column target \
            --predictions_column prediction \
            --metric metrics/$metrics \
            --output_path $output_file \
        "
    eval $job
fi            
