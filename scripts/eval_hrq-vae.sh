#!/bin/bash

base_path="../hrq-vae"
metrics="my_metric"
output_file="$base_path/$metrics.csv"
dataset="paranmt"
diversity="" # "diversity_"

if [ ! -f "$output_file" ]; then
    job="CUDA_VISIBLE_DEVICES=1 python compute_metrics.py \
            --input_path ${base_path}/hrq-vae_${dataset}_${diversity}predictions.csv \
            --source_column source \
            --target_column target \
            --predictions_column prediction \
            --metric metrics/$metrics \
            --output_path $output_file \
        "
    eval $job
fi            
