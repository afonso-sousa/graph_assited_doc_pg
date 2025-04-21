#!/bin/bash

model_name="facebook/bart-base"
data_dir="processed-data"
dataset="qqppos"
lr="1e-4"
checkpoint=$model_name-$dataset-lr$lr-no-delimiters
model_dir=output/${checkpoint}
scores_file="eval_scores.csv"
predictions_file="eval_generations.csv"
splits_suffix="with_graph"

CUDA_VISIBLE_DEVICES=1 python evaluation.py \
    --model_name_or_path $model_dir \
    --with_graph \
    --delimiters \
    --dataset_name $data_dir/$dataset \
    --splits_suffix $splits_suffix \
    --output_dir $model_dir \
    --scores_file $scores_file \
    --predictions_file $predictions_file