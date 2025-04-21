#!/bin/bash

model_name="facebook/bart-base"
data_dir="processed-data"
dataset="paranmt-small"
lr="1e-4"
checkpoint=$model_name-$dataset-lr$lr-with_tokens
model_dir=output/${checkpoint}
scores_file="eval_scores.csv"
predictions_file="eval_generations.csv"
splits_suffix="with_graph"

CUDA_VISIBLE_DEVICES=1 python evaluation.py \
    --model_name_or_path $model_dir \
    --with_graph \
    --with_token_types \
    --dataset_name $data_dir/$dataset \
    --splits_suffix $splits_suffix \
    --output_dir $model_dir \
    --scores_file $scores_file \
    --predictions_file $predictions_file