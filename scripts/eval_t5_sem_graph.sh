#!/bin/bash

model_name="t5-large"
data_dir="processed-data"
dataset="qqppos" # paranmt-small, qqppos, paws
lr="1e-4"
data_type="graph" # "amr" # "dep_tree" # "graph"
checkpoint=$model_name-$dataset-lr$lr-$data_type
model_dir=output/${checkpoint}
scores_file="eval_scores.csv"
predictions_file="eval_generations.csv"
file_suffix="with_$data_type"

CUDA_VISIBLE_DEVICES=1 python evaluation.py \
    --model_name_or_path $model_dir \
    --with_graph \
    --graph_type graph \
    --relations \
    --dataset_name $data_dir/$dataset \
    --splits_suffix $file_suffix \
    --output_dir $model_dir \
    --scores_file $scores_file \
    --predictions_file $predictions_file