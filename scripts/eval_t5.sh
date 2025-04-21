#!/bin/bash

model_name="t5-large"
data_dir="processed-data"
dataset="qqppos" # paranmt-small, qqppos, paws
lr="1e-4"
checkpoint=$model_name-$dataset-lr$lr-standard
model_dir=output/${checkpoint}
scores_file="eval_scores.csv"
predictions_file="eval_generations.csv"
data_type="dep_tree" # "graph"
file_suffix="with_$data_type"

python evaluation.py \
    --model_name_or_path $model_dir \
    --dataset_name $data_dir/$dataset \
    --splits_suffix $file_suffix \
    --output_dir $model_dir \
    --scores_file $scores_file \
    --predictions_file $predictions_file