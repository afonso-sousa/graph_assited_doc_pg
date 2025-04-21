#!/bin/bash

model_name="facebook/bart-base"
data_dir="processed-data"
dataset="qqppos" # paranmt-small, qqppos, paws
lr="1e-4"
checkpoint=$model_name-$dataset-lr$lr-graph
model_dir=output/${checkpoint}
scores_file="multi_eval_scores.csv"
predictions_file="multi_eval_generations.csv"
splits_suffix="with_graph"

python evaluation.py \
    --model_name_or_path $model_dir \
    --with_graph \
    --dataset_name $data_dir/$dataset \
    --splits_suffix $splits_suffix \
    --output_dir $model_dir \
    --scores_file $scores_file \
    --predictions_file $predictions_file \
    --num_return_sequences 4 \
    --beam_width 2 \
    --num_beam_groups 2 \
    --repetition_penalty 1.2 \
    --diversity_penalty 0.3
