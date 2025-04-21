
seed=1
predictions_dir="output"
output_dir=$predictions_dir
model_name="facebook/bart-base"
dataset="paws"
lr="1e-4"
name=$model_name-$dataset-lr$lr-no-reentrancy
metrics="my_metric"
output_file="$output_dir/$name/tagged_metrics.csv"
input_file="eval_generations.csv"

if [ ! -f "$output_file" ]; then
    job="CUDA_VISIBLE_DEVICES=1 python compute_metrics.py \
            --input_path $predictions_dir/$name/$input_file \
            --source_column source \
            --target_column target \
            --predictions_column prediction \
            --metric metrics/$metrics \
            --output_path $output_file \
            --compute_pair_wise
        "
    eval $job
fi         