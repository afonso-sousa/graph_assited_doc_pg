raw_datasets_dir="raw-data"
processed_datasets_dir="processed-data"
dataset="qqppos"

for split in "train" "validation" "test"
do
    output_file=$processed_datasets_dir/$dataset/${split}_with_graph.jsonl
    command="python prepare_data_with_graph.py \
            --dataset_name $raw_datasets_dir/$dataset \
            --split $split \
            --output_path $output_file \
            --drop_exemplars"
    
    if [ ! -f "$output_file" ]; then
        eval $command
    fi
done