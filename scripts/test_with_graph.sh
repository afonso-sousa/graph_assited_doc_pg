dataset_name="data/par3_with_semantic_graphs.json"
split="test"
model_name="google/long-t5-tglobal-base"
output_dir="output"
metric="metrics/my_metric"

CUDA_VISIBLE_DEVICES=1 python test.py \
  --dataset_name $dataset_name \
  --model_name_or_path $output_dir/$model_name \
  --split $split \
  --predict_output_file $output_dir/$model_name/pred.tsv \
  --max_seq_length 1024 \
  --metric $metric \
  --predict_batch_size 32 \
  --with_graph