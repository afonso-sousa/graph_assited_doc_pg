dataset_name="data/par3"
model_name="google/bigbird-pegasus-large-arxiv"
output_dir="output"
block_size=2
metric="metrics/my_metric"

CUDA_VISIBLE_DEVICES=0 python test.py \
  --dataset_name $dataset_name \
  --model_name_or_path $output_dir/${model_name}_${block_size} \
  --predict_output_file $output_dir/${model_name}_${block_size}/pred.tsv \
  --max_seq_length 512 \
  --metric $metric \
  --predict_batch_size 32