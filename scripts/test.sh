dataset_name="data/par3"
model_name="google/long-t5-tglobal-base"
output_dir="output"
metric="metrics/my_metric"

CUDA_VISIBLE_DEVICES=1 python test.py \
  --dataset_name $dataset_name \
  --model_name_or_path $output_dir/$model_name \
  --predict_output_file $output_dir/$model_name/pred.tsv \
  --max_seq_length 512 \
  --metric $metric \
  --predict_batch_size 32