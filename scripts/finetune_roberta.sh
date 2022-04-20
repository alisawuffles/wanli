#!/bin/bash
#SBATCH --job-name=train
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=xlab
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output="slurm/train/slurm-%J-%x.out"

cat $0
echo "--------------------"

train_file=annotation/all_batches/duplicate_shuffles/train_9.jsonl
output_dir=models/duplicate_shuffles/roberta-large-wanli-9

python -m classification.run_nli \
  --model_name_or_path roberta-large \
  --do_train \
  --do_eval \
  --train_file $train_file \
  --validation_file data/mnli/dev_matched.jsonl \
  --per_device_train_batch_size 32 \
  --per_device_eval_batch_size 32 \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --logging_strategy steps \
  --learning_rate 1e-5 \
  --num_train_epochs 5 \
  --warmup_ratio 0.06 \
  --weight_decay 0.1 \
  --overwrite_cache \
  --output_dir $output_dir
