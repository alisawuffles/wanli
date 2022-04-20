#!/bin/bash
#SBATCH --job-name=embed_examples
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=xlab
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output="slurm/representations/slurm-%J-%x.out"

cat $0
echo "--------------------"

python -m representations.embed_examples \
    --model_path models/roberta-large-mnli \
    --data_file data/mnli/train.jsonl \
    --dataset_name mnli
