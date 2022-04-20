#!/bin/bash
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=xlab
#SBATCH --nodes=1
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --output="slurm/generate/slurm-%J.out"

cat $0
echo "--------------------"

python -m pipeline \
    --model_path models/roberta-large-mnli \
    --num_gens_per_prompt 5 \
    --num_incontext_examples 5 \
    --ambiguity_quantile 0.75 \
    --num_examples 400000
