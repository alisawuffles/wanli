#!/bin/bash
#SBATCH --job-name=keep-ambiguous
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=xlab
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output="slurm/filter/slurm-%J-%x.out"

cat $0
echo "--------------------"

python -m filtering.keep_ambiguous \
    --data_file generated_data/filtered_examples.jsonl \
    --td_metrics_file models/roberta-large-mnli/training_dynamics_gen/td_metrics.jsonl
