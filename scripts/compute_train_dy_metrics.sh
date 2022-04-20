#!/bin/bash
#SBATCH --job-name=cartography
#SBATCH --partition=ckpt
#SBATCH --account=xlab
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --output="slurm/cartography/slurm-%J-%x.out"

cat $0
echo "--------------------"
model_path=models/roberta-large-mnli
evaluation_file=data/mnli/train.jsonl
dynamics_dir_name=training_dynamics_mnli

python -m cartography.compute_training_dynamics_hate \
    --model_path $model_path \
    --evaluation_file $evaluation_file \
    --dynamics_dir_name $dynamics_dir_name
