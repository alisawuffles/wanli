#!/bin/bash
#SBATCH --job-name=filter
#SBATCH --partition=gpu-rtx6k
#SBATCH --account=xlab
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --time=3:00:00
#SBATCH --output="slurm/filter/slurm-%J-%x.out"

cat $0
echo "--------------------"
data_file=generated_data/examples.jsonl

python -m filtering.filter \
    --data_file $data_file
