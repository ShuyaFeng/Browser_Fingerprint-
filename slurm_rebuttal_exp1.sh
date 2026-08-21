#!/bin/bash
#SBATCH --job-name=rebuttal_minentropy
#SBATCH --output=logs/rebuttal_exp1_%j.out
#SBATCH --error=logs/rebuttal_exp1_%j.err
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=emily.feng19@gmail.com

module load Anaconda3
conda activate fp-entropy  # change to your conda env name

mkdir -p logs results/rebuttal/figures

echo "=== Experiment 1: Min-entropy Shapley ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

python scripts/rebuttal_minentropy_shapley.py \
    --nrows 300000 \
    --mode monte_carlo \
    --n-perm 1000

echo "End: $(date)"
