#!/bin/bash
#SBATCH --job-name=rebuttal_magnitude
#SBATCH --output=logs/rebuttal_exp2_%j.out
#SBATCH --error=logs/rebuttal_exp2_%j.err
#SBATCH --partition=short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=emily.feng19@gmail.com

module load Anaconda3
conda activate fp-entropy  # change to your conda env name

mkdir -p logs results/rebuttal/figures

echo "=== Experiment 2: Magnitude-sensitive Validation ==="
echo "Start: $(date)"
echo "Node: $(hostname)"

python scripts/rebuttal_magnitude_validation.py \
    --nrows 300000 \
    --n-per-size 40

echo "End: $(date)"
