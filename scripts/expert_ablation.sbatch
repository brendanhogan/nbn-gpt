#!/bin/bash
#SBATCH --job-name=expert_ablation
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00

set -euo pipefail
cd /mnt/home/brendan/post2/funstuff/nbn-gpt
mkdir -p slurm_logs

echo "[ablation] starting at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

.venv/bin/python expert_ablation.py \
    --checkpoint output_moe/step_5000/checkpoint.pt \
    --num_val_steps 30 \
    --output output_moe/expert_ablation.json

echo "[ablation] finished at $(date)"
