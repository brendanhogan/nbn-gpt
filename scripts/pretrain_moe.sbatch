#!/bin/bash
#SBATCH --job-name=pretrain_moe
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00

set -euo pipefail

# Run from the project root regardless of where sbatch was launched from.
cd /mnt/home/brendan/post2/funstuff/nbn-gpt
mkdir -p slurm_logs

echo "[moe] starting at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

.venv/bin/python main_pretrain.py \
    --model_name gpt2small \
    --dataset fineweb \
    --output_dir output_moe \
    --use_moe \
    --num_experts 4 \
    --experts_per_token 2 \
    --moe_aux_loss_weight 0.01

echo "[moe] finished at $(date)"
