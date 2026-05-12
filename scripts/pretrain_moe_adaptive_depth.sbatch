#!/bin/bash
#SBATCH --job-name=pretrain_moe_adaptive_depth
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=05:00:00

set -euo pipefail
cd /mnt/home/brendan/post2/funstuff/nbn-gpt
mkdir -p slurm_logs

echo "[moe+adaptive_depth] starting at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Same MoE config as the baseline MoE run. Adaptive-depth adds:
#   * a tiny linear depth_router (d -> L=12)
#   * stack-all-12-layer-outputs + per-token softmax-weighted sum
#   * a compute-cost penalty λ · mean(expected_depth)
# B=64 — adds only ~1.2 GB activation memory (one stack), unlike AttnRes which
# stacked 25 times.
.venv/bin/python main_pretrain.py \
    --model_name gpt2small \
    --dataset fineweb \
    --output_dir output_moe_adaptive_depth \
    --use_moe \
    --num_experts 4 \
    --experts_per_token 2 \
    --moe_aux_loss_weight 0.01 \
    --use_adaptive_depth \
    --adaptive_depth_cost_weight 0.01

echo "[moe+adaptive_depth] finished at $(date)"
