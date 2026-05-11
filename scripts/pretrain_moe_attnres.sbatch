#!/bin/bash
#SBATCH --job-name=pretrain_moe_attnres
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00

set -euo pipefail
cd /mnt/home/brendan/post2/funstuff/nbn-gpt
mkdir -p slurm_logs

echo "[moe+attnres] starting at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# B=32 (vs B=64 for the dense/MoE runs) because AttnRes holds the full 25-element
# history of sublayer outputs live through the forward pass (~2.5 GB of bf16
# activations), pushing the H100 over its 80 GB limit at B=64 once cross_entropy
# allocates its fp32 logits buffer. With total_batch_size unchanged the
# grad_accum_steps auto-doubles, so the effective gradient batch is identical to
# the MoE run — only the wall-clock per step changes slightly.
.venv/bin/python main_pretrain.py \
    --model_name gpt2small \
    --dataset fineweb \
    --output_dir output_moe_attnres \
    --batch_size 32 \
    --use_moe \
    --num_experts 4 \
    --experts_per_token 2 \
    --moe_aux_loss_weight 0.01 \
    --use_attn_res

echo "[moe+attnres] finished at $(date)"
