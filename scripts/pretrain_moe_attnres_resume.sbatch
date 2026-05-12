#!/bin/bash
#SBATCH --job-name=pretrain_moe_attnres_resume
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=15:00:00

set -euo pipefail
cd /mnt/home/brendan/post2/funstuff/nbn-gpt
mkdir -p slurm_logs

echo "[moe+attnres resume] starting at $(date) on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Resume from output_moe_attnres/step_1000/checkpoint.pt — the last model
# checkpoint saved before SLURM cut the first run at step 1707. Training
# resumes at step 1001 with restored model + optimizer state and the LR
# scheduler fast-forwarded. ~4100 more steps × ~12.4 s/step ≈ 14 h.
.venv/bin/python main_pretrain.py \
    --model_name gpt2small \
    --dataset fineweb \
    --output_dir output_moe_attnres \
    --batch_size 32 \
    --use_moe \
    --num_experts 4 \
    --experts_per_token 2 \
    --moe_aux_loss_weight 0.01 \
    --use_attn_res \
    --resume_training

echo "[moe+attnres resume] finished at $(date)"
