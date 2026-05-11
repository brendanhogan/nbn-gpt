"""
Expert ablation experiment.

Loads the trained MoE checkpoint, then for each (layer, expert) combination:
    1. Disables that expert (router can never pick it)
    2. Measures validation loss on a fixed subset

Output: JSON mapping (layer, expert) -> ablated val loss, plus a baseline.

Run:
    .venv/bin/python expert_ablation.py \
        --checkpoint output_moe/step_5000/checkpoint.pt \
        --num_val_steps 30 \
        --output output_moe/expert_ablation.json
"""

import os
import json
import time
import argparse

import torch
import torch.nn.functional as F

import models
import datasets
import tokenizers


@torch.no_grad()
def evaluate(model, val_loader, num_steps, device_type="cuda"):
    model.eval()
    val_loader.reset()
    total = 0.0
    for _ in range(num_steps):
        x, y = val_loader.get_batch()
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            _, loss = model(x, y, return_logits=False)
        total += loss.item()
    return total / num_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--num_val_steps", type=int, default=30)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--num_experts", type=int, default=4)
    parser.add_argument("--experts_per_token", type=int, default=2)
    parser.add_argument("--moe_aux_loss_weight", type=float, default=0.01)
    parser.add_argument("--model_name", type=str, default="gpt2small")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # Build the same model architecture used for training, then load weights.
    tokenizer = tokenizers.get_tokenizer("gpt2")
    model = models.get_model(
        model_name=args.model_name,
        vocab_size=tokenizer.vocab_size,
        use_moe=True,
        num_experts=args.num_experts,
        experts_per_token=args.experts_per_token,
        moe_aux_loss_weight=args.moe_aux_loss_weight,
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    # The training run wrapped the model in torch.compile which prefixes keys with "_orig_mod.".
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model_state_dict"].items()}
    model.load_state_dict(state_dict)
    print(f"loaded checkpoint at step {ckpt.get('step', '?')}, ckpt val_loss = {ckpt.get('val_loss', '?')}")

    # Build val loader on the same fineweb val shards used at training time.
    _, val_loader = datasets.get_data_loaders(
        dataset_name="fineweb",
        tokenizer=tokenizer,
        batch_size=64,
        sequence_length=1024,
        split_ratio=0.9,
        process_rank=0,
        num_processes=1,
    )

    num_layers = len(model.transformer_blocks.transformers)
    moe_blocks = [b.feed_forward_layer for b in model.transformer_blocks.transformers]
    assert all(isinstance(b, models.MixtureOfExperts) for b in moe_blocks), "all blocks should be MoE"

    # Helper to set/clear all blocks' disabled_expert flags.
    def set_disabled(layer_idx: int | None, expert_idx: int | None) -> None:
        for L, blk in enumerate(moe_blocks):
            blk.disabled_expert = expert_idx if L == layer_idx else None

    results = {}

    # Baseline (no expert disabled).
    set_disabled(None, None)
    t0 = time.time()
    baseline = evaluate(model, val_loader, args.num_val_steps)
    print(f"baseline val_loss = {baseline:.4f}   ({time.time() - t0:.1f}s)")
    results["baseline"] = baseline
    results["per_ablation"] = {}

    # Ablate each (layer, expert).
    for L in range(num_layers):
        for E in range(args.num_experts):
            set_disabled(L, E)
            t0 = time.time()
            loss = evaluate(model, val_loader, args.num_val_steps)
            dt = time.time() - t0
            delta = loss - baseline
            results["per_ablation"][f"L{L}_E{E}"] = {
                "layer": L,
                "expert": E,
                "ablated_val_loss": loss,
                "delta_vs_baseline": delta,
            }
            print(f"L{L:>2d} E{E}: val_loss = {loss:.4f}   Δ = {delta:+.4f}   ({dt:.1f}s)")

    set_disabled(None, None)

    results["num_val_steps"] = args.num_val_steps
    results["checkpoint"] = args.checkpoint
    results["num_layers"] = num_layers
    results["num_experts"] = args.num_experts

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
