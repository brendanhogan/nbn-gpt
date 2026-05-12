"""
Plots for the adaptive-depth experiment.

Produces 3 figures into figs/:

  1. adaptive_depth_vs_moe_loss.png       — overlaid val loss curves with deltas
  2. adaptive_depth_alpha_evolution.png   — alpha weights per layer over training (heatmap)
  3. adaptive_depth_per_token.png         — at the final checkpoint, which tokens
                                            want shallower vs deeper readout
"""

import os
import json
import glob
from collections import Counter

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

import tokenizers


plt.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MOE_DIR = "output_moe"
AD_DIR  = "output_moe_adaptive_depth"
FIGS_DIR = "figs"
os.makedirs(FIGS_DIR, exist_ok=True)

MOE_COLOR = "#d1410c"
AD_COLOR  = "#9e2fb8"   # purple


# -- Helpers ------------------------------------------------------------------
def load_val_curve(output_dir):
    paths = sorted(glob.glob(os.path.join(output_dir, "step_*/val_loss.json")),
                   key=lambda p: int(p.rsplit("/", 2)[-2].split("_")[1]))
    steps, losses = [], []
    for path in paths:
        step = int(path.rsplit("/", 2)[-2].split("_")[1])
        with open(path) as f:
            losses.append(json.load(f)["validation_loss"])
            steps.append(step)
    return np.array(steps), np.array(losses)


def load_alpha_evolution(output_dir, num_layers=12, sample_every=25):
    """Read training_log.json for every (sub)sampled step, return (steps, alpha_LxN matrix)."""
    step_dirs = sorted(glob.glob(os.path.join(output_dir, "step_*")),
                       key=lambda p: int(p.rsplit("_", 1)[1]))
    last_step = int(step_dirs[-1].rsplit("_", 1)[1]) if step_dirs else 0
    steps, alphas, expected, entropy = [], [], [], []
    for d in step_dirs:
        step = int(d.rsplit("_", 1)[1])
        if step % sample_every != 0 and step != last_step:
            continue
        path = os.path.join(d, "training_log.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            log = json.load(f)
        if "depth/alpha_layer_1" not in log:
            continue
        steps.append(step)
        alphas.append([log[f"depth/alpha_layer_{k}"] for k in range(1, num_layers + 1)])
        expected.append(log["depth/mean_expected_depth"])
        entropy.append(log["depth/router_entropy"])
    return np.array(steps), np.array(alphas), np.array(expected), np.array(entropy)


def load_probe(probe_path):
    return torch.load(probe_path, map_location="cpu", weights_only=False)


# -- Plot 1: Loss curves ------------------------------------------------------
def plot_loss_curves():
    print("[1/3] Loss curves...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # Train loss not super interesting here; focus on val
    for run, color, label in [(MOE_DIR, MOE_COLOR, "MoE"),
                              (AD_DIR,  AD_COLOR,  "MoE + adaptive depth (λ=0.01)")]:
        if not os.path.exists(run):
            print(f"  skipping {run}")
            continue
        steps, val = load_val_curve(run)
        keep = steps > 0
        axes[0].plot(steps[keep], val[keep], color=color, marker="o", ms=5, lw=2, label=label)
        axes[1].plot(steps[keep], val[keep], color=color, marker="o", ms=5, lw=2, label=label)

    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Validation loss")
    axes[0].set_title("Validation loss (full range, step ≥ 500)")
    axes[0].legend(loc="upper right")

    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Validation loss")
    axes[1].set_title("Zoomed in (final stretch)")
    axes[1].set_xlim(3000, 5200)
    axes[1].set_ylim(3.28, 3.55)
    axes[1].legend(loc="upper right")

    # Annotate each step's Δ on the right panel
    sm, vm = load_val_curve(MOE_DIR)
    sa, va = load_val_curve(AD_DIR)
    m_by_step = dict(zip(sm.tolist(), vm.tolist()))
    for s, vA in zip(sa.tolist(), va.tolist()):
        if s in m_by_step and s >= 3500:
            axes[1].annotate(f"+{vA - m_by_step[s]:.3f}",
                             xy=(s, vA), xytext=(s, vA + 0.02),
                             fontsize=8, ha="center", color="#444")

    fig.suptitle("MoE vs MoE + adaptive-depth: same data, same steps, same H100", fontsize=14)
    out = os.path.join(FIGS_DIR, "adaptive_depth_vs_moe_loss.png")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print(f"    → {out}")


# -- Plot 2: Alpha evolution over training ------------------------------------
def plot_alpha_evolution():
    print("[2/3] Alpha evolution...")
    steps, alphas, expected, entropy = load_alpha_evolution(AD_DIR, num_layers=12, sample_every=25)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5),
                             gridspec_kw={"width_ratios": [1.4, 1]})

    # LEFT: heatmap of alpha[step, layer]
    ax = axes[0]
    im = ax.imshow(alphas.T, aspect="auto", origin="lower", cmap="magma",
                   extent=[steps[0], steps[-1], 0.5, 12.5], vmin=0, vmax=1)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Layer")
    ax.set_yticks(range(1, 13))
    ax.set_title("Depth-router α[layer k] over training\n"
                 "(uniform at init → collapse to L1)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("α (softmax weight on layer k)")

    # RIGHT: scalar evolution — expected depth and entropy
    ax2 = axes[1]
    ax2.plot(steps, expected, color="#1f6feb", lw=2, label="mean expected depth")
    ax2.set_xlabel("Training step")
    ax2.set_ylabel("Mean expected depth", color="#1f6feb")
    ax2.set_ylim(1, 12)
    ax2.tick_params(axis='y', colors="#1f6feb")
    ax2.axhline(6.5, color="gray", ls="--", lw=1, alpha=0.6, label="uniform avg (6.5)")
    ax2.legend(loc="upper right")
    ax2.set_title("Router decisiveness over training")

    ax3 = ax2.twinx()
    ax3.plot(steps, entropy, color="#c94e1c", lw=2, alpha=0.85)
    ax3.set_ylabel("Router entropy (nats)", color="#c94e1c")
    ax3.tick_params(axis='y', colors="#c94e1c")
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(True)
    ax3.spines["right"].set_color("#c94e1c")

    out = os.path.join(FIGS_DIR, "adaptive_depth_alpha_evolution.png")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print(f"    → {out}")


# -- Plot 3: Per-token depth distribution at final checkpoint -----------------
def plot_per_token():
    print("[3/3] Per-token depth distribution...")

    candidates = sorted(glob.glob(os.path.join(AD_DIR, "step_*/depth_probe.pt")),
                        key=lambda p: int(p.rsplit("/", 2)[-2].split("_")[1]))
    if not candidates:
        print("  no depth_probe.pt found")
        return
    probe = load_probe(candidates[-1])
    step = probe["step"]
    token_ids = probe["token_ids"].long()            # (B, T)
    alpha     = probe["depth_alpha"].float()         # (B, T, L)
    B, T, L = alpha.shape

    # Per token type, compute mean alpha across all its occurrences.
    flat_tokens = token_ids.reshape(-1).numpy()
    flat_alpha  = alpha.reshape(-1, L).numpy()
    counter = Counter(flat_tokens.tolist())

    min_count = 200
    eligible = {tid: c for tid, c in counter.items() if c >= min_count}

    # Mean alpha per eligible token, plus overall mean for reference
    by_token = {}
    for tid in eligible:
        mask = flat_tokens == tid
        by_token[tid] = flat_alpha[mask].mean(axis=0)
    mean_alpha = flat_alpha.mean(axis=0)

    # Rank tokens by their α_L12 (which ones want the deepest readout most)
    tok = tokenizers.get_tokenizer("gpt2")
    items = sorted(by_token.items(), key=lambda kv: -kv[1][-1])  # sort by L12 descending
    top_deep    = items[:15]
    top_shallow = items[-15:][::-1]   # smallest L12 → biggest L1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, L))

    for ax, group, title in [
        (axes[0], top_deep,    "Tokens that want the MOST L12 weight"),
        (axes[1], top_shallow, "Tokens that want the LEAST L12 weight (most L1)"),
    ]:
        labels = [repr(tok.decode([tid])) for tid, _ in group]
        dist   = np.stack([a for _, a in group])          # (N, L)
        y = np.arange(len(group))
        left = np.zeros(len(group))
        for k in range(L):
            ax.barh(y, dist[:, k], left=left, color=colors[k], edgecolor="white",
                    label=f"L{k+1}" if (k % 2 == 0 or k == L - 1) else None)
            left += dist[:, k]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, 1)
        ax.set_xlabel("Mean α across all occurrences")
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", alpha=0.3)

    handles, lbls = axes[0].get_legend_handles_labels()
    fig.legend(handles, lbls, loc="lower center", ncol=12, fontsize=8,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle(f"Per-token depth distribution at step {step}  "
                 f"(global mean expected_depth ≈ {(mean_alpha * np.arange(1, L+1)).sum():.2f}, "
                 f"L1={mean_alpha[0]:.2f}, L12={mean_alpha[-1]:.2f})", fontsize=13)
    out = os.path.join(FIGS_DIR, "adaptive_depth_per_token.png")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])
    fig.savefig(out); plt.close(fig)
    print(f"    → {out}")


if __name__ == "__main__":
    plot_loss_curves()
    plot_alpha_evolution()
    plot_per_token()
    print("Done.")
