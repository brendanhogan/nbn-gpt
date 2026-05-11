"""
Plots for the dense-vs-MoE pretraining experiment.

Produces four figures into figs/:
  1. moe_vs_dense_loss.png      — overlaid train and val loss curves
  2. utilization_evolution.png  — per-layer expert utilization, evolving over training
  3. router_entropy_curves.png  — router entropy per layer over training
  4. token_specialization.png   — for a late layer, the tokens most decisively routed

Run from project root:
    .venv/bin/python plot_experiment.py
"""

import os
import json
import glob
from collections import Counter

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import tokenizers

# -- Style --------------------------------------------------------------------
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

DENSE_COLOR = "#1f6feb"   # blue
MOE_COLOR   = "#d1410c"   # warm orange
EXPERT_CMAP = plt.cm.viridis

FIGS_DIR = "figs"
os.makedirs(FIGS_DIR, exist_ok=True)


# -- Loaders ------------------------------------------------------------------
def load_train_curve(output_dir: str, sample_every: int = 25) -> tuple[np.ndarray, np.ndarray]:
    """Read training_log.json from each step (subsampled) and return (steps, train_losses)."""
    step_dirs = sorted(glob.glob(os.path.join(output_dir, "step_*")), key=lambda p: int(p.rsplit("_", 1)[1]))
    steps, losses = [], []
    for d in step_dirs:
        step = int(d.rsplit("_", 1)[1])
        if step % sample_every != 0 and step != int(step_dirs[-1].rsplit("_", 1)[1]):
            continue
        path = os.path.join(d, "training_log.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            log = json.load(f)
        if "train_loss" not in log:
            continue
        steps.append(step)
        losses.append(log["train_loss"])
    return np.array(steps), np.array(losses)


def load_val_curve(output_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Read val_loss.json wherever it exists. Returns (steps, val_losses)."""
    paths = sorted(glob.glob(os.path.join(output_dir, "step_*/val_loss.json")),
                   key=lambda p: int(p.rsplit("/", 2)[-2].split("_")[1]))
    steps, losses = [], []
    for path in paths:
        step = int(path.rsplit("/", 2)[-2].split("_")[1])
        with open(path) as f:
            losses.append(json.load(f)["validation_loss"])
            steps.append(step)
    return np.array(steps), np.array(losses)


def load_moe_per_layer_metrics(output_dir: str, num_layers: int, sample_every: int = 25):
    """
    Returns dicts:
      entropy_by_layer[L] -> (steps, entropies)
      imbalance_by_layer[L] -> (steps, imbalance ratios)
    """
    step_dirs = sorted(glob.glob(os.path.join(output_dir, "step_*")), key=lambda p: int(p.rsplit("_", 1)[1]))
    rows = []
    for d in step_dirs:
        step = int(d.rsplit("_", 1)[1])
        if step % sample_every != 0 and step != int(step_dirs[-1].rsplit("_", 1)[1]):
            continue
        path = os.path.join(d, "training_log.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            log = json.load(f)
        if f"moe/layer_0/router_entropy" not in log:
            continue
        rows.append((step, log))

    entropies = {L: ([], []) for L in range(num_layers)}
    imbalances = {L: ([], []) for L in range(num_layers)}
    for step, log in rows:
        for L in range(num_layers):
            entropies[L][0].append(step)
            entropies[L][1].append(log[f"moe/layer_{L}/router_entropy"])
            imbalances[L][0].append(step)
            imbalances[L][1].append(log[f"moe/layer_{L}/imbalance_ratio"])
    return entropies, imbalances


def utilization_from_probe(probe_path: str) -> np.ndarray:
    """Returns (num_layers, num_experts) matrix of fraction-of-token-slots per expert."""
    probe = torch.load(probe_path, map_location="cpu", weights_only=False)
    routing = probe["routing_top_indices"].long()        # (L, B, T, k)
    num_layers = routing.shape[0]
    num_experts = probe["num_experts"]
    util = np.zeros((num_layers, num_experts))
    for L in range(num_layers):
        counts = torch.bincount(routing[L].reshape(-1), minlength=num_experts).float()
        util[L] = (counts / counts.sum()).numpy()
    return util


def ema(values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Exponential moving average smoothing for a noisy curve."""
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


# -- Plot 1: Loss curves -------------------------------------------------------
def plot_loss_curves() -> None:
    print("[1/4] Loss curves...")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    # --- Train loss ---
    for run, color, label in [("output_dense", DENSE_COLOR, "Dense"),
                              ("output_moe",   MOE_COLOR,   "MoE (4 experts, top-2)")]:
        steps, losses = load_train_curve(run, sample_every=25)
        axes[0].plot(steps, losses, color=color, alpha=0.20, lw=0.7)
        axes[0].plot(steps, ema(losses, 0.08), color=color, lw=2.2, label=label)
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Train loss")
    axes[0].set_title("Train loss")
    axes[0].set_yscale("log")
    axes[0].legend(loc="upper right")

    # --- Val loss ---
    for run, color, marker, label in [("output_dense", DENSE_COLOR, "o", "Dense"),
                                      ("output_moe",   MOE_COLOR,   "s", "MoE")]:
        steps, losses = load_val_curve(run)
        # Skip the step-0 init loss (~16) so the curve isn't dominated by it.
        keep = steps > 0
        axes[1].plot(steps[keep], losses[keep], color=color, marker=marker, ms=6, lw=2, label=label)
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Validation loss")
    axes[1].set_title("Validation loss (step ≥ 500)")
    axes[1].legend(loc="upper right")

    # Annotate the final gap
    steps_d, val_d = load_val_curve("output_dense")
    steps_m, val_m = load_val_curve("output_moe")
    final_d, final_m = val_d[-1], val_m[-1]
    axes[1].annotate(
        f"Δ = +{final_m - final_d:.3f}",
        xy=(steps_d[-1], (final_d + final_m) / 2),
        xytext=(steps_d[-1] - 1200, (final_d + final_m) / 2 + 0.04),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="gray", lw=1),
    )

    fig.suptitle("Dense vs MoE: same params, same tokens, 1×H100", fontsize=14)
    fig.tight_layout()
    out_path = os.path.join(FIGS_DIR, "moe_vs_dense_loss.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"    → {out_path}")


# -- Plot 2: Per-layer utilization heatmap evolution ---------------------------
def plot_utilization_evolution() -> None:
    print("[2/4] Utilization evolution heatmap...")

    snapshot_steps = [0, 500, 2500, 5099]
    fig, axes = plt.subplots(1, len(snapshot_steps), figsize=(4 * len(snapshot_steps), 5.5), sharey=True)

    vmin, vmax = 0.0, 0.45  # 0.25 is uniform; cap at 0.45 to show imbalance clearly

    for ax, step in zip(axes, snapshot_steps):
        probe_path = f"output_moe/step_{step}/routing_probe.pt"
        util = utilization_from_probe(probe_path)
        im = ax.imshow(util, aspect="auto", cmap=EXPERT_CMAP, vmin=vmin, vmax=vmax, origin="lower")
        ax.set_title(f"step {step}")
        ax.set_xlabel("Expert")
        ax.set_xticks(range(util.shape[1]))
        ax.set_xticklabels([f"E{e}" for e in range(util.shape[1])])
        # Annotate each cell
        for L in range(util.shape[0]):
            for e in range(util.shape[1]):
                ax.text(e, L, f"{util[L, e]:.2f}", ha="center", va="center",
                        color="white" if util[L, e] < 0.28 else "black", fontsize=7.5)

    axes[0].set_ylabel("Layer")
    axes[0].set_yticks(range(util.shape[0]))

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("Fraction of token-slots\n(0.25 = perfectly balanced)")

    fig.suptitle("Expert utilization per layer — random init → trained\n"
                 "(aux loss keeps every expert active; specialization happens within the balance constraint)",
                 fontsize=13)
    out_path = os.path.join(FIGS_DIR, "utilization_evolution.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"    → {out_path}")


# -- Plot 3: Router entropy per layer over training ----------------------------
def plot_router_entropy() -> None:
    print("[3/4] Router entropy curves...")

    # Read num_layers from the first probe
    probe = torch.load("output_moe/step_0/routing_probe.pt", map_location="cpu", weights_only=False)
    num_layers = probe["num_layers"]
    uniform_entropy = float(np.log(probe["num_experts"]))

    entropies, imbalances = load_moe_per_layer_metrics("output_moe", num_layers, sample_every=25)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # --- Entropy: one line per layer, color by depth ---
    cmap = plt.cm.plasma
    for L in range(num_layers):
        steps, ent = entropies[L]
        axes[0].plot(steps, ent, color=cmap(L / max(1, num_layers - 1)), lw=1.6, alpha=0.9)
    axes[0].axhline(uniform_entropy, color="gray", ls="--", lw=1, label=f"uniform = ln({probe['num_experts']}) ≈ {uniform_entropy:.2f}")
    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Router entropy")
    axes[0].set_title("Router entropy per layer\n(lower = router making more decisive picks)")
    axes[0].legend(loc="upper right")
    # Layer-color legend on the right edge
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=num_layers - 1))
    cbar = fig.colorbar(sm, ax=axes[0], fraction=0.04, pad=0.02)
    cbar.set_label("Layer (early → late)")

    # --- Imbalance: one line per layer ---
    for L in range(num_layers):
        steps, imb = imbalances[L]
        axes[1].plot(steps, imb, color=cmap(L / max(1, num_layers - 1)), lw=1.6, alpha=0.9)
    axes[1].axhline(1.0, color="gray", ls="--", lw=1, label="perfectly balanced (1.0)")
    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("max / min expert traffic")
    axes[1].set_title("Load imbalance per layer\n(aux loss keeps this near 1)")
    axes[1].set_yscale("log")
    axes[1].legend(loc="upper right")
    sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=num_layers - 1))
    fig.colorbar(sm2, ax=axes[1], fraction=0.04, pad=0.02).set_label("Layer (early → late)")

    fig.tight_layout()
    out_path = os.path.join(FIGS_DIR, "router_entropy_curves.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"    → {out_path}")


# -- Plot 4: Token specialization at a late layer ------------------------------
def plot_token_specialization(layer_to_show: int = 11, min_count: int = 200, top_n: int = 25) -> None:
    print(f"[4/4] Token specialization (layer {layer_to_show})...")

    probe = torch.load("output_moe/step_5099/routing_probe.pt", map_location="cpu", weights_only=False)
    routing   = probe["routing_top_indices"].long()        # (L, B, T, k)
    token_ids = probe["token_ids"].long()                  # (B, T)
    num_experts = probe["num_experts"]
    k = probe["experts_per_token"]

    layer_routing = routing[layer_to_show]                  # (B, T, k)
    B, T, _ = layer_routing.shape

    # For each token TYPE, count how many of its token-slot occurrences went to each expert.
    # token_ids_per_slot[(b,t,s)] = token_ids[b,t]
    token_ids_per_slot = token_ids.unsqueeze(-1).expand_as(layer_routing)  # (B, T, k)
    flat_tokens   = token_ids_per_slot.reshape(-1).numpy()
    flat_experts  = layer_routing.reshape(-1).numpy()

    # Count occurrences per token (each occurrence in a slot counts once).
    token_counter = Counter(flat_tokens.tolist())
    # Keep only tokens with enough mass to be statistically meaningful.
    eligible = {tid: c for tid, c in token_counter.items() if c >= min_count}

    # Build distribution per token: (num_eligible, num_experts), rows sum to 1.
    tids = sorted(eligible)
    tid_to_row = {tid: i for i, tid in enumerate(tids)}
    dist = np.zeros((len(tids), num_experts))
    for tid, e in zip(flat_tokens, flat_experts):
        if tid in tid_to_row:
            dist[tid_to_row[tid], e] += 1
    dist = dist / dist.sum(axis=1, keepdims=True)

    # Score = KL divergence from uniform — higher means more decisive routing.
    uniform = np.full(num_experts, 1.0 / num_experts)
    eps = 1e-9
    kl = (dist * (np.log(dist + eps) - np.log(uniform))).sum(axis=1)

    # Top-N most specialized tokens.
    top_idx = np.argsort(-kl)[:top_n]
    tok = tokenizers.get_tokenizer("gpt2")
    labels = [repr(tok.decode([tids[i]])) for i in top_idx]
    dist_top = dist[top_idx]

    # Plot: horizontal stacked bars.
    fig, ax = plt.subplots(figsize=(10, 0.32 * top_n + 1.5))
    colors = [EXPERT_CMAP(e / max(1, num_experts - 1)) for e in range(num_experts)]
    left = np.zeros(top_n)
    y = np.arange(top_n)
    for e in range(num_experts):
        ax.barh(y, dist_top[:, e], left=left, color=colors[e], edgecolor="white",
                label=f"Expert {e}")
        left += dist_top[:, e]

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of this token's top-k slots routed to each expert")
    ax.set_title(f"Most decisively-routed tokens at layer {layer_to_show} (final checkpoint, step 5099)\n"
                 f"Sorted by KL divergence from uniform routing  •  min count = {min_count} occurrences")
    ax.legend(loc="lower right", ncol=num_experts)
    ax.grid(axis="x", alpha=0.3)
    ax.grid(axis="y", visible=False)

    fig.tight_layout()
    out_path = os.path.join(FIGS_DIR, "token_specialization.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"    → {out_path}")


if __name__ == "__main__":
    plot_loss_curves()
    plot_utilization_evolution()
    plot_router_entropy()
    plot_token_specialization()
    print("\nDone. Plots are in figs/")
