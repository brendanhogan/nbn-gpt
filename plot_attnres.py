"""
Plots for the AttnRes experiment.

Generates 3 figures into figs/:

  1. attnres_vs_moe_loss.png     — overlaid train+val loss curves for the two runs
                                   (output_moe vs output_moe_attnres)
  2. attnres_attention_matrix.png — the headline figure: heatmap showing, for each
                                   sublayer (rows, 0 = first attn, up through "final"),
                                   how much weight it gave to each previous output
                                   (cols, 0 = token embedding, 1.. = sublayer outputs).
                                   Upper-triangular by construction.
  3. attnres_entropy_evolution.png — per-sublayer alpha entropy over training. Shows
                                   the router going from "uniform at init" (entropy
                                   = ln(history_size)) to whatever it learns.

Run after the MoE+AttnRes run finishes:
    .venv/bin/python plot_attnres.py
"""

import os
import json
import glob
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl


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

MOE_DIR        = "output_moe"
MOE_ATTNRES_DIR = "output_moe_attnres"
FIGS_DIR       = "figs"
os.makedirs(FIGS_DIR, exist_ok=True)

MOE_COLOR     = "#d1410c"
ATTNRES_COLOR = "#2f9e44"


# -- Helpers ------------------------------------------------------------------
def load_train_curve(output_dir: str, sample_every: int = 25):
    """(steps, train_losses) from training_log.json files."""
    step_dirs = sorted(glob.glob(os.path.join(output_dir, "step_*")),
                       key=lambda p: int(p.rsplit("_", 1)[1]))
    last_step = int(step_dirs[-1].rsplit("_", 1)[1]) if step_dirs else 0
    steps, losses = [], []
    for d in step_dirs:
        step = int(d.rsplit("_", 1)[1])
        if step % sample_every != 0 and step != last_step:
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


def load_val_curve(output_dir: str):
    paths = sorted(glob.glob(os.path.join(output_dir, "step_*/val_loss.json")),
                   key=lambda p: int(p.rsplit("/", 2)[-2].split("_")[1]))
    steps, losses = [], []
    for path in paths:
        step = int(path.rsplit("/", 2)[-2].split("_")[1])
        with open(path) as f:
            losses.append(json.load(f)["validation_loss"])
            steps.append(step)
    return np.array(steps), np.array(losses)


def load_attnres_probe(probe_path: str):
    """Returns list of (label, alpha_vector) tuples."""
    data = torch.load(probe_path, map_location="cpu", weights_only=False)
    return data["rows"]


def ema(x: np.ndarray, alpha: float = 0.08) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    return out


# -- Plot 1: Loss curves ------------------------------------------------------
def plot_loss_curves() -> None:
    print("[1/3] Loss curves...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))

    for run, color, label in [(MOE_DIR, MOE_COLOR, "MoE"),
                              (MOE_ATTNRES_DIR, ATTNRES_COLOR, "MoE + AttnRes")]:
        if not os.path.exists(run):
            print(f"  skipping {run} (not found)")
            continue
        steps_t, train = load_train_curve(run, sample_every=25)
        steps_v, val   = load_val_curve(run)

        axes[0].plot(steps_t, train, color=color, alpha=0.2, lw=0.7)
        axes[0].plot(steps_t, ema(train), color=color, lw=2.2, label=label)
        keep = steps_v > 0
        axes[1].plot(steps_v[keep], val[keep], color=color, marker="o", ms=6, lw=2, label=label)

    axes[0].set_xlabel("Training step"); axes[0].set_ylabel("Train loss"); axes[0].set_title("Train loss"); axes[0].set_yscale("log"); axes[0].legend()
    axes[1].set_xlabel("Training step"); axes[1].set_ylabel("Validation loss"); axes[1].set_title("Validation loss (step ≥ 500)"); axes[1].legend()

    # Δ annotations at every step where both runs have a val checkpoint —
    # the AttnRes run is shorter, so don't compare last-points across curves.
    if os.path.exists(MOE_DIR) and os.path.exists(MOE_ATTNRES_DIR):
        sm, vm = load_val_curve(MOE_DIR)
        sa, va = load_val_curve(MOE_ATTNRES_DIR)
        # Match steps that exist in both
        m_by_step = dict(zip(sm.tolist(), vm.tolist()))
        for s, v_attnres in zip(sa.tolist(), va.tolist()):
            if s == 0 or s not in m_by_step:
                continue
            delta = v_attnres - m_by_step[s]
            # Label above the AttnRes point
            axes[1].annotate(
                f"Δ={delta:+.3f}",
                xy=(s, v_attnres),
                xytext=(s, v_attnres + 0.03),
                fontsize=8, ha="center", color="#444",
            )

    fig.suptitle("MoE vs MoE + AttnRes: same data, same steps, same H100", fontsize=14)
    out = os.path.join(FIGS_DIR, "attnres_vs_moe_loss.png")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print(f"    → {out}")


# -- Plot 2: Attention matrix at final checkpoint -----------------------------
def plot_attention_matrix(probe_path: str | None = None) -> None:
    print("[2/3] Attention matrix...")

    if probe_path is None:
        # Find latest probe in output_moe_attnres
        candidates = sorted(glob.glob(os.path.join(MOE_ATTNRES_DIR, "step_*/attnres_probe.pt")),
                            key=lambda p: int(p.rsplit("/", 2)[-2].split("_")[1]))
        if not candidates:
            print("    no attnres_probe.pt found, skipping")
            return
        probe_path = candidates[-1]

    rows = load_attnres_probe(probe_path)
    # Each row: (label, alpha vector of length k)
    num_rows = len(rows)
    max_len  = max(len(a) for _, a in rows)
    M = np.full((num_rows, max_len), np.nan)
    labels = []
    for i, (label, alpha) in enumerate(rows):
        M[i, :len(alpha)] = alpha
        labels.append(label)

    step = int(probe_path.rsplit("/", 2)[-2].split("_")[1])

    fig, ax = plt.subplots(figsize=(11, 8))
    im = ax.imshow(M, aspect="auto", cmap="magma", origin="upper",
                   vmin=0, vmax=np.nanmax(M))
    ax.set_xlabel("Source: previous sublayer output (0 = token embedding)")
    ax.set_ylabel("Destination: which sublayer is reading the residual")
    ax.set_yticks(range(num_rows))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(range(max_len))
    ax.set_xticklabels(range(max_len), fontsize=8)
    # Annotate above-threshold cells so we can read at-a-glance
    threshold = 0.10  # only annotate alphas > 10%
    for i in range(num_rows):
        for j in range(max_len):
            if not np.isnan(M[i, j]) and M[i, j] >= threshold:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        color="white" if M[i, j] < np.nanmax(M) * 0.55 else "black",
                        fontsize=6.5)
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("α (softmax weight over depth)")
    ax.set_title(f"AttnRes depth-attention matrix at step {step}\n"
                 f"Row i = sublayer i's residual is α-weighted sum of columns 0..i-1.\n"
                 f"Cells ≥ {threshold:.0%} are labeled. Cells above the diagonal are zero (causal-in-depth).")
    out = os.path.join(FIGS_DIR, "attnres_attention_matrix.png")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print(f"    → {out}")


# -- Plot 3: Entropy evolution over training ----------------------------------
def plot_entropy_evolution() -> None:
    print("[3/3] Entropy evolution...")

    candidates = sorted(glob.glob(os.path.join(MOE_ATTNRES_DIR, "step_*/attnres_probe.pt")),
                        key=lambda p: int(p.rsplit("/", 2)[-2].split("_")[1]))
    if len(candidates) < 2:
        print("    not enough probes, skipping")
        return

    # For every probe, compute entropy of each row (sublayer's alpha distribution),
    # normalized by ln(history_size) so 1.0 = perfectly uniform, 0.0 = fully decisive.
    series_by_label = defaultdict(list)  # label -> list of (step, normalized_entropy)
    eps = 1e-9
    for probe_path in candidates:
        step = int(probe_path.rsplit("/", 2)[-2].split("_")[1])
        rows = load_attnres_probe(probe_path)
        for label, alpha in rows:
            ent = -(alpha * np.log(alpha + eps)).sum()
            max_ent = np.log(len(alpha)) if len(alpha) > 1 else 1.0
            series_by_label[label].append((step, ent / max_ent))

    fig, ax = plt.subplots(figsize=(12, 5.5))
    cmap = plt.cm.viridis
    labels_sorted = list(series_by_label.keys())  # already in sublayer order from probe
    for i, label in enumerate(labels_sorted):
        if label.startswith("L"):
            # Color by depth — extract layer index from "L<n>_attn" / "L<n>_ffn"
            L = int(label.split("_")[0][1:])
            num_layers = max(int(l.split("_")[0][1:]) for l in labels_sorted if l.startswith("L"))
            color = cmap(L / max(1, num_layers))
            lw = 1.2
            alpha = 0.85
        else:
            # "final" — make it stand out
            color = "black"
            lw = 2.2
            alpha = 1.0
        xs, ys = zip(*series_by_label[label])
        ax.plot(xs, ys, color=color, lw=lw, alpha=alpha, label=label if label == "final" else None)

    ax.axhline(1.0, color="gray", ls="--", lw=1, label="uniform / random init")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Normalized entropy (1 = uniform attention)")
    ax.set_title("AttnRes depth-attention entropy per sublayer over training\n"
                 "(each line is one sublayer's α distribution; color = depth, early=purple → late=yellow)")
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=0, vmax=11))
    cbar = fig.colorbar(sm, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Transformer block index")
    ax.legend(loc="lower left")
    out = os.path.join(FIGS_DIR, "attnres_entropy_evolution.png")
    fig.tight_layout(); fig.savefig(out); plt.close(fig)
    print(f"    → {out}")


if __name__ == "__main__":
    plot_loss_curves()
    plot_attention_matrix()
    plot_entropy_evolution()
    print("Done.")
