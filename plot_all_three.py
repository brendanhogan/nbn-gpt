"""
Headline plot for the three May 2026 experiments — all four val loss curves overlaid.

Generates figs/all_three_loss_curves.png with:
  - Dense baseline
  - MoE
  - MoE + AttnRes
  - MoE + adaptive depth (collapsed to L1)
"""

import os
import json
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

RUNS = [
    ("Dense baseline",       "output_dense",              "#1f6feb", "o"),
    ("MoE",                  "output_moe",                "#d1410c", "o"),
    ("MoE + AttnRes",        "output_moe_attnres",        "#2f9e44", "s"),
    ("MoE + adaptive depth", "output_moe_adaptive_depth", "#9e2fb8", "^"),
]


def load_val_curve(output_dir):
    paths = sorted(glob.glob(os.path.join(output_dir, "step_*/val_loss.json")),
                   key=lambda p: int(p.rsplit("/", 2)[-2].split("_")[1]))
    steps, losses = [], []
    for p in paths:
        steps.append(int(p.rsplit("/", 2)[-2].split("_")[1]))
        losses.append(json.load(open(p))["validation_loss"])
    return np.array(steps), np.array(losses)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))

    for label, run, color, marker in RUNS:
        if not os.path.exists(run):
            continue
        steps, val = load_val_curve(run)
        keep = steps > 0
        for ax in axes:
            ax.plot(steps[keep], val[keep], color=color, marker=marker, ms=5, lw=2, label=label)

    axes[0].set_xlabel("Training step")
    axes[0].set_ylabel("Validation loss")
    axes[0].set_title("Validation loss — full run")
    axes[0].legend(loc="upper right")

    axes[1].set_xlabel("Training step")
    axes[1].set_ylabel("Validation loss")
    axes[1].set_title("Final stretch (step ≥ 3000)")
    axes[1].set_xlim(2800, 5200)
    axes[1].set_ylim(3.27, 3.55)
    axes[1].legend(loc="upper right")

    # Annotate the four final points on the right panel
    finals = {}
    for label, run, color, _ in RUNS:
        if not os.path.exists(run):
            continue
        steps, val = load_val_curve(run)
        finals[label] = (steps[-1], val[-1], color)
    for label, (s, v, c) in finals.items():
        axes[1].annotate(f"{v:.3f}", xy=(s, v), xytext=(s + 30, v),
                         fontsize=9, color=c, va="center", ha="left", fontweight="bold")

    fig.suptitle("May 2026 — all three experiments vs dense baseline\n"
                 "Same data (FineWeb), same step count (5100), same H100",
                 fontsize=14, y=1.02)
    out = "figs/all_three_loss_curves.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
