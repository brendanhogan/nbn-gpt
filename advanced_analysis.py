"""
Advanced routing analyses on the trained MoE checkpoint.

Three analyses, all using the routing probe saved at the final step:

  (1) context-dependent routing
      For each frequent token TYPE, look at every occurrence in the probe and
      record the (sorted) pair of experts it routed to at a given layer. Then
      compute the entropy of that distribution over expert-pairs. Low entropy
      means the token always picks the same pair regardless of context
      ("router as lookup table"). High entropy means routing varies with
      context, which is the signature of useful learned routing.

  (3) token paths through the network
      For a sample sentence, trace each token's top-1 expert at every layer.
      Visualize as a heatmap (tokens × layers, cell colored by expert id).
      Reveals "highways" through the experts.

Analysis (2), the expert-ablation experiment, runs on a GPU via expert_ablation.py
and dumps a JSON; this script's plot for it just reads that JSON.

Plots written to figs/:
    context_dependence.png
    token_paths.png
    expert_ablation.png       (only if output_moe/expert_ablation.json exists)
"""

import os
import json
import math
from collections import Counter, defaultdict

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

import tokenizers


# -- Style (same as plot_experiment.py) ---------------------------------------
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

EXPERT_CMAP = plt.cm.viridis
FIGS_DIR = "figs"
os.makedirs(FIGS_DIR, exist_ok=True)


# -- Probe loading ------------------------------------------------------------
PROBE_PATH = "output_moe/step_5099/routing_probe.pt"


def load_probe() -> dict:
    return torch.load(PROBE_PATH, map_location="cpu", weights_only=False)


# -- Analysis 1: context-dependent routing -----------------------------------
def analyze_context_dependence(probe: dict, layers_to_show: list[int],
                                top_n_tokens: int = 12, min_count: int = 500) -> None:
    """
    For each (layer, token type), compute the entropy over the 6 possible
    sorted expert-pairs that the token's occurrences routed to.

    Plot two things in one figure:
      LEFT  — for a few representative layers, a histogram of per-token entropy.
              A bar at 0 means "every occurrence picks the same pair";
              a bar at log2(6) ≈ 2.585 means "occurrences split uniformly across all pairs".
      RIGHT — for a chosen layer, the top-N most frequent tokens with the
              non-uniform distribution they actually have, as stacked bars over pairs.
    """
    routing   = probe["routing_top_indices"].long()   # (L, B, T, k)
    token_ids = probe["token_ids"].long()             # (B, T)
    num_layers, B, T, k = routing.shape
    num_experts = probe["num_experts"]
    assert k == 2, "this analysis assumes top-k=2"

    # Enumerate all 6 sorted pairs (a < b) over 4 experts.
    pairs = [(a, b) for a in range(num_experts) for b in range(a + 1, num_experts)]
    pair_to_idx = {p: i for i, p in enumerate(pairs)}

    # Compute the (sorted) chosen pair at every (layer, batch, position).
    sorted_routing, _ = routing.sort(dim=-1)                          # (L, B, T, k)
    a = sorted_routing[..., 0].numpy()
    b = sorted_routing[..., 1].numpy()
    # Map (a, b) -> idx via a small lookup table.
    pair_idx_per_token = np.full(a.shape, -1, dtype=np.int64)
    for (pa, pb), idx in pair_to_idx.items():
        pair_idx_per_token[(a == pa) & (b == pb)] = idx              # (L, B, T)

    # For each token TYPE, count occurrences and accumulate per-pair counts per layer.
    flat_tids = token_ids.reshape(-1).numpy()                         # (B*T,)
    counter = Counter(flat_tids.tolist())
    eligible = sorted([t for t, c in counter.items() if c >= min_count])
    tid_to_row = {tid: i for i, tid in enumerate(eligible)}
    # Layer x type x pair_idx counts
    counts = np.zeros((num_layers, len(eligible), len(pairs)), dtype=np.int64)
    for L in range(num_layers):
        layer_pairs = pair_idx_per_token[L].reshape(-1)                # (B*T,)
        for tid, pidx in zip(flat_tids, layer_pairs):
            if tid in tid_to_row:
                counts[L, tid_to_row[tid], pidx] += 1

    # Distribution per (layer, token).
    totals = counts.sum(axis=-1, keepdims=True)                        # (L, T, 1)
    dist = counts / np.maximum(totals, 1)                              # (L, T, 6)
    # Entropy over pairs.
    eps = 1e-9
    entropy = -(dist * np.log2(dist + eps)).sum(axis=-1)               # (L, T)
    max_entropy = math.log2(len(pairs))                                # log2(6) ≈ 2.585

    # ---- Figure ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5),
                             gridspec_kw={"width_ratios": [1.2, 1.5]})

    # Left: histogram of entropies at a few layers
    ax = axes[0]
    layer_colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(layers_to_show)))
    for color, L in zip(layer_colors, layers_to_show):
        ax.hist(entropy[L], bins=np.linspace(0, max_entropy + 0.1, 25),
                histtype="step", lw=2, color=color, label=f"Layer {L}")
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.axvline(max_entropy, color="gray", ls=":", lw=1)
    ax.text(0.05, ax.get_ylim()[1] * 0.95, "fully deterministic\nrouting",
            color="gray", fontsize=8, va="top")
    ax.text(max_entropy - 0.05, ax.get_ylim()[1] * 0.95,
            "uniform over\nall pairs", color="gray", fontsize=8, va="top", ha="right")
    ax.set_xlabel("Entropy of pair-choice across occurrences  (bits)")
    ax.set_ylabel("# token types")
    ax.set_title(f"Per-token routing entropy by layer\n(token types with ≥ {min_count} occurrences)")
    ax.legend(title="layer", loc="upper right")

    # Right: top-N most-frequent tokens at a focal layer, stacked bar over pairs
    focal_layer = layers_to_show[-1]
    top_tids = sorted(eligible, key=lambda t: -counter[t])[:top_n_tokens]
    top_rows = [tid_to_row[t] for t in top_tids]
    dist_top = dist[focal_layer, top_rows]                              # (top_n, 6)
    tok = tokenizers.get_tokenizer("gpt2")
    labels = [repr(tok.decode([t])) for t in top_tids]

    pair_colors = plt.cm.tab10(np.linspace(0, 1, len(pairs)))
    y = np.arange(len(top_tids))
    left = np.zeros(len(top_tids))
    ax2 = axes[1]
    for i, pair in enumerate(pairs):
        ax2.barh(y, dist_top[:, i], left=left, color=pair_colors[i],
                 edgecolor="white", label=f"{{E{pair[0]}, E{pair[1]}}}")
        left += dist_top[:, i]
    ax2.set_yticks(y)
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1)
    ax2.set_xlabel("Fraction of this token's occurrences using each expert pair")
    ax2.set_title(f"Most frequent tokens — pair distribution at layer {focal_layer}")
    ax2.legend(title="expert pair", loc="lower right", ncol=2, fontsize=8)
    ax2.grid(axis="y", visible=False)
    ax2.grid(axis="x", alpha=0.3)

    fig.suptitle("Context dependence: does the SAME token route differently in different sentences?",
                 fontsize=14)
    out_path = os.path.join(FIGS_DIR, "context_dependence.png")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"context_dependence: → {out_path}")

    # Also print a small numerical summary the README writeup can quote.
    print(f"  Mean entropy per layer:")
    for L in range(num_layers):
        print(f"    layer {L:>2d}:  {entropy[L].mean():.3f} bits  (max possible {max_entropy:.3f})")


# -- Analysis 3: token paths --------------------------------------------------
def plot_token_paths(probe: dict, sentence_indices: list[int], max_tokens: int = 32) -> None:
    """
    For each chosen sentence (= one row of the batch in the probe), draw a
    (tokens × layers) heatmap where each cell is colored by the TOP-1 expert
    that token chose at that layer. Reveals "highways" — runs of consecutive
    tokens that share an expert path.
    """
    routing   = probe["routing_top_indices"].long()   # (L, B, T, k)
    token_ids = probe["token_ids"].long()             # (B, T)
    num_layers = routing.shape[0]
    num_experts = probe["num_experts"]

    tok = tokenizers.get_tokenizer("gpt2")

    fig, axes = plt.subplots(len(sentence_indices), 1,
                             figsize=(14, 3.2 * len(sentence_indices)),
                             squeeze=False)
    expert_colors = plt.cm.viridis(np.linspace(0, 1, num_experts))
    cmap = mpl.colors.ListedColormap(expert_colors)
    bounds = np.arange(num_experts + 1) - 0.5
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    for ax_row, sent_idx in zip(axes.flatten(), sentence_indices):
        # Top-1 expert per token per layer, just first `max_tokens` tokens.
        # routing shape (L, B, T, k); the highest-weight slot is index 0 in top_indices
        # (router's top-k returns sorted by value, descending). Use slot 0 as "top-1".
        paths = routing[:, sent_idx, :max_tokens, 0].numpy()           # (L, max_tokens)
        tids = token_ids[sent_idx, :max_tokens].tolist()
        token_strs = [tok.decode([t]) for t in tids]

        im = ax_row.imshow(paths, aspect="auto", cmap=cmap, norm=norm)
        ax_row.set_xticks(range(max_tokens))
        # Pretty token labels: replace whitespace tokens with visible glyphs
        clean = [(s.replace(" ", "·").replace("\n", "↵"))[:8] for s in token_strs]
        ax_row.set_xticklabels(clean, rotation=80, fontsize=8)
        ax_row.set_yticks(range(num_layers))
        ax_row.set_yticklabels(range(num_layers))
        ax_row.set_ylabel("Layer")
        ax_row.set_title(f"Sentence #{sent_idx}: top-1 expert per (token, layer)")
        ax_row.invert_yaxis()

    # Shared colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), ticks=range(num_experts),
                        fraction=0.025, pad=0.02)
    cbar.set_label("Top-1 expert")
    cbar.ax.set_yticklabels([f"E{e}" for e in range(num_experts)])

    fig.suptitle("Token paths through the experts\n"
                 "(each cell shows the top-1 expert that token picked at that layer)",
                 fontsize=14)
    out_path = os.path.join(FIGS_DIR, "token_paths.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"token_paths: → {out_path}")


# -- Analysis 2 plot: expert ablation ----------------------------------------
def plot_expert_ablation(json_path: str = "output_moe/expert_ablation.json") -> None:
    if not os.path.exists(json_path):
        print(f"expert_ablation: skipping ({json_path} not found yet)")
        return

    with open(json_path) as f:
        data = json.load(f)
    baseline = data["baseline"]
    num_layers = data["num_layers"]
    num_experts = data["num_experts"]
    delta_matrix = np.zeros((num_layers, num_experts))
    for entry in data["per_ablation"].values():
        delta_matrix[entry["layer"], entry["expert"]] = entry["delta_vs_baseline"]

    fig, ax = plt.subplots(figsize=(7.5, 7))
    vmax = max(0.05, delta_matrix.max())  # at least show some scale
    im = ax.imshow(delta_matrix, aspect="auto", cmap="Reds", vmin=0, vmax=vmax, origin="lower")
    for L in range(num_layers):
        for E in range(num_experts):
            ax.text(E, L, f"{delta_matrix[L, E]:+.3f}",
                    ha="center", va="center",
                    color="white" if delta_matrix[L, E] > vmax * 0.55 else "black",
                    fontsize=8)
    ax.set_xticks(range(num_experts))
    ax.set_xticklabels([f"E{e}" for e in range(num_experts)])
    ax.set_yticks(range(num_layers))
    ax.set_ylabel("Layer")
    ax.set_xlabel("Disabled expert")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(f"Δ validation loss vs baseline ({baseline:.3f})")
    ax.set_title(f"Expert ablation: how much does silencing each (layer, expert) hurt?\n"
                 f"Bigger number = more load-bearing  •  baseline val_loss = {baseline:.3f}")
    out_path = os.path.join(FIGS_DIR, "expert_ablation.png")
    fig.savefig(out_path)
    plt.close(fig)
    print(f"expert_ablation: → {out_path}")


if __name__ == "__main__":
    probe = load_probe()

    # Analysis (1)
    analyze_context_dependence(
        probe,
        layers_to_show=[0, 3, 6, 9, 11],
        top_n_tokens=15,
        min_count=500,
    )

    # Analysis (3) — pick a few different sentences from the probe
    plot_token_paths(probe, sentence_indices=[0, 1, 7], max_tokens=36)

    # Analysis (2) — plot if data is available
    plot_expert_ablation()

    print("\nDone.")
