"""
Offline analysis of MoE routing decisions.

Loads a `routing_probe.pt` file (saved by save_routing_probe in utils.py during
training) and prints summary statistics about which experts the router picked,
broken down by layer and by token.

Usage:
    python analyze_routing.py output/step_500/routing_probe.pt
    python analyze_routing.py output/step_500/routing_probe.pt --top_n_tokens 30
    python analyze_routing.py output/step_500/routing_probe.pt --sample_paragraph
"""

import argparse
import torch
from collections import Counter

import tokenizers


def load_probe(probe_path: str) -> dict:
    """Load a routing_probe.pt file. Returns the raw saved dict."""
    return torch.load(probe_path, map_location="cpu", weights_only=False)


def print_per_layer_utilization(routing: torch.Tensor, num_experts: int) -> None:
    """
    For each layer, print the fraction of token-slots that went to each expert.
    A perfectly balanced router shows 1/num_experts in every column.
    """
    print("\n--- Per-layer expert utilization (fraction of token-slots) ---")
    header = "layer  " + "  ".join(f"E{e:<5d}" for e in range(num_experts))
    print(header)
    num_layers = routing.shape[0]
    for L in range(num_layers):
        counts = torch.bincount(routing[L].reshape(-1), minlength=num_experts).float()
        fractions = counts / counts.sum()
        row = f"{L:5d}  " + "  ".join(f"{f.item():.3f} " for f in fractions)
        print(row)


def print_top_tokens_per_expert(
    routing: torch.Tensor,
    token_ids: torch.Tensor,
    num_experts: int,
    tokenizer: tokenizers.AbstractTokenizer,
    top_n: int,
) -> None:
    """
    For each (layer, expert), list the most common tokens routed to it.
    Counts each (token, slot) pair as one occurrence — so a token routing to
    expert E in 2 of its top-k slots counts twice.
    """
    num_layers, batch_size, seq_len, k = routing.shape
    # Broadcast token ids to (L, B, T, k) so we can mask per slot.
    token_ids_per_slot = token_ids.unsqueeze(0).unsqueeze(-1).expand(num_layers, -1, -1, k)

    print(f"\n--- Top {top_n} tokens routed to each expert (per layer) ---")
    for L in range(num_layers):
        print(f"\n[Layer {L}]")
        for e in range(num_experts):
            mask = (routing[L] == e)                                    # (B, T, k)
            tokens_for_expert = token_ids_per_slot[L][mask]             # (M,)
            if tokens_for_expert.numel() == 0:
                print(f"  E{e}: (no tokens routed here)")
                continue
            counter = Counter(tokens_for_expert.tolist())
            top = counter.most_common(top_n)
            preview = ", ".join(f"{repr(tokenizer.decode([tid]))}×{c}" for tid, c in top)
            print(f"  E{e}: {preview}")


def print_sample_paragraph(
    routing: torch.Tensor,
    token_ids: torch.Tensor,
    tokenizer: tokenizers.AbstractTokenizer,
) -> None:
    """
    Print one sequence of tokens, colored by the top-1 expert each token
    chose at layer 0 and at the last layer. Useful for spotting specialization.
    """
    num_layers = routing.shape[0]
    seq_token_ids       = token_ids[0]              # (T,)
    routing_first_layer = routing[0,  0, :, 0]      # (T,) top-1 expert per token at layer 0
    routing_last_layer  = routing[-1, 0, :, 0]

    # ANSI foreground colors. Cycle through if more experts than colors.
    color_codes = [31, 32, 33, 34, 35, 36, 91, 92, 93, 94, 95, 96]

    def colorize(text: str, expert: int) -> str:
        c = color_codes[expert % len(color_codes)]
        return f"\x1b[{c}m{text}\x1b[0m"

    decoded_tokens = [tokenizer.decode([t.item()]) for t in seq_token_ids]

    print("\n--- Sample paragraph routing (top-1 expert per token, color-coded) ---")
    print(f"\nLayer 0:")
    print("".join(colorize(s, e.item()) for s, e in zip(decoded_tokens, routing_first_layer)))
    print(f"\nLayer {num_layers - 1}:")
    print("".join(colorize(s, e.item()) for s, e in zip(decoded_tokens, routing_last_layer)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("probe_path", type=str, help="Path to routing_probe.pt")
    parser.add_argument("--tokenizer", type=str, default="gpt2", help="Tokenizer to decode token ids")
    parser.add_argument("--top_n_tokens", type=int, default=20, help="Top tokens per expert to show")
    parser.add_argument("--sample_paragraph", action="store_true", help="Print a colored per-token routing for one paragraph")
    args = parser.parse_args()

    probe = load_probe(args.probe_path)
    routing     = probe["routing_top_indices"].long()   # (num_layers, B, T, k)
    token_ids   = probe["token_ids"].long()             # (B, T)
    num_experts = probe["num_experts"]
    num_layers  = probe["num_layers"]
    k           = probe["experts_per_token"]
    step        = probe["step"]

    tokenizer = tokenizers.get_tokenizer(args.tokenizer)

    print(f"\n=== Routing probe summary (step {step}) ===")
    print(f"Layers: {num_layers}  |  Experts: {num_experts}  |  Top-k: {k}")
    print(f"Probe size: batch={token_ids.shape[0]} seq_len={token_ids.shape[1]}  ({token_ids.numel():,} tokens)")

    print_per_layer_utilization(routing, num_experts)
    print_top_tokens_per_expert(routing, token_ids, num_experts, tokenizer, args.top_n_tokens)
    if args.sample_paragraph:
        print_sample_paragraph(routing, token_ids, tokenizer)


if __name__ == "__main__":
    main()
