
"""
GPT Architecture Implementation

This module implements a GPT-style decoder-only transformer architecture with modern improvements.
Key features:
1. RMSNorm for more stable training compared to LayerNorm
2. Rotary positional embeddings (RoPE) for better relative position modeling
3. Flash attention for efficient memory usage and faster training
4. Parallel computation of attention patterns across multiple heads
5. Residual connections and dropout for regularization

The model follows a standard transformer decoder architecture with:
- Token + positional embeddings
- Multiple transformer decoder layers with:
  - RMSNorm
  - Multi-head self attention with RoPE
  - Feed-forward network
- Final RMSNorm and projection to vocabulary

Designed for efficient training and inference on modern hardware while maintaining
strong language modeling capabilities.

Largely adapted from: https://github.com/KellerJordan/modded-nanogpt
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class Rotary(torch.nn.Module):
    """
    Implements rotary positional embeddings (RoPE) for transformers.
    RoPE encodes relative positional information through rotation matrices
    applied to pairs of vectors.
    """

    def __init__(self, dim: int, base: int = 10000) -> None:
        """
        Initialize the rotary embeddings.

        Args:
            dim (int): Dimension of the embeddings (must be even)
            base (int): Base for the frequency calculations
        """
        super().__init__()
        # Calculate frequency bands - lower frequencies for higher dimensions
        self.inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        # Cache for avoiding recomputation
        self.seq_len_cached: int | None = None
        self.cos_cached: torch.Tensor | None = None  
        self.sin_cached: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute rotary embeddings for input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, seq_len, ...]

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Cosine and sine embeddings
                shaped for broadcasting with attention heads
        """
        seq_len = x.shape[1]
        if seq_len != self.seq_len_cached:
            # Only recompute if sequence length changes
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            # Outer product of positions and frequencies
            freqs = torch.outer(t, self.inv_freq).to(x.device)
            # Cache the embeddings in bfloat16 for efficiency
            self.cos_cached = freqs.cos().bfloat16()
            self.sin_cached = freqs.sin().bfloat16()
        # Add dimensions for batch and head broadcasting
        return self.cos_cached[None, :, None, :], self.sin_cached[None, :, None, :]

def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary positional embeddings to input tensor.

    Args:
        x (torch.Tensor): Input tensor of shape [batch, seq_len, heads, dim]
        cos (torch.Tensor): Cosine embeddings for rotation
        sin (torch.Tensor): Sine embeddings for rotation

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied, same shape as input
    """
    assert x.ndim == 4, "Input tensor must have 4 dimensions (batch, seq_len, heads, dim)"
    d = x.shape[3] // 2  # Split last dimension in half for rotation
    x1, x2 = x[..., :d], x[..., d:]  # Split vectors to rotate
    
    # Apply rotation using rotation matrix multiplication
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    
    return torch.cat([y1, y2], dim=3).type_as(x)


class MultiAttentionHead(nn.Module):
    """
    Implements multiple attention heads using a single forward pass and flash attention.
    
    """

    def __init__(self, embedding_size: int, number_of_attention_heads: int, dropout_rate: float) -> None:
        """
        Initialize the MultiAttentionHead module.

        Args:
            embedding_size (int): The size of the input embeddings.
            number_of_attention_heads (int): The number of attention heads to use.
            dropout_rate (float): The dropout rate to apply after the projection.
        """
        super().__init__()

        self.embedding_size = embedding_size
        self.number_of_attention_heads = number_of_attention_heads

        # Key, Query and Value layers 
        self.key_values = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.query_values = nn.Linear(self.embedding_size, self.embedding_size, bias=False)
        self.values = nn.Linear(self.embedding_size, self.embedding_size, bias=False)

        # Project the output back to the original embedding size
        self.projection = nn.Linear(embedding_size, embedding_size, bias=False)
        self.projection.weight.data.zero_() # From modded-nanogpt

        # Calculate dimension of attention heads
        self.head_dim = self.embedding_size // self.number_of_attention_heads

        # Add rotary embeddings
        self.rotary = Rotary(self.head_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the MultiAttentionHead module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, embedding_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, embedding_size).
        """
        batch_size, sequence_length, _ = x.shape

        # Compute keys, queries, and values
        key_vals = self.key_values(x).view(batch_size, sequence_length, self.number_of_attention_heads, self.head_dim)
        que_vals = self.query_values(x).view(batch_size, sequence_length, self.number_of_attention_heads, self.head_dim)
        vals = self.values(x).view(batch_size, sequence_length, self.number_of_attention_heads, self.head_dim)
        
        # Get rotary positional embeddings for the queries
        cos, sin = self.rotary(que_vals)

        # Apply RMS normalization to queries and keys for better training stability
        que_vals = F.rms_norm(que_vals, (que_vals.size(-1),))
        key_vals = F.rms_norm(key_vals, (key_vals.size(-1),))

        # Apply rotary embeddings to both queries and keys
        que_vals = apply_rotary_emb(que_vals, cos, sin)
        key_vals = apply_rotary_emb(key_vals, cos, sin)

        # Perform scaled dot-product attention with causal masking
        # Transpose to get shape (batch, heads, seq_len, head_dim)
        attention_output = F.scaled_dot_product_attention(
            que_vals.transpose(1, 2),  # queries 
            key_vals.transpose(1, 2),  # keys
            vals.transpose(1, 2),      # values
            is_causal=True             # Apply causal masking
        )

        # Reshape attention output back to original dimensions
        attention_output = attention_output.transpose(1, 2)                  # Restore sequence dimension
        attention_output = attention_output.contiguous()                     # Ensure memory is contiguous
        attention_output = attention_output.view_as(x)                       # Reshape to original input shape

        # Project to final output
        output = self.projection(attention_output)
        return output



class FeedForward(nn.Module):
    """A simple feed-forward network with a single hidden layer and non-linearity."""

    def __init__(self, input_embedding_size: int, dropout_rate: float, hidden_size: int | None = None) -> None:
        """
        Initialize the FeedForward module.

        Args:
            input_embedding_size (int): The size of the input embeddings.
            dropout_rate (float): The dropout rate to apply after the second linear layer.
            hidden_size (int | None): Size of the hidden layer. Defaults to 4 * input_embedding_size.
                Used by MixtureOfExperts to make each expert smaller than a dense FFN.
        """
        super().__init__()
        if hidden_size is None:
            hidden_size = 4 * input_embedding_size
        self.feed_forward_layer = nn.Linear(input_embedding_size, hidden_size, bias=False)
        self.feed_forward_projection = nn.Linear(hidden_size, input_embedding_size, bias=False)
        self.feed_forward_projection.weight.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the FeedForward module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = self.feed_forward_layer(x)
        x = F.relu(x).square()
        x = self.feed_forward_projection(x)
        return x



class MixtureOfExperts(nn.Module):
    """
    Drop-in replacement for FeedForward.

    Has `num_experts` small FFNs ("experts"). For each token, a tiny "router"
    network picks the top `experts_per_token` experts. Each token's output is
    a weighted blend of those experts' outputs, where the weights come from
    the router's softmax scores.

    Each expert's hidden size is (4 * embedding_size) / num_experts, so the
    total parameter count is roughly the same as a single dense FeedForward.

    Also returns an auxiliary "load balancing" loss that penalizes the router
    for sending too many tokens to too few experts. Without this, the router
    collapses to always picking the same expert.
    """

    def __init__(self, embedding_size: int, num_experts: int, experts_per_token: int) -> None:
        """
        Args:
            embedding_size (int): Model dimension (d_model).
            num_experts (int): Total number of experts (e.g. 4 or 8).
            experts_per_token (int): How many experts each token routes to (top-k, usually 2).
        """
        super().__init__()
        self.num_experts = num_experts
        self.experts_per_token = experts_per_token

        # Each expert is a small FeedForward with hidden size (4 * d) / num_experts.
        # That keeps total expert params equal to one dense FeedForward of hidden 4 * d.
        expert_hidden_size = (4 * embedding_size) // num_experts
        self.experts = nn.ModuleList([
            FeedForward(embedding_size, dropout_rate=0.0, hidden_size=expert_hidden_size)
            for _ in range(num_experts)
        ])

        # The router: a single linear layer that scores each expert for each token.
        self.router = nn.Linear(embedding_size, num_experts, bias=False)

        # For ablation experiments: if set to an integer, that expert's router logit
        # is forced to -inf so the top-k never picks it. Default None = normal behavior.
        self.disabled_expert: int | None = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x (torch.Tensor): Input of shape (batch, seq_len, embedding_size).

        Returns:
            output (torch.Tensor): Same shape as input.
            aux_loss (torch.Tensor): Scalar load balancing loss.
        """
        batch_size, seq_len, embedding_size = x.shape

        # Step 1: Score every expert for every token, then take softmax.
        router_logits = self.router(x)                              # (B, T, num_experts)
        # Ablation: mask out a disabled expert so top-k can never pick it.
        if self.disabled_expert is not None:
            router_logits = router_logits.clone()
            router_logits[..., self.disabled_expert] = float("-inf")
        router_probs = F.softmax(router_logits, dim=-1)             # (B, T, num_experts)

        # Step 2: For each token, pick the top-k experts and renormalize their weights to sum to 1.
        top_weights, top_indices = router_probs.topk(self.experts_per_token, dim=-1)
        top_weights = top_weights / top_weights.sum(dim=-1, keepdim=True)
        # top_weights and top_indices are both shape (B, T, k)

        # Step 3: Run each expert on the tokens that picked it; blend by weight.
        # Flatten batch and seq for easier indexing:
        x_flat           = x.reshape(-1, embedding_size)             # (B*T, d)
        top_indices_flat = top_indices.reshape(-1, self.experts_per_token)
        top_weights_flat = top_weights.reshape(-1, self.experts_per_token)
        output_flat      = torch.zeros_like(x_flat)

        for expert_id, expert in enumerate(self.experts):
            # Per-token weight assigned to this expert (0 if it wasn't in this token's top-k).
            expert_weight_per_token = (
                top_weights_flat * (top_indices_flat == expert_id)
            ).sum(dim=-1, keepdim=True)                              # (B*T, 1)

            # Only run the expert on tokens that actually need it.
            chosen = expert_weight_per_token.squeeze(-1) > 0         # (B*T,)
            if not chosen.any():
                continue

            expert_output = expert(x_flat[chosen])                   # (M, d)
            output_flat[chosen] = output_flat[chosen] + expert_weight_per_token[chosen] * expert_output

        output = output_flat.view(batch_size, seq_len, embedding_size)

        # Step 4: Auxiliary load balancing loss (Switch Transformer style).
        # Penalizes the router for assigning too much traffic AND probability to the same expert.
        # Minimized when both are uniform across experts.
        chosen_mass = torch.zeros_like(router_probs)                 # (B, T, num_experts)
        chosen_mass.scatter_(-1, top_indices, top_weights)
        fraction_per_expert  = chosen_mass.mean(dim=(0, 1))           # (num_experts,)
        mean_prob_per_expert = router_probs.mean(dim=(0, 1))          # (num_experts,)
        aux_loss = self.num_experts * (fraction_per_expert * mean_prob_per_expert).sum()

        # Step 5: Save stats for logging and routing decisions for offline analysis.
        # These don't affect gradients — they're just observables for us to look at.
        with torch.no_grad():
            eps = 1e-9
            # Router entropy: low = decisive picks, high = uniform picks
            router_entropy = -(router_probs * (router_probs + eps).log()).sum(dim=-1).mean()
            self.last_stats = {
                "fraction_per_expert": fraction_per_expert.detach(),
                "router_entropy": router_entropy,
                "imbalance_ratio": fraction_per_expert.max() / fraction_per_expert.clamp(min=eps).min(),
            }
            # Routing decisions for each token: which experts they picked.
            # Read by save_routing_probe() at eval time for offline analysis.
            self.last_top_indices = top_indices.detach()

        return output, aux_loss



def attn_res(history: list[torch.Tensor], pseudo_query: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Full Attention Residuals (AttnRes) from the Kimi Team paper (arxiv 2603.15031).

    Replaces the standard `x = x + f(x)` residual update with an attention-weighted
    sum over ALL previous sublayer outputs. The query is a single learnable d-vector
    per layer (initialized to zero so the attention starts as a uniform average).

    Args:
        history: list of L tensors, each shape (B, T, d). v_0 is the token embedding,
                 v_1..v_{L-1} are previous sublayer outputs (attn or ffn).
        pseudo_query: shape (d,). The per-layer learnable w_l vector.

    Returns:
        h: shape (B, T, d). The attention-weighted combination, used as input to the
           next sublayer.
        alpha: shape (L, B, T). The attention weights (saved by the caller for analysis).
    """
    # Iterative: never materialize a stacked V tensor. Each `history[i]` is already
    # saved as a sublayer output (would be saved anyway for that sublayer's own
    # backward), so the weighted sum below doesn't add new big tensors to the
    # autograd graph. This is the only version we've found that fits in 80 GB at
    # B=32 with full autograd active during training. Empirically ~12 s/step.
    eps = 1e-6
    L = len(history)
    logits_per_item = []
    for v in history:
        rms = torch.sqrt(v.float().pow(2).mean(-1) + eps).to(v.dtype)                    # (B, T)
        wv = torch.einsum('d, b t d -> b t', pseudo_query, v)                            # (B, T)
        logits_per_item.append(wv / rms)
    logits = torch.stack(logits_per_item, dim=0)                                          # (L, B, T) — tiny
    alpha = F.softmax(logits.float(), dim=0).to(history[0].dtype)                         # (L, B, T)
    h = alpha[0].unsqueeze(-1) * history[0]
    for i in range(1, L):
        h = h + alpha[i].unsqueeze(-1) * history[i]
    return h, alpha


class TransformerBlock(nn.Module):
    """Implements a full transformer block with multi-head attention and feed-forward layers."""

    def __init__(
        self,
        embedding_size: int,
        number_of_attention_heads: int,
        dropout_rate: float,
        use_moe: bool = False,
        num_experts: int = 4,
        experts_per_token: int = 2,
        use_attn_res: bool = False,
    ) -> None:
        """
        Initialize the TransformerBlock module.

        Args:
            embedding_size (int): The size of the input embeddings.
            number_of_attention_heads (int): The number of attention heads to use.
            dropout_rate (float): The dropout rate to apply in various components.
            use_moe (bool): If True, replace the FeedForward with a MixtureOfExperts.
            num_experts (int): Number of experts when use_moe is True.
            experts_per_token (int): Top-k experts each token routes to when use_moe is True.
        """
        super().__init__()
        self.multihead_attention = MultiAttentionHead(embedding_size, number_of_attention_heads, dropout_rate)
        self.use_moe = use_moe
        self.use_attn_res = use_attn_res
        if use_moe:
            self.feed_forward_layer = MixtureOfExperts(embedding_size, num_experts, experts_per_token)
        else:
            self.feed_forward_layer = FeedForward(embedding_size, dropout_rate)
        # AttnRes pseudo-queries (one before attention, one before FFN). Zero-init so
        # attention starts uniform across depth — matches the paper's recommendation.
        if use_attn_res:
            self.attn_pseudo_query = nn.Parameter(torch.zeros(embedding_size))
            self.ffn_pseudo_query  = nn.Parameter(torch.zeros(embedding_size))

    def forward(self, x: torch.Tensor, history: list[torch.Tensor] | None = None) -> tuple[torch.Tensor, torch.Tensor | float]:
        """
        Perform the forward pass of the TransformerBlock module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            output (torch.Tensor): Output tensor after attention and feed-forward layers.
            aux_loss (torch.Tensor | float): MoE load balancing loss, or 0.0 if MoE is off.
        """
        if self.use_attn_res:
            # AttnRes path: input to each sublayer comes from softmax-weighted attention
            # over the history of all previous sublayer outputs.
            assert history is not None, "use_attn_res=True requires history list"

            # --- Attention sublayer ---
            attn_in, attn_alpha = attn_res(history, self.attn_pseudo_query)
            attn_out = self.multihead_attention(F.rms_norm(attn_in, (attn_in.size(-1),)))
            history.append(attn_out)

            # --- FFN / MoE sublayer ---
            ffn_in, ffn_alpha = attn_res(history, self.ffn_pseudo_query)
            if self.use_moe:
                ff_out, aux_loss = self.feed_forward_layer(F.rms_norm(ffn_in, (ffn_in.size(-1),)))
            else:
                ff_out = self.feed_forward_layer(F.rms_norm(ffn_in, (ffn_in.size(-1),)))
                aux_loss = 0.0
            history.append(ff_out)

            # Save alpha vectors for analysis (detached, mean over batch+seq to keep size small).
            with torch.no_grad():
                self.last_attn_alpha = attn_alpha.detach().mean(dim=(1, 2))   # (L_history_at_attn,)
                self.last_ffn_alpha  = ffn_alpha.detach().mean(dim=(1, 2))     # (L_history_at_ffn,)

            # The "x" return value isn't used in the AttnRes path (GPTModel reads `history` directly),
            # but we return ff_out so the signature stays uniform.
            return ff_out, aux_loss
        else:
            # Standard residual path (unchanged behavior).
            x = x + self.multihead_attention(F.rms_norm(x, (x.size(-1),)))
            if self.use_moe:
                ff_out, aux_loss = self.feed_forward_layer(F.rms_norm(x, (x.size(-1),)))
                x = x + ff_out
                return x, aux_loss
            else:
                x = x + self.feed_forward_layer(F.rms_norm(x, (x.size(-1),)))
                return x, 0.0


class GPTModel(nn.Module):

    def __init__(
        self,
        vocab_size: int,
        input_embedding_size: int,
        context_length: int,
        number_of_transformer_layers: int,
        number_of_heads: int,
        dropout_rate: float,
        use_moe: bool = False,
        num_experts: int = 4,
        experts_per_token: int = 2,
        moe_aux_loss_weight: float = 0.01,
        use_attn_res: bool = False,
        use_adaptive_depth: bool = False,
        adaptive_depth_cost_weight: float = 0.01,
    ) -> None:
        super().__init__()
        if use_adaptive_depth and use_attn_res:
            raise ValueError("--use_adaptive_depth is not composable with --use_attn_res in this implementation")

        self.context_length = context_length
        self.use_moe = use_moe
        self.moe_aux_loss_weight = moe_aux_loss_weight
        self.use_attn_res = use_attn_res
        self.use_adaptive_depth = use_adaptive_depth
        self.adaptive_depth_cost_weight = adaptive_depth_cost_weight
        self.number_of_transformer_layers = number_of_transformer_layers

        # Setup actual transformer blocks
        self.transformer_blocks = nn.ModuleDict(dict(
            token_embedding_table = nn.Embedding(vocab_size, input_embedding_size),
            transformers = nn.ModuleList([
                TransformerBlock(
                    input_embedding_size,
                    number_of_heads,
                    dropout_rate,
                    use_moe=use_moe,
                    num_experts=num_experts,
                    experts_per_token=experts_per_token,
                    use_attn_res=use_attn_res,
                )
                for _ in range(number_of_transformer_layers)
            ]),
        ))

        # Setup final linear layer to make projection
        self.lm_head = nn.Linear(input_embedding_size, vocab_size, bias=False)

        # Share weights between first and last layer
        self.transformer_blocks.token_embedding_table.weight = self.lm_head.weight

        # Final AttnRes pseudo-query (used to aggregate all sublayer outputs before LM head).
        if use_attn_res:
            self.final_pseudo_query = nn.Parameter(torch.zeros(input_embedding_size))

        # Adaptive-depth router: per-token softmax over the L layer outputs. Reads only
        # the token embedding (no positional info / no deep features), so the router has
        # to learn "how much computation does this TOKEN TYPE deserve". Output of layer k
        # is then weighted by alpha[token, k] and summed before the LM head.
        if use_adaptive_depth:
            self.depth_router = nn.Linear(input_embedding_size, number_of_transformer_layers, bias=False)


    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, return_logits=True) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length = idx.shape
        # B, T = idx.shape

        # Get token and position embedding
        x = self.transformer_blocks.token_embedding_table(idx) # batch size x sequence length x embedding size

        # Pass through transformer blocks. Each block returns its output and an aux loss
        # (the MoE load balancing loss, or 0.0 when MoE is off).
        total_aux_loss = 0.0
        if self.use_attn_res:
            # AttnRes path: maintain a list of all sublayer outputs. Each block reads
            # from `history` and appends its two sublayer outputs (attn, ffn).
            history: list[torch.Tensor] = [x]
            for block in self.transformer_blocks.transformers:
                _, block_aux_loss = block(None, history=history)
                total_aux_loss = total_aux_loss + block_aux_loss
            # Final AttnRes over the full history before the LM head.
            x, final_alpha = attn_res(history, self.final_pseudo_query)
            with torch.no_grad():
                self.last_final_alpha = final_alpha.detach().mean(dim=(1, 2))   # (L_history,)
        else:
            # If adaptive-depth is on, predict the per-token depth distribution NOW
            # (from the raw token embedding, before any block runs) and collect every
            # layer's output as we go. Otherwise the loop is just the standard path.
            if self.use_adaptive_depth:
                depth_logits = self.depth_router(x)                                 # (B, T, L)
                depth_alpha  = F.softmax(depth_logits, dim=-1)                       # (B, T, L)
                layer_outputs: list[torch.Tensor] = []

            for block in self.transformer_blocks.transformers:
                x, block_aux_loss = block(x)
                total_aux_loss = total_aux_loss + block_aux_loss
                if self.use_adaptive_depth:
                    layer_outputs.append(x)

            if self.use_adaptive_depth:
                # Stack layer outputs and take per-token weighted sum over depth.
                stacked = torch.stack(layer_outputs, dim=2)                          # (B, T, L, d)
                x = torch.einsum('btl, btld -> btd', depth_alpha, stacked)           # (B, T, d)
                # Expected depth (1-indexed: depth=1 means "used layer 1's output").
                depths = torch.arange(1, len(layer_outputs) + 1, device=x.device, dtype=depth_alpha.dtype)
                expected_depth = (depth_alpha * depths).sum(dim=-1)                  # (B, T)
                # Stash for logging / analysis.
                with torch.no_grad():
                    self.last_depth_alpha    = depth_alpha.detach().mean(dim=(0, 1)) # (L,)
                    self.last_expected_depth = expected_depth.detach().mean()
                    self.last_depth_alpha_per_token = depth_alpha.detach()           # (B, T, L) — saved for probes

        # Do final normalization
        x = F.rms_norm(x, (x.size(-1),))

        # Output depends if loss and/or logists are needed
        if targets is not None:
            # Then we need to calcualte loss
            logits = self.lm_head(x)
            logits = logits.float()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            # If MoE is on, average aux loss across layers and add it to main loss.
            # If MoE is off, total_aux_loss is just 0.0 and this line is a no-op.
            if self.use_moe:
                loss = loss + self.moe_aux_loss_weight * (total_aux_loss / len(self.transformer_blocks.transformers))
            # Adaptive-depth cost: penalize the model for using more layers per token.
            # `expected_depth` is the mean depth (in {1..L}) the router chose across all tokens
            # in the batch. With λ=0 the model will collapse to using depth L; as λ grows
            # the model trades main_loss for cheaper "average depth".
            if self.use_adaptive_depth:
                loss = loss + self.adaptive_depth_cost_weight * expected_depth.mean()
        else:
            # Only do final layer for last token
            # logits = self.lm_head(x[:, [-1], :])
            logits = self.lm_head(x)

            logits = logits.float()
            loss = None

        if not return_logits:
            logits = None

        return logits, loss

    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -self.context_length:]
            # get the predictions
            logits, _ = self.forward(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

def get_model(
    model_name: str,
    vocab_size=None,
    use_moe: bool = False,
    num_experts: int = 4,
    experts_per_token: int = 2,
    moe_aux_loss_weight: float = 0.01,
    use_attn_res: bool = False,
    use_adaptive_depth: bool = False,
    adaptive_depth_cost_weight: float = 0.01,
) -> GPTModel:
    """
    Get a preconfigured GPT model based on the specified model name.

    Args:
        model_name (str): The name of the model configuration (currently only supports 'gpt2').
        vocab_size (int): The size of the vocabulary.
        use_moe (bool): If True, replace each FeedForward with a MixtureOfExperts.
        num_experts (int): Number of experts per MoE layer.
        experts_per_token (int): Top-k experts each token routes to.
        moe_aux_loss_weight (float): Weight for the MoE load balancing loss.

    Returns:
        GPTModel: A preconfigured GPT model.

    Raises:
        ValueError: If an unsupported model name is provided.
    """
    moe_kwargs = dict(
        use_moe=use_moe,
        num_experts=num_experts,
        experts_per_token=experts_per_token,
        moe_aux_loss_weight=moe_aux_loss_weight,
        use_attn_res=use_attn_res,
        use_adaptive_depth=use_adaptive_depth,
        adaptive_depth_cost_weight=adaptive_depth_cost_weight,
    )
    if model_name.lower() == 'gpt2small':
        return GPTModel(
            vocab_size=50304, # make rounder number
            input_embedding_size=768,
            context_length=1024,
            number_of_transformer_layers=12,
            number_of_heads=6,
            dropout_rate=0.1,
            **moe_kwargs,
        )
    elif model_name.lower() == 'gpt2full':
        return GPTModel(
            vocab_size=50304,
            input_embedding_size=1536,
            context_length=1024,
            number_of_transformer_layers=52,
            number_of_heads=12,
            dropout_rate=0.1,
            **moe_kwargs,
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
