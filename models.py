
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
        if use_moe:
            self.feed_forward_layer = MixtureOfExperts(embedding_size, num_experts, experts_per_token)
        else:
            self.feed_forward_layer = FeedForward(embedding_size, dropout_rate)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | float]:
        """
        Perform the forward pass of the TransformerBlock module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            output (torch.Tensor): Output tensor after attention and feed-forward layers.
            aux_loss (torch.Tensor | float): MoE load balancing loss, or 0.0 if MoE is off.
        """
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
    ) -> None:
        super().__init__()

        self.context_length = context_length
        self.use_moe = use_moe
        self.moe_aux_loss_weight = moe_aux_loss_weight

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
                )
                for _ in range(number_of_transformer_layers)
            ]),
        ))

        # Setup final linear layer to make projection
        self.lm_head = nn.Linear(input_embedding_size, vocab_size, bias=False)

        # Share weights between first and last layer
        self.transformer_blocks.token_embedding_table.weight = self.lm_head.weight


    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None, return_logits=True) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length = idx.shape
        # B, T = idx.shape

        # Get token and position embedding
        x = self.transformer_blocks.token_embedding_table(idx) # batch size x sequence length x embedding size

        # Pass through transformer blocks. Each block returns its output and an aux loss
        # (the MoE load balancing loss, or 0.0 when MoE is off).
        total_aux_loss = 0.0
        for block in self.transformer_blocks.transformers:
            x, block_aux_loss = block(x)
            total_aux_loss = total_aux_loss + block_aux_loss

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
