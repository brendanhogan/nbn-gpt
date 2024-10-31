
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

    def __init__(self, input_embedding_size: int, dropout_rate: float) -> None:
        """
        Initialize the FeedForward module.

        Args:
            input_embedding_size (int): The size of the input embeddings.
            dropout_rate (float): The dropout rate to apply after the second linear layer.
        """
        super().__init__()
        self.feed_forward_layer = nn.Linear(input_embedding_size, 4 * input_embedding_size, bias=False)
        self.feed_forward_projection = nn.Linear(4 * input_embedding_size, input_embedding_size, bias=False)
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



class TransformerBlock(nn.Module):
    """Implements a full transformer block with multi-head attention and feed-forward layers."""

    def __init__(self, embedding_size: int, number_of_attention_heads: int, dropout_rate: float) -> None:
        """
        Initialize the TransformerBlock module.

        Args:
            embedding_size (int): The size of the input embeddings.
            number_of_attention_heads (int): The number of attention heads to use.
            dropout_rate (float): The dropout rate to apply in various components.
        """
        super().__init__()
        self.multihead_attention = MultiAttentionHead(embedding_size, number_of_attention_heads, dropout_rate)
        self.feed_forward_layer = FeedForward(embedding_size, dropout_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Perform the forward pass of the TransformerBlock module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after applying attention and feed-forward layers.
        """
        x = x + self.multihead_attention(F.rms_norm(x, (x.size(-1),)))
        x = x + self.feed_forward_layer(F.rms_norm(x, (x.size(-1),)))
        return x


class GPTModel(nn.Module):

    def __init__(self, vocab_size: int, input_embedding_size: int, context_length: int, number_of_transformer_layers: int, number_of_heads: int, dropout_rate: float) -> None:
        super().__init__()

        self.context_length = context_length

        # Setup actual transformer blocks
        self.transformer_blocks = nn.ModuleDict(dict(
            token_embedding_table = nn.Embedding(vocab_size, input_embedding_size),
            transformers = nn.ModuleList([TransformerBlock(input_embedding_size, number_of_heads, dropout_rate) for _ in range(number_of_transformer_layers)]),
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

        # Pass through transformer blocks
        for block in self.transformer_blocks.transformers:
            x = block(x)
        
        # Do final normalization
        x = F.rms_norm(x, (x.size(-1),))

        # Output depends if loss and/or logists are needed 
        if targets is not None:
            # Then we need to calcualte loss 
            logits = self.lm_head(x)
            logits = logits.float() 
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
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

def get_model(model_name: str, vocab_size: int) -> GPTModel:
    """
    Get a preconfigured GPT model based on the specified model name.

    Args:
        model_name (str): The name of the model configuration (currently only supports 'gpt2').
        vocab_size (int): The size of the vocabulary.

    Returns:
        GPTModel: A preconfigured GPT model.

    Raises:
        ValueError: If an unsupported model name is provided.
    """
    if model_name.lower() == 'gpt2small':
        return GPTModel(
            vocab_size=50304, # make rounder number
            input_embedding_size=768,
            context_length=1024,
            number_of_transformer_layers=12,
            number_of_heads=6,
            dropout_rate=0.1,
        )
    elif model_name.lower() == 'gpt2full':
        return GPTModel(
            vocab_size=50304,
            input_embedding_size=1536,
            context_length=1024,
            number_of_transformer_layers=52,
            number_of_heads=12,
            dropout_rate=0.1,
        )
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
