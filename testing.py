"""

Before building the full scale GPT2-ish remake, I want to have (like kaparthy) a relatively simpler implementation, 
to better get a grasp on the fundmenetals. 

For the dataset - I will use the creepypasta dataset (which will later be used for finetuning). Tokenization will be by the 
character level, and I will build a basic transformer architecture. 

Things probably arent, or almost definitely are not, optimzied, this is just to explore and inital training. 



Very much taken directly form, just re-written somtimes in things that make more sense to me, 
more just working though 
https://github.com/karpathy/ng-video-lecture

"""
import random 
import argparse
from tqdm import tqdm


import torch
import torch.nn as nn
from torch.nn import functional as F

class TxtFileDatasetOriginal(): 

    def __init__(self, txt_file_pth):
        # Open text file 
        with open(txt_file_pth, 'r', encoding='utf-8') as file:
            self.text = file.read()

        # Get all chars - which will serve as tokens 
        self.chars = sorted(list(set(self.text)))
        # Get number of tokens 
        self.vocab_size = len(self.chars)

        # Build dictionary index -> token, and token -> index 
        self.index_to_token = {ind: token for ind, token in enumerate(self.chars)}
        self.token_to_index = {token: ind for ind, token in enumerate(self.chars)}

    def encode(self, input):
        """Give a string, return a tokenized version of that string"""
        return [self.token_to_index[char] for char in input]
    
    def decode(self, indices):
        """Given a list of indices, return the corresponding string"""
        return ''.join([self.index_to_token[idx] for idx in indices])

    def dataset_statistics(self):
        """Calculate and print statistics about the dataset."""
        print("Dataset Statistics:")
        print(f"Total characters: {len(self.text)}")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Unique characters: {''.join(self.chars)}")
        
        # Calculate character frequency
        char_freq = {}
        for char in self.text:
            char_freq[char] = char_freq.get(char, 0) + 1
        
        # Find most and least common characters
        most_common = max(char_freq, key=char_freq.get)
        least_common = min(char_freq, key=char_freq.get)
        
        print(f"Most common character: '{most_common}' (occurs {char_freq[most_common]} times)")
        print(f"Least common character: '{least_common}' (occurs {char_freq[least_common]} times)")
        
        # Calculate average word length (assuming words are separated by spaces)
        words = self.text.split()
        avg_word_length = sum(len(word) for word in words) / len(words)
        print(f"Average word length: {avg_word_length:.2f} characters")
        
        # Print a sample of the text
        print("\nSample of the text:")
        print(self.text[:200] + "...")

class TxtFileDatasetFull(): 

    def __init__(self, txt_file_pth, split='train',ratio=.9):
        # Open text file 
        with open("data/"+txt_file_pth, 'r', encoding='utf-8') as file:
            self.text = file.read()

        # Get all chars - which will serve as tokens 
        self.chars = sorted(list(set(self.text)))
        # Get number of tokens 
        self.vocab_size = len(self.chars)

        # Build dictionary index -> token, and token -> index 
        self.index_to_token = {ind: token for ind, token in enumerate(self.chars)}
        self.token_to_index = {token: ind for ind, token in enumerate(self.chars)}

        # Split the text into train and test sets based on the ratio
        split_index = int(len(self.text) * ratio)
        if split == 'train':
            self.text = self.text[:split_index]
        elif split == 'test':
            self.text = self.text[split_index:]
        else:
            raise ValueError("Split must be either 'train' or 'test'")

        # Encode the entire text, and make tensor 
        self.encoded_text = self.encode(self.text)
        self.encoded_text = torch.tensor(self.encoded_text, dtype=torch.long)

    def get_batch(self, batch_size, block_size, device="cpu"):
        """
        Generate a small batch of data of inputs x and targets y.
        
        Args:
        - batch_size (int): Number of sequences in the batch
        - block_size (int): Length of each sequence
        
        Returns:
        - x (list): List of input sequences
        - y (list): List of target sequences
        """
        x, y = [], []
        for _ in range(batch_size):
            random_indices = torch.randint(len(self.encoded_text) - block_size-1, (batch_size,))
            x = torch.stack([self.encoded_text[i:i+block_size] for i in random_indices])
            y = torch.stack([self.encoded_text[i+1:i+block_size+1] for i in random_indices])
        return x.to(device), y.to(device)

    def encode(self, input):
        """Give a string, return a tokenized version of that string"""
        return [self.token_to_index[char] for char in input]
    
    def decode(self, indices):
        """Given a list of indices, return the corresponding string"""
        return ''.join([self.index_to_token[idx] for idx in indices])

    def dataset_statistics(self):
        """Calculate and print statistics about the dataset."""
        print("Dataset Statistics:")
        print(f"Total characters: {len(self.text)}")
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Unique characters: {''.join(self.chars)}")
        
        # Calculate character frequency
        char_freq = {}
        for char in self.text:
            char_freq[char] = char_freq.get(char, 0) + 1
        
        # Find most and least common characters
        most_common = max(char_freq, key=char_freq.get)
        least_common = min(char_freq, key=char_freq.get)
        
        print(f"Most common character: '{most_common}' (occurs {char_freq[most_common]} times)")
        print(f"Least common character: '{least_common}' (occurs {char_freq[least_common]} times)")
        
        # Calculate average word length (assuming words are separated by spaces)
        words = self.text.split()
        avg_word_length = sum(len(word) for word in words) / len(words)
        print(f"Average word length: {avg_word_length:.2f} characters")
        
        # Print a sample of the text
        print("\nSample of the text:")
        print(self.text[:200] + "...")


class BigramLanguageModel(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx) # (B,T,C)

        if targets is None:
            loss = None
        else:
            batches, time_component, output_characters = logits.shape
            logits = logits.view(batches*time_component, output_characters) # Flatten out 
            targets = targets.view(batches*time_component)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self.forward(idx)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx



def testing_dataset(): 
     # 1. Build dataset and print stats 
    ds = TxtFileDatasetOriginal("data/all_stories.txt")
    ds.dataset_statistics()

    # 2. Test encode/decode 
    test_string = "Creepypasta dataset"
    encoded = ds.encode(test_string)
    decoded = ds.decode(encoded)
    print(f"Original string: {test_string}")
    print(f"Encoded: {encoded}")
    print(f"Decoded: {decoded}")
    print(f"Original and decoded match: {test_string == decoded}")
   
def testing_bigram(): 
     # 1. Build dataset and print stats 
    ds = TxtFileDatasetFull("all_stories.txt")

    # 2. Gt a batch 
    x_batch, y_batch = ds.get_batch(2, 8) 

    # 3. Do sample input/output
    m = BigramLanguageModel(ds.vocab_size)
    logits, loss = m(x_batch, y_batch)

    print("Untrained")
    print(logits.shape)
    print(loss)

    print(ds.decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

    # 4. Now train 
    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

    for steps in tqdm(range(10000), desc="Training", unit="step"):

        # sample a batch of data
        xb, yb = ds.get_batch(4, 8)

        # evaluate the loss
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("Trained")
    print(loss.item())
    print(ds.decode(m.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))


class SelfAttentionHead(nn.Module):
    """ one head of self-attention """

    def __init__(self, input_embedding_size: int, output_embedding_size:int , context_length:int, dropout=.1):
        """
        Single attention head, go from input_size (some embedding size) to some other embedding size (head size) 

        The keys will be the search value a token has to others 
        The queries will be what a token 'looks' for in other token 
        The values will be what value a token has to others


        The triangle buffer ensures that attentions can only look backwards.      

        input_embeddings_size - input size of input 
        output_embedding_size - output embedding size, 
        context_length - how many tokens the model can handle at once (aka block_size)
        
        """

        super().__init__()
        self.key = nn.Linear(input_embedding_size, output_embedding_size, bias=False)
        self.query = nn.Linear(input_embedding_size, output_embedding_size, bias=False)
        self.value = nn.Linear(input_embedding_size, output_embedding_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_length, context_length)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # B,T,C = x.shape
        batch_size, number_of_time_points, embedding_size = x.shape

        # Compute key, quey and values for all input tokens 
        key_value_of_tokens = self.key(x)   # (B,T,C)
        query_value_of_tokens = self.query(x) # (B,T,C)
        conceptual_value_of_tokens = self.value(x) # (B,T,C)

        # Compute attention with direct dot product - and normalize with 1/sqrt(input size) 
        computed_attention_mask = query_value_of_tokens @ key_value_of_tokens.transpose(-2,-1) * embedding_size**-0.5
        
        # FIll in - up to times points incase less than full context length 
        # Then set all 0 elements to -inf, for better softmax probabilities 
        computed_attention_mask = computed_attention_mask.masked_fill(self.tril[:number_of_time_points, :number_of_time_points] == 0, float('-inf'))

        # Now perform softmax - so that sum=1
        computed_attention_mask = F.softmax(computed_attention_mask, dim=-1) # (B, T, T)

        # Apply drop out 
        computed_attention_mask = self.dropout(computed_attention_mask)

        # Now perform weighted average of values, based off attention- ensures that all tokens can only attend to 
        # previous ones, so can use loss for all tokens 
        # Intutition - high dot product of key and query means more beneficial - so lets pass more of that value 
        # and vice versa 
        # At each time point can only sum of ones before it, so can use loss over all tokens 
        weighted_combine_values = computed_attention_mask @ conceptual_value_of_tokens

        return weighted_combine_values

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """

    def __init__(self, input_embedding_size, number_of_heads, head_embedding_size, context_length, dropout_rate, atsp):
        """
        Run number_of_heads self attention heads. Each embedding head has input size, 
        input_embedding size and output head_embedding size. 
        
        I mistakenly thought the input head size was divided up but I dont think it is. That might 
        be interesting 
        
        """
        super().__init__()
        self.atsp = atsp
        if self.atsp:
            self.heads = nn.ModuleList([SelfAttentionHead(head_embedding_size,head_embedding_size, context_length, dropout_rate) for _ in range(number_of_heads)])
        else:
            self.heads = nn.ModuleList([SelfAttentionHead(input_embedding_size, head_embedding_size, context_length, dropout_rate) for _ in range(number_of_heads)])
        self.proj = nn.Linear(input_embedding_size, input_embedding_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.dropout_rate =dropout_rate

    def forward(self, x):
        if self.atsp:
            # Calculate the size of each chunk
            chunk_size = x.shape[-1] // len(self.heads)
            head_outputs = []
            
            # Check if the model is in evaluation mode
            is_eval_mode = not self.training
            
            # Iterate through the channel dimension
            active_heads = 0
            head_outputs = []
            for i, head in enumerate(self.heads):
                # Extract the chunk for this head
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size
                x_chunk = x[..., start_idx:end_idx]
                
                # Pass the chunk through the head
                head_output = head(x_chunk)
                head_outputs.append(head_output)
            
            # Concatenate all head outputs
            out = torch.cat(head_outputs, dim=-1)
        else:
            out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

    # def forward(self, x):
    #     if self.atsp:
    #         # Calculate the size of each chunk
    #         chunk_size = x.shape[-1] // len(self.heads)
    #         head_outputs = []
            
    #         # Check if the model is in evaluation mode
    #         is_eval_mode = not self.training
            
    #         # Iterate through the channel dimension
    #         active_heads = 0
    #         for i, head in enumerate(self.heads):
    #             # If in eval mode, use all heads. Otherwise, apply dropout
    #             if is_eval_mode or (i == len(self.heads) - 1 and active_heads == 0) or random.random() >= 0.6:
    #                 # Extract the chunk for this head
    #                 start_idx = i * chunk_size
    #                 end_idx = (i + 1) * chunk_size
    #                 x_chunk = x[..., start_idx:end_idx]
                    
    #                 # Pass the chunk through the head
    #                 head_output = head(x_chunk)
    #                 if active_heads == 0:
    #                     accumulated_output = head_output
    #                 else:
    #                     accumulated_output = accumulated_output + head_output
                    
    #                 active_heads += 1
            
    #         # Ensure at least one head is active (this should always be true now)
    #         assert active_heads > 0, "At least one head should always be active"
            
    #         out = accumulated_output / active_heads
    #     else:
    #         out = torch.cat([h(x) for h in self.heads], dim=-1)
    #     out = self.dropout(self.proj(out))
    #     return out    

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, input_embedding_size, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_embedding_size, 4 * input_embedding_size),
            nn.ReLU(),
            nn.Linear(4 * input_embedding_size, input_embedding_size),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    """ Transformer block: communication followed by computation """

    def __init__(self, input_embedding_size, number_of_heads, context_length, dropout_rate, atsp):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_embedding_size = input_embedding_size // number_of_heads
        self.multi_head_self_attention = MultiHeadAttention(input_embedding_size, number_of_heads, head_embedding_size, context_length,  dropout_rate, atsp)
        self.feed_forward_layer = FeedFoward(input_embedding_size, dropout_rate)

        self.ln1 = nn.LayerNorm(input_embedding_size)
        self.ln2 = nn.LayerNorm(input_embedding_size)

    def forward(self, x):
        x = x + self.multi_head_self_attention(self.ln1(x))
        x = x + self.feed_forward_layer(self.ln2(x))
        return x


class GPTModel(nn.Module):

    def __init__(self, vocab_size, input_embedding_size, context_length, number_of_transformer_layers, number_of_heads, dropout_rate, device, atsp=False):
        super().__init__()

        # Setup look up table for token and positional embeddings 
        self.token_embedding_table = nn.Embedding(vocab_size, input_embedding_size)
        self.position_embedding_table = nn.Embedding(context_length, input_embedding_size)

        # Setup actual transformer blocks
        self.context_length = context_length
        self.transformer_blocks = nn.Sequential(*[TransformerBlock(input_embedding_size, number_of_heads, context_length, dropout_rate, atsp) for _ in range(number_of_transformer_layers)])

        # Setup last layer norm 
        self.final_layer_norm = nn.LayerNorm(input_embedding_size) # final layer norm

        # Setup final linear layer to make projection 
        self.lm_head = nn.Linear(input_embedding_size, vocab_size)

        self.device = device

    def forward(self, idx, targets=None):
        batch_size, sequence_length = idx.shape
        # B, T = idx.shape

        # Get token and position embedding 
        tok_emb = self.token_embedding_table(idx) # batch size x sequence length x embedding size 
        pos_emb = self.position_embedding_table(torch.arange(sequence_length, device=self.device)) # sequence length x embedding size

        # Combine the embeddings 
        x = tok_emb + pos_emb # batch size x sequence length x embedding size 

        # Pass through transformer blocks
        x = self.transformer_blocks(x) # batch size x sequence length x embedding size 

        # Do last layer normalization, and predict logits
        x = self.final_layer_norm(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
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


class GPTModelFlat(nn.Module):

    def __init__(self, vocab_size, input_embedding_size, context_length, number_of_transformer_layers, number_of_heads, dropout_rate, device, atsp=False):
        super().__init__()

        # Setup look up table for token and positional embeddings 
        self.token_embedding_table = nn.Embedding(vocab_size, input_embedding_size)
        self.position_embedding_table = nn.Embedding(context_length, input_embedding_size)

        # Setup actual transformer blocks
        self.context_length = context_length
        self.transformer_blocks = nn.ModuleList([TransformerBlock(input_embedding_size, number_of_heads, context_length, dropout_rate, False) for _ in range(number_of_transformer_layers)])
        self.number_of_transformer_layers = number_of_transformer_layers

        # Setup last layer norm 
        self.final_layer_norm = nn.LayerNorm(input_embedding_size) # final layer norm

        # Setup final linear layer to make projection 
        self.lm_head = nn.Linear(input_embedding_size, vocab_size)

        self.device = device

    def forward(self, idx, targets=None):
        batch_size, sequence_length = idx.shape
        # B, T = idx.shape

        # Get token and position embedding 
        tok_emb = self.token_embedding_table(idx) # batch size x sequence length x embedding size 
        pos_emb = self.position_embedding_table(torch.arange(sequence_length, device=self.device)) # sequence length x embedding size

        # Combine the embeddings 
        x = tok_emb + pos_emb # batch size x sequence length x embedding size 

        # Pass through transformer blocks
        transformer_outputs = []
        for i in range(self.number_of_transformer_layers):
            transformer_output = self.transformer_blocks[i](x)
            transformer_outputs.append(transformer_output)
        
        # Combine all outputs
        x = torch.stack(transformer_outputs, dim=0).mean(dim=0)  # Average across transformer layers

        # Do last layer normalization, and predict logits
        x = self.final_layer_norm(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
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

@torch.no_grad()
def estimate_loss(model, eval_iters, get_batch, batch_size, block_size, device):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(batch_size, block_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

def testing_gpt(): 
     # 1. Build dataset and print stats 
    ds = TxtFileDatasetFull("all_stories.txt")
    ds_test = TxtFileDatasetFull("all_stories.txt",split="test")

    # 2. Setup network 
    vocab_size = ds.vocab_size
    input_embedding_size = 64 
    context_length = 32 
    number_of_transformer_layers = 4 
    number_of_heads = 4
    dropout_rate = 0.0 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device {device}")
    max_iters = 5000
    eval_iters = 200
    eval_interval = 100
    learning_rate = 1e-3
    batch_size = 16 
    model = GPTModel(vocab_size, input_embedding_size, context_length, number_of_transformer_layers, number_of_heads, dropout_rate, device)
    model.to(device)
    torch.manual_seed(1994)

    # 4. Now train 
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, eval_iters, ds_test.get_batch, batch_size, context_length, device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = ds.get_batch(batch_size, context_length, device)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(ds.decode(model.generate(context, max_new_tokens=2000)[0].tolist()))


def testing_gpt_atsp(): 
     # 1. Build dataset and print stats 
    ds = TxtFileDatasetFull("all_stories.txt")
    ds_test = TxtFileDatasetFull("all_stories.txt",split="test")

    # 2. Setup network 
    vocab_size = ds.vocab_size
    input_embedding_size = 64 
    context_length = 32 
    number_of_transformer_layers = 4 
    number_of_heads = 4
    dropout_rate = 0.0 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device {device}")
    max_iters = 5000
    eval_iters = 200
    eval_interval = 100
    learning_rate = 1e-3
    batch_size = 16 
    model = GPTModel(vocab_size, input_embedding_size, context_length, number_of_transformer_layers, number_of_heads, dropout_rate, device, atsp=True)
    model.to(device)
    torch.manual_seed(1994)

    # 4. Now train 
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, eval_iters, ds_test.get_batch, batch_size, context_length, device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = ds.get_batch(batch_size, context_length, device)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(ds.decode(model.generate(context, max_new_tokens=2000)[0].tolist()))


def testing_gpt_flat(): 
     # 1. Build dataset and print stats 
    ds = TxtFileDatasetFull("all_stories.txt")
    ds_test = TxtFileDatasetFull("all_stories.txt",split="test")

    # 2. Setup network 
    vocab_size = ds.vocab_size
    input_embedding_size = 64 
    context_length = 32 
    number_of_transformer_layers = 4 
    number_of_heads = 4
    dropout_rate = 0.0 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device {device}")
    max_iters = 5000
    eval_iters = 200
    eval_interval = 100
    learning_rate = 1e-3
    batch_size = 16 
    model = GPTModelFlat(vocab_size, input_embedding_size, context_length, number_of_transformer_layers, number_of_heads, dropout_rate, device, atsp=False)
    model.to(device)
    torch.manual_seed(1994)

    # 4. Now train 
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    # create a PyTorch optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    for iter in range(max_iters):
        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, eval_iters, ds_test.get_batch, batch_size, context_length, device)
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

        # sample a batch of data
        xb, yb = ds.get_batch(batch_size, context_length, device)

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    # generate from the model
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(ds.decode(model.generate(context, max_new_tokens=2000)[0].tolist()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the dataset functionality")
    parser.add_argument("--test_original_dataset", action="store_true", help="Run tests on the original dataset")
    parser.add_argument("--bigram_model", action="store_true", help="Run tests on the bigram model")
    parser.add_argument("--gpt_model", action="store_true", help="Run tests on the bigram model")
    parser.add_argument("--at_split", action="store_true", help="Run tests on the bigram model")
    parser.add_argument("--gpt_flat", action="store_true", help="Run tests on the bigram model")
    args = parser.parse_args()

    if args.test_original_dataset:
        testing_dataset()

    if args.bigram_model:
        testing_bigram()
    
    if args.gpt_model:
        testing_gpt()

    if args.at_split:
        testing_gpt_atsp()

    if args.gpt_flat:
        testing_gpt_flat()






























#