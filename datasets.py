"""
Data processing and loading utilities for all datasets, supporting both training and inference.
"""

import os
import glob
import torch
import numpy as np
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod

import tokenizers


class AbstractDataLoader(ABC):
    """
    Abstract base class for data loaders.

    This class defines the interface that all data loaders should implement.
    It provides a common structure for data loading operations across different datasets.
    """

    @abstractmethod
    def __init__(self, tokenizer: tokenizers.AbstractTokenizer, batch_size: int, sequence_length: int, process_rank: int, num_processes: int, split: str):
        """
        Initialize the data loader.

        Args:
            tokenizer (tokenizers.AbstractTokenizer): Tokenizer for encoding and decoding text.
            batch_size (int): Number of sequences per batch.
            sequence_length (int): Length of each sequence.
            process_rank (int): Rank of the current process in distributed setting.
            num_processes (int): Total number of processes.
            split (str): Data split, e.g., 'train' or 'val'.
        """
        pass


    @abstractmethod
    def reset(self):
        """
        Reset the data loader to its initial state.
        """
        pass
    @abstractmethod
    def advance(self): 
        """
        This will be called when data needs to loop around (new epoch). 

        You can handle it here (like in the FineWeb example below) or just
        in the get_batch (like in the creepypasta example). 
        
        """
        pass 

    @abstractmethod
    def __len__(self): 
        """
        Return length of dataset in tokens. 
        """

    @abstractmethod
    def get_vocab_size(self) -> int:
        """
        Get the vocabulary size.

        Returns:
            int: Size of the vocabulary.
        """
        pass

    @abstractmethod
    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the next batch of data.

        Data is expected to be tokenized at this point. 

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing input sequences and target sequences.
        """
        pass


class CreepyPastaDataLoader(AbstractDataLoader):
    """
    DataLoader for the CreepyPasta dataset.

    This class handles loading and batching of the CreepyPasta dataset, which is small enough
    to be loaded entirely into memory. It supports distributed processing and data splitting.
    """

    def __init__(self, tokenizer: tokenizers.AbstractTokenizer, batch_size: int, sequence_length: int, process_rank: int, num_processes: int, split: str, split_ratio: float):
        """
        Initialize the CreepyPastaDataLoader.

        Args:
            tokenizer (tokenizers.AbstractTokenizer): Tokenizer that can encode and decode any arbitrary string. 
            batch_size (int): Number of sequences per batch.
            sequence_length (int): Length of each sequence.
            process_rank (int): Rank of the current process in distributed setting.
            num_processes (int): Total number of processes.
            split (str): Data split, either 'train' or 'val'.
        """
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.process_rank = process_rank
        self.num_processes = num_processes
        if split not in {'train', 'val'}:
            raise ValueError("Split must be either 'train' or 'val'")
        self.split = split

        self.data_root = "data/all_stories.txt"
        self.encoded_text = self._load_and_encode_data()
        if self.split == 'train':
            self.encoded_text = self.encoded_text[:int(len(self.encoded_text) * split_ratio)]
        else:  # 'val'
            self.encoded_text = self.encoded_text[int(len(self.encoded_text) * split_ratio):]
        print(f"Number of tokens in {self.split} split: {len(self.encoded_text)}")
        self.reset()

    def _load_and_encode_data(self) -> torch.Tensor:
        """
        Load and encode the entire dataset.

        Returns:
            torch.Tensor: Encoded dataset as a tensor of token indices.
        """
        with open(self.data_root, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Create character-level vocabulary
        # Use the tokenizer to encode the text
        encoded = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        self.vocab_size = self.tokenizer.vocab_size
        return encoded

    def advance(self):
        """Dont actually need to implement this here, handled in get batch"""
        pass

    def reset(self):
        """Reset the data loader to the beginning of the dataset."""
        self.current_position = self.batch_size * self.sequence_length * self.process_rank

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the next batch of data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input sequences and target sequences.
        """
        buf = self.encoded_text[self.current_position : self.current_position + self.batch_size*self.sequence_length + 1]
        x = buf[:-1].view(self.batch_size, self.sequence_length)  # inputs
        y = buf[1:].view(self.batch_size, self.sequence_length)   # targets
        
        # Advance the position
        self.current_position += self.batch_size * self.sequence_length * self.num_processes
        
        # Wrap around if we've reached the end of the dataset
        if self.current_position + (self.batch_size * self.sequence_length * self.num_processes + 1) > len(self.encoded_text):
            self.current_position = self.batch_size * self.sequence_length * self.process_rank
        
        return x.cuda(), y.cuda()

    def __len__(self) -> int:
        """
        Get the total number of batches in the dataset.

        Returns:
            int: Number of batches.
        """
        return len(self.encoded_text) // (self.batch_size * self.sequence_length)

    def get_vocab_size(self) -> int:
        """
        Get the vocabulary size.

        Returns:
            int: Size of the vocabulary.
        """
        return self.vocab_size

class RLHFDataset(AbstractDataLoader):
    """
    A data loader for RLHF training that loads text data from a file.
    """

    def __init__(self, tokenizer: tokenizers.AbstractTokenizer, text_file: str,
                 batch_size: int, sequence_length: int, process_rank: int = 0, 
                 num_processes: int = 1, split: str = 'train', split_ratio: float = 0.9) -> None:
        """
        Initialize the RLHF dataset loader.

        Args:
            tokenizer (AbstractTokenizer): Tokenizer instance used for encoding text
            text_file (str): Path to text file containing the data
            batch_size (int): Number of sequences per batch
            sequence_length (int): Length of each sequence
            process_rank (int): Rank of current process in distributed setting
            num_processes (int): Total number of distributed processes
            split (str): Data split, either 'train' or 'val'
            split_ratio (float): Ratio to split data between train and validation
        """
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.vocab_size = tokenizer.vocab_size
        if split not in {'train', 'val'}:
            raise ValueError("Split must be either 'train' or 'val'")
        self.split = split

        # Load and encode text data
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        encoded = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)
        
        # Split data according to split ratio
        if self.split == 'train':
            self.encoded_text = encoded[:int(len(encoded) * split_ratio)]
        else:  # 'val'
            self.encoded_text = encoded[int(len(encoded) * split_ratio):]
            
        print(f"Number of tokens in {self.split} split: {len(self.encoded_text)}")
        
        # Ensure we have enough tokens for at least one batch
        assert len(self.encoded_text) >= num_processes * batch_size * sequence_length + 1, \
               "Text file too small for batch configuration"
        
        self.reset()

    def advance(self):
        """Don't actually need to implement this here, handled in get_batch"""
        pass

    def reset(self):
        """Reset the data loader to the beginning of the dataset."""
        self.current_position = self.batch_size * self.sequence_length * self.process_rank

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the next batch of data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Input sequences and target sequences.
        """
        buf = self.encoded_text[self.current_position : self.current_position + self.batch_size*self.sequence_length + 1]
        x = buf[:-1].view(self.batch_size, self.sequence_length)  # inputs
        y = buf[1:].view(self.batch_size, self.sequence_length)   # targets
        
        # Advance the position
        self.current_position += self.batch_size * self.sequence_length * self.num_processes
        
        # Wrap around if we've reached the end of the dataset
        if self.current_position + (self.batch_size * self.sequence_length * self.num_processes + 1) > len(self.encoded_text):
            self.current_position = self.batch_size * self.sequence_length * self.process_rank
        
        return x.cuda(), y.cuda()

    def __len__(self) -> int:
        """
        Get the total number of batches in the dataset.

        Returns:
            int: Number of batches.
        """
        return len(self.encoded_text) // (self.batch_size * self.sequence_length)

    def get_vocab_size(self) -> int:
        """
        Get the vocabulary size.

        Returns:
            int: Size of the vocabulary.
        """
        return self.vocab_size

class CachedFineweb(AbstractDataLoader):
    """
    A data loader for the Fineweb dataset that loads pre-tokenized data from binary files.

    This loader is optimized for distributed training by loading data in shards and
    advancing through them in a coordinated way across processes. It expects data
    files in a specific binary format with headers containing metadata.

    Adapted from: https://github.com/KellerJordan/modded-nanogpt/blob/master/train_gpt2.py
    with modifications to fit the dataloader interface of this codebase.
    """

    def __init__(self, tokenizer: tokenizers.AbstractTokenizer, filename_pattern: str, 
                 batch_size: int, sequence_length: int, process_rank: int, num_processes: int) -> None:
        """
        Initialize the CachedFineweb data loader.

        Args:
            tokenizer (AbstractTokenizer): Tokenizer instance used for vocabulary size
            filename_pattern (str): Glob pattern to match data shard files
            batch_size (int): Number of sequences per batch
            sequence_length (int): Length of each sequence
            process_rank (int): Rank of current process in distributed setting
            num_processes (int): Total number of distributed processes
        """
        self.tokenizer = tokenizer
        self.filename_pattern = filename_pattern
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.split = "train" if "train" in filename_pattern else "val"
        self.vocab_size = tokenizer.vocab_size

        # Get all bin files 
        self.files = sorted(glob.glob(filename_pattern))

        # Load and validate all data shards, count total tokens
        total_number_of_tokens = 0
        for fname in self.files:
            shard_number_of_tokens = self._peek_data_shard(fname)
            assert shard_number_of_tokens >= num_processes * batch_size * sequence_length + 1, \
                   "Shard size too small for batch configuration"
            total_number_of_tokens += int(shard_number_of_tokens)
        self.total_number_of_tokens = total_number_of_tokens

        print(f"Number of tokens in {self.split} split: {self.total_number_of_tokens}")
        self.reset()

    def _peek_data_shard(self, filename: str) -> int:
        """
        Read the header of a data shard file to get metadata.

        Args:
            filename (str): Path to the data shard file

        Returns:
            int: Number of tokens in the shard

        Raises:
            SystemExit: If magic number validation fails
        """
        with open(filename, "rb") as f:
            header = np.frombuffer(f.read(256*4), dtype=np.int32)
            
        if header[0] != 20240520:
            print("ERROR: magic number mismatch in the data .bin file!")
            print("---> HINT: Are you passing in a correct file with --input_bin?")
            print("---> HINT: Dataset encoding changed recently, re-run data prepro or refer again to README")
            print("---> HINT: For example re-run: `python dev/data/tinyshakespeare.py`, then re-try")
            exit(1)
            
        assert header[1] == 1, "unsupported version"
        return header[2]  # number of tokens

    def _load_data_shard(self, filename: str) -> np.ndarray:
        """
        Load token data from a shard file.

        Args:
            filename (str): Path to the data shard file

        Returns:
            np.ndarray: Array of token IDs from the shard

        Raises:
            AssertionError: If token count doesn't match header or version is unsupported
        """
        with open(filename, "rb") as f:
            header = np.frombuffer(f.read(256*4), dtype=np.int32)
            assert header[0] == 20240520, "magic number mismatch in the data .bin file"
            assert header[1] == 1, "unsupported version"
            ntok = header[2]
            tokens = np.frombuffer(f.read(), dtype=np.uint16)
            
        assert len(tokens) == ntok, "number of tokens read does not match header"
        return tokens

    def reset(self) -> None:
        """Reset the data loader to start of the first shard."""
        self.current_shard = 0
        self.current_position = self.process_rank * self.batch_size * self.sequence_length
        self.tokens = self._load_data_shard(self.files[self.current_shard])

    def advance(self) -> None:
        """Advance to the next data shard and reset position."""
        self.current_shard = (self.current_shard + 1) % len(self.files)
        self.current_position = self.process_rank * self.batch_size * self.sequence_length
        self.tokens = self._load_data_shard(self.files[self.current_shard])

    def get_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get the next batch of sequences and targets.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Tuple of (input sequences, target sequences),
                each of shape (batch_size, sequence_length)
        """
        B = self.batch_size
        T = self.sequence_length
        buf = self.tokens[self.current_position : self.current_position + B*T + 1]
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)
        x = (buf[:-1]).view(B, T)  # inputs
        y = (buf[1:]).view(B, T)   # targets

        # Advance position and load next shard if necessary
        self.current_position += B * T * self.num_processes
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.advance()
        return x.cuda(), y.cuda()

    def get_vocab_size(self) -> int:
        """
        Get the vocabulary size of the tokenizer.

        Returns:
            int: Size of the vocabulary
        """
        return self.vocab_size


def get_data_loaders(dataset_name: str, tokenizer: tokenizers.AbstractTokenizer, batch_size: int, sequence_length: int, split_ratio: float, process_rank: int, num_processes: int, text_file: Optional[str] = "data/best_stories.txt") -> Tuple[AbstractDataLoader, AbstractDataLoader]:
    """
    Get train and validation data loaders for a specified dataset.

    Args:
        dataset_name (str): Name of the dataset to load.
        tokenizer: The tokenizer to use for encoding the text.
        batch_size (int): Batch size for the data loaders.
        sequence_length (int): Sequence length for the data loaders.
        split_ratio (float): Ratio to split the dataset into train and validation sets.
        process_rank (int): Rank of the current process in distributed setting.
        num_processes (int): Total number of processes.
        text_file (Optional[str]): Path to text file for RLHF dataset. Only used if dataset_name is "rlhf".

    Returns:
        Tuple[AbstractDataLoader, AbstractDataLoader]: Train and validation data loaders.

    Raises:
        NotImplementedError: If the specified dataset is not implemented.
    """
    if dataset_name.lower() == "creepypasta":
        train_loader = CreepyPastaDataLoader(
            tokenizer=tokenizer,
            batch_size=batch_size,
            sequence_length=sequence_length,
            process_rank=process_rank,
            num_processes=num_processes,
            split="train",
            split_ratio=split_ratio
        )
        
        val_loader = CreepyPastaDataLoader(
            tokenizer=tokenizer,
            batch_size=batch_size,
            sequence_length=sequence_length,
            process_rank=process_rank,
            num_processes=num_processes,
            split="val",
            split_ratio=split_ratio
        )
        
        return train_loader, val_loader
    elif dataset_name.lower() == "fineweb":
        train_loader = CachedFineweb(
            tokenizer=tokenizer,
            filename_pattern='data/fineweb10B/fineweb_train_*.bin',
            batch_size=batch_size,
            sequence_length=sequence_length,
            process_rank=process_rank,
            num_processes=num_processes,
        )
        
        val_loader = CachedFineweb(
            tokenizer=tokenizer,
            filename_pattern='data/fineweb10B/fineweb_val_*.bin',
            batch_size=batch_size,
            sequence_length=sequence_length,
            process_rank=process_rank,
            num_processes=num_processes,
        )
        
        return train_loader, val_loader
    elif dataset_name.lower() == "rlhf":
        if text_file is None:
            raise ValueError("text_file must be provided for RLHF dataset")
            
        train_loader = RLHFDataset(
            tokenizer=tokenizer,
            text_file=text_file,
            batch_size=batch_size,
            sequence_length=sequence_length,
            process_rank=process_rank,
            num_processes=num_processes,
            split="train",
            split_ratio=split_ratio
        )
        
        val_loader = RLHFDataset(
            tokenizer=tokenizer,
            text_file=text_file,
            batch_size=batch_size,
            sequence_length=sequence_length,
            process_rank=process_rank,
            num_processes=num_processes,
            split="val", 
            split_ratio=split_ratio
        )
        
        return train_loader, val_loader
    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' is not implemented.")
