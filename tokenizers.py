"""
Tokenizer module for text processing.

This module provides a base Tokenizer class and implementations for specific tokenizers.
It also includes a factory function to get the appropriate tokenizer based on a string identifier.
"""

import tiktoken
from abc import ABC, abstractmethod

class AbstractTokenizer(ABC):
    """
    Abstract base class for tokenizers.
    """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """
        Returns the vocabulary size of the tokenizer.
        """
        pass

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """
        Encodes the input text into a list of token ids.

        Args:
            text (str): The input text to encode.

        Returns:
            list[int]: A list of token ids.
        """
        pass

    @abstractmethod
    def decode(self, token_ids: list[int]) -> str:
        """
        Decodes a list of token ids back into text.

        Args:
            token_ids (list[int]): A list of token ids to decode.

        Returns:
            str: The decoded text.
        """
        pass

class TiktokenGPT2Tokenizer(AbstractTokenizer):
    """
    Tokenizer implementation using tiktoken for GPT-2.
    """

    def __init__(self):
        self._tokenizer = tiktoken.get_encoding("gpt2")

    @property
    def vocab_size(self) -> int:
        return self._tokenizer.n_vocab

    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text)

    def decode(self, token_ids: list[int]) -> str:
        return self._tokenizer.decode(token_ids)

def get_tokenizer(tokenizer_type: str) -> AbstractTokenizer:
    """
    Factory function to get the appropriate tokenizer based on the input string.

    Args:
        tokenizer_type (str): A string identifier for the desired tokenizer.

    Returns:
        Tokenizer: An instance of the requested tokenizer.

    Raises:
        NotImplementedError: If the requested tokenizer is not implemented.
    """
    if tokenizer_type.lower() == "gpt2":
        return TiktokenGPT2Tokenizer()
    else:
        raise NotImplementedError(f"Tokenizer '{tokenizer_type}' is not implemented.")
