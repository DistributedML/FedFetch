from abc import ABC, abstractmethod

import torch


class AbstractCompressor(ABC):
    """Base class for compressing and decompressing a single tensor."""

    @abstractmethod
    def compress(self, tensor, name):
        """Compresses a tensor and returns it with the context needed to decompress it."""
        raise NotImplemented("compress was not implemented.")

    @abstractmethod
    def decompress(self, tensors, ctx):
        """Decompress the tensor with the given context."""
        raise NotImplemented("decompress was not implemented.")

    @property
    @abstractmethod
    def compressed_size(self):
        """Returns the compressed size of the tensor in number of bits"""
        raise NotImplemented("compressed_size was not implemented.")


def calculate_tensor_size_bits(tensor: torch.Tensor):
    """Calculate the size of a tensor in bits"""
    return tensor.numel() * tensor.element_size() * 8
