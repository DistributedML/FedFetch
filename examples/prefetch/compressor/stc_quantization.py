import logging
import math

import numpy as np
import torch

from examples.prefetch.compressor import AbstractCompressor
from examples.prefetch.compressor.topk import TopKCompressor

"""
Implements

Robust and Communication-Efficient Federated Learning from Non-IID Data
Felix Sattler, Simon Wiedemann, Klaus-Robert Müller*, Member, IEEE, and Wojciech Samek*, Member, IEEE
"""

class STCQuantizationCompressor(AbstractCompressor):

    def __init__(self, compress_ratio):
        super().__init__()
        self.compress_ratio = compress_ratio

    def compress(self, tensor: torch.Tensor, name=""):
        # Assumes that tensor is already masked
        shape = tensor.size()
        tensor = tensor.flatten()
        k = max(1, int(tensor.numel() * self.compress_ratio))
        topk_compressor = TopKCompressor(self.compress_ratio)
        masked_tensor, ctx = topk_compressor.compress(tensor)
        masked_tensor = topk_compressor.decompress(masked_tensor, ctx)
        mu = torch.sum(torch.abs(masked_tensor)) / k
        tensor_compressed = torch.sign(masked_tensor)
        ctx = (mu, shape)

        self.nonzero_numel = k
        return tensor_compressed, ctx

    def decompress(self, tensor_compressed, ctx):
        mu, shape = ctx
        tensor_decompressed = tensor_compressed * mu
        tensor_decompressed = tensor_compressed.view(shape)
        return tensor_decompressed
    
    @property
    def compressed_size(self):
        # TODO implement STC Golomb code and measure real size

        # Estimates size using Golomb code on quantized values
        # Assumes random sparsity pattern and large k
        phi = (math.sqrt(5) + 1) / 2
        b_star = 1 + math.floor(math.log2(math.log(phi - 1) / math.log(1 - self.compress_ratio)))
        b_pos = b_star + 1 / (1 - (1 - self.compress_ratio)**(2**b_star))
        return math.ceil(self.nonzero_numel + self.nonzero_numel * b_pos) + 32