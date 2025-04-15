import math

import numpy as np

from examples.prefetch.compressor import AbstractCompressor
from examples.prefetch.compressor.eden.eden import eden_builder

"""
Adapted from
https://github.com/amitport/EDEN-Distributed-Mean-Estimation

EDEN: Communication-Efficient and Robust Distributed Mean Estimation for Federated Learning
Shay Vargaftik, Ran Ben Basat, Amit Portnoy, Gal Mendelson, Yaniv Ben Itzhak, Michael Mitzenmacher 
Proceedings of the 39th International Conference on Machine Learning, PMLR 162:21984-22014, 2022.
"""

class EDENCompressor(AbstractCompressor):
    def __init__(self, bit, device):
        super().__init__()
        self.bit = bit
        self.eden = eden_builder(bits=self.bit, device=device)
        # For size calculation
        self.numel = 0

    def compress(self, tensor, name=""):
        tensor_compressed, ctx = self.eden.forward(tensor)
        self.numel = tensor_compressed.numel()
        return tensor_compressed, ctx

    def decompress(self, tensor_compressed, ctx):
        tensor_decompressed, metrics = self.eden.backward(tensor_compressed, ctx) # metrics is always None
        return tensor_decompressed
    
    @property
    def compressed_size(self):
        # Assumes tensor shape and seed is already shared between server and clients
        return self.numel * self.bit + 32
    