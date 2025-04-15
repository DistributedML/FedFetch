"""
Implements

[LFL] "Federated Learning With Quantized Global Model Updates".
Mohammad Mohammadi Amiri, Deniz Gündüz, Sanjeev R. Kulkarni, H. Vincent Poor. arXiv 2020.
"""

import torch

from examples.prefetch.compressor import AbstractCompressor


class LFLCompressor(AbstractCompressor):

    def __init__(self, bit):
        super().__init__()
        self.bit = bit
        # 1 bit needed for sign, so actual bits for quantums is one less than the total bit budget
        self.quantum_num = 2 ** (self.bit - 1)

    def compress(self, tensor: torch.Tensor, name=""):
        shape = tensor.size()
        tensor = tensor.flatten()

        abs_gradient = tensor.abs()
        max_gradient = abs_gradient.max().flatten()
        min_gradient = abs_gradient.min().flatten()

        level_float = self.quantum_num * (abs_gradient - min_gradient) / (max_gradient - min_gradient) 
        previous_level = level_float.floor()
        prob = torch.empty_like(tensor).uniform_()
        is_next_level = (prob < (level_float - previous_level)).type(torch.float32)
        new_level = (previous_level + is_next_level)

        sign = tensor.sign()
        tensor_compressed = (new_level * sign).type(torch.int16)
        tensor_compressed = tensor_compressed.type(torch.int8 if self.quantum_num < 128 else torch.half)
        tensor_compressed = tensor_compressed, max_gradient, min_gradient
        ctx = (shape, name)

        self.numel = tensor.numel()
        return tensor_compressed, ctx

    def decompress(self, tensor_compressed, ctx):
        tensor_compressed, max_gradient, min_gradient = tensor_compressed
        shape, name = ctx

        decode_output = tensor_compressed.type(torch.float32)
        tensor_decompressed = min_gradient + (max_gradient - min_gradient) * decode_output / self.quantum_num
        tensor_decompressed = tensor_decompressed.view(shape)
        return tensor_decompressed
    
    @property
    def compressed_size(self):
        return self.numel * self.bit + 64