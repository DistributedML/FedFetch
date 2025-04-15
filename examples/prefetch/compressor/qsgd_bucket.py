"""
Adapted from
https://github.com/sands-lab/grace/blob/master/grace_dl/dist/compressor/qsgd.py

GRACE: A Compressed Communication Framework for Distributed Machine Learning.
H. Xu, C.-Y. Ho, A. M. Abdelmoniem, A. Dutta, E. H. Bergou, K. Karatsenidis, M. Canini, P. Kalnis.
In Proc. of ICDCS, 2021.
"""

import torch

from examples.prefetch.compressor import AbstractCompressor


class QSGDBucketCompressor(AbstractCompressor):

    def __init__(self, bit, bucket_size=128):
        super().__init__()
        self.bit = bit
        # 1 bit needed for sign, so actual bits for quantums is one less than the total bit budget
        self.quantum_num = 2 ** (self.bit - 1)
        self.bucket_size = bucket_size

    def compress(self, tensor, name=""):
        shape = tensor.size()
        tensor = tensor.flatten()
        abs_gradient = tensor.abs()

        if tensor.numel() % self.bucket_size != 0:
            pad_size = self.bucket_size - tensor.numel() % self.bucket_size
            pad_tensor = torch.cat([tensor, torch.zeros(pad_size, dtype=tensor.dtype, device=tensor.device)])
        else:
            pad_tensor = tensor
        pad_tensor = pad_tensor.view([-1, self.bucket_size])
        pad_tensor_sqsum = torch.sum(pad_tensor ** 2, dim=1)
        bucket_norm = torch.sqrt(pad_tensor_sqsum)
        b = torch.ones([1, self.bucket_size], device=tensor.device)
        expand_norm = torch.matmul(bucket_norm.view([-1, 1]), b)
        norm = expand_norm.flatten()[:tensor.numel()]

        level_float = self.quantum_num / norm * abs_gradient
        previous_level = level_float.floor()
        prob = torch.empty_like(tensor).uniform_()
        is_next_level = (prob < (level_float - previous_level)).type(torch.float32)
        new_level = (previous_level + is_next_level)

        sign = tensor.sign()
        tensor_compressed = (new_level * sign).type(torch.int16)
        tensor_compressed = tensor_compressed.type(torch.int8 if self.quantum_num < 128 else torch.half)

        self.numel = tensor.numel()
        return (tensor_compressed, bucket_norm), shape

    def decompress(self, tensor_compressed, ctx):
        tensor_compressed, bucket_norm = tensor_compressed
        shape = ctx
        b = torch.ones([1, self.bucket_size], device=tensor_compressed.device)
        expand_norm = torch.matmul(bucket_norm.view([-1, 1]), b)
        norm = expand_norm.flatten()[:shape.numel()]
        decode_output = tensor_compressed.type(torch.float32)
        tensor_decompressed = norm / self.quantum_num * decode_output
        tensor_decompressed = tensor_decompressed.view(shape)

        return tensor_decompressed
    
    @property
    def compressed_size(self):
        return self.numel * self.bit + self.bucket_size * 32
