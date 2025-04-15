import logging

import torch

from examples.prefetch.compressor import AbstractCompressor, calculate_tensor_size_bits

"""
Implements Entropy-Constrained Uniform Quantization (ECUQ) from

DoCoFL: Downlink Compression for Cross-Device Federated Learning
Ron Dorfman, Shay Vargaftik, Yaniv Ben-Itzhak, Kfir Yehuda Levy 
Proceedings of the 40th International Conference on Machine Learning, PMLR 202:8356-8388, 2023.
"""

class ECUQCompressor(AbstractCompressor):
    def __init__(self, bit, tolerance=0.1, device=None):
        super().__init__()
        self.device = device
        self.bit = bit
        self.tolerance = tolerance


    def compress(self, x: torch.Tensor, name=""):
        # TODO hack to force compression on GPU, need to change later 
        orig_device = x.device
        x = x.to(device="cuda:0")
        self.device = x.device 
        can_compress = True

        shape = x.size()
        x = x.flatten()

        # logging.info("Otherwise start ecuq")
        x_max = torch.max(x)
        x_min = torch.min(x)
        K = 2 ** self.bit
        delta = (x_max - x_min) / K

        # Find quantization values Q
        Q = x_min + (torch.linspace(0, K-1, K, device=self.device) + 0.5) * delta

        # Quantize x
        x_hat_Q = torch.argmin(torch.abs(x - Q.view([-1,1])), 0)

        # Find empirical density of quantized x
        p_Q = torch.bincount(x_hat_Q) / x.numel()

        # Find entropy of quantized x
        min_real = torch.finfo(x.dtype).min
        H_p_Q = - torch.sum(p_Q * torch.clamp(torch.log2(p_Q), min=min_real))

        if H_p_Q < self.bit - self.tolerance:
            x_hat_Q, delta, can_compress = self.double_binary_search_num_quantization_levels(x)

        """
        If we use huffman encoding
        bin_count = torch.bincount(x_hat_Q)
        freq_table = {k:v for k, v in enumerate(bin_count)}
        from dahuffman import HuffmanCodec
        codec = HuffmanCodec.from_frequencies(freq_table)
        encoded = codec.encode(x_hat_Q.tolist())
        print(f"Encoded size bits {len(encoded) * 8} bit budget {x_hat_Q.numel() * self.bit} ratio {len(encoded) * 8 / (x_hat_Q.numel() * self.bit)}")
        self._compressed_size = len(encoded) * 8 + 64
        return encoded, (shape, x_min, delta, codec)
        """
        self.device = orig_device # TODO Remove
        if not can_compress:
            orig_size = calculate_tensor_size_bits(x)
            self._compressed_size = calculate_tensor_size_bits(x)
            logging.info(f"ECUQ cannot compress using full size instead {orig_size}")
            # TODO Remove cast
            return x.to(device=orig_device), (shape, x_min.to(device=orig_device), delta.to(device=orig_device), can_compress)
        
        self.numel = x.numel()
        self._compressed_size = x.numel() * self.bit + 64
        # TODO Remove cast
        return x_hat_Q.to(device=orig_device), (shape, x_min.to(device=orig_device), delta.to(device=orig_device), can_compress)
    
    def decompress(self, tensor_compressed: bytes, ctx):
        # shape, x_min, delta, codec = ctx
        # x = torch.tensor(codec.decode(tensor_compressed), device=self.device)

        shape, x_min, delta, can_compress = ctx
        if not can_compress:
            return tensor_compressed.view(shape)
        
        x = tensor_compressed.to(device=self.device)

        tensor_decompressed = x_min + (x + 0.5) * delta
        tensor_decompressed = tensor_decompressed.view(shape)
        return tensor_decompressed
    
    @property
    def compressed_size(self):
        # Compressed size is set in the compress method
        return self._compressed_size
    
    def double_binary_search_num_quantization_levels(self, x: torch.Tensor):
        iterations = 0
        low = 2**self.bit
        high = torch.inf
        p = -1
        x_max = torch.max(x)
        x_min = torch.min(x)

        while low <= high and iterations < 20:
            if high == torch.inf:
                p += 1
                mid = 2**self.bit + 2**p
            else:
                mid = (low + high) / 2

            K = int(mid)
            delta = (x_max - x_min) / mid 
            # logging.info(f"low: {low} high: {high} mid: {mid} p: {p} b: {self.bit} K: {K} Delta: {delta}")

            Q = x_min + (torch.linspace(0, K-1, K, device=self.device) + 0.5) * delta
            x_hat_Q = torch.argmin(torch.abs(x - Q.view([-1,1])), 0)

            p_Q = torch.bincount(x_hat_Q) / x.numel()

            min_real = torch.finfo(x.dtype).min
            H_p_Q = - torch.sum(p_Q * torch.clamp(torch.log2(p_Q), min=min_real))

            iterations += 1
            if H_p_Q > self.bit:
                high = mid - 1
            elif H_p_Q < self.bit - self.tolerance:
                low = mid + 1
            else:
                return x_hat_Q, delta, True
        # logging.info(f"ECUQ cannot find solution with bit budget {self.bit} and tolerance {self.tolerance}")
        return x, delta, False


    