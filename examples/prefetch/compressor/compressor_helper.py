
import numpy as np
import torch
from attr import dataclass

from examples.prefetch.compressor import AbstractCompressor, calculate_tensor_size_bits
from examples.prefetch.compressor.compressor_constants import CompressorType
from examples.prefetch.compressor.ecuq import ECUQCompressor
from examples.prefetch.compressor.eden_wrapper import EDENCompressor
from examples.prefetch.compressor.identity import IdentityCompressor
from examples.prefetch.compressor.lfl import LFLCompressor
from examples.prefetch.compressor.powersgd import PowerSGDCompressor
from examples.prefetch.compressor.qsgd import QSGDCompressor
from examples.prefetch.compressor.qsgd_bucket import QSGDBucketCompressor
from examples.prefetch.compressor.stc_quantization import STCQuantizationCompressor
from examples.prefetch.utils import is_batch_norm_layer


@dataclass
class CompressorConfig:
    compressor_type: CompressorType
    quantization_bit: int | None
    spar_ratio: float | None
    matrix_decomposition_rank: int | None
    device: str | None
    compress_batch_norm: bool = True


@dataclass
class CompressedWeight:
    weight: torch.tensor
    ctx: any

class StateDictCompressionWrapper:
    """
    Class for compressing a model's state dict
    WARNING sparsifiction/masking methods are handled separately!
    """

    def __init__(
        self,
        compressor_config: CompressorConfig,
        keys: list[str],
        compressor: AbstractCompressor | None = None,
        compressed_sizes: list[int] | None = None,
        orig_sizes: list[int] | None = None,
    ):
        self.compressor_config = compressor_config
        self.compressor = (
            compressor if compressor else self.get_compressor(compressor_config)
        )
        self.keys = keys
        self.orig_sizes = orig_sizes
        self.compressed_sizes = compressed_sizes
        self._orig_size = None
        self._compressed_size = None
    
    def __init__(
        self,
        compressor_config: CompressorConfig,
        weights: list[torch.tensor],
        keys: list[str],
        compressed_weights: list[CompressedWeight] | None = None,
        compressed_sizes: list[int] | None = None,
        orig_sizes: list[int] | None = None,
        compressor: AbstractCompressor | None = None,
    ):
        self.compressor_config = compressor_config
        self.weights = weights
        self.keys = keys
        self.compressed_weights = compressed_weights
        self.compressed_sizes = compressed_sizes
        self.orig_sizes = (
            orig_sizes
            if orig_sizes
            else [calculate_tensor_size_bits(weight) for weight in self.weights]
        )
        self.compressor = (
            compressor if compressor else self.get_compressor(compressor_config)
        )        
        self._orig_size = None
        self._compressed_size = None

    @classmethod
    def init_with_statedict(
        cls,
        state_dict: dict[str, any],
        compressor_config: CompressorConfig,
        compressor: AbstractCompressor | None = None,
        clone_weights=True,
    ):
        weights = (
            [params.data.clone().detach() for params in state_dict.values()]
            if clone_weights
            else state_dict.values()
        )
        keys = list(state_dict.keys())
        return cls(compressor_config, weights, keys, compressor=compressor)

    @property
    def orig_size(self):
        if not self._orig_size:
            self._orig_size = np.sum(self.orig_sizes)
        return self._orig_size

    @property
    def compressed_size(self):
        if not self.compressed_sizes:
            raise RuntimeError(
                "The state dict must be compressed first before you can access the compressed size"
            )
        if not self._compressed_size:
            self._compressed_size = np.sum(self.compressed_sizes)
        return self._compressed_size
    
    def compress(self):
        self.compressed_weights = []
        self.compressed_sizes = []
        for key, weight in zip(self.keys, self.weights):
            # Do not compress batch norm layers
            if not self.compressor_config.compress_batch_norm and is_batch_norm_layer(key):
                cur_compressed_update, ctx = weight, (weight.shape, "batch_norm")
            else:
                cur_compressed_update, ctx = self.compressor.compress(weight, key)

            self.compressed_weights.append(CompressedWeight(cur_compressed_update, ctx))
            self.compressed_sizes.append(self.compressor.compressed_size)
        return self
    
    def decompress(self):
        self.weights = []
        for key, compressed_weight in zip(self.keys, self.compressed_weights):
            # Do not decompress batch norm layers
            if not self.compressor_config.compress_batch_norm and is_batch_norm_layer(key):
                cur_decompressed_update = compressed_weight.weight
            else:
                cur_decompressed_update = self.compressor.decompress(compressed_weight.weight, compressed_weight.ctx)

            self.weights.append(cur_decompressed_update)
        return self

    def get_compressor(self, config: CompressorConfig):
        if config.compressor_type == CompressorType.NO_COMPRESSION:
            return IdentityCompressor()
        if config.compressor_type == CompressorType.QSGD:
            return QSGDCompressor(config.quantization_bit)
        elif config.compressor_type == CompressorType.QSGD_BUCKET:
            return QSGDBucketCompressor(config.quantization_bit)
        elif config.compressor_type == CompressorType.LFL:
            return LFLCompressor(config.quantization_bit)
        elif config.compressor_type == CompressorType.EDEN:
            return EDENCompressor(config.quantization_bit, device=config.device)
        elif config.compressor_type == CompressorType.ECUQ:
            return ECUQCompressor(config.quantization_bit, device=config.device)
        elif config.compressor_type == CompressorType.STC_QUANT:
            return STCQuantizationCompressor(config.spar_ratio)
        elif config.compressor_type == CompressorType.POWERSGD:
            return PowerSGDCompressor(rank=config.matrix_decomposition_rank)
        else:
            raise NotImplementedError(
                f"Download compression method {config.compressor_type} is not implemented"
            )
