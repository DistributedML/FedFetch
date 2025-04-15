from examples.prefetch.compressor import AbstractCompressor, calculate_tensor_size_bits


class IdentityCompressor(AbstractCompressor):

    def __init__(self):
        super().__init__()

    def compress(self, tensor, name=""):
        self._compressed_size = calculate_tensor_size_bits(tensor)
        return tensor, None

    def decompress(self, tensor_compressed, ctx={}):
        return tensor_compressed
    
    @property
    def compressed_size(self):
        return self._compressed_size