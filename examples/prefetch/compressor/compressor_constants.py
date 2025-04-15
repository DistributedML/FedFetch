from enum import Enum

import numpy as np


class CompressorType(str, Enum):
    NO_COMPRESSION = "None"
    
    # QUANTIZATION METHODS
    QSGD = "QSGD"
    QSGD_BUCKET = "QSGD_bucket"
    LFL = "LFL"
    EDEN = "EDEN"
    ECUQ = "ECUQ"
    STC_QUANT = "STC_QUANT"

    # MATRIX DECOMPOSITION METHODS
    POWERSGD = "POWERSGD"

# STC Golomb encoded position coordinate size
PHI_GOLDEN_RATIO = (np.sqrt(5) + 1) / 2
B_STAR = lambda p: 1 + np.floor(np.log2(np.log(PHI_GOLDEN_RATIO - 1) / np.log(1 - p)))
B_POS = lambda p: B_STAR(p) + (1 / (1 - (1 - p) ** (2 ** B_STAR(p))))