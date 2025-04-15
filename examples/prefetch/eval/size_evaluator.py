import logging
import math
from typing import Dict, List

import numpy as np
import torch

from examples.prefetch.compressor.compressor_constants import B_POS, CompressorType
from examples.prefetch.compressor.compressor_helper import (
    CompressorConfig,
    StateDictCompressionWrapper,
)
from examples.prefetch.constants import *
from examples.prefetch.utils import is_batch_norm_layer

model_update_1_round_overhead = 0
model_update_accurate_cache = {}

class SizeEvaluator(object):
    def __init__(self, args, state_dict, device) -> None:
        self.device = device
        self.args = args
        self.c_base = args.prefetch_compressor_type
        self.c_dl = args.download_compressor_type
        self.c_ul = args.upload_compressor_type
        self.fl_method = args.fl_method

        self.total_mask_ratio = args.total_mask_ratio
        self.shared_mask_ratio = args.shared_mask_ratio

        self.layer_numels = []
        self.layer_element_sizes = []
        self.layer_sizes = []
        self.batchnorm_numel = 0 # number of batchnorm elements
        self.batchnorm_size = 0

        for key, tensor in state_dict.items():
            self.layer_numels.append(tensor.numel())
            self.layer_element_sizes.append(tensor.element_size())
            self.layer_sizes.append(tensor.numel() * tensor.element_size())
            if is_batch_norm_layer(key):
                self.batchnorm_numel += tensor.numel()
                self.batchnorm_size += tensor.numel() * tensor.element_size()

        self.model_size = sum(self.layer_sizes) * 8 # bytes to bits
        self.model_numel = sum(self.layer_numels) # equivalent to the size of a bitmap
        self.batchnorm_size *= 8 # bytes to bits

        self.delta_cache: Dict[tuple[int, int], float] = {}
        self.avg_delta_sizes: Dict[int, tuple[int, int]] = {} # Maps from delta to avg_size and count
        # self.partial_sums_cache: Dict[tuple[int, int], List[torch.Tensor]] = {} # non-overlapping partial sums
        self.sparsification_ratio_cache = {0: self.total_mask_ratio}
        # self.partial_sums_highest = None
        self.no_calc_after = 49

        # Find single round compressed sizes
        self.init_compressed_sizes(state_dict) 

    def init_compressed_sizes(self, state_dict:Dict[str, List[torch.Tensor]]):
        state_dict_dl = StateDictCompressionWrapper.init_with_statedict(state_dict, CompressorConfig(self.c_dl, self.args.download_quantization_bit, self.total_mask_ratio, self.args.matrix_decomposition_rank, self.device, compress_batch_norm=self.args.compress_batch_norm)).compress()
        state_dict_ul = StateDictCompressionWrapper.init_with_statedict(state_dict, CompressorConfig(self.c_ul, self.args.upload_quantization_bit, self.total_mask_ratio, self.args.matrix_decomposition_rank, self.device, compress_batch_norm=self.args.compress_batch_norm)).compress()
        self.compressed_model_size_base = 0
        self.compressed_model_size_dl = state_dict_dl.compressed_size + self.batchnorm_size
        self.compressed_model_size_ul = state_dict_ul.compressed_size + self.batchnorm_size
    
    def estimate_size_base_and_delta(self, r_min, r, last_synced_round=1):
        # logging.info(f"Estimate size and base r_min {r_min} r {r}")
        last_synced_round = max(last_synced_round, 1)
        if self.c_base == CompressorType.NO_COMPRESSION:
            if self.fl_method == FEDAVG:
                return self.model_size
            elif self.fl_method in [STC, GLUEFL]:
                return min(
                    self.estimate_delta_size(last_synced_round, r-1),
                    self.model_size
                )
            else:
                raise RuntimeError(f"Used a non-supported fl method {self.fl_method}")
        elif self.c_base == CompressorType.ECUQ and self.c_dl == CompressorType.EDEN:
            return self.compressed_model_size_base
        else:
            raise RuntimeError(f"Used a non-supported base compressor {self.c_base}")
    
    def estimate_delta_size(self, l, r):
        l = max(l, 1)
        # logging.info(f"Try to find size for l {l} r {r}")

        if l > r:
            return 0
        
        if l == r:
            if self.fl_method in [STC, GLUEFL]:
                return self.model_size * self.total_mask_ratio
            else:
                return self.compressed_model_size_dl
        delta = r - l

        if delta in self.avg_delta_sizes:
            return self.avg_delta_sizes[delta][0]
        else:
            # delta > len(self.avg_delta_sizes)
            # logging.warning(f"estimate delta size saw delta {delta} not in self.avg_delta_size")
            if self.fl_method in [STC, GLUEFL]:
                if self.c_dl == CompressorType.STC_QUANT:
                    return self.model_numel * 2 * np.sum([self.total_mask_ratio * (1-self.total_mask_ratio)**i for i in range(delta + 1)])
                else:
                    return self.model_size * np.sum([self.total_mask_ratio * (1-self.total_mask_ratio)**i for i in range(delta + 1)])
            else:
                return delta * self.compressed_model_size_dl

    def calculate_size_base_and_delta(self, r_min, p, mask_list, last_synced_round):
        last_synced_round = max(last_synced_round, 1)
        if self.c_base == CompressorType.NO_COMPRESSION:
            if self.fl_method == FEDAVG:
                if self.args.use_single_delta:
                    res = self.model_size
                    for i in range(r_min, p):
                        res += self.calculate_delta_size(i,i)
                    return res
                
                return self.model_size
            elif self.fl_method in [STC, GLUEFL]:
                if self.args.use_single_delta:
                    res = self.model_size
                    for i in range(r_min, p):
                        res += self.calculate_delta_size(i,i)
                    return res
                
                return min(
                    self.calculate_delta_size(last_synced_round, p-1, mask_list),
                    self.model_size
                )
            else:
                raise RuntimeError(f"Used a non-supported fl method {self.fl_method}")
        elif self.c_base == CompressorType.ECUQ and self.c_dl == CompressorType.EDEN:
            return self.compressed_model_size_base
        else:
            raise RuntimeError(f"Used a non-supported base compressor {self.c_base}")
    
    def calculate_ul_size(self):
        if self.fl_method == STC:
            coordinate_size = min(
                self.model_numel, # Standard bitmap
                self.model_numel * self.total_mask_ratio * min(
                    math.ceil(np.log2(self.model_numel + 1)), # Standard position coordinates
                    B_POS(self.total_mask_ratio) # Golomb encoded position coordinates
                )
            )
            return self.total_mask_ratio * self.model_size + coordinate_size
        elif self.fl_method == GLUEFL:
            coordinate_size = min(
                self.model_numel, # Standard bitmap
                self.model_numel * (self.total_mask_ratio - self.shared_mask_ratio) * min(
                    math.ceil(np.log2(self.model_numel + 1)), # Standard position coordinates
                    B_POS(self.total_mask_ratio - self.shared_mask_ratio) # Golomb encoded position coordinates
                )
            )
            return self.total_mask_ratio * self.model_size + coordinate_size
        else:
            return self.compressed_model_size_ul
      
    def calculate_delta_size(self, l: int, r: int, mask_list:List[List[torch.Tensor]]=None):
        l = max(l, 1)
        # logging.info(f"Calculate delta size l {l} r {r} len mask list {len(mask_list)} compression ratio {self.compressed_model_size_dl / self.model_size}")
        if l > r or self.fl_method == DOCOFL:
            return 0
        elif r - l >= self.no_calc_after:
            return self.model_size # Technically not the case for compress full model but that is out of scope of this project
        elif (l, r) in self.delta_cache:
            return self.delta_cache[(l, r)]
        elif  self.args.use_single_delta:
            return (r - l + 1) * self.compressed_model_size_dl
        else:
            if self.fl_method in [STC, GLUEFL]:
                sparsification_ratio = self.calculate_sparsification_ratio(l, r, mask_list)
                logging.info(f"sparsification ratio {sparsification_ratio} of {r-l + 1} updates")
                coordinate_size = min(
                    self.model_numel, # A bitmap
                    self.model_numel * sparsification_ratio * min (
                        math.ceil(np.log2(self.model_numel + 1)), # Standard position coordinates
                        B_POS(sparsification_ratio) # Golomb encoded position coordinates
                    )
                )
                if self.c_dl == CompressorType.STC_QUANT:
                    delta_size = min(sparsification_ratio * self.model_numel * 2 + coordinate_size, self.model_size)
                else:
                    delta_size = min(sparsification_ratio * self.model_size + coordinate_size, self.model_size)
            else: # A quantization methods
                if self.c_dl == CompressorType.NO_COMPRESSION:
                    delta_size = self.model_size
                else:
                    delta_size = min(self.compressed_model_size_dl * (r - l + 1), self.model_size)

            # No need to calculate deltas after this point because it is too small
            # With top_k and q=20%, this is typically about 40 rounds
            if (self.model_size - delta_size) / self.model_size < 0.01:
                self.no_calc_after = r-l

            self.delta_cache[(l, r)] = delta_size
            self.update_avg_delta_size(l, r, delta_size)
            return delta_size

    def update_avg_delta_size(self, l, r, size):
        # if l == 0:
        #     return
        
        round_diff = r - l
        if round_diff not in self.avg_delta_sizes:
            self.avg_delta_sizes[round_diff] = (size, 1)
        else:
            avg_size, count = self.avg_delta_sizes[round_diff]
            self.avg_delta_sizes[round_diff] = ((count / (count + 1)) * avg_size + (1 / (count + 1)) * size, count + 1)

    def calculate_sparsification_ratio(self, l: int, r: int, mask_list:List[List[torch.Tensor]]=None):
        # if l == 0:
        #     return 1. # Client needs to sync the initial model
        
        round_diff = r - l
        if round_diff in self.sparsification_ratio_cache:
            return self.sparsification_ratio_cache[round_diff]
        
        if round_diff > self.no_calc_after or l < len(mask_list) - 50:
            return 1.
        
        logging.info(f"Mask list len {len(mask_list)} round diff {round_diff} l {l} r {r}")
        partial_sum = []
        latest_mask = mask_list[-1]
        for idx, mask in enumerate(latest_mask):
            partial_sum.append(mask.clone().detach().to(dtype=torch.bool, device=self.device))
        # partial_sum = copy.deepcopy(mask_list[-1])
        for i in range(len(mask_list)-1, max(0, l-1), -1):
            for j in range(len(partial_sum)):
                tmp  = mask_list[i][j].clone().detach().to(dtype=torch.bool, device=self.device)
                partial_sum[j] |= tmp
            self.sparsification_ratio_cache[len(mask_list) - i - 1] = self.__calc_spar_ratio(partial_sum)
        
        # if self.partial_sums_highest is None:
        #     partial_sum = copy.deepcopy(mask_list[r])
        #     start = r
        #     self.sparsification_ratio_cache[0] = self.__calc_spar_ratio(partial_sum)
        # else:
        #     start, partial_sum = self.partial_sums_highest
        
        # for i in range(start, l-1, -1):
        #     for j in range(len(partial_sum)):
        #         partial_sum[j] |= mask_list[i][j]
        #     self.sparsification_ratio_cache[r-i] = self.__calc_spar_ratio(partial_sum)
            
        # self.partial_sums_highest = (l, partial_sum)
        return self.sparsification_ratio_cache[round_diff]
    
    def __calc_spar_ratio(self, partial_sum):
        tot_nonzero = 0
        for i in range(len(partial_sum)):
            tot_nonzero += partial_sum[i].sum()
        return (float(tot_nonzero) + self.batchnorm_numel) / self.model_numel
    
    def update_avg_spar_ratio(self):
        pass

    def clean_spar_ratio_cache(self):
        for i in range(20):
            if i in self.sparsification_ratio_cache:
                self.sparsification_ratio_cache.pop(i)
            # self.sparsification_ratio_cache.clear()
        # self.partial_sums_highest = None

    @staticmethod
    def check_sparsification_ratio(global_g_list):
        worker_number = len(global_g_list)
        spar_ratio = 0.

        total_param = 0
        for g_idx, g_param in enumerate(global_g_list[0]):
            total_param += len(torch.flatten(global_g_list[0][g_idx]))

        for i in range(worker_number):
            non_zero_param = 0
            for g_idx, g_param in enumerate(global_g_list[i]):
                mask = g_param != 0.
                # print(mask)
                non_zero_param += float(torch.sum(mask))

            spar_ratio += (non_zero_param / total_param) / worker_number

        return spar_ratio

    @staticmethod
    def check_tensor_difference(tensor_list_1, tensor_list_2):
        tot_nonzero = 0
        tot_param = 0
        
        for idx, val in enumerate(tensor_list_1):
            tmp = (tensor_list_1[idx] - tensor_list_2[idx]) != 0.
            tot_nonzero += tmp.sum()
            tot_param += tmp.numel()

        spar_ratio = tot_nonzero / tot_param if tot_param > 0 else -1
        return spar_ratio

    @staticmethod
    def check_model_update_overhead(l, r, global_model, mask_record_list, device, use_accurate_cache=False):
        # logging.info(f"check_model_update_overhead {l} {r} {len(mask_record_list)}")
        if r - l < 0:
            raise RuntimeError(f"check_model_update_overhead() saw r{r} which is less than l{l}")

        if l < 0:
            # Client does not yet have the global model
            return 1

        if r - l == 0:
            return 0
            
        if r - l == 1:
            if SizeEvaluator.model_update_1_round_overhead > 0:
                return SizeEvaluator.model_update_1_round_overhead

        elif use_accurate_cache:
            if (r << 16 + l) in SizeEvaluator.model_update_accurate_cache:
                return SizeEvaluator.model_update_accurate_cache[r << 16 + l]

        mask_accum_list = []
        
        for p_idx, key in enumerate(global_model.state_dict().keys()):
            mask_accum_list.append(torch.zeros_like(global_model.state_dict()[key], dtype=torch.bool, device=torch.device("cpu")))

        for idx in range(l, r):
            for p_idx, key in enumerate(global_model.state_dict().keys()):
                mask_accum_list[p_idx] |= mask_record_list[idx][p_idx]
        
        tot_nonzero = 0
        tot_param = 0
        for p_idx, key in enumerate(global_model.state_dict().keys()):
            tot_nonzero += mask_accum_list[p_idx].sum()
            tot_param += mask_accum_list[p_idx].numel()
            
        res = float(tot_nonzero / tot_param)

        if r - l == 1:
            SizeEvaluator.model_update_1_round_overhead = res
        elif use_accurate_cache:
            SizeEvaluator.model_update_accurate_cache[r << 16 + l] = res

        return res