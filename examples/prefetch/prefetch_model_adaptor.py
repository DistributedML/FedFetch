
from typing import List

import torch

from fedscale.cloud.aggregation.optimizers import TorchServerOptimizer
from fedscale.cloud.internal.torch_model_adapter import TorchModelAdapter

"""
A model adaptor for FedFetch
"""
class PrefetchModelAdaptor(TorchModelAdapter):
    def __init__(self, model, optimizer = None):
        super().__init__(model, optimizer)

    def get_weights(self) -> List[torch.Tensor]:
        """
        Get the model's weights as a Tensor array. Note that it doesn't contain layer names. Rather, index 0
        contains the model's first layer weights, and index N contains the N+1 layer's weights.
        :return: A list of torch tensors

        This fixes the return type of get_weights in TorchModelAdapter
        """
        return [params.data.clone().detach() for params in self.model.state_dict().values()]

    def get_keys(self) -> List[str]:
        return list(self.model.state_dict().keys())
