import copy
import logging
import math
import os
import pickle

import numpy as np
import torch
from torch.autograd import Variable

from examples.prefetch.compressor.compressor_constants import CompressorType
from examples.prefetch.compressor.compressor_helper import (
    CompressorConfig,
    StateDictCompressionWrapper,
)
from examples.prefetch.compressor.topk import TopKCompressor
from examples.prefetch.constants import *
from examples.prefetch.utils import is_batch_norm_layer
from fedscale.cloud.config_parser import args
from fedscale.cloud.execution.torch_client import TorchClient

"""A customized client for Prefetch FL"""
class PrefetchClient(TorchClient):
    """Basic client component in Federated Learning"""

    def load_compensation(self, temp_path):
        # load compensation vector
        with open(temp_path, "rb") as model_in:
            model = pickle.load(model_in)
        return model

    def save_compensation(self, model, temp_path):
        # save compensation vector
        model = [i.to(device="cpu") for i in model]
        with open(temp_path, "wb") as model_out:
            pickle.dump(model, model_out)
    
    def train(
        self, client_data, model: torch.nn.Module, conf, mask_model, epochNo, agg_weight
    ):
        client_id = conf.clientId
        np.random.seed(1) # TODO verify why this is needed
        total_mask_ratio = args.total_mask_ratio
        fl_method = args.fl_method
        regenerate_epoch = args.regenerate_epoch
        compensation_dir = os.path.join(
            args.compensation_dir, args.job_name, args.time_stamp
        )

        trained_unique_samples = min(
            len(client_data.dataset), conf.local_steps * conf.batch_size
        )

        logging.info(
            f"Start to train (CLIENT: {client_id}) (WEIGHT: {agg_weight}) (LR: {conf.learning_rate})..."
        )

        model = model.to(device=self.device)
        model.train()

        # ===== load compensation =====
        if args.use_compensation:
            os.makedirs(compensation_dir, exist_ok=True)
            temp_path = os.path.join(compensation_dir, 'compensation_c'+str(client_id)+'.pth.tar')
            compensation_model = []
            if (agg_weight > 100.0) or (not os.path.exists(temp_path)):
                # create a new compensation model
                for idx, param in enumerate(model.state_dict().values()):
                    tmp = torch.zeros_like(param.data).to(device=self.device)
                    compensation_model.append(tmp)
            else:
                # load existing compensation models
                try:
                    compensation_model = self.load_compensation(temp_path)
                    compensation_model = [c.to(device=self.device) for c in compensation_model]
                except EOFError:
                    logging.info(f"Compensation model from {temp_path} cannot be loaded")
                    compensation_model = []

        keys = []
        for idx, key in enumerate(model.state_dict()):
            keys.append(key)

        last_model_copy = [param.data.clone().detach() for param in model.state_dict().values()]

        # TODO use FedScale's helper functions instead
        # optimizer = self.get_optimizer(model, conf)
        # criterion = self.get_criterion(conf)
        optimizer = torch.optim.SGD(model.parameters(), lr=conf.learning_rate, momentum=0.9, weight_decay=5e-4)
        criterion = torch.nn.CrossEntropyLoss().to(device=self.device)

        epoch_train_loss = 1e-4
        error_type = None
        completed_steps = 0

       # TODO: One may hope to run fixed number of epochs, instead of iterations
        while completed_steps < conf.local_steps:
            for data_pair in client_data:
                (data, target) = data_pair
                data, target = Variable(data).to(device=self.device), Variable(target).to(device=self.device)

                if args.task == "speech":
                    data = torch.unsqueeze(data, 1)

                output = model(data)
                loss = criterion(output, target)

                # only measure the loss of the first epoch
                epoch_train_loss = (1. - conf.loss_decay) * epoch_train_loss + conf.loss_decay * loss.item()
                # logging.info(f"local {clientId} {completed_steps} {loss.item()}")

                # ========= Define the backward loss ==============
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                completed_steps += 1

                if completed_steps == conf.local_steps:
                    break

        # ===== calculate gradient =====
        model_gradient = []
        for idx, param in enumerate(model.state_dict().values()):
            model_gradient.append(
                (last_model_copy[idx] - param.data).type(torch.FloatTensor).to(device=self.device)
            )

        # ===== apply compensation =====
        if args.use_compensation:
            for idx, param in enumerate(compensation_model):
                model_gradient[idx] += (compensation_model[idx] / agg_weight)
        gradient_original = copy.deepcopy(model_gradient)

        # ===== apply sparsification =====
        if fl_method in [FEDAVG, DOCOFL] or args.upload_compressor_type == CompressorType.STC_QUANT:
            pass 
        else:
            c_ul = TopKCompressor(compress_ratio=total_mask_ratio)
            for idx, gradient_tmp in enumerate(model_gradient):
                if is_batch_norm_layer(keys[idx]):
                    continue 
                    
                if fl_method == STC or (fl_method == GLUEFL and epochNo % regenerate_epoch == 1):
                    # STC or GlueFL with shared mask regneration
                    tmp_compressed_gradient, ctx_tmp = c_ul.compress(
                            gradient_tmp)
                    model_gradient[idx] = c_ul.decompress(tmp_compressed_gradient, ctx_tmp)

                elif fl_method == GLUEFL:
                    # GlueFL shared mask + unique mask
                    max_value = float(gradient_tmp.abs().max())
                    largest_tmp = gradient_tmp.clone().detach()
                    largest_tmp[mask_model[idx] == True] = max_value
                    largest_tmp, ctx_tmp = c_ul.compress(largest_tmp)
                    largest_tmp = c_ul.decompress(largest_tmp, ctx_tmp)
                    largest_tmp = largest_tmp.to(torch.bool)
                    gradient_tmp[largest_tmp != True] = 0.0
                    model_gradient[idx] = gradient_tmp # FIXME might be redundant
                
                else:
                    raise NotImplementedError(f"Upload sparsification method {fl_method} is not implemented")

        # ===== apply quantization =====
        ul_c_config = CompressorConfig(args.upload_compressor_type, args.upload_quantization_bit, total_mask_ratio, args.matrix_decomposition_rank, self.device, compress_batch_norm=args.compress_batch_norm)
        model_gradient_state_dict = StateDictCompressionWrapper(ul_c_config, model_gradient, keys).compress().decompress()
        model_gradient = model_gradient_state_dict.weights

        # ===== save compensation =====
        if args.use_compensation:
            for idx, param in enumerate(compensation_model):
                compensation_model[idx] = (gradient_original[idx] - model_gradient[idx]) * agg_weight
            compensation_model = [e.to(device='cpu') for e in compensation_model]
            self.save_compensation(compensation_model, temp_path)

        # ===== collect results =====
        results = {
            "client_id": client_id,
            "moving_loss": epoch_train_loss,
            "trained_size": completed_steps * conf.batch_size,
            "success": completed_steps > 0,
        }
        results["utility"] = math.sqrt(epoch_train_loss) * float(trained_unique_samples)

        if error_type is None:
            logging.info(f"Training of (CLIENT: {client_id}) completes, {results}")
        else:
            logging.info(f"Training of (CLIENT: {client_id}) failed as {error_type}")

        # results['update_weight'] = model_param
        model_gradient = [g.to(device="cpu").numpy() for g in model_gradient]
        results["update_gradient"] = model_gradient
        results["wall_duration"] = 0

        return results