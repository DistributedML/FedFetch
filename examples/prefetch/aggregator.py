import copy
import math
import os
import pickle
import random
import sys
import time
from collections import deque
from typing import Dict, List

import numpy
import torch

import fedscale.cloud.channels.job_api_pb2 as job_api_pb2
from examples.prefetch.client_manager import PrefetchClientManager
from examples.prefetch.compressor.compressor_constants import CompressorType
from examples.prefetch.compressor.compressor_helper import (
    CompressorConfig,
    StateDictCompressionWrapper,
)
from examples.prefetch.compressor.topk import TopKCompressor
from examples.prefetch.constants import *
from examples.prefetch.eval.round_evaluator import RoundEvaluator
from examples.prefetch.eval.size_evaluator import SizeEvaluator
from examples.prefetch.prefetch_model_adaptor import PrefetchModelAdaptor
from examples.prefetch.utils import is_batch_norm_layer
from fedscale.cloud import commons
from fedscale.cloud.aggregation.aggregator import Aggregator
from fedscale.cloud.aggregation.optimizers import TorchServerOptimizer
from fedscale.cloud.logger.aggregator_logging import *


class PrefetchAggregator(Aggregator):
    """Feed aggregator using tensorflow models"""

    def __init__(self, args):
        super().__init__(args)
        self.client_manager = self.init_client_manager(args)

        self.feasible_client_count = 0 # Initialized after executors have finished client registration
        self.num_participants = args.num_participants

        self.total_mask_ratio = args.total_mask_ratio  # = shared_mask + local_mask
        self.shared_mask_ratio = args.shared_mask_ratio
        self.regenerate_epoch = args.regenerate_epoch
        self.max_prefetch_round = args.max_prefetch_round

        self.sampling_strategy = args.sampling_strategy
        self.sticky_group_size = args.sticky_group_size
        self.sticky_group_change_num = args.sticky_group_change_num
        self.sampled_sticky_client_set = []
        self.sampled_sticky_clients = []
        self.sampled_changed_clients = []

        self.fl_method = args.fl_method

        self.compressed_gradient = []
        self.mask_record_list = [[]] # Use 1-index
        self.shared_mask = []
        self.apf_ratio = 1.0

        self.last_update_index = []
        self.round_lost_clients = []
        self.clients_to_run = []
        self.slowest_client_id = -1
        self.round_evaluator = RoundEvaluator()

        # TODO Extract scheduler logic
        self.enable_prefetch = args.enable_prefetch
        self.max_prefetch_round = args.max_prefetch_round
        self.warmup_round = args.warmup_round
        self.prefetch_schedules_dict:Dict[int, List[int]] = {} # P
        self.min_prefetch_round:Dict[int, int] = {} # min P
        self.exp_mov_avg_dur = None

        self.sampled_clients = []
        self.sampled_sticky_clients = []
        self.sampled_changed_clients = []
        
        self.download_compressor_type = args.download_compressor_type
        self.upload_compressor_type = args.upload_compressor_type
        self.prefetch_compressor_type = args.prefetch_compressor_type

        self.dl_c_config = CompressorConfig(self.download_compressor_type, self.args.download_quantization_bit, self.total_mask_ratio, self.args.matrix_decomposition_rank, self.device, compress_batch_norm=self.args.compress_batch_norm)
        self.base_c_config = CompressorConfig(self.prefetch_compressor_type, self.args.prefetch_quantization_bit, self.total_mask_ratio, self.args.matrix_decomposition_rank, self.device, compress_batch_norm=self.args.compress_batch_norm)

        self.server_updates_queue = deque(maxlen=args.max_prefetch_round)
        self.server_models_queue = deque(maxlen=args.max_prefetch_round+1) # List of previous model weights of length max_prefetch_round
        self.next_client_model = None

        # DoCoFL
        self.server_anchors_queue = deque(maxlen=args.anchor_count)
        self.anchor_deployment_rate = args.anchor_deployment_rate

    def save_current_model(self):
        save_path = os.path.join(self.args.compensation_dir, 'model_'+str(self.round)+'.pth.tar')
        with open(save_path, "wb") as model_out:
            pickle.dump(self.model_wrapper.get_weights(), model_out)

    def load_model(self, round):
        file_path = os.path.join(self.args.compensation_dir, 'model_'+str(round)+'.pth.tar')
        with open(file_path, "rb") as model_in:
            model = pickle.load(model_in)
        return model


    def init_model(self):
        """Initialize the model"""
        if self.args.engine != commons.PYTORCH:
            raise ValueError(f"{self.args.engine} is not a supported engine for prefetch")
        
        self.model_wrapper = PrefetchModelAdaptor(
            init_model(),
            optimizer=TorchServerOptimizer(
                    self.args.gradient_policy, self.args, self.device
            ),
        )

    def init_client_manager(self, args):
        """Initialize Prefetch FL client sampler

        Args:
            args (dictionary): Variable arguments for fedscale runtime config. defaults to the setup in arg_parser.py

        Returns:
            PrfetchClientManager: The client manager class
        """

        # sample_mode: random or oort
        client_manager = PrefetchClientManager(args.sample_mode, args=args)

        return client_manager

    def init_mask(self):
        self.shared_mask = []
        for idx, param in enumerate(self.model_wrapper.get_model().state_dict().values()):
            self.shared_mask.append(
                torch.zeros_like(param, dtype=torch.bool).to(dtype=torch.bool)
            )

    def client_register_handler(self, executorId, info):
        """Triggered once after all executors have registered

        Args:
            executorId (int): Executor Id
            info (dictionary): Executor information

        """
        logging.info(f"Loading {len(info['size'])} client traces ...")
        for _size in info["size"]:
            # since the worker rankId starts from 1, we also configure the initial dataId as 1
            mapped_id = (
                (self.num_of_clients + 1) % len(self.client_profiles)
                if len(self.client_profiles) > 0
                else 1
            )
            systemProfile = self.client_profiles.get(
                mapped_id,
                {
                    "computation": 1.0,
                    "communication": 1.0,
                    "dl_kbps": 1.0,
                    "ul_kbps": 1.0,
                },
            )

            client_id = (
                (self.num_of_clients + 1)
                if self.experiment_mode == commons.SIMULATION_MODE
                else executorId
            )
            self.client_manager.register_client(
                executorId, client_id, size=_size, speed=systemProfile
            )
            self.client_manager.registerDuration(
                client_id,
                batch_size=self.args.batch_size,
                local_steps=self.args.local_steps,
                upload_size=self.model_update_size,
                download_size=self.model_update_size,
            )
            self.num_of_clients += 1

        logging.info(
            "Info of all feasible clients {}".format(self.client_manager.getDataInfo())
        )

        # Post client registration aggregator initialization
        self.feasible_client_count = len(self.client_manager.feasibleClients)
        self.last_update_index = [
            0 for _ in range(self.feasible_client_count * 2)
        ]

    def run(self):
        """Start running the aggregator server by setting up execution
        and communication environment, and monitoring the grpc message.
        """
        self.setup_env()
        self.client_profiles = self.load_client_profile(
            file_path=self.args.device_conf_file
        )

        self.init_control_communication()
        self.init_data_communication()

        self.init_model()
        self.init_mask()

        self.model_update_size = (
            sys.getsizeof(pickle.dumps(self.model_wrapper)) / 1024.0 * 8.0
        )  # kbits
        self.model_bitmap_size = self.model_update_size / 32


        # Quantization
        initial_model_weights = self.model_wrapper.get_weights_torch()
        self.server_models_queue.append(initial_model_weights)
        self.size_eval = SizeEvaluator(
            self.args,
            self.model_wrapper.get_model().state_dict(),
            self.device
        )
        
        self.event_monitor()

    def get_shared_mask(self):
        """Get shared mask that would be used by all FL clients (in default FL)

        Returns:
            List of PyTorch tensor: Based on the executor's machine learning framework, initialize and return the mask for training.

        """
        return [p.to(device="cpu") for p in self.shared_mask]

    def tictak_client_tasks(self, sampled_clients, num_clients_to_collect):
        """
        Record sampled client execution information in last round. In the SIMULATION_MODE,
        further filter the sampled_client and pick the top num_clients_to_collect clients.

        Args:
            sampled_clients (list of int): Sampled clients from client manager
            num_clients_to_collect (int): The number of clients actually needed for next round.

        Returns:
            tuple: Return the sampled clients and client execution information in the last round.

        """

        if len(sampled_clients) == 0:
            return [], [], [], {}, 0, []

        if self.experiment_mode == commons.SIMULATION_MODE:
            # NOTE: We try to remove dummy events as much as possible in simulations,
            # by removing the stragglers/offline clients in overcommitment"""
            sampledClientsReal = []
            sampledClientsLost = []
            completionTimes = []
            virtual_client_clock = {}

            # prefetch stats
            prefetched_clients = set()
            prefetch_lost_clients = []
            replaced_clients = []


            if self.round > 1:
                if self.enable_prefetch and self.round in self.min_prefetch_round:
                    min_p = self.min_prefetch_round[self.round]
                    self.calculate_next_client_model(min_p - (self.round - self.max_prefetch_round))
                else:
                    self.calculate_next_client_model(len(self.server_models_queue)-1)


            # 1. remove dummy clients that are not available to the end of training
            for client_to_run in sampled_clients:
                client_cfg = self.client_conf.get(client_to_run, self.args)
                exe_cost = {
                    "computation": 0,
                    "downstream": 0,
                    "upstream": 0,
                    "round_duration": 0,
                }
                # =================================================
                dl_size = 0
                ul_size = 0
                pre_size = 0

                train_round = self.round
                lost_during_prefetch = False

                update_size = 0
                if self.enable_prefetch and self.round > self.warmup_round + self.max_prefetch_round:
                    p = self.prefetch_schedules_dict[train_round][client_to_run]
                    min_p = self.min_prefetch_round[self.round]
                    
                    dl_bw = self.client_manager.get_dl_bw_bits(client_to_run)
                    round_durations = self.round_evaluator.round_durations_aggregated
                    update_size = self.size_eval.calculate_size_base_and_delta(
                        min_p, p, self.mask_record_list, self.last_update_index[client_to_run])
                    l = p
                    time_budget = 0

                    pre_size += update_size
                    cur_time = self.global_virtual_clock - sum(round_durations[p:])

                    if p < train_round:
                        # The client will perform prefetching
                        prefetched_clients.add(client_to_run)

                    for j in range(p, train_round):
                        if not self.client_manager.isClientActive(client_to_run, cur_time):
                            prefetch_lost_clients.append(client_to_run)
                            if self.args.enable_replace_offline:
                                """
                                The client now goes offline, replace with a new client 
                                For simplicity, set the prefetch round for replacements to be the next round
                                One could run the prefetch scheduler again on the replacment client to get a better prefetch start round
                                One could also immediately start prefetching after the replacement is decided
                                """
                                replacement = self.client_manager.get_replacement_client(cur_time, j, client_to_run)
                                self.prefetch_schedules_dict[train_round][replacement] = j + 1
                                replaced_clients.append(replacement)
                                sampled_clients.append(replacement)

                            lost_during_prefetch = True
                            break

                        time_budget = round_durations[j]
                        cur_time += round_durations[j]
                        
                        time_budget = max(0, time_budget - update_size / dl_bw)
                        if time_budget > 0:
                            next_delta_size = self.size_eval.calculate_delta_size(
                                l, j - 1, self.mask_record_list)
                            
                            if self.fl_method in [STC, GLUEFL]:
                                # TODO Don't compress the coordinates (the impact of this is only marginal...)
                                # An optimization for masking models, the shared mask will always be changed 
                                # so there is no point in trying to transfer the model corresponding the to the shared mask
                                next_delta_size = next_delta_size * (1 - self.shared_mask_ratio) 
                            
                            pre_size += next_delta_size
                            update_size = max(0, next_delta_size - time_budget * dl_bw)
                            l = j
                        else:
                            update_size -= round_durations[j] * dl_bw

                    pre_size -= update_size # Final update size is part of the download size
                    final_delta = self.size_eval.calculate_delta_size(l, train_round-1, self.mask_record_list)
                    if self.fl_method == DOCOFL or self.prefetch_compressor_type != NO_COMPRESSION:
                        final_delta = self.size_eval.compressed_model_size_dl
                    
                    dl_size = update_size + final_delta
                    logging.info(f"Final prefetch size {pre_size}")
                    logging.info(f"Final download size {dl_size} remaining_updates {update_size} final_delta {final_delta} l {l} r {train_round-1}")
                else:
                    dl_size = self.size_eval.calculate_delta_size(self.last_update_index[client_to_run], train_round-1, self.mask_record_list)
                    if self.fl_method == DOCOFL:
                        dl_size = self.size_eval.compressed_model_size_base
                ul_size = self.size_eval.calculate_ul_size()                        
                exe_cost = self.client_manager.get_completion_time(client_to_run, batch_size=client_cfg.batch_size, local_steps=client_cfg.local_steps, upload_size=ul_size, download_size=dl_size)


                # =================================================
                
                roundDuration = (
                    exe_cost["computation"]
                    + exe_cost["downstream"]
                    + exe_cost["upstream"]
                )
                exe_cost["round"] = roundDuration
                virtual_client_clock[client_to_run] = exe_cost
                self.last_update_index[client_to_run] = (
                    self.round - 1
                )  # Client knows the global state from the previous round
                
                if lost_during_prefetch:
                    logging.info(f"Client {client_to_run} lost during prefetch")
                    self.round_evaluator.record_lost_prefetch(pre_size)
                    # sampledClientsLost.append(client_to_run)
                else:
                    self.round_evaluator.record_client(client_to_run, dl_size, ul_size, exe_cost, prefetch_dl_size=pre_size)
                    if self.client_manager.isClientActive(client_to_run, roundDuration + self.global_virtual_clock):
                        # if the client is not active by the time of collection, we consider it is lost in this round
                        sampledClientsReal.append(client_to_run)
                        completionTimes.append(roundDuration)
                    else:
                        sampledClientsLost.append(client_to_run)

            num_clients_to_collect = min(num_clients_to_collect, len(completionTimes))
            
            # 2. get the top-k completions to remove stragglers
            workers_sorted_by_completion_time = sorted(
                range(len(completionTimes)), key=lambda k: completionTimes[k]
            )
            top_k_index = workers_sorted_by_completion_time[:num_clients_to_collect]
            clients_to_run = [sampledClientsReal[k] for k in top_k_index]

            dummy_clients = [
                sampledClientsReal[k]
                for k in workers_sorted_by_completion_time[num_clients_to_collect:]
            ]
            round_duration = completionTimes[top_k_index[-1]]
            completionTimes.sort()

            slowest_client_id = sampledClientsReal[top_k_index[-1]]
            logging.info(
                f"Successfully prefetch {len(prefetched_clients)} slowest client {slowest_client_id} time {completionTimes[-1]} is prefetched {slowest_client_id in prefetched_clients}"
            )
            logging.info(f"During prefetch, the following clients went offline {prefetch_lost_clients}\nReplacement map{replaced_clients}")
            # logging.info(f"Successfully prefetch {len(prefetched_clients)} slowest client {slowest_client_id} is prefetched {slowest_client_id in prefetched_clients}  {completionTimes}")

            return (
                clients_to_run,
                dummy_clients,
                sampledClientsLost,
                virtual_client_clock,
                round_duration,
                completionTimes[:num_clients_to_collect],
            )
        else:
            virtual_client_clock = {
                client: {"computation": 1, "communication": 1}
                for client in sampled_clients
            }
            completionTimes = [1 for c in sampled_clients]
            return (
                sampled_clients,
                sampled_clients,
                [],
                virtual_client_clock,
                1,
                completionTimes,
                -1,
            )
        
    def calculate_next_client_model(self, base_idx):
        keys = self.model_wrapper.get_keys()
        logging.info(f"server_updates_queue {len(self.server_updates_queue)} server_model_queue {len(self.server_models_queue)} base_idx {base_idx}")
        if self.fl_method == DOCOFL and self.round > 2 and base_idx < len(self.server_models_queue)-1:
            # The vanilla DoCoFL implementation, base model is an anchor compressed with ECUQ
            if self.args.use_latest_model: # Alternative DoCoFL + FedFetch implementation  
                base_model = copy.deepcopy(self.server_models_queue[base_idx])
                base_state_dict = StateDictCompressionWrapper(self.base_c_config, base_model, keys).compress().decompress()
                base_model = base_state_dict.weights
                self.size_eval.compressed_model_size_base = base_state_dict.compressed_size
            else:
                base_model = copy.deepcopy(self.server_anchors_queue[0])
            cur_model = self.model_wrapper.get_weights_torch()
            model_diff = []
            for i, _ in enumerate(keys):
                model_diff.append(cur_model[i] - base_model[i])
            model_diff_state_dict = StateDictCompressionWrapper(self.dl_c_config, model_diff, keys).compress().decompress()
            model_diff = model_diff_state_dict.weights
            self.size_eval.compressed_model_size_dl = model_diff_state_dict.compressed_size
            self.next_client_model = [(base + diff).numpy() for base, diff in zip(base_model, model_diff)]
        elif self.prefetch_compressor_type == CompressorType.ECUQ and self.download_compressor_type == CompressorType.EDEN:
            # The default DoCoFL + FedFetch implementation, base model is one that is compressed with ECUQ
            # The final client side model is the base model plus updates compressed with EDEN
            base_model = copy.deepcopy(self.server_models_queue[base_idx])
            base_state_dict = StateDictCompressionWrapper(self.base_c_config, base_model, keys).compress().decompress()
            base_model = base_state_dict.weights
            self.size_eval.compressed_model_size_base = base_state_dict.compressed_size

            remain_server_models = list(self.server_models_queue)[base_idx+1:]
            
            for m in remain_server_models:
                model_diff = []
                for i, key in enumerate(keys):
                    model_diff.append(m[i] - base_model[i])
                model_diff_state_dict = StateDictCompressionWrapper(self.dl_c_config, model_diff, keys).compress().decompress()
                base_model = [base + diff for base, diff in zip(base_model, model_diff_state_dict.weights)]
            self.next_client_model = [t.numpy() for t in base_model] 
        else:
            # If no compressor is applied to the base model, then clients will have the exact server-side model
            self.next_client_model = copy.deepcopy(self.server_models_queue[-1])


    def client_completion_handler(self, results):
        """We may need to keep all updates from clients,
        if so, we need to append results to the cache

        Args:
            results (dictionary): client's training result

        """
        # Format:
        #       -results = {'client_id':client_id, 'update_weight': model_param, 'update_gradient': gradient_param, 'moving_loss': round_train_loss,
        #       'trained_size': count, 'wall_duration': time_cost, 'success': is_success 'utility': utility}

        if self.args.gradient_policy in ["q-fedavg"]:
            self.client_training_results.append(results)
        # Feed metrics to client sampler
        self.stats_util_accumulator.append(results["utility"])
        self.loss_accumulator.append(results["moving_loss"])

        self.client_manager.register_feedback(
            results["client_id"],
            results["utility"],
            auxi=math.sqrt(results["moving_loss"]),
            time_stamp=self.round,
            duration=self.virtual_client_clock[results["client_id"]]["computation"]
            + self.virtual_client_clock[results["client_id"]]["communication"],
        )

        # ================== Aggregate weights ======================
        self.update_lock.acquire()

        self.model_in_update += 1
        self.update_weight_aggregation(results)
        
        self.update_lock.release()

    def update_weight_aggregation(self, results):
        """May aggregate client updates on the fly"""

        # ===== initialize compressed gradients =====
        if self._is_first_result_in_round():
            self.compressed_gradient = [
                torch.zeros_like(param.data)
                .to(device=self.device, dtype=torch.float32)
                for param in self.model_wrapper.get_model().state_dict().values()
            ]
            self.residual = [
                torch.zeros_like(param.data)
                .to(device=self.device, dtype=torch.float32)
                for param in self.model_wrapper.get_model().state_dict().values()
            ]

        keys = list(self.model_wrapper.get_model().state_dict().keys())

        # Perform decompression if needed
        # For simulation efficiency, the compressed then decompressed gradient is transmitted
        # so there is no need for decompression
        update_gradient = [
            torch.from_numpy(param).to(device=self.device) for param in results["update_gradient"]
        ]

        # Aggregate gradients with specific gradient weights
        gradient_weight = self.get_gradient_weight(results["client_id"])
        
        for idx, key in enumerate(keys):
            if is_batch_norm_layer(key):
                # Batch norm layer is not weighted
                self.compressed_gradient[idx] += update_gradient[idx] * (1.0 / self.tasks_round)
            else:
                self.compressed_gradient[idx] += update_gradient[idx] * gradient_weight

        # All clients are done
        if self._is_last_result_in_round():
            full_compressed_gradient = copy.deepcopy(self.compressed_gradient)
            keys = self.model_wrapper.get_keys()
            if self.args.enable_server_residuals:
                for i, _ in enumerate(keys):
                    self.compressed_gradient[i] += self.residual[i]

            self.apply_and_update_mask()

            update_state_dict = StateDictCompressionWrapper(self.dl_c_config, self.compressed_gradient, keys).compress().decompress()
            self.compressed_gradient = update_state_dict.weights
            self.size_eval.compressed_model_size_dl = update_state_dict.compressed_size   
            
            if self.args.enable_server_residuals:
                for i, _ in enumerate(keys):
                    self.residual[i] = full_compressed_gradient[i] - self.compressed_gradient[i]

            spar_ratio = SizeEvaluator.check_sparsification_ratio([self.compressed_gradient])
            mask_ratio = SizeEvaluator.check_sparsification_ratio([self.shared_mask])
            logging.info(f"Gradients sparsification: {spar_ratio}")
            logging.info(f"Mask sparsification: {mask_ratio}")

            # ==== update global model =====
            model_state_dict = self.model_wrapper.get_model().state_dict()
            for idx, param in enumerate(model_state_dict.values()):
                param.data = (
                    param.data.to(device=self.device).to(dtype=torch.float32)
                    - self.compressed_gradient[idx]
                )
            
            self.model_wrapper.get_model().load_state_dict(model_state_dict)

            # ===== update mask list =====
            mask_list = []
            for p_idx, key in enumerate(self.model_wrapper.get_model().state_dict().keys()):
                mask = (self.compressed_gradient[p_idx] != 0).to(
                    dtype=torch.bool,
                    device=torch.device("cpu")
                )
                mask_list.append(mask)

            self.mask_record_list.append(mask_list)

            if self.round > 50: # Cleanup mask record list 
                self.mask_record_list[self.round - 50] = None

            # ==== update quantized update =====

            curr_weights = [p.to(dtype=torch.float32) for p in model_state_dict.values()]
            if self.fl_method == DOCOFL and self.round % self.anchor_deployment_rate == 1 and not self.args.use_latest_model:
                base_state_dict = StateDictCompressionWrapper(self.base_c_config, curr_weights, keys).compress().decompress()
                self.server_anchors_queue.append(base_state_dict.weights)
                self.size_eval.compressed_model_size_base = base_state_dict.compressed_size

            self.server_models_queue.append(curr_weights)
            self.server_updates_queue.append(self.compressed_gradient)
        

    def get_gradient_weight(self, client_id):
        # Note that the gradient weight is normalized to be in the interval [0, 1] by multiplying 1 / N
        weight = 0 
        if self.sampling_strategy == "STICKY":
            if self.round <= 1:
                weight = 1.0 / float(self.tasks_round)
            elif client_id in self.sampled_sticky_client_set:
                weight = (1.0 / float(self.feasible_client_count)) * (
                    1.0
                    / (
                        (float(self.tasks_round) - float(self.sticky_group_change_num))
                        / float(self.sticky_group_size)
                    )
                ) # (1 / N) * (S / C), where prob of client being sampled is C / S
            else:
                weight = (1.0 / float(self.feasible_client_count)) * (
                    1.0
                    / (
                        float(self.sticky_group_change_num)
                        / (
                            float(self.feasible_client_count)
                            - float(self.sticky_group_size)
                        )
                    )
                ) # (1 / N) * ((N - S) / (K - C)), where prob of client being sampled is (K-C)/(N-S)
        else:
            """
            [FedAvg] "Communication-Efficient Learning of Deep Networks from Decentralized Data".
            H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Aguera y Arcas. AISTATS, 2017
            """
            # probability of client being sampled is 1 / N
            # the weight is actually 1 / (K * (1 / N)) which when normalized by 1 / N leads to 1 / K
            weight = 1.0 / self.tasks_round # (1 / K)
            
        return weight

    def apply_and_update_mask(self):
        if self.fl_method not in [STC, GLUEFL]:
            return
        
        compressor_tot = TopKCompressor(compress_ratio=self.total_mask_ratio)
        compressor_shr = TopKCompressor(compress_ratio=self.shared_mask_ratio)

        state_dict_keys = self.model_wrapper.get_keys()

        for idx, key in enumerate(state_dict_keys):
            if is_batch_norm_layer(key):
                continue

            # --- STC ---
            if (
                self.fl_method in [STC]
                or self.round % self.regenerate_epoch == 1
            ):
                # local mask
                self.compressed_gradient[idx], ctx_tmp = compressor_tot.compress(
                    self.compressed_gradient[idx]
                )

                self.compressed_gradient[idx] = compressor_tot.decompress(
                    self.compressed_gradient[idx], ctx_tmp
                )
            else:
                # shared + local mask
                update_mask = self.compressed_gradient[idx].clone().detach()
                update_mask[self.shared_mask[idx] == True] = numpy.inf
                update_mask, ctx_tmp = compressor_tot.compress(update_mask)
                update_mask = compressor_tot.decompress(update_mask, ctx_tmp)
                update_mask = update_mask.to(torch.bool)
                self.compressed_gradient[idx][update_mask != True] = 0.0

        # --- update shared mask ---
        for idx, key in enumerate(state_dict_keys):
            if is_batch_norm_layer(key):
                continue

            determined_mask = self.compressed_gradient[idx].clone().detach()
            determined_mask, ctx_tmp = compressor_shr.compress(determined_mask)
            determined_mask = compressor_shr.decompress(determined_mask, ctx_tmp)
            self.shared_mask[idx] = determined_mask.to(torch.bool)

    def get_client_conf(self, client_id):
        """Training configurations that will be applied on clients,
        developers can further define personalized client config here.

        Args:
            client_id (int): The client id.

        Returns:
            dictionary: TorchClient training config.

        """
        conf = {
            "learning_rate": self.args.learning_rate,
            "download_compressor_type": self.download_compressor_type,
            "upload_compressor_type": self.download_compressor_type
        }
        return conf
    
    def create_client_task(self, executorId):
        """Issue a new client training task to specific executor

        Args:
            executorId (int): Executor Id.

        Returns:
            tuple: Training config for new task. (dictionary, PyTorch or TensorFlow module)

        """
        next_client_id = self.resource_manager.get_next_task(executorId)
        train_config = None
        if next_client_id != None:
            config = self.get_client_conf(next_client_id)
            train_config = {
                "client_id": next_client_id,
                "task_config": config,
                # TODO move following to be part of task_config
                "agg_weight": (
                    self.get_gradient_weight(next_client_id)
                    * float(self.feasible_client_count)
                ),
            }
        return train_config, self.get_train_update_virtual(next_client_id)
    
    def get_train_update_virtual(self, client_id):
        """
        Transfer the client-side model that already applies the quantized update 
        """
        
        if (self.download_compressor_type == CompressorType.NO_COMPRESSION) or self.round == 1:
            return self.model_wrapper.get_weights()
        else:
            return self.next_client_model

    def CLIENT_PING(self, request, context):
        """Handle client ping requests

        Args:
            request (PingRequest): Ping request info from executor.

        Returns:
            ServerResponse: Server response to ping request

        """
        # NOTE: client_id = executor_id in deployment,
        # while multiple client_id may use the same executor_id (VMs) in simulations
        executor_id, client_id = request.executor_id, request.client_id
        response_data = response_msg = commons.DUMMY_RESPONSE

        if len(self.individual_client_events[executor_id]) == 0:
            # send dummy response
            current_event = commons.DUMMY_EVENT
            response_data = response_msg = commons.DUMMY_RESPONSE
        else:
            current_event = self.individual_client_events[executor_id].popleft()
            if current_event == commons.CLIENT_TRAIN:
                response_msg, response_data = self.create_client_task(executor_id)

                if response_msg is None:
                    current_event = commons.DUMMY_EVENT
                    if self.experiment_mode != commons.SIMULATION_MODE:
                        self.individual_client_events[executor_id].append(
                            commons.CLIENT_TRAIN
                        )
            elif current_event == commons.MODEL_TEST:
                response_msg, response_data = self.get_test_config(client_id)
            elif current_event == commons.UPDATE_MODEL:
                # Transfer the entire model weights instead of partial model weights in real-life
                response_data = self.model_wrapper.get_weights()
            elif current_event == commons.UPDATE_MASK:
                response_data = self.get_shared_mask()
            elif current_event == commons.SHUT_DOWN:
                response_msg = self.get_shutdown_config(executor_id)

        response_msg, response_data = self.serialize_response(
            response_msg
        ), self.serialize_response(response_data)
        # NOTE: in simulation mode, response data is pickle for faster (de)serialization
        response = job_api_pb2.ServerResponse(
            event=current_event, meta=response_msg, data=response_data
        )
        if current_event != commons.DUMMY_EVENT:
            logging.info(f"Issue EVENT ({current_event}) to EXECUTOR ({executor_id})")

        return response

    def select_participants(self):
        """
        Selects next round's participants depending on the sampling strategy

        If sampling_strategy == "STICKY", then use sticky sampling and, if possible, ensure that the number of sticky
        and change/new/non-sticky clients is the same as specified in the config args.
        Relevant args include num_participants, sticky_group_size, sticky_group_change_num, overcommitment, overcommit_weight

        Otherwise, use uniform sampling.
        """
        if self.sampling_strategy == "STICKY":
            if self.enable_prefetch:

                logging.info(f"avg time {self.exp_mov_avg_dur}")
                if self.round > self.warmup_round:
                    self.client_manager.presample_sticky(self.global_virtual_clock)
                    # Schedule prefetch
                    training_round = self.round + self.max_prefetch_round
                    self.prefetch_schedules_dict[training_round], self.min_prefetch_round[training_round] = self.schedule_prefetch(self.client_manager.schedule_clients_queue.popleft(), self.round)
                    logging.info(f"Determined prefetch schedule {self.prefetch_schedules_dict[training_round]}")

                if self.round <= self.warmup_round + self.max_prefetch_round:
                    self.sampled_sticky_clients, self.sampled_changed_clients = self.client_manager.select_participants_sticky(self.global_virtual_clock)
                else:
                    self.sampled_sticky_clients, self.sampled_changed_clients = self.client_manager.get_presampled_sticky_clients()
            else:
                self.sampled_sticky_clients, self.sampled_changed_clients = (
                    self.client_manager.select_participants_sticky(
                        cur_time=self.global_virtual_clock
                    )
                )

            self.sampled_sticky_client_set = set(self.sampled_sticky_clients)

            # Choose fastest online changed clients
            (
                change_to_run,
                change_stragglers,
                change_lost,
                change_virtual_client_clock,
                change_round_duration,
                change_flatten_client_duration,
            ) = self.tictak_client_tasks(
                self.sampled_changed_clients,
                (
                    self.args.sticky_group_change_num
                    if self.round > 1
                    else self.args.num_participants
                ),
            )
            logging.info(
                f"Selected change participants to run: {sorted(change_to_run)}\nchange stragglers: {sorted(change_stragglers)}\nchange lost: {sorted(change_lost)}"
            )

            # Randomly choose from online sticky clients
            sticky_to_run_count = (
                self.args.num_participants - self.args.sticky_group_change_num
            )
            (
                sticky_fast,
                sticky_slow,
                sticky_lost,
                sticky_virtual_client_clock,
                _,
                sticky_flatten_client_duration,
            ) = self.tictak_client_tasks(
                self.sampled_sticky_clients, sticky_to_run_count
            )
            all_sticky = sticky_fast + sticky_slow
            all_sticky.sort(key=lambda c: sticky_virtual_client_clock[c]["round"])
            faster_sticky_count = sum(
                1
                for c in all_sticky
                if sticky_virtual_client_clock[c]["round"] <= change_round_duration
            )

            if faster_sticky_count >= sticky_to_run_count:
                sticky_to_run = random.sample(
                    all_sticky[:faster_sticky_count], sticky_to_run_count
                )
            else:
                extra_sticky_clients = sticky_to_run_count - faster_sticky_count
                sticky_to_run = all_sticky[: faster_sticky_count + extra_sticky_clients]
                logging.info(
                    f"Sticky group has only {faster_sticky_count} clients that are faster than change group, fastest sticky clients will be used"
                )

            if (
                len(self.sampled_sticky_clients) > 0
            ):  # There are no sticky clients in round 1
                slowest_sticky_client_id = max(
                    sticky_to_run, key=lambda k: sticky_virtual_client_clock[k]["round"]
                )
                sticky_round_duration = sticky_virtual_client_clock[
                    slowest_sticky_client_id
                ]["round"]
            else:
                sticky_round_duration = 0

            sticky_ignored = [c for c in all_sticky if c not in sticky_to_run]

            logging.info(
                f"Selected sticky participants to run: {sorted(sticky_to_run)}\nunselected sticky participants: {sorted(sticky_ignored)}\nsticky lost: {sorted(sticky_lost)}"
            )

            # Combine sticky and changed clients together
            self.clients_to_run = sticky_to_run + change_to_run
            self.round_stragglers = sticky_ignored + change_stragglers
            self.round_lost_clients = sticky_lost + change_lost
            self.virtual_client_clock = {
                **sticky_virtual_client_clock,
                **change_virtual_client_clock,
            }
            self.round_duration = max(sticky_round_duration, change_round_duration)
            self.flatten_client_duration = numpy.array(
                sticky_flatten_client_duration + change_flatten_client_duration
            )
            self.clients_to_run.sort(
                key=lambda k: self.virtual_client_clock[k]["round"]
            )
            self.slowest_client_id = self.clients_to_run[-1]

            # Make sure that there are change_num number of new clients added each epoch
            if self.round > 1:
                self.client_manager.update_sticky_group(change_to_run)

        else:
            if self.enable_prefetch:
                logging.info(f"avg time {self.exp_mov_avg_dur}")
                if self.round > self.warmup_round:
                    self.client_manager.presample(self.global_virtual_clock)
                    # Schedule prefetch
                    training_round = self.round + self.max_prefetch_round
                    if self.args.use_fixed_round:
                        clients = self.client_manager.schedule_clients_queue.popleft()
                        self.prefetch_schedules_dict[training_round] = {i : self.round for i in clients}
                        self.min_prefetch_round[training_round] = self.round
                    else:
                        self.prefetch_schedules_dict[training_round], self.min_prefetch_round[training_round] = self.schedule_prefetch(self.client_manager.schedule_clients_queue.popleft(), self.round)
                    logging.info(f"Determined prefetch schedule {self.prefetch_schedules_dict[training_round]}")

                if self.round <= self.warmup_round + self.max_prefetch_round:
                    self.sampled_participants = self.client_manager.select_participants(self.global_virtual_clock)
                else:
                    self.sampled_participants = self.client_manager.get_presampled_clients()
            else:
                self.sampled_participants = sorted(
                    self.client_manager.select_participants(
                        cur_time=self.global_virtual_clock,
                    )
                )
            logging.info(f"Sampled clients: {sorted(self.sampled_participants)}")

            (
                self.clients_to_run,
                self.round_stragglers,
                self.round_lost_clients,
                self.virtual_client_clock,
                self.round_duration,
                self.flatten_client_duration,
            ) = self.tictak_client_tasks(
                self.sampled_participants, self.args.num_participants
            )
            self.slowest_client_id = self.clients_to_run[-1]
            self.flatten_client_duration = numpy.array(self.flatten_client_duration)

        logging.info(
            f"Selected participants to run: {sorted(self.clients_to_run)}\nstragglers: {sorted(self.round_stragglers)}\nlost: {sorted(self.round_lost_clients)}"
        )

    def schedule_prefetch(self, clients: List[int], cur_round: int):
        prefetch_schedule = {c: cur_round for c in clients}
        r_min = cur_round
        T_percentile = numpy.inf
        clients_can_finish = clients.copy()        
        for r in range(cur_round, cur_round+self.max_prefetch_round+1):
            fetch_times = {i: self.estimate_fetch_time(i, r, r_min) for i in clients_can_finish}
            if T_percentile == numpy.inf:
                T_percentile = numpy.percentile(list(fetch_times.values()), 100 * ((1 + self.args.prefetch_scheduler_beta * (self.args.overcommitment - 1)) / self.args.overcommitment)) # Equal to T_max if OC = 1
                clients_can_finish = [i for i in clients_can_finish if fetch_times[i] < T_percentile]
            
            clients_can_finish = [i for i in clients_can_finish if fetch_times[i] <= T_percentile]
            if len(clients_can_finish) ==  len(clients):
                r_min = min(r, cur_round + self.max_prefetch_round)
                T_percentile = numpy.percentile(list(fetch_times.values()), 100 * ((1 + self.args.prefetch_scheduler_beta * (self.args.overcommitment - 1)) / self.args.overcommitment)) # Equal to T_max if OC = 1
            T_curr_max = max(list(fetch_times.values())) if len(fetch_times) > 0 else -1 # Time needed for roll back client
            T_succ_max = max([fetch_times[i] for i in clients_can_finish]) # Time needed for successful client
            logging.info(
f"""Prefetch scheduler:
{len(clients_can_finish)} client need no more than {cur_round + self.max_prefetch_round - r} prefetch rounds
Rollback {len(fetch_times) - len(clients_can_finish)} clients to prefetch at earlier round
Cur r_min {r_min} T_all_max {T_curr_max} T_succ_max {T_succ_max} T_percentile {T_percentile}"""
            )
            for i in clients_can_finish:
                prefetch_schedule[i] = r
        return prefetch_schedule, r_min

    def estimate_fetch_time(self, i: int, r: int, r_min: int):
        # logging.info(f"Estimate fetch time i {i} r {r} r_min {r_min}")
        dl_bw = self.client_manager.get_dl_bw_bits(i)
        round_download_amount = self.exp_mov_avg_dur * dl_bw

        update_size = self.size_eval.estimate_size_base_and_delta(r_min, r, self.last_update_index[i])
        train_round = self.round + self.max_prefetch_round
        l = r
        time_budget = 0
        for j in range(r, train_round):
            time_budget = self.exp_mov_avg_dur
            # logging.info(f"1 update size {update_size} time_budge {time_budget} j {j} l {l}")
            time_budget = max(0, time_budget - (update_size / dl_bw))
            # logging.info(f"2 update size {update_size} time_budge {time_budget} j {j} l {l}")
            if time_budget > 0:
                # logging.info(f"3a1 update size {update_size} time_budge {time_budget} j {j} l {l}")
                # update_size = max(0, self.size_eval.estimate_delta_size(l, j-1) - time_budget * dl_bw)
                update_size = self.size_eval.estimate_delta_size(l, j-1)
                # logging.info(f"3a2 update size {update_size} time_budge {time_budget} j {j} l {l}")
                update_size = max(0, update_size - time_budget * dl_bw)
                l = j
                time_budget = 0
                # logging.info(f"3a3 update size {update_size} time_budge {time_budget} j {j} l {l}")
            else:
                update_size -= round_download_amount
                # logging.info(f"3b update size {update_size} time_budge {time_budget} j {j} l {l}")
            # logging.info(f"4 update size {update_size} time_budge {time_budget} j {j} l {l}")
        final_delta = self.size_eval.estimate_delta_size(l, train_round-1)
        if self.fl_method == DOCOFL or self.prefetch_compressor_type != CompressorType.NO_COMPRESSION:
            final_delta = self.size_eval.compressed_model_size_dl
                    
        return (update_size + final_delta)/ dl_bw

    def calculate_exp_weighted_avg(self, new_dur):
        alpha = self.args.avg_alpha
        if self.exp_mov_avg_dur is None:
            self.exp_mov_avg_dur = new_dur
        else:
            self.exp_mov_avg_dur = alpha * new_dur + (1 - alpha) * self.exp_mov_avg_dur

    def round_completion_handler(self):
        """Triggered upon the round completion, it registers the last round execution info,
        broadcast new tasks for executors and select clients for next round.
        """
        self.global_virtual_clock += self.round_duration
        self.round += 1

        last_round_avg_util = sum(self.stats_util_accumulator) / max(
            1, len(self.stats_util_accumulator)
        )
        # assign avg reward to explored, but not ran workers
        for client_id in self.round_stragglers:
            self.client_manager.register_feedback(
                client_id,
                last_round_avg_util,
                time_stamp=self.round,
                # TODO switch to using both download and upload time when we eventually test Oort
                duration=self.virtual_client_clock[client_id]["computation"] + self.virtual_client_clock[client_id]["communication"], 
                success=False,
            )

        avg_loss = sum(self.loss_accumulator) / max(1, len(self.loss_accumulator))
        logging.info(
            f"Wall clock: {round(self.global_virtual_clock)} s, round: {self.round-1}, Planned participants: "
            + f"{len(self.sampled_participants)}, Succeed participants: {len(self.stats_util_accumulator)}, Training loss: {avg_loss}"
        )

        # dump round completion information to tensorboard
        if len(self.loss_accumulator):
            self.log_train_result(avg_loss)

        if self.round > 1:
            self.round_evaluator.record_round_completion(
                self.clients_to_run,
                self.round_stragglers + self.round_lost_clients,
                self.slowest_client_id,
            )
            self.round_evaluator.print_stats()
            self.round_evaluator.start_new_round()
            self.calculate_exp_weighted_avg(self.round_evaluator.round_durations_aggregated[-1])

        # Perform one more profiling of sparsification
        self.size_eval.clean_spar_ratio_cache()
        # if self.round > 1 and self.round <= self.size_eval.no_calc_after and self.fl_method not in [FEDAVG, DOCOFL]:
        #     self.size_eval.calculate_delta_size(1, self.round - 1, self.mask_record_list)
        logging.info(f"avg_delta_sizes {self.size_eval.avg_delta_sizes}")
        
        logging.info(f"Start round {self.round}")
        # Select next round's participants
        self.select_participants()

        # Issue requests to the resource manager; Tasks ordered by the completion time
        self.resource_manager.register_tasks(self.clients_to_run)
        self.tasks_round = len(self.clients_to_run)

        # Update executors and participants
        if self.experiment_mode == commons.SIMULATION_MODE:
            self.sampled_executors = list(self.individual_client_events.keys())
        else:
            self.sampled_executors = [str(c_id) for c_id in self.sampled_participants]

        self.model_in_update = 0
        self.test_result_accumulator = []
        self.stats_util_accumulator = []
        self.client_training_results = []
        self.loss_accumulator = []
        self.update_default_task_config()

        if self.round >= self.args.rounds:
            self.broadcast_aggregator_events(commons.SHUT_DOWN)
        elif self.round % self.args.eval_interval == 0 or self.round == 10:
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            self.broadcast_aggregator_events(commons.UPDATE_MASK)
            self.broadcast_aggregator_events(commons.MODEL_TEST) # Issues a START_ROUND after testing completes
        else:
            self.broadcast_aggregator_events(commons.UPDATE_MODEL)
            self.broadcast_aggregator_events(commons.UPDATE_MASK)
            self.broadcast_aggregator_events(commons.START_ROUND)

    def event_monitor(self):
        """Activate event handler according to the received new message"""
        logging.info("Start monitoring events ...")

        while True:
            # Broadcast events to clients
            if len(self.broadcast_events_queue) > 0:
                current_event = self.broadcast_events_queue.popleft()

                if current_event in (
                    commons.UPDATE_MODEL,
                    commons.MODEL_TEST,
                    commons.UPDATE_MASK,
                ):
                    self.dispatch_client_events(current_event)

                elif current_event == commons.START_ROUND:
                    self.dispatch_client_events(commons.CLIENT_TRAIN)

                elif current_event == commons.SHUT_DOWN:
                    self.dispatch_client_events(commons.SHUT_DOWN)
                    break

            # Handle events queued on the aggregator
            elif len(self.server_events_queue) > 0:
                client_id, current_event, meta, data = self.server_events_queue.popleft()

                if current_event == commons.UPLOAD_MODEL:
                    self.client_completion_handler(self.deserialize_response(data))
                    if len(self.stats_util_accumulator) == self.tasks_round:
                        self.round_completion_handler()

                elif current_event == commons.MODEL_TEST:
                    self.testing_completion_handler(
                        client_id, self.deserialize_response(data)
                    )

                else:
                    logging.error(f"Event {current_event} is not defined")

            else:
                # execute every 100 ms
                time.sleep(0.1)


if __name__ == "__main__":
    aggregator = PrefetchAggregator(parser.args)
    aggregator.run()
