import torch
import numpy as np
import torch.nn as nn
import os
import sys
import copy
import threading
import pickle
import logging
import numpy as np
from pprint import pformat

import utils
import plotting
import argparse
from batch_systems import *
from load_models import load_model
# from dataloader.data_loaders import load_data
from utils import set_seeds, get_batch, parse_profile
from utils import get_queuing_delay, get_ramp_latencies, get_remaining_rate
from utils import serve_batch, tune_threshold, earlyexit_inference, get_optimal_exitable_ramps
from utils import earlyexit_infer_per_sample, get_batch_perf, get_overall_exit_info, get_ramp_scores, get_ramp_utility
from utils import ramp_addition, ramp_pruning, ramp_pruning_garbage_only, ramp_addition_tail_latency
sys.path.insert(1, os.path.join(os.getcwd(), 'profiling'))  # for loading profile pickles
# from profiler import TIDSProfiler

# suppress matplotlib font manager logger
logging.getLogger('matplotlib.font_manager').disabled = True

# format string for logging
LOG_FORMAT = '%(asctime)s [%(name)s:%(levelname)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
# utility threshold for pruning. our system is not sensitive to this
# hyperparameter, but we set it to 0.1 instead of 0 to prevent noise.
PRUNE_THRESHOLD = 0.1  # -0.05
# max accuracy loss (compared to the original model output) we can afford
ACC_LOSS_BUDGET_ACTUAL = 0.01  # 1.5% acc loss
# max accuracy loss (compared to the original model output) we can afford
# when doing threshold tuning (0.5% slack)
ACC_LOSS_BUDGET_TUNING = 0.0001  # 1% acc loss
# max tail latency degradation (compared to the original model tail latency) we can afford
TAIL_LATENCY_BUDGET = 0.05
RAMP_CHECK_INTERVAL = 30 
NUM_RAMP_BUDGET = 3


class Controller():
    def __init__(self, args, log_level="INFO"):
        # pytorch model instance (?)
        self._model = None
        # latency/memory profile of the model with all exits enabled
        # NOTE(ruipan): for now, all profiles assume batch size bs=1
        self._model_profile = None
        # key: batch size ranging from 1 to 64. each item is a
        # list of tuples for easier querying of early exit latencies.
        # index x: (latency of vanilla model up to ramp x, latency of ramp x).
        # last entry: (latency of vanilla model, None).
        self._latency_calc_list = {}
        # key: layer name, value: output shape
        self._layer_output_size = {}
        # keeps a queue of incoming serving requests
        self._requests_queue = None
        # current early-exit configuration
        self._ramp_ids = None  # list of 0-indexed sorted ramp IDs
        self._shadow_ramp_id = None  # set of shadow ramp IDs
        self._shadow_ramp_idx = None  # index of shadow ramp in ramp_ids
        self._shadow_ramp_num_ = 0  # number of good sample batches for shadow ramp
        self._thresholds = None  # list of exit thresholds associated with each ramp
        # entropies ("conf") / predictions ("acc") of past data, each of which is a
        # list (len: num ramps + 1) of lists (len: num samples)
        self._historical_data = None
        self._historical_exit_rates = None
        self._historical_ramp_utility = None
        # whether to use pre-computed entropies (stored in pickle files) during serving
        self._simulate = False
        
        # seed things just in case
        set_seeds()
        # configure logger
        self._logger = logging.getLogger(__name__)
        logging_level_dict = {"INFO": logging.INFO, "DEBUG": logging.DEBUG}
        self._logger.setLevel(logging_level_dict[log_level])
        # overwrite previous output log file
        # ch = logging.FileHandler(f"../logs/{args.model}_{args.dataset}.log", mode="w+")
        # ch.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))
        # self._logger.addHandler(ch)
        ch = logging.FileHandler(
            f"./logs/output_{args.arch}_{args.dataset}.log", mode="w+")
        ch.setFormatter(logging.Formatter(fmt=LOG_FORMAT, datefmt=DATE_FORMAT))
        self._logger.addHandler(ch)
        self._logger.debug(f"Logger set up!")
        self._logger.info(f"args: {args}")

        # multiprocessing lock
        self._lock = threading.Lock()

        # temporary code for testing
        self._args = args
        self._dataloader = None
        self._latest_deactivated_ramp = None

        self._recovery_mode = False
        self._last_violation_idx = None

        self._device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")

        self.nlp = False
        if self._args.dataset != "video":
            self.nlp = True
            global NUM_RAMP_BUDGET
            global RAMP_CHECK_INTERVAL
            NUM_RAMP_BUDGET = 2
            RAMP_CHECK_INTERVAL = 100


    def get_batch_decision(self):

        """
        Get batch decision from batch decision pickle file or generate batch decision
        """
        if os.path.exists(self._args.batch_decision_path):
            print(f"self._args.batch_decision_path: {self._args.batch_decision_path}")
            with open(self._args.batch_decision_path, "rb") as f:
                self._batch_info = pickle.load(f)
                self._batch_decision = self._batch_info['batching_decision']
        else:
            assert os.path.exists(self._args.batch_decision_path), f"batch decision path {self._args.batch_decision_path} does not exist"
            

    def plot_latency_cdfs(self):

        per_request_stats = self._batch_info["per_request_stats"]
        total_num_requests = self._batch_info["total_num_requests"]
        batching_scheme = self._batch_info["batching_scheme"]
        arch = self._batch_info["arch"]
        slo = self._batch_info["slo"]
        avg_qps = self._batch_info["avg_qps"]
        total_time = self._batch_info["end_time"]
        batch_decision = self._batch_info["batching_decision"]

        vanilla_median, apparate_median, apparate_optimal_median = \
                get_latency_plots(self._args.dataset, True, batching_scheme, arch, slo, avg_qps, \
                    per_request_stats, total_num_requests, total_time, 1/self._args.qps, batch_decision, self._all_vanilla_latencies)

        self._logger.info(f"apparate median savings {(vanilla_median - apparate_median) / vanilla_median }" )

    def bootstrap(self, generate_pickle=False):
        """Set up the training environment for model dataset pair

        Assume if latency profile is found
            model is already trained and entropy profile is generated

        if latency profile is found
            1. Load the entropy pickle file and latency profile
        else 
            1. Load the vanilla model and inject ramps.
            2. Train all the ramps dump the model and generate entropy pickle file and latency profile

        ramp addition: generate initial ramp ids and thresholds

        Load the model with the ramp ids and set the thresholds

        """
        """TODO(ruipan):
        load the model with all exits, run dataset through the model, and record
        the entropies of all samples at all ramps in a pickle file.
        Later, ramp_addition is done on this pickle file in simulation (w/o having
        to actually serve the model).
        """

        for batch_size in utils.supported_batch_sizes:
            if self._args.dataset == 'video':
                profile_path = os.path.join(
                    "../", self._args.profile_dir, f"{self._args.arch.split('_')[0]}_{batch_size}_earlyexit_profile.pickle")
            else:
                profile_path = os.path.join(
                    "../", self._args.profile_dir, f"{self._args.arch}_{batch_size}_earlyexit_profile.pickle")

            if os.path.exists(profile_path):
                with open(profile_path, 'rb') as f:
                    profile = pickle.load(f)
                if not any([x in self._args.arch for x in ["vgg", "resnet"]]):
                    self._latency_calc_list[batch_size] = parse_profile(profile)
                else:  # NOTE(ruipan): all cv models' branched_module latencies aren't properly recorded.
                    # workaround: load vanilla model profile for vanilla model's runtime, and manually add
                    # the ramps' overheads
                    """
                    latency_calc_list format: list of tuples. for each ramp, the tuple is 
                    (vanilla_latency_before_ramp, ramp_latency,). In addition, 
                    (vanilla_model_latency, None,) is appended to latency_calc_list.
                    """
                    latency_calc_list = []
                    vanilla_profile_path = profile_path.replace(f"_earlyexit", '')
                    # vanilla_profile_path = os.path.join("./profile_pickles_bs", f"{self._args.arch.split('_')[0]}_{batch_size}_profile.pickle")
                    with open(vanilla_profile_path, "rb") as f:
                        vanilla_profile = pickle.load(f)
                    # print(f"vanilla_profile_path {vanilla_profile_path}")
                    """
                    Traverse through all named modules in vanilla_profile.
                    for every full name, check if the corresponding module in the ee profile
                    is a BranchPoint. If so, add to latency_calc_list.
                    """
                    all_childrens_fullname = vanilla_profile.get_all_childrens_fullname()
                    # print(f"all_childrens_fullname {all_childrens_fullname}")
                    for child_fullname in all_childrens_fullname:
                        module_in_ee = profile.get_child_with_name(child_fullname.split('.'))  # module with same name in ee profile
                        if module_in_ee is not None:
                            if module_in_ee.type == "BranchPoint":
                                # print(f"child_fullname {child_fullname}, found twin module in ee with name {module_in_ee.full_name} that's a r=branchpoint")
                                module_in_vanilla = vanilla_profile.get_child_with_name(child_fullname.split('.'))
                                assert module_in_vanilla.full_name == module_in_ee.full_name
                                latency_calc_list.append((
                                    module_in_vanilla.vanilla_latency_up_until_me, 
                                    module_in_ee.get_child_with_name(["branch_net"]).fwd_latency,
                                ))
                            
                    latency_calc_list.append((vanilla_profile.fwd_latency_orig, None,))
                    # print(f"latency_calc_list {latency_calc_list}")
                    self._latency_calc_list[batch_size] = latency_calc_list
            else:
                raise Exception(
                    f"No profile found for model {self._args.arch} at {profile_path}!")
            
        self._batch_decision = None
        if self._args.batching_scheme != 'uniform':
            self.get_batch_decision()

        if generate_pickle:  # activate all ramps
            self._total_num_ramps = len(entropy_dict['conf'])
            self._ramp_ids = list(range(self._total_num_ramps - 1))
            self._thresholds = [0.0] * len(self._ramp_ids)
        elif self._args.bootstrap_pickle_path is None:
            self._total_num_ramps = 13
            self._ramp_quota = 0
            self._latest_possible_ramp = self.get_boundary(latency_calc_list=self._latency_calc_list[self._args.batch_size])
            self._ramp_ids = [5, 8]
            self._thresholds = [1.0, 1.0]
        else:
            with open(self._args.bootstrap_pickle_path, 'rb') as f:
                entropy_dict = pickle.load(f)

            self._ramp_ids, self._thresholds, latency_savings, acc, exit_rate, (self._ramp_efficacy_order, ramp_efficacies) = ramp_addition_tail_latency(
                entropy_dict,
                latency_calc_list=self._latency_calc_list[self._args.batch_size],
                # latency_calc_list=self._latency_calc_list[16],
                num_ramp_budget= 1 if ('resnet18' in self._args.arch or 'vgg' in self._args.arch) else NUM_RAMP_BUDGET,
                # num_ramp_budget=1,
                acc_loss_budget=ACC_LOSS_BUDGET_TUNING,
                tail_latency_budget= 0.05 if ('resnet18' in self._args.arch or 'vgg' in self._args.arch) else TAIL_LATENCY_BUDGET
            )
            # print(self._latency_calc_list[self._args.batch_size])
            # exit()
            # overhead = 0.0
            # vanila_latency = self._latency_calc_list[self._args.batch_size][-1][0]
            # for ramp_id in self._ramp_ids:
            #     overhead += self._latency_calc_list[self._args.batch_size][ramp_id][1]
            # self._logger.info(
            #     f"bootstrap: ramp addition with tail latency optimization done, ramp ids: {self._ramp_ids}, thresholds: {self._thresholds}")
            # self._logger.info(
            #     f"expected latency savings: {latency_savings}, expected acc: {acc}, exit rate: {exit_rate} tail latency {overhead / vanila_latency * 100}% worse")

            # # # NOTE: considers vanilla model as a ramp
            self._total_num_ramps = len(entropy_dict['conf'])
            # self._logger.info(
            #     f"bootstrap: total number of ramps: {self._total_num_ramps}")
            # # (ID of first ramp, its associated exit rate), for checking the signal for ramp location changes
            # self._prev_avg_exit_rate_info = (
            #     min(self._ramp_ids), exit_rate)
            # self._ramp_avg_confidence = [np.average(entropy_dict['conf'][ramp_id]) for ramp_id in self._ramp_ids]
            # self._logger.info(f"average ramp confidence {self._ramp_avg_confidence}")
            # self._logger.info(
            #     f"ramp efficacy order {self._ramp_efficacy_order}, ramp efficacies {ramp_efficacies}")
            # optimal_exitable_ramps = \
            #     get_optimal_exitable_ramps(entropy_dict, [i for i in range(self._total_num_ramps)], self._total_num_ramps)
            
            # ramp_ids = self._ramp_ids
            # ramp_acc = [1 - np.mean(entropy_dict['acc'][i]) for i in range(self._total_num_ramps - 1)]
            
            # _ = utils.get_shadow_ramp_order(optimal_exitable_ramps, ramp_ids, self._total_num_ramps, self._latency_calc_list[self._args.batch_size], ramp_acc)

            self._latest_possible_ramp = self.get_boundary(latency_calc_list=self._latency_calc_list[self._args.batch_size])
            print(f"self._latest_possible_ramp: {self._latest_possible_ramp}")
            if self._ramp_ids is None:
                self._ramp_ids = self.get_new_ramps(0, self._latest_possible_ramp, 1)
                self._thresholds = [0.0]*len(self._ramp_ids)
            else:
                l = len(self._ramp_ids)
                self._ramp_ids = self.get_new_ramps(0, self._latest_possible_ramp, l)
                self._thresholds = [0.0]*len(self._ramp_ids)
            self._ramp_quota = 0
            print(self._ramp_ids)
            print(self._latency_calc_list[1])

        if self._args.optimal_exiting:
            self._ramp_ids = list(range(self._total_num_ramps - 1))
            self._thresholds = [0.0] * len(self._ramp_ids)

        self._historical_data = {'conf': [[] for _ in range(self._total_num_ramps)],
                                'acc': [[] for _ in range(self._total_num_ramps)]}

        self._batch_idx = 0
        self._last_latency_improvement = 0.0
        self._curr_latency_improvement = 0.0
        self._after_ramp_adjustment = False

        self._postive_threshold = 0.6
        self._negative_threshold = 0.6

        self._historical_data_size = 4

        self._historical_data = {'conf': [[] for _ in range(self._total_num_ramps)],
                                    'acc': [[] for _ in range(self._total_num_ramps)]}
        self._batch_size_info = []

        self.set_meta_data()

        self._violation_counter = 0

        if not self._simulate:
            # self._model is an EarlyExitModel wrap up of the vanilla model
            self._model, self._tokenizer, self._all_exit_def = \
                load_model(self._args.dataset, self._args.arch,
                        self._args.model_dir, self._args.num_classes,
                        self._args.pretrained, self._args.earlyexit)   
            # now inject the ramps into the model together with pretrained ramp weights
            self._model.activate_ramps(self._ramp_ids, self._all_exit_def)

        self._ramp_history = []
        self._ramp_history.append(self._ramp_ids)

    def get_boundary(self, latency_calc_list):
        """
        get the latest possible ramp location

        Args:
            latency_calc_list (list): a list of latency calculation results
        """
        latest_ramp = 0

        for i in range(self._total_num_ramps - 1):
            latency_config, baseline_latency = get_ramp_latencies(
                    [i], latency_calc_list)
            if latency_config[0] > baseline_latency:
                break
            latest_ramp = i

        return latest_ramp
            


    def set_meta_data(self):
        """
        initialize the meta data of the model e.g. historical data
        """

        self._historical_exit_rates = []
        self._historical_ramp_utility = [[]
                                        for _ in range(self._total_num_ramps)]

        self._acc_violation_info = []
        self._last_latency_improvement = 0.0
        self._curr_latency_improvement = 0.0
        self._good_count = 0
        self._bad_count = 0

    def clear_meta_data(self, ramp_id):
        """
        clear historical data of a given ramp

        Args:
            ramp_id (int): ramp id
        """
        for key, _ in self._historical_data.items():
            self._historical_data[key][ramp_id] = []

        self._historical_ramp_utility[ramp_id] = []

    def setup_serving(self):
        """Set up the serving environment for model dataset pair
        """
        if not self._simulate:
            # # 1. Load the model
            # self._model, tokenizer = load_model(self._args.dataset, self._args.arch, \
            #                             self._args.model_dir, self._args.pretrained, \
            #                             self._ramp_ids, self._args.earlyexit)
            self._model.eval()

            # 2. Load the dataset
            _, self._dataloader = \
                load_data(self._args.dataset, self._args.data_dir,
                        self._args.batch_size, self._args.arch, test_only=True, tokenizer=self._tokenizer)
        else:
            if self._args.simulation_pickle_path is not None:  # TODO: remove hardcode
                pickle_path = self._args.simulation_pickle_path
            else:
                raise ValueError(f"simulation pickle path is not provided")
                # pickle_path = f"../{self._args.dataset}_{self._args.arch}_entropies.pickle"
                # pickle_path = os.path.join(os.getenv("HOME"), f"{self._args.dataset}_{self._args.arch}_entropies.pickle")
            with open(pickle_path, "rb") as f:
                entropy_dict = pickle.load(f)
            self._dataloader = utils.get_batches(
                entropy_dict, batch_size=self._args.batch_size)

    def get_batch(self, batch_size):
        """Get a batch of data from the queue

        Args:
            batch_size (int): number of requests to get from the queue

        Returns:
            tensor: batch of data
            batch_size (int): number of requests in the batch
        """
        pass

    def get_new_ramps(self, left, right, num):
        """Get a list of new ramp ids

        Args:
            left (int): left index of the ramp
            right (int): right index of the ramp
            num (int): number of new ramps to get

        Returns:
            list: list of new ramp ids
        """
        return list(np.linspace(left, right, num=num+2, dtype=int))[1:-1]
            
    def serve(self, simulate: bool = False, store_entropy_pickle: bool = False):
        """Serve a batch of requests

        Args:
            simulate (bool): use pre-computed entropies stored in pickle files
                FIXME(ruipan): simulation/physical mismatch
            store_entropy_pickle (bool): stores the entropies of all data
                at all ramps in a pickle file. Defaults to false.

        Returns:
            tensor: batch of predictions
        """

        """
            While True:
                1. Aquire lock Check if there is a new config
                2. Serve the batch
                3. Update the historical data
                4. Check signals
        """
        self._simulate = simulate
        self._store_entropy_pickle = store_entropy_pickle

        # for plotting
        self._all_latencies = []
        self._all_accuracies = []
        self._all_exit_ramp = []
        self._all_vanilla_latencies = []
        self._threshold_tuning_history = []

        self.bootstrap(store_entropy_pickle)

        if store_entropy_pickle:
            # assert self._args.bootstrap_pickle_path is None, f"Trying to generate an entropy pickle, but also using that entropy pickle for bootstrapping..."
            entropy_dict = {'conf': [[] for _ in range(self._total_num_ramps)],
                            'acc': [[] for _ in range(self._total_num_ramps)]}
            
        online_configs = []

        self.setup_serving()

        configs = []
        
        # with open(f"../{self._args.dataset}_{self._args.arch}_entropies.pickle", 'rb') as f:
        #     debug_entropy_dict = pickle.load(f)
        #     debug_entropy_loader = get_batch(debug_entropy_dict, self._args.batch_size)

        with torch.no_grad():
            # for inputs in self._dataloader
            curr_data = None
            i = 0
            self._dataloader = iter(self._dataloader)
            while True:
                if self._batch_decision is None or self._store_entropy_pickle:
                    curr_data = next(self._dataloader, None)
                    if curr_data is None:
                        break
                    i += 1
                    inputs = curr_data
                else:
                    if i >= len(self._batch_decision):
                        break
                    curr_batch_size = self._batch_decision[i]
                    if not self._simulate:
                        if curr_data is None or curr_data[0].shape[0] <= curr_batch_size:
                            load = True
                        else:
                            load = False

                        if load:
                            try :
                                next_data = next(self._dataloader)
                            except StopIteration:
                                self.setup_serving()
                                self._dataloader = iter(self._dataloader)
                                next_data = next(self._dataloader)

                            if curr_data is None:
                                curr_data = next_data
                            else:
                                for j in range(len(curr_data)):
                                    curr_data[j] = torch.cat((curr_data[j], next_data[j]), dim=0)
                        
                        if curr_data[0].shape[0] < curr_batch_size:
                            continue
                        elif curr_data[0].shape[0] == curr_batch_size:
                            inputs = curr_data
                            curr_data = None
                            i += 1
                        else:
                            inputs = [None] * len(curr_data)
                            indices = torch.tensor(range(0, curr_batch_size))
                            indices_left = torch.tensor(range(curr_batch_size, curr_data[0].shape[0]))
                            for j in range(len(curr_data)):
                                inputs[j] = torch.index_select(curr_data[j], 0, indices)
                                curr_data[j] = torch.index_select(curr_data[j], 0, indices_left)
                            i += 1
                    else:
                        if curr_data is None or len(curr_data['conf'][0]) < curr_batch_size:
                            load = True
                        else:
                            load = False
                        if load:
                            try :
                                next_data = next(self._dataloader)
                            except StopIteration:
                                break
                                self.setup_serving()
                                self._dataloader = iter(self._dataloader)
                                next_data = next(self._dataloader)
                            if curr_data is None:
                                curr_data = next_data
                            else:
                                for key, _ in curr_data.items():
                                    for ramp_id in range(self._total_num_ramps - 1):
                                        curr_data[key][ramp_id] += next_data[key][ramp_id]
                        if len(curr_data['conf'][0]) < curr_batch_size:
                            continue
                        elif len(curr_data['conf'][0]) == curr_batch_size:
                            inputs = curr_data
                            curr_data = None
                            i += 1
                        else:
                            inputs = {'conf': [[] for _ in range(self._total_num_ramps)],
                                        'acc': [[] for _ in range(self._total_num_ramps)]}
                            for key, _ in inputs.items():
                                for ramp_id in range(self._total_num_ramps - 1):
                                    inputs[key][ramp_id] = curr_data[key][ramp_id][:curr_batch_size]
                                    curr_data[key][ramp_id] = curr_data[key][ramp_id][curr_batch_size:]
                            i += 1

                self._lock.acquire()
                if not self._simulate:
                    if len(inputs) == 2:  # CV
                        inputs = inputs[0]
                        inputs = inputs.to(self._device)
                        x, ee_outputs = self._model(inputs)
                        output = ee_outputs + [x]
                        _, target = output[-1].clone().detach().max(dim=1)
                    elif len(inputs) == 4:  # NLP
                        batch = inputs
                        batch = tuple(t.to(self._device) for t in batch)
                        inputs = {
                            "input_ids": batch[0],
                            "attention_mask": batch[1],
                            "labels": batch[3]
                        }
                        if "distilbert" not in self._args.arch:
                            inputs["token_type_ids"] = (
                                batch[2] if self._args.arch in [
                                    "bert-base-uncased", "bert-large-uncased", "xlnet"] else None
                            )  # XLM, DistilBERT and RoBERTa don't use segment_ids
                        # x is a tuple of length 8, containing outputs (e.g., loss, logits, ...) of deebert
                        # ee_outputs: list of length num_ramps, each of which is the output stored in each branchpoint:
                        # (logits, pooled_output). Logits can be processed to obtain preds or entropies.
                        # logits: tensor of torch.Size([batch_size, num_labels])
                        # print(f"inputs {inputs}")
                        x, ee_outputs = self._model(**inputs)

                        # extract logits from (logits, pooled_output) so that we can directly run softmax on the logits
                        output = [ee_output[0].detach().cpu()
                                for ee_output in ee_outputs] + [x[1].detach().cpu()]
                        # NOTE(ruipan): this assumes classification workloads. if regression, use something like np.squeeze
                        # FIXME(ruipan): target is wrong for some datasets (bert-base: rte and mrpc wrong, but sst-2 is correct)
                        _, target = output[-1].max(dim=1)
                        # print(f"output {output}, target {target}")
                    else:
                        raise NotImplementedError

                request_rate = 240
                batch_size = len(inputs["conf"][0]) if self._simulate else target.size(0)
                queuing_delay = get_queuing_delay(request_rate, batch_size)
                batch_size = utils.round_up_batch_size(batch_size)
                ramp_latencies, vanilla_latency = get_ramp_latencies(
                    self._ramp_ids, self._latency_calc_list[batch_size])

                apparate_optimal = False    

                if not self._simulate:
                    batch_meta_data, sample_latencies, sample_acc, sample_exit_points = \
                        earlyexit_infer_per_sample(output, target, self._ramp_ids,
                                                self._thresholds, self._total_num_ramps,
                                                queuing_delay, ramp_latencies, optimal=self._args.optimal_exiting)

                elif apparate_optimal:
                    thresholds_new, latency_improvement_new, exit_rate, acc = \
                            tune_threshold(self._ramp_ids, self._shadow_ramp_idx, inputs, acc_loss_budget=ACC_LOSS_BUDGET_TUNING, latency_calc_list=self._latency_calc_list[batch_size])

                    # self._ramp_ids, self._thresholds, latency_savings, acc, exit_rate, _ = ramp_addition_tail_latency(
                    #     inputs,
                    #     latency_calc_list=self._latency_calc_list[self._args.batch_size],
                    #     num_ramp_budget=5,
                    #     acc_loss_budget=ACC_LOSS_BUDGET_TUNING,
                    #     tail_latency_budget=TAIL_LATENCY_BUDGET
                    # )       

                    # ramp_latencies, vanilla_latency = get_ramp_latencies(
                    #     self._ramp_ids, self._latency_calc_list[batch_size])

                    batch_meta_data, sample_latencies, sample_acc, sample_exit_points = \
                        earlyexit_infer_per_sample(None, None, self._ramp_ids,
                                                self._thresholds, self._total_num_ramps,
                                                queuing_delay, ramp_latencies, optimal=self._args.optimal_exiting,
                                                simulated_pickle=inputs)

                    acc, latency_improvement_old, exit_rate = \
                        get_batch_perf(sample_latencies, sample_acc, sample_exit_points,
                                    vanilla_latency, self._ramp_ids, self._total_num_ramps)

                    if latency_improvement_new - latency_improvement_old > 2.0 or acc < 1 - ACC_LOSS_BUDGET_ACTUAL:
                        self._thresholds = thresholds_new
                        self._logger.info(f"should change! latency_improvement_new {latency_improvement_new}, latency_improvement_old {latency_improvement_old}, acc {acc}")

                    if acc < 1 - ACC_LOSS_BUDGET_ACTUAL:
                        self._thresholds = thresholds_new
                        self._logger.info("must change!")
                        if i > 1:

                            thresholds_tune, latency_improvement_tune, exit_rate, acc = \
                                tune_threshold(self._ramp_ids, self._shadow_ramp_idx, self._historical_data, acc_loss_budget=ACC_LOSS_BUDGET_TUNING, latency_calc_list=self._latency_calc_list[batch_size])

                            acc, latency_improvement_tune, exit_rate = \
                                get_batch_perf(sample_latencies, sample_acc, sample_exit_points,
                                            vanilla_latency, self._ramp_ids, self._total_num_ramps)
                            
                            batch_meta_data_1, sample_latencies_1, sample_acc_1, sample_exit_points_1 = \
                                earlyexit_infer_per_sample(None, None, self._ramp_ids,
                                                        thresholds_tune, self._total_num_ramps,
                                                        queuing_delay, ramp_latencies, optimal=self._args.optimal_exiting,
                                                        simulated_pickle=inputs)

                            acc_tune, latency_improvement_tune, exit_rate = \
                                get_batch_perf(sample_latencies_1, sample_acc_1, sample_exit_points_1,
                                            vanilla_latency, self._ramp_ids, self._total_num_ramps)

                            if acc_tune > 1 - ACC_LOSS_BUDGET_ACTUAL:
                                self._logger.info("can be improve!, {}".format(acc_tune))
                else:
                    # self._thresholds, latency_improvement, exit_rate, acc = \
                    #         tune_threshold(self._ramp_ids, self._shadow_ramp_idx, inputs, acc_loss_budget=ACC_LOSS_BUDGET_TUNING, latency_calc_list=self._latency_calc_list[batch_size])

                    # self._ramp_ids_new, self._thresholds_new, latency_improvement_new, acc, exit_rate, _ = ramp_addition_tail_latency(
                    #     inputs,
                    #     latency_calc_list=self._latency_calc_list[self._args.batch_size],
                    #     num_ramp_budget=NUM_RAMP_BUDGET,
                    #     acc_loss_budget=ACC_LOSS_BUDGET_TUNING,
                    #     tail_latency_budget=TAIL_LATENCY_BUDGET
                    # )
                    # self._logger.info("optimal: batch {}, ramp_ids {}, thresholds {}, actual acc {}, latency_improvement {}, exit_rate {}"
                    #               .format(i, self._ramp_ids_new, self._thresholds_new, acc, latency_improvement_new, exit_rate))
                    # if self._ramp_ids_new is not None and latency_improvement_new - latency_improvement > 0.0:
                    #     self._ramp_ids = self._ramp_ids_new
                    #     self._thresholds = self._thresholds_new

                    #     ramp_latencies, vanilla_latency = get_ramp_latencies(
                    #         self._ramp_ids, self._latency_calc_list[batch_size])

                    batch_meta_data, sample_latencies, sample_acc, sample_exit_points = \
                        earlyexit_infer_per_sample(None, None, self._ramp_ids,
                                                self._thresholds, self._total_num_ramps,
                                                queuing_delay, ramp_latencies, optimal=self._args.optimal_exiting,
                                                simulated_pickle=inputs)

                        
                if self._args.optimal_exiting:
                    sample_latencies = [(s[0], min(s[1], vanilla_latency)) for s in sample_latencies]
                
                if self._recovery_mode == True:
                    self._all_latencies += [(p[0], vanilla_latency) for p in sample_latencies]
                else:
                    self._all_latencies += sample_latencies
                # self._logger.info(f"{self._all_latencies[-1]}, {self._recovery_mode}, {sample_latencies}")
                self._all_accuracies += sample_acc
                self._all_exit_ramp += sample_exit_points
                self._all_vanilla_latencies += [vanilla_latency] * len(sample_latencies)

                if store_entropy_pickle:
                    for ramp_id in range(self._total_num_ramps):
                        entropy_dict["conf"][ramp_id] += batch_meta_data["conf"][ramp_id]
                        entropy_dict["acc"][ramp_id] += batch_meta_data["acc"][ramp_id]
                        # print(f"batch_meta_data['conf'][ramp_id] {batch_meta_data['conf'][ramp_id]}")

                acc, latency_improvement, exit_rate = \
                    get_batch_perf(sample_latencies, sample_acc, sample_exit_points,
                                   vanilla_latency, self._ramp_ids, self._total_num_ramps)

                self._acc_violation_info.append(
                    [acc < 1 - ACC_LOSS_BUDGET_ACTUAL, acc, batch_size])
                
                if self._acc_violation_info[-1][0]:
                    self._violation_counter += 1
                # print(self._acc_violation_info[-1])

                _, curr_ramp_acc = get_overall_exit_info(
                    sample_exit_points, sample_acc)
                
                self._curr_ramp_acc = curr_ramp_acc

                self._logger.info("serve_batch: batch {}, current bs {}, ramp_ids {}, thresholds {}, actual acc {}, latency_improvement {}, exit_rate {}, ramp acc {}"
                                  .format(i, batch_size, self._ramp_ids, self._thresholds, acc, latency_improvement if not self._recovery_mode else 0.0 , exit_rate, curr_ramp_acc))

                if not store_entropy_pickle:
                    # 3. Update the historical data
                    self.update_historical_data(
                        batch_meta_data, exit_rate, len(inputs["conf"][0]) if self._simulate else target.size(0), latency_improvement)
                    # 4. Check signals
                    if self.nlp:
                        is_threshold_tuned, is_ramp_adjusted = self.check_signals_nlp(i, batch_size)
                    else:
                        is_threshold_tuned, is_ramp_adjusted = self.check_signals_cv(i, batch_size)
                    if i % RAMP_CHECK_INTERVAL == 0:
                        self._ramp_history.append(self._ramp_ids)
                    self._threshold_tuning_history.append(is_threshold_tuned)
                self._batch_idx += 1
                self._lock.release()

        if store_entropy_pickle:
            # with open(f"./entropy_pickles/{self._args.dataset}_{self._args.arch}_entropies.pickle", "wb") as f:
            with open(f"../{self._args.dataset}_{self._args.arch}.pickle", "wb") as f:
                pickle.dump(entropy_dict, f)

        # plotting.plot_latency_cdf(
        #     self._all_latencies, vanilla_latency=vanilla_latency)
        overall_accuracy = 100 * \
            (sum(self._all_accuracies) / len(self._all_accuracies))

        all_serving_latencies = [l[1] for l in self._all_latencies]


        # print(np.array([l for l in all_serving_latencies]).mean())
        if self._args.batch_decision_path is not None:
            if not self._args.optimal_exiting:
                if self.nlp:
                    path = f"../apparate_latency/{self._args.arch}_{self._args.dataset}_azure.pickle"
                    with open(path, "wb") as f:
                        pickle.dump(all_serving_latencies, f)
                else:
                    path = f"../apparate_latency/{self._args.arch}_{self._args.dataset}_{int(self._args.slo)}_fixed_{int(self._args.qps)}.pickle"
                    with open(path, "wb") as f:
                        pickle.dump(all_serving_latencies, f)
            else:
                path = f"../optimal_latency/{self._args.arch}_{self._args.dataset}_{int(self._args.slo)}_fixed_{int(self._args.qps)}_optimal.pickle"
                with open(path, "wb") as f:
                    pickle.dump(all_serving_latencies, f)

        overall_exit_rate, overall_exit_accuracy = get_overall_exit_info(
            self._all_exit_ramp, self._all_accuracies)

        all_serving_latencies = np.array(all_serving_latencies)
        self._all_vanilla_latencies = np.array(self._all_vanilla_latencies)

        average_latency_improvement = 100 * np.mean((self._all_vanilla_latencies - all_serving_latencies) / self._all_vanilla_latencies) 

        # if self.nlp:
        #     self.plot_latency_cdfs()

        self._logger.info(
            f"[{self._args.arch}, {self._args.dataset}]: Serving with complete, overall accuracy {overall_accuracy}%, "
            f"overall serving latency improvement {average_latency_improvement}%, "
            f"overall exit rate {overall_exit_rate}, overall ramp accuracy {overall_exit_accuracy}")

    def update_historical_data(self, data, exit_rate, batch_size, latency_improvement):
        """Update historical data

        Args:
            data (tensor): batch of entropy data
            exit_rate (np.ndarray): index x: samples exited at xth ramp, 
                normalized to 1.0. Last position: samples exited at the 
                end of vanilla model, also normalized.
            batch_size (int): number of requests in the current batch
            latency_improvement (float): latency improvement of the current batch
        """
        assert len(self._ramp_ids) > 0, "No ramp enabled"
        self._curr_ramp_avg_confidence = [np.average(data['conf'][ramp_id]) for ramp_id in self._ramp_ids]
        # self._logger.info(f"curren ramp avg confidence {self._curr_ramp_avg_confidence}")

        self._last_latency_improvement = self._curr_latency_improvement    
        self._curr_latency_improvement = latency_improvement


        self._batch_size_info.append(batch_size)
        for key, _ in data.items():
            for ramp_id in self._ramp_ids:
                self._historical_data[key][ramp_id] += data[key][ramp_id]


        if len(self._batch_size_info) > self._historical_data_size:
            size = self._batch_size_info.pop(0)
            for key, _ in data.items():
                for ramp_id in self._ramp_ids:
                    # if ramp_id == self._shadow_ramp_id:
                    #     if len(self._historical_data[key][ramp_id]) < \
                    #             sum(self._batch_size_info):
                    #         continue
                    if len(self._historical_data[key][ramp_id]) <= size:
                        continue
                    self._historical_data[key][ramp_id] \
                        = self._historical_data[key][ramp_id][size:]

        # update historical exit rates and ramp utility scores
        self._historical_exit_rates.append([exit_rate, batch_size])

        # NOTE(ruipan): get_ramp_utilities() is better at capturing which ramp to deactivate
        # TODO: incorporate tail latency into consideration
        batch_size = utils.round_up_batch_size(batch_size)
        latency_config, _ = get_ramp_latencies(
            self._ramp_ids, self._latency_calc_list[batch_size])
        utilites = get_ramp_utility(
            self._ramp_ids, exit_rate, latency_config, self._latency_calc_list[batch_size])
        for ramp_id in self._ramp_ids:
            self._historical_ramp_utility[ramp_id].append(utilites.pop(0))

        if len(self._historical_exit_rates) > 20:
            self._historical_exit_rates.pop(0)
            for ramp_id in self._ramp_ids:
                if len(self._historical_ramp_utility[ramp_id]) > 20:
                    self._historical_ramp_utility[ramp_id].pop(0)
        self._logger.debug("historical exit rates: {}".format(
            self._historical_exit_rates))
        for ramp_id in self._ramp_ids:
            self._logger.debug("ramp {} utility: {}".format(
                ramp_id, self._historical_ramp_utility[ramp_id]))


    def check_signals_cv(self, batch_id: int, batch_size: int):
        """Check signals for ramp activation/deactivation and threshold tuning

        Args:
            batch_id (int): batch index
            batch_size (int): batch sizes

        Returns:
            tune_threshold (bool): True if threshold tuning is conducted
            ramp_adjustment (bool): True if ramp activation/deactivation is conducted
        """
        # threading.Thread(target=self._threshold_tuner.greedy_search, args=).start()

        is_threshold_tuned, is_ramp_adjusted = False, False

        # return is_threshold_tuned, is_ramp_adjusted  # XXX: uncomment for optimal exiting
        
        if self._args.optimal_exiting:
            return is_threshold_tuned, is_ramp_adjusted
        
        num_samples = 0.0
        correct_samples = 0.0
        for acc_info in self._acc_violation_info:
            num_samples += acc_info[2]
            correct_samples += acc_info[1] * acc_info[2]

        curr_overall_acc = correct_samples / num_samples

        if self._violation_counter >= 2 or curr_overall_acc < 1 - ACC_LOSS_BUDGET_ACTUAL:
            # self._logger.info("violation counter is {}".format(
            #     self._violation_counter))
            for idx, ramp_id in enumerate(self._ramp_ids):
                if ramp_id in self._curr_ramp_acc:
                    if self._curr_ramp_acc[ramp_id] < 1 - ACC_LOSS_BUDGET_ACTUAL:
                        self._thresholds[idx] = 0.0
        else:
            if self._acc_violation_info[-1][0] == True or self._after_ramp_adjustment or batch_id == 1: 
                thresholds, _, _, _ \
                    = tune_threshold(self._ramp_ids, None, self._historical_data, acc_loss_budget=ACC_LOSS_BUDGET_TUNING, latency_calc_list=self._latency_calc_list[batch_size])
                self._thresholds = thresholds
                self._after_ramp_adjustment = False
            
        if batch_id % RAMP_CHECK_INTERVAL == 0:  # with ramp changes
        # if False:  # no ramp changes
            if curr_overall_acc >= 1 - ACC_LOSS_BUDGET_ACTUAL:
                self._recovery_mode = False
            self._violation_counter = 0
            negative_ramps = []
            negative_ramp_idxs = []

            ramp_scores = []
      
            for idx, ramp_id in enumerate(self._ramp_ids):
                # self._logger.info("ramp {} utility: {}".format(
                #     ramp_id, self._historical_ramp_utility[ramp_id]))
                if all(i <= 0.0 for i in self._historical_ramp_utility[ramp_id]):
                    # self._logger.info(f"ramp {ramp_id} is negative")
                    negative_ramps.append(ramp_id)
                    negative_ramp_idxs.append(idx)
                ramp_scores.append([idx, ramp_id, np.array(self._historical_ramp_utility[ramp_id]).mean()])

            ramp_scores = sorted(ramp_scores, key=lambda x: x[2])
            self._logger.info(f"ramp scores: {ramp_scores}")

            # self._logger.info(f"negative ramp: {negative_ramps} negative ramp idxs: {negative_ramp_idxs}")
            if len(negative_ramps) > 0: # there is at least one negative ramp
                thresholds, latency_improvement, exit_rate, acc \
                    = tune_threshold(self._ramp_ids, self._shadow_ramp_idx, self._historical_data, acc_loss_budget=ACC_LOSS_BUDGET_TUNING, latency_calc_list=self._latency_calc_list[batch_size])
                latency_gap = 5.0 if self.nlp else 2.0
                if latency_improvement - self._curr_latency_improvement > latency_gap:
                    self._thresholds = thresholds 
                else:
                    # return is_threshold_tuned, is_ramp_adjusted
                    total_samples = sum([bz for _, bz in self._historical_exit_rates])
                    avg_exit_rate_info = sum([exit_rate * bz for exit_rate, bz in self._historical_exit_rates]) / total_samples
                    # self._logger.info("threshold tuning is not enough")

                    if self._latest_possible_ramp in negative_ramps and len(self._ramp_ids) == len(negative_ramps):
                        if len(self._ramp_ids) > 1:
                            for idx, ramp_id in enumerate(negative_ramps[:-1]):
                                self.clear_meta_data(ramp_id)
                                self._ramp_quota += 1
                            self._ramp_ids = [negative_ramps[-1]]
                            self._thresholds = [self._thresholds[-1]]
                            self._after_ramp_adjustment = True
                        else:
                            self._after_ramp_adjustment = False
                        # self._logger.info("all ramps are negative, including last ramp")
                        
                        return is_threshold_tuned, is_ramp_adjusted
                    
    
                    for idx, ramp_id in enumerate(negative_ramps):
                        self.clear_meta_data(ramp_id)
                        self._ramp_quota += 1


                    left = max(negative_ramps[-1], self._ramp_ids[-1]) + 1
                    right = self._latest_possible_ramp
                        
                    remained_ramp_idx = [idx for idx in range(len(self._ramp_ids)) if idx not in negative_ramp_idxs]
                    self._ramp_ids = [ramp_id for idx, ramp_id in enumerate(self._ramp_ids) if idx in remained_ramp_idx]
                    self._thresholds = [threshold for idx, threshold in enumerate(self._thresholds) if idx in remained_ramp_idx]

                    if sum([avg_exit_rate_info[idx] for idx in remained_ramp_idx]) > 0.9 or left > right:
                        self._after_ramp_adjustment = True
                        return is_threshold_tuned, is_ramp_adjusted
                    else:
                        new_ramps = self.get_new_ramps(left, right, 1)
                        self._ramp_ids += new_ramps
                        self._thresholds += [0.0]
                        self._after_ramp_adjustment = True
                        self._ramp_quota -= 1
                        for idx, ramp_id in enumerate(self._ramp_ids):
                            self.clear_meta_data(ramp_id)
                        return is_threshold_tuned, is_ramp_adjusted

            else:
                if self._ramp_quota > 0:
                    if self._ramp_ids[0] > 1:
                        new_ramp = (self._ramp_ids[0] - 1) // 2
                        self._ramp_ids = [new_ramp] + self._ramp_ids
                        self._thresholds = [0.0] + self._thresholds
                        self._ramp_quota -= 1
                        for idx, ramp_id in enumerate(self._ramp_ids):
                            self.clear_meta_data(ramp_id)
                        self._after_ramp_adjustment = True
                    else:
                        self._after_ramp_adjustment = False
                else:
                    if self._ramp_ids[0] < 2:
                        self._after_ramp_adjustment = False
                    else:
                        for idx, ramp_id in enumerate(self._ramp_ids):
                            self.clear_meta_data(ramp_id)
                        new_ramp = (self._ramp_ids[0] - 1) // 2
                        self._ramp_ids = [ramp_id for idx, ramp_id in enumerate(self._ramp_ids) if idx != ramp_scores[-1][0]]
                        self._thresholds = [threshold for idx, threshold in enumerate(self._thresholds) if idx != ramp_scores[-1][0]]
                        self._ramp_ids = [new_ramp] + self._ramp_ids
                        self._thresholds = [0.0] + self._thresholds
                        self._after_ramp_adjustment = True

        return is_threshold_tuned, is_ramp_adjusted

    def check_signals_nlp(self, batch_id: int, batch_size: int):
        """Check signals for ramp activation/deactivation and threshold tuning

        Args:
            batch_id (int): batch index
            batch_size (int): batch sizes

        Returns:
            tune_threshold (bool): True if threshold tuning is conducted
            ramp_adjustment (bool): True if ramp activation/deactivation is conducted
        """
        # threading.Thread(target=self._threshold_tuner.greedy_search, args=).start()

        is_threshold_tuned, is_ramp_adjusted = False, False
        
        if self._args.optimal_exiting:
            return is_threshold_tuned, is_ramp_adjusted
    
        
        if self._acc_violation_info[-1][0] == True \
            or (self._after_ramp_adjustment and batch_id % RAMP_CHECK_INTERVAL == 40) or batch_id == 1: 

            fixed_ramps_info = {}

            if not (self._after_ramp_adjustment and batch_id % RAMP_CHECK_INTERVAL == 40):
                fixed_ramps_info = {}
                for idx, ramp_id in enumerate(self._ramp_ids):
                    if ramp_id in self._curr_ramp_acc:
                        if self._curr_ramp_acc[ramp_id] > 1 - ACC_LOSS_BUDGET_ACTUAL:
                            fixed_ramps_info[ramp_id] = self._thresholds[idx]
                    else:
                        fixed_ramps_info[ramp_id] = self._thresholds[idx]

            thresholds, _, _, _ \
                = tune_threshold(self._ramp_ids, None, self._historical_data, \
                                    acc_loss_budget=ACC_LOSS_BUDGET_TUNING, \
                                    latency_calc_list=self._latency_calc_list[batch_size], fixed_ramps_info=fixed_ramps_info)
            self._thresholds = thresholds
            self._after_ramp_adjustment = False

        if self._violation_counter >= 2:
            for idx, ramp_id in enumerate(self._ramp_ids):
                if ramp_id in self._curr_ramp_acc:
                    if self._curr_ramp_acc[ramp_id] < 1 - ACC_LOSS_BUDGET_ACTUAL:
                        self._thresholds[idx] = self._thresholds[idx] * 0.1
            
        if batch_id % RAMP_CHECK_INTERVAL == 0:  # with ramp changes
            self._after_ramp_adjustment = True

            self._violation_counter = 0
            negative_ramps = []
            negative_ramp_idxs = []
            num_positive_ramps = 0.0
            ramp_scores = []

            for idx, ramp_id in enumerate(self._ramp_ids):
                # self._logger.info("ramp {} utility: {}".format(
                #     ramp_id, self._historical_ramp_utility[ramp_id]))
                positive_counter = 0.0
                negative_counter = 0.0
                for i in self._historical_ramp_utility[ramp_id]:
                    if i >= 0.0:
                        positive_counter += 1
                    elif i < 0.0:
                        negative_counter += 1
                if negative_counter / len(self._historical_ramp_utility[ramp_id]) > self._negative_threshold:
                    negative_ramps.append(ramp_id)
                    negative_ramp_idxs.append(idx)
            
                if positive_counter / len(self._historical_ramp_utility[ramp_id]) > self._postive_threshold:
                    num_positive_ramps += 1
                   
                ramp_scores.append([idx, ramp_id, np.array(self._historical_ramp_utility[ramp_id]).mean()])
                self._logger.info(f"ramp {ramp_id} positive counter {positive_counter} negative counter {negative_counter}") 
            ramp_scores = sorted(ramp_scores, key=lambda x: x[2])
            # self._logger.info(f"ramp scores: {ramp_scores}")

            if len(self._ramp_history) > 4:
                self._logger.info(f"self._ramp_history {self._ramp_history[-2][0]} self._ramp_ids[0] {self._ramp_ids[0]}")
                
                condition = all(x == self._ramp_history[-1] for x in self._ramp_history[-5:])
                if condition:
                    # self._postive_threshold -= 0.15
                    # self._postive_threshold = max(0.6, self._postive_threshold)
                    self._postive_threshold = 0.6

            if len(negative_ramps) == 0 and num_positive_ramps == 0 and len(self._ramp_ids) == 1:
                num_positive_ramps = 1

            if len(negative_ramps) == 1 and len(self._ramp_ids) == 1 and self._ramp_ids[-1] == self._latest_possible_ramp:
                negative_ramps = []
                num_positive_ramps = 1

            self._logger.info(f"negative ramp: {negative_ramps} negative ramp idxs: {negative_ramp_idxs}, num_positive_ramps {num_positive_ramps} positive threshold {self._postive_threshold}, negative threshold {self._negative_threshold} self._ramp_quota {self._ramp_quota}")
            if len(negative_ramps) > 0: # there is at least one negative ramp

                if len(self._ramp_history) > 1:
                    self._logger.info(f"self._ramp_history {self._ramp_history[-2][0]} self._ramp_ids[0] {self._ramp_ids[0]}")
                    
                    if utils.compare_lists(self._ramp_ids, self._ramp_history[-2]):
                    # if self._ramp_history[-2][0] > self._ramp_ids[0]:
                        # self._postive_threshold += 0.1
                        # self._postive_threshold = min(0.95, self._postive_threshold)
                        self._postive_threshold = 1.0

                thresholds, latency_improvement, exit_rate, acc \
                    = tune_threshold(self._ramp_ids, self._shadow_ramp_idx, self._historical_data, acc_loss_budget=ACC_LOSS_BUDGET_TUNING, latency_calc_list=self._latency_calc_list[batch_size])
                if latency_improvement - self._curr_latency_improvement > 5.0:
                    self._thresholds = thresholds
                else:
                    # return is_threshold_tuned, is_ramp_adjusted
                    total_samples = sum([bz for _, bz in self._historical_exit_rates])
                    avg_exit_rate_info = sum([exit_rate * bz for exit_rate, bz in self._historical_exit_rates]) / total_samples
                    if self._latest_possible_ramp in negative_ramps and len(self._ramp_ids) == len(negative_ramps):
                        if len(self._ramp_ids) > 1:
                            for idx, ramp_id in enumerate(negative_ramps[:-1]):
                                self.clear_meta_data(ramp_id)
                                self._ramp_quota += 1
                            self._ramp_ids = [negative_ramps[-1]]
                            self._thresholds = [self._thresholds[-1]]                         
                        return is_threshold_tuned, is_ramp_adjusted
                    
    
                    for idx, ramp_id in enumerate(negative_ramps):
                        self.clear_meta_data(ramp_id)
                        self._ramp_quota += 1


                    left = max(negative_ramps[-1], self._ramp_ids[-1]) + 1
                    right = self._latest_possible_ramp
                        
                    remained_ramp_idx = [idx for idx in range(len(self._ramp_ids)) if idx not in negative_ramp_idxs]
                    self._ramp_ids = [ramp_id for idx, ramp_id in enumerate(self._ramp_ids) if idx in remained_ramp_idx]
                    self._thresholds = [threshold for idx, threshold in enumerate(self._thresholds) if idx in remained_ramp_idx]

                    if (sum([avg_exit_rate_info[idx] for idx in remained_ramp_idx]) > 0.9 or left > right):
                        return is_threshold_tuned, is_ramp_adjusted
                    else:
                        new_ramps = self.get_new_ramps(left, right, 1)
                        self._ramp_ids += new_ramps
                        self._thresholds += [0.01]
                        self._ramp_quota -= 1
                        for idx, ramp_id in enumerate(self._ramp_ids):
                            self.clear_meta_data(ramp_id)
                
                        return is_threshold_tuned, is_ramp_adjusted

            else:
                if num_positive_ramps < 1:
                    return is_threshold_tuned, is_ramp_adjusted

               
                if len(self._ramp_history) > 1:
                    self._logger.info(f"self._ramp_history {self._ramp_history[-2][0]} self._ramp_ids[0] {self._ramp_ids[0]}")
                    if utils.compare_lists(self._ramp_history[-2], self._ramp_ids):
                        self._postive_threshold = 1.0


                if self._ramp_quota > 0:
                    if self._ramp_ids[0] > 1:
                        new_ramp = (self._ramp_ids[0] - 1) // 2
                        self._ramp_ids = [new_ramp] + self._ramp_ids
                        self._thresholds = [0.01] + self._thresholds
                        self._ramp_quota -= 1
                        for idx, ramp_id in enumerate(self._ramp_ids):
                            self.clear_meta_data(ramp_id)
                else:
                    if self._ramp_ids[0] >= 2:
                        for idx, ramp_id in enumerate(self._ramp_ids):
                            self.clear_meta_data(ramp_id)
                        new_ramp = (self._ramp_ids[0] - 1) // 2
                        self._ramp_ids = [ramp_id for idx, ramp_id in enumerate(self._ramp_ids) if idx != ramp_scores[-1][0]]
                        self._thresholds = [threshold for idx, threshold in enumerate(self._thresholds) if idx != ramp_scores[-1][0]]
                        self._ramp_ids = [new_ramp] + self._ramp_ids
                        self._thresholds = [0.01] + self._thresholds

        return is_threshold_tuned, is_ramp_adjusted

    def ensemble_historical_data(self):
        """Ensemble historical data for shadow ramp.

        Returns:
            temporal_good_data: a dict good samples
            temporal_bad_data: a dict bad samples
            temporal_data: a dict of all samples
            trim_good_size: the number of good samples trimmed
            trim_bad_size: the number of bad samples trimmed
        """
        num_good_samples = len(
            self._historical_data_good['conf'][self._shadow_ramp_id])
        num_bad_samples = len(
            self._historical_data_bad['conf'][self._shadow_ramp_id])

        active_id = self._ramp_ids[(
            self._shadow_ramp_idx + 1) % len(self._ramp_ids)]
        trim_good_size = len(
            self._historical_data_good['conf'][active_id]) - num_good_samples
        trim_bad_size = len(
            self._historical_data_bad['conf'][active_id]) - num_bad_samples

        self._logger.info(
            f"num_good_samples {num_good_samples}, num_bad_samples {num_bad_samples}")

        temporal_good_data = {'conf': [[] for _ in range(self._total_num_ramps)],
                              'acc': [[] for _ in range(self._total_num_ramps)]}
        temporal_bad_data = {'conf': [[] for _ in range(self._total_num_ramps)],
                             'acc': [[] for _ in range(self._total_num_ramps)]}

        for key, _ in temporal_good_data.items():
            for ramp_id in self._ramp_ids:
                if num_good_samples > 0:
                    temporal_good_data[key][ramp_id] = self._historical_data_good[key][ramp_id][-num_good_samples:]
                if num_bad_samples > 0:
                    temporal_bad_data[key][ramp_id] = self._historical_data_bad[key][ramp_id][-num_bad_samples:]

        temporal_data = copy.deepcopy(temporal_good_data)
        for key, _ in temporal_good_data.items():
            for ramp_id in self._ramp_ids:
                temporal_data[key][ramp_id] += temporal_bad_data[key][ramp_id]

        min_size = len(self._historical_data['conf'][self._shadow_ramp_id])
        for key, _ in self._historical_data.items():
            for ramp_id in self._ramp_ids:
                if ramp_id == self._shadow_ramp_id:
                    continue
                else:
                    self._historical_data[key][ramp_id] = self._historical_data[key][ramp_id][-min_size:]
                    
        return temporal_good_data, temporal_bad_data, temporal_data, trim_good_size, trim_bad_size

    def get_exitable_ramps(self):
        """For each historcal data, returns a list of possible ramps
        it can exit from.

        Returns:
            all_exitable_ramps (list): list of lists. index x: all
                ramp IDs historical data x can exit from.
        """
        num_samples = len(self._historical_data["conf"][self._ramp_ids[0]])
        all_exitable_ramps = [[] for _ in range(num_samples)]
        for sample_id in range(num_samples):
            # confidence score (higher: easier) at all ramps
            all_conf = [self._historical_data['conf'][ramp_id][sample_id]
                        for ramp_id in self._ramp_ids]
            all_exitable_ramps[sample_id] = [
                self._ramp_ids[i] for i, (conf, threshold) in enumerate(zip(all_conf, self._thresholds))
                if (1 - conf) <= threshold
            ]
        return all_exitable_ramps


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Online EE Controller")
    parser.add_argument('--dataset', type=str, default="urban")
    parser.add_argument('--arch', type=str, default="resnet18_urban", help="vanilla model architecture")
    parser.add_argument('--earlyexit', type=bool, default=True)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_classes', type=int, default=3)
    parser.add_argument('--data_dir', type=str,
                        default=os.path.join(os.getenv("HOME"), "urban"))
    parser.add_argument('--model_dir', type=str,
                        default=os.path.join(os.getenv("HOME"), "model_checkpoints"))
    parser.add_argument('--pretrained', type=bool, default=True)
    parser.add_argument('--profile_dir', type=str, default='')
    parser.add_argument('--request_rate', type=int, default=300,
                        help="Request rate in requests per second")
    parser.add_argument('--slo', type=float, default=45,
                        help="Default SLO in ms")
    parser.add_argument('--qps', type=float, default=350,
                        help="expected query per second")
    parser.add_argument('--batching_scheme', type=str,
                        default='clockwork', choices=['clockwork', 'tf_serve', 'uniform'])
    parser.add_argument('--bootstrap_pickle_path', type=str,
                        default=None)
    parser.add_argument('--simulation_pickle_path', type=str,
                        default=None)
    parser.add_argument('--batch_decision_path', type=str,
                        default=None)
    parser.add_argument('--optimal_exiting', action='store_true')
    args, unknown = parser.parse_known_args()
    
    controller = Controller(args)
    # controller.serve()
    controller.serve(simulate=True, store_entropy_pickle=False)