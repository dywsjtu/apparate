import copy
import logging
import multiprocessing
import os
import pickle
from pprint import pformat
import random
import sys
import argparse
os.environ['OPENBLAS_NUM_THREADS'] = '1'  # https://stackoverflow.com/a/57549064/9601555
# TODO: del os.environ['OPENBLAS_NUM_THREADS']?
import numpy as np
from collections import OrderedDict
from itertools import product, combinations
from threshold_tuner import *

from utils import *


logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s [%(levelname)s] %(message)s', 
    datefmt='%Y-%m-%d %H:%M:%S'
)


class RampSelector():
    def __init__(self, args):
        self.golden_threshold_dict = {
            # FIXME(ruipan): remove me
            # thresholds for bert + dataset s.t. if this threshold is set
            # for all ramps, the overall acc loss is < ~2%.
            "rte": 0.0625,
            "mrpc": 0.0225,
            "sst-2": 0.005,
            "qnli": 0.02,
            "qqp": 0.03,
            "mnli": 0.03,
            "mnli-mm": 0.025,
            "cifar10": 0.1,
            "cifar100": 0.1,  # 0.05?
        }
        self.model_overhead_dict = {
            # TODO(ruipan): use model profiling result to replace me
            "bert": (0.62, 0.11),  # (encoder overhead, ramp overhead)
            "resnet56": (0.31, 0.11),
            "resnet50": (0.31, 0.11),
            "resnet18": (0.31, 0.11),
            # "resnet56": (0.31, 0) 
        }
        self.pickle_path = args.offline_data_path
        # if pruning ramps from a full set of ramps, prune a ramp
        # if its utility is smaller than prune_threshold
        # NOTE(ruipan): larger values like 3, 5, or 10 also works (although not as good as 0)
        self.prune_threshold = 0.1
        # different methods for calculating savings in get_savings_and_overhead_of_ramp().
        self.favor_early_ramps = False  # True or False
        # if selecting ramps starting from an empty set, add the best
        # ramps until we reach the ramp (memory) budget
        self.num_ramp_budget = args.ramp_budget 
        # self.num_ramp_budget = args.ramp_budget = 5
        # if true, ignore the ramp budget and automatically selects
        # the best number of ramps by stopping ramp addition when adding
        # a ramp decreases the latency savings
        self.auto_num_ramps = False
        # max amount of accuracy loss (compared to the original model output) we can afford
        self.acc_loss_budget = 1.5
        # FIXME(ruipan)
        self.dataset = self.pickle_path.split('/')[-1].split('_')[0]
        self.model_type = self.pickle_path.split('/')[-1].split('.')[0]
        self.model_type = self.model_type.split('_')[1].split('-')[0]
        
        self.overhead_encoder, self.overhead_ramp = self.model_overhead_dict[self.model_type]
        self.args = args
        self.threshold_tuner = ThresholdTuner(args)
        

    def get_num_ramps(self, pickle_path: str) -> int:
        """Return the number of ramps (and encoders)
        in a model based on its name

        Args:
            pickle_path (str): absolute path of pickle file

        Returns:
            int: number of ramps
        """
        if "distilbert" in pickle_path:
            return 6
        elif "bert-large" in pickle_path:
            return 24
        elif "bert-base" in pickle_path:
            return 12
        elif "resnet56" in pickle_path:
            return 27
        elif "resnet50" in pickle_path:
            return 16
        elif "resnet18" in pickle_path:
            return 8
        assert False


    def evaluate_savings(self, exit_counter: dict, ramp_configuration: dict, total_num_samples: int, num_encoders: int) -> float:
        """Evaluate the latency savings of a ramp configuration
        and its resulting exit counters.

        Args:
            exit_counter (dict): key: 0-indexed ramp id, value: num samples exited
            ramp_configuration (dict): key: 0-indexed ramp id,
            value: exit threshold at that ramp. Requires ramp
            ids to be monotonically increasing (from shallow to deep)
            total_num_samples (int): total number of samples in the dataset
            num_encoders (int): number of encoders

        Returns:
            float: percentage of overall latency savings, normalized to [0, 100].
            E.g., 0.0 is vanilla, while 30.0 stands for 30% savings.
        """
        # TODO(ruipan): incorporate model profiling result to calculate this
        orig_cost = total_num_samples * num_encoders * self.overhead_encoder

        ee_cost = 0
        for ramp_id, samples_exited in exit_counter.items():
            # encoder overhead: ramp_id + 1 is the num of encoders traversed
            num_encoders_traversed = ramp_id + 1
            ee_cost += samples_exited * num_encoders_traversed * self.overhead_encoder
            # ramp overhead
            # num_ramps_traversed = sum([1 for x in ramp_configuration.keys() if x <= ramp_id])  # dict
            num_ramps_traversed = sum([1 for x in ramp_configuration if x <= ramp_id])  # list
            ee_cost += samples_exited * num_ramps_traversed * self.overhead_ramp
        
        # factor in samples that did not exit through any ramps
        num_unexited_samples = total_num_samples - sum(exit_counter.values())
        ee_cost += num_unexited_samples * (num_encoders * self.overhead_encoder + len(ramp_configuration) * self.overhead_ramp)

        savings = 100 * (1 - ee_cost / orig_cost)
        return savings


    def evaluate_savings_v2(self, task: str, exit_counter: dict, ramp_configuration: dict, total_num_samples: int) -> float:
        """Evaluate the latency savings of a ramp configuration
        and its resulting exit counters.

        Args:
            task (str): cv or nlp. 
            exit_counter (dict): key: 0-indexed ramp id, value: num samples exited
            ramp_configuration (dict): key: 0-indexed ramp id,
            value: exit threshold at that ramp. Requires ramp
            ids to be monotonically increasing (from shallow to deep)
            total_num_samples (int): total number of samples in the dataset

        Returns:
            float: percentage of overall latency savings, normalized to [0, 100].
            E.g., 0.0 is vanilla, while 30.0 stands for 30% savings.
        """
        # NOTE(ruipan): for now, assumes nlp workloads use bert-base-uncased,
        # while cv workloads use resnet50_waymo.
        if task == "nlp":
            profile_name = "./profile_pickles/bert-base-uncased_profile.pickle"
        elif task == "cv":
            profile_name = "./profile_pickles/resnet50_waymo_earlyexit_profile.pickle"
        else:
            raise NotImplementedError

        with open(profile_name, "rb") as f:
            profile = pickle.load(f)

        # latency in ms with all ramps enabled
        overall_latency = profile.fwd_latency
        vanilla_latency = None  # latency in ms of the vanilla model without any ramps
        all_exits_latency = None  # latency in ms of traversing the whole model with ramp_configuration
        latency_savings = []  # index x: percentage of latency savings if a sample exits at the ramp after layer x

        if profile.type == "ResNetEarlyExit":
            all_branchpoints = self.get_all_children_with_type("BranchPoint")
            last_ramp = all_branchpoints[-1]
            vanilla_latency = overall_latency - sum(last_ramp.ramp_latencies_up_until_me)
            all_exits_latency = vanilla_latency
            all_exits_latency += sum([
                latency for ramp_id, latency in enumerate(last_ramp.ramp_latencies_up_until_me) 
                if ramp_id in ramp_configuration
            ])

            latency_savings = []
            for ramp_id, ramp in enumerate(all_branchpoints):
                ee_latency = ramp.vanilla_latency_up_until_me
                ee_latency += sum([
                    latency for id, latency in enumerate(self.ramp_latencies_up_until_me) 
                    if id <= ramp_id and id in ramp_configuration
                ])
                latency_savings.append(100 * (
                    1 - ee_latency / vanilla_latency
                ))
        elif "BertForSequenceClassification" in profile.type:
            embeddings_latency = profile.children[0].children[0].fwd_latency
            all_bert_layers = profile.get_all_children_with_type(
                "BertLayer" if "Distil" not in profile.type else "TransformerBlock")
            all_bert_highways = profile.get_all_children_with_type("BertHighway")
            avg_layer_latency = sum([x.fwd_latency for x in all_bert_layers]) / len(all_bert_layers)
            avg_highway_latency = sum([x.fwd_latency for x in all_bert_highways]) / len(all_bert_highways)
            vanilla_latency = overall_latency - sum([x.fwd_latency for x in all_bert_highways])
            all_exits_latency = vanilla_latency + len(ramp_configuration) * avg_highway_latency

            latency_savings = []
            for i in range(1, len(all_bert_highways) + 1):
                num_ramps_traversed = sum([1 for ramp_id in ramp_configuration if ramp_id < i])
                ee_latency = embeddings_latency + i * avg_layer_latency + num_ramps_traversed * avg_highway_latency
                latency_savings.append(100 * (
                    1 - ee_latency / vanilla_latency
                ))
        else:
            # need to calculate the following:
            # latency_savings, vanilla_latency, all_exits_latency
            raise NotImplementedError(f"Latency saving calculation for model type {profile.type} not implemented!")

        orig_cost = total_num_samples * 100  # 100% of vanilla latency

        ee_cost = 0
        for ramp_id, samples_exited in exit_counter.items():
            curr_latency_savings = latency_savings[ramp_id]
            ee_cost += samples_exited * (100 - curr_latency_savings)
        # factor in samples that did not exit through any ramps
        num_unexited_samples = total_num_samples - sum(exit_counter.values())
        ee_cost += num_unexited_samples * (all_exits_latency / vanilla_latency * 100)

        savings = 100 * (1 - ee_cost / orig_cost)
        return savings


    def emulate_inference(self, pickle_dict: dict, ramp_configuration: dict, pickle_path: str, task: str):
        """Given a pickle file and some ramp configurations,
        emulates serving a workload and returns an accuracy.

        Args:
            pickle_dict (dict or list): key: either "conf" or "acc", 
            value: list with length = num samples
            ramp_configuration (dict): key: 0-indexed ramp id,
            value: exit threshold at that ramp. Requires ramp
            ids to be monotonically increasing (from shallow to deep)
            pickle_path (str): absolute path to the pickle file
            task (str): either cv or nlp, this determines the entropy definition

        Returns:
            accuracy (float): inference accuracy (%)
            exit_counter (dict): key: 0-indexed ramp id, value: num samples exited
        """
        assert is_monotonic(ramp_configuration.keys(), increasing=True), \
            "Ramp IDs in configuration are not in order!"

        all_ramp_ids = list(range(self.get_num_ramps(pickle_path)))
        exit_counter = {}  # key: 0-indexed ramp IDs, value: num of samples exited
        for id in all_ramp_ids:
            exit_counter[id] = 0

        num_samples = len(pickle_dict["conf"][0])
        num_correct_samples = 0
        exited_data = set()  # indexes of samples that have already exited the model
        for ramp_id, exit_threshold in ramp_configuration.items():
            # find indexes of all samples that can exit from current ramp
            entropies = np.array(pickle_dict["conf"][ramp_id])
            # if task == "cv":
            if True:
                entropies = 1 - entropies  # confidence score -> entropy
            exitable_samples = list(np.where(entropies < exit_threshold)[0])
            newly_exit_samples = set(exitable_samples) - exited_data
            exit_counter[ramp_id] = len(newly_exit_samples)
            exited_data.update(newly_exit_samples)
            num_correct_samples += sum([pickle_dict["acc"][ramp_id][id] for id in newly_exit_samples])
        # the rest of the data didn't exit through any ramps, 
        # and went through the whole model
        num_correct_samples += (num_samples - len(exited_data))

        accuracy = 100 * num_correct_samples / num_samples
        # logging.debug(f"accuracy, exit_counter {accuracy, exit_counter}")
        return accuracy, exit_counter


    def get_exitable_ramps(self, task: str, pickle_dict: dict, ramp_configuration: dict):
        """For each sample, return the ID of the ramp it exits
        from along with a list of possible ramps to exit from 
        given a ramp configuration.

        Args:
            pickle_dict (dict or list): key: either "conf" or "acc", 
            value: list with length = num samples
            ramp_configuration (dict): key: 0-indexed ramp id,
            value: exit threshold at that ramp. Requires ramp
            ids to be monotonically increasing (from shallow to deep)

        Returns:
            exitable_ramps (dict): key: 0-indexed sample ids, 
            value: list of 0-indexed ramp ids
            actual_exit_ramps (dict): key: 0-indexed sample ids,
            value: 0-indexed id of the first ramp
        """
        exitable_ramps = {}
        actual_exit_ramps = {}
        # if task == "nlp":
        #     sample_ids = list(pickle_dict.keys())
        # elif task == "cv":
        sample_ids = list(range(len(pickle_dict["conf"][0])))
        
        for sample_id in sample_ids:
            exitable_ramps[sample_id] = []
            for ramp_id in ramp_configuration.keys():  # go through ramps sequentially
                ramp_entropy = None
                # FIXME(ruipan): unify cv and nlp by getting rid of these ifs
                if task == "nlp":
                    # ramp_entropy = pickle_dict[sample_id]["all_entropies"][ramp_id]
                    ramp_entropy = 1 - pickle_dict["conf"][ramp_id][sample_id]
                elif task == "cv":
                    ramp_entropy = 1 - pickle_dict["conf"][ramp_id][sample_id]

                if ramp_entropy <= ramp_configuration[ramp_id]:
                    if sample_id not in actual_exit_ramps:
                        # first available ramp
                        actual_exit_ramps[sample_id] = ramp_id
                    exitable_ramps[sample_id].append(ramp_id)
                
        return exitable_ramps, actual_exit_ramps


    def get_savings_and_overhead_of_ramp(self, pickle_dict: dict, exitable_ramps: dict, 
        actual_exit_ramps: dict, exit_counter: dict, remaining_rate: dict, 
        total_num_samples: int, ramp_configuration: dict, ramp_id: int):
        """Calculates the overhead and savings of every ramp

        Args:
            pickle_dict (dict or list): key: either "conf" or "acc", 
            value: list with length = num samples
            exitable_ramps (dict): key: 0-indexed sample ids, 
            value: list of 0-indexed ramp ids
            actual_exit_ramps (dict): key: 0-indexed sample ids,
            value: 0-indexed id of the first ramp
            exit_counter (dict): key: 0-indexed ramp id, value: num samples exited
            remaining_rate (dict): key: 0-indexed ramp IDs, value: percentage of remaining samples (0%-100%)
            ramp_configuration (dict): key: 0-indexed ramp id, value: exit threshold at that ramp. 
            Requires ramp ids to be monotonically increasing (from shallow to deep)
            ramp_id (int): 0-indexed ramp ID of the target ramp

        Returns:
            overhead (float), savings (float): Overhead/savings scores
        """
        num_ramps = len(exit_counter)

        # calculate overhead
        # percentage of samples that passes through this ramp
        samples_passed_through_ramp = 100 if ramp_id - 1 not in remaining_rate else remaining_rate[ramp_id - 1]
        samples_remaining_after_ramp = remaining_rate[ramp_id]
        exit_rate = samples_passed_through_ramp - samples_remaining_after_ramp
        # logging.debug(f"Ramp {ramp_id}, samples_passed_through_ramp {samples_passed_through_ramp:<5}, samples_remaining_after_ramp {samples_remaining_after_ramp:<5}, exit_rate {exit_rate:<5}")

        if samples_remaining_after_ramp != 100:  # some samples exited, not totally garbage ramp
            overhead = self.overhead_ramp * samples_passed_through_ramp
        else:  # totally garbage ramp, mark as infinite overhead to be pruned in the first round
            overhead = float("inf")

        # calculate savings
        if not self.favor_early_ramps:
            # method 1: distance saved is from ramp to next exitable location
            savings = 0
            for sample_id in exitable_ramps.keys():
                if sample_id not in actual_exit_ramps:
                    # sample doesn't take any exits
                    continue
                if actual_exit_ramps[sample_id] == ramp_id:
                    # current sample takes exit here
                    if len(exitable_ramps[sample_id]) == 1:
                        distance_to_last_ramp = num_ramps - ramp_id
                        savings += distance_to_last_ramp * (self.overhead_encoder + self.overhead_ramp)
                    else:
                        # calculate the actual distance to the next ramp
                        num_encoders_remaining = (exitable_ramps[sample_id][1]) - ramp_id
                        savings += num_encoders_remaining * self.overhead_encoder
                        # how many other ramps are after the current one
                        num_ramps_remaining = sum([1 for x in ramp_configuration.keys() if x > ramp_id])
                        # if sample can exit from an earlier ramp, take that instead
                        # still wrong because those remaining encoders it needs to pass through
                        # don't all necessarily have ramps attached afterward,
                        # but don't worry about this for now, as num_encoders_remaining is 
                        # empirically small (1 or 2).
                        num_ramps_remaining = min(num_encoders_remaining, num_ramps_remaining)
                        savings += num_ramps_remaining * self.overhead_ramp
            # normalize so that savings and overhead are both
            # calculated using rates, not number of samples
            savings /= total_num_samples / 100
        else:
            # method 2: distance saved is from ramp to end of model
            # compared to method 1, this favors earlier ramps, since the 
            # "distance saved" is higher
            distance_to_back_of_model = num_ramps - ramp_id
            savings = exit_rate * (distance_to_back_of_model * self.overhead_encoder)

        return overhead, savings


    def get_ramp_utility(self, task: str, exit_counter: dict, remaining_rate: dict, 
        ramp_configuration: dict, pickle_dict: dict, total_num_samples: int) -> dict:
        """Calculates the utility score of all ramps

        Args:
            exit_counter (dict): key: 0-indexed ramp id, value: num samples exited
            remaining_rate (dict): key: 0-indexed ramp IDs, value: percentage of remaining samples (0%-100%)
            ramp_configuration (dict): key: 0-indexed ramp id, value: exit threshold at that ramp. 
            Requires ramp ids to be monotonically increasing (from shallow to deep)
            pickle_dict (dict or list): key: either "conf" or "acc", 
            value: list with length = num samples
            total_num_samples (int): total number of samples

        Returns:
            utilities (dict): key: 0-indexed ramp IDs, value: utility
        """
        utilities = {}
        exitable_ramps, actual_exit_ramps = self.get_exitable_ramps(task, pickle_dict, ramp_configuration)

        for ramp_id in [x for x in ramp_configuration.keys()]:
            overhead, savings = self.get_savings_and_overhead_of_ramp(pickle_dict, exitable_ramps, 
                actual_exit_ramps, exit_counter, remaining_rate, total_num_samples, ramp_configuration, ramp_id)
            utility = savings - overhead
            utilities[ramp_id] = utility
            logging.debug(f"Ramp {ramp_id}, utility {utility} = savings {savings} - overhead {overhead}")
        return utilities


    def progressive_pruning(self, task: str, pickle_path: str = None, ramp_configuration: dict = None, pickle_dict: dict = None, ):
        """Run progressive pruning for the model-dataset pair

        Args:
            pickle_path (str): absolute path to the pickle file
            pickle_dict (dict, optional): dict read from an 
            entropy-prediction profile pickle. Defaults to None.
            ramp_configuration (dict, optinal): key: 0-indexed ramp id,
            value: exit threshold at that ramp. Requires ramp
            ids to be monotonically increasing (from shallow to deep).
            If None, enable all ramps with the same golden threshold.

        Returns:
            ramp_configuration, utilities, savings: resulting ramp configurations,
            per-ramp utilities, and their latency savings
        """
        if pickle_path is None:
            pickle_path = self.pickle_path

        if pickle_dict is None:
            with open(pickle_path, "rb") as f:
                pickle_dict = pickle.load(f)

                # HACK(ruipan): temp hack, transform format into cv pickle
                if task == "nlp":
                    pickle_dict = pickle_format_convert(pickle_dict)
        total_num_samples = len(pickle_dict["conf"][0])
        
        num_encoders = num_ramps = self.get_num_ramps(pickle_path)
        if ramp_configuration is None:
            # FIXME(ruipan): ugly, make ramp_configuration non-optional later!
            all_ramp_ids = list(range(num_ramps))
            threshold = self.golden_threshold_dict[self.dataset]
            ramp_configuration = {ramp_id: threshold for ramp_id in all_ramp_ids}

        epoch = 0
        savings_every_epoch = []

        while len(ramp_configuration) > 0:
            accuracy, exit_counter = self.emulate_inference(
                pickle_dict=pickle_dict,
                ramp_configuration=ramp_configuration,
                pickle_path=pickle_path,
                task=task
            )

            savings = self.evaluate_savings(exit_counter, ramp_configuration, total_num_samples, num_encoders)
            logging.debug(f"exit_counter {exit_counter}")
            exit_rate = normalize_exit_rate(exit_counter, total_num_samples)
            logging.debug(f"exit_rate {exit_rate}")
            remaining_rate = get_remaining_rate(exit_rate)
            savings_every_epoch.append(round(savings * 100, 3))
            logging.info('=' * 25 + f"Ramp pruning epoch {epoch}" + '=' * 25)
            logging.debug(f"[{pickle_path}] New emulation with ramp_configuration {ramp_configuration}\nAccuracy {accuracy}\nExit_counter {exit_counter}")
            logging.debug(f"Exit_rate {exit_rate}\nRemaining_rate {remaining_rate}\nTotal savings {savings}\n")

            # plot_exit_rate(exit_rate, epoch=epoch)

            utilities = self.get_ramp_utility(task, exit_counter, remaining_rate, ramp_configuration, pickle_dict, total_num_samples)
            logging.debug(f"Utilities: {utilities}")

            if epoch == 0:  # prune totally-garbage ramps
                garbage_ramps = [ramp_id for ramp_id in utilities.keys() if utilities[ramp_id] == float("-inf")]
                logging.info(f"Garbage ramps: {garbage_ramps}")
                for garbage_ramp in garbage_ramps:
                    ramp_configuration.pop(garbage_ramp)
            else:  # prune ramp with lowest, negative utility
                ramp_id_to_prune = min(utilities, key=utilities.get)
                logging.info(f"Ramp with the lowest utility: {ramp_id_to_prune}")

                # prune until only rmaps with positive utilities are left
                if utilities[ramp_id_to_prune] > self.prune_threshold:
                    logging.info(f"Ramp {ramp_id_to_prune} has positive utility, terminating pruning process")
                    logging.info(f"Remaining #ramps: {len(utilities)}")
                    break
                else:
                    # prune ramp with negative utility
                    ramp_configuration.pop(ramp_id_to_prune)

            logging.info(f"new ramp_configuration: {ramp_configuration}")
            epoch += 1

        logging.info(f"[{pickle_path}]\nLatency savings: all active: {savings_every_epoch[0]}%, after pruning garbage: {savings_every_epoch[1]}%, after pruning all: {savings_every_epoch[-1]}%")
        if any([x < 0 for x in savings_every_epoch]):
            epochs_where_neg_savings = list(np.where(np.array(savings_every_epoch) < 0)[0])
            logging.warning(f"Negative savings at epochs {epochs_where_neg_savings}!")
            logging.warning(f"The following latency savings might be invalid!")
        logging.info(f"Absolute latency improvement: {round(savings_every_epoch[-1] - savings_every_epoch[0], 3)}%")
        logging.info(f"Relative latency improvement: {round(100 * (savings_every_epoch[-1] / savings_every_epoch[0] - 1), 3)}%")
        logging.info("="*75)
        return ramp_configuration.keys(), utilities, savings


    def progressive_addition(self, task: str, pickle_path: str = None, pickle_dict: dict = None):
        """Run progressive addition to come up with a set of ramp configurations.
        NOTE(ruipan): compared to progressive pruning, this may not lead to
        a near-optimal configuration

        Args:
            task (str): cv or nlp
            pickle_path (str): absolute path to the pickle file.
            Defaults to None.
            pickle_dict (dict, optional): dict read from an 
            entropy-prediction profile pickle. Defaults to None.

        Returns:
            list, list, float, float, dict: list of ramp 
            IDs/thresholds in the configuration and their 
            latency savings & accuracy, and its associated
            exit rate
        """
        # start from empty ramp configuration
        curr_ramp_ids = []

        # setup
        if pickle_path is None:
            pickle_path = self.pickle_path

        if pickle_dict is None:
            with open(pickle_path, "rb") as f:
                pickle_dict = pickle.load(f)

                # HACK(ruipan): temp hack, transform format into cv pickle
                if task == "nlp":
                    pickle_dict = pickle_format_convert(pickle_dict)
                
        total_num_samples = len(pickle_dict["conf"][0])
        num_encoders = num_ramps = self.get_num_ramps(pickle_path)
        all_ramp_ids = list(range(num_ramps))
        config_every_epoch = []

        # while we have memory budget remaining:
        while len(curr_ramp_ids) < self.num_ramp_budget:
            logging.debug("{:=^50}".format(f"Ramp selection epoch {len(curr_ramp_ids)}"))
            logging.debug(f"curr_ramp_ids {curr_ramp_ids}")
            candidate_ramps = [x for x in all_ramp_ids if x not in curr_ramp_ids]

            # keep track of current best ramp and its max latency savings
            max_savings, best_ramp_id, best_acc, best_exit_rate = float("-inf"), None, None, None
            # for the remaining inactive ramps, add each one...
            for candidate_ramp_id in candidate_ramps:
                candidate_ramp_ids = sorted(curr_ramp_ids + [candidate_ramp_id])
                
                # do threshold tuning on the current set of ramps
                search_results = self.threshold_tuner.greedy_search(self.args.task, pickle_path, 
                                                                    candidate_ramp_ids, self.args.step_size, data=pickle_dict)
                # search_results: best_config, best_latency_improvement, best_exit_rates, best_acc
                candidate_config = {k: v for k, v in zip(candidate_ramp_ids, search_results[0])}

                # ...evaluate each one...
                accuracy, exit_counter = self.emulate_inference(
                    pickle_dict=pickle_dict,
                    ramp_configuration=candidate_config,
                    pickle_path=pickle_path,
                    task=task
                )

                if accuracy < 100 - self.acc_loss_budget:
                    # FIXME(ruipan): on subdatasets, the emulated accuracy does not match with the 
                    # accuracy returned by the search algorithm
                    logging.debug(f"candidate_ramp_ids {candidate_ramp_ids}, accuracy {accuracy}, continuing")
                    continue
            
                savings = self.evaluate_savings(exit_counter, candidate_ramp_ids, total_num_samples, num_encoders)  # FIXME(ruipan)
                # savings = self.evaluate_savings_v2(task, exit_counter, candidate_ramp_ids, total_num_samples)
                exit_rate = normalize_exit_rate(exit_counter, total_num_samples)
                remaining_rate = get_remaining_rate(exit_rate)
                logging.debug(f"candidate_ramp_id {candidate_ramp_id}, accuracy {accuracy}, savings {savings}")
                if savings > max_savings:
                    max_savings = savings
                    best_ramp_id = candidate_ramp_id
                    best_acc = 100 * search_results[3]
                    best_exit_rate = exit_rate

            if best_ramp_id == None:
                logging.info(f"No more ramps can be added without overflowing the accuracy loss budget, stopping")
                break

            # ...and pick the best one
            logging.debug(f"Among all candidates, ramp {best_ramp_id} brings a max saving of {max_savings}")
            if self.auto_num_ramps and len(curr_ramp_ids) > 0 and \
                max_savings < config_every_epoch[-1][0]:
                logging.info(f"Ignoring ramp budget, latency savings have reached maximum in last epoch, stopping")
                break
            curr_ramp_ids = sorted(curr_ramp_ids + [best_ramp_id])
            logging.debug(f"curr_ramp_ids {curr_ramp_ids}, thresholds {search_results[0]}")
            config_every_epoch.append((max_savings, curr_ramp_ids, best_acc, best_exit_rate, search_results[0]))

        savings_every_epoch = [x[0] for x in config_every_epoch]
        if not is_monotonic(savings_every_epoch, increasing=True):
            logging.error(f"Latency savings every epoch ({savings_every_epoch}) is not monotonically increasing!")

        # TODO: also bookkeep the exit thresholds of the resulting configuration
        logging.info(f"config_every_epoch: {pformat(config_every_epoch)}")
        logging.info(f"curr_ramp_ids {curr_ramp_ids}, thresholds {config_every_epoch[-1][2]} latency savings {savings_every_epoch[-1]}%, acc {best_acc}%")
        return curr_ramp_ids, config_every_epoch[-1][4], savings_every_epoch[-1], config_every_epoch[-1][2], config_every_epoch[-1][3]
    

    def ramp_modification_simpler_data(self, suboptimal_config, blacklist):
        # add an earlier ramp to handle easier data
        curr_earliest = min(suboptimal_config)
        candidate = curr_earliest - 1
        while candidate in blacklist:
            candidate -= 1
        result = [candidate] + list(suboptimal_config)
        logging.info(f"ramp_modification_simpler_data: input {suboptimal_config}, output {result}")
        return result

    def ramp_modification_harder_data(self, suboptimal_config, blacklist, removed_useless_ramp_in_last_round=False):
        # add a deeper ramp to handle easier data
        ###############version w/o blacklist###############
        # splus_ramp_ids = list(suboptimal_config)
        # fallback_ramp_index = next((i for i, v in enumerate(splus_ramp_ids) if v != splus_ramp_ids[0] + i), -1)
        # if fallback_ramp_index != -1:
        #     # left: move all ramps in first chunk of continuous ramps to right by one position
        #     # right: remaining ramps
        #     move_by_distance = 0 if removed_useless_ramp_in_last_round else 1
        #     splus_ramp_ids = [x + move_by_distance for x in splus_ramp_ids[:fallback_ramp_index]] + splus_ramp_ids[fallback_ramp_index:]
        # else:  # already continuous, directly add a ramp to the end
        #     splus_ramp_ids = splus_ramp_ids[1:] + [splus_ramp_ids[-1] + 1]
        # logging.info(f"ramp_modification_harder_data: input {suboptimal_config}, output {splus_ramp_ids}")
        # return splus_ramp_ids
        ###############version with blacklist###############
        splus_ramp_ids = list(suboptimal_config)
        candidate_ramp_id = splus_ramp_ids[0]
        while candidate_ramp_id in suboptimal_config or candidate_ramp_id in blacklist:
            # move pointer backward until we find a position where the ramp is neither active
            # nor was just deactivated
            candidate_ramp_id += 1
        
        result = sorted([candidate_ramp_id] + list(suboptimal_config))
        logging.info(f"ramp_modification_harder_data: input {suboptimal_config}, output {result}")
        return result
        
        
    def online_retuning(self, task: str, pickle_path: str = None, pickle_dict: dict = None):
        """Evaluate the latency savings/accuracy loss of: 
        (1) retuning thresholds and locations on every subdataset
        (2) tune locations once, and then only retune thresholds
            on every subdataset
        """
        # setup
        if pickle_path is None:
            pickle_path = self.pickle_path

        if pickle_dict is None:
            with open(pickle_path, "rb") as f:
                pickle_dict = pickle.load(f)
                # HACK(ruipan): temp hack, transform format into cv pickle
                if task == "nlp":
                    pickle_dict = pickle_format_convert(pickle_dict)

        subdatasets = get_subdatasets(pickle_dict, by_hardness=False)
        initial_config = None
        savings_acc_config_every_subdataset = []

        # ################################################################################
        # # hardness descending, or random
        for subdataset_id, subdataset in enumerate(subdatasets):
        ################################################################################
        # hardness descending, but leave the first subdataset, so rmap 3 doesnt get added
        # for subdataset_id, subdataset in enumerate(list(subdatasets)[1:]):
        ################################################################################
        # # temporary change on qqp to use subdatasets with big gap between optimal & suboptimal
        # # qqp: 0, 1, 8, 9
        # subdatasets = list(subdatasets)
        # subdatasets = subdatasets[0:2] + subdatasets[-2:]
        # for subdataset_id, subdataset in enumerate(subdatasets):
        ################################################################################
        # reverse the order of subdatasets, so they are ordered by hardness ascending
        # for subdataset_id, subdataset in enumerate(reversed(list(subdatasets))):
        ################################################################################
            """
            Naming notion:
            optimal: re-tune thresholds and locations for every subdataset
            suboptimal: re-tune thresholds for every subdataset
            initial: reuse the config from first subdataset
            """
            total_num_samples = len(subdataset['conf'][0])
            logging.info(f"subdataset_id {subdataset_id}, len {total_num_samples}")
            num_samples = len(subdataset["conf"][0])  # len(subdataset)
            num_ramps = len(subdataset["conf"])  # len(subdataset[0]["all_entropies"])
            ################################################################################
            # optimal: retune both locations and thresholds
            ramp_ids, thresholds, _, optimal_accuracy, optimal_exit_rate = self.progressive_addition(task, pickle_dict=subdataset)
            optimal_config = {k: v for k, v in zip(ramp_ids, thresholds)}
            _, exit_counter = self.emulate_inference(
                pickle_dict=subdataset,
                ramp_configuration=optimal_config,
                pickle_path=pickle_path,
                task=task
            )
            optimal_savings = self.evaluate_savings(exit_counter, optimal_config, total_num_samples, 12)
            # XXX(ruipan): actually, the optimal should be progressive_addition done on the whole dataset, not the first subdataset?
            ################################################################################
            # suboptimal: only tune thresholds
            only_tune_threshold = self.threshold_tuner.greedy_search(self.args.task, pickle_path, 
                                                                     list(initial_config.keys()) if initial_config is not None else ramp_ids, 
                                                                     self.args.step_size, data=subdataset)
            if subdataset_id == 0:
                initial_config = {k: v for k, v in zip(ramp_ids, thresholds)}
                logging.debug(f"initial_config set to {initial_config}")
            suboptimal_config = {k: v for k, v in zip(list(initial_config.keys()), only_tune_threshold[0])}
            logging.debug(f"suboptimal_config set to {suboptimal_config}")
            suboptimal_accuracy = only_tune_threshold[3]
            suboptimal_exit_rate = only_tune_threshold[2]
            # suboptimal_savings = only_tune_threshold[1]  # NOTE(ruipan): this uses the savings calculation as defined in threshold_tuner.py
            _, exit_counter = self.emulate_inference(
                pickle_dict=subdataset,
                ramp_configuration=suboptimal_config,
                pickle_path=pickle_path,
                task=task
            )
            suboptimal_savings = self.evaluate_savings(exit_counter, suboptimal_config, total_num_samples, 12)
            exit_rate = normalize_exit_rate(exit_counter, total_num_samples)
            remaining_rate = get_remaining_rate(exit_rate)
            utilities = self.get_ramp_utility(task, exit_counter, remaining_rate, suboptimal_config, subdataset, total_num_samples)
            logging.info(f"suboptimal utilities: {utilities}")

            # suboptimal+: start suboptimal, and run progressive pruning until all ramps have positive utility
            suboptimal_plus_config = copy.deepcopy(suboptimal_config)
            blacklist = []  # ramps that will not be considered as shadow ramps
            while True:
                if all([x > self.prune_threshold for x in utilities.values()]):
                    logging.debug(f"suboptimal+: all ramps now have positive utility!")
                    # already good
                    break
                else:
                    worst_ramp = min(utilities, key=utilities.get)
                    blacklist.append(worst_ramp)
                    logging.debug(f"suboptimal+: pruning ramp {worst_ramp}, suboptimal_plus_config {suboptimal_plus_config}")
                    suboptimal_plus_config.pop(worst_ramp)
                    suboptimal_plus_accuracy, suboptimal_plus_exit_counter = self.emulate_inference(
                        pickle_dict=subdataset,
                        ramp_configuration=suboptimal_plus_config,
                        pickle_path=pickle_path,
                        task=task
                    )
                    suboptimal_plus_savings = self.evaluate_savings(suboptimal_plus_exit_counter, suboptimal_plus_config, total_num_samples, 12)
                    suboptimal_plus_exit_rate = normalize_exit_rate(suboptimal_plus_exit_counter, total_num_samples)
                    suboptimal_plus_remaining_rate = get_remaining_rate(suboptimal_plus_exit_rate)
                    utilities = self.get_ramp_utility(task, suboptimal_plus_exit_counter, suboptimal_plus_remaining_rate, suboptimal_plus_config, subdataset, total_num_samples)

            ################################################################################
            # improve suboptimal by selecting ramp location without a holistic re-search
            ################################################################################
            # when samples become simpler: try adding 1 ramp before the current earliest
            s_simpler = suboptimal_plus_config.keys()
            s_simpler_blacklist = copy.deepcopy(blacklist)
            s_simpler_past_stats = [(suboptimal_plus_savings, suboptimal_plus_config, suboptimal_plus_accuracy, suboptimal_plus_exit_rate)]
            while True:
                s_simpler = self.ramp_modification_simpler_data(s_simpler, s_simpler_blacklist)
                if -1 in s_simpler:
                    logging.info(f"added ramp -1, breaking")
                    break
                s_simpler_search_result = self.threshold_tuner.greedy_search(self.args.task, pickle_path, 
                                                                        s_simpler,
                                                                        self.args.step_size, data=subdataset)
                s_simpler_config = {k: v for k, v in zip(s_simpler, s_simpler_search_result[0])}
                # s_simpler_accuracy = s_simpler_search_result[3]
                # s_simpler_exit_rate = s_simpler_search_result[2]
                s_simpler_accuracy, s_simpler_exit_counter = self.emulate_inference(
                    pickle_dict=subdataset,
                    ramp_configuration=s_simpler_config,
                    pickle_path=pickle_path,
                    task=task
                )
                s_simpler_savings = self.evaluate_savings(s_simpler_exit_counter, s_simpler_config, total_num_samples, 12)
                s_simpler_exit_rate = normalize_exit_rate(s_simpler_exit_counter, total_num_samples)
                s_simpler_remaining_rate = get_remaining_rate(s_simpler_exit_rate)

                if s_simpler_past_stats[-1][0] > s_simpler_savings:
                    break
                # NOTE(ruipan): add a step here that deactivates ramps with low (<2%?) exit rates
                # UPDATE: instead of based on exit rate, deactivate based on utilities?
                # if any([x < 0.02 for x in s_simpler_exit_rate]):
                #     useless_ramp_ids = [id for i, id in enumerate(s_simpler) if s_simpler_exit_rate[i] < 0.02]
                #     useful_ramp_ids = [x for x in s_simpler if x not in useless_ramp_ids]
                #     logging.info(f"useless_ramp_ids {useless_ramp_ids}, s_simpler updated from {s_simpler} to {useful_ramp_ids}")
                #     s_simpler = useful_ramp_ids

                # if adding the new ramp caused some other ramp to go useless, deactivate that ramp, and add it to the blacklist
                utilities = self.get_ramp_utility(task, s_simpler_exit_counter, s_simpler_remaining_rate, s_simpler_config, subdataset, total_num_samples)
                logging.debug(f"s_simpler utilities: {utilities}")
                if any([x > self.prune_threshold for x in utilities.values()]):
                    worst_ramp = min(utilities, key=utilities.get)
                    s_simpler_blacklist.append(worst_ramp)
                    s_simpler_copy_config = copy.deepcopy(s_simpler_config)
                    s_simpler_copy_config.pop(worst_ramp)

                    s_simpler_copy_accuracy, s_simpler_copy_exit_counter = self.emulate_inference(
                        pickle_dict=subdataset,
                        ramp_configuration=s_simpler_copy_config,
                        pickle_path=pickle_path,
                        task=task
                    )
                    s_simpler_copy_savings = self.evaluate_savings(s_simpler_copy_exit_counter, s_simpler_copy_config, total_num_samples, 12)
                    s_simpler_copy_exit_rate = normalize_exit_rate(s_simpler_copy_exit_counter, total_num_samples)
                    if s_simpler_copy_savings > s_simpler_savings:
                        s_simpler_past_stats.append((s_simpler_copy_savings, s_simpler_copy_config, s_simpler_copy_accuracy, s_simpler_copy_exit_rate))
                        continue
                # otherwise, just add a ramp and not remove one
                s_simpler_past_stats.append((s_simpler_savings, s_simpler_config, s_simpler_accuracy, s_simpler_exit_rate))
            ################################################################################
            # when samples become harder: try removing earliest ramp, and add one at 
            s_harder = suboptimal_plus_config.keys()
            s_harder_past_stats = [(suboptimal_plus_savings, suboptimal_plus_config, suboptimal_plus_accuracy, suboptimal_plus_exit_rate)]
            removed_useless_ramp_in_last_round = False
            while True:
                s_harder = self.ramp_modification_harder_data(s_harder, blacklist, removed_useless_ramp_in_last_round)
                if removed_useless_ramp_in_last_round:  # reset flag
                    removed_useless_ramp_in_last_round = False
                if 12 in s_harder:
                    logging.info(f"added ramp 12, breaking")
                    break
                s_harder_search_result = self.threshold_tuner.greedy_search(self.args.task, pickle_path, 
                                                                        s_harder,
                                                                        self.args.step_size, data=subdataset)
                s_harder_config = {k: v for k, v in zip(s_harder, s_harder_search_result[0])}
                # s_harder_accuracy = s_harder_search_result[3]
                # s_harder_exit_rate = s_harder_search_result[2]
                s_harder_accuracy, s_harder_exit_counter = self.emulate_inference(
                    pickle_dict=subdataset,
                    ramp_configuration=s_harder_config,
                    pickle_path=pickle_path,
                    task=task
                )
                s_harder_savings = self.evaluate_savings(s_harder_exit_counter, s_harder_config, total_num_samples, 12)
                s_harder_exit_rate = normalize_exit_rate(s_harder_exit_counter, total_num_samples)
                if s_harder_past_stats[-1][0] > s_harder_savings:
                    break
                # NOTE(ruipan): add a step here that deactivates ramps with low (<2%?) exit rates
                # UPDATE: instead of based on exit rate, deactivate based on utilities?
                # if any([x < 0.02 for x in s_harder_exit_rate]):
                #     useless_ramp_ids = [id for i, id in enumerate(s_harder) if s_harder_exit_rate[i] < 0.02]
                #     useful_ramp_ids = [x for x in s_harder if x not in useless_ramp_ids]
                #     logging.info(f"useless_ramp_ids {useless_ramp_ids}, s_harder updated from {s_harder} to {useful_ramp_ids}")
                #     s_harder = useful_ramp_ids
                #     removed_useless_ramp_in_last_round = True
                s_harder_past_stats.append((s_harder_savings, s_harder_config, s_harder_accuracy, s_harder_exit_rate))
            ################################################################################
            logging.info(f"Subdataset {subdataset_id},\n\t"
                         f"optimal: savings {optimal_savings}, acc {optimal_accuracy}, exit rate {optimal_exit_rate}\n\t"
                         f"suboptimal: savings {suboptimal_savings}, acc {suboptimal_accuracy}, exit rate {suboptimal_exit_rate}\n\t"
                         f"suboptimal_plus: savings {suboptimal_plus_savings}, acc {suboptimal_plus_accuracy}, exit rate {suboptimal_plus_exit_rate}\n\t"
                         f"s_simpler: savings {s_simpler_past_stats[-1][0]}, acc {s_simpler_past_stats[-1][2]}, exit rate {s_simpler_past_stats[-1][3]}\n\t"
                         f"s_harder: savings {s_harder_past_stats[-1][0]}, acc {s_harder_past_stats[-1][2]}, exit rate {s_harder_past_stats[-1][3]}\n\t"
                         f"optimal_config {optimal_config},\n\t"
                         f"suboptimal_config {suboptimal_config},\n\t"
                         f"suboptimal_plus_config {suboptimal_plus_config},\n\t"
                         f"s_simpler {s_simpler_past_stats[-1][1]},\n\t"
                         f"s_harder {s_harder_past_stats[-1][1]}")
            savings_acc_config_every_subdataset.append((
                optimal_savings, optimal_accuracy, optimal_config,
                suboptimal_savings, suboptimal_accuracy, suboptimal_config,
                s_simpler_past_stats[-1][0], s_simpler_past_stats[-1][2], s_simpler_past_stats[-1][1],
                s_harder_past_stats[-1][0], s_harder_past_stats[-1][2], s_harder_past_stats[-1][1],
            ))
        
        return savings_acc_config_every_subdataset



    def ramp_configs_generator(self, num_ramps: int, threshold_options: list = None):
        """Given a number of ramp ids and a list of thresholds to choose from,
        yield all combinations of ramp ids and threshold options in a memory-efficient way.

        Args:
            num_ramps (int): number of all available ramps
            threshold_options (list, optional): list of floats that represents
            possible thresholds an exit can use. Defaults to None.
        
        Yields:
            dict: a ramp configuration
        """
        if not threshold_options:
            # TODO(ruipan): remove self.golden_threshold_dict later
            threshold_options = [self.golden_threshold_dict[self.dataset]]

        # XXX(ruipan): shrink search space
        # all_ramp_ids = np.arange(num_ramps)
        # all_ramp_ids = np.arange(3, num_ramps)
        all_ramp_ids = np.array([3, 4, 5, 6, 7, 8])
        
        curr_repeat_num = len(all_ramp_ids)
        logging.info(f"all_ramp_ids: {all_ramp_ids}")

        combinations_generator = combinations(all_ramp_ids, curr_repeat_num)
        while True:
            try:  # generate all configs with curr_repeat_num ramps
                ramp_ids = next(combinations_generator)  # choose a set of ramps first
                # then, generate all possible threshold combinations
                for threshold_combination in product(threshold_options, repeat=curr_repeat_num):
                    config = {}
                    for ramp_id, threshold in zip(ramp_ids, threshold_combination):
                        config[ramp_id] = threshold
                    yield config
            except StopIteration:
                # finished generating configs with curr_repeat_num ramps,
                logging.info("{:=^50}".format(f"Finished emulating inference on configs with {curr_repeat_num} ramps"))
                curr_repeat_num -= 1  
                if curr_repeat_num == 0:  # finished generating all configs
                    break
                # move on to generating configs with {curr_repeat_num-1} ramps
                combinations_generator = combinations(all_ramp_ids, curr_repeat_num)


    def config_consumer(self, config_queue, result_queue, pickle_dict, pickle_path, task, total_num_samples, num_encoders):
        """A parallel worker/consumer that reads configurations
        from a queue, emulates inference, and reports the best
        configuration that it has processed
        """
        best_savings = float("-inf")
        best_config = None
        while True:
            config = config_queue.get()
            if config is None:
                break
            accuracy, exit_counter = self.emulate_inference(
                pickle_dict=pickle_dict,
                ramp_configuration=config,
                pickle_path=pickle_path,
                task=task
            )
            savings = self.evaluate_savings(exit_counter, config, total_num_samples, num_encoders)

            if accuracy > (100 - self.acc_loss_budget) and savings > best_savings:
                best_savings = savings
                best_config = config
        worker_id = multiprocessing.current_process().pid
        logging.info(f"worker_id {worker_id}, best_savings {best_savings}, best_config {best_config}")
        result_queue.put((best_savings, best_config))


    def get_optimal_config(self, task: str, pickle_path: str = None, pickle_dict: dict = None, enable_multiprocessing: bool = False):
        """Search and return the global optimal ramp configuration 
        (location & thresholds)

        Args:
            task (str): cv or nlp
            pickle_path (str): absolute path to the pickle file.
            Defaults to None.
            pickle_dict (dict, optional): dict read from an 
            entropy-prediction profile pickle. Defaults to None.
            enable_multiprocessing (bool, optional): if enabled, spawn
            multipule processes to parallelize the search. Defaults to False.
        """
        # setup
        if pickle_path is None:
            pickle_path = self.pickle_path
        if pickle_dict is None:
            with open(pickle_path, "rb") as f:
                pickle_dict = pickle.load(f)
                if task == "nlp":  # HACK(ruipan): temp hack, transform format into cv pickle
                    pickle_dict = pickle_format_convert(pickle_dict)
        
        total_num_samples = len(pickle_dict["conf"][0])
        num_encoders = self.get_num_ramps(pickle_path)
        # ramp_configs_generator = self.ramp_configs_generator(num_encoders)  # uniform threshold
        # threshold_options = [x / 10000 for x in range(0, 5125, 125)]
        threshold_options = [x / 10000 for x in range(0, 6000, 1000)]  # XXX(ruipan): attempt to shrink the search space
        logging.info(f"Threshold options: {threshold_options}")
        ramp_configs_generator = self.ramp_configs_generator(num_encoders, threshold_options=threshold_options)
        if not enable_multiprocessing:
            best_config = None
            best_latency_savings = float("-inf")
            for config in ramp_configs_generator:
                accuracy, exit_counter = self.emulate_inference(
                    pickle_dict=pickle_dict,
                    ramp_configuration=config,
                    pickle_path=pickle_path,
                    task=task
                )
                savings = self.evaluate_savings(exit_counter, config, total_num_samples, num_encoders)
                if (accuracy >= 100 - self.acc_loss_budget) and (savings > best_latency_savings):
                    best_config = config
                    best_latency_savings = savings
            logging.info(f"Best config {best_config}, latency savings {best_latency_savings * 100}%")
        else:
            num_workers = multiprocessing.cpu_count() * 2
            logging.info(f"Parallelizing search with {num_workers} workers")
            # producer put configs in this queue, consumers get from this queue
            # NOTE(ruipan) reference: https://stackoverflow.com/a/43079667/9601555
            config_queue = multiprocessing.Queue(maxsize=num_workers)
            # consumers each put their globally-optimal results in this queue
            result_queue = multiprocessing.Queue()

            pool = multiprocessing.Pool(num_workers, initializer=self.config_consumer,
                                        initargs=(config_queue, result_queue, pickle_dict, pickle_path, task, total_num_samples, num_encoders,))
            # with get_context("spawn").Pool(num_workers, initializer=config_consumer) as pool:
            for config in ramp_configs_generator:
                config_queue.put(config)  # blocks until config_queue falls below its max size
            for _ in range(num_workers):
                config_queue.put(None)
            logging.info(f"Doing pool.close()")
            pool.close()
            logging.info(f"Doing pool.join()")
            pool.join()  # https://docs.python.org/3/library/multiprocessing.html

            # aggregate optimal across all workers
            best_savings = float("-inf")
            best_config = None
            for _ in range(num_workers):
                savings, config = result_queue.get()
                # logging.debug(f"worker ?, savings {savings}")
                if savings > best_savings:
                    best_savings = savings
                    best_config = config

            logging.info(f"Best savings {best_savings} comes from config {best_config}")
            return best_config, best_savings


if __name__ == "__main__":
    # NOTE(ruipan): this might prevent multiprocessing Pool from 
    # getting stuck: https://pythonspeed.com/articles/python-multiprocessing/
    # https://github.com/pytorch/pytorch/issues/3492#issuecomment-341965218

    parser = argparse.ArgumentParser(description="PyTorch Cifar10 Example")
    parser.add_argument('--offline_data_path', type=str, default='./pickles/resnet56_cifar10.pickle')
    args, unknown = parser.parse_known_args()
    multiprocessing.set_start_method("spawn")

    rp = RampSelector(args)
    # rp.get_optimal_config("nlp", enable_multiprocessing=True)
    # rp.get_optimal_config("nlp")
    # rp.progressive_pruning("nlp")
    rp.progressive_addition("nlp")

    # rp = RampSelector("/home/ruipan/deebert/entropy_pickles/cifar10_resnet56.pickle")
    # rp.get_optimal_config("cv", enable_multiprocessing=True)
    # rp.get_optimal_config("cv")
    # rp.progressive_pruning("cv")
    # rp.progressive_addition("cv")
