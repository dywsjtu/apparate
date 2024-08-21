import math
import copy
import os
import random
import torch
import pickle
import torch.nn as nn
import numpy as np
from collections import defaultdict

# all possible batch sizes to choose from
supported_batch_sizes = [1, 2, 4, 8, 16, 32, 64]
# supported model architectures
all_cv_models = [
    # distiller
    "resnet20_cifar10", "resnet32_cifar10", "resnet44_cifar10",
    "resnet56_cifar10", "resnet110_cifar10", "resnet1202_cifar10",
    "resnet20_cifar100", "resnet32_cifar100", "resnet44_cifar100",
    "resnet56_cifar100", "resnet110_cifar100", "resnet1202_cifar100",
    # custom datasets
    "resnet18_waymo", "resnet50_waymo",
    "resnet18_urban", "resnet50_urban",
]
all_nlp_models = [
    # dee{bert,roberta,distilbert}
    "bert-base-uncased",
    "bert-large-uncased",
    # "roberta",
    "distilbert-base-uncased",
]
all_supported_models = all_cv_models + all_nlp_models
# supported datasets
all_cv_datasets = [
    # CV
    "cifar10", "cifar100",
    # custom CV datasets
    "waymo", "urban",
]
all_nlp_datasets = [
    # NLP - GLUE
    "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst-2", "wnli",
]
all_supported_datasets = all_cv_datasets + all_nlp_datasets


def round_up_batch_size(batch_size):
    """Rounds up the batch size to 2^n.
    This is mostly used to accommodate the last batch in a dataset,
    where the bs might not be 2^n.

    Args:
        batch_size (int): batch size

    Returns:
        int: rounded-up batch size
    """
    return min([x for x in supported_batch_sizes if x >= batch_size])


def set_seeds(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def nth_repl(s, sub, repl, n):
    # https://stackoverflow.com/a/35092436/9601555
    """Replace the n-th occurrence of sub in s with repl

    Args:
        s (str): original string
        sub (str): substring to be substituted
        repl (str): replacement substring
        n (int): 1-indexed occurrence ID

        In [14]: s = "foobarfoofoobarbar"
        In [15]: nth_repl(s, "bar", "replaced", 3)
        Out[15]: 'foobarfoofoobarreplaced'

    Returns:
        _type_: _description_
    """
    find = s.find(sub)
    # If find is not -1 we have found at least one match for the substring
    i = find != -1
    # loop util we find the nth or we find no match
    while find != -1 and i != n:
        # find + 1 means we start searching from after the last match
        find = s.find(sub, find + 1)
        i += 1
    # If i is equal to n we found nth match so replace
    if i == n:
        return s[:find] + repl + s[find+len(sub):]
    return s


def compare_lists(list1, list2):
    # Compare the lists element-wise
    for element1, element2 in zip(list1, list2):
        if element1 < element2:
            return True
        elif element1 > element2:
            return False

    return len(list1) < len(list2)


def is_monotonic(l, increasing):
    """Checks whether items in a list are monotonically
    increasing/decreasing

    Args:
        l (list): list of items
        increasing (bool): True if checking
        for monotonic increasing, and False if
        checking for monotonic decreasing.

    Returns:
        bool: whether items in a list are monotonically 
        increasing/decreasing
    """
    l = list(l)
    if increasing:
        return all(l[i] <= l[i+1] for i in range(len(l)-1))
    else:  # check for monotonic decreasing
        return all(l[i] >= l[i+1] for i in range(len(l)-1))


def normalize_exit_rate(exit_counter, total_num_samples):
    """Given an exit counter as a dict,
    return the normalized exit rates such that 
    the exit rate at each ramp sum up to 100.

    Args:
        exit_counter (dict): key: 0-indexed ramp id, value: number of samples exited
        total_num_samples (int): total number of samples in the dataset

    Returns:
        dict: key: 0-indexed ramp id, value: exit rate %
    """
    exit_rate = {}
    for ramp_id, num_samples_exited in exit_counter.items():
        rate = round(num_samples_exited / total_num_samples * 100, 5)
        exit_rate[ramp_id] = rate
    return exit_rate


def get_remaining_rate(exit_rate):
    """Given the normalized exit rate at each ramp,
    return the remaining sample rate (CDF) at
    each ramp

    Args:
        exit_rate (dict): key: 0-indexed ramp IDs,
        value: normalized exit rate (0-1)
        OR: (np.ndarray): index x: normalized exit
        rate at ramp of index x

    Returns:
        remaining_rate (dict): key: 0-indexed ramp IDs,
        value: percentage of remaining samples (0-1).
        OR: (np.ndarray): index x: normalized remaining
        rate after ramp of index x
    """
    if type(exit_rate) is dict:
        remaining_rate = {}
        for ramp_id, exit_rate_at_ramp in exit_rate.items():
            last_ramp_rate = 1 if ramp_id == 0 else remaining_rate[ramp_id - 1]
            remaining_rate[ramp_id] = last_ramp_rate - exit_rate_at_ramp
        return remaining_rate
    elif type(exit_rate) is np.ndarray:
        remaining_rate = []
        for ramp_index, exit_rate_at_ramp in enumerate(exit_rate):
            last_ramp_rate = 1 if ramp_index == 0 else remaining_rate[ramp_index - 1]
            remaining_rate.append(last_ramp_rate - exit_rate_at_ramp)
        return np.array(remaining_rate)
    else:
        raise NotImplementedError
    

def merge_batches(batches, total_num_ramps):
    merged = {'conf': [[] for _ in range(total_num_ramps)],
        'acc': [[] for _ in range(total_num_ramps)]}
    for i, batch in enumerate(batches):
        for ramp_id in range(total_num_ramps):
            merged["conf"][ramp_id] += batch["conf"][ramp_id]
            merged["acc"][ramp_id] += batch["acc"][ramp_id]
    return merged


def get_avg_exit_point(ramp_ids, exit_rate, latency_calc_list):
    """Returns the average exit point of all samples.

    Args:
        ramp_ids (list): list of ramp ids
        exit_rate (np.ndarray): exit rate at all ramps and the final model, 
            sums up to 1
        latency_calc_list (list): list (len = num_all_ramps + 1) of tuples.
            Index x: vanilla model latency before ramp x and the latency of ramp x.
            Last index: (vanilla model latency, None).

    Returns:
        avg_exit_point (float): normalized (0-1) average exit point of all samples
    """
    # vanilla model latency
    vanilla_latency = latency_calc_list[-1][0]
    # normalized latency (w.r.t to vanilla model latency) of all current exit points
    vanilla_distances = np.array(
        [latency_calc_list[i][0] / vanilla_latency for i in ramp_ids] + [1.0])
    print(f"vanilla_distances of ramps {ramp_ids}: {vanilla_distances}")
    avg_exit_point = sum(exit_rate * vanilla_distances)
    return avg_exit_point


def get_batches(pickle_dict: dict, batch_size: int, batch_size_schedule: list = None):
    """Given a fixed batch size or a schedule of the per-batch bs, 
    return the entropy dict for emulating serving

    Args:
        pickle_dict (dict): _description_
        batch_size (int): fixed batch size
        batch_size_schedule (list): list of ints of batch sizes in every batch

    Yields:
        _type_: _description_
    """
    total_num_samples = len(pickle_dict["conf"][0])
    num_ramps = len(pickle_dict["conf"]) - 1

    if batch_size_schedule is None:  # fixed batch size
        start_indices = list(
            range(0, total_num_samples, batch_size))
        end_indices = [
            x + batch_size for x in start_indices][:-1] + [total_num_samples - 1]
    else:  # dynamic/adaptive batch size, read from schedule
        start_indices = [sum(batch_size_schedule[:i]) for i in range(len(batch_size_schedule))]
        end_indices = [sum(x) for x in zip(start_indices, batch_size_schedule)]

    for start, end in zip(start_indices, end_indices):
        assert end - start <= max(supported_batch_sizes), \
            f"Batch size in schedule exceeded max supported by our system!"
        batch = {"conf": [], "acc": []}
        if start == end:
            continue
        for ramp_id in range(num_ramps):
            batch["conf"].append(
                pickle_dict["conf"][ramp_id][start:end])
            batch["acc"].append(
                pickle_dict["acc"][ramp_id][start:end])
        yield batch


def get_subdatasets(pickle_dict: dict, by_hardness: bool = False, num_subdatasets: int = 10):
    print(f"pickle_dict.keys() {pickle_dict.keys()}")  # seems to be a pickle format issue (key rn is sample id, needs to be conf and acc)
    total_num_samples = len(pickle_dict["conf"][0])
    num_ramps = len(pickle_dict["conf"]) - 1
    print(f"total_num_samples {total_num_samples}, num_ramps {num_ramps}")
    if by_hardness:
        #############################################################################
        # different methods of partitioning a subdataset
        #############################################################################
        # method 1: use entropy at first ramp as proxy for hardness
        # works well for QQP, but less so for others
        num_samples_in_subdataset = math.ceil(
            total_num_samples / num_subdatasets)

        # get the indices of sample ids sorted by entropy at first ramp ascending
        sorted_index = np.argsort(pickle_dict["conf"][0])

        start_indices = list(
            range(0, total_num_samples, num_samples_in_subdataset))
        end_indices = [
            x + num_samples_in_subdataset for x in start_indices][:-1] + [total_num_samples - 1]

        for start, end in zip(start_indices, end_indices):
            subdataset = {"conf": [], "acc": []}
            for ramp_id in range(num_ramps + 1):
                conf = np.array(pickle_dict["conf"][ramp_id])[
                    sorted_index[start:end]]
                conf = list(conf)
                acc = np.array(pickle_dict["acc"][ramp_id])[
                    sorted_index[start:end]]
                acc = list(acc)
                subdataset["conf"].append(conf)
                subdataset["acc"].append(acc)
            yield subdataset
        #############################################################################
        # method 2.1: look at the series of predictions made by the ramps.
        # the earlier "a correct prediction" appears, the easier an input is.
        #############################################################################
        # method 2.2: look at the series of predictions made by the ramps.
        # the earlier "a series of correct predictions until the end of the model"
        # appears, the easier an input is.
        #############################################################################
        #
        #############################################################################
    else:
        num_samples_in_subdataset = math.ceil(
            total_num_samples / num_subdatasets)

        start_indices = list(
            range(0, total_num_samples, num_samples_in_subdataset))
        end_indices = [
            x + num_samples_in_subdataset for x in start_indices][:-1] + [total_num_samples - 1]

        for start, end in zip(start_indices, end_indices):
            subdataset = {"conf": [], "acc": []}
            for ramp_id in range(num_ramps + 1):
                subdataset["conf"].append(
                    pickle_dict["conf"][ramp_id][start:end])
                subdataset["acc"].append(
                    pickle_dict["acc"][ramp_id][start:end])
            yield subdataset


# def get_subdatasets_v2(pickle_dict: dict, by_hardness: bool = False):
#     # outdated, nlp pickle format
#     total_num_samples = len(pickle_dict)
#     num_ramps = len(pickle_dict[0]["all_entropies"])

#     if by_hardness:
#         num_subdatasets = 10
#         num_samples_in_subdataset = math.ceil(total_num_samples / num_subdatasets)

#         # get the indices of sample ids sorted by entropy at first ramp ascending
#         sorted_index = np.argsort([sample[0]["all_entropies"] for sample in pickle_dict.values()])

#         start_indices = list(range(0, total_num_samples, num_samples_in_subdataset))
#         end_indices = [x + num_samples_in_subdataset for x in start_indices][:-1] + [total_num_samples - 1]

#         for start, end in zip(start_indices, end_indices):
#             subdataset = {}
#             for sample_id, profile in pickle_dict.items():
#                 if sample_id in sorted_index[start:end]:
#                     subdataset[sample_id] = profile
#             yield subdataset
#     else:
#         # at least 1000 samples in each subdataset for statistical significance
#         num_samples_in_subdataset = 1000

#         start_indices = list(range(0, total_num_samples, num_samples_in_subdataset))
#         end_indices = [x + num_samples_in_subdataset for x in start_indices][:-1] + [total_num_samples - 1]

#         for start, end in zip(start_indices, end_indices):
#             subdataset = {k: v for k, v in pickle_dict.items() if start <= k < end}
#             yield subdataset


def pickle_format_convert(pickle_dict):
    """
    Converts an NLP pickle's format to match with that of CV pickles.
    nlp/deebert pickle format: {
        sample_id: {
            "all_entropies": [],  # list of entropies of this sample at each ramp
            "all_logits": [],  # list of logits of this sample at each ramp
            "all_predictions": [],  # list of predictions of this sample at each ramp
            "orig_model_prediction": [],  # ground-truth label from dataset
        }
    }

    cv/distiller pickle format: {
        "conf": [  # confidence = 1 - entropy
            [sample_0_entropy_ramp_0, sample_1_entropy_ramp_0, ...],  # float between 0 and 0.5 
            [sample_0_entropy_ramp_1, sample_1_entropy_ramp_1, ...],
            ...
            # original model is considered a ramp
        ],
        "acc": [
            [sample_0_ee_prediction_ramp_0, sample_1_ee_prediction_ramp_0, ...],  # bool
            [sample_0_ee_prediction_ramp_1, sample_1_ee_prediction_ramp_1, ...],
            ...
        ],
    }

    Args:
        pickle_dict (dict): nlp pickle

    Returns:
        dict: nlp pickle with cv pickle's format
    """
    num_samples = len(pickle_dict)
    num_ramps = len(pickle_dict[0]["all_entropies"])
    # print(f"num_samples {num_samples}, num_ramps {num_ramps}")
    """
    for now, conf and acc are: [
        [sample_0_entropy_ramp_0, sample_0_entropy_ramp_1, ...],
        [sample_1_entropy_ramp_0, sample_1_entropy_ramp_1, ...], ...
    ]
    """
    conf = [[] for _ in range(num_samples)]
    acc = [[] for _ in range(num_samples)]
    
    if type(pickle_dict) == dict:
        pickle_dict = list(pickle_dict.values())

    for sample_id, sample in enumerate(pickle_dict):
        # also change entropy definition to match cv pickle format
        # removes deebert ramp at end of model
        conf[sample_id] = [
            1 - x for x in list(sample["all_entropies"])][:-1] + [None]
        label = sample["orig_model_prediction"][0]
        acc[sample_id] = [label == p[0]
                          for p in sample["all_predictions"]][:-1] + [True]

    # transpose list of lists
    conf = list(map(list, zip(*conf)))
    acc = list(map(list, zip(*acc)))

    return {
        "conf": conf,
        "acc": acc,
    }


def query_performance(config, all_ramps_conf, all_ramps_acc, ramp_ids, latency_config, baseline):
    """
    Given a configuration, return the performance of the model
    under this configuration.

    Args:
        config (list): list of exit thresholds
        all_ramps_conf (list): confidences of all ramps for all samples
        all_ramps_acc (list): accuracies of all ramps for all samples
        ramp_ids (list): list of ramp ids
        latency_config (numpy array): latency configuration
        baseline (float): baseline latency

    Returns:
        acc (float): accuracy of the model under this configuration
        latency (float): latency of the model under this configuration
        exit_rate (numpy array): exit rate of each ramp
    """
    correct = 0
    nums_exit = [0 for i in range(len(ramp_ids) + 1)]
    for i in range(len(all_ramps_conf[ramp_ids[0]])):
        earlyexit_taken = False
        for j in range(len(ramp_ids)):
            id = ramp_ids[j]
            if 1 - all_ramps_conf[id][i] < config[j]:
                nums_exit[j] += 1
                earlyexit_taken = True
                if all_ramps_acc[id][i]:
                    correct += 1
                break
        if not earlyexit_taken:
            nums_exit[-1] += 1
            correct += 1
    exit_rate = np.array(
        [(n+0.0)/len(all_ramps_conf[ramp_ids[0]]) for n in nums_exit])
    acc = round((correct+0.0)/len(all_ramps_conf[ramp_ids[0]]), 7)
    latency_improvement = (
        baseline - sum(exit_rate * latency_config)) / baseline * 100
    return acc, latency_improvement, exit_rate


def get_latency_config(path, ramp_ids):
    if "resnet" in path and "cifar" in path:
        res = [0.0 for _ in ramp_ids]
        for i in range(len(ramp_ids)):
            res[i] = 0.25 + 0.11*(i+1) + 0.31*(ramp_ids[i] + 1)
        res.append(8.72 + 0.11*len(ramp_ids))
        return np.array(res), 8.72
    elif "bert" in path:
        res = [0.0 for _ in ramp_ids]
        for i in range(len(ramp_ids)):
            res[i] = 0.106*(i+1) + 0.622*(ramp_ids[i] + 1)
        res.append(7.464 + 0.106*len(ramp_ids))
        # print(res)
        return np.array(res), 7.464
    elif "resnet50" in path and "phx" in path:
        res = [0.0 for _ in ramp_ids]
        for i in range(len(ramp_ids)):
            res[i] = 0.25 + 0.11*(i+1) + 0.31*(ramp_ids[i] + 1)
        res.append(5.31 + 0.11*len(ramp_ids))
        return np.array(res), 5.31
    elif "resnet18" in path and "urban" in path:
        res = [0.0 for _ in ramp_ids]
        for i in range(len(ramp_ids)):
            res[i] = 0.25 + 0.11*(i+1) + 0.31*(ramp_ids[i] + 1)
        res.append(2.83 + 0.11*len(ramp_ids))
        return np.array(res), 2.83


def num_trials_for_global_optimal(num_ramps, num_threshold_options):
    total_num = 0
    for curr_num_ramps in reversed(range(1, num_ramps+1)):
        # number of ramp combinations for curr_num_ramps
        num_ramp_combinations = math.comb(num_ramps, curr_num_ramps)
        # number of threshold combinations for curr_num_ramps
        num_threshold_combinations = num_threshold_options ** curr_num_ramps
        total_num += num_ramp_combinations * num_threshold_combinations
    print(
        f"total num trials for optimal grid search (num_ramps {num_ramps}, num_threshold_options {num_threshold_options}): {total_num}")
    return total_num


def get_batch(data, batch_size):
    """Get a batch of data.

    Args:
        data (list): list of data
        batch_size (int): batch size

    Yields:
        list: batch of data
    """
    max_idx = len(data['conf'][0])

    for i in range(0, max_idx, batch_size):
        res = {'conf': [data['conf'][j][i: min(i + batch_size, max_idx)] for j in range(len(data['conf']))],
               'acc': [data['acc'][j][i: min(i + batch_size, max_idx)] for j in range(len(data['acc']))]}
        yield res

    # for i in range(50):
    #     with open('./pickles/urban-{}_resnet18.pickle'.format(i), 'rb') as f:
    #         data = pickle.load(f)
    #     yield data


def parse_profile(profile):
    """Perform intermediate processing to the raw model profile, so that
    we don't need to access the profile object every time we calculate
    latency savings.

    Args:
        profile (Profiler.profile): model profile

    Returns:
        latency_calc_list (list): list (len = num_all_ramps + 1) of tuples.
            Index x: vanilla model latency before ramp x and the latency of ramp x.
            Last index: (vanilla model latency, None).
    """
    # Distiller injection method (sequential execution)
    latency_calc_list = []
    all_branchpoints = profile.get_all_children_with_name(
        "branch_net")
    
    assert all_branchpoints != [], f"No branchpoints found!"
    for ramp in all_branchpoints:
        vanilla_latency_before_ramp = ramp.vanilla_latency_up_until_me  # TODO(ruipan): change to me once all profile pickles are unified
        # vanilla_latency_before_ramp = ramp.vanilla_latency_after_me  # NOTE(ruipan): this is for old versions of profile pickles
        ramp_latency = ramp.fwd_latency
        latency_calc_list.append(
            (vanilla_latency_before_ramp, ramp_latency,))

    # add an entry for the vanilla model latency
    vanilla_model_latency = profile.vanilla_latency_up_until_me  # TODO(ruipan): change to me once all profile pickles are unified
    # vanilla_model_latency = profile.vanilla_latency_after_me  # NOTE(ruipan): this is for old versions of profile pickles
    latency_calc_list.append((vanilla_model_latency, None,))

    return latency_calc_list


def get_ramp_latencies(active_ramp_ids: list, latency_calc_list: list):
    """Compute the latency of exiting from different ramp locations 
    in a numpy array given a ramp configuration.

    Args:
        active_ramp_ids (list): 0-indexed ramp ids that are currently active
        latency_calc_list (list): list of tuples, each tuple contains
            the vanilla latency before the ramp and the ramp latency

    Returns:
        np.array: index x: latency in ms of exiting from the xth ramp.
        float: latency in ms of traversing the vanilla model
    """
    assert is_monotonic(active_ramp_ids, increasing=True), \
        "Ramp IDs in configuration are not in order!"
    assert latency_calc_list != [], f"latency_calc_list not yet calculated!"

    # latency in ms of the vanilla model without any ramps
    vanilla_latency = latency_calc_list[-1][0]

    latencies = []
    for ramp_id in active_ramp_ids:
        latencies.append(
            latency_calc_list[ramp_id][0] +
            # vanilla latency + sum of all ramp latencies prior to current ramp
            sum([latency_calc_list[id][1]
                for id in active_ramp_ids if id <= ramp_id])
        )
    # add latency for samples that went through all exits but did not exit
    latencies.append(vanilla_latency +
                     sum(latency_calc_list[id][1] for id in active_ramp_ids))
    latencies = np.array(latencies)
    return latencies, vanilla_latency


def serve_batch(thresholds, batch_data, ramp_ids, latency_calc_list):
    """Given a configuration, return the performance of the model
        under this configuration.

    Args:
        thresholds (list): list of exit thresholds
        batch_data (dict): batch of data
        ramp_ids (list): list of ramp ids
        latency_calc_list (list): list (len = num_all_ramps + 1) of tuples.
            Index x: vanilla model latency before ramp x and the latency of ramp x.
            Last index: (vanilla model latency, None).

    Returns:
        acc (float): accuracy
        latency_improvement (float): latency improvement
        exit_rate (numpy array): exit rate of each ramp
    """
    all_ramps_conf = batch_data['conf']
    all_ramps_acc = batch_data['acc']
    # latency_config, baseline = \
    #     get_latency_config('resnet18_urban_config', ramp_ids)

    latency_config, baseline = get_ramp_latencies(ramp_ids, latency_calc_list)

    return query_performance(thresholds, all_ramps_conf, all_ramps_acc,
                             ramp_ids, latency_config, baseline)


def tune_threshold(ramp_ids, shadow_ramp_idx, data, acc_loss_budget, latency_calc_list, min_step_size=0.01, fixed_ramps_info = {}):
    """Tune the exit threshold for each ramp.

    Args:
        ramp_ids (list): list of ramp ids
        shadow_ramp_idx (int): idx of shadow ramp in ramp_ids
        data (dict): data
        acc_loss_budget (float): max accuracy loss (compared to the original model output) we can afford
        latency_calc_list (list): list (len = num_all_ramps + 1) of tuples.
            Index x: vanilla model latency before ramp x and the latency of ramp x.
            Last index: (vanilla model latency, None).
        min_step_size (float): minimum step sizes

    Returns:
        thresholds (list): list of exit thresholds
        latency_improvement (float): latency improvement
        exit_rate (numpy array): exit rate of each ramp
        acc (float): accuracy
    """
    assert ramp_ids != [], f"Empty ramp_ids!"
    if shadow_ramp_idx is not None:
        activate_ramp_ids = ramp_ids[:shadow_ramp_idx] + \
            ramp_ids[shadow_ramp_idx + 1:]
    else:
        activate_ramp_ids = ramp_ids

    thresholds, latency_improvement, exit_rate, acc = None, float(
        "-inf"), None, None
    
    min_step_size = 0.001
    

    for s in [0.01]:
    # for s in [0.01, 0.02, 0.04]:
        s = round(s, 4)
        cur_config, curr_latency_improvement, curr_exit_rates, curr_acc = \
            greedy_search_step(activate_ramp_ids, min_step_size, s, acc_loss_budget,
                               data, latency_calc_list, fixed_ramps_info)

        if curr_latency_improvement > latency_improvement:
            thresholds = cur_config
            latency_improvement = curr_latency_improvement
            exit_rate = curr_exit_rates
            acc = curr_acc

    # print("greedy search: ", ramp_ids, thresholds,
    #       latency_improvement, exit_rate, acc, flush=True)
    if shadow_ramp_idx is not None:
        thresholds.insert(shadow_ramp_idx, 0.0)

    return thresholds, latency_improvement, exit_rate, acc


def greedy_search_step(ramp_ids, min_step_size, step_size, acc_loss_budget, data, latency_calc_list, fixed_ramps_info = {}):
    """Perform greedy search.

    Args:
        ramp_ids (list): list of ramp ids
        min_step_size (float): minimum step sizes
        step_size (float): step sizes
        acc_loss_budget (float): max accuracy loss (compared to the original model output) we can afford
        data (dict): data
        latency_calc_list (list): list (len = num_all_ramps + 1) of tuples.
            Index x: vanilla model latency before ramp x and the latency of ramp x.
            Last index: (vanilla model latency, None).

    Returns:
        thresholds (list): list of exit thresholds
        latency_improvement (float): latency improvement
        exit_rate (list): exit rate
        acc (float): accuracy
    """
    latency_config, baseline = get_ramp_latencies(ramp_ids, latency_calc_list)
    step_sizes = [step_size if id not in fixed_ramps_info else min_step_size for id in ramp_ids]
    thresholds = [0.0 if id not in fixed_ramps_info else fixed_ramps_info[id] for id in ramp_ids]
    
    acc, latency_improvement, exit_rate = None, None, None

    acc, latency_improvement, exit_rate = \
        query_performance(
            thresholds, data['conf'], data['acc'], ramp_ids, latency_config, baseline)

    while True:
        next_exit_rate, positive_dirs = None, None
        next_direction, next_acc, next_latency_improvement = None, None, None
        next_direction, next_acc, next_latency_improvement, next_exit_rate, positive_dirs = \
            explore_direction(data, ramp_ids, thresholds, step_sizes, latency_config,
                              baseline, acc, latency_improvement, exit_rate, acc_loss_budget, fixed_ramps_info)

        if next_direction != None and thresholds[next_direction] <= 1:
            acc = next_acc
            latency_improvement = next_latency_improvement
            exit_rate = next_exit_rate
            thresholds[next_direction] =\
                round(thresholds[next_direction] +
                      step_sizes[next_direction], 4)
            step_sizes[next_direction] *= 2
            for i in positive_dirs:
                if i != next_direction:
                    step_sizes[i] *= 2
        else:
            flag = True
            for i in range(len(step_sizes)):
                if round(step_sizes[i], 4) <= min_step_size \
                        or thresholds[i] > 1:
                    continue
                else:
                    flag = False
                    step_sizes[i] /= 2
            if flag:
                break

    return thresholds, latency_improvement, exit_rate, acc


def explore_direction(data, ramp_ids, thresholds, step_sizes, latency_config,
                      baseline, curr_acc, curr_latency_improvement, curr_exit_rate,
                      acc_loss_budget, fixed_ramps_info=[None]):
    """Explore the direction of the next step.

    Args:
        data (dict): data
        ramp_ids (list): list of ramp ids
        thresholds (list): list of exit thresholds
        step_sizes (list): list of step sizes
        latency_config (list): list of latencies
        baseline (float): baseline latency
        curr_acc (float): current accuracy
        curr_latency_improvement (float): current latency improvement
        curr_exit_rate (list): current exit rate
        acc_loss_budget (float): max accuracy loss (compared to the original model output) we can afford

    Returns:
        best_direction (int): best direction
        res_acc (float): accuracy in best direction
        res_latency_improvement (float): latency improvement in best direction
        res_exit_rate (list): exit rate in best direction
        positive_dirs (list): list of directions that have positive improvement
    """
    best_direction = None
    best_score = float("inf")
    res_acc = None
    res_latency_improvement = None
    res_exit_rate = None
    equal_num = 0
    positive_dirs = []
    positive_dirs_data = []

    for direction in range(len(ramp_ids)):
        if ramp_ids[direction] in fixed_ramps_info:
            continue
        temp_config = copy.deepcopy(thresholds)
        temp_config[direction] = round(
            temp_config[direction] + step_sizes[direction], 4)

        temp_acc, temp_latency_improvement, temp_exit_rate = \
            query_performance(
                temp_config, data['conf'], data['acc'], ramp_ids, latency_config, baseline)

        if abs(1 - temp_acc) < acc_loss_budget:
            if temp_latency_improvement != curr_latency_improvement:
                score = abs(temp_acc - curr_acc) / \
                    abs(temp_latency_improvement - curr_latency_improvement)
                if score < best_score:
                    best_score = score
                    best_direction = direction
                    res_acc = temp_acc
                    res_exit_rate = temp_exit_rate
                    res_latency_improvement = temp_latency_improvement
            else:
                equal_num += 1

            if temp_latency_improvement == curr_latency_improvement or \
                    temp_acc == curr_acc:
                positive_dirs += [direction]
                positive_dirs_data += [[temp_acc,
                                        temp_latency_improvement, temp_exit_rate]]

    if equal_num == len(ramp_ids):
        return 0, curr_acc, curr_latency_improvement, curr_exit_rate, positive_dirs
    if not best_direction and len(positive_dirs) > 0:
        return positive_dirs[0], positive_dirs_data[0][0], \
            positive_dirs_data[0][1], positive_dirs_data[0][2], positive_dirs
    return best_direction, res_acc, \
        res_latency_improvement, res_exit_rate, positive_dirs


def earlyexit_infer_per_sample(output, target, ramp_ids, thresholds, total_num_ramps, queuing_delay, ramp_latencies, optimal=False, simulated_pickle=None):
    """
    Early exit inference per sample.

    Args:
        output (torch.Tensor): output tensor consisting of multiple exit outputs and final output
        target (torch.Tensor): model prediction
        ramp_ids (list): list of activated ramp ids
        thresholds (list): list of exit thresholds
        total_num_ramps (int): total possible exits including final exit
        queuing_delay (list of float): queuing delay for each data sample in the batch
        ramp_latencies (list of float): ramp latencies for each ramp including final exit
        optimal (bool): whether to use optimal early exit, say only exit if the prediction is correct

    Returns:
        batch_meta_data (dict): for historical data update}
        sample_latencies (list of float): latencies for each data sample (inference + queuing delay)
        sample_acc (list of bool): prediction correctness for each data sample
        sample_exit_points (list of int): exit ramp id for each data sample 
    """
    res_conf = [[] for _ in range(total_num_ramps)]
    res_acc = [[] for _ in range(total_num_ramps)]

    if simulated_pickle is None:
        this_batch_size = target.size(0)
        softmax = nn.Softmax(dim=1)
        # calculate confidence and accuracy for each ramp
        for exitnum in range(len(ramp_ids)):
            out = softmax(output[exitnum])
            out, inds = torch.max(out, dim=1)
            res_conf[ramp_ids[exitnum]] += out.cpu().tolist()
            res_acc[ramp_ids[exitnum]] += (inds == target).cpu().tolist()
        res_conf[-1] += output[-1].cpu().tolist()
        res_acc[-1] += [True for _ in range(this_batch_size)]
    else:
        this_batch_size = len(simulated_pickle["conf"][0])
        for ramp_id in ramp_ids:
            res_conf[ramp_id] = simulated_pickle["conf"][ramp_id]
            res_acc[ramp_id] = simulated_pickle["acc"][ramp_id]

    sample_latencies = []
    sample_acc = []
    sample_exit_points = []

    for i in range(this_batch_size):
        earlyexit = False
        for j in range(len(ramp_ids)):
            if not optimal:
                ramp_id = ramp_ids[j]
                if 1 - res_conf[ramp_id][i] < thresholds[j]:
                    sample_latencies += [(queuing_delay[i],
                                          ramp_latencies[j],)]
                    sample_acc += [res_acc[ramp_id][i]]
                    sample_exit_points += [ramp_id]
                    earlyexit = True
                    break
            else:
                ramp_id = ramp_ids[j]
                if res_acc[ramp_id][i]:
                # if 1 - res_conf[ramp_id][i] < thresholds[j] and res_acc[ramp_id][i]:
                    sample_latencies += [(queuing_delay[i],
                                          ramp_latencies[j],)]
                    sample_acc += [res_acc[ramp_id][i]]
                    sample_exit_points += [ramp_id]
                    earlyexit = True
                    break
        if not earlyexit:
            sample_latencies += [(queuing_delay[i], ramp_latencies[-1],)]
            sample_exit_points += [total_num_ramps - 1]
            sample_acc += [True]
    batch_meta_data = {"conf": res_conf, "acc": res_acc}
    return batch_meta_data, sample_latencies, sample_acc, sample_exit_points


def earlyexit_inference(output, target, ramp_ids, thresholds, total_num_ramps):
    """
    Early exit inference.

    Args:
        output (torch.Tensor): output tensor consisting of multiple exit outputs and final output
        target (torch.Tensor): model prediction
        ramp_ids (list): list of activated ramp ids
        thresholds (list): list of exit thresholds
        total_num_ramps (int): total possible exits including final exit

    Returns:
        batch_meta_data (dict): for historical data update}
        exit_rate (numpy array): exit rate for each ramp
    """
    this_batch_size = target.size(0)

    softmax = nn.Softmax(dim=1)

    num_exits = len(ramp_ids) + 1

    res_conf = [[] for _ in range(total_num_ramps)]
    res_acc = [[] for _ in range(total_num_ramps)]

    # calculate confidence and accuracy for each ramp
    for exitnum in range(len(ramp_ids)):
        out = softmax(output[exitnum])
        out, inds = torch.max(out, dim=1)
        res_conf[ramp_ids[exitnum]] += out.cpu().tolist()
        res_acc[ramp_ids[exitnum]] += (inds == target).cpu().tolist()

    # calculate confidence and accuracy for final exit
    res_conf[-1] += output[-1].cpu().tolist()
    res_acc[-1] += [True for _ in range(this_batch_size)]

    exit_counter = [0 for i in range(len(ramp_ids) + 1)]

    for batch_index in range(this_batch_size):
        earlyexit_taken = False
        # take the exit using CrossEntropyLoss as confidence measure (lower is more confident)
        for exitnum in range(num_exits - 1):
            if 1 - res_conf[ramp_ids[exitnum]][batch_index] < thresholds[exitnum]:
                # take the results from early exit since lower than threshold
                exit_counter[exitnum] += 1
                earlyexit_taken = True
                break
        if not earlyexit_taken:
            exit_counter[-1] += 1

    exit_rate = np.array([(n+0.0)/this_batch_size for n in exit_counter])

    batch_meta_data = {"conf": res_conf, "acc": res_acc}

    return batch_meta_data, exit_rate


def get_queuing_delay(request_rate, batch_size):
    """
    Get queuing delay.

    Args:
        request_rate (float): requests per second

    Returns:
        queuing_delay (list): list of queuing delays for each sample in the batch (ms)
    """
    # time gap between two requests (ms)
    gap = 1.0 / request_rate * 1000
    queuing_delay = []
    for i in range(batch_size):
        queuing_delay.append(gap * i)
    return queuing_delay[::-1]


def calculate_batch_size(request_rate, model_inference_time, slo):
    """
    Calculate batch size based on request rate and latency.

    Args:
        request_rate (float): requests per second
        model_inference_time (float): model inference time (ms)
        slo (float): slo for request (ms)

    Returns:
        batch_size (int): batch size that can satisfy the slo 
        based on the given request rate
    """
    batch_size = int((slo - model_inference_time) * request_rate / 1000.0)
    return batch_size


def get_batch_perf(sample_latencies, sample_acc, sample_exit_points, vanilla_latency, ramp_ids, total_num_ramps):
    """
    Query batch performance.

    Args:
        sample_latencies (list): list of tuples of (queuing_delay, ramp_latency)
        sample_acc (list): list of bool of accuracy
        sample_exit_points (list): list of int of exit points
        vanilla_latency (float): latency of vanilla model
        ramp_ids (list): list of activated ramp ids
        total_num_ramps (int): total number of ramps

    Returns:
        acc (float): batch accuracy
        latency_improvement (float): batch latency improvement
        exit_rate (list): list of exit rate for each ramp
    """

    acc = sum(sample_acc) / float(len(sample_acc))
    curr_serving_latencies = [l[1] for l in sample_latencies]
    latency_improvement = 100 * \
        (vanilla_latency - sum(curr_serving_latencies) /
         len(curr_serving_latencies)) / vanilla_latency

    num_exit = [0 for _ in range(len(ramp_ids) + 1)]
    for i, exit_point in enumerate(sample_exit_points):
        for j, ramp_id in enumerate(ramp_ids):
            if exit_point == ramp_id:
                num_exit[j] += 1
                break
            elif exit_point == total_num_ramps - 1:
                num_exit[-1] += 1
                break
    exit_rate = np.array(
        [(n+0.0)/len(sample_exit_points) for n in num_exit])
    return acc, latency_improvement, exit_rate


def get_overall_exit_info(all_exit_ramp, all_accuracies):
    """
    Get overall exit info.

    Args:
        all_exit_ramp (list): list of exit ramp ids for each sample
        all_accuracies (list): list of bool of accuracy for each sample

    Returns:
        overall_exit_info (dict): overall exit info for each ramp
        overall_exit_acc (dict): overall exit accuracy for each ramp
    """
    overall_exit_info = {}

    overall_exit_acc = {}

    for ramp_id in all_exit_ramp:
        if ramp_id not in overall_exit_info:
            overall_exit_info[ramp_id] = 1
        else:
            overall_exit_info[ramp_id] += 1

    for key, val in overall_exit_info.items():
        overall_exit_info[key] = val / float(len(all_exit_ramp))

    for exit_point, correct in zip(all_exit_ramp, all_accuracies):
        if exit_point not in overall_exit_acc:
            overall_exit_acc[exit_point] = []
        overall_exit_acc[exit_point].append(int(correct))

    for key, val in overall_exit_acc.items():
        overall_exit_acc[key] = sum(val) / float(len(val))

    return dict(sorted(overall_exit_info.items())), dict(sorted(overall_exit_acc.items()))


def get_exitable_ramps(
    entropy_dict: dict,
    ramp_ids: list,
    thresholds: list,
):
    """For each historcal data, returns a list of possible ramps
    it can exit from.

    Args:
        entropy_dict (dict): format similar to self._historical_data. Note 
            that this dict should record the entropies at all possible ramps.
        ramp_ids (list): list of int, 0-indexed ramp IDs
        thresholds (list): list of float of associated exit thresholds

    Returns:
        all_exitable_ramps (list): list of lists. index x: all
            ramp IDs historical data x can exit from.
    """
    num_samples = len(entropy_dict["conf"][ramp_ids[0]])
    all_exitable_ramps = [[] for _ in range(num_samples)]
    for sample_id in range(num_samples):
        # confidence score (higher: easier) at all ramps
        all_conf = [entropy_dict['conf'][ramp_id][sample_id]
                    for ramp_id in ramp_ids]
        all_exitable_ramps[sample_id] = [
            ramp_ids[i] for i, (conf, threshold) in enumerate(zip(all_conf, thresholds))
            if (1 - conf) <= threshold
        ]
    return all_exitable_ramps


def get_ramp_utilities(
    ramp_ids: list,
    thresholds: list,
    entropy_dict: dict,
    exit_rate: np.ndarray,
    latency_calc_list: list,
    batch_size: int,
):
    """Returns the utilities of ramps in the current configuration.

    Args:
        ramp_ids (list): list of ramp ids
        thresholds (list): list of exit thresholds
        entropy_dict (dict): format similar to self._historical_data. Note 
            that this dict should record the entropies at all possible ramps.
        exit_rate (np.ndarray): exit rate at all ramps and the final model, 
            sums up to 1
        latency_calc_list (list): list (len = num_all_ramps + 1) of tuples.
            Index x: vanilla model latency before ramp x and the latency of ramp x.
            Last index: (vanilla model latency, None).
        batch_size (int): batch size            

    Returns:
        utilities (list): index x: utility of ramp at index x.
    """
    assert len(exit_rate) == len(ramp_ids) + 1, \
        f"Metadata length mismatch: len(exit_rate) {len(exit_rate)}, but has {len(ramp_ids)} active ramps!"
    utilities = []  # index x: numerical value of utility of ramp at index x
    remaining_rate = get_remaining_rate(exit_rate)
    latencies, _ = get_ramp_latencies(ramp_ids, latency_calc_list)
    all_exitable_ramps = get_exitable_ramps(
        entropy_dict, ramp_ids, thresholds)

    for ramp_index, ramp_id in enumerate(ramp_ids):
        # print("="*30)
        samples_passed_through_ramp = 1.0 if ramp_index == 0 else remaining_rate[
            ramp_index - 1]
        # samples_remaining_after_ramp = remaining_rate[ramp_index]
        ramp_overhead = latency_calc_list[ramp_id][1]
        # if samples_remaining_after_ramp == 1:
        #     # totally garbage ramp, mark as infinite overhead to be pruned in the first round
        #     overhead = float("inf")
        # else:  # some samples exited, not totally garbage ramp
        #     overhead = samples_passed_through_ramp * ramp_overhead
        overhead = samples_passed_through_ramp * ramp_overhead
        # print(f"overhead: samples_passed_through_ramp {samples_passed_through_ramp} * ramp_overhead {ramp_overhead}")

        saving = 0
        num_samples_exited_at_ramp = 0
        for exitable_ramps in all_exitable_ramps:  # for each sample:
            # for sample_id, exitable_ramps in enumerate(all_exitable_ramps):
            if exitable_ramps == []:  # sample doesn't take any exits
                continue
            if exitable_ramps[0] == ramp_id:  # current sample exits here
                num_samples_exited_at_ramp += 1
                if len(exitable_ramps) == 1:
                    next_exitable_ramp_index = -1
                else:
                    all_actual_exitable_ramps = [
                        x for x in exitable_ramps if x in ramp_ids]
                    next_exitable_ramp_id = all_actual_exitable_ramps[1]
                    next_exitable_ramp_index = ramp_ids.index(
                        next_exitable_ramp_id)
                saving += (
                    latencies[next_exitable_ramp_index] -
                    latencies[ramp_index]
                )
        # normalize savings
        total_num_samples = len(all_exitable_ramps)
        if num_samples_exited_at_ramp == 0:  # no samples exits here
            saving = 0
        else:
            # normalize from absolute values to 0-1
            saving /= total_num_samples
        utility = saving - overhead
        utilities.append(utility)
        # print(f"ramp {ramp_id}, utility {utility} = saving {saving} - overhead {overhead}")
    return utilities


def ramp_addition(
    entropy_dict: dict,
    latency_calc_list: list,
    num_ramp_budget: int = float('inf'),
    acc_loss_budget: float = 0.01,
    curr_ramp_ids: list = [],
):
    """Given an entropy pickle that records the entropies of some dataset
    at all possible ramps, iteratively activates ramps to obtain a
    configuration with maximum latency savings under an accuracy loss budget

    Args:
        entropy_dict (dict): format similar to self._historical_data. Note 
            that this dict should record the entropies at all possible ramps.
        latency_calc_list (list): list (len = num_all_ramps + 1) of tuples.
            Index x: vanilla model latency before ramp x and the latency of ramp x.
            Last index: (vanilla model latency, None).
        num_ramp_budget (int): max number of ramps to insert into the model
        acc_loss_budget (float): max acc loss affordable
        curr_ramp_ids (list): list of ramp IDs to start with. Defaults to 
            an empty ramp configuration.

    Returns:
        ramp_ids (list): list of int, 0-indexed ramp IDs
        thresholds (list): list of float of associated exit thresholds
        latency_savings (float): percentage of latency savings
        acc (float): accuracy of the resulting config from 0-1
        exit_rate (np.ndarray): exit rate at all ramps and the final model, 
            sums up to 1
        (ramp_efficacy_order, ramp_efficacies) (list, list): list of ramp IDs, ordered by
            individual efficacy descending; latency savings of each individual ramp
    """
    total_num_ramps = len(entropy_dict["conf"]) - 1
    stats_every_epoch = []  # stats in every epoch of ramp addition
    # index x: latency improvement of ramp x if only 1 ramp is activated
    ramp_efficacies = [float('-inf') for _ in range(total_num_ramps)]

    while len(curr_ramp_ids) < num_ramp_budget:
        # print("{:=^50}".format(f"Ramp addition epoch {len(curr_ramp_ids)}"))
        # print(f"curr_ramp_ids {curr_ramp_ids}")
        candidate_ramps = [x for x in list(
            range(total_num_ramps)) if x not in curr_ramp_ids]

        # keep track of current best ramp and its max latency savings
        max_savings, best_ramp_id, best_thresholds, best_acc, best_exit_rate = float(
            "-inf"), None, None, None, None
        # for the remaining inactive ramps, add each one...
        # print('='*30)
        for candidate_ramp_id in candidate_ramps:
            candidate_ramp_ids = sorted(
                curr_ramp_ids + [candidate_ramp_id])
            thresholds, latency_improvement, exit_rate, acc \
                = tune_threshold(candidate_ramp_ids, None, entropy_dict, acc_loss_budget=acc_loss_budget, latency_calc_list=latency_calc_list)

            # #######calculate utility of each ramp in 1st round#######
            # if len(curr_ramp_ids) == 0:  # first round
            #     utilities = get_ramp_utilities(
            #         candidate_ramp_ids,
            #         thresholds,
            #         entropy_dict,
            #         exit_rate,
            #         latency_calc_list,
            #         batch_size=16,
            #     )
            #     print(f"Utility of ramp {candidate_ramp_id}: {utilities[0]}")
            # #######calculate utility of each ramp in 1st round#######

            if acc < 1 - acc_loss_budget:
                # print(f"candidate_ramp_ids {candidate_ramp_ids}, accuracy {acc}, continuing")
                continue
            else:
                ramp_efficacies[candidate_ramp_id] = latency_improvement
            if latency_improvement > max_savings:
                max_savings = latency_improvement
                best_ramp_id = candidate_ramp_id
                best_thresholds = thresholds
                best_acc = acc
                best_exit_rate = exit_rate
        if best_ramp_id == None:
            # print(f"No more ramps can be added without overflowing the accuracy loss budget, stopping")
            break
        # print(f"Among all candidates, adding ramp {best_ramp_id} brings a max saving of {max_savings}")
        if len(curr_ramp_ids) > 0 and \
                (stats_every_epoch != [] and stats_every_epoch[-1][0] > max_savings):
            # print(f"Latency savings have reached maximum in last epoch, stopping")
            break
        curr_ramp_ids = sorted(curr_ramp_ids + [best_ramp_id])
        stats_every_epoch.append(
            (max_savings, curr_ramp_ids, best_thresholds, best_acc, best_exit_rate))
        # print(f"Current ramps: {curr_ramp_ids}, thresholds {best_thresholds}, exit_rate {best_exit_rate}")

    # print(f"Ramp addition result: {curr_ramp_ids}, thresholds {stats_every_epoch[-1][2]}, latency savings {stats_every_epoch[-1][0]}, acc {stats_every_epoch[-1][3]}, exit rate {stats_every_epoch[-1][4]}")
    ramp_ids = curr_ramp_ids
    thresholds, latency_savings, acc, exit_rate = stats_every_epoch[-1][
        2], stats_every_epoch[-1][0], stats_every_epoch[-1][3], stats_every_epoch[-1][4]

    # order the ramps by efficacy descending
    ramp_efficacy_order = list((-np.array(ramp_efficacies)).argsort())

    return ramp_ids, thresholds, latency_savings, acc, exit_rate, (ramp_efficacy_order, ramp_efficacies)


def ramp_pruning(
    entropy_dict: dict,
    latency_calc_list: list,
    batch_size: int,
    ramp_ids: list = None,
    acc_loss_budget: float = 0.01,
    prune_threshold: float = 0.05,
):
    """Given an entropy pickle that records the entropies of some dataset
    at all possible ramps, activate all ramps, tune thresholds on all ramps,
    and iteratively prunes ramps with the lowest negative utility.

    Args:
        entropy_dict (dict): format similar to self._historical_data
        latency_calc_list (list): list (len = num_all_ramps + 1) of tuples.
            Index x: vanilla model latency before ramp x and the latency of ramp x.
            Last index: (vanilla model latency, None).
        batch_size (int): batch size
        num_ramp_budget (int): max number of ramps to insert into the model
        acc_loss_budget (float): max acc loss affordable
        prune_threshold (float): utility threshold for pruning. 
            our system is not sensitive to this hyperparameter, 
            but we default it to 0.1 instead of 0 to prevent noise.

    Returns:
        ramp_ids (list): list of int, 0-indexed ramp IDs
        thresholds (list): list of float of associated exit thresholds
        latency_savings (float): percentage of latency savings
        acc (float): accuracy of the resulting config from 0-1
        exit_rate (np.ndarray): exit rate at all ramps and the final model, 
            sums up to 1
    """
    if ramp_ids is None:
        total_num_ramps = len(entropy_dict["conf"]) - 1
        ramp_ids = list(range(total_num_ramps))

    epoch = 0
    stats_every_epoch = []

    while True:
        if ramp_ids == []:
            # all ramps have been pruned
            thresholds, latency_improvement, acc = [], 0.0, 1.0
            exit_rate = np.array([1.0])
            break
        # activate all and run threshold tuning
        thresholds, latency_improvement, exit_rate, acc \
            = tune_threshold(ramp_ids, None, entropy_dict, acc_loss_budget=acc_loss_budget, latency_calc_list=latency_calc_list)

        stats_every_epoch.append((
            latency_improvement, ramp_ids, thresholds, acc, exit_rate,
        ))

        # calculate utilities of the current ramps & thresholds on this subdataset
        utilities = get_ramp_utilities(
            ramp_ids, thresholds, entropy_dict, exit_rate, latency_calc_list, batch_size)
        # if any negatives, pop ramp with lowest utility
        if epoch == 0:  # prune totally-garbage ramps
            ramp_index_to_prune = list(np.where(exit_rate[:-1] == 0)[0])
        else:  # prune ramp with lowest, negative utility
            if min(utilities) > prune_threshold:
                # print(f"All ramps have positive utility, terminating pruning process")
                break
            ramp_index_to_prune = [utilities.index(min(utilities))]
        ramp_ids_to_prune = [ramp_ids[x] for x in ramp_index_to_prune]
        ramp_ids = [x for x in ramp_ids if x not in ramp_ids_to_prune]
        epoch += 1
    # print(stats_every_epoch)  # savings might not monotonically decrease
    return ramp_ids, thresholds, latency_improvement, acc, exit_rate


def ramp_pruning_garbage_only(
    entropy_dict: dict,
    latency_calc_list: list,
    acc_loss_budget=0.01,
):
    """Given an entropy pickle that records the entropies of some dataset
    at all possible ramps, activate all ramps, tune thresholds on all ramps,
    and only deactivate the ramps that are totally garbage. Then, retune
    thresholds.

    Args:
        entropy_dict (dict): format similar to self._historical_data
        latency_calc_list (list): list (len = num_all_ramps + 1) of tuples.
            Index x: vanilla model latency before ramp x and the latency of ramp x.
            Last index: (vanilla model latency, None).
        acc_loss_budget (float): max acc loss affordable

    Returns:
        ramp_ids (list): list of int, 0-indexed ramp IDs
        thresholds (list): list of float of associated exit thresholds
        latency_savings (float): percentage of latency savings
        acc (float): accuracy of the resulting config from 0-1
        exit_rate (np.ndarray): exit rate at all ramps and the final model, 
            sums up to 1
    """
    total_num_ramps = len(entropy_dict["conf"]) - 1
    all_ramp_ids = list(range(total_num_ramps))

    # activate all and run threshold tuning
    thresholds, latency_improvement, exit_rate, acc \
        = tune_threshold(all_ramp_ids, None, entropy_dict, acc_loss_budget=acc_loss_budget, latency_calc_list=latency_calc_list)
    garbage_ramp_ids = list(np.where(exit_rate == 0)[0])
    # print(f"Ramp IDs that are totally garbage and beyond saving: {garbage_ramp_ids}")
    all_ramp_ids = [x for x in all_ramp_ids if x not in garbage_ramp_ids]
    thresholds, latency_improvement, exit_rate, acc \
        = tune_threshold(all_ramp_ids, None, entropy_dict, acc_loss_budget=acc_loss_budget, latency_calc_list=latency_calc_list)
    # print(f"Ramp pruning (garbage only) result: {all_ramp_ids}, thresholds {thresholds}, latency savings {latency_improvement}, acc {acc}, exit rate {exit_rate}")
    return all_ramp_ids, thresholds, latency_improvement, acc, exit_rate


def get_ramp_scores(ramp_ids, exit_rates, latency_config, latency_calc_list):
    """
    Calculate the utility score of a ramp, which is the expected latency improvement
    considering tail latency

    Args:
        ramp_ids (list): list of int, 0-indexed ramp IDs
        exit_rates (np.ndarray): exit rate at all ramps and the final model, 
            sums up to 1
        latency_config (np.ndarray): latency of the config at all ramps and the final model
        latency_calc_list (list): list of tuples. Index x: vanilla model latency before ramp x and the latency of ramp x.

    Return: 
        utility_scores (list): list of utility score of the ramp 
    """

    utility_scores = []

    baseline_latency = latency_calc_list[-1][0]

    for i, ramp_id in enumerate(ramp_ids):
       
        # savings = exit_rates[i] * (baseline_latency - latency_config[i])  # distance saved = end_of_model - curr_ramp
        savings = exit_rates[i] * (latency_config[i + 1] - latency_config[i])  # distance saved = next_ramp - curr_ramp
        overhead = sum(exit_rates[i+1:]) * latency_calc_list[ramp_id][1]
        utility = savings - overhead
        tail_latency_overhead = latency_calc_list[ramp_id][1]
        score = utility / tail_latency_overhead
        utility_scores.append(score)

    return utility_scores


def get_ramp_utility(ramp_ids, exit_rates, latency_config, latency_calc_list):
    """
    Calculate the utility score of a ramp, which is the expected latency improvement
    considering tail latency

    Args:
        ramp_ids (list): list of int, 0-indexed ramp IDs
        exit_rates (np.ndarray): exit rate at all ramps and the final model, 
            sums up to 1
        latency_config (np.ndarray): latency of the config at all ramps and the final model
        latency_calc_list (list): list of tuples. Index x: vanilla model latency before ramp x and the latency of ramp x.

    Return: 
        utility_scores (list): list of utility score of the ramp 
    """

    utility_scores = []

    baseline_latency = latency_calc_list[-1][0]

    for i, ramp_id in enumerate(ramp_ids):
       
        # savings = exit_rates[i] * (baseline_latency - latency_config[i])  # distance saved = end_of_model - curr_ramp
        savings = exit_rates[i] * (latency_config[i + 1] - latency_config[i])  # distance saved = next_ramp - curr_ramp
        overhead = sum(exit_rates[i+1:]) * latency_calc_list[ramp_id][1]
        utility = savings - overhead
        utility_scores.append(utility)

    return utility_scores


def ramp_addition_tail_latency(
    entropy_dict: dict,
    latency_calc_list: list,
    num_ramp_budget: int = float('inf'),
    tail_latency_budget: float = 0.02,
    acc_loss_budget: float = 0.01,
    curr_ramp_ids: list = [],
):
    """Given an entropy pickle that records the entropies of some dataset
    at all possible ramps, iteratively activates ramps to obtain a
    configuration with maximum latency savings per tail latency under an accuracy loss budget

    Args:
        entropy_dict (dict): format similar to self._historical_data. Note 
            that this dict should record the entropies at all possible ramps.
        latency_calc_list (list): list (len = num_all_ramps + 1) of tuples.
            Index x: vanilla model latency before ramp x and the latency of ramp x.
            Last index: (vanilla model latency, None).
        num_ramp_budget (int): max number of ramps to insert into the model
        acc_loss_budget (float): max acc loss affordable
        curr_ramp_ids (list): list of ramp IDs to start with. Defaults to 
            an empty ramp configuration.

    Returns:
        ramp_ids (list): list of int, 0-indexed ramp IDs
        thresholds (list): list of float of associated exit thresholds
        latency_savings (float): percentage of latency savings
        acc (float): accuracy of the resulting config from 0-1
        exit_rate (np.ndarray): exit rate at all ramps and the final model, 
            sums up to 1
        (ramp_efficacy_order, ramp_efficacies) (list, list): list of ramp IDs, ordered by
            individual efficacy descending; latency savings of each individual ramp
    """
    
    total_num_ramps = len(entropy_dict["conf"]) - 1
    stats_every_epoch = []  # stats in every epoch of ramp addition
    # index x: latency improvement of ramp x if only 1 ramp is activated
    ramp_efficacies = [float('-inf') for _ in range(total_num_ramps)]
    # print(f"num_ramp_budget: {num_ramp_budget}")
    while len(curr_ramp_ids) < num_ramp_budget:
        candidate_ramps = [x for x in list(
            range(total_num_ramps)) if x not in curr_ramp_ids]

        # keep track of current best ramp and its max latency savings
        max_savings, best_ramp_id, best_thresholds, best_acc, best_exit_rate = float(
            "-inf"), None, None, None, None

        max_score, best_tail_latency = 0, None

        # for the remaining inactive ramps, add each one...
        # print('='*30)
        for candidate_ramp_id in candidate_ramps:
            candidate_ramp_ids = sorted(
                curr_ramp_ids + [candidate_ramp_id])
            thresholds, latency_improvement, exit_rate, acc \
                = tune_threshold(candidate_ramp_ids, None, entropy_dict, acc_loss_budget=acc_loss_budget, latency_calc_list=latency_calc_list)

            latency_config, baseline_latency = get_ramp_latencies(
                candidate_ramp_ids, latency_calc_list)
            ramp_scores = get_ramp_scores(
                candidate_ramp_ids, exit_rate, latency_config, latency_calc_list)
            # latency_config, baseline = get_ramp_latencies(candidate_ramp_ids, latency_calc_list)
            # idx = candidate_ramp_ids.index(candidate_ramp_id)
            # savings = exit_rate[idx] * (baseline - latency_config[idx])
            # tail_latency_overhead = latency_calc_list[candidate_ramp_id][1]
            # score = savings / tail_latency_overhead
            idx = candidate_ramp_ids.index(candidate_ramp_id)
            score = ramp_scores[idx]

            if acc < 1 - acc_loss_budget:
                continue
            else:
                ramp_efficacies[candidate_ramp_id] = latency_improvement

            if score > max_score:
                if (latency_config[-1] - baseline_latency) / baseline_latency < tail_latency_budget:
                    max_score = score
                    # best_tail_latency = tail_latency_overhead
                    max_savings = latency_improvement
                    best_ramp_id = candidate_ramp_id
                    best_thresholds = thresholds
                    best_acc = acc
                    best_exit_rate = exit_rate
                else:
                    print(f"Latency budget exceeded! Latency overhead of ramp {candidate_ramp_id} is {(latency_config[-1] - baseline_latency) / baseline_latency}.")
        if best_ramp_id == None:
            break

        curr_ramp_ids = sorted(curr_ramp_ids + [best_ramp_id])
        stats_every_epoch.append(
            (max_savings, curr_ramp_ids, best_thresholds, best_acc, best_exit_rate))

    if stats_every_epoch == []:
        return None, None, None, None, None, (None, None)
        # raise RuntimeError(
        #     f"No ramps can be added within the tail latency budget of {tail_latency_budget}!")

    ramp_ids = curr_ramp_ids
    thresholds, latency_savings, acc, exit_rate = stats_every_epoch[-1][
        2], stats_every_epoch[-1][0], stats_every_epoch[-1][3], stats_every_epoch[-1][4]

    # order the ramps by efficacy descending
    ramp_efficacy_order = list((-np.array(ramp_efficacies)).argsort())
    return ramp_ids, thresholds, latency_savings, acc, exit_rate, (ramp_efficacy_order, ramp_efficacies)


def get_optimal_exit_rate(ramp_ids, entropy_dict):
    num_samples = len(entropy_dict['acc'][ramp_ids[0]])
    exit_rate = {-1: 0}  # final model output
    for i in ramp_ids:
        exit_rate[i] = 0
    for i in range(num_samples):
        exited = False
        for id in ramp_ids:
            if entropy_dict['acc'][id][i]:  # true
                exit_rate[id] += 1
                exited = True
                break
        if not exited:
            exit_rate[-1] += 1
    exit_rate_list = [exit_rate[i] for i in ramp_ids] + [exit_rate[-1]]
    exit_rate_list = np.array([x / sum(exit_rate_list)
                              for x in exit_rate_list])
    return exit_rate_list


def ramp_arch_addition_tail_latency(
    entropy_dict_list: dict,
    latency_calc_list: list,
    num_ramp_budget=float('inf'),
    tail_latency_budget=0.01,
    acc_loss_budget=0.01
):
    """Given an entropy pickle that records the entropies of some dataset
    at all possible ramps, iteratively activates ramps to obtain a
    configuration with maximum latency savings per tail latency under an accuracy loss budget
    also considers the architecture of the ramp

    Args:
        entropy_dict_list (list): list of dict, whose format similar to self._historical_data. Note 
            that this dict should record the entropies at all possible ramps.
        latency_calc_list (list of list): each entry is a list (len = num_all_ramps + 1) of tuples.
            Index x: vanilla model latency before ramp x and the latency of ramp x.
            Last index: (vanilla model latency, None).
        num_ramp_budget (int): max number of ramps to insert into the model
        acc_loss_budget (float): max acc loss affordable

    Returns:
        ramp_ids (list): list of int, 0-indexed ramp IDs
        thresholds (list): list of float of associated exit thresholds
        latency_savings (float): percentage of latency savings
        acc (float): accuracy of the resulting config from 0-1
        exit_rate (np.ndarray): exit rate at all ramps and the final model, 
            sums up to 1
        ramp_arch_map (dict): map from ramp id to arch id
    """
    curr_ramp_ids = []  # start from empty ramp configuration
    total_num_ramps = len(entropy_dict_list[0]["conf"]) - 1
    stats_every_epoch = []  # stats in every epoch of ramp addition
    total_num_archs = len(entropy_dict_list)
    ramp_arch_map = {}

    while len(curr_ramp_ids) < num_ramp_budget:
        candidate_ramps = [x for x in list(
            range(total_num_ramps)) if x not in curr_ramp_ids]

        # keep track of current best ramp and its max latency savings
        max_savings, best_ramp_id, best_arch, best_thresholds, best_acc, best_exit_rate = float(
            "-inf"), None, None, None, None, None

        max_score, best_tail_latency = float("-inf"), None

        # for the remaining inactive ramps, add each one...
        # print('='*30)
        for candidate_ramp_id in candidate_ramps:
            candidate_ramp_ids = sorted(
                curr_ramp_ids + [candidate_ramp_id])

            for i in range(total_num_archs):
                thresholds, latency_improvement, exit_rate, acc \
                    = tune_threshold(candidate_ramp_ids, None, entropy_dict_list[i], acc_loss_budget=acc_loss_budget, latency_calc_list=latency_calc_list[i])

                latency_config, baseline_latency = get_ramp_latencies(
                    candidate_ramp_ids, latency_calc_list[i])
                ramp_scores = get_ramp_scores(
                    candidate_ramp_ids, exit_rate, latency_config, latency_calc_list[i])
                idx = candidate_ramp_ids.index(candidate_ramp_id)
                score = ramp_scores[idx]

                if acc < 1 - acc_loss_budget:
                    continue

                if score > max_score and (latency_config[-1] - baseline_latency) / baseline_latency < tail_latency_budget:
                    max_score = score
                    # best_tail_latency = tail_latency_overhead
                    max_savings = latency_improvement
                    best_ramp_id = candidate_ramp_id
                    best_thresholds = thresholds
                    best_acc = acc
                    best_exit_rate = exit_rate
                    best_arch = i

        if best_ramp_id == None:
            break

        curr_ramp_ids = sorted(curr_ramp_ids + [best_ramp_id])
        stats_every_epoch.append(
            (max_savings, curr_ramp_ids, best_thresholds, best_acc, best_exit_rate))
        ramp_arch_map[best_ramp_id] = best_arch

    ramp_ids = curr_ramp_ids
    thresholds, latency_savings, acc, exit_rate = stats_every_epoch[-1][
        2], stats_every_epoch[-1][0], stats_every_epoch[-1][3], stats_every_epoch[-1][4]
    return ramp_ids, thresholds, latency_savings, acc, exit_rate, ramp_arch_map




def get_optimal_exitable_ramps(entropy_dict, ramp_ids, total_num_ramps):
    """
    get the optimal exitable ramps for each data sample

    Args:
        entropy_dict (dict): entropy dict for a single data sample
        ramp_ids (list): list of ramp ids
        total_num_ramps (int): total number of ramps
    Returns:
        optimal_exitable_ramps (list): list of optimal exitable ramps for each data sample
    """

    num_samples = len(entropy_dict["conf"][ramp_ids[0]])
    optimal_exitable_ramps = [[] for _ in range(num_samples)]
    for sample_id in range(num_samples):
        exitable = False
        for ramp_id in ramp_ids:
            if entropy_dict["acc"][ramp_id][sample_id]:
                optimal_exitable_ramps[sample_id].append(ramp_id)
                exitable = True
        if not exitable:
            optimal_exitable_ramps[sample_id].append(total_num_ramps)

    return optimal_exitable_ramps


def get_shadow_ramp_order(optimal_exitable_ramps, ramp_ids, total_num_ramps, latency_calc_list, ramp_acc):
    """
    get the shadow ramp order based on conditional probabilities

    Args:
        optimal_exitable_ramps (list): list of optimal exitable ramps for each data sample
        ramp_ids (list): list of ramp ids
        total_num_ramps (int): total number of ramps

    Returns:
        shadow_ramp_order (list): list of shadow ramp order
    """

    candidate_ramps = [x for x in list(
            range(total_num_ramps - 1)) if x not in ramp_ids]

    print("candidate_ramps", candidate_ramps)

    print("ramp_acc", ramp_acc)

    ramp_efficacies = [float('-inf') for _ in range(total_num_ramps - 1)]
    
    for i, candidate_ramp_id in enumerate(candidate_ramps):

        candidate_ramp_ids = sorted(
            ramp_ids + [candidate_ramp_id])

        exit_rates = [0 for _ in range(len(candidate_ramp_ids))]

        for j in range(len(candidate_ramp_ids)):
            query_set = set(candidate_ramp_ids[:j])
            divident = 0.0
            divisor = 0.0
            for exit_ramps in optimal_exitable_ramps:
                if len(query_set) == 0:
                    # first ramp
                    if candidate_ramp_ids[j] in exit_ramps:
                        divident += 1
                    divisor += 1
                else:
                    if not query_set.intersection(set(exit_ramps)) and candidate_ramp_ids[j] in exit_ramps:
                        divident += 1
                    divisor += 1

            exit_rates[j] = divident / divisor
    
        exit_rates.append(1 - sum(exit_rates))
        exit_rates = np.array(exit_rates)
        print(candidate_ramp_ids)
        print(exit_rates)

        latency_config, baseline_latency = get_ramp_latencies(
                candidate_ramp_ids, latency_calc_list)
        latency_improvement = (baseline_latency - sum(exit_rates * latency_config)) / baseline_latency * 100

        print(f"candidate_ramp_id {candidate_ramp_id}: {latency_improvement / latency_calc_list[candidate_ramp_id][1] / ramp_acc[candidate_ramp_id]}")

def find_universal_threshold(
    data,
    ramp_ids: list,
    latency_calc_list: list,
):
    BATCH_SIZE = 16
    ACC_BUDGET = 0.01
    
    threshold = 0.0
    step_size = 0.005
    latency_config, baseline = get_ramp_latencies(ramp_ids, latency_calc_list[BATCH_SIZE])
    while True:
        acc, latency_improvement, exit_rate = query_performance(
            config=[threshold + step_size] * len(ramp_ids),
            all_ramps_conf=data["conf"], 
            all_ramps_acc=data["acc"], 
            ramp_ids=ramp_ids, 
            latency_config=latency_config, 
            baseline=baseline,
        )
    
        if acc > 1 - ACC_BUDGET:  # accuracy not violated, increase threshold
            threshold += step_size
        else:
            print(f"Threshold {threshold + step_size} will violate acc ({acc}), breaking at {threshold}")
            return threshold
