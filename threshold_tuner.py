import os, sys
import time
import pickle
import argparse
import copy
import numpy as np

from matplotlib import cm
from matplotlib.ticker import LinearLocator, FixedLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import itertools
from itertools import repeat
import multiprocessing
from utils import *
# from profiling.profiler import *


class ThresholdTuner (object):

    def __init__(self, args):
        self.args = args

    def get_chunks(self, iterable, chunks=1):
        lst = list(iterable)
        return [lst[i::chunks] for i in range(chunks)]

    def index_to_float(self, config, step_size):
        return [i*step_size for i in config]
        
    def generate_ramp_configs(self, ramp_ids, step_size, grid_size):
        return [[round(0 + i*step_size, 4) for i in range(grid_size)] for _ in ramp_ids]

    
    def emulate_inference(self, configs, pickle_dict, ramp_ids, latency_config, baseline):

        best_config = []
        best_latency_improvement = float("-inf")
        best_config_acc = 0
        best_exit_rate = None

        for config in configs:
            config = list(config)
            nums_exit = [0 for i in range(len(ramp_ids) + 1)]
            correct = 0

            for profile in pickle_dict.values():
                orig_model_prediction = profile["orig_model_prediction"][0]
                all_entropies = profile["all_entropies"]
                all_predictions = profile["all_predictions"]
                
                has_exited = False
                for i in range(len(ramp_ids)):
                    if all_entropies[ramp_ids[i]] <= config[i]:
                        ramp_prediction = all_predictions[ramp_ids[i]][0]
                        if orig_model_prediction == ramp_prediction:
                            correct += 1
                        has_exited = True
                        nums_exit[i] += 1
                        break

                if not has_exited:
                    correct += 1
                    nums_exit[-1] += 1

            exit_rate = np.array([(n+0.0)/len(pickle_dict) for n in nums_exit])
            acc = round((correct+0.0)/len(pickle_dict), 9)
            latency_improvement = (baseline - sum(exit_rate * latency_config)) / baseline * 100
            
            if abs(1 - acc) < 0.015 and latency_improvement > best_latency_improvement:
                # print(config, acc, latency_improvement, exit_rate, flush=True)
                best_config = config
                best_latency_improvement = latency_improvement
                best_config_acc = acc
                best_exit_rate = exit_rate

        return best_config, best_latency_improvement, best_config_acc, best_exit_rate

    def query_performance_mp(self, configs, all_ramps_conf, all_ramps_acc, ramp_ids, latency_config, baseline):

        best_config = []
        best_latency_improvement = 0
        best_config_acc = 0
        best_exit_rate = None

        for config in configs:
            correct = 0
            config = list(config)
            nums_exit = [0 for i in range(len(ramp_ids) + 1)] 
            for i in range(len(all_ramps_conf[0])):
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
            
            exit_rate = np.array([(n+0.0)/len(all_ramps_conf[0]) for n in nums_exit])
            acc = round((correct+0.0)/len(all_ramps_conf[0]), 7)
            latency_improvement = (baseline - sum(exit_rate * latency_config)) / baseline * 100

            # print(config, acc, nums_exit, latency_improvement)

            if abs(1 - acc) < 0.015 and latency_improvement > best_latency_improvement:
                # print(config, acc, latency_improvement)
                best_config = config
                best_latency_improvement = latency_improvement
                best_config_acc = acc
                best_exit_rate = exit_rate

        # print("my partition: ", configs, "my best config is: ", best_config, best_latency_improvement, best_config_acc)
            
        return best_config, best_latency_improvement, best_exit_rate, best_config_acc

    def explore_direction(self, task, offline_data, ramp_ids, config, step_sizes, latency_config, baseline, curr_acc, curr_latency_improvement, curr_exit_rate):
        best_direction = None
        best_score = float("inf")
        res_acc = None
        res_latency_improvement = None
        res_exit_rate = None
        equal_num = 0
        positive_dirs = []
        positive_dirs_data = []

        for direction in range(len(ramp_ids)):
            
            temp_config = copy.deepcopy(config)
            temp_config[direction] = round(temp_config[direction] + step_sizes[direction], 4)

            # if task == "cv":
            #     temp_acc, temp_latency_improvement, temp_exit_rate = \
            #         query_performance(temp_config, offline_data[0], offline_data[1], ramp_ids, latency_config, baseline)
            # elif task == "nlp":
            #     _, temp_latency_improvement, temp_acc, temp_exit_rate = \
            #     self.emulate_inference([temp_config], offline_data, ramp_ids, latency_config, baseline)
            temp_acc, temp_latency_improvement, temp_exit_rate = \
                query_performance(temp_config, offline_data[0], offline_data[1], ramp_ids, latency_config, baseline)

            # print("explore direction: ", direction, temp_acc, temp_latency_improvement, abs(temp_acc - curr_acc), abs(temp_latency_improvement - curr_latency_improvement), abs(temp_acc - curr_acc) / abs(temp_latency_improvement - curr_latency_improvement))
            if abs(1 - temp_acc) < 0.015:
                if temp_latency_improvement != curr_latency_improvement:
                    score =  abs(temp_acc - curr_acc) / abs(temp_latency_improvement - curr_latency_improvement)
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
                    positive_dirs_data += [[temp_acc, temp_latency_improvement, temp_exit_rate]]


        if equal_num == len(ramp_ids):
            return 0, curr_acc, curr_latency_improvement, curr_exit_rate, positive_dirs
        if not best_direction and len(positive_dirs) > 0:
            return positive_dirs[0], positive_dirs_data[0][0], positive_dirs_data[0][1], positive_dirs_data[0][2], positive_dirs
        # print("explore result: ", best_direction, res_acc, res_latency_improvement)
        return best_direction, res_acc, res_latency_improvement, res_exit_rate, positive_dirs

    def greedy_search_step(self, task, path, ramp_ids, min_step_size, s, data=None):

        with open(path, "rb") as f:
            if data: 
                offline_data = data
            else:
                offline_data = pickle.load(f)

        latency_config, baseline = get_latency_config(path, ramp_ids)
        step_sizes = [s]*len(ramp_ids)
        # print(step_sizes)
        config = [0.0 for _ in ramp_ids]

        curr_acc, curr_latency_improvement, curr_exit_rate = None, None, None

        # if task == "cv":
        #     all_ramps_conf, all_ramps_acc = offline_data["conf"], offline_data["acc"]
        #     curr_acc, curr_latency_improvement, curr_exit_rate = \
        #         query_performance(config, all_ramps_conf, all_ramps_acc, ramp_ids, latency_config, baseline)
        # elif task == "nlp":
        #     _, curr_latency_improvement, curr_acc, curr_exit_rate = \
        #         self.emulate_inference([config], offline_data, ramp_ids, latency_config, baseline)
        all_ramps_conf, all_ramps_acc = offline_data["conf"], offline_data["acc"]
        curr_acc, curr_latency_improvement, curr_exit_rate = \
            query_performance(config, all_ramps_conf, all_ramps_acc, ramp_ids, latency_config, baseline)

        while True:
            # print("curr ", config, curr_acc, curr_latency_improvement, step_sizes)
            next_direction, next_acc, next_latency_improvement, next_exit_rate, positive_dirs = None, None, None, None, None
            # if task == "cv":
            #     next_direction, next_acc, next_latency_improvement, next_exit_rate, positive_dirs = \
            #         self.explore_direction(task, [all_ramps_conf, all_ramps_acc], ramp_ids, config, step_sizes, latency_config, baseline, curr_acc, curr_latency_improvement, curr_exit_rate)
            # elif task == "nlp":
            #     next_direction, next_acc, next_latency_improvement, next_exit_rate, positive_dirs = \
            #         self.explore_direction(task, offline_data, ramp_ids, config, step_sizes, latency_config, baseline, curr_acc, curr_latency_improvement, curr_exit_rate)
            next_direction, next_acc, next_latency_improvement, next_exit_rate, positive_dirs = \
                self.explore_direction(task, [all_ramps_conf, all_ramps_acc], ramp_ids, config, step_sizes, latency_config, baseline, curr_acc, curr_latency_improvement, curr_exit_rate)

            if next_direction != None and config[next_direction] <= 1:
                curr_acc = next_acc
                curr_latency_improvement = next_latency_improvement
                curr_exit_rate = next_exit_rate
                config[next_direction] = round(config[next_direction] + step_sizes[next_direction], 4)
                step_sizes[next_direction] *= 2
                for i in positive_dirs:
                    if i != next_direction:
                        step_sizes[i] *= 2
                # print("next ", config, curr_acc, curr_latency_improvement, step_sizes)
            else:
                flag = True
                for i in range(len(step_sizes)):
                    if round(step_sizes[i], 4) <= min_step_size \
                        or config[i] > 1:
                        continue
                    else:
                        flag = False
                        step_sizes[i] /= 2
                if flag:
                    break
                
        return config, curr_latency_improvement, curr_exit_rate, curr_acc

    def greedy_search(self, task, path, ramp_ids, min_step_size=0.0125, data=None):
        '''
            task (str): cv or nlp
            path (str): path to the offline data
            ramp_ids (list): list of ramp ids
            min_step_size (float): the minimum step size
        '''
        best_config, best_latency_improvement, best_exit_rates, best_acc = None, float("-inf"), None, None

        for s in [0.0125, 0.025, 0.05]:
            s = round(s, 4)
            cur_config, curr_latency_improvement, curr_exit_rates, curr_acc = \
                self.greedy_search_step(task, path, ramp_ids, min_step_size, s, data=data)

            if curr_latency_improvement > best_latency_improvement:
                best_config = cur_config
                best_latency_improvement = curr_latency_improvement
                best_exit_rates = curr_exit_rates
                best_acc = curr_acc

        print("greedy search: ", ramp_ids, best_config, best_latency_improvement, best_exit_rates, best_acc, flush=True)

        return best_config, best_latency_improvement, best_exit_rates, best_acc

    def grid_search(self, task, path, ramp_ids, step_size, grid_size):
        '''
            task (str): "cv" or "nlp"
            path (str): path to the offline data
            ramp_ids (list): list of 0-indexed ramp ids
            step_size (float): step size for the grid search
            grid_size (int): number of grid points for each ramp
        '''
        
        best_config = []
        best_latency_improvement = 0
        best_config_acc = 0
        best_exit_rate = None

        if task == "cv":
            with open(path,'rb') as f:
                offline_data = pickle.load(f)
            
            all_ramps_conf, all_ramps_acc = offline_data["conf"], offline_data["acc"]
            ramp_configs = self.generate_ramp_configs(ramp_ids, step_size, grid_size)
            latency_config, baseline = get_latency_config(path, ramp_ids)
            
            all_configs = itertools.product(*ramp_configs)
            chunked_pairs = self.get_chunks(all_configs, chunks=multiprocessing.cpu_count())
            # chunked_pairs = self.get_chunks(all_configs, chunks=1)

            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                results = pool.starmap(self.query_performance_mp, \
                    zip(chunked_pairs, repeat(all_ramps_conf), repeat(all_ramps_acc), repeat(ramp_ids), repeat(latency_config), repeat(baseline)))
                for result in results:
                    if result[1] > best_latency_improvement:
                        best_config = result[0]
                        best_latency_improvement = result[1]
                        best_exit_rate = result[2]
                        best_config_acc = result[3]
            
            print("grid search: ", ramp_ids, best_config, best_latency_improvement, best_config_acc, flush=True)

        elif task == "nlp":
            with open(path, "rb") as f:
                pickle_dict = pickle.load(f)

            ramp_configs = self.generate_ramp_configs(ramp_ids, step_size, grid_size)
            latency_config, baseline = get_latency_config(path, ramp_ids)
            all_configs = itertools.product(*ramp_configs)
            chunked_pairs = self.get_chunks(all_configs, chunks=multiprocessing.cpu_count())

            with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
                results = pool.starmap(self.emulate_inference, \
                    zip(chunked_pairs, repeat(pickle_dict), repeat(ramp_ids), repeat(latency_config), repeat(baseline)))
                for result in results:
                    if result[1] > best_latency_improvement:
                        best_config = result[0]
                        best_latency_improvement = result[1]
                        best_config_acc = result[2]
                        best_exit_rate = result[3]

            print("grid search: ", ramp_ids, best_config, best_latency_improvement, best_exit_rate, best_config_acc, flush=True)


        return best_config, best_latency_improvement, best_exit_rate, best_config_acc