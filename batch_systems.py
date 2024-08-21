# refer to sec 5.3, "Scheduling INFER" of the Clockwork OSDI paper for a description of its batching mechanism
# this assumes single model, single stream (same SLO for all requests)
import os
import pickle
import random
import sys
import csv
import time
import utils
import numpy as np
import pandas as pd
from statistics import median, mean
from collections import Counter
pd.options.mode.chained_assignment = None  # default='warn'
from pprint import pprint
import plotting
sys.path.insert(1, os.path.join(os.getcwd(), 'profiling'))  # for loading profile pickles

"""
======================================================================
    class definitions of request and strategy and helper functions
======================================================================
"""
class Request:
    def __init__(self, request_id: int, slo: int, arrival_time: float):
        self.request_id = request_id  # unique int id
        self.slo = slo  # SLO in ms
        self.arrival_time = arrival_time  # current time in ms
        self.deadline = self.arrival_time + self.slo  # timestamp after which an SLO violation will occur
    
    def __str__(self):
        return f"<Request {self.request_id}, arrival time {self.arrival_time}, deadline {self.deadline}>"
    
    def __repr__(self):
        return self.__str__()
    
    def has_expired(self, serving_time: float, current_time: float):
        """Checks if a request has expired

        Args:
            serving_time (float, optional): model inference latency in ms
            current_time (float, optional): current timestamp in ms (starting from 0)

        Returns:
            bool: whether a request has expired
        """
        return (current_time + serving_time) > self.deadline

class Strategy:
    def __init__(self, batch_size: int, deadline):
        self.batch_size = batch_size
        # latest time to start serving using this strategy to not violate
        # SLOs for any request in the batch
        self.deadline = deadline
    
    def __str__(self):
        return f"<Strategy bs {self.batch_size}, deadline {self.deadline}>"
    
    def __repr__(self):
        return self.__str__()

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

# functions for parsing azure workload trace
# AZURE_TRACE_DIR = "/home/ruipan/azure-functions"
AZURE_TRACE_DIR = "/data2/ruipan/azure-functions"
INTERVAL_DURATION_SECONDS = 60  # time between entries in the azure trace
arrival_rates_columns = [str(x) for x in range(1, 1441)]

def parse_azure_trace(trace_id: int = 1):
    if trace_id < 10:
        trace_id = f"0{trace_id}"
    filename = f"invocations_per_function_md.anon.d{trace_id}.csv"
    filename = os.path.join(AZURE_TRACE_DIR, filename)
    df = pd.read_csv(filename, sep=",")
    df["total_requests"] = df[arrival_rates_columns].sum(axis=1)
    df["avg_qps"] = df["total_requests"] / (24 * 60 * 60)
    num_functions = len(df)
    print(f"Processed trace {trace_id}, {num_functions} workloads")
    return df


def get_model_serving_time(arch, profile_dir, latency_calc_list):

    model_serving_time = []

    for bs_idx, batch_size in enumerate(utils.supported_batch_sizes):
        profile_path = os.path.join(profile_dir, f"{arch}_{batch_size}_earlyexit_profile.pickle")
        if os.path.exists(profile_path):
            with open(profile_path, 'rb') as f:
                profile = pickle.load(f)
            latency_calc_list = utils.parse_profile(profile)
            vanilla_model_serving_latency = latency_calc_list[-1][0]
            model_serving_time.append(vanilla_model_serving_latency)
        else:
            raise Exception(f"No profile found for model {arch} at {profile_path}")

    # pprint(f"model_serving_time: {model_serving_time}")
    return model_serving_time

def create_request(fixed_arrival_rate, poisson_arrival, slo, qps):

    curr_request_id = 0  # counter for assigning IDs to requests
    curr_time = 0.0  # timestamp for generating arrival times
    all_requests = []  # all requests in the trace
    

    if fixed_arrival_rate:
        # total_num_requests = 30000  # number of total requests
        total_num_requests = 70000  # amazon
        # interarrival_time = 1000.0 / 180 # 19 fixed arrival rate: time in ms between request arrivals
        interarrival_time = 1000.0 / qps
        if poisson_arrival:
            np.random.seed(2023)
            time_between_arrivals \
                = np.random.exponential(scale=interarrival_time, size=total_num_requests)

    if fixed_arrival_rate:  # fixed arrival of requests
        avg_qps = 1000 / interarrival_time
        
        for _ in range(total_num_requests):
            request = Request(curr_request_id, slo, 
                            arrival_time=curr_time)
            all_requests.append(request)
            if poisson_arrival:
                delta_t = time_between_arrivals[curr_request_id]
            else:
                delta_t = interarrival_time
            curr_time += delta_t
            curr_request_id += 1
    else:  # azure trace
        # pick a trace with avg qps between 250 and 300 (15000-18000 queries/minute)
        # qps_lowest, qps_highest = 242, 249
        qps_lowest, qps_highest = 50, float("inf")
        df = parse_azure_trace()
        # # filter out traces that satisfies the qps range condition
        # df = df[(qps_lowest < df["avg_qps"]) & (df["avg_qps"] < qps_highest)]
        # df = df[arrival_rates_columns]
        
        # XXX: pick a function in the trace
        SELECTED_FUNCTION_ID = 7188
        TOTAL_NUM_REQUESTS = 700000  # 250050 for amazon, 179767 for imdb

        selected_func = df.loc[[SELECTED_FUNCTION_ID]]
        selected_func = selected_func[arrival_rates_columns]
        selected_func = selected_func.values.flatten().tolist()
        arrivals_per_min = []
        for v in selected_func:
            arrivals_per_min.append(int(v))
            if sum(arrivals_per_min) > TOTAL_NUM_REQUESTS:
                arrivals_per_min[-1] -= (sum(arrivals_per_min) - TOTAL_NUM_REQUESTS)
                break

        start_index = 0
        avg_qps = np.average(arrivals_per_min) / 60
        # print(f"Using trace {row}, avg qps {avg_qps}")
        for i in range(len(arrivals_per_min)):
            if arrivals_per_min[i] != 0: 
                interarrival_time = INTERVAL_DURATION_SECONDS * 1000.0 / arrivals_per_min[i]  # fixed arrival interval within each minute in ms
                for request_id in range(arrivals_per_min[i]):
                    print(f"adding request with id {curr_request_id}, arrival_time {curr_time + interarrival_time * request_id}")
                    request = Request(curr_request_id, slo, arrival_time=curr_time + interarrival_time * request_id)
                    curr_request_id += 1
                    all_requests.append(request)
            curr_time += INTERVAL_DURATION_SECONDS * 1000
        print(f"Total number of requests: {len(all_requests)}")
        assert sum(arrivals_per_min) == len(all_requests)
    return all_requests, avg_qps, interarrival_time

def get_batch_decision(batching_scheme, all_requests, model_serving_time, slo=0.0, max_batch_size=8, batch_timeout_ms=60, max_enqueued_batches=2):
    round_id = 0
    curr_time = 0.0  # timestamp for emulating serving
    total_num_requests = len(all_requests)
    batch_decision = []
    per_request_stats = [None for _ in range(len(all_requests))]  # (queueing delay, inference time)
    # request queue per batch size. new requests are enqueued into every batch queue.
    batch_queues = [[] for _ in utils.supported_batch_sizes]
    strategy_queue = []  # a strategy is created per batch queue, and all strategies are enqueued in here
    next_request_id = 0  # id of the next request that will be pulled in the scheduler
    last_served_request_id = -1  # id of the last successfully served request
    last_dropped_request_id = -1  # id of the last dropped request
    # max_batch_size = 8 # max batch size supported by the tf serving system
    # batch_timeout_ms = 60 # ms The maximum amount of time to wait before executing a batch (even if it hasn't reached max_batch_size).
    # max_enqueued_batches = 2 # The maximum number of batches to enqueue before rejecting them.
    last_served_time = -1 # timestamp of the last served batch
    tf_request_queue = [] # queue of requests for tf serving

    while max(last_served_request_id, last_dropped_request_id) < total_num_requests - 1:
        # pprint("{:=^75}".format(f"Serving round {round_id}, curr_time {curr_time}"))

        # pull in newly arrived requests, and add to all batch queues.
        # logically, find all requests in all_requests that are:
        # newly arrived & not expired & not already served/dropped & not currently in any batch queues

        incoming_requests = []
        for request_id in range(next_request_id, len(all_requests)):
            r = all_requests[request_id]
            if r.arrival_time > curr_time:  # arriving in the future, schedule in next round
                next_request_id = request_id
                break
            else:
                if r.deadline > curr_time:
                    # not expired, can still be scheduled, add to all batch queues
                    incoming_requests.append(all_requests[request_id])
                else:  # expired before getting the chance to be enqueued in batchqueue
                    if request_id > last_dropped_request_id:
                        last_dropped_request_id = request_id
        if curr_time > all_requests[-1].arrival_time:
            next_request_id = len(all_requests)
            if incoming_requests == []:
                print("No more incoming requests")
                break

        if batching_scheme == "clockwork":
             # add every incoming requests to all batch queues
            for bs_idx, batch_queue in enumerate(batch_queues):
                batch_queue += incoming_requests
            # pprint(f"incoming_requests {incoming_requests}")
            # dequeue expired/already served requests from all batch queues
            for bs_idx, batch_queue in enumerate(batch_queues):
                request_ids_before_dropping = [r.request_id for r in batch_queue]
                batch_queues[bs_idx] = [r for r in batch_queue
                                if not r.has_expired(serving_time=model_serving_time[bs_idx], current_time=curr_time)  # request has not expired
                                and r.request_id > last_served_request_id]  # request has not been served in prior rounds
                request_ids_after_dropping = [r.request_id for r in batch_queues[bs_idx]]
                dropped_requests = list(set(request_ids_before_dropping) - set(request_ids_after_dropping))
                # last_dropped_request_id = max(last_dropped_request_id, max(dropped_requests) if dropped_requests != [] else -1)
                # pprint(f"batch_queue {utils.supported_batch_sizes[bs_idx]}, remaining requests: {[r.request_id for r in batch_queues[bs_idx]]}")
            
            if all([q == [] for q in batch_queues]):  # no requests received, nothing to schedule, sleep for a while
                print(f"no requests received, skipping to the next request")
                print(f"curr_time {total_num_requests}, next_request_id {next_request_id}")
                if next_request_id < total_num_requests:
                    curr_time = all_requests[next_request_id].arrival_time
                else:
                    # NOTE(ruipan): list index out of range can be triggered 
                    break
                continue

            # update requests that are dropped
            curr_min_request_id = min([q[0].request_id for q in batch_queues if q != []])  # all requests with smaller IDs are either served or dropped
            last_dropped_request_id = max(last_dropped_request_id, curr_min_request_id - 1)  # update last_dropped_request_id

            # create and enqueue serving strategies produced by each batch queue
            for bs_idx, batch_queue in enumerate(batch_queues):
                if batch_queue == []:  # skip empty queues
                    deadline_hoq = float("inf")  # use inf to indicate invalid strategy
                else:  # deadline of request at head of queue
                    deadline_hoq = batch_queue[0].deadline
                strategy_queue.append(Strategy(
                    batch_size=utils.supported_batch_sizes[bs_idx],
                    deadline=deadline_hoq - model_serving_time[bs_idx],  # subtract batch execution time from deadline of request at head of queue
                ))
                    
            # pprint(f"strategy_queue before sorting: {strategy_queue}")
            strategy_queue.sort(key=lambda x: x.deadline)
            # pprint(f"strategy_queue after sorting: {strategy_queue}")

            # iterate starting from the tightest deadline
            final_strategy = None
            for strategy in strategy_queue:
                deadline = strategy.deadline
                batch_size = strategy.batch_size
                batch_idx = utils.supported_batch_sizes.index(batch_size)
                if (strategy.deadline == float("inf")  # empty batch queue, invalid strategy
                    or deadline < curr_time  # deadline of strategy has elapsed
                    or len(batch_queues[batch_idx]) < batch_size):  # batch queue doesn't have sufficient requests
                    continue
                
                # found valid strategy
                final_strategy = strategy
                strategy_queue = []  # remove old strategies
                # NOTE(ruipan): clockwork also speculatively tries to increase the bs as much as possible
                # but this doesn't make sense?? since increasing the bs might violate SLOs for requests at head of queue
                break

            # pprint(f"final_strategy {final_strategy}")

            if final_strategy is not None:  # if valid: run! 
                batch_size = final_strategy.batch_size
                batch_idx = utils.supported_batch_sizes.index(batch_size)
                requests = batch_queues[batch_idx][:batch_size] # requests to be served
                last_served_request_id = requests[-1].request_id
                # newly_served_requests = [r.request_id for r in requests]
                for r in requests:
                    per_request_stats[r.request_id] = (
                        curr_time - r.arrival_time,
                        model_serving_time[batch_idx],
                    )
                # pprint(f"emulating serving, sleeping for {model_serving_time[batch_idx]} ms...")
                curr_time += model_serving_time[batch_idx]
                batch_decision.append(batch_size)
            else:  # if not found: don't consider this for now...
                assert False, f"Corner case: no valid strategy found. still do something?"
        else:
            should_serve = False
            tf_request_queue += incoming_requests
            max_num_requests = max_enqueued_batches * max_batch_size

            if len(tf_request_queue) > max_num_requests:
                # print(f"tf_request_queue too long, drop requests from {tf_request_queue[max_num_requests].request_id} to {tf_request_queue[-1].request_id}")
                tf_request_queue = tf_request_queue[:max_num_requests]
                last_dropped_request_id = max(last_dropped_request_id, tf_request_queue[-1].request_id)

            for idx, r in enumerate(tf_request_queue):
                if r.arrival_time - last_served_time > batch_timeout_ms:
                    should_serve = True
                    break
            should_serve = should_serve or len(tf_request_queue) >= max_batch_size or next_request_id == len(all_requests)

            if not should_serve:
                # print(f"should not serve now, skipping to the next request")
                curr_time = all_requests[next_request_id].arrival_time  
                continue
           
            if idx + 1 > max_batch_size:
                tf_requests = tf_request_queue[:max_batch_size]
                tf_request_queue = tf_request_queue[max_batch_size:]
                batch_idx = utils.supported_batch_sizes.index(max_batch_size)
            else:
                for batch_idx, bz in enumerate(utils.supported_batch_sizes):
                    if idx + 1 <= bz:
                        tf_requests = tf_request_queue[:idx+1]
                        tf_request_queue = tf_request_queue[idx+1:]
                        break
            # print(f"tf_requests {tf_requests}")
            if tf_requests == []:
                continue
            last_served_time = tf_requests[-1].arrival_time
            last_served_request_id = max([r.request_id for r in tf_requests])

            for r in tf_requests:
                per_request_stats[r.request_id] = (
                    curr_time - r.arrival_time,
                    model_serving_time[batch_idx],
                )
            # pprint(f"emulating serving, sleeping for {model_serving_time[batch_idx]} ms...")
            curr_time += model_serving_time[batch_idx]
            batch_decision.append(utils.supported_batch_sizes[batch_idx])
        round_id += 1

    print(len(per_request_stats), sum(batch_decision))
    print("="*50)
    print(f"Serving complete!")

    num_served_requests = sum([1 for s in per_request_stats if s is not None and s[0] + s[1] < slo])
    num_dropped_requests = sum([1 for s in per_request_stats if s is None])
    num_slo_violations = sum([1 for s in per_request_stats if s is not None and s[0] + s[1] >= slo])

    serve_rate = num_served_requests / total_num_requests * 100
    drop_rate = num_dropped_requests / total_num_requests * 100
    slo_violation_rate = num_slo_violations / total_num_requests * 100
    avg_bs = mean(batch_decision)
    bs_frequency = Counter(batch_decision)
    print(f"Served: {num_served_requests} ({round(serve_rate, 3)}% of requests)")
    print(f"Dropped: {num_dropped_requests} ({round(drop_rate, 3)}% of requests)")
    print(f"SLO violations: {num_slo_violations} ({round(slo_violation_rate, 3)}% of requests)")
    print(f"Avg batch size: {round(avg_bs, 5)}, bs frequency: {bs_frequency}")
    print(f"Avg queueing delay: {round(np.average([s[0] for s in per_request_stats if s is not None and s[0] + s[1] < slo]), 3)}")
    print(f"Avg inference time: {round(np.average([s[1] for s in per_request_stats if s is not None and s[0] + s[1] < slo]), 3)}")
    print(f"Avg serving latency: {round(np.average([s[0] + s[1] for s in per_request_stats if s is not None and s[0] + s[1] < slo]), 3)}")
    
    return max_batch_size, batch_timeout_ms, max_enqueued_batches, batch_decision, per_request_stats, total_num_requests, curr_time

def get_latency_plots(dataset, fixed_arrival_rate, batching_scheme, arch, slo, avg_qps, per_request_stats, total_num_requests, total_time, interarrival_time, batch_decision, all_vanilla_latencies):
    
    # ee_file = os.path.join("../", file)
    # ee_file = os.path.join(os.getenv("HOME"), file)
    ee_file = f"../apparate_latency/{arch}_{dataset}_azure.pickle"
    apparate_optimal_file = f"../apparate_optimal_latency/{arch}_{dataset}_azure.pickle"
        
    print(f"ee_file {ee_file}")
    if os.path.exists(ee_file):
        with open(ee_file, "rb") as f:
            ee_serving_latency = pickle.load(f)
    else:
        ee_serving_latency = None

    print(f"optimal apparate file {ee_file}")
    if os.path.exists(apparate_optimal_file):
        with open(apparate_optimal_file, "rb") as f:
            apparate_optimal_latency = pickle.load(f)
    else:
        apparate_optimal_latency = None
    # apparate_optimal_latency = None

    optimal_ee_file = f"../optimal_latency/{arch}_{dataset}_optimal.pickle"

    if os.path.exists(optimal_ee_file):
        with open(optimal_ee_file, "rb") as f:
            optimal_ee_serving_latency = pickle.load(f)
    else:
        optimal_ee_serving_latency = None
        print("not found")
        print(optimal_ee_file)
    optimal_ee_serving_latency = None
    
    # pprint(f"per_request_stats {per_request_stats}")
    print(len(per_request_stats), sum(batch_decision))
    # print("="*50)
    # print(f"Serving complete!")
    # print(f"SLO = {slo} ms, trace average qps = {avg_qps}, avg interarrival time = {round(interarrival_time * 1000, 3)} ms")

    num_served_requests = sum([1 for s in per_request_stats if s is not None and s[0] + s[1] < slo])
    num_dropped_requests = sum([1 for s in per_request_stats if s is None])
    num_slo_violations = sum([1 for s in per_request_stats if s is not None and s[0] + s[1] >= slo])

    serve_rate = num_served_requests / total_num_requests * 100
    drop_rate = num_dropped_requests / total_num_requests * 100
    slo_violation_rate = num_slo_violations / total_num_requests * 100
    avg_bs = mean(batch_decision)
    bs_frequency = Counter(batch_decision)
    # print(f"Served: {num_served_requests} ({round(serve_rate, 3)}% of requests)")
    # print(f"Dropped: {num_dropped_requests} ({round(drop_rate, 3)}% of requests)")
    # print(f"SLO violations: {num_slo_violations} ({round(slo_violation_rate, 3)}% of requests)")
    # print(f"Average throughput: {round(num_served_requests / (total_time / 1e3), 3)} qps")
    # print(f"Avg batch size: {round(avg_bs, 5)}, bs frequency: {bs_frequency}")
    # print(f"Avg queueing delay: {round(np.average([s[0] for s in per_request_stats if s is not None and s[0] + s[1] < slo]), 3)}")
    # print(f"Avg inference time: {round(np.average([s[1] for s in per_request_stats if s is not None and s[0] + s[1] < slo]), 3)}")
    # print(f"Avg serving latency: {round(np.average([s[0] + s[1] for s in per_request_stats if s is not None and s[0] + s[1] < slo]), 3)}")

    queueing_delay_list = [s[0] for s in per_request_stats if s is not None and s[0] + s[1] < slo]
    inference_time_list = [s[1] for s in per_request_stats if s is not None and s[0] + s[1] < slo]
    latency_list = [s[0] + s[1] for s in per_request_stats if s is not None and s[0] + s[1] < slo]

    # print(len(ee_serving_latency), len(latency_list), len(all_vanilla_latencies))

    vanillar_latency = []
    if ee_serving_latency is not None:
        ee_latency_list = []
        idx = 0
        for s in per_request_stats:
            if s is not None and s[0] + s[1] < slo and idx < len(ee_serving_latency):
                ee_latency_list.append(s[0] + ee_serving_latency[idx])
                vanillar_latency.append(s[0] + all_vanilla_latencies[idx])
                idx += 1
    else:
        ee_latency_list = None

    # print(np.array(l1).mean(), np.array(l2).mean(), (np.array(l1).mean() - np.array(l2).mean()) / np.array(l2).mean())

    # print(np.array(l1).mean(), np.array(all_vanilla_latencies).mean(), (np.array(l1).mean() - np.array(all_vanilla_latencies).mean()) / np.array(all_vanilla_latencies).mean())

    if apparate_optimal_latency is not None:
        apparate_optimal_latency_list = []
        idx = 0
        for s in per_request_stats:
            if s is not None and s[0] + s[1] < slo and idx < len(apparate_optimal_latency):
                apparate_optimal_latency_list.append(s[0] + apparate_optimal_latency[idx])
                idx += 1
    else:
        apparate_optimal_latency_list = None
    apparate_optimal_latency_list = None

    if optimal_ee_serving_latency is not None:
        optimal_ee_latency_list = []
        idx = 0
        for s in per_request_stats:
            if s is not None and s[0] + s[1] < slo and idx < len(optimal_ee_serving_latency):
                optimal_ee_latency_list.append(s[0] + optimal_ee_serving_latency[idx])
                idx += 1
    else:
        optimal_ee_latency_list = None
    optimal_ee_latency_list = None

    latency_list = vanillar_latency

    # optimal_ee_latency_list = None
    # print(max(ee_latency_list))

    # plot latency cdf
    # profile_dir = "./profile_pickles_bs"
    # profile_dir = "/home/ruipan/deebert/profile_pickles_hf"
    # profile_dir = "/home/yinwei/apparate/profile_pickles_bs/profile_pickles_a6000_rampv0"

    # print(np.median(np.array(latency_list)), np.median(np.array(ee_latency_list)), np.median(np.array(apparate_optimal_latency_list)))

    plotting.plot_latency_cdf_different_slo(
        dataset,
        queueing_delay_list,
        inference_time_list,
        vanillar_latency,
        ee_latency_list,
        apparate_optimal_latency_list,
        optimal_ee_latency_list,
        batching_scheme=batching_scheme,
        slo=slo,
        fixed_arrival_rate=fixed_arrival_rate,
        interarrival_time=interarrival_time,
        avg_qps=avg_qps,
        arch=arch,
        drop_rate=drop_rate,
    )

    return np.median(np.array(latency_list)), np.median(np.array(ee_latency_list)), None




"""
======================================================================
    initialization
======================================================================
"""



if __name__ == "__main__":
    # populate model serving times
    model_serving_time = []

    profile_dir = "./profile_pickles_bs"
    # profile_dir = "./motivation_profile_pickles"
    # profile_dir = "./profile_pickles_bs/apparate_profile_pickles_t4"
    # profile_dir = "/home/ruipan/apparate/profile_pickles_bs"

    # arch = "bert-base-uncased"
    arch = "resnet18_urban"
    # arch = "resnet50_waymo"
    dataset = "urban"
    batching_scheme = "clockwork"  # clockwork, tf_serve
    set_seed()
    model_serving_time = get_model_serving_time(arch, profile_dir)
    """
    ======================================================================
        emulate arrival of requests
    ======================================================================
    """
    fixed_arrival_rate = True  # either use fixed arrival rate, or use arrival pattern from azure trace
    poisson_arrival = True  # whether to use poisson arrival pattern
    slo = 25  # 150 SLO in ms for all requests, typically 20 - 200
    qps = 375

    all_requests, avg_qps, interarrival_time = create_request(fixed_arrival_rate, poisson_arrival, slo, qps)
    # pprint(f"all_requests {len(all_requests)}")
    """
    ======================================================================
        get batch decision
    ======================================================================
    """

    if fixed_arrival_rate:
        filename_suffix = f"fixed_{int(avg_qps)}"
        # TODO(ruipan): use "poisson_" for poisson arrival?
    else:
        filename_suffix = f"azure_{round(avg_qps)}"
    filename = f"./batch_decisions/{batching_scheme}_{arch}_{slo}_{filename_suffix}.pickle"
    # filename = f"../{batching_scheme}_{arch}_{slo}_{filename_suffix}.pickle"
    if not os.path.exists(filename):
        print(f"{filename} does not exist, generating batching decisions...")
        max_batch_size, batch_timeout_ms, max_enqueued_batches, \
            batch_decision, per_request_stats, total_num_requests, total_time = get_batch_decision(batching_scheme, all_requests, model_serving_time, max_batch_size=8, batch_timeout_ms=60, max_enqueued_batches=2)
        with open(filename, "wb") as f:
            batch_info = {
                "batching_decision": batch_decision,
                "per_request_stats": per_request_stats,
                "total_num_requests": total_num_requests,
                "batching_scheme": batching_scheme,
                "arch": arch,
                "slo": slo,
                "dataset": dataset,
                "avg_qps": avg_qps,
                "end_time": total_time,
                "max_batch_size": max_batch_size,
                "batch_timeout_ms": batch_timeout_ms,
                "max_enqueued_batches": max_enqueued_batches,
            }
            pickle.dump(batch_info, f)
    else:
        print(f"loading batching decisions from {filename}...")
        with open(filename, "rb") as f:
            batch_info = pickle.load(f)
            batch_decision = batch_info["batching_decision"]
            per_request_stats = batch_info["per_request_stats"]
            total_num_requests = batch_info["total_num_requests"]
            batching_scheme = batch_info["batching_scheme"]
            arch = batch_info["arch"]
            slo = batch_info["slo"]
            dataset = batch_info["dataset"]
            avg_qps = batch_info["avg_qps"]
            total_time = batch_info["end_time"]
            max_batch_size = batch_info["max_batch_size"]
            batch_timeout_ms = batch_info["batch_timeout_ms"]
            max_enqueued_batches = batch_info["max_enqueued_batches"]

    get_latency_plots(dataset, fixed_arrival_rate, batching_scheme, arch, slo, avg_qps, per_request_stats, total_num_requests, total_time, interarrival_time, batch_decision)