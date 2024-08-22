import pickle
import numpy as np

def values_to_cdf(values):
    cdf_list = []
    values.sort()
    count = 0
    for v in values:
        count += 1
        cdf_list.append(count / len(values))
    return cdf_list


def parse_result_file(
    model: str,
    dataset: str,
    slo_multiplier: int,
    arrival: str,
    BATCH_DECISION_PATH: str,
    APPARATE_LATENCY_PATH: str,
    OPTIMAL_LATENCY_PATH: str,
):
    print(f"model {model}, dataset {dataset}")
    
    if "slo_multiplier" in BATCH_DECISION_PATH:  # CV workload, with SLO multiplier
        batch_decision_path = BATCH_DECISION_PATH.format(model=model, 
                                                        slo_multiplier=slo_multiplier, arrival=arrival)
        apparate_latency_path = APPARATE_LATENCY_PATH.format(
            model=model, dataset=dataset, slo_multiplier=slo_multiplier, arrival=arrival
        )
        optimal_latency_path = OPTIMAL_LATENCY_PATH.format(
            model=model, dataset=dataset, slo_multiplier=4, arrival=arrival
        )
    else:  # NLP workload, azure arrival traice
        batch_decision_path = BATCH_DECISION_PATH.format(model=model, arrival=arrival)
        apparate_latency_path = APPARATE_LATENCY_PATH.format(
            model=model, dataset=dataset, 
            arrival=arrival,
        )
        optimal_latency_path = OPTIMAL_LATENCY_PATH.format(
            model=model, dataset=dataset, 
            arrival=arrival,
        )
    
    with open(batch_decision_path, "rb") as f1, open(apparate_latency_path, "rb") as f2, open(optimal_latency_path, "rb") as f3:
        batch_decision, apparate_latency, optimal_latency = pickle.load(f1), pickle.load(f2), pickle.load(f3)
        
    per_request_stats = batch_decision["per_request_stats"]  # every item: queuing delay, inference time
    per_request_stats = [x for x in per_request_stats if x is not None]
    total_num_requests = sum([1 for x in per_request_stats if x is not None])
    
    length = min(len(apparate_latency), len(optimal_latency))
    apparate_latency = apparate_latency[:length]
    optimal_latency = optimal_latency[:length]
    num_served_requests = len(apparate_latency)  # NOTE(ruipan): might be smaller than total_num_requests b/c some are dropped
    print(f"num_served_requests {num_served_requests}")
    queuing_delays = [s[0] for s in per_request_stats if s is not None]
    queuing_delays = queuing_delays[:num_served_requests]
    model_inference_time_vanilla = [s[1] for s in per_request_stats[:num_served_requests]]
    model_inference_time_ee = apparate_latency
    model_inference_time_optimal = optimal_latency

    serving_time_vanilla = [sum(x) for x in zip(queuing_delays, model_inference_time_vanilla)]
    serving_time_ee = [sum(x) for x in zip(queuing_delays, model_inference_time_ee)]
    serving_time_optimal = [sum(x) for x in zip(queuing_delays, model_inference_time_optimal)]

    apparate_serving_improvement = 100 * (1 - np.median(serving_time_ee) / np.median(serving_time_vanilla))
    optimal_serving_improvement = 100 * (1 - np.median(serving_time_optimal) / np.median(serving_time_vanilla))
        
    return {
        "apparate_serving_improvement": apparate_serving_improvement,
        "optimal_serving_improvement": optimal_serving_improvement,
        "serving_time_vanilla": serving_time_vanilla,
        "serving_time_ee": serving_time_ee,
        "serving_time_optimal": serving_time_optimal,
    }
