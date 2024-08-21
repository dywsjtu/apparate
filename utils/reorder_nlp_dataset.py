# evaluates different methods of partitioning and rearranging NLP datasets (by hardness)
import math
import os
import sys
import copy
import pickle
import random
import numpy as np

# this script can:
# - check if the notion of hardness changes much across different models
# - check if the notion of hardness changes much across partitioning methods
# M: use entropy at first ramp
# M: use sum of entropies across all ramps
# M: use ramp index at which first correct prediction is made
# M: use ramp index after which all predictions are correct

filename_template = "{dataset}_{model}_entropies.pickle"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print("Seeded everything")

set_seed(0)

def find_consecutive_true(lst):
    # given a list of booleans, return the first True where
    # all elements afterward are also True
    for i in range(len(lst)):
        if lst[i] and all(lst[i+1:]):
            return i
    return -1  # if no such elements are found


def find_easyness_index(easyness_buckets, index):
    # given a list of easyness buckets and the index of a sample,
    # find the index of the bucket containing the sample 
    for i in range(len(easyness_buckets)):
        if index in easyness_buckets[i]:
            return i
    assert False, f"Sample {index} not in any bucket!"
        
def read_entropy_pickle(dataset, model):
    # entropy_pickle_dir = os.path.join(os.path.expanduser("~"), "deebert", "entropy_pickles_hf", "serving")
    entropy_pickle_dir = os.path.join(os.path.expanduser("~"), "nlp_ee", "entropy_pickles_hf", "serving")
    filename = os.path.join(entropy_pickle_dir, filename_template.format(dataset=dataset, model=model))
    with open(filename, "rb") as f:
        # assumes nlp pickles: smaller entropy -> more confident prediction
        p = pickle.load(f)
    return p

def bucket_similarity(b1, b2):
    # deprecated
    assert len(b1) == len(b2)
    num_samples = sum([len(x) for x in b1])
    num_buckets = len(b1)
    num_samples_in_same_bucket = 0
    for bucket_id in range(num_buckets):
        s1, s2 = b1[bucket_id], b2[bucket_id]
        union = set(s1).union(s2)
        intersection = set(s1).intersection(s2)
        num_samples_in_same_bucket += len(intersection)
        print(f"bucket {bucket_id}, s1 {len(s1)}, s2 {len(s2)}, superset {len(union)}, intersection {len(intersection)}")
    print(f"overlapping rate: {round(num_samples_in_same_bucket / num_samples * 100, 2)}%")

def partition_dataset(dataset, model, method):
    p = read_entropy_pickle(dataset, model)
    
    # num_samples = len(p)
    # num_ramps = len(p[0]["all_entropies"]) - 1  # last ramp is not used
    num_samples = len(p['conf'][0])
    num_ramps = len(p['conf'])  # includes original model, which is considered as a ramp
    # print(f"num_samples {num_samples}, num_ramps {num_ramps}")
    
    easyness_buckets = [[] for _ in range(num_ramps)]  # put sample IDs into different buckets, ordered by easyness ascending
    
    if "rampindex" in method:
        for sample_id in range(num_samples):
            if method == "rampindex_first":
                all_predictions = [p["acc"][ramp_id][sample_id] for ramp_id in range(num_ramps)]
                try:
                    easyness_index = all_predictions.index(True)
                except ValueError:  # True is not in list
                    easyness_index = num_ramps  # exit from end of vanilla model
            elif method == "rampindex_all":
                all_predictions = [p["acc"][ramp_id][sample_id] for ramp_id in range(num_ramps)]
                easyness_index = find_consecutive_true(all_predictions)
                if easyness_index == -1:
                    easyness_index = num_ramps
            elif method == "rampindex_top3":
                all_predictions = [p["acc"][ramp_id][sample_id] for ramp_id in range(num_ramps)]
                exitable_locations = [index for index, accurate in enumerate(all_predictions) if accurate]
                # TODO
            # print(sample_id, all_predictions, easyness_index)
            
            # add sample id to corresponding easyness bucket
            easyness_buckets[easyness_index].append(sample_id) 
    elif "prediction_" in method:
        ramp_id = int(method[method.find('_')+1:])
        assert ramp_id + 1 <= num_ramps, f"Invalid ramp ID {ramp_id} (num_ramps {num_ramps})"
        predictions = p["acc"][ramp_id]
        entropies = p["conf"][ramp_id]
        pred_correct_sample_ids = [index for index, pred in enumerate(predictions) if pred]
        pred_incorrect_sample_ids = [index for index, pred in enumerate(predictions) if not pred]
        print(f"num correct {len(pred_correct_sample_ids)}, num incorrect {len(pred_incorrect_sample_ids)}")
        
        # all correct predictions at ramp -> all incorrect predictions
        # among the two segments, rank samples according to entropy
        sorted_index = np.argsort(entropies)  # sort by confidence/easyness descending
        correct_ids_ordered = [x for x in sorted_index if x in pred_correct_sample_ids]
        incorrect_ids_ordered = [x for x in sorted_index if x in pred_incorrect_sample_ids]
        return (correct_ids_ordered, incorrect_ids_ordered)
    elif "entropy_" in method:
        num_samples_in_bucket = math.ceil(num_samples / num_ramps)
        ramp_id = method[method.find('_')+1:]
        if ramp_id.isdigit():
            ramp_id = int(ramp_id)
            assert ramp_id + 1 <= num_ramps, f"Invalid ramp ID {ramp_id} (num_ramps {num_ramps})"
            entropies = p["conf"][ramp_id]
        else:
            assert method == "entropy_sum"
            entropies = [sum([p[ramp_id][sample_id] for ramp_id in range(num_ramps)]) 
                         for sample_id in range(num_samples)]

        # for sample_id in range(num_samples):
        #     print(sample_id, p[sample_id]["all_entropies"][:-1], entropies[sample_id])
        
        sorted_index = np.argsort(entropies)[::-1]  # sort by confidence/easyness descending
        start_indices = list(range(0, num_samples + num_samples_in_bucket, num_samples_in_bucket))[:-1]
        end_indices = [x + num_samples_in_bucket for x in start_indices][:-1] + [num_samples + 1]
        for bucket_id, (start, end) in enumerate(zip(start_indices, end_indices)):
            easyness_buckets[bucket_id] = list(sorted_index[start:end])

    return easyness_buckets
        
def reorder_dataset(dataset, model, partition_method, easyness_index_order):
    p = read_entropy_pickle(dataset, model)
    if easyness_index_order is None:  # original order
        return p, p
    num_ramps = len(p['conf'])
    partition_results = partition_dataset(dataset, model, partition_method)
    if type(partition_results) is list:  # partition_results is an easyness_buckets  
        easyness_buckets = partition_results
        streaming_sample_ids = []
        for easyness_index in easyness_index_order:
            streaming_sample_ids += easyness_buckets[easyness_index]
    elif type(partition_results) is tuple:
        correct_ids_ordered, incorrect_ids_ordered = partition_results
        streaming_sample_ids = []
        ################################################################
        # method 1: all correct + all incorrect
        # streaming_sample_ids = correct_ids_ordered + incorrect_ids_ordered
        ################################################################
        # method 2: 
        chunk_size = 50  # num samples in each correct/incorrect chunk
        total_num_samples = 30000  # num samples to include in streaming workload
        
        # # method 1: 5000 samples. say 100 samples in orig dataset, 75-25 correct, chunk size 25.
        # # for all 50 chunks in streaming workload, each chunk contains 100 samples from orig dataset.
        # while len(streaming_sample_ids) < total_num_samples:
        #     correct, incorrect = copy.deepcopy(correct_ids_ordered), copy.deepcopy(incorrect_ids_ordered)
        #     while len(correct) != 0 or len(incorrect) != 0:  # some samples not yet added to streaming sample IDs
        #         if len(streaming_sample_ids) < chunk_size:
        #             k = 0  # cannot have an incorrect chunk at the beginning
        #         elif len(correct) != 0 and len(incorrect) != 0:
        #             k = random.randint(0, 1)
        #         elif len(correct) != 0:
        #             k = 0
        #         else:
        #             k = 1
        #         if k == 0:  # pop from correct
        #             new_chunk = correct[:chunk_size]
        #             del correct[:chunk_size]
        #         else:  # pop from incorrect
        #             if streaming_sample_ids == []:
        #                 continue  # cannot have an incorrect chunk at the beginning
        #             new_chunk = incorrect[:chunk_size]
        #             del incorrect[:chunk_size]
        #         streaming_sample_ids += new_chunk
        
        # # method 2: only include correct predictions
        # while len(streaming_sample_ids) < total_num_samples:
        #     correct = copy.deepcopy(correct_ids_ordered)
        #     streaming_sample_ids += correct
        
        # method 3: 
        num_samples_in_dataset = len(correct_ids_ordered) + len(incorrect_ids_ordered)
        num_repeats = math.ceil(total_num_samples / num_samples_in_dataset)
        print(f"num_repeats {num_repeats}")
        chunk_size = len(incorrect_ids_ordered) * num_repeats  # chunk size s.t. all correct samples are grouped together
        discreteness_factor = 1.0
        chunk_size = int(chunk_size / discreteness_factor)
        print(f"chunk_size {chunk_size}")
        correct, incorrect = copy.deepcopy(correct_ids_ordered), copy.deepcopy(incorrect_ids_ordered)
        correct *= num_repeats
        incorrect *= num_repeats
        while len(streaming_sample_ids) < (len(correct) + len(incorrect)):
            while len(correct) != 0 or len(incorrect) != 0:  # some samples not yet added to streaming sample IDs
                print(len(correct), len(incorrect))
                if len(correct) != 0 and len(incorrect) != 0:
                    k = random.randint(0, 1)
                elif len(correct) != 0:
                    k = 0
                else:
                    k = 1
                if k == 0:  # pop from correct
                    new_chunk = correct[:chunk_size]
                    del correct[:chunk_size]
                else:  # pop from incorrect
                    if streaming_sample_ids == []:
                        continue  
                    new_chunk = incorrect[:chunk_size]
                    del incorrect[:chunk_size]
                streaming_sample_ids += new_chunk
                
            
        print(f"total num samples {len(streaming_sample_ids)}")
        ################################################################
    
        print(f"chunk_size {chunk_size}")
    
    new_dataset = {"conf": [], "acc": []}
    easiest_dataset = {"conf": [], "acc": []}  # samples in first bucket only
    
    for ramp_id in range(num_ramps):
        new_dataset["conf"].append(
            [p["conf"][ramp_id][id] for id in streaming_sample_ids]
        )
        new_dataset["acc"].append(
            [p["acc"][ramp_id][id] for id in streaming_sample_ids]
        )

        # easiest_dataset["conf"].append(
        #     [p["conf"][ramp_id][id] for id in easyness_buckets[0]]
        # )
        # easiest_dataset["acc"].append(
        #     [p["acc"][ramp_id][id] for id in easyness_buckets[0]]
        # )
        
    return new_dataset, easiest_dataset
    


#########create new orderings of datasets#########
dataset = "qqp"
# model = "bert-large-uncased"
model = "dev_gpt2"
partition_method = "prediction_2"
# rampindex_first, rampindex_all, entropy_0, entropy_sum, prediction_5

# easyness_buckets = partition_dataset(dataset, model, method=partition_method)
# print(f"Total num samples: {sum([len(x) for x in easyness_buckets])}")
# for b_i, bucket in enumerate(easyness_buckets):
#     # print(f"{b_i}, num samples in bucket {len(bucket)}")
#     print(len(bucket))

    
# different ordering schemes:
# original -- preserve ordering in original dataset
# harder -- order samples by easyness descending (becomes harder): buckets 0, 1, ..., 11
# 0.1.2.11.10.9.3.4.5.8.7.6
    
ordering = {
    "original": None,
    "harder": list(range(12)),
    # "0.1.2.11.10.9.3.4.5.8.7.6": [0, 1, 2, 11, 10, 9, 3, 4, 5, 8, 7, 6]
}

for scheme_name, easyness_index_order in ordering.items():
    new_dataset, easiest_dataset = reorder_dataset(dataset, model, partition_method, easyness_index_order)
    if scheme_name == "original":
        assert read_entropy_pickle(dataset, model) == new_dataset
    else:
        assert read_entropy_pickle(dataset, model) != new_dataset
    entropy_pickle_dir = os.path.join(os.path.expanduser("~"), "apparate", "nlp_ordered_datasets", partition_method, scheme_name)
    if not os.path.exists(entropy_pickle_dir):
        os.makedirs(entropy_pickle_dir)
    entropy_pickle_path = os.path.join(entropy_pickle_dir, filename_template.format(dataset=dataset, model=model))
    print(entropy_pickle_path)
    with open(entropy_pickle_path, "wb") as f:
        pickle.dump(new_dataset, f)


    # entropy_pickle_dir = os.path.join(os.path.expanduser("~"), "apparate", "nlp_ordered_datasets", "easiest_dataset", partition_method, scheme_name)
    # if not os.path.exists(entropy_pickle_dir):
    #     os.makedirs(entropy_pickle_dir)
    # entropy_pickle_path = os.path.join(entropy_pickle_dir, filename_template.format(dataset=dataset, model=model))
    # print(entropy_pickle_path)
    # with open(entropy_pickle_path, "wb") as f:
    #     pickle.dump(easiest_dataset, f)

    
exit()


#########code for testing different partitioning methods#########

# dataset = "qnli"
# model1 = "bert-base-uncased"
# model2 = "bert-large-uncased"

# # rampindex_first, rampindex_all, entropy_1, entropy_sum
# b1 = partition_dataset(dataset, model1, method="rampindex_first")
# b2 = partition_dataset(dataset, model2, method="rampindex_first")

# print("="*50)
# for b_i, bucket in enumerate(b1):
#     print(f"{b_i}, num samples in bucket {len(bucket)}")
# assert sum([len(x) for x in b1]) == sum([len(x) for x in b2])

# b2_new = []
# for i in range(len(b1)):
#     newlist = b2[2 * i] + b2[2 * i + 1]
#     newlist.sort()
#     b2_new.append(newlist)
# b2 = b2_new

# # 12 buckets. if across models, the same sample is placed in buckets distance within 2, 
# # we consider the two models similar.
# num_samples = sum([len(x) for x in b1])
# num_buckets = len(b1)
# num_similar_easyness_samples = 0
# for i in range(num_samples):
#     e1 = find_easyness_index(b1, i)
#     e2 = find_easyness_index(b2, i)
    
#     if abs(e1 - e2) <= 2:
#         num_similar_easyness_samples += 1
# print(f"similar rate: {round(num_similar_easyness_samples / num_samples * 100, 2)}%")

