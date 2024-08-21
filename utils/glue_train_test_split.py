import os
import sys
import copy
import pickle
import itertools
from sklearn.model_selection import train_test_split

"""
Reads an entropy pickle file obtained by running generate_entropy_pickle.sh in deebert,
partitions it into a train/test (serving/bootstrapping) (90-10) set split.
The "test set" is used to run bootstrapping, while the "train set" is used for crafting a streaming workload.
"""

# entropy_pickle_dir = "/home/ruipan/deebert/entropy_pickles_hf"
entropy_pickle_dir = "/home/ruipan/nlp_ee/entropy_pickles_hf"
serving_entropy_pickle_dir = os.path.join(entropy_pickle_dir, "serving")
bootstrapping_entropy_pickle_dir = os.path.join(entropy_pickle_dir, "bootstrapping")
for dir in [serving_entropy_pickle_dir, bootstrapping_entropy_pickle_dir]:
    if not os.path.exists(dir):
        os.mkdir(dir)
entropy_pickle_filename = "{dataset}_dev_{model}_entropies.pickle"
# models = ["bert-base-uncased", "bert-large-uncased"]
models = ["gpt2"]
datasets = ["rte", "mrpc", "sst-2", "qnli", "qqp"]


def filter_samples(entropy_file, sample_ids):
    """
    Given an entropy pickle file and a list of sample IDs to keep,
    return the filtered pickle file
    """
    num_exits = len(entropy_file["acc"])
    filtered_file = copy.deepcopy(entropy_file)
    for key in ["acc", "conf"]:
        for exit_id in range(num_exits):
            new_list = [entropy_file[key][exit_id][id] for id in sample_ids]
            filtered_file[key][exit_id] = new_list
    return filtered_file


for model, dataset in itertools.product(models, datasets):
    print("{:=^50}".format(f"Splitting {model}+{dataset}"))
    pickle_filename = entropy_pickle_filename.format(model=model, dataset=dataset)
    filename = os.path.join(entropy_pickle_dir, pickle_filename)
    with open(filename, "rb") as f:
        p = pickle.load(f)
    
    num_samples = len(p["acc"][0])
    all_ids = list(range(num_samples))
    serving_ids, bootstrapping_ids = train_test_split(all_ids, test_size=0.1, train_size=0.9, random_state=0)
    
    for dir, sample_ids in [[serving_entropy_pickle_dir, serving_ids], 
                            [bootstrapping_entropy_pickle_dir, bootstrapping_ids]]:
        new_filename = os.path.join(dir, pickle_filename)
        filtered_file = filter_samples(p, sample_ids)
        with open(new_filename, "wb") as f:
            pickle.dump(filtered_file, f)
