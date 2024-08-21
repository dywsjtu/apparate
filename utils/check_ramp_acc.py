import os, sys
import pickle
sys.path.insert(1, os.path.join(os.path.dirname(os.getcwd())))
import utils

# filename = "../entropy_pickles/qnli_distilbert-base-uncased_entropies.pickle"

# filename = "../entropy_pickles/rte_bert-base-uncased_entropies.pickle"
# filename = "../../deebert/entropy_pickles/rte_bert-base-uncased_entropies.pickle"
# filename = "../../deebert/entropy_pickles_hf/rte_bert-large-uncased_entropies.pickle"

# filename_template = "../../deebert/entropy_pickles_numepochs/sst-2_bert-base-uncased_entropies_{num_epochs}.pickle"
# filename_template = "../../deebert/entropy_pickles_hf/{dataset}_bert-base-uncased_entropies.pickle"
filename_template = "../../nlp_ee/entropy_pickles_hf/{dataset}_dev_gpt2_entropies.pickle"

# for num_epochs in range(0, 32, 2):
#     filename = filename_template.format(num_epochs=num_epochs)

datasets = ["rte", "mrpc", "sst-2", "qnli", "qqp"]  # "mnli"


# for _ in range(1):
for dataset in datasets:
    
    filename = filename_template.format(dataset=dataset)
    print('='*50)
    print(filename)

    with open(filename, "rb") as f:
        p = pickle.load(f)
        
    if "acc" not in p:  # nlp format pickle
        p = utils.pickle_format_convert(p)
        
    num_samples = len(p["acc"][0])
    num_ramps = len(p["acc"])  # last entry is original model output, second last is last ramp (ignore)

    for i in range(num_ramps):
        acc = sum(p["acc"][i]) / len(p["acc"][i])
        print(f"ramp {i}, acc {acc}")
        