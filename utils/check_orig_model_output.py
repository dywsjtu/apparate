import pickle

############ report orig model output on all datasets ############

filename_template = "/home/ruipan/deebert/orig_model_output_hf/{dataset}_bert-base-uncased_eval.pickle"
datasets = ["rte", "mrpc", "sst-2", "qnli", "mnli", "qqp"]
for dataset in datasets:
    filename = filename_template.format(dataset=dataset)
    print('='*50)
    print(filename)
    with open(filename, "rb") as f:
        p = pickle.load(f)
    acc = sum([p[i]["accurate"] for i in range(len(p))]) / len(p)
    print(f"acc {acc}")
        

############ compare between two ############

# # file_1 = "/home/ruipan/deebert/orig_model_output_hf/rte_bert-base-uncased_eval.pickle"  # huggingface weights
# # file_2 = "/home/ruipan/deebert/orig_model_output/rte_bert-base-uncased_eval.pickle"  # our own trained model weights

# file_1 = "/home/ruipan/deebert/orig_model_output_hf/qqp_bert-large-uncased_eval.pickle"  # deebert output
# file_2 = "/home/ruipan/deebert/orig_model_output_hf/qqp_bert-large-uncased_eval.pickle"  # bert output

# with open(file_1, "rb") as f1, open(file_2, "rb") as f2:
#     p1 = pickle.load(f1)
#     p2 = pickle.load(f2)
    
#     exactly_same = (p1 == p2)
#     print(f"exactly_same: {exactly_same}")
    
#     same_predictions = [p1[i]["pred"] == p2[i]["pred"] for i in range(len(p1))]
#     print(f"similarity between orig prediction of different weights: {sum(same_predictions) / len(same_predictions)}")  # 222/277
#     acc_1 = sum([p1[i]["accurate"] for i in range(len(p1))]) / len(p1)
#     acc_2 = sum([p2[i]["accurate"] for i in range(len(p1))]) / len(p2)
#     print(f"acc_1 {acc_1}, acc_2 {acc_2}")
#     # for sample_id in range(len(p1)):
#     #     if 
#     #     pass
    
