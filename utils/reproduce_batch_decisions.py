import os
import pickle

data_dir = "../batch_decisions_temp"

slos = [80, 100, 120, 140]
batching_scheme = "tf_serve"

arch = "bert-base-uncased"
# arch = "resnet50_waymo"

filename_suffix = f"fixed_{180}"
# filename_suffix = f"fixed_{350}"


for slo in slos:
    filename = os.path.join(data_dir, f"{batching_scheme}_{arch}_{slo}_{filename_suffix}.pickle")
    with open(filename, "rb") as f:
        p = pickle.load(f)
    max_batch_size = p["max_batch_size"]
    batch_timeout_ms = p["batch_timeout_ms"]
    max_enqueued_batches = p["max_enqueued_batches"]
    
    print('='*50)
    print(f"slo {slo}, max_batch_size {max_batch_size}, batch_timeout_ms {batch_timeout_ms}, max_enqueued_batches {max_enqueued_batches}")
