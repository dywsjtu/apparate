import re
import os, sys
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plotting

model = "bert-large-uncased"
dataset = "qnli"
# model = "resnet18_urban"
# dataset = "urban"
methods = ["nochange", "nextbest", "rampaddition", "exitpoint", ]
# methods = ["nochange"]
avg_savings_dict = {k: None for k in methods}

for method in methods:
    with open(f"../logs/output_{model}_{dataset}_{method}.log", "r") as f:
        lines = f.readlines()

    per_batch_latency_improvement = []

    for i, line in enumerate(lines):
        if "serve_batch" in line:
            batch_id = int(re.search(r"serve_batch: batch (\d+)", line).group(1))
            thresholds = re.search(r"thresholds ([\[\]\d\.,\s]+)", line).group(1)
            acc = float(re.search(r"actual acc ([\d.]+)", line).group(1))
            latency_improvement = float(re.search(r"latency_improvement ([+-]?[\d.]+)", line).group(1))
            per_batch_latency_improvement.append(latency_improvement)
        elif "shadow_ramp_id " in line:
            pass

    # group the latency improvements into every 10 batches
    avg_savings = []
    while len(per_batch_latency_improvement) >= 10:
        avg_savings.append(np.average(per_batch_latency_improvement[:10]))
        per_batch_latency_improvement = per_batch_latency_improvement[10:]
    avg_savings.append(np.average(per_batch_latency_improvement))
    avg_savings_dict[method] = avg_savings
    
plotting.plot_latency_savings_over_batches(
    model, dataset, avg_savings_dict
)