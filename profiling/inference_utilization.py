import os
import sys
import time
import torch
import pickle
import nvidia_smi
import multiprocessing as mp
sys.path.insert(1, os.path.dirname(os.getcwd()))
import models

# how many inferences to run
NUM_INFERENCES = 1000
NUM_WARMUPS = 200
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64]
MODELS = [
    ("waymo", "resnet50_waymo"),
    ("waymo", "resnet18_waymo"),
    ("urban", "resnet18_urban"),
]

# initialize nvml and obtain GPU device name
# TODO(ruipan): maybe using torch.cuda.utilization() is easier
nvidia_smi.nvmlInit()
handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
device_name = nvidia_smi.nvmlDeviceGetName(handle).decode("utf-8").replace(" ", "_")

def measure_utilization(running_flag, avg_util):
    # code for gpu utilization profiling
    nvidia_smi.nvmlInit()
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
    utils = []
    while True:
        if running_flag.value:
            util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
            utils.append(util.gpu)
        else:
            avg_util.value = sum(utils) / len(utils)
            exit(0)

def main():
    all_profiles = {}
    for dataset, model_name in MODELS:
        # model = models.waymo.resnet18_waymo()
        model = models.__getattribute__(dataset).__getattribute__(model_name)()
        model.to("cuda:0")
        
        util_at_bs = {}
        for i, batch_size in enumerate(BATCH_SIZES):
            running_flag = mp.Value("i", 1)
            avg_util = mp.Value("f", 0.0)
            
            input_size = (batch_size, 3, 224, 224)  # for resnet50_waymo and resnet18_urban
            inputs = torch.randn(input_size)
            inputs = inputs.to("cuda:0")
            
            for _ in range(NUM_WARMUPS):  # warmup
                model(inputs)
            
            # launch new process to do utilization profiling
            p = mp.Process(target=measure_utilization, args=(running_flag, avg_util))
            p.start()
            
            for _ in range(NUM_INFERENCES):
                model(inputs)
            
            # terminate process, collect result
            running_flag.value = 0
            p.join()
            print(f"model {model_name}, batch_size {batch_size}, value of avg: {avg_util.value}")
            util_at_bs[batch_size] = avg_util.value
        print(f"model {model_name}, util_at_bs {util_at_bs}")
        all_profiles[model_name] = util_at_bs
    print(all_profiles, file=open(f"{device_name}.txt", "w"))
    with open(f"{device_name}.pickle", "wb") as f:
        pickle.dump(all_profiles, f)

if __name__ == '__main__':
    main()
