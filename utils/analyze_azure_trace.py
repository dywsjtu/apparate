import os
import pickle
import random
import sys
import csv
import time
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from pprint import pprint
sys.path.insert(1, os.path.join(os.getcwd(), 'profiling'))  # for loading profile pickles
sys.path.insert(1, os.path.abspath('..'))
import plotting


AZURE_TRACE_DIR = "/home/ruipan/azure-functions"
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


qps_lowest, qps_highest = 30, float("inf")
df = parse_azure_trace()
# filter out traces that satisfies the qps range condition
df = df[(qps_lowest < df["avg_qps"]) & (df["avg_qps"] < qps_highest)]
df = df[arrival_rates_columns]
print(f"Functions after filtering: {df}")

plotting.analyze_azure_trace(df)

# for i, row in df.iterrows():
#     print(f"trace {i}")
    