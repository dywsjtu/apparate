# Apparate: Rethinking Early Exits to Tame Latency-Throughput Tensions in ML Serving

This repository contains the source code implementation of the [SOSP '24](https://sigops.org/s/conferences/sosp/2024/) paper [Apparate: Rethinking Early Exits to Tame Latency-Throughput Tensions in ML Serving](https://arxiv.org/abs/2312.05385).

## Directory Structure

### `profile_pickles_bs`

Contains the operator-level latency profile of different models at different batch sizes, all measured on an NVIDIA RTX A6000 GPU.

Format: `{model_name}_{batch_size}_profile.pickle` for vanilla models, and `{model_name}_{batch_size}_earlyexit_profile.pickle` for EE models.

### `batch_decisions`

Contains the batching decisions of Clockwork using different models and request arrival traces. 

Format: `{model_name}_1_fixed_30.pickle` for 30FPS video traces (CV workloads), and `{model_name}_azure.pickle` for Microsoft Azure MAF traces (NLP workloads).

### `optimal_latency`

Contains the per-sample optimal latency for different workloads.

Format: `{model_name}_{dataset}_optimal.pickle`. The pickled object is a list of floats, with each one denoting the queuing delay + optimal model inference latency of a request.

### `{bootstrap,simulation}_pickles`

Contains the confidence and accuracy of the bootstrapping/simulation dataset at all EE ramps.

Format: `{bootstrap,simulation}_{dataset}_{model_name}.pickle`. The pickled object `p` is a dict with two keys: "conf" and "acc". The confidence/accuracy of sample i at ramp r can be accessed via: `p["conf"/"acc"][r][i]`.

## Getting Started

Apparate is implemented in Python. We have tested Apparate on Ubuntu xx with Python xx.

Detailed instructions on how to reproduce the main results from our SOSP paper are in [EXPERIMENTS.md](EXPERIMENTS.md).


## References

```
@article{dai2023apparate,
  title={Apparate: Rethinking Early Exits to Tame Latency-Throughput Tensions in ML Serving},
  author={Dai, Yinwei and Pan, Rui and Iyer, Anand and Li, Kai and Netravali, Ravi},
  journal={arXiv preprint arXiv:2312.05385},
  year={2023}
}
```