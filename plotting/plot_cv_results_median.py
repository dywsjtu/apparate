# %%
import os
import sys
import time
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce

from plotting_utils import parse_result_file

matplotlib.rcParams["figure.figsize"] = (7.5, 1.85)  # (4, 1.3)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# %%
models = ["ResNet18", "ResNet50", "ResNet101", "VGG11", "VGG13", "VGG16"]
datasets = ["Auburn", "Bellevue1", "Bellevue2", "Calgary", "Coral", "Hampton", "Oxford"]

# NOTE(ruipan): use these for the final version. use the abs path here for testing.
# BATCH_DECISION_PATH = "../../batch_decisions/{model}_{slo_multiplier}_{arrival}.pickle"
# APPARATE_LATENCY_PATH = "../../apparate_latency/{model}_{dataset}_{slo_multiplier}_{arrival}.pickle"
# OPTIMAL_LATENCY_PATH = "../../optimal_latency/{model}_{dataset}_{slo_multiplier}_{arrival}_optimal.pickle"
BATCH_DECISION_PATH = "/home/ruipan/apparate-ae/data/batch_decisions/{model}_{slo_multiplier}_{arrival}.pickle"
APPARATE_LATENCY_PATH = "/home/ruipan/apparate-ae/data/apparate_latency/{model}_{dataset}_{slo_multiplier}_{arrival}.pickle"
OPTIMAL_LATENCY_PATH = "/home/ruipan/apparate-ae/data/optimal_latency/{model}_{dataset}_{slo_multiplier}_{arrival}_optimal.pickle"

bar_width = 0.3  # the width of the bars
x_pos = np.arange(len(models))
patterns = ["", "\\", "/"]
colors = ["#d9ffbf", "#4f963c"]
latency_dict = {
    "Apparate": [],  # median runtime across all workloads for ResNet18, ResNet50, ...
    "Optimal": [],
}
latency_minmax_dict = {
    "Apparate": [],
    "Optimal": [],
}

label_names = ["Apparate", "Optimal"]
multiplier = 0




# %%
# fill in data dict
for model in models:
    all_apparate_serving_improvement = []
    all_optimal_serving_improvement = []
    for dataset in datasets:
        print(f"parsing", model, dataset)
        results = parse_result_file(
            model.lower(),
            dataset.lower() + "_video",
            slo_multiplier=1,
            arrival="fixed_30",
            BATCH_DECISION_PATH=BATCH_DECISION_PATH,
            APPARATE_LATENCY_PATH=APPARATE_LATENCY_PATH,
            OPTIMAL_LATENCY_PATH=OPTIMAL_LATENCY_PATH,
        )
        
        apparate_serving_improvement = results["apparate_serving_improvement"]
        optimal_serving_improvement = results["optimal_serving_improvement"]
        serving_time_vanilla = results["serving_time_vanilla"]
        serving_time_ee = results["serving_time_ee"]
        serving_time_optimal = results["serving_time_optimal"]

        if np.median(apparate_serving_improvement) < 0:
            print(f"model {model} dataset {dataset} median improvement is negative! {np.median(apparate_serving_improvement)}")
        #### median
        # print(f"dataset {dataset}, first 10 apparate_serving_improvement {apparate_serving_improvement[:10]}")
        all_apparate_serving_improvement.append(apparate_serving_improvement)
        all_optimal_serving_improvement.append(optimal_serving_improvement)
        # print(f"model {model} dataset {dataset} apparate_serving_improvement {(apparate_serving_improvement)}")
        print(f"within {1 - apparate_serving_improvement / optimal_serving_improvement}")
        # print(f"absolute median saving: {np.median(serving_time_vanilla) - np.median(serving_time_ee)}")
        
        # print(f"p95 are within xx of vanilla model: {1 - np.percentile(serving_time_ee, 95) / np.percentile(serving_time_vanilla, 95)}")
        
        
        #### p95
        # FIXME: something wrong with the plotting script??
        # all_apparate_serving_improvement.append(np.percentile(apparate_serving_improvement, 5))
        # all_optimal_serving_improvement.append(np.percentile(optimal_serving_improvement, 5))
        # print(f"model {model} dataset {dataset} apparate_serving_improvement p95 {np.percentile(apparate_serving_improvement, 5)}")
        ####
    latency_dict["Apparate"].append(np.median(all_apparate_serving_improvement))
    latency_dict["Optimal"].append(np.median(all_optimal_serving_improvement))
    
    latency_minmax_dict["Apparate"].append([
        max(np.median(all_apparate_serving_improvement) - min(all_apparate_serving_improvement), 2),
        max(max(all_apparate_serving_improvement) - np.median(all_apparate_serving_improvement), 2),
    ])
    latency_minmax_dict["Optimal"].append([
        np.median(all_optimal_serving_improvement) - min(all_optimal_serving_improvement),
        max(all_optimal_serving_improvement) - np.median(all_optimal_serving_improvement),
    ])

    print(f"model {model}, all_apparate_serving_improvement {all_apparate_serving_improvement}")
    print(f"model {model}, all_apparate_serving_improvement median {np.median(all_apparate_serving_improvement)}, min {min(all_apparate_serving_improvement)}, max {max(all_apparate_serving_improvement)}")


# %%

fig, ax = plt.subplots()
for scheme_idx, (scheme_name, median_latencies) in enumerate(latency_dict.items()):
    offset = bar_width * multiplier
    print(f"scheme {scheme_idx}, median_latencies {median_latencies}, median {np.median(median_latencies)}")
    yerr = list(map(list, zip(*latency_minmax_dict[scheme_name])))  # transpose the list to be passed in yerr
    print(f"yerr: {yerr}")
    rects = ax.bar(
        x_pos + offset, [round(x, 2) for x in median_latencies], 
        bar_width, 
        # https://stackoverflow.com/a/33857966: yerr values are relative to the data...
        yerr=yerr,
        error_kw=dict(ecolor='#666666', capsize=3,),
        color=colors[scheme_idx], edgecolor='#666666', 
        label=scheme_name, hatch=patterns[scheme_idx], 
        alpha=.99,
    )
    # ax.bar_label(rects, padding=2, fontsize=9, color="#666666")
    multiplier += 1

# ax.set_yscale("log")
ax.set_ylabel('Med. Latency Wins\nvs. Vanilla (%)', fontsize=13)
# ax.set_ylabel('P95 Latency Wins\nvs. Vanilla', fontsize=13)
ax.set_xticks(x_pos + bar_width * 0.5, models, fontsize=13)
# plt.setp(ax.get_xticklabels(), rotation=15, ha="center")
# ax.set_ylim(0, 25)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncols=2, fontsize=10)
ax.set_axisbelow(True)  # puts the grid below the bars
ax.grid(color='lightgrey', linestyle='dashed', axis="both", linewidth=0.8)

fig.savefig(f'cv_results_median.pdf', bbox_inches='tight', dpi=500)
# fig.savefig(f'cv_results_p95.png', bbox_inches='tight', dpi=500)





# # %%
# # Fig. 13: p95
# latency_dict = {
#     "Apparate": [],  # median runtime across all workloads for ResNet18, ResNet50, ...
#     "Optimal": [],
# }
# latency_minmax_dict = {
#     "Apparate": [],
#     "Optimal": [],
# }


# for model in models:
#     all_apparate_serving_improvement = []
#     all_optimal_serving_improvement = []
#     for dataset in datasets:
#         print(f"parsing", model, dataset)
#         results = parse_result_file(
#             model.lower(),
#             dataset.lower() + "_video",
#             slo_multiplier=1,
#             arrival="fixed_30",
#             BATCH_DECISION_PATH=BATCH_DECISION_PATH,
#             APPARATE_LATENCY_PATH=APPARATE_LATENCY_PATH,
#             OPTIMAL_LATENCY_PATH=OPTIMAL_LATENCY_PATH,
#         )
        
#         apparate_serving_improvement = results["apparate_serving_improvement"]
#         optimal_serving_improvement = results["optimal_serving_improvement"]
#         serving_time_vanilla = results["serving_time_vanilla"]
#         serving_time_ee = results["serving_time_ee"]
#         serving_time_optimal = results["serving_time_optimal"]

#         if np.median(apparate_serving_improvement) < 0:
#             print(f"model {model} dataset {dataset} median improvement is negative! {np.median(apparate_serving_improvement)}")

#         #### p95
#         # FIXME: something wrong with the plotting script??
#         all_apparate_serving_improvement.append(np.percentile(apparate_serving_improvement, 5))
#         all_optimal_serving_improvement.append(np.percentile(optimal_serving_improvement, 5))
#         print(f"model {model} dataset {dataset} apparate_serving_improvement p95 {np.percentile(apparate_serving_improvement, 5)}")
#         ####
#     latency_dict["Apparate"].append(np.median(all_apparate_serving_improvement))
#     latency_dict["Optimal"].append(np.median(all_optimal_serving_improvement))
    
#     latency_minmax_dict["Apparate"].append([
#         np.percentile(all_apparate_serving_improvement, 5) - min(all_apparate_serving_improvement),
#         max(all_apparate_serving_improvement) - np.percentile(all_apparate_serving_improvement, 5),
#     ])
#     latency_minmax_dict["Optimal"].append([
#         np.percentile(all_optimal_serving_improvement, 5) - min(all_optimal_serving_improvement),
#         max(all_optimal_serving_improvement) - np.percentile(all_optimal_serving_improvement, 5),
#     ])

#     print(f"model {model}, all_apparate_serving_improvement {all_apparate_serving_improvement}")
#     print(f"model {model}, all_apparate_serving_improvement median {np.median(all_apparate_serving_improvement)}, min {min(all_apparate_serving_improvement)}, max {max(all_apparate_serving_improvement)}")


# # %%

# fig, ax = plt.subplots()
# for scheme_idx, (scheme_name, median_latencies) in enumerate(latency_dict.items()):
#     offset = bar_width * multiplier
#     print(f"scheme {scheme_idx}, median_latencies {median_latencies}, median {np.median(median_latencies)}")
#     yerr = list(map(list, zip(*latency_minmax_dict[scheme_name])))  # transpose the list to be passed in yerr
#     print(f"yerr: {yerr}")
#     rects = ax.bar(
#         x_pos + offset, [round(x, 2) for x in median_latencies], 
#         bar_width, 
#         # https://stackoverflow.com/a/33857966: yerr values are relative to the data...
#         yerr=yerr,
#         error_kw=dict(ecolor='#666666', capsize=3,),
#         color=colors[scheme_idx], edgecolor='#666666', 
#         label=scheme_name, hatch=patterns[scheme_idx], 
#         alpha=.99,
#     )
#     # ax.bar_label(rects, padding=2, fontsize=9, color="#666666")
#     multiplier += 1

# # ax.set_yscale("log")
# ax.set_ylabel('P95 Latency Wins\nvs. Vanilla', fontsize=13)
# ax.set_xticks(x_pos + bar_width * 0.5, models, fontsize=13)
# # plt.setp(ax.get_xticklabels(), rotation=15, ha="center")
# # ax.set_ylim(0, 25)
# ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncols=2, fontsize=10)
# ax.set_axisbelow(True)  # puts the grid below the bars
# ax.grid(color='lightgrey', linestyle='dashed', axis="both", linewidth=0.8)

# # fig.savefig(f'cv_results_median.pdf', bbox_inches='tight', dpi=500)
# fig.savefig(f'cv_results_p95.png', bbox_inches='tight', dpi=500)
# # %%
