# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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
latency_dict = {
    "Apparate": [],  # median runtime across all workloads for ResNet18, ResNet50, ...
    "Vanilla": [],
}
latency_minmax_dict = {
    "Apparate": [],  # median runtime across all workloads for ResNet18, ResNet50, ...
    "Vanilla": [],
}

label_names = ["Apparate", "Vanilla"]
multiplier = 0

# %%
# fill in data dict
for model in models:
    all_apparate_p95 = []
    all_vanilla_p95 = []
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
        
        serving_time_vanilla = results["serving_time_vanilla"]
        serving_time_ee = results["serving_time_ee"]
        
        all_apparate_p95.append(np.percentile(serving_time_ee, 95))
        all_vanilla_p95.append(np.percentile(serving_time_vanilla, 95))
    latency_dict["Apparate"].append(np.median(all_apparate_p95))
    latency_dict["Vanilla"].append(np.median(all_vanilla_p95))
    
    latency_minmax_dict["Apparate"].append([
        np.median(all_apparate_p95) - min(all_apparate_p95),
        max(all_apparate_p95) - np.median(all_apparate_p95),
    ])
    latency_minmax_dict["Vanilla"].append([
        np.median(all_vanilla_p95) - min(all_vanilla_p95),
        max(all_vanilla_p95) - np.median(all_vanilla_p95),
    ])
    print(f"model {model}, all_apparate_p95 {all_apparate_p95}, all_vanilla_p95 {all_vanilla_p95}")
    print(f"model {model}, median all_apparate_p95 {np.median(all_apparate_p95)}, min {min(all_apparate_p95)}, max {max(all_apparate_p95)}")


# %%

fig, ax = plt.subplots()
for scheme_idx, (scheme_name, all_p95) in enumerate(latency_dict.items()):
    offset = bar_width * multiplier
    print(f"scheme {scheme_idx}, all_p95 {all_p95}")
    yerr = list(map(list, zip(*latency_minmax_dict[scheme_name])))  # transpose the list to be passed in yerr
    print(f"yerr: {yerr}")
    rects = ax.bar(
        x_pos + offset, [round(x, 2) for x in all_p95], 
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

ax.set_ylabel('P95 Latency (ms)', fontsize=13)
ax.set_xticks(x_pos + bar_width * 0.5, models, fontsize=13)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3), ncols=2, fontsize=10)
ax.set_axisbelow(True)  # puts the grid below the bars
ax.grid(color='lightgrey', linestyle='dashed', axis="both", linewidth=0.8)

fig.savefig(f'cv_results_p95.pdf', bbox_inches='tight', dpi=500)
# %%
