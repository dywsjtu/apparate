import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
from functools import reduce

hardness = True  # True False
hardness_descending = True  # True for descending, False for ascending

matplotlib.rcParams["figure.figsize"] = (8 if hardness else 12, 3)  # 8 and 12
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# if hardness:
#     if hardness_descending:
#         suffix = "hardness_descending"
#     else:
#         suffix = "hardness_ascending"
# else:
#     suffix = "random"
suffix = "qqp_ascending"  # qqp_ascending, qqp_descending, qqp_descending_except_first, qqp_random

with open(f"../savings_acc_config_{suffix}.pickle", "rb") as f:
    p = pickle.load(f)

x_index = list(range(len(p)))

for i, expr_name in enumerate([
    "compare_latency_optimal_suboptimal",
    "optimal_ramps_across_subdatasets",
]):
    fig, ax = plt.subplots()

    optimal_savings = [x[0] for x in p]
    optimal_accuracy = [x[1] for x in p]
    optimal_configs = [x[2] for x in p]
    suboptimal_savings = [x[3] for x in p]
    suboptimal_accuracy = [x[4] for x in p]
    suboptimal_configs = [x[5] for x in p]
    s_simpler_savings = [x[6] for x in p]
    s_simpler_accuracy = [x[7] for x in p]
    s_simpler_configs = [x[8] for x in p]
    s_harder_savings = [x[9] for x in p]
    s_harder_accuracy = [x[10] for x in p]
    s_harder_configs = [x[11] for x in p]

    alpha = 0.7
        
    if i == 0:
        plt.plot(x_index, 
                optimal_savings, 
                marker="o", color="#1f77b4", alpha=alpha, label=f"Retune both")  # color="#1f77b4"
        plt.plot(x_index, 
                suboptimal_savings, 
                marker="o", color="red", alpha=alpha, label=f"Ramp fixed")  # color="red"
        # plt.plot(x_index,  # data becomes simpler, move ramps forward
        #         s_simpler_savings, 
        #         marker="o", color="orange", alpha=alpha, label=f"Ramp forward")
        plt.plot(x_index,  # data becomes harder, move ramps backward
                s_harder_savings, 
                marker="o", color="green", alpha=alpha, label=f"Ramp backward")
        
        print(f"optimal_savings {optimal_savings}")
        print(f"suboptimal_savings {suboptimal_savings}")
        print(f"s_simpler_savings {s_simpler_savings}")
        print(f"s_harder_savings {s_harder_savings}")

        # latency_diff = [x - y for x, y in zip(optimal_savings, suboptimal_savings)]
        # print(f"max difference {max(latency_diff)}")
        # print(f"avg difference {sum(latency_diff) / len(latency_diff)}")

        # yticks = list(range(35, 44, 2))
        yticks = list(range(25, 65, 5))
        # yticks = list(range(25, 85, 5))
        ylabel = f"Latency savings (%)"
    elif i == 1:
        for subdataset_idx, optimal_config_in_subdataset in enumerate(optimal_configs):
            for ramp, threshold in optimal_config_in_subdataset.items():
                alpha = threshold
                alpha *= 2  # normalize from 0-0.5 to 0-1 to make dots more obvious
                if alpha < 0.8:  # make shallow-colored dots more obvious
                    alpha += 0.2
                # alpha = 1 - alpha  # flip color scheme
                alpha = min(alpha, 1)  # in case crazy thresholds like 2.4 pop up
                plt.plot(subdataset_idx, ramp, alpha=alpha, marker="o", color="#1f77b4")
        yticks = list(range(12))
        ylabel = f"Optimal ramp IDs"
    
    ax.set_xticks(x_index)
    ax.set_xlabel(f"Subdataset ID", fontsize=12)
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel, fontsize=12)
    # ax.yaxis.set_label_coords(-0.11, 0.3)  # x, y
    # ax.set_ylim([min_y - 1e10, max_y + 1e10])

    ax.legend()
    # ax.legend(loc="lower right")
    ax.set_axisbelow(True)  # puts the grid below the bars
    ax.grid(color='lightgrey', linestyle='dashed', axis="both", linewidth=0.8)

    filename_dict = {
        0: "nlp_retuning",
        1: "optimal_ramp_locations",
    }
    fig.savefig(f'{filename_dict[i]}_{suffix}.png', bbox_inches='tight', dpi=500)

    # plt.show()
    plt.close()
