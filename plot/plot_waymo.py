import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from itertools import combinations

# plt.rcParams["figure.figsize"] = (25, 5)
# plt.rcParams["font.size"] = 4
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["font.size"] = 15

for ramp_ids in list(combinations([4, 8, 10, 12], 4)):
    ramp_ids = list(ramp_ids)
    print("ramp config: ", ramp_ids)

    with open("../ramp-{}.pickle".format("-".join([str(i) for i in ramp_ids])), 'rb') as f:
        p = pickle.load(f)

    optimal = p[0]
    greedy = p[1]
    tune_first_only = p[2]
    tune_every_2_chunks = p[3]
    tune_every_4_chunks = p[4]
    greedy_w_addition = p[5]
    greedy_ramp_adjustment = p[6]
        
    acc_first = [(1 - a[3]) < 0.015 for a in tune_first_only]
    acc_two = [(1 - a[3]) < 0.015 for a in tune_every_2_chunks]
    acc_four = [(1 - a[3]) < 0.015 for a in tune_every_4_chunks]
    acc_greedy_w_addition = [(1 - a[3]) < 0.015 for a in greedy_w_addition]
    acc_greedy_ramp_adjustment = [(1 - a[3]) < 0.015 for a in greedy_ramp_adjustment]

    latency_savings_optimal = [a[1] for a in optimal]
    latency_savings_greedy = [max(a[1], 0) for a in greedy]
    latency_savings_greedy_w_addition = [max(a[1], 0) for a in greedy_w_addition]
    latency_savings_greedy_ramp_adjustment = [max(a[1], 0) for a in greedy_ramp_adjustment]
    latency_savings_first = [a[1] for a in tune_first_only]
    latency_savings_two = [a[1] for a in tune_every_2_chunks]
    latency_savings_four = [a[1] for a in tune_every_4_chunks]

    latency_savings = [
        latency_savings_greedy, latency_savings_first, latency_savings_two, latency_savings_four, latency_savings_greedy_w_addition
    ]

    
    labels = ["greedy", "first_only", "every_2_chunks", "every_4_chunks"]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    linestyles = ["solid", "solid", "dashed"]
    colors = ["blue", "orange", "red"]

    # for savings, label in zip(latency_savings, labels):
    # for savings, label in [(latency_savings_optimal, "optimal"), (latency_savings_two, "2_chunks")]:
    temp_latency = latency_savings_greedy_ramp_adjustment
    temp_acc = acc_greedy_ramp_adjustment
    # for savings, label in [(latency_savings_greedy, "greedy"), (temp_latency, "greedy_w_addition")]:
    for savings, label in [(latency_savings_greedy, "greedy-"+"-".join([str(i) for i in ramp_ids])), (temp_latency, "greedy_core_ramps"), (latency_savings_greedy_w_addition, "greedy_w_addition")]:
        plt.plot(list(range(len(savings))), savings, label=label, linestyle=linestyles.pop(0), color=colors.pop(0))
    
    for i, saving in enumerate(temp_latency):
        if not temp_acc[i]:
            acc_violation = matplotlib.patches.Rectangle((i-0.5, 0), 1, 60, color='#FF8888')
            ax.add_patch(acc_violation)
    
    plt.legend(loc="lower left")
    plt.xlabel(f"Video chunk ID")
    plt.ylabel(f"Latency savings compared to vanilla (%)")
    # plt.xticks(list(range(0, 161)))
    plt.xticks(list(range(0, 41)))
    plt.yticks(list(range(0, 60, 5)))
    # plt.title(f"Ramps {' '.join([str(i) for i in ramp_ids])}")
    plt.grid(color='grey', linewidth=0.5)
    filename = "./ramp-{}.png".format("-".join([str(i) for i in ramp_ids]))
    dpi = 400
    plt.savefig(filename, dpi=dpi)
    plt.close()
