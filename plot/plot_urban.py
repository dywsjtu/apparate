import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from itertools import combinations

# plt.rcParams["figure.figsize"] = (25, 5)
# plt.rcParams["font.size"] = 4
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["font.size"] = 15

for ramp_ids in list(combinations([1, 3, 5], 2)):
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
    lazy_tune = p[6]
    lazy_tune_1 = p[7]
    js_tune = p[8]
        
    acc_first = [(1 - a[3]) < 0.015 for a in tune_first_only]
    acc_two = [(1 - a[3]) < 0.015 for a in tune_every_2_chunks]
    acc_four = [(1 - a[3]) < 0.015 for a in tune_every_4_chunks]
    acc_greedy_w_addition = [(1 - a[3]) < 0.015 for a in greedy_w_addition]
    acc_greedy = [(1 - a[3]) < 0.015 for a in greedy]
    acc_optimal = [(1 - a[3]) < 0.015 for a in optimal]
    acc_js = [(1 - a[3]) < 0.015 for a in js_tune[:-1]]
    # acc_greedy_ramp_adjustment = [(1 - a[3]) < 0.015 for a in greedy_ramp_adjustment]

    latency_savings_optimal = [a[1] for a in optimal]
    latency_savings_greedy = [max(a[1], 0) for a in greedy]
    latency_savings_greedy_w_addition = [max(a[1], 0) for a in greedy_w_addition]
    latency_savings_lazy = [max(a[1], 0) for a in lazy_tune[:-1]]
    latency_savings_js = [max(a[1], 0) for a in js_tune[:-1]]

    # latency_savings_greedy_ramp_adjustment = [max(a[1], 0) for a in greedy_ramp_adjustment]
    latency_savings_first = [a[1] for a in tune_first_only]
    latency_savings_two = [a[1] for a in tune_every_2_chunks]
    latency_savings_four = [a[1] for a in tune_every_4_chunks]

    latency_savings = [
        latency_savings_greedy, latency_savings_first, latency_savings_two, latency_savings_four
    ]

    
    labels = ["greedy", "first_only", "every_2_chunks", "every_4_chunks"]

    fig = plt.figure()
    ax = fig.add_subplot(111)

    linestyles = ["solid", "solid", "dashed"]
    colors = ["blue", "orange", "red"]
    
    temp_latency = latency_savings_js
    temp_acc = acc_js
    retune_idxs = js_tune[-1]
    print(retune_idxs)
    for savings, label in [(latency_savings_greedy, "greedy"), (temp_latency, "retune on JS")]:
    # for savings, label in [(latency_savings_greedy, "greedy-"+"-".join([str(i) for i in ramp_ids])), (temp_latency, "greedy_core_ramps"), (latency_savings_greedy_w_addition, "greedy_w_addition")]:
        plt.plot(list(range(len(savings))), savings, label=label, linestyle=linestyles.pop(0), color=colors.pop(0))
    
    for i, saving in enumerate(temp_latency):
        if temp_acc != None:
            if not temp_acc[i]:
                acc_violation = matplotlib.patches.Rectangle((i-0.5, 0), 1, 60, color='#FF8888')
                ax.add_patch(acc_violation)
        # if i in retune_idxs:
        #     ax.add_patch(matplotlib.patches.Rectangle((i-0.5, 0), 1, 60, color='#88acff')) 
    
    plt.legend(loc="lower left")
    plt.xlabel(f"Video chunk ID")
    plt.ylabel(f"Latency savings compared to vanilla (%)")
    # plt.xticks(list(range(0, 161)))
    plt.xticks(list(range(0, 51)))
    plt.yticks(list(range(0, 60, 5)))
    # plt.title(f"Ramps {' '.join([str(i) for i in ramp_ids])}")
    plt.grid(color='grey', linewidth=0.5)
    filename = "./ramp-{}.png".format("-".join([str(i) for i in ramp_ids]))
    dpi = 400
    plt.savefig(filename, dpi=dpi)
    plt.close()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # linestyles = ["solid", "solid"]
    # colors = ["blue", "orange"]

    # for i in range(len(ramp_ids)):
    #     exit_rate = [js_tune[j][2][i]*100 for j in range(len(js_tune))]
    #     plt.plot(list(range(len(js_tune))), exit_rate, label="ramp {}".format(str(ramp_ids[i])), linestyle=linestyles.pop(0), color=colors.pop(0))

    # for i, saving in enumerate(js_tune):
    #     if i in retune_idxs:
    #         retune_idx = matplotlib.patches.Rectangle((i-0.5, 0), 1, 60, color='#88acff')
    #         ax.add_patch(retune_idx) 

    # plt.legend(loc="lower left")
    # plt.xlabel(f"Video chunk ID")
    # plt.ylabel(f"Exit rate %")
    # # plt.xticks(list(range(0, 161)))
    # plt.xticks(list(range(0, 51)))
    # plt.yticks(list(range(0, 80, 10)))
    # # plt.title(f"Ramps {' '.join([str(i) for i in ramp_ids])}")
    # plt.grid(color='grey', linewidth=0.5)
    # filename = "./ramp-{}-exit-rate.png".format("-".join([str(i) for i in ramp_ids]))
    # dpi = 400
    # plt.savefig(filename, dpi=dpi)
    # plt.close()
