import os, sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from itertools import combinations

arch = "resnet18_urban"
idx = 0
pickle_path = os.path.join("./", "{}_plot_{}.pickle".format(arch, str(idx)))

with open(pickle_path, "rb") as f:
    all_latencies, all_accuracies, all_exit_ramp, threshold_tuning_history, vanilla_latency = pickle.load(f)

# print(len(all_latencies)), print(len(all_accuracies)), print(len(all_exit_ramp)), print(len(threshold_tuning_history))

bs = 64

print(sum(threshold_tuning_history))

accumulated_acc = []
accumulated_latency_improvement = []

individual_acc = []
individual_latency_improvement = []


batch_ids_violate_acc = []

idx = 0
for i in range(len(threshold_tuning_history)):
    max_idx = min((idx+1)*bs, len(all_latencies))
    curr_latency = all_latencies[: max_idx]
    curr_accuracy = all_accuracies[: max_idx]
    curr_exit_ramp = all_exit_ramp[: max_idx]
    curr_threshold_tuning_history = threshold_tuning_history[: max_idx]
    # print(sum(curr_accuracy)/(max_idx+0.0))
    
    if threshold_tuning_history[i]:
        print("threshold tuning triggered at {}".format(i))

    accumulated_acc.append(100*sum(curr_accuracy)/(max_idx+0.0))
    accumulated_latency_improvement.append(100 * (vanilla_latency - sum(curr_latency) / len(curr_latency)) / vanilla_latency)

    curr_batch_latency = all_latencies[idx*bs: max_idx]
    curr_batch_accuracy = all_accuracies[idx*bs: max_idx]

    print(sum(curr_batch_accuracy)/(max_idx-idx*bs))
    print(idx*bs, max_idx, len(curr_batch_accuracy))

    if sum(curr_batch_accuracy)/(max_idx-idx*bs) < 0.985:
        batch_ids_violate_acc.append((i, sum(curr_batch_accuracy)/(max_idx-idx*bs)))

    individual_acc.append(100*sum(curr_batch_accuracy)/(len(curr_batch_accuracy)+0.0))
    individual_latency_improvement.append(100 * (vanilla_latency - sum(curr_batch_latency) / len(curr_batch_latency)) / vanilla_latency)

    idx += 1
    
print("violate", len(batch_ids_violate_acc), batch_ids_violate_acc, len(batch_ids_violate_acc) / float(len(threshold_tuning_history)))

def generate_plot(l, titles, yaxis_range, filename=None):
    plt.rcParams["figure.figsize"] = (20, 5)
    plt.rcParams["font.size"] = 15
    fig = plt.figure()
    ax = fig.add_subplot(111)

    linestyles = ["solid", "solid", "dashed"]
    colors = ["blue", "orange", "red"]
    for l_sub in l:
        t = titles.pop(0)
        plt.plot(l_sub, label=t, color=colors.pop(0), linestyle=linestyles.pop(0))

    # for i in range(len(threshold_tuning_history)):
    #     if threshold_tuning_history[i]:
    #         tune_batch = matplotlib.patches.Rectangle((i-0.5, 0), 1, 60, color='#88acff')
    #         ax.add_patch(tune_batch)

        plt.legend(loc="lower left")
        plt.xlabel(f"batch id")
        plt.ylabel(t)

        # plt.xticks(list(range(0, 51)))
        plt.yticks(yaxis_range)
        # plt.title(f"Ramps {' '.join([str(i) for i in ramp_ids])}")
        plt.grid(color='grey', linewidth=0.5)
        # filename = t + ".png"
    dpi = 400
    plt.savefig(filename+'.png', dpi=dpi)
    plt.close()

# generate_plot([accumulated_acc], ["accumulated_accuracy"], list(range(98, 100)))
# generate_plot([accumulated_latency_improvement], ["accumulated_latency_improvement"], list(range(20, 30)))

# generate_plot([individual_acc],["accuracy"], list(range(90, 100)))
# generate_plot([individual_latency_improvement], ["latency_improvement"], list(range(10, 30)))


print("----------------------------------")


arch = "resnet18_urban"
idx = 1
pickle_path = os.path.join("./", "{}_plot_{}.pickle".format(arch, str(idx)))

with open(pickle_path, "rb") as f:
    all_latencies_1, all_accuracies_1, all_exit_ramp_1, threshold_tuning_history_1, vanilla_latency_1 = pickle.load(f)

accumulated_acc_1 = []
accumulated_latency_improvement_1 = []

individual_acc_1 = []
individual_latency_improvement_1 = []

batch_ids_violate_acc_1 = []

idx = 0
for i in range(len(threshold_tuning_history_1)):
    max_idx = min((idx+1)*bs, len(all_latencies_1))
    curr_latency = all_latencies_1[: max_idx]
    curr_accuracy = all_accuracies_1[: max_idx]
    curr_exit_ramp = all_exit_ramp_1[: max_idx]
    curr_threshold_tuning_history = threshold_tuning_history_1[: max_idx]
    # print(sum(curr_accuracy)/(max_idx+0.0))
    
    if threshold_tuning_history_1[i]:
        print("threshold tuning triggered at {}".format(i))

    accumulated_acc_1.append(100*sum(curr_accuracy)/(max_idx+0.0))
    accumulated_latency_improvement_1.append(100 * (vanilla_latency_1 - sum(curr_latency) / len(curr_latency)) / vanilla_latency_1)

    curr_batch_latency = all_latencies_1[idx*bs: max_idx]
    curr_batch_accuracy = all_accuracies_1[idx*bs: max_idx]

    print(sum(curr_batch_accuracy)/(max_idx-idx*bs))


    if sum(curr_batch_accuracy)/(max_idx-idx*bs) < 0.985:
        batch_ids_violate_acc_1.append((i, sum(curr_batch_accuracy)/(max_idx-idx*bs)))

    individual_acc_1.append(100*sum(curr_batch_accuracy)/(len(curr_batch_accuracy)+0.0))
    individual_latency_improvement_1.append(100 * (vanilla_latency_1 - sum(curr_batch_latency) / len(curr_batch_latency)) / vanilla_latency_1)

    idx += 1

counter = 0
for batch_id, acc in batch_ids_violate_acc:
    print(batch_id, 100*acc, individual_acc_1[batch_id])
    if 100*acc <= individual_acc_1[batch_id]:
        counter += 1
print(counter)


generate_plot([[100*m[1] for m in batch_ids_violate_acc], [individual_acc_1[m[0]] for m in batch_ids_violate_acc]], ['without tuning', 'with tuning'] , list(range(90, 100)), filename="w_tuning")