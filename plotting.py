import pickle
from statistics import mean
from itertools import product
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


def values_to_cdf(values):
    """Turns a list of values into a list of cumulative probabilities
    for easier plotting. Note that the values will be sorted
    in-place in this function.

    Args:
        values (list): list of values

    Returns:
        list: list of cumulative probabilities (0-1)
    """
    cdf_list = []
    # values.sort()  # in-place sort
    count = 0
    for v in values:
        count += 1
        cdf_list.append(count / len(values))
    return cdf_list



def plot_latency_cdf(all_latencies, vanilla_latency: float):
    fig, ax = plt.subplots()
    
    total_latencies = [x[0] + x[1] for x in all_latencies]
    queueing_delays = [x[0] for x in all_latencies]
    inference_latencies = [x[1] for x in all_latencies]
    vanilla_latencies = [x + vanilla_latency for x in queueing_delays]  # Hmm... 

    ax.plot(
        total_latencies,
        values_to_cdf(total_latencies),
        label="total_latencies"
    )
    ax.plot(
        queueing_delays,
        values_to_cdf(queueing_delays),
        label="queueing_delays"
    )
    ax.plot(
        inference_latencies,
        values_to_cdf(inference_latencies),
        label="inference_latencies"
    )
    ax.plot(
        vanilla_latencies,
        values_to_cdf(vanilla_latencies),
        label="vanilla_latencies"
    )
    ax.axvline(x=vanilla_latency, color="red", linestyle="--")
    
    ax.set_ylabel("Cumulative probability")
    ax.set_xlabel("Inference latency (ms)")
    ax.set_yticks([x / 100 for x in list(range(0, 110, 10))])
    ax.legend()
    ax.set_axisbelow(True)  # puts the grid below the bars
    ax.grid(color='lightgrey', linestyle='dashed', axis="both")
    fig.savefig(f"latency_cdf.png", dpi=200)
    plt.close()


def plot_latency_cdf_different_slo(
    dataset,
    queueing_delay_list,
    inference_time_list,
    latency_list,
    ee_latency_list,
    optimal_apparate_latency_list,
    optimal_ee_latency_list,
    batching_scheme,
    slo,
    fixed_arrival_rate,
    interarrival_time,
    avg_qps,
    arch,
    drop_rate,
):
    fig, ax = plt.subplots()
    # ax.plot(
    #     queueing_delay_list,
    #     values_to_cdf(queueing_delay_list),
    #     label="queueing delay"
    # )
    # ax.plot(
    #     inference_time_list,
    #     values_to_cdf(inference_time_list),
    #     label="model inference"
    # )
    if ee_latency_list is not None:
        ee_latency_list = sorted(ee_latency_list)
        ax.plot(
            ee_latency_list,
            values_to_cdf(ee_latency_list),
            label="apparate"
        )
    if optimal_ee_latency_list is not None:
        optimal_ee_latency_list = sorted(optimal_ee_latency_list)
        ax.plot(
            optimal_ee_latency_list,
            values_to_cdf(optimal_ee_latency_list),
            label="optimal ee"
        )
    if optimal_apparate_latency_list is not None:
        optimal_apparate_latency_list = sorted(optimal_apparate_latency_list)
        ax.plot(
            optimal_apparate_latency_list,
            values_to_cdf(optimal_apparate_latency_list),
            label="optimal apparate"
        )

    latency_list = sorted(latency_list)
    ax.plot(
        latency_list,
        values_to_cdf(latency_list),
        label="vanilla model"
    )

    # ax.axvline(x=slo, color="red", linestyle="--", label="SLO")
    # ax.axvline(x=lower_bound, color="gray", linestyle="--", label="minimum latency")
    
    ax.set_ylabel("Cumulative probability")
    ax.set_xlabel("Inference latency (ms)")
    ax.set_yticks([x / 100 for x in list(range(0, 110, 10))])
    ax.legend(loc='upper left')
    if fixed_arrival_rate:
        description = f"fixed interarrival {round(interarrival_time, 3)} ms"
        filename = f"./plots/{batching_scheme}_{arch.split('_')[0]}_{dataset}_{slo}_fixed_{round(avg_qps)}.png"
    else:
        description = f"azure trace, avg qps {round(avg_qps)}"
        filename = f"./plots/{batching_scheme}_{arch.split('_')[0]}_{dataset}_{slo}_azure_{round(avg_qps)}.png"
        
    ax.set_title(f"{batching_scheme} on {arch.split('_')[0]} and {dataset} with azure trace")
    ax.set_axisbelow(True)  # puts the grid below the bars
    ax.grid(color='lightgrey', linestyle='dashed', axis="both")
    # fig.savefig(f"./latency_motivation.png", dpi=200)
    fig.savefig(filename, dpi=200)
    plt.close()
    


def plot_latency_savings_comparison(
    model: str,
    dataset: str,
    by_hardness: bool,
    num_samples_in_subdataset: int,
    data_dict: dict,
):
    fig, ax = plt.subplots()
    line_colors = ["c", "m", "y", "tab:gray"]
        
    for scheme_name, metrics_every_dataset in data_dict.items():
        num_subdatasets = len(metrics_every_dataset)
        latency_savings_every_dataset = [x[0] for x in metrics_every_dataset]
        acc_every_dataset = [x[1] for x in metrics_every_dataset]
        ax.plot(
            list(range(num_subdatasets)),
            latency_savings_every_dataset,
            label=scheme_name,
            color=line_colors.pop(0)
            # marker="o",
        )
        for subdataset_id, acc in enumerate(acc_every_dataset):
            marker, color = ("o", "tab:green") if acc >= 0.985 else ("X", "tab:red")
            ax.plot(subdataset_id, latency_savings_every_dataset[subdataset_id], marker=marker, color=color)
    
    ax.set_ylabel("Latency savings (%)")
    ax.set_xlabel("Subdataset ID")
    ax.set_xticks(list(range(num_subdatasets)))
    # model + dataset + by hardness
    ax.set_title(f"{model}+{dataset}, by_hardness {by_hardness},\nnum samples in subdataset {num_samples_in_subdataset}")
    ax.legend()
    ax.set_axisbelow(True)  # puts the grid below the bars
    ax.grid(color='lightgrey', linestyle='dashed', axis="both")
    fig.savefig(f"./plots/latency_savings_comparison/{model}_{dataset}_{by_hardness}.png", dpi=200)
    plt.clf()
    plt.close()


def plot_optimal_ramps(
    model: str,
    dataset: str,
    by_hardness: bool,
    num_samples_in_subdataset: int,
    data_list: list,
    latency_savings_list: list = None
):
    num_dp = len(data_list)
    # matplotlib.rcParams["figure.figsize"] = (num_dp / 3, 4)
    fig, ax = plt.subplots()
        
    for subdataset_id, (optimal_ramp_ids, _, _, _, _) in enumerate(data_list):
        for ramp_id in optimal_ramp_ids:
            ax.plot(
                subdataset_id,
                ramp_id,
                color="tab:blue",
                marker="o",
            )
    
    if latency_savings_list is not None:
        ax1 = ax.twinx()  # instantiate a second axes that shares the same x-axis
        ax1.set_ylabel('Latency savings (%)', color='tab:orange')  # we already handled the x-label with ax1
        trend_list = []
        for i in range(len(latency_savings_list)):
            try:
                trend_list.append(latency_savings_list[i-2] - latency_savings_list[i-1])
            except IndexError:
                trend_list.append(0)
        ax1.plot(list(range(len(latency_savings_list))), trend_list, label="Latency savings (%)", color='tab:orange')
        ax1.tick_params(axis='y', labelcolor='tab:orange')
        ax1.axhline(y=0, color="red")
    
    ax.set_ylabel("Optimal ramp IDs")
    ax.set_xlabel("Subdataset ID")
    num_ramps = {
        "distilbert-base-uncased": 6,
        "bert-base-uncased": 12,
        "bert-large-uncased": 24,
        "resnet18_urban": 8,
        "resnet18_waymo": 8,
        "resnet50_urban": 16,
        "resnet50_waymo": 16,
    }[model]
    ax.set_yticks(list(range(num_ramps)))
    ax.set_xticks(list(range(len(data_list))))
    if latency_savings_list is not None:
        ax1.legend()
    # model + dataset + by hardness
    ax.set_title(f"{model}+{dataset}, by_hardness {by_hardness},\nnum samples in subdataset {num_samples_in_subdataset}")
    ax.set_axisbelow(True)  # puts the grid below the bars
    ax.grid(color='lightgrey', linestyle='dashed', axis="both")
    fig.savefig(f"./plots/optimal_ramps/{model}_{dataset}_{by_hardness}.png", dpi=200)
    plt.clf()
    plt.close()
    return



def plot_avg_exit_point_intuition(
    model: str,
    dataset: str,
    by_hardness: bool,
    num_samples_in_subdataset: int,
    avg_exit_point_list: list,
    fixed_ramp_latency_savings: list,
):
    fig, ax = plt.subplots()

    ax.plot(
        list(range(len(avg_exit_point_list))),
        [x[0] for x in avg_exit_point_list],
        color="tab:blue",
        marker="o",
        label="optimal exiting"
    )
    ax.plot(
        list(range(len(avg_exit_point_list))),
        [x[1] for x in avg_exit_point_list],
        color="tab:green",
        marker="o",
        label="approximation"
    )
    ax.set_ylabel(f"Average exit point", color="tab:blue")
    ax.tick_params(axis='y', labelcolor='tab:blue')
    
    ax1 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax1.set_ylabel('Optimal latency savings (ramp fixed)', color='tab:orange')  # we already handled the x-label with ax1
    ax1.plot(list(range(len(fixed_ramp_latency_savings))), fixed_ramp_latency_savings, color='tab:orange')
    ax1.tick_params(axis='y', labelcolor='tab:orange')
    
    ax.set_xlabel("Subdataset ID")
    ax.set_xticks(list(range(len(avg_exit_point_list))))
    ax.legend()
    # model + dataset + by hardness
    ax.set_title(f"{model}+{dataset}, by_hardness {by_hardness},\nnum samples in subdataset {num_samples_in_subdataset}")
    ax.set_axisbelow(True)  # puts the grid below the bars
    ax.grid(color='lightgrey', linestyle='dashed', axis="both")
    fig.savefig(f"./plots/avg_exit_point_intuition/{model}_{dataset}_{by_hardness}.png", dpi=200)
    plt.clf()
    plt.close()
    return



def plot_batch_size_latencies(latencies_per_batch_size: dict):
    fig, ax = plt.subplots()
    
    model_latencies = [x[0] for x in latencies_per_batch_size.values()]
    ramp_latencies = [x[1] for x in latencies_per_batch_size.values()]
        
    ax.plot(
        latencies_per_batch_size.keys(),
        model_latencies,
        label="whole_model",
        marker="o",
        color="tab:blue",
    )
    ax.plot(
        latencies_per_batch_size.keys(),
        ramp_latencies,
        label="first_ramp",
        marker="o",
        color="tab:orange",
    )
    
    # ax.set_yscale("log")
    ax.set_ylabel("Latency (ms)")
    ax.set_xlabel("Batch size")
    # ax.set_xticks(list(latencies_per_batch_size.keys()))
    # ax.set_xticks(list(range(0, 35, 5)))
    ax.set_xticks([1, 2, 4, 8, 16, 32])
    # ax.set_xticks([1, 2, 4, 8, 16, 32, 64])
    # model + dataset + by hardness
    ax.legend()
    ax.set_axisbelow(True)  # puts the grid below the bars
    ax.grid(color='lightgrey', linestyle='dashed', axis="both")
    fig.savefig(f"./plots/batch_size_latencies.png", dpi=200)
    plt.clf()
    plt.close()
    return

def plot_different_bootstrap_methods(latency_savings_different_bootstrap_schemes: list):
    fig, ax = plt.subplots()
    
    m1_savings = mean([mdp[0][0] for mdp in latency_savings_different_bootstrap_schemes])
    m2_savings = mean([mdp[1][0] for mdp in latency_savings_different_bootstrap_schemes])
    m3_savings = mean([mdp[2][0] for mdp in latency_savings_different_bootstrap_schemes])
    m4_savings = mean([mdp[3][0] for mdp in latency_savings_different_bootstrap_schemes])
    
    bottom, top = min([m1_savings, m2_savings, m3_savings, m4_savings]), max([m1_savings, m2_savings, m3_savings, m4_savings])
    
    ax.bar(0, m1_savings)
    ax.bar(1, m2_savings)
    ax.bar(2, m3_savings)
    ax.bar(3, m4_savings)
    
    ax.set_ylim(bottom - 0.5, top + 0.5)
    ax.set_ylabel("Avg latency savings (%)")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Method1", "Method2", "Method3", "Method4"])
    
    fig.savefig(f"./plots/bootstrap_methods_savings.png", dpi=200)
    plt.clf()
    plt.close()
    
    fig, ax = plt.subplots()
    
    m1_ramps, m2_ramps, m3_ramps, m4_ramps = [], [], [], []
    
    for mdp in latency_savings_different_bootstrap_schemes:
        num_ramps_remaining = [len(scheme[2]) for scheme in mdp]
        num_ramps_in_model = {
            "bert-base-uncased": 12,
            "bert-large-uncased": 24,
            "distilbert-base-uncased": 6,
        }[mdp[0][1]]
        num_ramps_remaining = [x / num_ramps_in_model * 100 for x in num_ramps_remaining]
        m1_ramps.append(num_ramps_remaining[0])
        m2_ramps.append(num_ramps_remaining[1])
        m3_ramps.append(num_ramps_remaining[2])
        m4_ramps.append(num_ramps_remaining[3])
    
    m1_ramps, m2_ramps, m3_ramps, m4_ramps = mean(m1_ramps), mean(m2_ramps), mean(m3_ramps), mean(m4_ramps)
    bottom, top = min([m1_ramps, m2_ramps, m3_ramps, m4_ramps]), max([m1_ramps, m2_ramps, m3_ramps, m4_ramps])
    
    ax.bar(0, m1_ramps)
    ax.bar(1, m2_ramps)
    ax.bar(2, m3_ramps)
    ax.bar(3, m4_ramps)
    
    ax.set_ylim(bottom - 5, top + 5)
    ax.set_ylabel("Avg ramp activation proportion (%)")
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["Method1", "Method2", "Method3", "Method4"])
    
    fig.savefig(f"./plots/bootstrap_methods_ramps.png", dpi=200)
    plt.clf()
    plt.close()

def plot_entropy_visualization(model, dataset, entropy_dict):
    fig, ax = plt.subplots()
    
    linewidth_dict = {
        "mrpc": 0.1,  # ~500
        "rte": 0.1,  # ~300
        "sst-2": 0.1,  # ~900
        "qnli": 0.02,  # ~5500
        "urban": 0.02,
        "qqp": 0.005,  # ~40000
        "mnli": 0.01,  # ~10000
        "mnli-mm": 0.01  # ~10000
    }
    
    ramp_entropies = entropy_dict["conf"][:-1]
    all_ramp_ids = list(range(len(ramp_entropies)))
    active_ramp_ids = [i for i, x in enumerate(ramp_entropies) if x != []]
    num_samples = len(ramp_entropies[active_ramp_ids[0]])
    print(f"Plotting {model}+{dataset}, num_samples {num_samples}")
    print(f"all_ramp_ids {all_ramp_ids}, active_ramp_ids {active_ramp_ids}")
    for i in range(num_samples):
        entropies_across_ramps = [
            1 - ramp_entropies[id][i] for id in active_ramp_ids
        ]
        ax.plot(active_ramp_ids, entropies_across_ramps, color="red", linewidth=linewidth_dict[dataset])
        if dataset in ["urban", "qqp"] and i > 5000:
            break
        
        print(f"entropies_across_ramps {entropies_across_ramps}")
    
    ax.set_xticks(active_ramp_ids)
    ax.set_xticklabels(active_ramp_ids)
    ax.set_ylabel("Entropy (lower: more confident)")
    ax.set_xlabel("Ramp IDs")
    ax.set_axisbelow(True)
    ax.grid(color='grey', linewidth=0.08)
    
    fig.savefig(f"./plots/entropies_visualization/{dataset}_{model}.png", dpi=200)
    plt.clf()
    plt.close()
        
    return



def plot_all_combo_avg_exit_points(
    model: str,
    dataset: str,
    all_combo_avg_exit_points: list,
):
    fig, ax = plt.subplots()
    
    for avg_exit_point, latency_improvement in all_combo_avg_exit_points:
        ax.scatter(
            latency_improvement,
            avg_exit_point,
            color="tab:blue",
            marker="o",
            s=0.3,  # marker size
        )
    
    ax.set_ylabel("Average exit point")
    ax.set_xlabel("Latency improvement of config (%)")
    ax.set_title(f"{model}+{dataset}")
    ax.set_axisbelow(True)
    ax.grid(color='grey', linewidth=0.08)
    
    fig.savefig(f"./plots/all_combo_avg_exit_points/{dataset}_{model}.png", dpi=200)
    plt.clf()
    plt.close()
    

def plot_latency_savings_over_batches(
    model: str,
    dataset: str,
    avg_savings_dict: dict,
):
    num_dp = len(avg_savings_dict["nochange"])
    # matplotlib.rcParams["figure.figsize"] = (num_dp / 3, 4)
    fig, ax = plt.subplots()
        
    for method, avg_savings in avg_savings_dict.items():
        ax.plot(
            list(range(len(avg_savings))),
            avg_savings,
            marker="o",
            label=method
        )
    
    ax.set_ylabel("Average latency savings (%)")
    ax.set_xlabel("Batch ID (x10)")
    ax.set_xticks(list(range(len(avg_savings))))
    ax.legend()
    # model + dataset + by hardness
    ax.set_title(f"{model}+{dataset}")
    ax.set_axisbelow(True)  # puts the grid below the bars
    ax.grid(color='lightgrey', linestyle='dashed', axis="both")
    fig.savefig(f"../plots/savings_over_batches/{model}_{dataset}.png", dpi=200)
    plt.clf()
    plt.close()
    return


def plot_latency_savings_comparison_different_addition_policies(
    model: str,
    dataset: str,
    by_hardness: bool,
    num_samples_in_subdataset: int,
    data_list: list,
):
    num_dp = len(data_list)
    matplotlib.rcParams["figure.figsize"] = (num_dp / 3, 4)
    
    fig, ax = plt.subplots()
    line_colors = ["c", "m", "y", "tab:gray"]
    
    data_list_by_scheme = [
        [data_list[i][0] for i in range(len(data_list))],
        [data_list[i][1] for i in range(len(data_list))],
        [data_list[i][2] for i in range(len(data_list))],
        [data_list[i][3] for i in range(len(data_list))],
    ]
    scheme_names = ["nochange", "rampaddition", "onestep", "startavg"]
    
    for i, x in enumerate(data_list_by_scheme):
        print(f"{scheme_names[i]}: avg {np.average(x)}")
        
    for i, metrics_every_dataset in enumerate(data_list_by_scheme):
        num_subdatasets = len(metrics_every_dataset)
        # latency_savings_every_dataset = [x[0] for x in metrics_every_dataset]
        # acc_every_dataset = [x[1] for x in metrics_every_dataset]
        ax.plot(
            list(range(num_subdatasets)),
            metrics_every_dataset,
            label=scheme_names[i],
            color=line_colors.pop(0)
            # marker="o",
        )
        # for subdataset_id, acc in enumerate(acc_every_dataset):
        #     marker, color = ("o", "tab:green") if acc >= 0.985 else ("X", "tab:red")
        #     ax.plot(subdataset_id, latency_savings_every_dataset[subdataset_id], marker=marker, color=color)
    
    ax.set_ylabel("Latency savings (%)")
    ax.set_xlabel("Batch ID")
    ax.set_xticks(list(range(num_subdatasets)))
    # model + dataset + by hardness
    ax.set_title(f"{model}+{dataset}, by_hardness {by_hardness},\nbatch size {num_samples_in_subdataset}")
    ax.legend()
    ax.set_axisbelow(True)  # puts the grid below the bars
    ax.grid(color='lightgrey', linestyle='dashed', axis="both")
    fig.savefig(f"./plots/plot_latency_savings_comparison_different_addition_policies/{model}_{dataset}_{by_hardness}.png", dpi=200)
    plt.clf()
    plt.close()


def analyze_azure_trace(df):
    """
    Plots the request arrival pattern for each filtered function in an azure trace
    """
    for i, row in df.iterrows():
        sum = row.sum()
        avg_qps = int(sum / (24 * 60 * 60))
        print(f"plotting function {i}, avg_qps {avg_qps}")
        
        row = [x / 60 for x in row]  # queries per minute to qps
        
        matplotlib.rcParams["figure.figsize"] = (20, 5)
        fig, ax = plt.subplots()
        ax.axhline(y=avg_qps, color="red", linestyle="dashed")
        
        ax.plot(
            list(range(len(row))),
            row,
        )
        
        ax.set_ylabel("QPS")
        ax.set_xlabel("Minute")
        ax.set_title(f"function {i}, avg_qps {avg_qps}")
        ax.set_axisbelow(True)  # puts the grid below the bars
        ax.grid(color='lightgrey', linestyle='dashed', axis="both")
        fig.savefig(f"../plots/analyze_azure_trace/{i}_{avg_qps}.png", dpi=200)
        plt.clf()
        plt.close()
    



if __name__ == "__main__":
    # models, datasets = ["bert-base-uncased", "bert-large-uncased", "distilbert-base-uncased"], ["rte", "mrpc", "sst-2", "mnli", "qnli", "qqp"]
    # models, datasets = ["bert-large-uncased"], ["mnli", "qqp"]  # broken
    models, datasets = ["resnet18"], ["urban"]  # broken
    model_dataset_pairs = product(models, datasets)
    for (model, dataset) in model_dataset_pairs:
        with open(f"./entropy_pickles/{dataset}_{model}_entropies.pickle", "rb") as f:
            p = pickle.load(f)
        plot_entropy_visualization(model, dataset, p)
