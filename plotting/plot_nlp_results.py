# %%
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from plotting_utils import parse_result_file, values_to_cdf

# NOTE(ruipan): use these for the final version. use the abs path here for testing.
# BATCH_DECISION_PATH = "../../batch_decisions/{model}_{arrival}.pickle"
# APPARATE_LATENCY_PATH = "../../apparate_latency/{model}_{dataset}_{arrival}.pickle"
# OPTIMAL_LATENCY_PATH = "../../optimal_latency/{model}_{dataset}_{arrival}_optimal.pickle"
BATCH_DECISION_PATH = "/home/ruipan/apparate-ae/data/batch_decisions/{model}_{arrival}.pickle"
APPARATE_LATENCY_PATH = "/home/ruipan/apparate-ae/data/apparate_latency/{model}_{dataset}_{arrival}.pickle"
OPTIMAL_LATENCY_PATH = "/home/ruipan/apparate-ae/data/optimal_latency/{model}_{dataset}_optimal.pickle"


matplotlib.rcParams["figure.figsize"] = (4, 2)  # (4, 1.3)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

datasets = ["amazon_reviews", "imdb"]
models = ["distilbert-base", "bert-base", "bert-large", "gpt2-medium"]


for model in models:
    fig, ax = plt.subplots()
    dataset_colors = ["tab:green", "tab:red"]
    dataset_linestyles = ["dashed", "dotted", ]
    for dataset in datasets:  
        dataset_label_name = {
            "amazon_reviews": "Amazon",
            "imdb": "IMDB",
        }[dataset]
        dataset_color = dataset_colors.pop(0)
        dataset_linestyle = dataset_linestyles.pop(0)
        results = parse_result_file(
            model.lower(),
            dataset.lower(),
            slo_multiplier=None,
            arrival="azure",
            BATCH_DECISION_PATH=BATCH_DECISION_PATH,
            APPARATE_LATENCY_PATH=APPARATE_LATENCY_PATH,
            OPTIMAL_LATENCY_PATH=OPTIMAL_LATENCY_PATH,
        )
        serving_time_ee = results["serving_time_ee"]
        serving_time_vanilla = results["serving_time_vanilla"]

        print("="*50)
        print(f"model {model} dataset {dataset}")

        print(f"50 percentile win {np.percentile(serving_time_vanilla, 50) - np.percentile(serving_time_ee, 50)}ms")
        print(f"25 percentile win {np.percentile(serving_time_vanilla, 25) - np.percentile(serving_time_ee, 25)}ms")

        ax.plot(
            serving_time_ee,
            values_to_cdf(serving_time_ee),
            label=f"{dataset_label_name}",
            color={
                "amazon_reviews": "tab:green",
                "imdb": "tab:red",                
            }[dataset],
            linestyle=dataset_linestyle,
        )
        ax.plot(
            serving_time_vanilla,
            values_to_cdf(serving_time_vanilla),
            label=f"{dataset_label_name}-V",
            color="tab:blue",
            # linestyle="solid",
            linestyle=dataset_linestyle,
        )

    ax.set_xlabel(f"Latency (ms)", fontsize=15)
    ax.set_ylabel(f"CDF", fontsize=15)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    
    ax.legend(loc="upper left", ncols=1, fontsize=10)
    ax.set_axisbelow(True)  # puts the grid behind the bars
    ax.grid(color='lightgrey', linestyle='dashed', axis="both", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(f'./nlp_results_{model.lower()}.pdf', bbox_inches='tight', dpi=500)  
    plt.close()
# %%
