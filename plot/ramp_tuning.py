import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce


matplotlib.rcParams["figure.figsize"] = (8, 3)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


config_every_epoch = {
    "rte": [
        (0.22588894064671416, [6], [0.4125]),  # best savings, best ramp IDs, and their associated thresholds
        (0.2756686464034782, [3, 6], [0.1125, 0.4]),
        (0.3310042312022049, [3, 6, 8], [0.2375, 0.075, 0.4]),
        (0.342610923488995, [3, 5, 6, 8], [0.2375, 0.075, 0.1125, 0.4]),
        (0.35138872714568536, [3, 5, 6, 8, 9], [0.2375, 0.075, 0.075, 0.1125, 0.4]),
        (0.3492003416016459, [3, 5, 6, 8, 9, 11], [0.2375, 0.075, 0.075, 0.1125, 0.1375, 0.4]),
        (0.34502736695004066, [3, 4, 5, 6, 8, 9, 11], [0.2125, 0.075, 0.075, 0.1875, 0.1375, 0.2125, 0.425]),
        (0.34017992313962975, [3, 4, 5, 6, 7, 8, 9, 11], [0.2125, 0.0625, 0.075, 0.075, 0.1875, 0.1375, 0.2125, 0.425]),
        (0.33273164861612503, [3, 4, 5, 6, 7, 8, 9, 10, 11], [0.2125, 0.0625, 0.075, 0.075, 0.05, 0.1875, 0.1375, 0.2125, 0.425])
    ],
    "qnli": [
        (0.2214100626108385, [4], ),  # 1: curr_ramp_ids [4], thresholds [0.4125]
        (0.2287834680617803, [4, 6], ),  # 2: curr_ramp_ids [4, 6], thresholds [0.075, 0.35]
        (0.22492564839910323, [4, 6, 7], ),  # 3: curr_ramp_ids [4, 6, 7], thresholds [0.0625, 0.0375, 0.375]
        (0.2199867535069746, [3, 4, 6, 7], ),  # 4: curr_ramp_ids [3, 4, 6, 7], thresholds [0.05, 0.05, 0.0375, 0.35]
        (0.22020744637925915, [3, 4, 5, 6, 7], ),  # 5: curr_ramp_ids [3, 4, 5, 6, 7], thresholds [0.05, 0.05, 0.025, 0.0375, 0.3625]
        (0.21292728797245997, [3, 4, 5, 6, 7, 10], ),  # 6: curr_ramp_ids [3, 4, 5, 6, 7, 10], thresholds [0.0375, 0.0375, 0.0375, 0.05, 0.0375, 0.3625]
        (0.21335932440917305, [3, 4, 5, 6, 7, 10, 11], ),  # 7: curr_ramp_ids [3, 4, 5, 6, 7, 10, 11], thresholds [0.025, 0.0375, 0.0375, 0.025, 0.0375, 0.0125, 0.375]
        (0.20116241420779868, [2, 3, 4, 5, 6, 7, 10, 11], ),  # 8: curr_ramp_ids [2, 3, 4, 5, 6, 7, 10, 11], thresholds [0.025, 0.0375, 0.0375, 0.025, 0.0125, 0.0, 0.0125, 0.4]
        (0.19330309078276342, [2, 3, 4, 5, 6, 7, 9, 10, 11], ),  # 9: curr_ramp_ids [2, 3, 4, 5, 6, 7, 9, 10, 11], thresholds [0.075, 0.025, 0.0375, 0.0375, 0.025, 0.0375, 0.0, 0.0125, 0.375]
        # No more ramps can be added without overflowing the accuracy loss budget, stopping
    ], 
    "qqp": [
        (0.3247978718028506, [4], ),  # 1: curr_ramp_ids [4], thresholds [0.45]
        (0.34258565182354217, [3, 4], ),  # 2: curr_ramp_ids [3, 4], thresholds [0.1, 0.425]
        (0.3624489360344044, [3, 4, 10], ),  # 3: curr_ramp_ids [3, 4, 10], thresholds [0.05, 0.1, 0.4125]
        (0.36664472511894997, [3, 4, 7, 10], ),  # 4: curr_ramp_ids [3, 4, 7, 10], thresholds [0.0875, 0.0875, 0.0125, 0.425]
        (0.3650371410562264, [3, 4, 7, 10, 11], ),  # 5: curr_ramp_ids [3, 4, 7, 10, 11], thresholds [0.0875, 0.0875, 0.0125, 0.0125, 0.4125]
        (0.36307111189125496, [3, 4, 6, 7, 10, 11], ),  # 6: curr_ramp_ids [3, 4, 6, 7, 10, 11], thresholds [0.075, 0.075, 0.025, 0.0125, 0.025, 0.4125]
        (0.362943950648805, [3, 4, 6, 7, 8, 10, 11], ),  # 7: curr_ramp_ids [3, 4, 6, 7, 8, 10, 11], thresholds [0.075, 0.075, 0.025, 0.025, 0.0, 0.05, 0.425]
        (0.35847016880364035, [3, 4, 5, 6, 7, 8, 10, 11], ),  # 8: curr_ramp_ids [3, 4, 5, 6, 7, 8, 10, 11], thresholds [0.075, 0.075, 0.025, 0.025, 0.0125, 0.0, 0.0375, 0.4125]
        (0.3534861462397505, [3, 4, 5, 6, 7, 8, 9, 10, 11], ),  # 9: curr_ramp_ids [3, 4, 5, 6, 7, 8, 9, 10, 11], thresholds [0.075, 0.075, 0.025, 0.025, 0.025, 0.0125, 0.0, 0.025, 0.425]
        (0.3444674387431882, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ),  # 10: curr_ramp_ids [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], thresholds [0.0375, 0.0875, 0.0375, 0.0375, 0.0375, 0.025, 0.0125, 0.0125, 0.0125, 0.425]
        (0.3332063050699604, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ),  # 11: curr_ramp_ids [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], thresholds [0.025, 0.0375, 0.0875, 0.0375, 0.0375, 0.0375, 0.025, 0.0125, 0.0125, 0.0125, 0.4125]
        # No more ramps can be added without overflowing the accuracy loss budget, stopping
    ],
    "mnli": [
        (0.1517288931249624, [7], ),  # 1: curr_ramp_ids [7], thresholds [0.5625]
        (0.17589368368582226, [5, 7], ),  # 2: curr_ramp_ids [5, 7], thresholds [0.075, 0.5375]
        (0.1888123839416298, [5, 7, 10], ),  # 3: curr_ramp_ids [5, 7, 10], thresholds [0.075, 0.05, 0.55]
        (0.19221457172749645, [5, 7, 9, 10], ),  # 4: curr_ramp_ids [5, 7, 9, 10], thresholds [0.0625, 0.05, 0.0375, 0.55]
        (0.1893479642197866, [5, 7, 9, 10, 11], ),  # 5: curr_ramp_ids [5, 7, 9, 10, 11], thresholds [0.075, 0.0375, 0.0125, 0.025, 0.5375]
        (0.17972723886524355, [5, 6, 7, 9, 10, 11], ),  # 6: curr_ramp_ids [5, 6, 7, 9, 10, 11], thresholds [0.05, 0.05, 0.0125, 0.0125, 0.025, 0.55]
        (0.17235619717461215, [5, 6, 7, 8, 9, 10, 11], ),  # 7: curr_ramp_ids [5, 6, 7, 8, 9, 10, 11], thresholds [0.0375, 0.0375, 0.0375, 0.025, 0.0125, 0.0375, 0.5625]
        (0.15857722708822908, [3, 5, 6, 7, 8, 9, 10, 11], ),  # 8: curr_ramp_ids [3, 5, 6, 7, 8, 9, 10, 11], thresholds [0.0625, 0.0375, 0.0375, 0.0375, 0.0125, 0.0125, 0.0375, 0.5625]
        (0.14260636287446826, [3, 4, 5, 6, 7, 8, 9, 10, 11], ),  # 9: curr_ramp_ids [3, 4, 5, 6, 7, 8, 9, 10, 11], thresholds [0.0875, 0.0625, 0.0375, 0.0375, 0.0375, 0.0125, 0.0125, 0.0375, 0.5625]
        (0.1250523118553456, [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], ),  # 10: curr_ramp_ids [2, 3, 4, 5, 6, 7, 8, 9, 10, 11], thresholds [0.15, 0.075, 0.05, 0.0375, 0.0375, 0.0375, 0.0125, 0.0125, 0.0375, 0.55]
        # No more ramps can be added without overflowing the accuracy loss budget, stopping
    ]
}

x_index = list(range(1, 13))

for i, expr_name in enumerate([
    "latency_saving_as_more_ramps_are_added",
    "ramp_addition_ordering",
]):
    fig, ax = plt.subplots()

    # search for "format strings" in 
    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
    for dataset in config_every_epoch.keys():
        if i == 0:
            data = [100 * x[0] for x in config_every_epoch[dataset]]
            ylabel = f"Latency savings (%)"
            ax.set_yticks(list(range(15, 40, 5)))
            # max_y, min_y = max(savings_by_num_ramps), min(savings_by_num_ramps)
        elif i == 1:
            ramps_added_every_epoch = config_every_epoch[dataset][0][1]
            for epoch_data in config_every_epoch[dataset][1:]:
                print(epoch_data[1], ramps_added_every_epoch)
                new_ramp = set(epoch_data[1]) - set(ramps_added_every_epoch)
                ramps_added_every_epoch.append(list(new_ramp)[0])
            data = ramps_added_every_epoch
            ylabel = f"Ramp ID added"
            ax.set_yticks(list(range(1, 13)))
        
        plt.plot(list(range(1, len(config_every_epoch[dataset]) + 1)), 
                data, 
                marker="o", label=f"bert-{dataset}")  # color="#1f77b4"
    
    ax.set_xticks(x_index)
    ax.set_xlabel(f"Number of ramps", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    # ax.yaxis.set_label_coords(-0.11, 0.3)  # x, y
    # ax.set_ylim([min_y - 1e10, max_y + 1e10])

    ax.legend(loc="center right")
    ax.set_axisbelow(True)  # puts the grid below the bars
    ax.grid(color='lightgrey', linestyle='dashed', axis="both", linewidth=0.8)

    fig.savefig(f'{expr_name}.png', bbox_inches='tight', dpi=500)

    # plt.show()
    plt.close()
