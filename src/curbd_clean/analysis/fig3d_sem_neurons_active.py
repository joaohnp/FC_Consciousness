import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from joblib import Parallel, delayed
from simdata_utils import get_configs, get_neurons, psth_to_area
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

sys.path.insert(0, "/home/joaohnp/github/curbd_jp/curbd_folder")
from curbd import computeCURBD, trainMultiRegionRNN
from curbd_clean.plotting.io import save_figure

(
    Neurons_E4,
    timing_Neurons_E4,
    Neurons_E23,
    timing_Neurons_E23,
    Neurons_E5,
    timing_Neurons_E5,
    Neurons_E6,
    timing_Neurons_E6,
) = get_neurons()


def process_simulation(picked_size, i):
    area1, area1_raster = psth_to_area(
        Neurons_E23,
        timing_Neurons_E23,
        picked_size,
        which_mode=WHICH_MODE,
        binsize=binsize,
    )
    area2, area2_raster = psth_to_area(
        Neurons_E4,
        timing_Neurons_E4,
        picked_size,
        which_mode=WHICH_MODE,
        binsize=binsize,
    )
    area3, area3_raster = psth_to_area(
        Neurons_E5,
        timing_Neurons_E5,
        picked_size,
        which_mode=WHICH_MODE,
        binsize=binsize,
    )
    area4, area4_raster = psth_to_area(
        Neurons_E6,
        timing_Neurons_E6,
        picked_size,
        which_mode=WHICH_MODE,
        binsize=binsize,
    )

    area_responses = {"E23": area1, "E4": area2, "E5": area3, "E6": area4}
    concatenated_matrix = np.concatenate(
        [area_responses[key] for key in area_responses]
    )
    region_index_list = []
    row_counter = 0

    for region_name in area_responses:
        vectors = area_responses[region_name]
        end_index = row_counter + len(vectors)
        indices_array = np.arange(row_counter, end_index)
        region_index_list.append([region_name, indices_array])
        row_counter = end_index

    region_index_array = np.array(region_index_list, dtype=object)
    model = trainMultiRegionRNN(
        concatenated_matrix,
        dtData=0.01,
        dtFactor=1,
        g=1.5,
        regions=region_index_array,
        tauRNN=0.1,
        nRunTrain=5,
        plotStatus=toplot,
        verbose=False,
        nRunFree=0,
    )
    curbd_arr, curbd_labels = computeCURBD(model)
    return curbd_arr, curbd_labels


WHICH_MODE = "active"
(
    TOTAL_NEURONS,
    TOTAL_SIMULATION,
    n_initial,
    toplot,
    all_values,
    picked_size,
    binsize,
) = get_configs()
PROJECT_PATH = "/home/joaohnp/github/curbd_jp/CURBD_Simulated_Data"

toplot = False
all_values = []
agent_pair_values = {}
TOTAL_NEURONS = 10
for picked_size in range(n_initial, TOTAL_NEURONS):
    print(f"Checking for {picked_size} neurons")
    results = Parallel(n_jobs=-1)(
        delayed(process_simulation)(picked_size, i) for i in range(TOTAL_SIMULATION)
    )

    pair_values = {
        ("E23", "E4"): {"E23_to_E4": [], "E4_to_E23": []},
        ("E4", "E5"): {"E4_to_E5": [], "E5_to_E4": []},
        ("E5", "E6"): {"E5_to_E6": [], "E6_to_E5": []},
        ("E6", "E23"): {"E6_to_E23": [], "E23_to_E6": []},
    }

    for curbd_arr, curbd_labels in results:
        n_regions = curbd_arr.shape[0]

        Labels_curbd = ["E23", "E4", "E5", "E6"]
        df = pd.DataFrame(index=Labels_curbd, columns=Labels_curbd)
        for iTarget in range(n_regions):
            for iSource in range(n_regions):
                split_string = curbd_labels[iTarget, iSource].split()
                df_src = split_string[0]
                df_trg = split_string[2]
                df[df_src][df_trg] = np.abs(
                    np.mean(np.mean(curbd_arr[iTarget, iSource], axis=1), axis=0)
                )
        for pair, directions in pair_values.items():
            for direction in directions:
                values = (
                    df.loc[pair[0], pair[1]]
                    if direction.startswith(pair[0])
                    else df.loc[pair[1], pair[0]]
                )
                pair_values[pair][direction].append(values)

    for pair, directions in pair_values.items():
        for direction in directions:
            for create_dict in range(n_initial, TOTAL_NEURONS):
                if create_dict not in agent_pair_values:
                    agent_pair_values[create_dict] = {}
                if pair not in agent_pair_values[create_dict]:
                    agent_pair_values[create_dict][pair] = {}
                if direction not in agent_pair_values[create_dict][pair]:
                    agent_pair_values[create_dict][pair][direction] = []

    for pair, directions in pair_values.items():
        for direction, values in directions.items():
            for value in values:
                agent_pair_values[picked_size][pair][direction].append(value)

histograms = {}

for picked_size, interactions in agent_pair_values.items():
    histograms[picked_size] = {}
    for pair, directions in interactions.items():
        histograms[picked_size][pair] = {}
        for direction, values in directions.items():
            histograms[picked_size][pair][direction] = (
                np.average(values),
                (np.std(values) / np.sqrt(TOTAL_SIMULATION)),
            )

number_of_agents = []
interactions = []
values = []
for picked_size, pairs in histograms.items():
    for pair, directions in pairs.items():
        for direction, value in directions.items():
            number_of_agents.append(picked_size)
            interactions.append(f"{direction}")
            values.append(value)

heatmap_data = pd.DataFrame(
    {"N": number_of_agents, "Interaction": interactions, "Value": values}
)
heatmap_data["mean"] = heatmap_data["Value"].apply(lambda x: x[0])
heatmap_data["std"] = heatmap_data["Value"].apply(lambda x: x[1])
x_values = range(0, TOTAL_NEURONS)
df = pd.DataFrame(heatmap_data)
anova_results = {}
for interaction in df["Interaction"].unique():
    subset = df[df["Interaction"] == interaction]
    print(subset)
    print(subset["Value"])
    # model = ols('Value ~ C(N)', data=subset).fit()
    # anova_table = sm.stats.anova_lm(model, typ=2)
    # anova_results[interaction] = anova_table

    # print(f"ANOVA results for {interaction}:")
    # print(anova_table)

    # # Perform post hoc test if ANOVA is significant
    # if anova_table['PR(>F)'][0] < 0.05:
    #     tukey = pairwise_tukeyhsd(subset['Value'], subset['N'], alpha=0.05)
    #     print(f"Tukey HSD results for {interaction}:")
    #     print(tukey)
    # else:
    #     print(f"No significant differences found for {interaction}")

plt.figure(figsize=(14, 10))
for interaction in heatmap_data["Interaction"].unique():
    interaction_data = heatmap_data[heatmap_data["Interaction"] == interaction]
    x_values = interaction_data["N"]
    mean_values = interaction_data["mean"]
    std_values = interaction_data["std"]

    plt.plot(x_values, mean_values, label=interaction)
    plt.fill_between(
        x_values, mean_values - std_values, mean_values + std_values, alpha=0.3
    )

# Adding title and labels
plt.ylim([0, 0.8])
plt.show()
plt.xlabel("Number of Neurons")
plt.ylabel("Mean Value")
plt.legend(title="Interaction")


# plt.ylabel('Interaction')
if WHICH_MODE == "active":
    save_figure(plt.gcf(), "analysis/sem_active+{TOTAL_SIMULATION}")
