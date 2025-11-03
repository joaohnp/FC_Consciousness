import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ptitprince as pt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from curbd_clean.utils.paths import resolve_path
from curbd_clean.analysis.simdata_utils import get_configs, get_neurons, psth_to_area
from curbd_clean.analysis.curbd import computeCURBD, trainMultiRegionRNN
from curbd_clean.plotting.io import save_figure


logger = logging.getLogger(__name__)

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

area1, area1_raster = psth_to_area(
    Neurons_E23, timing_Neurons_E23, picked_size, which_mode=WHICH_MODE, binsize=binsize
)
area2, area2_raster = psth_to_area(
    Neurons_E4, timing_Neurons_E4, picked_size, which_mode=WHICH_MODE, binsize=binsize
)
area3, area3_raster = psth_to_area(
    Neurons_E5, timing_Neurons_E5, picked_size, which_mode=WHICH_MODE, binsize=binsize
)
area4, area4_raster = psth_to_area(
    Neurons_E6, timing_Neurons_E6, picked_size, which_mode=WHICH_MODE, binsize=binsize
)

areas = ["E23", "E4", "E5", "E6"]
area_responses = {"E23": area1, "E4": area2, "E5": area3, "E6": area4}

pair_values = {
    ("E23", "E4"): {"E23_to_E4": [], "E4_to_E23": []},
    ("E4", "E5"): {"E4_to_E5": [], "E5_to_E4": []},
    ("E5", "E6"): {"E5_to_E6": [], "E6_to_E5": []},
    ("E6", "E23"): {"E6_to_E23": [], "E23_to_E6": []},
}

for i in range(0, TOTAL_SIMULATION):
    concatenated_matrix = np.concatenate(
        [area_responses[key] for key in area_responses]
    )
    region_index_list = []
    ROW_COUNTER = 0
    for region_name in area_responses:
        vectors = area_responses[region_name]
        end_index = ROW_COUNTER + len(vectors)
        indices_array = np.arange(ROW_COUNTER, end_index)
        region_index_list.append([region_name, indices_array])
        ROW_COUNTER = end_index

    region_index_array = np.array(region_index_list, dtype=object)
    model = trainMultiRegionRNN(
        concatenated_matrix,
        dtData=0.01,
        dtFactor=1,
        g=1.5,
        regions=region_index_array,
        tauRNN=0.1,
        nRunTrain=500,
        plotStatus=toplot,
        verbose=False,
        nRunFree=0,
    )
    curbd_arr, curbd_labels = computeCURBD(model)
    n_regions = curbd_arr.shape[0]

    base = resolve_path("src", "curbd_clean", "data", "simdata", "spike_data")
    connectivitymatrix = base / "connectivity.txt"
    labels = ["E23", "PV23", "E4", "PV4", "E5", "PV5", "E6", "PV6"]
    data_array = np.loadtxt(connectivitymatrix, delimiter=",")
    df = pd.DataFrame(np.reshape(data_array, [8, 8]))
    df.index = labels
    df.columns = labels
    pv = ["PV23", "PV4", "PV6", "PV5"]
    matrix_connectivity = df.drop(index=pv, columns=pv)

    strength_matrix = base / "connections_FINAL.txt"
    data_array2 = np.loadtxt(strength_matrix, delimiter=",")
    df_connections = pd.DataFrame(np.reshape(data_array2, [8, 8]))
    df_connections.index = labels
    df_connections.columns = labels
    matrix_connections = df_connections.drop(index=pv, columns=pv)

    labels_curbd = ["E23", "E4", "E5", "E6"]
    dfc = pd.DataFrame(index=labels_curbd, columns=labels_curbd)
    for iTarget in range(n_regions):
        for iSource in range(n_regions):
            split_string = curbd_labels[iTarget, iSource].split()
            df_src = split_string[0]
            df_trg = split_string[2]
            dfc[df_src][df_trg] = np.abs(
                np.mean(np.mean(curbd_arr[iTarget, iSource], axis=1), axis=0)
            )
    for pair, directions in pair_values.items():
        for direction in directions:
            values = (
                dfc.loc[pair[0], pair[1]]
                if direction.startswith(pair[0])
                else dfc.loc[pair[1], pair[0]]
            )
            pair_values[pair][direction].append(values)

num_directions = sum(len(directions) for directions in pair_values.values())
cols = 2
rows = (num_directions + 1) // cols

fig, axes = plt.subplots(cols, rows)
axes = axes.flatten()

idx = 0
for pair, directions in pair_values.items():
    for direction, values in directions.items():
        ax = axes[idx]
        data = pd.DataFrame(
            {
                "value": values,
                "direction": [
                    f"{pair[0]} to {pair[1]}"
                    if direction.startswith(pair[0])
                    else f"{pair[1]} to {pair[0]}"
                ]
                * len(values),
            }
        )
        pt.RainCloud(
            x="direction",
            y="value",
            hue="direction",
            orient="v",
            data=data,
            palette="Set2",
            bw=0.2,
            width_viol=0.6,
            ax=ax,
            alpha=0.65,
        )
        ax.set_ylim([-0.05, 3])
        ax.set_ylabel("Value")
        ax.set_title(
            f"{pair[0]} to {pair[1]}"
            if direction.startswith(pair[0])
            else f"{pair[1]} to {pair[0]}"
        )
        idx += 1

plt.suptitle("Distribution of Relationship Values for Each Pair and Direction", y=1.02)
plt.tight_layout()
plt.suptitle("Distribution of Relationship Values for Each Direction", y=1.02)

save_figure(fig, f"analysis/active_raincloud+{str(picked_size)}+")
