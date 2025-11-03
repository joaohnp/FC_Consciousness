import os
import pickle

import numpy as np
import pandas as pd

from utils_expinfo import get_experimental_info
from utils_mapconvert import convert_folders


def units_area(TT_per_area, spks_folder, ChMap):
    units_in_area = []
    for folder_interest in TT_per_area:
        path_hd = convert_folders(folder_interest, spks_folder, ChMap)
        if os.path.exists(path_hd):
            print(path_hd)
            # % Loading cluster info
            cluster_info_loc = os.path.join(
                path_hd, "phy_tdc", "cluster_info.tsv"
            )
            cluster_info = pd.read_csv(cluster_info_loc, sep="\t")
            # % Loading cluster spikes
            cluster_spikes_loc = os.path.join(
                path_hd, "phy_tdc", "spike_clusters.npy"
            )
            cluster_spikes = np.load(cluster_spikes_loc)
            # % Loading cluster spike times
            cluster_spktimes_loc = os.path.join(
                path_hd, "phy_tdc", "spike_times.npy"
            )
            cluster_spiketimes = np.load(cluster_spktimes_loc)
            units = len(cluster_info["cluster_id"][
                cluster_info["group"] != "noise"
            ])
            # units_mua = len(cluster_info["cluster_id"][
            #     cluster_info["group"] == "mua"
            # ])
            # units_good = len(cluster_info["cluster_id"][
            #     cluster_info["group"] == "good"
            # ])
            print(units)
            units_in_area.append(units)
    return units_in_area
(   animal,
    sessions,
    stages,
    behaviour,
    hd_mapping,
    areas_mapping,
    cs_naming,
    stage_names,
    stage_mapping,
) = get_experimental_info()

animal_idx = 3
sessions_idx = 2
selected_animal = animal[animal_idx]
selected_session = sessions[sessions_idx]
hd_session = hd_mapping[sessions_idx]
projectPath = os.path.dirname(os.path.abspath(__name__))
print(f"Analysis will be run at {selected_animal} session {selected_session}")
ChMap_path = os.path.join(projectPath, "ChMaps", selected_animal + ".pickle")
with open(ChMap_path, "rb") as handle:
    ChMap = pickle.load(handle)

hard_drive = "/run/media/joaohnp/Elements"
spikes_folder = os.path.join(hard_drive, selected_animal, hd_session)
TT_list = list(set(ChMap))  # Finding the name of each unique tetrode
area_units = {area_name: [] for area_name in areas_mapping.values()}
for area in areas_mapping:
    area_of_interest = areas_mapping[area]
    TT_per_area = [name for name in TT_list if area_of_interest in name]
    print(TT_per_area)
    UnitsInArea = units_area(TT_per_area, spikes_folder, ChMap)
    area_units[area_of_interest].append(UnitsInArea)

print(area_units)
