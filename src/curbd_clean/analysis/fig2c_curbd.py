import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from curbd_clean.analysis.simdata_utils import get_configs, get_neurons, psth_to_area
from curbd_clean.analysis.curbd import computeCURBD, trainMultiRegionRNN
from curbd_clean.plotting.io import save_figure

logger = logging.getLogger(__name__)

(
    TOTAL_NEURONS,
    TOTAL_SIMULATION,
    n_initial,
    toplot,
    all_values,
    picked_size,
    binsize,
) = get_configs()

toplot = True
WHICH_MODE = "all"
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
area_responses = {"E23": area2, "E4": area1, "E5": area3, "E6": area4}

concatenated_matrix = np.concatenate([area_responses[key] for key in area_responses])
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
    nRunTrain=500,
    plotStatus=toplot,
    verbose=False,
    nRunFree=0,
)

curbd_arr, curbd_labels = computeCURBD(model)
n_regions = curbd_arr.shape[0]
