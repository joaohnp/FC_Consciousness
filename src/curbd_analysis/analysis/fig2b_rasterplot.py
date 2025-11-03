import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from curbd_analysis.analysis.simdata_utils import (
    get_configs,
    get_neurons,
    psth_to_area,
)
from curbd_analysis.plotting.io import save_figure

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

(
    TOTAL_NEURONS,
    TOTAL_SIMULATION,
    n_initial,
    toplot,
    all_values,
    neurons_per_area,
    binsize,
) = get_configs()

area1, area1_raster = psth_to_area(
    indices=Neurons_E23,
    times=timing_Neurons_E23,
    total_neurons=neurons_per_area,
    binsize=binsize,
)
area2, area2_raster = psth_to_area(
    indices=Neurons_E4,
    times=timing_Neurons_E4,
    total_neurons=neurons_per_area,
    binsize=binsize,
)
area3, area3_raster = psth_to_area(
    indices=Neurons_E5,
    times=timing_Neurons_E5,
    total_neurons=neurons_per_area,
    binsize=binsize,
)
area4, area4_raster = psth_to_area(
    indices=Neurons_E6,
    times=timing_Neurons_E6,
    total_neurons=neurons_per_area,
    binsize=binsize,
)


TOTAL_LENGTH = 4000
STEP = 4000 / binsize
time_axis = np.arange(0, TOTAL_LENGTH, STEP)
a = [area4_raster, area3_raster, area2_raster, area1_raster]
flattened_a = [item for sublist in a for item in sublist]

fig, ax = plt.subplots(1, 1, figsize=(12, 12))
ax.eventplot(flattened_a, colors="k")
ax.axvline(1, color="red", label="Stimulus start")
ax.set_xlabel("Time")
ax.set_ylabel("Unit #")
ax.set_title("Raster plots")

save_figure(fig, "analysis/fig2b_rasterplot")
