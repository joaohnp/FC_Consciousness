import logging
from pathlib import Path
import random

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from curbd_clean.utils.paths import resolve_path

logger = logging.getLogger(__name__)


def get_neurons():
    base = resolve_path("src", "curbd_clean", "data", "simdata")
    INPUT_TOCHECK = "input4_0.3_50"
    neurons_e4 = base / INPUT_TOCHECK / "S_e4i.txt"
    timing_e4 = base / INPUT_TOCHECK / "S_e4t.txt"
    neurons_e5 = base / INPUT_TOCHECK / "S_e5i.txt"
    timing_e5 = base / INPUT_TOCHECK / "S_e5t.txt"
    neurons_e6 = base / INPUT_TOCHECK / "S_e6i.txt"
    timing_e6 = base / INPUT_TOCHECK / "S_e6t.txt"
    neurons_e23 = base / INPUT_TOCHECK / "S_e23i.txt"
    timing_e23 = base / INPUT_TOCHECK / "S_e23t.txt"
    logger.info("Loaded neuron paths from %s", base)
    return (
        neurons_e4,
        timing_e4,
        neurons_e23,
        timing_e23,
        neurons_e5,
        timing_e5,
        neurons_e6,
        timing_e6,
    )


def spk_to_psth(neuron, binsize=25, sigma_bins=2):
    x = neuron.to_numpy() if hasattr(neuron, "to_numpy") else np.asarray(neuron)
    if x.size == 0:
        return np.array([])
    x = x.astype(float)
    max_t = int(np.ceil(x.max()))
    if not np.isfinite(max_t) or max_t < 0:
        return np.array([])
    bins = 4000 / binsize
    counts, _ = np.histogram(x, bins=int(bins))
    rate_hz = counts * (1000.0 / float(binsize))
    gaus = gaussian_filter1d(rate_hz, sigma_bins)

    return gaus


def psth_to_area(
    indices: Path,
    times: Path,
    total_neurons: int = 20,
    which_mode: str = "all",
    binsize: int = 25,
):
    psths_neurons = []
    stacked_raster = []
    neuron_indices = pd.read_csv(indices, names=["Number"], header=None)
    spike_times = pd.read_csv(times, names=["Number"], header=None)
    freq_idx = random.sample(list(neuron_indices["Number"].unique()), total_neurons)
    for idx in freq_idx:
        neuron1 = spike_times["Number"][neuron_indices["Number"] == idx]
        stacked_raster.append(neuron1)
        gaus = spk_to_psth(neuron1, binsize)
        if which_mode == "baseline":
            psths_neurons.append(gaus[5:50])
        elif which_mode == "active":
            psths_neurons.append(gaus[100:150])
        elif which_mode == "all":
            psths_neurons.append(gaus[3:-3])
    return psths_neurons, stacked_raster


def get_configs():
    total_neurons = 30
    total_simulation = 1000
    n_initial = 1
    toplot = False
    all_values = []
    picked_size = 30
    binsize = 25
    return (
        total_neurons,
        total_simulation,
        n_initial,
        toplot,
        all_values,
        picked_size,
        binsize,
    )


def print_connectivity():
    base = resolve_path("src", "curbd_jp", "data", "simdata", "spike_data")
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
    logger.info("Connectivity: %s", matrix_connectivity.to_string())
    logger.info("Connections: %s", matrix_connections.to_string())
    logger.info("Product: %s", (matrix_connectivity * matrix_connections).to_string())
