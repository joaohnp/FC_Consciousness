from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import logging
import random
import numpy as np
import pandas as pd
from scipy.signal import convolve
from scipy.signal.windows import gaussian
from .mapconvert import convert_folders


def calculate_firing_rate(
    spike_times,
    events,
    baseline,
    after_stimulus,
    bin_size,
    sampling_frequency,
    window_size,
    sigma,
):
    baseline_bins = int(baseline * sampling_frequency)
    after_stimulus_bins = int(after_stimulus * sampling_frequency)
    firing_rate_matrix = np.zeros((len(events), baseline_bins + after_stimulus_bins))
    for i, event_time in enumerate(events):
        event_bin = int(event_time)
        left_limit = event_bin - baseline_bins
        right_limit = event_bin + after_stimulus_bins
        num_bins = baseline_bins + after_stimulus_bins
        bin_edges = np.linspace(left_limit, right_limit, num=num_bins + 1)
        spike_times_in_window = spike_times[
            (spike_times >= left_limit) & (spike_times <= right_limit)
        ]
        if len(spike_times_in_window) == 0:
            continue
        spike_counts, _ = np.histogram(spike_times_in_window, bins=bin_edges)
        window_size_bins = int(window_size)
        sigma_bins = int(sigma)
        gaussian_window = gaussian(window_size_bins, sigma_bins)
        smoothed_event = convolve(spike_counts, gaussian_window, mode="same")
        firing_rate_matrix[i, :] = smoothed_event / 0.1
    return firing_rate_matrix


def psth_per_unit(units, cs, window, sigma, baseline):
    cs_responses = []
    for unit in units:
        firing_rate_matrix = calculate_firing_rate(
            unit, cs, 1, 3, 1, 1000, window, sigma
        )
        if baseline == 0:
            mean_fr = np.mean(firing_rate_matrix, axis=0)
            cs_responses.append(mean_fr)
        else:
            mean_fr = np.mean(firing_rate_matrix, axis=0)
            mean_fr2 = mean_fr[baseline:3000]
            baseline_mean_fr = np.mean(mean_fr2[0:baseline])
            baseline_std_fr = np.std(mean_fr2[0:baseline])
            mean_fr_zscored = (mean_fr2 - baseline_mean_fr) / baseline_std_fr
            cs_responses.append(mean_fr_zscored)
    return cs_responses


def units_area(TT_per_area, spks_folder, ChMap):
    units_in_area = []
    for folder_interest in TT_per_area:
        path_hd = convert_folders(folder_interest, spks_folder, ChMap)
        path_hd = Path(path_hd)
        if path_hd.exists():
            cluster_info_loc = path_hd / "phy_tdc" / "cluster_info.tsv"
            cluster_spikes_loc = path_hd / "phy_tdc" / "spike_clusters.npy"
            cluster_spktimes_loc = path_hd / "phy_tdc" / "spike_times.npy"
            cluster_info = pd.read_csv(cluster_info_loc, sep="\t")
            cluster_spikes = np.load(cluster_spikes_loc)
            cluster_spiketimes = np.load(cluster_spktimes_loc)
            unit_ids = cluster_info["cluster_id"][cluster_info["group"] != "noise"]
            for ii in unit_ids:
                SPK_ms = cluster_spiketimes[(cluster_spikes == ii)] / 30
                units_in_area.append(SPK_ms)
    return units_in_area


def find_events_in_stage(scored_epochs, cs, stage_of_interest):
    events_in_stage = []
    for timestamp in cs:
        epoch_index = timestamp // 5000
        if scored_epochs[epoch_index] == stage_of_interest:
            events_in_stage.append(timestamp)
    return events_in_stage


def curbd_get_avg(curbd_arr):
    num_rows, num_columns = curbd_arr.shape
    curbd_averages = np.empty((num_rows, num_columns), dtype=object)
    for iTarget in range(num_rows):
        for iSource in range(num_columns):
            curbd_quadrants = curbd_arr[iTarget, iSource]
            if curbd_quadrants.ndim > 1:
                curbd_quadrants = np.mean(curbd_quadrants, axis=0)
            quadrant_connectivity = np.empty(len(curbd_quadrants))
            for i in range(len(curbd_quadrants)):
                avg_connectivity_neuron = np.mean(curbd_quadrants[i])
                quadrant_connectivity[i] = avg_connectivity_neuron
            curbd_averages[iTarget, iSource] = quadrant_connectivity
    return curbd_averages


def prep_curbd(area_responses, areas, n_neurons):
    activity_matrix = []
    index_arrays = []
    regions = []
    for area_of_interest in areas:
        area_activity = area_responses[area_of_interest][0]
        neurons_area = len(area_activity)
        if n_neurons == -1:
            selected_neurons = area_activity
        elif neurons_area >= n_neurons:
            idx_neurons = random.sample(range(neurons_area), n_neurons)
            selected_neurons = [area_activity[i] for i in idx_neurons]
        else:
            raise ValueError(
                f"Number of requested neurons ({n_neurons}) exceeds n_neurons in area {area_of_interest} ({neurons_area})."
            )
        if len(activity_matrix) == 0:
            activity_matrix = selected_neurons
        else:
            activity_matrix = np.vstack((activity_matrix, selected_neurons))
        index_array = np.arange(
            len(activity_matrix) - len(selected_neurons),
            len(activity_matrix),
        )
        index_arrays.append(index_array)
        regions.append([f"Region {area_of_interest}", index_array])
    return activity_matrix, index_arrays, regions


def prep_sim(area_responses, areas, combination):
    activity_matrix = []
    index_arrays = []
    regions = []
    for area_of_interest, neuron_indices in zip(areas, combination):
        area_activity = area_responses[area_of_interest][0]
        if isinstance(neuron_indices, int):
            selected_neurons = [area_activity[neuron_indices]]
        else:
            selected_neurons = [area_activity[idx] for idx in neuron_indices]
        if len(activity_matrix) == 0:
            activity_matrix = np.array(selected_neurons)
        else:
            activity_matrix = np.vstack((activity_matrix, selected_neurons))
        index_array = np.arange(
            len(activity_matrix) - len(selected_neurons), len(activity_matrix)
        )
        index_arrays.append(index_array)
        regions.append([f"Region {area_of_interest}", index_array])
    return activity_matrix, index_arrays, regions


def generate_random_neuron_combinations(neurons_per_area, structure, max_combinations):
    combinations = set()
    while len(combinations) < max_combinations:
        random_combination = []
        for n, r in zip(neurons_per_area, structure):
            if r > 0:
                random_combination.append(random.sample(range(n), r))
            else:
                random_combination.append([])
        combinations.add(tuple(map(tuple, random_combination)))
    return [list(map(list, combo)) for combo in combinations]


def find_combinations(areas, total_neurons, current_combination=[]):
    if len(current_combination) == len(areas):
        if sum(current_combination) == total_neurons and all(
            n > 0 for n in current_combination
        ):
            yield current_combination
        return
    for i in range(1, areas[len(current_combination)] + 1):
        if sum(current_combination) + i > total_neurons:
            break
        yield from find_combinations(areas, total_neurons, current_combination + [i])
