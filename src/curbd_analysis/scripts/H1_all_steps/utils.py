import itertools
import os
import random

import numpy as np
import pandas as pd
from scipy.signal import convolve, gaussian

from utils_mapconvert import convert_folders


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
    # Convert baseline and after-stimulus periods to time bins
    baseline_bins = int(baseline * sampling_frequency)
    after_stimulus_bins = int(after_stimulus * sampling_frequency)

    # Initialize firing rate matrix
    firing_rate_matrix = np.zeros(
        (len(events), baseline_bins + after_stimulus_bins)
    )

    # Iterate over events
    for i, event_time in enumerate(events):
        # Convert event time to time bin
        event_bin = int(event_time)

        # Calculate left and right limits
        left_limit = event_bin - baseline_bins
        right_limit = event_bin + after_stimulus_bins

        # Create time bins
        num_bins = baseline_bins + after_stimulus_bins
        bin_edges = np.linspace(left_limit, right_limit, num=num_bins + 1)
        # Filter spike times within the specified time window
        spike_times_in_window = spike_times[
            (spike_times >= left_limit) & (spike_times <= right_limit)
        ]

        # Check if there are spike times in the period
        if len(spike_times_in_window) == 0:
            continue  # Skip this event if no spikes in the period

        # Calculate spike counts in each bin
        spike_counts, _ = np.histogram(spike_times_in_window, bins=bin_edges)
        # Parameters
        # window_size = 50  # Size of the Gaussian window in ms
        # sigma = 5  # Standard deviation of the Gaussian window in ms
        # Convert window_size and sigma to time bins
        window_size_bins = int(window_size)
        sigma_bins = int(sigma)
        # Create Gaussian window
        gaussian_window = gaussian(window_size_bins, sigma_bins)
        smoothed_event = convolve(spike_counts, gaussian_window, mode="same")

        # Store the firing rate in the matrix
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
        elif baseline != 0:
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
        if os.path.exists(path_hd):
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

            unit_ids = cluster_info["cluster_id"][
                cluster_info["group"] != "noise"
            ]

            for ii in unit_ids:
                SPK_ms = (
                    cluster_spiketimes[(cluster_spikes == ii)]
                    / 30  # Converting back to seconds
                )
                units_in_area.append(SPK_ms)

    return units_in_area


def find_events_in_stage(scored_epochs, cs, stage_of_interest):
    # Initialize a list to store events for the stage of interest
    events_in_stage = []

    # Iterate through the event timestamps (cs_novel)
    for timestamp in cs:
        # Calculate the epoch index for the current event based on seconds
        epoch_index = timestamp // 5000  # Assuming each epoch is 5 seconds
        # Check if the stage of the corresponding epoch matches the stage of interest
        if scored_epochs[epoch_index] == stage_of_interest:
            # Store the event timestamp in the list
            events_in_stage.append(timestamp)

    return events_in_stage


def curbd_get_avg(curbd_arr):
    num_rows, num_columns = curbd_arr.shape
    # Create an empty matrix to store the quadrant connectivity averages
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

            # Store the quadrant_connectivity as an array in curbd_averages
            curbd_averages[iTarget, iSource] = quadrant_connectivity

    return curbd_averages


def prep_curbd(area_responses, areas, n_neurons):
    activity_matrix = []
    index_arrays = []  # Initialize the list for index arrays
    regions = []  # Initialize the list for region information
    for area_of_interest in areas:
        area_activity = area_responses[area_of_interest][0]
        neurons_area = len(area_activity)
        if n_neurons == -1:
            # Include all neurons in the specified area
            selected_neurons = area_activity
        elif neurons_area >= n_neurons:
            idx_neurons = random.sample(range(neurons_area), n_neurons)
            selected_neurons = [area_activity[i] for i in idx_neurons]
        else:
            # Handle the case when n_neurons is greater than the available neurons
            raise ValueError(
                f"Number of requested neurons ({n_neurons}) exceeds "
                f"n_neurons in area {area_of_interest} ({neurons_area})."
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
    index_arrays = []  # Initialize the list for index arrays
    regions = []  # Initialize the list for region information

    for area_of_interest, neuron_indices in zip(areas, combination):
        area_activity = area_responses[area_of_interest][0]

        # Handle both single and multiple neuron selections
        if isinstance(neuron_indices, int):
            selected_neurons = [area_activity[neuron_indices]]
        else:  # It's a list of indices
            selected_neurons = [area_activity[idx] for idx in neuron_indices]

        # Stack the selected neurons to form the activity matrix
        if len(activity_matrix) == 0:
            activity_matrix = np.array(selected_neurons)
        else:
            activity_matrix = np.vstack((activity_matrix, selected_neurons))

        # Determine the indices in the activity matrix
        index_array = np.arange(
            len(activity_matrix) - len(selected_neurons), len(activity_matrix)
        )
        index_arrays.append(index_array)

        # Append the region information
        regions.append([f"Region {area_of_interest}", index_array])

    return activity_matrix, index_arrays, regions


# def generate_neuron_combinations(
#     neurons_per_area, pool_combination, max_combinations=10000
# ):
#     """
#     Generate a limited number of combinations of neuron selections for each area.
#     :param neurons_per_area: List of total neurons available in each area.
#     :param pool_combination: List of lists specifying the neuron indices to select from each area.
#     :param max_combinations: Maximum number of combinations to return.
#     :return: List of lists, each containing a specific combination.
#     """
#     area_combinations = []
#     for n, indices in zip(neurons_per_area, pool_combination):
#         if isinstance(indices, list) and len(indices) > 1:
#             area_combinations.append([tuple(indices)])
#         else:
#             r = indices[0] if isinstance(indices, list) else indices
#             area_combinations.append(list(itertools.combinations(range(n), r)))

#     all_combinations = list(itertools.product(*area_combinations))
#     formatted_combinations = [
#         [list(group) for group in combo] for combo in all_combinations
#     ]

#     # Randomly sample up to max_combinations from the list
#     if len(formatted_combinations) > max_combinations:
#         formatted_combinations = random.sample(
#             formatted_combinations, max_combinations
#         )


#     return formatted_combinations
def generate_random_neuron_combinations(
    neurons_per_area, structure, max_combinations
):
    """
    Generate a specified number of random combinations of neuron selections for each area.
    """
    combinations = set()

    while len(combinations) < max_combinations:
        random_combination = []
        for n, r in zip(neurons_per_area, structure):
            if r > 0:
                random_combination.append(random.sample(range(n), r))
            else:
                random_combination.append(
                    []
                )  # No neuron selected from this area
        combinations.add(tuple(map(tuple, random_combination)))

    return [list(map(list, combo)) for combo in combinations]


def find_combinations(areas, total_neurons, current_combination=[]):
    # Base case: if we've considered all areas
    if len(current_combination) == len(areas):
        # Check if the total number of neurons equals the target and at least one neuron is selected from each area
        if sum(current_combination) == total_neurons and all(
            n > 0 for n in current_combination
        ):
            yield current_combination
        return

    # Start from 1 since we need at least one neuron from each area
    for i in range(1, areas[len(current_combination)] + 1):
        # Stop if the current number of neurons already exceeds the target
        if sum(current_combination) + i > total_neurons:
            break

        # Recursive call
        yield from find_combinations(
            areas, total_neurons, current_combination + [i]
        )
