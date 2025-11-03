import logging
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import convolve
from scipy.signal.windows import gaussian  # from curbd_folder import curbd
import curbd

from utils import units_area
from utils_expinfo import get_experimental_info


def get_files_with_prefix(directory, prefix):
    all_files = list(directory.iterdir())
    matching_files = [file.name for file in all_files if file.name[:3] == prefix]

    return matching_files


def find_events_in_stage(scored_epochs, cs, stage_of_interest):
    events_in_stage = []
    epoch_events = []
    for timestamp in cs:
        epoch_index = timestamp // 5000  # Assuming each epoch is 5 seconds
        if 0 <= epoch_index < len(scored_epochs):
            if scored_epochs[epoch_index] == stage_of_interest:
                events_in_stage.append(timestamp)
                epoch_events.append(epoch_index)
    return events_in_stage, epoch_events


def calculate_firing_rate(
    spike_times,
    epoch_number,
    window_size,
    sigma,
):
    spike_times = spike_times / 1000
    firing_rate_matrix = np.zeros((1, 5000))
    left_limit = epoch_number * 5
    right_limit = epoch_number * 5 + 5
    num_bins = 5000
    bin_edges = np.linspace(left_limit, right_limit, num=num_bins + 1)
    bin_width = 0.1
    spike_times_in_window = spike_times[
        (spike_times > left_limit).all(axis=1) & (spike_times < right_limit).all(axis=1)
    ]
    if len(spike_times_in_window) == 0:
        ...
    else:
        spike_counts, _ = np.histogram(spike_times_in_window, bins=bin_edges)
        window_size_bins = int(window_size / bin_width)
        sigma_bins = int(sigma / bin_width)
        gaussian_window = gaussian(window_size_bins, sigma_bins)
        smoothed_event = convolve(spike_counts, gaussian_window, mode="same")
        firing_rate_matrix = smoothed_event / 0.1
    return firing_rate_matrix


def psth_per_unit(units, epoch_number, window, sigma):
    # check if the psth is how we need it
    all_fr = []
    for indx, unit in enumerate(units):
        df_unit = pd.DataFrame(unit)
        firing_rate_matrix = calculate_firing_rate(df_unit, epoch_number, window, sigma)
        all_fr.append(firing_rate_matrix)
    return all_fr


def prep_curbd(area_responses, areas):
    activity_matrix = []
    index_arrays = []
    regions = []
    current_index = 0
    for area_of_interest in areas:
        # Get and stack activity for this area
        area_activity = area_responses[area_of_interest][0]
        if len(area_activity) == 0:
            area_activity = [-500]
        activity_region = np.vstack(area_activity)
        activity_matrix.append(activity_region)
        # Create indices for this region
        n_neurons = len(area_activity)
        region_indices = np.arange(current_index, current_index + n_neurons)
        # Store indices and region info
        index_arrays.append(region_indices)
        regions.append([f"Region {area_of_interest}", region_indices])
        current_index += n_neurons
    return np.vstack(activity_matrix), index_arrays, regions


def process_epoch(
    epoch_n,
    scores,
    areas_mapping,
    TT_list,
    spikes_folder,
    ChMap,
    epochs_to_skip,
    logger,
    window=125,
    sigma=10,
    curbd_nrun=200,
):
    start_time = time.time()
    area_responses = {area_name: [] for area_name in areas_mapping.values()}
    # Check if epoch should be skipped (adjust this logic as needed)
    if sum(epochs_to_skip == epoch_n) > 0:
        ...
        return None
    for index, area in areas_mapping.items():
        area_of_interest = areas_mapping[index]
        TT_per_area = [name for name in TT_list if area_of_interest in name]

        UnitsInArea = units_area(TT_per_area, spikes_folder, ChMap)
        epoch_responses = psth_per_unit(UnitsInArea, epoch_n + 1, window=125, sigma=10)
        area_responses[area_of_interest].append(epoch_responses)
    areas = ["A1", "BLA", "HPC", "PFC"]
    activity_matrix, index_arrays, regions = prep_curbd(area_responses, areas)
    model = curbd.trainMultiRegionRNN(
        activity_matrix,
        dtData=0.001,
        dtFactor=1,
        g=3,
        regions=np.array(regions, dtype=object),
        tauRNN=0.1,
        nRunTrain=curbd_nrun,
        plotStatus=False,
        verbose=False,
        nRunFree=0,
    )
    elapsed_time = time.time() - start_time
    logger.info(f"Epoch {epoch_n} processed in {elapsed_time:.2f} seconds")
    [curbd_arr, curbd_labels] = curbd.computeCURBD(model)
    return {
        "epoch_n": epoch_n,
        "curbd_arr": curbd_arr,
        "curbd_labels": curbd_labels,
    }


projectPath = Path(__file__).resolve().parent.parent.parent
sleepScores_path = projectPath / "data" / "SleepScores"
(
    animal,
    sessions,
    stages,
    behaviour,
    hd_mapping,
    areas_mapping,
    cs_naming,
    stage_names,
    stage_mapping,
) = get_experimental_info()
curbd_nrun = 200  # in the pilot it was 200!
for animal_chosen_idx in animal:
    animal_chosen = animal[animal_chosen_idx]
    matching_files = get_files_with_prefix(sleepScores_path, animal_chosen)
    for file_of_interest in matching_files:
        with open(sleepScores_path / file_of_interest, "rb") as fp:
            scores = pickle.load(fp)
            if len(scores) == 3:
                scores_tocheck = scores[2]
            else:
                scores_tocheck = scores

        # Step 3: Decide which epochs are subjected to our analysis
        with open(projectPath / "data" / "TTLs" / file_of_interest, "rb") as fp:
            cs_minus, cs_novel, cs_plus = pickle.load(fp)
        # changes on lab pc which ran the full pipeline

        cs_mapping = {"cs_plus": cs_plus, "cs_minus": cs_minus, "cs_novel": cs_novel}
        stage_info = {
            stage: {stimulus: [] for stimulus in cs_naming} for stage in stage_names
        }
        for stage_name in stage_names:
            for stimulus in cs_naming:
                events_in_stage = find_events_in_stage(
                    scores_tocheck, cs_mapping[stimulus], stage_mapping[stage_name]
                )
                stage_info[stage_name][stimulus] = events_in_stage

        epochs_to_avoid_acc = []
        for stg in stage_info.keys():
            STG_TIMING_CS = stage_info[stg]
            df_stim_NREM = pd.DataFrame.from_dict(STG_TIMING_CS)
            epochs_to_avoid = np.concatenate(
                df_stim_NREM[["cs_novel", "cs_plus", "cs_minus"]].iloc[1].values
            )
            epochs_to_avoid_acc.append(epochs_to_avoid)
        if len(epochs_to_avoid_acc) == 5:
            epochs_to_avoid_certain = np.concatenate(
                [
                    epochs_to_avoid_acc[0],
                    epochs_to_avoid_acc[1],
                    epochs_to_avoid_acc[2],
                    epochs_to_avoid_acc[3],
                    epochs_to_avoid_acc[4],
                ]
            ).astype(int)

            scores_clean = np.delete(scores_tocheck, epochs_to_avoid_certain)
        else:
            scores_clean = np.delete(
                scores_tocheck, np.concatenate(epochs_to_avoid_acc)
            )

        ChMap_path = projectPath / "data" / "ChMaps" / f"{animal_chosen}.pickle"
        with open(ChMap_path, "rb") as handle:
            ChMap = pickle.load(handle)

        TT_list = list(set(ChMap))  # Finding the name of each unique tetrode
        area_psth = []
        area_responses = {area_name: [] for area_name in areas_mapping.values()}
        area_units = {area_name: [] for area_name in areas_mapping.values()}
        hard_drive = projectPath / "data" / "raw"
        hd_session = hd_mapping[2]
        spikes_folder = hard_drive / animal_chosen / hd_session

        epochs_to_skip = np.concatenate(epochs_to_avoid_acc)
        df_curbd_epochs = pd.DataFrame(columns=["curbd_arr", "epoch_n"])
        df_curbd_epochs["curbd_arr"] = np.zeros(len(scores_tocheck))

        logging.basicConfig(level=logging.INFO, format="%(message)s")
        logger = logging.getLogger()

        print("Starting!")
        hard_drive_save = projectPath / "data" / "curbd_output"
        save_path = hard_drive_save / "results" / file_of_interest
        save_path.mkdir(parents=True, exist_ok=True)
        for epoch_n in range(0, len(scores_tocheck)):
            result = process_epoch(
                epoch_n,
                scores,
                areas_mapping,
                TT_list,
                spikes_folder,
                ChMap,
                epochs_to_skip,
                logger,
            )
            tosave = save_path / f"{epoch_n}.pkl"
            with open(tosave, "wb") as file:
                pickle.dump(result, file)
        print("Finishing!")
