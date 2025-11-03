# %%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import convolve, welch
from scipy.signal.windows import gaussian

from curbd_folder import curbd as curbd
from utils import curbd_get_avg, units_area
from utils_expinfo import get_experimental_info
from utils_plotting import plot_raster, plotting_psth

projectPath = os.path.dirname(os.path.abspath(__name__))
sleepScores_path = os.path.join(projectPath, 'SleepScores')
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
animal_chosen = animal[3]

file_of_interest = 'r20ProbeSleep.pickle'
with open(os.path.join(sleepScores_path, file_of_interest),'rb') as fp:
    scores = pickle.load(fp)

# Step 3: Decide which epochs are subjected to our analysis
with open(os.path.join(projectPath, 'TTLs', 'r20ProbeSleep.pickle'),'rb') as fp:
    cs_minus, cs_novel, cs_plus = pickle.load(fp)

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

cs_mapping = {"cs_plus": cs_plus, "cs_minus": cs_minus, "cs_novel": cs_novel}
stage_info = {
    stage: {stimulus: [] for stimulus in cs_naming} for stage in stage_names
}
for stage_name in stage_names:
    for stimulus in cs_naming:
        events_in_stage = find_events_in_stage(
            scores[2], cs_mapping[stimulus], stage_mapping[stage_name]
        )
        stage_info[stage_name][stimulus] = events_in_stage

epochs_to_avoid_acc = []
for stg in stage_info.keys():
    STG_TIMING_CS = stage_info[stg]
    df_stim_NREM = pd.DataFrame.from_dict(STG_TIMING_CS)
    epochs_to_avoid = np.concatenate(df_stim_NREM[['cs_novel', 'cs_plus', 'cs_minus']].iloc[1].values)
    epochs_to_avoid_acc.append(epochs_to_avoid)

scores_clean = np.delete(scores[2], np.concatenate(epochs_to_avoid_acc))

ChMap_path = os.path.join(projectPath, "ChMaps", animal_chosen + '.pickle')
with open(ChMap_path, "rb") as handle:
    ChMap = pickle.load(handle)


TT_list = list(set(ChMap))  # Finding the name of each unique tetrode
area_psth = []
area_responses = {area_name: [] for area_name in areas_mapping.values()}
area_units = {area_name: [] for area_name in areas_mapping.values()}
hard_drive = "/run/media/joaohnp/Elements"
hd_session=hd_mapping[2]
spikes_folder = os.path.join(hard_drive, animal_chosen, hd_session)
def calculate_firing_rate(
    spike_times,
    epoch_number,
    window_size,
    sigma,
):
    spike_times = spike_times/1000
    firing_rate_matrix = np.zeros((1,5000))
    left_limit = epoch_number*5
    right_limit = epoch_number*5 + 5
    num_bins = 5000
    bin_edges = np.linspace(left_limit, right_limit, num=num_bins + 1)
    bin_width = 0.1
    spike_times_in_window = spike_times[(spike_times > left_limit).all(axis=1) & (spike_times < right_limit).all(axis=1)]
    if len(spike_times_in_window) == 0:
        print('No spikes')
    else:
        spike_counts, _ = np.histogram(spike_times_in_window, bins=bin_edges)
        window_size_bins = int(window_size/bin_width)
        sigma_bins = int(sigma/bin_width)
        gaussian_window = gaussian(window_size_bins, sigma_bins)
        smoothed_event = convolve(spike_counts, gaussian_window, mode="same")
        firing_rate_matrix = smoothed_event / 0.1
    return firing_rate_matrix
# check if the psth is how we need it
def psth_per_unit(units, epoch_number, window, sigma):
    all_fr = []
    for indx, unit in enumerate(units):
        print(indx)
        df_unit = pd.DataFrame(unit)
        firing_rate_matrix = calculate_firing_rate(df_unit,
                                                   epoch_number,
                                                   window,
                                                   sigma)
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
epochs_to_skip = np.concatenate(epochs_to_avoid_acc)
df_curbd_epochs = pd.DataFrame(columns=['curbd_arr', 'epoch_n'])
df_curbd_epochs['curbd_arr'] = np.zeros(len(scores[2]))
from time import time

import numpy as np
from joblib import Parallel, delayed

def process_epoch(epoch_n, scores, 
                  areas_mapping, TT_list, spikes_folder, 
                  ChMap, nrun, 
                  window=125, sigma=10):
    start_time = time()
    area_responses = {area_name: [] for area_name in areas_mapping.values()}
    # Check if epoch should be skipped (adjust this logic as needed)
    if sum(epochs_to_skip == epoch_n) > 0:
        print('Skipping epoch')
        return None
    for index, area in areas_mapping.items():
        area_of_interest = areas_mapping[index]
        TT_per_area = [name for name in TT_list if area_of_interest in name]
        print(TT_per_area)
        
        UnitsInArea = units_area(TT_per_area, spikes_folder, ChMap)
        epoch_responses = psth_per_unit(UnitsInArea, epoch_n+1, window=125, sigma=10)
        plt.plot(epoch_responses[0])
        plt.show()
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
        nRunTrain=nrun,
        plotStatus=True,
        verbose=False,
        nRunFree=0
    )
    [curbd_arr, curbd_labels] = curbd.computeCURBD(model)
    end_time = time()  # End timing
    elapsed_time = end_time - start_time
    print(f"Processing epoch {epoch_n} took {elapsed_time:.2f} seconds")
    return {
        'epoch_n': epoch_n,
        'curbd_arr': curbd_arr,
        'curbd_labels': curbd_labels
    }

a = process_epoch(418, 
        scores, 
        areas_mapping, 
        TT_list, 
        spikes_folder, 
        ChMap, 200)
# Parallelize the processing
results = Parallel(n_jobs=1)(
    delayed(process_epoch)(
        epoch_n, 
        scores, 
        areas_mapping, 
        TT_list, 
        spikes_folder, 
        ChMap
    ) for epoch_n in range(len(scores[2]))
)

# Process results
for result in results:
    if result is not None:
        df_curbd_epochs.loc[result['epoch_n'], 'curbd_arr'] = result['curbd_arr']
        df_curbd_epochs.loc[result['epoch_n'], 'epoch_n'] = result['epoch_n']

#check which folders are being open when grabbing spikes

# df_curbd = df_curbd_epochs.iloc[0:787]
# mean_values = []
# for index, row in df_curbd.iterrows():
#     mean_acc = []
#     if isinstance(row['curbd_arr'], np.ndarray):
#         for vec in row['curbd_arr']:
#             mean_acc.append(np.mean(np.mean(vec)))
#         mean_values.append(np.mean(mean_acc))
#     else:
#         mean_values.append(np.nan)

# # Assign all values at once
# df_curbd['mean_curbd'] = mean_values
# scores_tocheck = scores[2][:787]
# AW = df_curbd[scores_tocheck == 1]['mean_curbd']
# NREM = df_curbd[scores_tocheck == 3]['mean_curbd']
# REM = df_curbd[scores_tocheck == 4]['mean_curbd']
# plt.boxplot([AW.dropna(), NREM.dropna(), REM.dropna()], labels=['AW', 'NREM', 'REM'])
# plt.show()
