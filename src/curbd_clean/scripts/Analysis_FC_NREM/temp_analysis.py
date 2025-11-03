# %%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils_expinfo import get_experimental_info

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

hard_drive = "/run/media/joaohnp/Elements"
hd_session=hd_mapping[2]

pickles_path = os.path.join(hard_drive, 'session_result')
pickle_files = [f for f in os.listdir(pickles_path) if f.endswith('.pkl')]
pickle_files.sort(key=lambda x: int(x.split('.')[0]))
dfs = []
def process_matrix(input_matrix):
    output_matrix = np.zeros((4, 4))
    arr = input_matrix['curbd_arr']
    lbl = input_matrix['curbd_labels']
    # Loop through each position in the 4x4 matrix
    for i in range(4):
        for j in range(4):
            # Get the quadrant
            quadrant = arr[i, j]
            # print(lbl[i,j])
            # print(quadrant)
            # mean_axis0 = np.mean(np.mean(np.abs(quadrant), axis=0))
            mean_axis0 = np.mean(np.mean(quadrant, axis=0))
            # print(mean_axis0)
            output_matrix[i, j] = mean_axis0
    return output_matrix

def check_files(pkl_files, epoch_amount):
    mean_values = []
    median_values = []
    for file in pkl_files[:epoch_amount]:
        # print(file)
        file_path = os.path.join(pickles_path, file)
        df = pd.read_pickle(file_path)
        if df is None:
            mean_values.append(np.nan)
        elif isinstance(df['curbd_arr'], np.ndarray):
            vec = df
            if vec['curbd_arr'].size == 0:
                mean_values.append(0)
            else:
                value = process_matrix(vec)
                mask = np.array([
                                [1, 0, 0, 1],
                                [0, 1, 1, 0],
                                [1, 1, 1, 1], 
                                [0, 0, 1, 1]], dtype=bool)
                avg = np.mean(np.abs(value[~mask]))
                # avg = np.median(value[~np.eye(4, dtype=bool)])
                # avg = np.mean(value)
                # avg = np.mean(abs(value))
                # avg = np.median(value)
                # avg = np.median(np.abs(value))
                median_vl = np.median(np.abs(value[~np.eye(4, dtype=bool)]))
                if np.isfinite(avg):
                    mean_values.append(avg)
                    median_values.append(median_vl)
                else:
                    mean_values.append(0)
    return median_values, mean_values

median_values, mean_values = check_files(pickle_files, -1)
df_curbd = pd.DataFrame(columns=['mean_curbd', 'scores'])
df_curbd['mean_curbd'] = mean_values
df_curbd['scores'] = scores[2][:len(mean_values)]    
scores_tocheck = scores[2][:len(mean_values)]
AW = df_curbd[scores_tocheck == 1]['mean_curbd']
QW = df_curbd[scores_tocheck == 2]['mean_curbd']
WAKE = pd.concat([AW, QW])
NREM = df_curbd[scores_tocheck == 3]['mean_curbd']
REM = df_curbd[scores_tocheck == 4]['mean_curbd']
UND = df_curbd[scores_tocheck == 5]['mean_curbd']
plt.boxplot([WAKE.dropna(), NREM.dropna(), REM.dropna(), AW.dropna(), QW.dropna()],
            labels=['WAKE', 'NREM', 'REM', 'AW', 'QW'])
plt.show()

# len(AW) + len(QW) + len(NREM) + len(REM) + len(UND)
AW.describe()
NREM.describe()
REM.describe()
fig, ax = plt.subplots(figsize=(10, 6))
all_data = np.concatenate([WAKE.dropna(), NREM.dropna(), REM.dropna()])
bins = np.linspace(min(all_data), max(all_data), 50)  # 30 bins across the range
# plt.hist(WAKE.dropna(), bins=bins, alpha=0.2, label='WAKE', density=True)
plt.hist(NREM.dropna(), bins=bins, alpha=0.2, label='NREM', density=True)
plt.hist(WAKE.dropna(), bins=bins, alpha=0.2, label='REM', density=True)
plt.xlabel('Mean CURBD Score')
plt.ylabel('Amount')
plt.title('Distribution of CURBD Scores Across Sleep States')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

pkl_umberto = []
# for file in pickle_files[:-1]:
#     file_path = os.path.join(pickles_path, file)
#     df = pd.read_pickle(file_path)
#     if df is None:
#         pkl_umberto.append(np.nan)
#     elif isinstance(df['curbd_arr'], np.ndarray):
#         vec = df['curbd_arr']
#         if vec.size == 0:
#             pkl_umberto.append(0)
#         else:
#             value = process_matrix(vec)
#             pkl_umberto.append(value)

# with open('pkl_umberto', "wb") as output_file:
#     pickle.dump(pkl_umberto, output_file)
df_curbd = pd.DataFrame(columns=['mean_curbd', 'scores'])
df_curbd['mean_curbd'] = median_values
df_curbd['scores'] = scores[2][:len(median_values)]    
scores_tocheck = scores[2][:len(median_values)]
AW = df_curbd[scores_tocheck == 1]['mean_curbd']
QW = df_curbd[scores_tocheck == 2]['mean_curbd']
# WAKE = pd.concat([AW, QW])
WAKE = AW.dropna()[:300]
NREM = df_curbd[scores_tocheck == 3]['mean_curbd']
REM = df_curbd[scores_tocheck == 4]['mean_curbd']
plt.boxplot([WAKE.dropna(), NREM.dropna(), REM.dropna(), AW.dropna(), QW.dropna()],
            labels=['WAKE', 'NREM', 'REM', 'AW', 'QW'])
plt.show()


fig, ax = plt.subplots(figsize=(10, 6))
all_data = np.concatenate([WAKE.dropna(), NREM.dropna(), REM.dropna()])
bins = np.linspace(min(all_data), max(all_data), 50)  # 30 bins across the range
plt.hist(WAKE.dropna(), bins=bins, alpha=0.2, label='WAKE', density=True)
plt.hist(NREM.dropna(), bins=bins, alpha=0.2, label='NREM', density=True)
plt.hist(REM.dropna(), bins=bins, alpha=0.2, label='REM', density=True)
plt.xlabel('Mean CURBD Score')
plt.ylabel('Amount')
plt.title('Distribution of CURBD Scores Across Sleep States')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.tight_layout()
plt.show()

from scipy import stats

# For unpaired t-test
stats.mannwhitneyu(REM.dropna(), NREM.dropna())

import numpy as np
from scipy import stats
from statsmodels.stats.multitest import multipletests

# Bootstrap approach


def get_stats(stage1, stage2, nrep, group_size, replace):
    n_boots = nrep
    bootstrap_results = []
    for _ in range(n_boots):
        # Sample with replacement from both distributions
        boot1 = np.random.choice(WAKE.dropna(), size=group_size, replace=replace)
        boot2 = np.random.choice(NREM.dropna(), size=group_size, replace=replace)
        stat, pval = stats.mannwhitneyu(boot1, boot2)
        bootstrap_results.append(pval)
    _, adjusted_pvals, _, _ = multipletests(bootstrap_results, method='bonferroni')
    plt.hist(adjusted_pvals, bins=100)
    # plt.axvline(0.05, color= 'r')
    plt.show()

get_stats(NREM, WAKE, 1000, 500, True)