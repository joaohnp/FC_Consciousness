# %%
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sig
import seaborn as sns
from scipy import stats
from scipy.signal import welch
from statsmodels.stats.multitest import multipletests
import sys
from scipy.integrate import simpson as simps

sys.path.insert(0, "/home/joaohnp/github/curbd_jp/src")
from curbd_clean.plotting.io import save_figure as io_save_figure


def accelerometer_resultant(x_axis, y_axis, z_axis):
    sos = np.power(x_axis, 2) + np.power(y_axis, 2) + np.power(z_axis, 2)
    accel = np.sqrt(sos, dtype=np.float64)
    order = 2
    sampling_freq = 1000
    cutoff_freq = 2
    normalized_cutoff_freq = 2 * cutoff_freq / sampling_freq
    numerator_coeffs, denominator_coeffs = sig.butter(order, normalized_cutoff_freq)
    filtered_signal = sig.lfilter(numerator_coeffs, denominator_coeffs, accel)
    nogravity_sig = accel - filtered_signal
    return nogravity_sig


def get_top_10_percent(values):
    threshold = np.percentile(values, 90)
    top_indices = np.where(values >= threshold)[0]
    return top_indices


def process_matrix(input_matrix):
    output_matrix = np.zeros((4, 4))
    arr = input_matrix["curbd_arr"]
    lbl = input_matrix["curbd_labels"]
    # Loop through each position in the 4x4 matrix
    for i in range(4):
        for j in range(4):
            quadrant = arr[i, j]
            # mean_axis0 = np.mean(np.mean(np.abs(quadrant), axis=0))
            mean_axis0 = np.mean(np.mean(quadrant, axis=0))
            output_matrix[i, j] = mean_axis0
    return output_matrix


def check_files(
    pkl_files, epoch_amount, masks, mask_names=["all", "no_eye", "no_hpc", "mask_intra"]
):
    results = {name: [] for name in mask_names}
    for file in pkl_files[:epoch_amount]:
        file_path = curbd_results_path / file
        df = pd.read_pickle(file_path)
        if df is None:
            for name in mask_names:
                results[name].append(np.nan)
            continue
        elif isinstance(df["curbd_arr"], np.ndarray):
            vec = df
            if vec["curbd_arr"].size == 0:
                for name in mask_names:
                    results[name].append(0)
            else:
                value = process_matrix(vec)
                for mask, name in zip(masks, mask_names):
                    avg = np.mean(np.abs(value[mask]))
                    results[name].append(np.nan if not np.isfinite(avg) else avg)
    return pd.DataFrame(results)


projectPath = Path(__file__).resolve().parent.parent
sleepScores_path = projectPath / "data" / "SleepScores"
lfp_channels_path = projectPath / "data" / "pilot_data_fig6"


file_of_interest = "r20ProbeSleep.pickle"
with open(sleepScores_path / file_of_interest, "rb") as fp:
    scores = pickle.load(fp)


A1_path = lfp_channels_path / "r20ch_10_ds.pickle"
HPC_path = lfp_channels_path / "r20ch_41_ds.pickle"
PFC_path = lfp_channels_path / "r20ch_65_ds.pickle"
x_path = lfp_channels_path / "r20ch_130_ds.pickle"
y_path = lfp_channels_path / "r20ch_128_ds.pickle"
z_path = lfp_channels_path / "r20ch_129_ds.pickle"
with open(A1_path, "rb") as fp:
    A1_channel = pickle.load(fp)
with open(HPC_path, "rb") as fp:
    HPC_channel = pickle.load(fp)

with open(PFC_path, "rb") as fp:
    PFC_channel = pickle.load(fp)

with open(x_path, "rb") as fp:
    x_channel = pickle.load(fp)

with open(y_path, "rb") as fp:
    y_channel = pickle.load(fp)

with open(z_path, "rb") as fp:
    z_channel = pickle.load(fp)

list_epochs_NREM = np.where(scores[2] == 3)
list_epochs_awake = np.where((scores[2] == 1) | (scores[2] == 2) | (scores[2] == 2))

list_NREM_power = []
for epoch in list_epochs_NREM[0]:
    leftlim = int((epoch * 5))
    rightlim = int((epoch * 5) + 5)
    f, pxx = welch(
        A1_channel[(leftlim * 1000) : (rightlim * 1000)],
        fs=1000,
        window="flattop",
        nfft=1024,
        scaling="spectrum",
    )
    power = sum(pxx[(f > 0) & (f <= 4)])
    list_NREM_power.append(power)

list_awake = []
for epoch in list_epochs_awake[0]:
    leftlim = int((epoch * 5))
    rightlim = int((epoch * 5) + 5)
    movement = accelerometer_resultant(
        x_channel[(leftlim * 1000) : (rightlim * 1000)],
        y_channel[(leftlim * 1000) : (rightlim * 1000)],
        z_channel[(leftlim * 1000) : (rightlim * 1000)],
    )
    area = simps(abs(movement))
    list_awake.append(area)


top_indices_NREM = get_top_10_percent(list_NREM_power)
top_indices_wake = get_top_10_percent(list_awake)
wake_tocheck = list_epochs_awake[0][top_indices_wake]
nrem_tocheck = list_epochs_NREM[0][top_indices_NREM]

curbd_results_path = projectPath / "data" / "pilot_data_fig6" / "curbd_results_session"
pickle_files = [p.name for p in curbd_results_path.glob("*.pkl")]
pickle_files.sort(key=lambda x: int(x.split(".")[0]))

mask_all = np.array(
    [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]], dtype=bool
)
mask_no_eye = np.array(
    [[0, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1], [1, 1, 1, 0]], dtype=bool
)
mask_no_hpc = np.array(
    [[0, 1, 1, 0], [1, 0, 0, 1], [0, 0, 0, 0], [1, 1, 0, 0]], dtype=bool
)
mask_intra = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=bool
)
masks = [mask_all, mask_no_eye, mask_no_hpc, mask_intra]
mean_values = check_files(pickle_files, -1, masks=masks)
df_curbd = mean_values
df_curbd["scores"] = scores[2][: len(mean_values)]


AW = df_curbd[(df_curbd["scores"] == 1) | (df_curbd["scores"] == 2)]["no_eye"]
NREM = df_curbd[df_curbd["scores"] == 3]["no_eye"]


def analyze_quartile_overlap(dist1, dist2, labels=["Dist 1", "Dist 2"], fontsize=16):
    # Calculate quartiles
    n_boots = 1000
    bootstrap_results = []
    for _ in range(n_boots):
        # Sample with replacement from both distributions
        boot1 = np.random.choice(dist1.dropna(), size=500, replace=True)
        boot2 = np.random.choice(dist2.dropna(), size=500, replace=True)
        stat, pval = stats.mannwhitneyu(boot1, boot2)
        bootstrap_results.append(pval)
    _, adjusted_pvals, _, _ = multipletests(bootstrap_results, method="bonferroni")
    quartiles_d1 = np.percentile(dist1, [25, 50, 75])
    quartiles_d2 = np.percentile(dist2, [25, 50, 75])
    q1_overlap = np.sum((dist1 <= quartiles_d2[0])) / len(dist1) * 100
    q2_overlap = (
        np.sum((dist1 > quartiles_d2[0]) & (dist1 <= quartiles_d2[1]))
        / len(dist1)
        * 100
    )
    q3_overlap = (
        np.sum((dist1 > quartiles_d2[1]) & (dist1 <= quartiles_d2[2]))
        / len(dist1)
        * 100
    )
    q4_overlap = np.sum((dist1 > quartiles_d2[2])) / len(dist1) * 100
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.boxplot([dist1, dist2], tick_labels=labels)
    stat, pvalue = stats.mannwhitneyu(dist1, dist2)
    ax1.set_title(f"CURBD Scores per stage p={pvalue:.2e}", fontsize=fontsize + 2)
    ax1.set_ylabel("Mean CURBD Score", fontsize=fontsize + 1)
    ax1.tick_params(axis="both", labelsize=fontsize - 1)
    ax1.grid(True, linestyle="--", alpha=0.3)
    weights1 = np.ones_like(dist1) / len(dist1) * 100
    weights2 = np.ones_like(dist2) / len(dist2) * 100
    ax2.hist(dist1, bins=20, alpha=0.2, label=labels[0], weights=weights1)
    ax2.hist(dist2, bins=20, alpha=0.2, label=labels[1], weights=weights2)
    ax2.set_xlabel("Mean CURBD Score", fontsize=fontsize + 1)
    ax2.set_ylabel("Density (%)", fontsize=fontsize + 1)
    ax2.set_title("Distribution of CURBD Scores", fontsize=fontsize + 2)
    ax2.legend(fontsize=fontsize - 2)
    ax2.tick_params(axis="both", labelsize=fontsize - 1)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.text(
        0.6,
        0.5,
        f"{labels[0]} within \n{labels[1]} quartiles:\n"
        + f"Q1: {q1_overlap:.1f}%\n"
        + f"Q2: {q2_overlap:.1f}%\n"
        + f"Q3: {q3_overlap:.1f}%\n"
        + f"Q4: {q4_overlap:.1f}%",
        transform=ax2.transAxes,
    )
    return fig


# Usage:
fontsize = 16
plt.rcParams["svg.fonttype"] = "none"  # Keep text as text, not paths
figures_path = projectPath / "Figures" / "Overlap"
AW = df_curbd[(df_curbd["scores"] == 1) | (df_curbd["scores"] == 2)]["no_eye"]
NREM = df_curbd[df_curbd["scores"] == 3]["no_eye"]
fig = analyze_quartile_overlap(NREM.dropna(), AW.dropna(), labels=["NREM", "WAKE"])
fig.suptitle("Interareal connections - all epochs", fontsize=fontsize + 3)
io_save_figure(fig, "Overlap/Inter_noeye_all")

AW = df_curbd[(df_curbd["scores"] == 1) | (df_curbd["scores"] == 2)]["mask_intra"]
NREM = df_curbd[df_curbd["scores"] == 3]["mask_intra"]
fig = analyze_quartile_overlap(NREM.dropna(), AW.dropna(), labels=["NREM", "WAKE"])
fig.suptitle("Intra connections - all epochs", fontsize=fontsize + 3)
io_save_figure(fig, "Overlap/Intra_allepochs")

AW = df_curbd[(df_curbd["scores"] == 1) | (df_curbd["scores"] == 2)]["no_hpc"]
NREM = df_curbd[df_curbd["scores"] == 3]["no_hpc"]
fig = analyze_quartile_overlap(NREM.dropna(), AW.dropna(), labels=["NREM", "WAKE"])
fig.suptitle("Interareal connections - all epochs - no HPC", fontsize=fontsize + 3)
io_save_figure(fig, "Overlap/Inter_allepochs_noHPC")

AW = df_curbd["no_eye"].iloc[wake_tocheck]
NREM = df_curbd["no_eye"].iloc[nrem_tocheck]
fig = analyze_quartile_overlap(NREM.dropna(), AW.dropna(), labels=["NREM", "WAKE"])
fig.suptitle("Interareal connections - selected epochs", fontsize=fontsize + 3)
io_save_figure(fig, "Overlap/Inter_selected")

AW = df_curbd["no_hpc"].iloc[wake_tocheck]
NREM = df_curbd["no_hpc"].iloc[nrem_tocheck]
fig = analyze_quartile_overlap(NREM.dropna(), AW.dropna(), labels=["NREM", "WAKE"])
fig.suptitle("Interareal connections - selected epochs - no HPC", fontsize=fontsize + 3)
io_save_figure(fig, "Overlap/Inter_selected_nohpc")

REM = df_curbd[(df_curbd["scores"] == 4)]["no_eye"]
NREM = df_curbd[df_curbd["scores"] == 3]["no_eye"]
fig = analyze_quartile_overlap(NREM.dropna(), REM.dropna(), labels=["NREM", "REM"])
fig.suptitle("Interareal connections", fontsize=fontsize + 3)
io_save_figure(fig, "Overlap/Inter_allepochs_REM")

REM = df_curbd[(df_curbd["scores"] == 4)]["no_hpc"]
NREM = df_curbd[df_curbd["scores"] == 3]["no_hpc"]
fig = analyze_quartile_overlap(NREM.dropna(), REM.dropna(), labels=["NREM", "REM"])
fig.suptitle("Interareal connections - no HPC", fontsize=fontsize + 3)
io_save_figure(fig, "Overlap/Inter_allepochs_REM_nohpc")

permutation_results_path = (
    projectPath / "data" / "pilot_data_fig6" / "permutation_results"
)
perm_75 = permutation_results_path / "permutation_75.pkl"
your_object = pickle.load(open(perm_75, "rb"))
plt.hist(
    your_object["accu_perm"],
    weights=np.ones_like(your_object["accu_perm"])
    / len(your_object["accu_perm"])
    * 100,
    label="Shuffled dataset",
)
plt.axvline(x=your_object["accu_true"], color="r", label="Ground Truth")
plt.xlabel("Classification accuracy", fontsize=16)
plt.ylabel("Density (%)", fontsize=16)
plt.tick_params(axis="both", labelsize=16)
plt.legend(fontsize=14)
io_save_figure(plt.gcf(), "Overlap/panelG")
