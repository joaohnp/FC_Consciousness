import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

projectPath = os.path.dirname(os.path.abspath(__name__))
parentPath = os.path.dirname(projectPath)
scoresPath = os.path.join(parentPath, "SleepScores")
all_results_df = pd.DataFrame()
magnitudesPath = os.path.join(projectPath, "CurbdMagnitudes")
results_sessions = os.listdir(magnitudesPath)
all_results_df = pd.DataFrame()

alpha = 0.05
for chosen_results in results_sessions:
    print(f"Processing results for session {chosen_results}")
    file_of_interest = chosen_results + ".pickle"

    with open(os.path.join(scoresPath, chosen_results[0:-4] + ".pickle"), "rb") as fp:
        scores = pickle.load(fp)

    if len(scores) == 3:
        scores = scores[2]

    with open(os.path.join(magnitudesPath, chosen_results), "rb") as fp:
        df_curbd = pickle.load(fp)

    AW = df_curbd[(df_curbd["scores"] == 1) | (df_curbd["scores"] == 2)][
        "no_eye"
    ].dropna()
    NREM = df_curbd[df_curbd["scores"] == 3]["no_eye"].dropna()
    REM = df_curbd[df_curbd["scores"] == 4]["no_eye"].dropna()

    bottom_25_REM = REM[REM <= np.percentile(REM, 25)]
    top_5_NREM_cutoff = np.percentile(NREM, 5)
    percentage_below_NREM = (
        sum(bottom_25_REM <= top_5_NREM_cutoff) / len(bottom_25_REM)
    ) * 100

    dict_results = {
        "Session": chosen_results,
        "AW": len(AW),
        "NREM": len(NREM),
        "REM": len(REM),
        "NREM top 5%": np.round(top_5_NREM_cutoff, 3),
        "REM bottom 25%": len(bottom_25_REM),
        "Percentage below NREM": np.round(percentage_below_NREM, 2),
    }

    df_results = pd.DataFrame(dict_results, index=[0])
    all_results_df = pd.concat([all_results_df, df_results], ignore_index=True)
    dump_path = os.path.join(projectPath, "all_results_df_step2_H2.pkl")
    all_results_df.to_pickle(dump_path)
    print(f"Added results for session {chosen_results}")


def get_significance_stars(pvalue):
    if pvalue < 0.001:
        return "***"
    elif pvalue < 0.01:
        return "**"
    elif pvalue < 0.05:
        return "*"
    else:
        return "ns"  # not significant


def get_distribution_fig(NREM, REM, labels=["NREM", "REM"], fontsize=16):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.boxplot([NREM, REM], labels=labels)
    boxplot_positions = [1, 2]
    y_offset = 0.01
    # stat, pvalue = stats.mannwhitneyu(dist1, dist2)
    # ax1.set_title(f"CURBD Scores per stage p={pvalue:.2e}", fontsize=fontsize + 2)
    _, pvalue_nremrem = stats.mannwhitneyu(NREM, REM)
    # NREM vs REM
    stars_nremrem = get_significance_stars(pvalue_nremrem)
    y_nremrem = max(max(NREM), max(REM)) + y_offset
    ax1.text(
        (boxplot_positions[0] + boxplot_positions[1]) / 2,
        y_nremrem,
        stars_nremrem,
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )
    ax1.set_ylabel("Mean CURBD Score", fontsize=fontsize + 1)
    ax1.tick_params(axis="both", labelsize=fontsize - 1)
    ax1.grid(True, linestyle="--", alpha=0.3)
    weights2 = np.ones_like(NREM) / len(NREM) * 100
    weights3 = np.ones_like(REM) / len(REM) * 100
    all_data = np.concatenate([NREM, REM])
    bins = np.linspace(min(all_data), max(all_data), 31)
    ax2.hist(NREM, bins=bins, alpha=0.2, label=labels[0], weights=weights2)
    ax2.hist(REM, bins=bins, alpha=0.2, label=labels[1], weights=weights3)
    ax2.set_xlabel("Mean CURBD Score", fontsize=fontsize + 1)
    ax2.set_ylabel("Density (%)", fontsize=fontsize + 1)
    ax2.set_title("Distribution of CURBD Scores", fontsize=fontsize + 2)
    ax2.legend(fontsize=fontsize - 2)
    ax2.tick_params(axis="both", labelsize=fontsize - 1)
    ax2.grid(True, linestyle="--", alpha=0.3)
    return fig


# Plotting exemplar session
chosen_results = results_sessions[2]
with open(os.path.join(magnitudesPath, chosen_results), "rb") as fp:
    df_curbd = pickle.load(fp)
df_curbd = df_curbd.dropna()

NREM = df_curbd[df_curbd["scores"] == 3]["no_eye"].dropna()
REM = df_curbd[df_curbd["scores"] == 4]["no_eye"].dropna()

bottom_25_REM = REM[REM <= np.percentile(REM, 25)]
top_5_NREM = NREM[NREM >= np.percentile(NREM, 5)]


fontsize = 16
plt.rcParams["svg.fonttype"] = "none"  # Keep text as text, not paths
save_path = os.path.join(projectPath, "Figures", "exemplar_session.svg")

fig = get_distribution_fig(top_5_NREM, bottom_25_REM, labels=["NREM", "REM"])
fig.suptitle("Interareal connections - r20 Fear Test", fontsize=fontsize + 3)
plt.savefig(
    os.path.join(save_path),
    transparent=True,  # Removes white background
    format="svg",  # Ensures SVG format
    bbox_inches="tight",
)

animal_medians = []
for chosen_results in results_sessions:
    print(f"Processing results for session {chosen_results}")
    file_of_interest = chosen_results + ".pickle"
    with open(os.path.join(magnitudesPath, chosen_results), "rb") as fp:
        df_curbd = pickle.load(fp)
    AW = df_curbd[(df_curbd["scores"] == 1) | (df_curbd["scores"] == 2)][
        "no_eye"
    ].dropna()
    NREM = df_curbd[df_curbd["scores"] == 3]["no_eye"].dropna()
    REM = df_curbd[df_curbd["scores"] == 4]["no_eye"].dropna()
    top_5_NREM = NREM[NREM >= np.percentile(NREM, 5)]
    bottom_25_REM = REM[REM <= np.percentile(REM, 25)]
    toplot = [top_5_NREM.median(), bottom_25_REM.median()]
    animal_medians.append(toplot)


def plot_across_sessions(animal_medians):
    fig, ax1 = plt.subplots(1, 1, figsize=(12, 5))
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
    ]  # Example colors
    brain_states = ["NREM", "REM"]
    for i, animal_data in enumerate(animal_medians):
        ax1.plot(
            animal_data,
            "o-",
            color=colors[i],
            label=f"{results_sessions[i]}",
            markersize=12,
        )
    ax1.set_ylabel("Mean CURBD Score", fontsize=fontsize + 1)
    ax1.tick_params(axis="both", labelsize=fontsize - 1)
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend()  # Add a legend to identify the colors
    # Set x-axis ticks and labels to represent brain states
    ax1.set_xticks(range(len(brain_states)))
    ax1.set_xticklabels(brain_states)
    ax1.set_xlabel("Brain State", fontsize=fontsize + 1)
    return fig


fontsize = 16
fig = plot_across_sessions(animal_medians)
fig.suptitle("Top 5% NREM, bottom 25% REM - all sessions", fontsize=fontsize + 3)
save_path_all = os.path.join(projectPath, "Figures", "all_sessions.svg")
plt.savefig(
    os.path.join(save_path_all),
    transparent=True,  # Removes white background
    format="svg",  # Ensures SVG format
    bbox_inches="tight",
)
