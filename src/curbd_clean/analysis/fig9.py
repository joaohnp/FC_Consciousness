from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

projectPath = Path(__file__).resolve().parent.parent
sleepScores_path = projectPath / "data" / "SleepScores"
all_results_df = pd.DataFrame()
magnitudesPath = projectPath / "data" / "CurbdMagnitudes"
results_sessions = [p.name for p in magnitudesPath.iterdir()]
all_results_df = pd.DataFrame()
figures_path = projectPath.resolve().parent.parent / "figures" / "analysis"
alpha = 0.05
for chosen_results in results_sessions:
    print(f"Processing results for session {chosen_results}")
    file_of_interest = chosen_results + ".pickle"

    with open(sleepScores_path / (chosen_results[0:-4] + ".pickle"), "rb") as fp:
        scores = pickle.load(fp)

    if len(scores) == 3:
        scores = scores[2]

    with open(magnitudesPath / chosen_results, "rb") as fp:
        df_curbd = pickle.load(fp)

    AW = df_curbd[(df_curbd["scores"] == 1) | (df_curbd["scores"] == 2)][
        "no_eye"
    ].dropna()
    NREM = df_curbd[df_curbd["scores"] == 3]["no_eye"].dropna()
    REM = df_curbd[df_curbd["scores"] == 4]["no_eye"].dropna()

    top_25_NREM = NREM[NREM >= np.percentile(NREM, 75)]
    top_5_REM_cutoff = np.percentile(REM, 5)
    top_5_AW_cutoff = np.percentile(AW, 5)
    percentage_above_REM = (
        sum(top_25_NREM >= top_5_REM_cutoff) / len(top_25_NREM)
    ) * 100
    percentage_above_AW = (sum(top_25_NREM >= top_5_AW_cutoff) / len(top_25_NREM)) * 100

    dict_results = {
        "Session": chosen_results,
        "AW": len(AW),
        "NREM": len(NREM),
        "REM": len(REM),
        "AW top 5%": np.round(top_5_AW_cutoff, 3),
        "REM top 5%": np.round(top_5_REM_cutoff, 3),
        "NREM top 25%": len(top_25_NREM),
        "Percentage above AW": np.round(percentage_above_AW, 2),
        "Percentage above REM": np.round(percentage_above_REM, 2),
    }

    df_results = pd.DataFrame(dict_results, index=[0])
    all_results_df = pd.concat([all_results_df, df_results], ignore_index=True)
    dump_path = projectPath / "all_results_df_step2.pkl"
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


def get_distribution_fig(AW, NREM, REM, labels=["AW", "NREM", "REM"], fontsize=16):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.boxplot([AW, NREM, REM], tick_labels=labels)
    boxplot_positions = [1, 2, 3]
    y_offset = 0.01
    # stat, pvalue = stats.mannwhitneyu(dist1, dist2)
    # ax1.set_title(f"CURBD Scores per stage p={pvalue:.2e}", fontsize=fontsize + 2)
    _, pvalue_awrem = stats.mannwhitneyu(AW, REM)
    _, pvalue_nremrem = stats.mannwhitneyu(NREM, REM)
    _, pvalue_awnrem = stats.mannwhitneyu(AW, NREM)
    stars_awrem = get_significance_stars(pvalue_awrem)
    y_awrem = max(max(AW), max(REM)) + y_offset
    ax1.text(
        (boxplot_positions[0] + boxplot_positions[2]) / 2,
        y_awrem,
        stars_awrem,
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )
    # NREM vs REM
    stars_nremrem = get_significance_stars(pvalue_nremrem)
    y_nremrem = max(max(NREM), max(REM)) + y_offset
    ax1.text(
        (boxplot_positions[1] + boxplot_positions[2]) / 2,
        y_nremrem,
        stars_nremrem,
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )
    # AW vs NREM
    stars_awnrem = get_significance_stars(pvalue_awnrem)
    y_awnrem = (
        max(max(AW), max(NREM)) + y_offset * 2
    )  # Add more offset for the upper comparison
    ax1.text(
        (boxplot_positions[0] + boxplot_positions[1]) / 2,
        y_awnrem,
        stars_awnrem,
        ha="center",
        va="bottom",
        fontsize=fontsize,
    )
    ax1.set_ylabel("Mean CURBD Score", fontsize=fontsize + 1)
    ax1.tick_params(axis="both", labelsize=fontsize - 1)
    ax1.grid(True, linestyle="--", alpha=0.3)
    weights1 = np.ones_like(AW) / len(AW) * 100
    weights2 = np.ones_like(NREM) / len(NREM) * 100
    weights3 = np.ones_like(REM) / len(REM) * 100
    all_data = np.concatenate([AW, NREM, REM])
    bins = np.linspace(min(all_data), max(all_data), 31)
    ax2.hist(AW, bins=bins, alpha=0.2, label=labels[0], weights=weights1)
    ax2.hist(NREM, bins=bins, alpha=0.2, label=labels[1], weights=weights2)
    ax2.hist(REM, bins=bins, alpha=0.2, label=labels[2], weights=weights3)
    ax2.set_xlabel("Mean CURBD Score", fontsize=fontsize + 1)
    ax2.set_ylabel("Density (%)", fontsize=fontsize + 1)
    ax2.set_title("Distribution of CURBD Scores", fontsize=fontsize + 2)
    ax2.legend(fontsize=fontsize - 2)
    ax2.tick_params(axis="both", labelsize=fontsize - 1)
    ax2.grid(True, linestyle="--", alpha=0.3)
    return fig


# Plotting exemplar session
chosen_results = results_sessions[5]
with open(magnitudesPath / chosen_results, "rb") as fp:
    df_curbd = pickle.load(fp)
df_curbd = df_curbd.dropna()

AW = df_curbd[(df_curbd["scores"] == 1) | (df_curbd["scores"] == 2)]["no_eye"].dropna()
NREM = df_curbd[df_curbd["scores"] == 3]["no_eye"].dropna()
REM = df_curbd[df_curbd["scores"] == 4]["no_eye"].dropna()

top_25_NREM = NREM[NREM >= np.percentile(NREM, 75)]
bottom_5_REM = REM[REM <= np.percentile(REM, 5)]
bottom_5_AW = AW[AW <= np.percentile(AW, 5)]


fontsize = 16
plt.rcParams["svg.fonttype"] = "none"  # Keep text as text, not paths
save_path = figures_path / "fig_9_panel_a.svg"

fig = get_distribution_fig(
    bottom_5_AW, top_25_NREM, bottom_5_REM, labels=["AW", "NREM", "REM"]
)
fig.suptitle(
    "Top 25% NREM, bottom 5% AW and REM - r20 Fear Test", fontsize=fontsize + 3
)
plt.savefig(
    save_path,
    transparent=True,  # Removes white background
    format="svg",  # Ensures SVG format
    bbox_inches="tight",
)

all_AW = []
all_NREM = []
all_REM = []
for chosen_results in results_sessions:
    file_of_interest = chosen_results + ".pickle"
    with open(magnitudesPath / chosen_results, "rb") as fp:
        df_curbd = pickle.load(fp)
    AW = df_curbd[(df_curbd["scores"] == 1) | (df_curbd["scores"] == 2)][
        "no_eye"
    ].dropna()
    NREM = df_curbd[df_curbd["scores"] == 3]["no_eye"].dropna()
    REM = df_curbd[df_curbd["scores"] == 4]["no_eye"].dropna()
    top_25_NREM = NREM[NREM >= np.percentile(NREM, 75)]
    bottom_5_REM = REM[REM <= np.percentile(REM, 5)]
    bottom_5_AW = AW[AW <= np.percentile(AW, 5)]
    all_AW.append(bottom_5_AW)
    all_NREM.append(top_25_NREM)
    all_REM.append(bottom_5_REM)

all_AW = np.concatenate(all_AW)
all_NREM = np.concatenate(all_NREM)
all_REM = np.concatenate(all_REM)

animal_medians = []
for chosen_results in results_sessions:
    print(f"Processing results for session {chosen_results}")
    file_of_interest = chosen_results + ".pickle"
    with open(magnitudesPath / chosen_results, "rb") as fp:
        df_curbd = pickle.load(fp)
    AW = df_curbd[(df_curbd["scores"] == 1) | (df_curbd["scores"] == 2)][
        "no_eye"
    ].dropna()
    NREM = df_curbd[df_curbd["scores"] == 3]["no_eye"].dropna()
    REM = df_curbd[df_curbd["scores"] == 4]["no_eye"].dropna()
    top_25_NREM = NREM[NREM >= np.percentile(NREM, 75)]
    bottom_5_REM = REM[REM <= np.percentile(REM, 5)]
    bottom_5_AW = AW[AW <= np.percentile(AW, 5)]
    toplot = [bottom_5_AW.median(), top_25_NREM.median(), bottom_5_REM.median()]
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
    brain_states = ["AW", "NREM", "REM"]
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
fig.suptitle("Top 25% NREM, bottom 5% AW and REM - all sessions", fontsize=fontsize + 3)
save_path_all = figures_path / "fig_9_panel_b.svg"
plt.savefig(
    save_path_all,
    transparent=True,  # Removes white background
    format="svg",  # Ensures SVG format
    bbox_inches="tight",
)
