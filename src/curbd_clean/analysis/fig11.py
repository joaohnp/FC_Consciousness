from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import combine_pvalues

projectPath = Path(__file__).resolve().parent.parent
h2_path = projectPath / "data" / "H2"
conditions = ["NREMvsREM"]
figures_path = projectPath.resolve().parent.parent / "figures" / "analysis"
all_results_df = pd.DataFrame()
for condition in conditions:
    resultsPath = h2_path / condition
    prcnt = [p.name for p in resultsPath.iterdir()]
    for chosen_prcnt in prcnt:
        p_values = []
        results_percent = [p.name for p in (resultsPath / chosen_prcnt).iterdir()]
        for chosen_pkl in results_percent:
            file = open(resultsPath / chosen_prcnt / chosen_pkl, "rb")
            my_dict = pickle.load(file)
            accu_perm = my_dict["accu_perm"]
            accu_true = my_dict["accu_true"]
            p_value = len(accu_perm[accu_perm >= accu_true]) / (len(accu_perm))
            if p_value == 0:
                p_value = 1e-16
            p_values.append(p_value)
        fisher_p = combine_pvalues(p_values, method="fisher")
        dict_results = {
            "Condition": condition,
            "P values": [p_values],
            "Cutoff": chosen_prcnt,
            "Fisher p": fisher_p.pvalue,
            "stat": fisher_p.statistic,
        }
        df_results = pd.DataFrame(dict_results, index=[0])

        all_results_df = pd.concat([all_results_df, df_results], ignore_index=True)
        dump_path = projectPath / "combined_p.pkl"
        all_results_df.to_pickle(dump_path)


def get_figure(df_epochs, fontsize=16):
    hist1 = df_epochs["accu_perm"]
    upper_bound = np.percentile(df_epochs["accu_perm"], 95)
    # Create figure and axes using plt.subplots()
    fig, ax1 = plt.subplots(figsize=(6, 6))
    ax1.hist(
        hist1,
        weights=np.ones_like(hist1) / len(hist1) * 100,
        label="Shuffled dataset",
        bins=10,
    )
    # stat, pvalue = stats.mannwhitneyu(dist1, dist2)
    ax1.set_title("r20 Fear Test", fontsize=fontsize + 2)
    ax1.set_ylabel("Fraction (%)", fontsize=fontsize + 1)
    ax1.set_xlabel("Classification accuracy", fontsize=fontsize + 1)
    ax1.tick_params(axis="both", labelsize=fontsize - 1)
    ax1.axvline(x=df_epochs.iloc[0]["accu_true"], color="r", label="Ground truth")
    ax1.axvline(x=upper_bound, color="grey", label="Upper bound")
    ax1.legend(fontsize=fontsize - 2)
    ax1.grid(True, linestyle="--", alpha=0.3)
    # ax1.set_xlim(0, 1)
    return fig


plt.rcParams["svg.fonttype"] = "none"

for condition in conditions:
    resultsPath = h2_path / condition
    prcnt = [p.name for p in resultsPath.iterdir()]
    for chosen_prcnt in prcnt:
        p_values = []
        results_percent = [p.name for p in (resultsPath / chosen_prcnt).iterdir()]
        for result in results_percent:
            if "r20Probe" in result:
                chosen_pkl = result
        file = open(resultsPath / chosen_prcnt / chosen_pkl, "rb")
        df = pd.DataFrame.from_dict(pickle.load(file))
        fig = get_figure(df)
        fig.suptitle(f"{chosen_prcnt} {condition}", fontsize=19)
        save_path = figures_path / f"exemplar_session{chosen_prcnt}_{condition}.svg"
        plt.savefig(
            save_path,
            transparent=True,  # Removes white background
            format="svg",  # Ensures SVG format
            # bbox_inches="tight",
        )


def get_population_figure(df_population, fontsize=16):
    conditions = df_population["Condition"].unique()
    cutoffs = df_population["Cutoff"].unique()
    fig, axes = plt.subplots(
        len(conditions), len(cutoffs), figsize=(18, 6), constrained_layout=True
    )
    condition = conditions[0]
    for j, cutoff in enumerate(cutoffs):
        # Filter data for this cutoff
        subset = df_population[df_population["Cutoff"] == cutoff]
        # Create scatter plot
        ax = axes[j]
        ax.scatter(subset["Session"], subset["Diff accuracy"] * 100)
        # Add dashed red line at y=0
        ax.axhline(y=0, color="red", linestyle="--", alpha=0.7, label="0% Diff")
        ax.set_xticks(range(len(subset)))
        ax.set_xticklabels(subset["Label"])
        # Set labels and title
        ax.set_xlabel("Session", fontsize=fontsize + 1)
        ax.set_ylabel("Difference (%)", fontsize=fontsize + 1)
        ax.set_title(f"{condition} - Cutoff: {cutoff}", fontsize=fontsize + 2)
        # Optional: rotate x-axis labels if session names are long
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(axis="both", labelsize=fontsize - 1)
    # ax1.legend(fontsize=fontsize - 2)
    return fig


plt.rcParams["svg.fonttype"] = "none"
df_population = pd.DataFrame()
for condition in conditions:
    resultsPath = h2_path / condition
    prcnt = [p.name for p in resultsPath.iterdir()]
    for chosen_prcnt in prcnt:
        p_values = []
        for results in (resultsPath / chosen_prcnt).iterdir():
            chosen_pkl = results.name
            file = open(resultsPath / chosen_prcnt / chosen_pkl, "rb")
            df = pd.DataFrame.from_dict(pickle.load(file))
            upper_bound = np.percentile(df["accu_perm"], 95)
            diff_accuracy = df.iloc[0]["accu_true"] - upper_bound
            if "r20Probe" in chosen_pkl:
                label = "r20 FT"
            elif "r20Fear" in chosen_pkl:
                label = "r20 FC"
            elif "r20Habituation" in chosen_pkl:
                label = "r20 Hab"
            elif "r14Probe" in chosen_pkl:
                label = "r14 FT"
            elif "r14Fear" in chosen_pkl:
                label = "r14 FC"
            elif "r14Habituation" in chosen_pkl:
                label = "r14 Hab"
            dict_results = {
                "Session": chosen_pkl,
                "Condition": condition,
                "P values": [p_values],
                "Cutoff": chosen_prcnt,
                "Diff accuracy": diff_accuracy,
                "Label": label,
            }
            df_results = pd.DataFrame(dict_results, index=[0])
            df_population = pd.concat([df_population, df_results], ignore_index=True)
fig = get_population_figure(df_population)
save_path = figures_path / "population_fig_H2.svg"
plt.savefig(
    save_path,
    transparent=True,  # Removes white background
    format="svg",  # Ensures SVG format
    bbox_inches="tight",
)

