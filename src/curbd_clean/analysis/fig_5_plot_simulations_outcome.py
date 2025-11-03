from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.stats import ttest_ind
import sys

sys.path.insert(0, "/home/joaohnp/github/curbd_jp/src")
from curbd_clean.plotting.io import save_figure as io_save_figure


def plot_bar_poly(data, ax, ylabel, ylim):
    iterations = list(data.keys())
    avg_values = [np.mean(np.absolute(data[i])) for i in iterations]
    sem_values = [
        np.std(np.absolute(data[i])) / np.sqrt(len(data[i])) for i in iterations
    ]

    ax.bar(
        iterations,
        avg_values,
        yerr=sem_values,
        color="black",
        alpha=0.8,
        capsize=5,
    )
    ax.axvline(
        x=9.5,
        color="red",
        linestyle="--",
        label=f"Cutoff",
    )

    ax.set_xlabel("Number of neurons")
    ax.set_ylabel(ylabel)
    ax.set_xticks(iterations)
    ax.set_ylim([0, ylim])
    ax.legend()


def compare_subsets(reference_data, all_data, alpha=0.01):
    num_comparisons = len(all_data) - 1  # Minus 1 to exclude the reference subset
    corrected_alpha = alpha / num_comparisons
    significant_comparisons = {}

    for subset_name, subset_data in all_data.items():
        if subset_data is reference_data:
            continue

        res = ttest_ind(reference_data, subset_data, equal_var=False)
        p_value = res.pvalue
        if p_value < corrected_alpha:
            significant_comparisons[subset_name] = p_value

    return significant_comparisons


project_path = Path(__file__).resolve().parent.parent
pickle_path = project_path / "data" / "pvar_analysis"

pvar_results_path = pickle_path / "pvar_results.pkl"
with open(pvar_results_path, "rb") as handle:
    pvar_results = pickle.load(handle)

fig = plt.figure(figsize=(6, 5))
gs = GridSpec(1, 1)
ax1 = fig.add_subplot(gs[0, 0])
plot_bar_poly(pvar_results, ax1, "pVar", 0.15)
ax1.set_title("Model fit on different subsets of units")
plt.tight_layout()
io_save_figure(fig, "fig5.png", dpi=300)
plt.show()

# Statistical comparisons (as-is)
reference_subset_name = 24
reference_subset_data = pvar_results[reference_subset_name]
significant_results = compare_subsets(reference_subset_data, pvar_results)
for subset, p_value in significant_results.items():
    print(
        f"Subset {reference_subset_name} is significantly different from subset {subset} (p-value: {p_value:.3f})"
    )
