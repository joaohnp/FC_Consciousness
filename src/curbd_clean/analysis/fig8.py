from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import numpy as np

projectPath = Path(__file__).resolve().parent.parent
n_epochs_pkl_path = projectPath / "data" / "n_epochs"
figures_path = projectPath.resolve().parent.parent / "figures" / "analysis"
with open(n_epochs_pkl_path / "n_epochs.pkl", "rb") as fp:
    df_epochs = pickle.load(fp)
    df_epochs = df_epochs.drop(
        df_epochs[df_epochs["Session"] == "r16FearConditioningSleep"].index,
        inplace=False,
    )


def plot_histogram_with_markers(
    ax, hist_data, ground_truth, upper_bound, title, fontsize
):
    """
    Helper function to plot a single histogram with ground truth and upper bound markers.

    Args:
        ax (matplotlib.axes.Axes): The axes object to plot on.
        hist_data (numpy.ndarray): The array of histogram data.
        ground_truth (float): The ground truth accuracy value.
        upper_bound (float): The 95% upper bound value.
        title (str): The title for the subplot.
        fontsize (int): The base font size for the plot elements.
    """
    ax.hist(
        hist_data,
        weights=np.ones_like(hist_data) / len(hist_data) * 100,
        label="Shuffled dataset",
    )
    ax.set_title(title, fontsize=fontsize + 2)
    ax.set_ylabel("Fraction (%)", fontsize=fontsize + 1)
    ax.set_xlabel("Classification accuracy", fontsize=fontsize + 1)
    ax.tick_params(axis="both", labelsize=fontsize - 1)
    ax.axvline(x=ground_truth, color="r", label="Ground Truth")
    ax.axvline(
        x=upper_bound,
        color="grey",
        label="Upper bound",
    )
    ax.legend(fontsize=fontsize - 2)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_xlim(0.40, 0.65)


def get_figure(df_epochs, fontsize=16):
    data_points = [
        (0, "NREM vs AW - r20 Fear Test"),
        (1, "NREM vs REM - r20 Fear Test"),
        (2, "AW vs REM - r20 Fear Test"),
    ]
    nremaw_diff = (
        df_epochs[df_epochs["Pair"] == "NREM vs AW"]["Ground truth accuracy"]
        - df_epochs[df_epochs["Pair"] == "NREM vs AW"]["95% upper bound"]
    ) * 100
    remnrem_diff = (
        df_epochs[df_epochs["Pair"] == "NREM vs REM"]["Ground truth accuracy"]
        - df_epochs[df_epochs["Pair"] == "NREM vs REM"]["95% upper bound"]
    ) * 100
    remaw_diff = (
        df_epochs[df_epochs["Pair"] == "AW vs REM"]["Ground truth accuracy"]
        - df_epochs[df_epochs["Pair"] == "AW vs REM"]["95% upper bound"]
    ) * 100
    condition_pairs_diff = [
        nremaw_diff.tolist(),
        remnrem_diff.tolist(),
        remaw_diff.tolist(),
    ]  # Convert to list for histogram plotting
    # Create the figure and axes
    fig, ax = plt.subplots(2, 3, figsize=(18, 8))  # Keep the increased figsize
    # Plot each histogram using the helper function
    for i, (index, title) in enumerate(data_points):
        hist_data = df_epochs.iloc[index]["accu_perm"]
        ground_truth = df_epochs.iloc[index]["Ground truth accuracy"]
        upper_bound = df_epochs.iloc[index]["95% upper bound"]
        plot_histogram_with_markers(
            ax[0, i], hist_data, ground_truth, upper_bound, title, fontsize
        )
    pair_titles = [
        "NREM vs AW Difference",
        "NREM vs REM Difference",
        "AW vs REM Difference",
    ]
    legend_labels = [
        "r14 FT",
        "r20 FC",
        "r20 FT",
        "r14 Hab",
        "r14 FC",
        "r20 Hab",
    ]
    for i, data in enumerate(condition_pairs_diff):
        # Use the index of the data points for the x-axis and the difference values for the y-axis
        ax[1, i].scatter(
            range(len(data)),
            data,
            alpha=0.7,
        )
        ax[1, i].set_title(pair_titles[i], fontsize=fontsize)
        ax[1, i].set_xlabel(
            "Session",
            fontsize=fontsize - 1,  # Adjusted label
        )
        ax[1, i].set_ylabel(
            "Ground Truth - 95% Upper Bound (%)", fontsize=fontsize - 1
        )  # Adjusted label
        ax[1, i].tick_params(axis="both", labelsize=fontsize - 2)
        ax[1, i].grid(True, linestyle="--", alpha=0.3)
        ax[1, i].set_xticks(range(len(data)))
        ax[1, i].set_xticklabels(
            legend_labels[: len(data)], rotation=45, ha="right"
        )  # Added rotation for readability
        ax[1, i].axhline(y=0, color="red", linestyle="--", alpha=0.3)
        # You might want to adjust the x-axis limits or add x-ticks depending on how you want to represent the points.
        # For a small number of points, explicitly showing the indices might be helpful.
    plt.tight_layout()
    return fig


plt.rcParams["svg.fonttype"] = "none"
save_path = figures_path / "fig8.svg"
fig = get_figure(df_epochs)
plt.savefig(
    save_path,
    transparent=True,  # Removes white background
    format="svg",  # Ensures SVG format
    bbox_inches="tight",
)
