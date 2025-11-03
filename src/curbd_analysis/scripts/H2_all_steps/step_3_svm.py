# %%
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from scipy import stats
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, LeaveOneOut
from sklearn.svm import SVC

projectPath = os.path.dirname(os.path.abspath(__name__))
parentPath = os.path.dirname(projectPath)
scoresPath = os.path.join(parentPath, "SleepScores")
magnitudesPath = os.path.join(projectPath, "CurbdMagnitudes")
results_sessions = os.listdir(magnitudesPath)
all_results_df = pd.DataFrame()

alpha = 0.05


def get_xy(condition1, condition2):
    X = np.concatenate([condition1, condition2])
    y = np.concatenate([np.zeros(len(condition1)), np.ones(len(condition2))])
    return X, y


model = SVC(class_weight="balanced")
for chosen_results in tqdm.tqdm(results_sessions, desc="Processing files"):
    print(f"Processing results for session {chosen_results}")
    with open(os.path.join(magnitudesPath, chosen_results), "rb") as fp:
        df_curbd = pickle.load(fp)
    NREM = df_curbd[df_curbd["scores"] == 3]["no_eye"].dropna()
    REM = df_curbd[df_curbd["scores"] == 4]["no_eye"].dropna()
    bottom_25_REM = REM[REM <= np.percentile(REM, 25)]
    bottom_25_REM.name = "bottom_25_REM"
    top_5_NREM = NREM[NREM >= np.percentile(NREM, 5)]
    top_5_NREM.name = "top_5_NREM"
    pairs_to_test = [
        [bottom_25_REM, top_5_NREM],
    ]
    loo = LeaveOneOut()
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    for pair in pairs_to_test:
        X, y = get_xy(pair[0], pair[1])
        loo = LeaveOneOut()
        predictions = []
        true_labels = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            X_train = X_train.reshape(-1, 1)
            X_test = X_test.reshape(-1, 1)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions.append(y_pred[0])
            true_labels.append(y_test[0])
        loo_accuracy = accuracy_score(true_labels, predictions)
        print(f"Accuracy for {pair[0].name} and {pair[1].name}: {loo_accuracy}")
        accuracy_permutation = []
        print("Starting shuffling procedure")
        for rand_perm in range(1000):
            predictions_permutation = []
            true_labels_permutation = []
            for train_index, test_index in kf.split(X):
                y_perm = np.random.permutation(y)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y_perm[train_index], y_perm[test_index]
                X_train = X_train.reshape(-1, 1)
                X_test = X_test.reshape(-1, 1)
                _ = model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                predictions_permutation.append(y_pred[0])
                true_labels_permutation.append(y_test[0])
            loo_accuracy_permutation = accuracy_score(
                true_labels_permutation, predictions_permutation
            )
            accuracy_permutation.append(loo_accuracy_permutation)
        lower_bound = np.percentile(accuracy_permutation, 2.5)
        upper_bound = np.percentile(accuracy_permutation, 97.5)
        print(
            f"95% CI for {pair[0].name} and {pair[1].name}: {lower_bound} - {upper_bound}"
        )
        if lower_bound <= loo_accuracy <= upper_bound:
            exclude_session = "Yes"
        else:
            exclude_session = "No"
        print(f"{pair[0].name} and {pair[1].name} reject session? {exclude_session}")
        dict_results = {
            "Session": chosen_results[:-4],
            "Condition": f"{pair[0].name} vs {pair[1].name}",
            "Ground truth accuracy": loo_accuracy,
            "Shuffled 95% CI lower bound": lower_bound,
            "Shuffled 95% CI upper bound": upper_bound,
            "Exclude session?": exclude_session,
        }
        df_results = pd.DataFrame(dict_results, index=[0])
        all_results_df = pd.concat([all_results_df, df_results], ignore_index=True)
        dump_path = os.path.join(projectPath, "all_results_df_step3.pkl")
        all_results_df.to_pickle(dump_path)
        print("Added results for session")


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
    ax1.boxplot([AW, NREM, REM], labels=labels)
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


with open(os.path.join(magnitudesPath, "r20ProbeSleep.pkl"), "rb") as fp:
    df_curbd = pickle.load(fp)

fontsize = 16
plt.rcParams["svg.fonttype"] = "none"  # Keep text as text, not paths
save_path = os.path.join(projectPath, "Figures", "exemplar_session.svg")
AW = df_curbd[(df_curbd["scores"] == 1) | (df_curbd["scores"] == 2)]["no_eye"].dropna()
NREM = df_curbd[df_curbd["scores"] == 3]["no_eye"].dropna()
REM = df_curbd[df_curbd["scores"] == 4]["no_eye"].dropna()

bottom_5_NREM = NREM[NREM <= np.percentile(NREM, 5)]
bottom_5_NREM.name = "bottom_5_NREM"
top_5_REM = REM[REM >= np.percentile(REM, 95)]
top_5_REM.name = "top_5_REM"
top_5_AW = AW[AW >= np.percentile(AW, 95)]
top_5_AW.name = "top_5_AW"


fig = get_distribution_fig(
    top_5_AW, bottom_5_NREM, top_5_REM, labels=["AW", "NREM", "REM"]
)
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
    bottom_5_NREM = NREM[NREM <= np.percentile(NREM, 5)]
    top_5_REM = REM[REM >= np.percentile(REM, 95)]
    top_5_AW = AW[AW >= np.percentile(AW, 95)]
    toplot = [top_5_AW.median(), bottom_5_NREM.median(), top_5_REM.median()]
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
fig.suptitle(
    "Top 5% REM and awake, bottom 25% NREM - all sessions", fontsize=fontsize + 3
)
save_path_all = os.path.join(projectPath, "Figures", "all_sessions.svg")
plt.savefig(
    os.path.join(save_path_all),
    transparent=True,  # Removes white background
    format="svg",  # Ensures SVG format
    bbox_inches="tight",
)
