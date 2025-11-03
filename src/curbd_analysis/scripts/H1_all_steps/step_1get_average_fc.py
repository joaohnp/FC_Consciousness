# %%
import os
import pickle

import numpy as np
import pandas as pd
from scikit_posthocs import posthoc_dunn
from scipy import stats

projectPath = os.path.dirname(os.path.abspath(__name__))
parentPath = os.path.dirname(projectPath)
scoresPath = os.path.join(parentPath, "SleepScores")
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

    for name, data in [("AW", AW), ("NREM", NREM), ("REM", REM)]:
        stat, p_value = stats.shapiro(data)
        normality = (
            "Normally distributed" if p_value > alpha else "Not normally distributed"
        )
        print(f"{name}: stat={stat:.4f}, p-value={p_value:.4f} - {normality}")

    stat, p_value = stats.kruskal(AW, NREM, REM)
    alpha = 0.05
    if p_value < alpha:
        all_data = pd.concat([AW, NREM, REM])
        groups = pd.Series(["AW"] * len(AW) + ["NREM"] * len(NREM) + ["REM"] * len(REM))
        df_combined = pd.DataFrame({"value": all_data.values, "group": groups.values})
        dunn_result = posthoc_dunn(
            df_combined, val_col="value", group_col="group", p_adjust="fdr_bh"
        )
        nrem_vs_aw = dunn_result.loc["NREM", "AW"]
        nrem_vs_rem = dunn_result.loc["NREM", "REM"]
        print(f"NREM vs AW p-value: {dunn_result.loc['NREM', 'AW']:.4f}")
        print(f"NREM vs REM p-value: {dunn_result.loc['NREM', 'REM']:.4f}")
    elif p_value > alpha:
        nrem_vs_aw = nrem_vs_rem = np.nan

    dict_results = {
        "Session": chosen_results,
        "AW": len(AW),
        "NREM": len(NREM),
        "REM": len(REM),
        "Shapiro": normality,
        "Kruskal": p_value,
        "NREM vs AW": nrem_vs_aw,
        "NREM vs REM": nrem_vs_rem,
    }

    df_results = pd.DataFrame(dict_results, index=[0])
    all_results_df = pd.concat([all_results_df, df_results], ignore_index=True)
    dump_path = os.path.join(projectPath, "all_results_df.pkl")
    all_results_df.to_pickle(dump_path)
    print(f"Added results for session {chosen_results}")
