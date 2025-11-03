import os
import pickle
import time

import numpy as np
import pandas as pd
import tqdm

projectPath = os.path.dirname(os.path.abspath(__name__))
parentPath = os.path.dirname(projectPath)
scoresPath = os.path.join(parentPath, "SleepScores")
magnitudesPath = os.path.join(projectPath, "CurbdMagnitudes")
hard_drive = os.path.join(projectPath, "data", "curbd_outputs")
results_sessions = os.listdir(hard_drive)
all_results_df = pd.DataFrame()

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
alpha = 0.05


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
    pkl_files,
    epoch_amount,
    masks,
    mask_names=["all", "no_eye", "no_hpc", "mask_intra"],
):
    results = {name: [] for name in mask_names}
    start_time = time.time()
    for file in tqdm.tqdm(pkl_files[:epoch_amount], desc="Processing files"):
        file_path = os.path.join(pickles_path, file)
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
    elapsed_time = time.time() - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    return pd.DataFrame(results)


for chosen_results in results_sessions:
    print(f"Processing results for session {chosen_results}")
    pickles_path = os.path.join(hard_drive, chosen_results)
    pickle_files = [f for f in os.listdir(pickles_path) if f.endswith(".pkl")]
    pickle_files.sort(
        key=lambda x: int(x.split(".")[0])
    )  # setting which processed session we'll look into
    file_of_interest = chosen_results + ".pickle"

    with open(os.path.join(scoresPath, file_of_interest), "rb") as fp:
        scores = pickle.load(fp)

    if len(scores) == 3:
        scores = scores[2]

    mean_values = check_files(pickle_files, -1, masks=masks)
    df_curbd = mean_values
    df_curbd["scores"] = scores[: len(mean_values)]
    dump_path = os.path.join(magnitudesPath, chosen_results + ".pkl")
    df_curbd.to_pickle(dump_path)
