import os
import pickle

import numpy as np
import pandas as pd

projectPath = os.path.dirname(os.path.abspath(__name__))
hard_drive = os.path.joint(projectPath, "data", "curbd_output")
dirs = os.listdir(hard_drive)
for dir_of_interest in dirs:
    results_folder = os.path.join(
        hard_drive, dir_of_interest
    )  # Choosing one specific folder just to test

    pickle_files = [f for f in os.listdir(results_folder) if f.endswith(".pkl")]
    pickle_files.sort(key=lambda x: int(x.split(".")[0]))

    def process_matrix(input_matrix):
        output_matrix = np.zeros((4, 4))
        arr = input_matrix["curbd_arr"]
        lbl = input_matrix["curbd_labels"]
        # Loop through each position in the 4x4 matrix
        for i in range(4):
            for j in range(4):
                quadrant = arr[i, j]
                mean_axis0 = np.mean(np.mean(quadrant, axis=0))
                output_matrix[i, j] = mean_axis0
        return output_matrix

    to_pkl = []
    for file in pickle_files[:-1]:
        file_path = os.path.join(results_folder, file)
        df = pd.read_pickle(file_path)
        if df is None:
            to_pkl.append(0)
        elif isinstance(df["curbd_arr"], np.ndarray):
            vec = df["curbd_arr"]
            if vec.size == 0:
                to_pkl.append(0)
            else:
                value = process_matrix(df)
                to_pkl.append(value)

    with open(
        os.path.join(projectPath, "LOO-rdy", dir_of_interest), "wb"
    ) as output_file:
        pickle.dump(to_pkl, output_file)
