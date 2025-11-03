import os

import pandas as pd

projectPath = os.path.dirname(os.path.abspath(__name__))

unpickled_df_step2 = pd.read_pickle(
    os.path.join(projectPath, "all_results_df_step2_H2.pkl")
)

unpickled_df_step2.to_csv("all_results_df_step2.csv")

unpickled_df = pd.read_pickle(os.path.join(projectPath, "combined_p.pkl"))

unpickled_df.round(2).to_csv("combined_p_results_h2.csv")

unpickled_df = pd.read_pickle(os.path.join(projectPath, "all_results_df_step3.pkl"))

unpickled_df.round(2).to_csv("svm_h2.csv")
