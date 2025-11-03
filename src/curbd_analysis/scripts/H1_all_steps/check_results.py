import pickle
import os
import pandas as pd
import numpy as np
projectPath = os.path.dirname(os.path.abspath(__name__))

unpickled_df = pd.read_pickle(os.path.join(projectPath, "all_results_df.pkl"))

unpickled_df.round(2).to_csv("all_results_df.csv")

unpickled_df_step2 = pd.read_pickle(os.path.join(projectPath, "all_results_df_step2.pkl"))

unpickled_df_step2.to_csv("all_results_df_step2.csv")

unpickled_df_step4 = pd.read_pickle(os.path.join(projectPath, "all_results_df_step4.pkl"))

unpickled_df_step4.round(2).to_csv("all_results_df_step4.csv")

unpickled_df = pd.read_pickle(os.path.join(projectPath, "combined_p.pkl"))

unpickled_df.round(2).to_csv("combined_p_results.csv")