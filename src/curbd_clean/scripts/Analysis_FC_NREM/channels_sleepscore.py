import os
import pickle

import pandas as pd

projectPath = os.path.dirname(os.path.abspath(__name__))
target_folder = os.path.join(projectPath, 'Analysis_FC_NREM', 'ch_scoring.pkl')
SCORING_MASTER = {}

SCORING_MASTER["r16"] = {
    "Habituation": {
        "A1": "65",
        "HPC": "10",
        "PFC": "41"
    },
    "FearConditioning": {
        "A1": "value4",
        "HPC": "value5",
        "PFC": "value6"
    }
}

df = pd.DataFrame.from_dict(SCORING_MASTER)

with open(target_folder, 'wb') as fp:
    pickle.dump(df, fp)
