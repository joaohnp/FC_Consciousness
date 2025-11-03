from pathlib import Path
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from curbd_analysis.utils.paths import resolve_path
from curbd_analysis.plotting.io import save_figure

logger = logging.getLogger(__name__)

DATA_DIR = resolve_path("src", "curbd_analysis", "data", "simdata", "spike_data")

conn_matrix_path = DATA_DIR / "connectivity_matrix.txt"
labels = ["E23", "PV23", "E4", "PV4", "E5", "PV5", "E6", "PV6"]

data_array = np.loadtxt(conn_matrix_path, delimiter=",")
df = pd.DataFrame(np.reshape(data_array, [8, 8]))
df.index = labels
df.columns = labels
pv = ["PV23", "PV4", "PV6", "PV5"]
matrix = df.drop(index=pv, columns=pv)
logger.info(
    "Loaded connectivity matrix from %s with shape %s", conn_matrix_path, matrix.shape
)

connections_path = DATA_DIR / "connections_FINAL.txt"
data_array2 = np.loadtxt(connections_path, delimiter=",")
df2 = pd.DataFrame(np.reshape(data_array2, [8, 8]))
df2.index = labels
df2.columns = labels
matrix_connections = df2.drop(index=pv, columns=pv)
logger.info(
    "Loaded connections from %s with shape %s",
    connections_path,
    matrix_connections.shape,
)

fig, ax = plt.subplots()
sns.heatmap(matrix, annot=False, vmin=0, vmax=0.008, ax=ax)
ax.set_xlabel("Source")
ax.set_ylabel("Target")

save_figure(fig, "analysis/fig2a_heatmap")
