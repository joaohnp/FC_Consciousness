# %%
import logging
from pathlib import Path
import pickle

from utils_expinfo import get_experimental_info
from utils_sims import run_parallel_simulations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

(
    animal,
    sessions,
    stages,
    behaviour,
    hd_mapping,
    areas_mapping,
    cs_naming,
    stage_names,
    stage_mapping,
) = get_experimental_info()
# projectPath = Path(__file__).absolute().parent
projectPath = Path(__file__).resolve().parent.parent.parent


def main():
    # Loading the pickled data
    pickled_path = projectPath / "data" / "r14remsleep" / "csplus_rem_r14.pickle"
    with open(pickled_path, "rb") as handle:
        area_responses = pickle.load(handle)

    areas = ["A1", "BLA", "HPC", "PFC"]
    num_neurons_per_area = [
        len(area_responses["A1"][0]),
        len(area_responses["BLA"][0]),
        len(area_responses["PFC"][0]),
        len(area_responses["HPC"][0]),
    ]

    inter_area_results = {}
    pvar_results = {}

    for total_neurons in range(4, 13):  # Adjust range as needed
        print(f"Starting parallel simulation for {total_neurons} total neurons.")
        (
            inter_area_values_aggregated,
            pvar_values_aggregated,
        ) = run_parallel_simulations(
            total_neurons, area_responses, num_neurons_per_area
        )

        # Store the results for the current number of total neurons
        inter_area_results[total_neurons] = inter_area_values_aggregated
        pvar_results[total_neurons] = pvar_values_aggregated
        # Save the aggregated results
        print(f"SAVING THE COMBINATION OF {total_neurons}")
        with open("final_inter_area_results.pkl", "wb") as f:
            pickle.dump(inter_area_results, f)
        with open("final_pvar_results.pkl", "wb") as f:
            pickle.dump(pvar_results, f)


if __name__ == "__main__":
    main()

pickled_path = projectPath / "final_inter_area_results.pkl"
with open(pickled_path, "rb") as handle:
    area_result = pickle.load(handle)

# %%
