# Function to run simulations for a chunk of combinations
import logging
import multiprocessing
import os
import time
from functools import partial

import numpy as np

import curbd
from utils import (
    curbd_get_avg,
    find_combinations,
    generate_random_neuron_combinations,
    prep_sim,
)


def run_simulation_chunk(
    chunk, total_neurons, area_responses, num_neurons_per_area, worker_id
):
    logger = worker_setup(worker_id)
    logger.info(
        f"Worker {worker_id} started processing chunk. First few combinations:\
              {chunk[:3]}"
    )

    logger.debug(
        f"Chunk content for worker {worker_id}: {chunk}"
    )  # This might be a large output
    logger.info("Started processing chunk.")
    all_inter_area_values = []
    pvar_check = []

    for i in range(0, len(chunk)):
        areacombo = chunk[i]
        total_combinations = len(chunk)
        progress = (i / total_combinations) * 100
        logger.info(
            f"Worker {worker_id} processing combination \
                .{i+1}/{total_combinations} ({progress:.2f}% complete)"
        )
        if worker_id == 0:
            if i % (total_combinations / 100) == 0:
                progress = (i / total_combinations) * 100
                print(
                    f"Workers are {progress:.2f}% complete. \
                        Simulation of {total_neurons}"
                )

        if any(len(sublist) == 0 for sublist in areacombo):
            logger.error(
                f"Invalid combination {areacombo} \
                      received by worker {worker_id}."
            )
            continue  # Skip this combination
        logger.info(f"Starting combination {areacombo}.")
        start_time = time.time()
        activity_matrix, index_arrays, regions = prep_sim(
            area_responses, ["A1", "BLA", "PFC", "HPC"], areacombo
        )
        regions = np.array(regions, dtype=object)
        logger.info("Starting CURBD.")
        model = curbd.trainMultiRegionRNN(
            activity_matrix,
            dtData=0.001,
            dtFactor=1,
            regions=regions,
            tauRNN=2 * 0.1 / 2,
            nRunTrain=500,
            verbose=False,
            nRunFree=0,
            plotStatus=False,
        )
        [curbd_arr, curbd_labels] = curbd.computeCURBD(model)
        elapsed_time = time.time() - start_time  # End timing
        logger.info(
            f"CURBD training and computation finished. Elapsed time: \
                {elapsed_time:.2f} seconds."
        )  # Log elapsed time

        n_regions = curbd_arr.shape[0]

        curbd_avg = curbd_get_avg(curbd_arr)
        curbd_inter = []

        # Loop through simulations

        for iTarget in range(n_regions):
            for iSource in range(n_regions):
                # Ignore identity pairs
                if iTarget == iSource:
                    continue
                curbd_values = curbd_avg[iTarget, iSource]
                curbd_inter.append(curbd_values)

        inter_area_connectivity = np.mean(curbd_inter)
        all_inter_area_values.append(inter_area_connectivity)
        pvar_check.append(model["pVars"][-1])

    logger.info("Completed chunk.")
    return all_inter_area_values, pvar_check


def worker_setup(worker_id):
    # Define the directory for logs
    log_dir = "logs"
    os.makedirs(
        log_dir, exist_ok=True
    )  # Create the directory if it doesn't exist

    # Define the log file name, one for each worker
    log_file_name = f"worker_{worker_id}.log"
    log_file_path = os.path.join(log_dir, log_file_name)

    # Set up logging to file
    logger = logging.getLogger(f"worker_{worker_id}")
    logger.setLevel(logging.INFO)  # Or use another logging level like DEBUG
    logger.propagate = False

    # Create a file handler to write to the log file
    file_handler = logging.FileHandler(log_file_path)

    # Create a formatter and set it for the handler
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(file_handler)

    return logger


def worker(
    chunk, total_neurons, area_responses, num_neurons_per_area, worker_id
):
    return run_simulation_chunk(
        chunk, total_neurons, area_responses, num_neurons_per_area, worker_id
    )


def run_parallel_simulations(
    total_neurons, area_responses, num_neurons_per_area
):
    structure_combinations = list(
        find_combinations(num_neurons_per_area, total_neurons)
    )
    logging.info(f"Structure combinations: {structure_combinations}")
    all_specific_combinations = []
    max_combinations = int(500 / (len(structure_combinations)))
    print(
        f"there are {len(structure_combinations)} structural combinations, which will bring {max_combinations} neuron combinations"
    )
    for structure in structure_combinations:
        specific_combinations = generate_random_neuron_combinations(
            num_neurons_per_area, structure, max_combinations=max_combinations
        )
        all_specific_combinations.extend(specific_combinations)
    num_workers = multiprocessing.cpu_count()
    chunks = [
        all_specific_combinations[i::num_workers] for i in range(num_workers)
    ]

    pool = multiprocessing.Pool(num_workers)
    worker_args = [
        (chunk, total_neurons, area_responses, num_neurons_per_area, worker_id)
        for worker_id, chunk in enumerate(chunks)
    ]
    for worker_id, chunk in enumerate(chunks):
        logging.info(
            f"Worker {worker_id} received chunk with {len(chunk)}"
            "combinations."
        )
    print(f"Simulation of {total_neurons}")
    results = pool.starmap(worker, worker_args)
    pool.close()
    pool.join()

    # Aggregate results from all workers
    inter_area_values_aggregated = []
    pvar_values_aggregated = []
    for inter_area_values, pvar_values in results:
        inter_area_values_aggregated.extend(inter_area_values)
        pvar_values_aggregated.extend(pvar_values)

    return inter_area_values_aggregated, pvar_values_aggregated
