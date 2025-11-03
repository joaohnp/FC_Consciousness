def get_experimental_info():
    animal = {0: "r14", 1: "r16", 2: "r19", 3: "r20"}

    sessions = {
        0: "Habituation",
        1: "FearConditioning",
        2: "Probe",
        3: "ExtinctionTraining",
        4: "ExtinctionTest",
    }

    stages = {
        0: "NREM",
        1: "REM",
        2: "AW",
        3: "QW",
    }

    behaviour = {
        0: "Sleep",
        1: "Awake",
    }

    hd_mapping = {
        0: "00.Habituation",
        1: "01.FearConditioning",
        2: "02.ProbeTest",
        3: "03.ExtinctionTraining",
        4: "04.ExtinctionTest",
    }
    areas_mapping = {0: "A1", 1: "BLA", 2: "HPC", 3: "PFC"}

    cs_naming = ["cs_plus", "cs_minus", "cs_novel"]

    stage_names = ["AW", "QW", "NREM", "REM", "UND"]  # Stage names
    stage_mapping = {"AW": 1, "QW": 2, "NREM": 3, "REM": 4, "UND": 5}
    return (
        animal,
        sessions,
        stages,
        behaviour,
        hd_mapping,
        areas_mapping,
        cs_naming,
        stage_names,
        stage_mapping,
    )
