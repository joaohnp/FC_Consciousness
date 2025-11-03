from pathlib import Path


def find_indices_by_string(input_list, search_string):
    return [i + 1 for i, s in enumerate(input_list) if s == search_string]


def channels_to_prepTT(channels):
    if len(channels) > 0:
        return f"prepTT{channels[0] // 4}"
    return None


def convert_folders(folder_name, spks_folder, ChMap):
    result = find_indices_by_string(ChMap, folder_name)
    prepTT_value = channels_to_prepTT(result)
    path_hd = Path(spks_folder) / prepTT_value
    return str(path_hd)
