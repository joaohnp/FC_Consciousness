# %%
import os


def find_indices_by_string(input_list, search_string):
    indices = [i + 1 for i, s in enumerate(input_list) if s == search_string]
    return indices


def channels_to_prepTT(channels):
    # Check if the list is not empty
    if len(channels) > 0:
        # Calculate the prepTT value based on the first channel in the list
        prepTT = f"prepTT{channels[0] // 4}"
        return prepTT
    else:
        return None  # Return None for an empty list


# Example usage:


def convert_folders(folder_name, spks_folder, ChMap):
    # Extracting indices of the folder_name
    result = find_indices_by_string(ChMap, folder_name)

    prepTT_value = channels_to_prepTT(result)
    # print(f"Getting indices {result} from {prepTT_value} ")
    path_hd = os.path.join(spks_folder, prepTT_value)

    return path_hd
