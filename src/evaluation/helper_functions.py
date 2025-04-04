import pandas as pd
from typing import Dict, List, Any
import numpy as np
import itertools
from instances.instance_manager import InstanceManager


def extract_combinations(current_dict: dict) -> list[Any]:
    """This function extracts list of parameters, extracts them and return the list of
    all combinations"""
    config_lists = []
    config_keys = []

    for key, params in current_dict.items():
        if isinstance(params, dict) and not key == "mixer":
            nested_combos = extract_combinations(params)
            config_lists.append(nested_combos)
            config_keys.append((key, None))
        elif isinstance(params, list):
            config_lists.append(params)
            config_keys.append((key, None))
        elif params == {}:
            config_lists.append([None])
            config_keys.append((key, None))
        else:
            config_lists.append([params])
            config_keys.append((key, None))

    combinations = list(itertools.product(*config_lists))

    config_list = []
    for combo in combinations:
        result = {}
        for (key, _), value in zip(config_keys, combo):
            if isinstance(value, dict):
                result[key] = value
            elif value is not None:
                result[key] = value
        config_list.append(result)

    return config_list


def get_model_params(
    instance_manager: InstanceManager, file_selection: list[str] | None = None
) -> list:
    model_params = []
    if file_selection is not None:
        for file_name in file_selection:
            opt, _ = instance_manager.get_metadata(file_name)
            model_params.append(
                {
                    "cqm": file_name,
                    "dir": instance_manager.instance_directory,
                    "opt": opt,
                }
            )
    else:
        for x in instance_manager.instance_iterator():
            _, opt, id = x

            model_params.append(
                {"cqm": id, "dir": instance_manager.instance_directory, "opt": opt}
            )
    return model_params


def apply_dict_comparison(
    df: pd.DataFrame,
    column: str,
    reference_dict: Dict,
    ignore_keys: List | None = None,
) -> np.ndarray:
    """
    Apply dictionary comparison to a DataFrame column against a reference dictionary.

    Args:
        df (pd.DataFrame): Input DataFrame
        column (str): Name of column containing dictionaries
        reference_dict (dict): Dictionary to compare against
        ignore_keys (list, optional): Keys to ignore in comparison

    Returns:
        np.ndarray: Boolean numpy array indicating comparison result for each row
    """

    def compare_dicts(dict1: Dict, dict2: Dict, ignore_keys: List | None = None):
        ignore_keys = ignore_keys or []

        # Filter out ignored keys
        keys1 = {k for k in dict1.keys() if k not in ignore_keys}
        keys2 = {k for k in dict2.keys() if k not in ignore_keys}
        # Check key sets match
        if keys1 != keys2:
            return False

        # Compare each value
        for key in keys1:
            val1, val2 = dict1[key], dict2[key]

            # Recursively compare nested dictionaries
            if isinstance(val1, dict) and isinstance(val2, dict):
                if not compare_dicts(val1, val2, ignore_keys):
                    return False

            # Direct comparison for non-dict values
            elif val1 != val2:
                return False

        return True

    # Create comparison results as a list first
    comparison_results = [
        compare_dicts(x, reference_dict, ignore_keys) for x in df[column]
    ]

    # Convert to numpy array
    mask = np.array(comparison_results, dtype=bool)

    return mask
