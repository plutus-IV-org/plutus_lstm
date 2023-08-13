import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Dict, Union
from utilities.metrics import _directional_accuracy
from utilities.weights_generator import generate_weight_matrix


def sum_values(dictionary: Dict, weights: np.array, real_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the weighted sum of all values (dataframes) in the dictionary.

    """
    total = sum(values * weight for key, weight, values in zip(dictionary.keys(), weights, dictionary.values()))

    result_df, trades_coverage_df = _directional_accuracy(real_df, total,
                                                          best_model={'future_days': len(total.columns)},
                                                          reshape_required=False)
    directional_accuracy_score = result_df['6 months'].mean()
    return directional_accuracy_score, total


def compute_averages(initial_keys: List[str], data: Dict) -> Tuple[List[str], Dict]:
    """
    Calculate and append the average values for given names and intervals.

    Parameters:
    - initial_keys (List[str]): List of key strings in the format: 'name_otherparts_interval'.
    - data (Dict): Dictionary containing dataframes associated with initial_keys.

    Returns:
    - Tuple[List[str], Dict]: Updated initial_keys and data with averaged values appended.
    """

    # Extract unique names and intervals from the provided keys
    unique_names = {key.split('_')[0] for key in initial_keys}
    unique_intervals = {key.split('_')[5] for key in initial_keys}

    for name in unique_names:
        for interval in unique_intervals:

            # Filter keys that match the current name and interval
            filtered_keys = [k for k in initial_keys if
                             k.startswith(name) and (k.split('_') + ['dummy_ending'])[5] == interval]

            # Group dataframes by their number of columns
            dataframes_grouped_by_columns = {}
            for idx, key in enumerate(filtered_keys, 1):
                number_of_columns = data[key][1].shape[1]
                dataframes_grouped_by_columns[(idx, number_of_columns)] = data[key][1]

            unique_column_counts = {column_count for _, column_count in dataframes_grouped_by_columns.keys()}

            for column_count in unique_column_counts:
                relevant_dataframes = {idx: dataframe for (idx, col_count), dataframe in
                                       dataframes_grouped_by_columns.items() if col_count == column_count}

                # Compute the average for dataframes with current column count
                real_price = data[list(data.keys())[0]][0]
                real_price_df = pd.DataFrame()
                for x in range(column_count):
                    shifted_price = real_price.shift(-x)
                    shifted_price.columns = [x]
                    real_price_df = pd.concat([real_price_df, shifted_price], axis=1)

                run_times = 100  # You can adjust this value
                num_columns = len(real_price_df.columns)
                storage = {}
                selected_weights_combinations = generate_weight_matrix(num_columns, rows=run_times)
                print('Computing average prediction line based on different wights combinations.')
                for weights in tqdm(selected_weights_combinations, desc="Running Loop", unit="combination"):
                    weighted_score, weighted_average_df = sum_values(relevant_dataframes, weights=weights,
                                                                     real_df=real_price_df)
                    storage[weighted_score] = weighted_average_df,weights
                max_key = max(storage.keys())
                averaged_df = storage[max_key][0]
                print(f"The best weights combinations is {storage[max_key][1]}")
                print(f"The best weights score is {max_key}")
                # Add the averaged dataframe to the data dictionary
                new_key = f"{name}_average_{column_count}_future_steps_{interval}"
                sample_key_for_interval = next(filter(lambda x: interval in x, initial_keys))
                data[new_key] = data[sample_key_for_interval][0], averaged_df
                initial_keys.append(new_key)

    return initial_keys, data
