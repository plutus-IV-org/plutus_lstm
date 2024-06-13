import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Tuple, List, Dict, Union, Any
from utilities.metrics import _directional_accuracy
from utilities.weights_generator import generate_weight_matrix
from Const import ShortsBatches, LongsBatches, ModelBatches
import re


def sum_values(dictionary: Dict[int, pd.DataFrame], weights: np.array, real_df: pd.DataFrame) -> tuple[
    Any, Union[int, Any]]:
    """
    Compute the weighted sum of all values (dataframes) in the dictionary.
    """
    # Make sure weights array and dictionary have the same length
    if len(weights) != len(dictionary):
        raise ValueError("The number of weights must match the number of DataFrames in the dictionary")
    else:
        if len(weights) == 1:
            weights = [1]
        # Compute the weighted sum
        total = sum(df * weight for df, weight in zip(dictionary.values(), weights))

        # Other computations
        result_df, trades_coverage_df = _directional_accuracy(real_df, total,
                                                              best_model={'future_days': len(total.columns)},
                                                              reshape_required=False)
        directional_accuracy_score = result_df['6 months'].mean()
        return directional_accuracy_score, total


def count_mean_prediction_line(dictionary: Dict[int, pd.DataFrame], weights: np.array) -> pd.DataFrame:
    total = sum(df * weight for df, weight in zip(dictionary.values(), weights))
    return total


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

                # Simple mean
                weights_array = np.ones(len(relevant_dataframes)) * 1 / len(relevant_dataframes)
                weighted_average_df = count_mean_prediction_line(relevant_dataframes, weights=weights_array)
                # Add the averaged dataframe to the data dictionary
                new_key = f"{name}_simple-average_{column_count}_future_steps_{interval}"
                data[new_key] = data[filtered_keys[0]][0], weighted_average_df
                initial_keys.append(new_key)

                # # Long mean
                # # Loop through each batch in LongsBatches
                # for x in range(len(list(LongsBatches))):
                #     # Check if the current batch matches the specified interval and number of columns
                #     if list(LongsBatches)[x].name.split('_')[1] == interval and \
                #             list(LongsBatches)[x].name.split('_')[2] == str(column_count):
                #         # Process the batch if it has elements
                #         if len(list(LongsBatches)[x].value) > 0:
                #             # Initialize dictionaries for storing predictions and weights
                #             long_storage = {}
                #             weights_storage = []
                #             position_key = 1
                #             # Iterate over short model names in the batch
                #             for short_model_name in list(LongsBatches)[x].value.keys():
                #                 # Loop through each key in the data
                #                 for key in data.keys():
                #                     # Construct a regex pattern to match the short_model_name
                #                     pattern = f".*{re.escape(short_model_name)}.*"
                #                     # If the pattern matches the key, process the data
                #                     if re.match(pattern, key):
                #                         prediction_df = data[key][1]
                #                         # Retrieve the weight for the current model
                #                         weight = list(LongsBatches)[x].value[short_model_name]
                #                         # Store the prediction dataframe in long_storage
                #                         long_storage[position_key] = prediction_df
                #                         # Append the weight to the weights storage
                #                         weights_storage.append(weight)
                #                         # Increment the position key
                #                         position_key += 1
                #                         break
                #
                #             # Convert weights list to a NumPy array for processing
                #             weights_storage_array = np.array(weights_storage)
                #             # Calculate the weighted mean prediction
                #             long_mean = count_mean_prediction_line(long_storage, weights=weights_storage_array)
                #
                #             # Construct a new key for storing the long mean in data
                #             new_key = f"{name}_long-average_{column_count}_future_steps_{interval}"
                #             # Find a sample key for the interval
                #             sample_key_for_interval = next(filter(lambda x: interval in x, initial_keys))
                #             # Store the long mean in data with the new key
                #             data[new_key] = data[sample_key_for_interval][0], long_mean
                #             # Append the new key to initial_keys for future reference
                #             initial_keys.append(new_key)
                #
                # # Short mean
                # # Loop through each batch in ShortssBatches
                # for x in range(len(list(ShortsBatches))):
                #     # Check if the current batch matches the specified interval and number of columns
                #     if list(ShortsBatches)[x].name.split('_')[1] == interval and \
                #             list(ShortsBatches)[x].name.split('_')[2] == str(column_count):
                #         # Process the batch if it has elements
                #         if len(list(ShortsBatches)[x].value) > 0:
                #             # Initialize dictionaries for storing predictions and weights
                #             short_storage = {}
                #             weights_storage = []
                #             position_key = 1
                #             # Iterate over short model names in the batch
                #             for short_model_name in list(ShortsBatches)[x].value.keys():
                #                 # Loop through each key in the data
                #                 for key in data.keys():
                #                     # Construct a regex pattern to match the short_model_name
                #                     pattern = f".*{re.escape(short_model_name)}.*"
                #                     # If the pattern matches the key, process the data
                #                     if re.match(pattern, key):
                #                         prediction_df = data[key][1]
                #                         # Retrieve the weight for the current model
                #                         weight = list(ShortsBatches)[x].value[short_model_name]
                #                         # Store the prediction dataframe in short_storage
                #                         short_storage[position_key] = prediction_df
                #                         # Append the weight to the weights storage
                #                         weights_storage.append(weight)
                #                         # Increment the position key
                #                         position_key += 1
                #                         break
                #
                #             # Convert weights list to a NumPy array for processing
                #             weights_storage_array = np.array(weights_storage)
                #             # Calculate the weighted mean prediction
                #             short_mean = count_mean_prediction_line(short_storage, weights=weights_storage_array)
                #
                #             # Construct a new key for storing the short mean in data
                #             new_key = f"{name}_short-average_{column_count}_future_steps_{interval}"
                #             # Find a sample key for the interval
                #             sample_key_for_interval = next(filter(lambda x: interval in x, initial_keys))
                #             # Store the short mean in data with the new key
                #             data[new_key] = data[sample_key_for_interval][0], short_mean
                #             # Append the new key to initial_keys for future reference
                #             initial_keys.append(new_key)
                #
                # # Top mean
                # # Loop through each batch in TopBatches
                # for x in range(len(list(ModelBatches))):
                #     # Check if the current batch matches the specified interval and number of columns
                #     if list(ModelBatches)[x].name.split('_')[1] == interval and \
                #             list(ModelBatches)[x].name.split('_')[2] == str(column_count):
                #         # Process the batch if it has elements
                #         if len(list(ModelBatches)[x].value) > 0:
                #             # Initialize dictionaries for storing predictions and weights
                #             top_storage = {}
                #             weights_storage = []
                #             position_key = 1
                #             # Iterate over top model names in the batch
                #             for top_model_name in list(ModelBatches)[x].value.keys():
                #                 # Loop through each key in the data
                #                 for key in data.keys():
                #                     # Construct a regex pattern to match the top_model_name
                #                     pattern = f".*{re.escape(top_model_name)}.*"
                #                     # If the pattern matches the key, process the data
                #                     if re.match(pattern, key):
                #                         prediction_df = data[key][1]
                #                         # Retrieve the weight for the current model
                #                         weight = list(ModelBatches)[x].value[top_model_name]
                #                         # Store the prediction dataframe in top_storage
                #                         top_storage[position_key] = prediction_df
                #                         # Append the weight to the weights storage
                #                         weights_storage.append(weight)
                #                         # Increment the position key
                #                         position_key += 1
                #                         break
                #
                #             # Convert weights list to a NumPy array for processing
                #             weights_storage_array = np.array(weights_storage)
                #             # Calculate the weighted mean prediction
                #             top_mean = count_mean_prediction_line(top_storage, weights=weights_storage_array)
                #
                #             # Construct a new key for storing the top mean in data
                #             new_key = f"{name}_top-average_{column_count}_future_steps_{interval}"
                #             # Find a sample key for the interval
                #             sample_key_for_interval = next(filter(lambda x: interval in x, initial_keys))
                #             # Store the top mean in data with the new key
                #             data[new_key] = data[sample_key_for_interval][0], top_mean
                #             # Append the new key to initial_keys for future reference
                #             initial_keys.append(new_key)

                # Adjusted mean
                # run_times = 100  # You can adjust this value
                # num_columns = len(real_price_df.columns)
                # storage = {}
                # selected_weights_combinations = generate_weight_matrix(len(relevant_dataframes), rows=run_times)
                #
                #
                # print('Computing average prediction line based on different wights combinations.')
                # for weights in tqdm(selected_weights_combinations, desc="Running Loop", unit="combination"):
                #     weighted_score, weighted_average_df = sum_values(relevant_dataframes, weights=weights,
                #                                                      real_df=real_price_df)
                #     storage[weighted_score] = weighted_average_df, weights
                # max_key = max(storage.keys())
                # averaged_df = storage[max_key][0]
                # print(f"The best weights combinations is {storage[max_key][1]}")
                # print(f"The best weights score is {max_key}")

    return initial_keys, data
