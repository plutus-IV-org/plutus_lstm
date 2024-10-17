import os
import re
from typing import Tuple, Dict, List, Any, Union
import pandas as pd
import datetime as dt
from app_service.input_data_maker import generate_data
from app_service.app_utilities import auxiliary_dataframes, group_by_history_score
from db_service.SQLite import directional_accuracy_history_load
from Const import DA_TABLE


def prepare_data(run_type: str = 'prod', assets: str = 'crypto') -> Tuple[
    Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], List[str], str]:
    """
    Prepares data for running reports.

    :param run_type: str
        Run type. Default 'prod' will obtain full data for specified assets.
        'dev' will utilize locally saved data.
    :param assets: str
        Type of cluster. Default 'crypto'.
    :return: tuple
        All relevant data related to the underlying cluster, including predictions, prices, names, and intervals.
    """
    if run_type == 'dev':
        import json

        # Load locally saved data for development
        with open(r'C:\Users\ilsbo\PycharmProjects\plutus_lstm\vaults\data_vault\dev_output_data.json',
                  'r') as json_file:
            data = json.load(json_file)

        # Convert JSON data to pandas DataFrames
        predictions = {k: pd.read_json(v, orient='split') for k, v in data['asset_predictions'].items()}
        if isinstance(data['asset_prices'], str):
            prices = pd.read_json(data['asset_prices'], orient='split')
        else:
            prices = {k: pd.read_json(v, orient='split') for k, v in data['asset_prices'].items()}

        names = data['asset_names']
        intervals = data['interval']
    else:
        # Generate data for production run
        predictions, prices, names, intervals = generate_data(assets)

    return predictions, prices, names, intervals


def prepare_inverse_name_mappings(full_names: List[str], short_names: List[str]) -> Dict[str, List[str]]:
    """
    Prepares an inverse mapping where short asset names are keys and lists of full asset names are values.

    :param full_names: List of full asset names.
    :param short_names: List of short asset names.
    :return: Dictionary mapping short names to lists of full names.
    """
    output_dict = {s_name: [] for s_name in short_names}
    for f_name in full_names:
        for s_name in short_names:
            # Match short names to full names using regex
            if re.search(s_name, f_name.split('_')[0] + "_" + f_name.split('_')[5]):
                output_dict[s_name].append(f_name)
    return output_dict


def prepare_bulk_data(predictions: Dict[str, pd.DataFrame], prices: Dict[str, pd.DataFrame]) -> Dict[
    str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Prepares bulk data by combining predictions and prices.

    :param predictions: Dictionary of predictions dataframes.
    :param prices: Dictionary of prices dataframes.
    :return: Dictionary combining predictions and prices data.
    """
    output_dict = {}
    for name in prices.keys():
        for key, val in predictions.items():
            # Match prediction and price data using regex
            if re.search(name, key.split('_')[0] + "_" + key.split('_')[5]):
                output_dict[key] = (predictions[key], prices[name])
    return output_dict


def prepare_stat_data(
        predictions: Dict[str, pd.DataFrame], prices: Dict[str, pd.DataFrame], names: List[str],
        anomalies_toggle: int = 0, anomalies_window: int = 2, rolling_period: int = 20,
        zscore_lvl: int = 2, run_type: str = 'prod') -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Prepares statistical data for the given asset names.

    :param predictions: Dictionary of predictions dataframes.
    :param prices: Dictionary of prices dataframes.
    :param names: List of asset names.
    :param anomalies_toggle: Toggle for anomalies detection.
    :param anomalies_window: Window for anomalies detection.
    :param rolling_period: Rolling period for statistics calculation.
    :param zscore_lvl: Z-score level for anomalies detection.
    :param run_type: Run type, either 'prod' or 'dev'.
    :return: Dictionary of statistical data.
    """
    is_dev = run_type == 'dev'
    output_dict = {}
    for name in names:
        # Find relevant models for each asset name
        relevant_models = [key for key in predictions if re.search(name, key.split('_')[0] + "_" + key.split('_')[5])]

        for model in relevant_models:
            # Generate auxiliary dataframes for deviation and directional accuracy
            deviation_data, directional_accuracy_data = auxiliary_dataframes(
                model, prices, predictions, name, anomalies_toggle, anomalies_window, rolling_period, zscore_lvl,
                is_dev=is_dev)
            output_dict[model] = (deviation_data, directional_accuracy_data)
    return output_dict


def load_stat_history(models: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Loads historical statistics for the given models.

    :param models: List of model names.
    :return: Dictionary of grouped historical statistics data.
    """
    output_dict = {}
    for model in models:
        # Load historical data from database
        df = directional_accuracy_history_load(DA_TABLE)
        # Filter data for the specific model
        selected_df = df[df['model_name'] == model]
        # Group data by history score
        grouped_df = group_by_history_score(selected_df)
        output_dict[model] = grouped_df
    return output_dict


def merge_dictionaries(dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Merges a list of dictionaries into a single dictionary, combining values of the same keys into lists.

    :param dicts: List of dictionaries to merge.
    :return: Merged dictionary with lists of values for the same keys.
    """
    merged_dict = {}
    for d in dicts:
        for key, value in d.items():
            if key not in merged_dict:
                merged_dict[key] = []
            merged_dict[key].append(value)
    return merged_dict


def integrate_unified_dict_into_mappings(unified_dict: Dict[str, Any], name_mappings: Dict[str, List[str]]) -> Dict[
    str, Dict[str, Any]]:
    """
    Integrates the unified dictionary into the name mappings.

    :param unified_dict: The unified dictionary with lists of values for the same keys.
    :param name_mappings: The dictionary mapping short names to lists of full names.
    :return: A final dictionary where each key is a short name and items correspond to long name items from the unified dict.
    """
    final_dict = {}
    for short_name, full_names in name_mappings.items():
        final_dict[short_name] = {full_name: unified_dict.get(full_name, []) for full_name in full_names}
    return final_dict


def process_dataframe(df: pd.DataFrame, trim_length: int = 20) -> pd.DataFrame:
    """
    Processes a DataFrame by transposing if necessary, ensuring a date-like column is the index,
    and trimming to the last `trim_length` rows. Also applies styling to highlight the maximum value in each column.
    Args:
        df (pd.DataFrame): The DataFrame to process.
        trim_length (int): The number of rows to keep from the end of the DataFrame. Default is 20.

    Returns:
        pd.DataFrame: The processed DataFrame with styling applied.
    """
    # Transpose the DataFrame if the number of columns is greater than the number of rows
    if df.shape[1] > df.shape[0] and df.shape[1] > trim_length:
        df = df.transpose()

    # Check for a date-like column and set it as the index if not already set
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                df.set_index(col, inplace=True)
                break

    # Trim the DataFrame to the last `trim_length` rows
    df = df.tail(trim_length)

    return df


def process_dataframes(data_dict: Dict[str, List[Union[Tuple[Any, ...], pd.DataFrame]]], trim_length: int = 20) -> Dict[
    str, List[Union[Tuple[Any, ...], pd.DataFrame]]]:
    """
    Processes all DataFrames within a nested dictionary structure by transposing, setting a date-like column as the index,
    trimming, and applying styling. Handles DataFrames nested within lists and tuples.
    Args:
        data_dict (Dict[str, List[Union[Tuple[Any, ...], pd.DataFrame]]]): The dictionary containing nested structures with DataFrames.
        trim_length (int): The number of rows to keep from the end of each DataFrame. Default is 20.

    Returns:
        Dict[str, List[Union[Tuple[Any, ...], pd.DataFrame]]]: The dictionary with processed DataFrames.
    """

    def recursive_process(item: Any) -> Any:
        """
        Recursively processes items within the nested structure, applying the DataFrame processing where needed.
        Args:
            item (Any): The item to process, which could be a DataFrame, list, tuple, or other.

        Returns:
            Any: The processed item.
        """
        if isinstance(item, pd.DataFrame):
            return process_dataframe(item, trim_length)
        elif isinstance(item, tuple):
            return tuple(recursive_process(sub_item) for sub_item in item)
        elif isinstance(item, list):
            return [recursive_process(sub_item) for sub_item in item]
        else:
            return item

    # Iterate over each key in the dictionary and process the list of elements
    for key in data_dict:
        data_dict[key] = [recursive_process(item) for item in data_dict[key]]

    return data_dict


def save_final_dict_to_excel(dict_to_save: Dict[str, Dict[str, Any]], file_path: str, block_height: int = 30,
                             spacing: int = 2):
    """
    Saves the final dictionary to an Excel file with each key as a separate sheet.
    Within each sheet, elements are stored with some spacing and kept within a specified row range.

    :param dict_to_save: The final dictionary to save.
    :param file_path: Path to the Excel file to save.
    :param block_height: Number of rows each model block should occupy.
    :param spacing: Number of empty rows between each element within a block.
    """
    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        for sheet_name, sub_dict in dict_to_save.items():
            if sheet_name not in writer.sheets:
                worksheet = writer.book.add_worksheet(sheet_name)
                writer.sheets[sheet_name] = worksheet
            else:
                worksheet = writer.sheets[sheet_name]

            start_row = 0

            for full_name, elements in sub_dict.items():
                # Ensure each model block starts within the specified row range
                if start_row % block_height != 0:
                    start_row = ((start_row // block_height) + 1) * block_height

                # Write the full name to the current row
                worksheet.write(start_row, 0, full_name)
                start_row += 1

                max_length = 0  # To keep track of the maximum number of rows in the current section
                current_col = 0  # Starting column

                for element in elements:
                    if isinstance(element, tuple):
                        for df in element:
                            if isinstance(df, pd.DataFrame):
                                df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, startcol=current_col,
                                            index=True, header=True)
                                current_length = len(df) + 1  # Include header
                                max_length = max(max_length, current_length)
                                current_col += df.shape[1] + 2  # Move to the next column with spacing
                    elif isinstance(element, pd.DataFrame):
                        element.to_excel(writer, sheet_name=sheet_name, startrow=start_row, startcol=current_col,
                                         index=True, header=True)
                        current_length = len(element) + 1  # Include header
                        max_length = max(max_length, current_length)
                        current_col += element.shape[1] + 2  # Move to the next column with spacing

                start_row += max_length + spacing  # Move down after writing all tables side by side


def get_bulk_data_abs_path(run_type: str = 'prod'):
    data = dt.datetime.now().date()
    return os.path.join(os.getcwd(), 'reporting_service', 'bulk_data',
                        f'Bulk_data_{run_type}_{str(data)}.xlsx')


def get_output_data_abs_path(run_type: str = 'prod'):
    data = dt.datetime.now().date()
    return os.path.join(os.getcwd(), 'reporting_service', 'prepared_data',
                        f'Daily_report_{run_type}_{str(data)}.xlsx')


def generate_bulk_data(run_type: str = 'prod', asset_type: str = 'crypto') -> None:
    # Prepare data for the specified run type
    asset_predictions, asset_prices, asset_names, interval = prepare_data(run_type, asset_type)

    # Prepare name mappings between full and short asset names
    name_mappings = prepare_inverse_name_mappings(asset_predictions.keys(), asset_names)

    # Prepare bulk data combining predictions and prices
    bulk_data = prepare_bulk_data(asset_predictions, asset_prices)

    # Prepare statistical data
    stat_data = prepare_stat_data(asset_predictions, asset_prices, asset_names, run_type=run_type)

    # Load historical statistical data
    historical_data = load_stat_history(list(asset_predictions.keys()))

    # Combine all data into one output dictionary with unified keys
    output_dicts = [bulk_data, stat_data, historical_data]
    unified_dict = merge_dictionaries(output_dicts)

    # Trim and arrange the dataframes
    unified_dict = process_dataframes(unified_dict)
    # Integrate the unified dictionary into the name mappings
    final_dict = integrate_unified_dict_into_mappings(unified_dict, name_mappings)

    save_final_dict_to_excel(final_dict, get_bulk_data_abs_path(run_type))
