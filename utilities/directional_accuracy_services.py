import pandas as pd
import datetime
from PATH_CONFIG import _ROOT_PATH
from db_service.SQLite import directional_accuracy_history_add
from Const import DA_TABLE


def save_directional_accuracy_score(model_name: str, results: pd.Series) -> None:
    """
    Save directional accuracy scores along with corresponding model names and date.

    :param model_name: Name of the model for which the accuracy scores are being saved.
    :param results: A pandas Series containing the directional accuracy scores.
    :return: None
    """
    # # Construct the absolute file path for the data vault's directional accuracy register
    # abs_path = _ROOT_PATH() + "\\vaults\\data_vault\\directional_accuracy_register.xlsx"
    #
    # # Read the existing register into a DataFrame, selecting specific columns
    # df = pd.read_excel(abs_path)[['model_name', 'da_score', 'da_list', 'date']]

    # Prepare the data to be added as a new row in the DataFrame
    df_to_add = pd.DataFrame(
        {'model_name': model_name,
         'da_score': results.mean(),  # Placeholder value, should be replaced with the actual DA score
         'da_list': str(results.values.tolist()),
         'date': datetime.datetime.now()},  # Placeholder value, should be replaced with the actual date
        index=[0])

    # # Concatenate the new data with the existing DataFrame
    # df = pd.concat([df, df_to_add])
    #
    # # Write the updated DataFrame back to the Excel file
    # df.to_excel(abs_path)

    # Adding table to SQL db
    directional_accuracy_history_add(DA_TABLE, df_to_add)
    return None
