import sqlite3
import os
import pandas as pd
import json
from db_service.schemas.tables_schemas import DIRECTIONAL_ACCURACY_TABLE_SCHEMA
from Const import DA_TABLE

# Getting tables directory
current_file_path = os.path.abspath(__file__)
directory_path = os.path.dirname(current_file_path)
tables_directory = os.path.join(directory_path, 'tables')  # Use os.path.join for path concatenation


# Define the function to load DataFrame into SQLite
def directional_accuracy_history_load(table_name: str) -> pd.DataFrame:
    # Gets full path
    abs_path = os.path.join(tables_directory, table_name)
    conn = sqlite3.connect(abs_path)

    # Creates table if doesn't exist
    conn.execute(DIRECTIONAL_ACCURACY_TABLE_SCHEMA)

    # Load existing data into a DataFrame
    df = pd.read_sql_query(f"SELECT * FROM {table_name.split('.')[0]}", conn)
    conn.close()
    return df


# Define the function to add DataFrame to SQLite
def directional_accuracy_history_add(table_name: str, table_to_add: pd.DataFrame) -> None:
    try:
        # Gets full path
        abs_path = os.path.join(tables_directory, table_name)
        conn = sqlite3.connect(abs_path)

        # Creates table if doesn't exist
        conn.execute(DIRECTIONAL_ACCURACY_TABLE_SCHEMA)

        # Serializes the list value
        table_to_add['da_list'] = table_to_add['da_list'].apply(json.dumps)

        # Adds new data from the DataFrame to the specified table
        table_to_add.to_sql(table_name.split('.')[0], conn, if_exists='append', index=False)

        # Commit changes and close the connection
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == '__main__':
    # Sample data to create a DataFrame
    import datetime

    data = {
        'model_name': ['ETH-USD_Custom_150_150_10_1d_A'],
        'da_score': [0.607804918],
        'da_list': [[0.632445, 0.532754, 0.432635]],  # List of floats
        'date': [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')]  # Current date and time
    }
    df_to_add = pd.DataFrame(data)
    directional_accuracy_history_add(DA_TABLE, df_to_add)
    downloaded_data = directional_accuracy_history_load(DA_TABLE)
