import sqlite3
import os
import shutil
import pandas as pd
import datetime as dt
import json
import glob
from db_service.schemas.tables_schemas import *
from Const import DA_TABLE

# Getting tables directory
current_file_path = os.path.abspath(__file__)
directory_path = os.path.dirname(current_file_path)
tables_directory = os.path.join(directory_path, 'tables')  # Use os.path.join for path concatenation


def create_backup(db_path):
    backup_path = f"{db_path}.backup_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copyfile(db_path, backup_path)
    return backup_path


def clean_old_backups(directory: str, days: int = 20) -> None:
    """
    Cleans up backup files older than the specified number of days.

    Parameters:
    directory (str): The directory where the backup files are stored.
    days (int): The number of days to keep backups. Files older than this will be deleted.
    """
    current_time = dt.datetime.now()
    cutoff_time = current_time - dt.timedelta(days=days)

    # List all backup files in the directory
    backup_files = glob.glob(os.path.join(directory, '*.backup_*'))

    for backup_file in backup_files:
        # Get the creation time of the file
        creation_time = dt.datetime.fromtimestamp(os.path.getctime(backup_file))

        # Delete the file if it is older than the cutoff time
        if creation_time < cutoff_time:
            os.remove(backup_file)
            print(f"Deleted backup: {backup_file}")


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


# Define the function to load DataFrame into SQLite
def technical_data_load(table_name: str) -> pd.DataFrame:
    # Gets full path
    abs_path = os.path.join(tables_directory, table_name)
    conn = sqlite3.connect(abs_path)

    # Load existing data into a DataFrame

    query = f"SELECT time, open, high, low, close, volume FROM {table_name.split('.')[0]}"
    df = pd.read_sql_query(query, conn)

    conn.close()
    # Rename the columns
    df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    return df


def technical_data_add(table_name: str, table_to_add: pd.DataFrame, table_schema: str) -> None:
    try:
        # Get the full path to the database
        abs_path = os.path.join(tables_directory, table_name)

        # Create a backup of the database
        backup_path = create_backup(abs_path)
        print(f"Backup created at {backup_path}")

        # Removing old backups
        clean_old_backups(tables_directory)

        conn = sqlite3.connect(abs_path)

        # Create the table if it doesn't exist
        conn.execute(table_schema)

        # Load existing data
        table_to_add.reset_index(inplace=True)
        load_df = technical_data_load(table_name)
        load_df['Time'] = load_df['Time'].astype('datetime64')

        if load_df.empty:
            final_df = table_to_add.copy()
        else:
            final_df = pd.concat([load_df, table_to_add]).drop_duplicates(subset='Time', keep='last')
            if not final_df.empty:
                query = f"DELETE FROM {table_name.split('.')[0]}"
                conn.execute(query)

        # Add new data from the DataFrame to the specified table
        final_df.to_sql(table_name.split('.')[0], conn, if_exists='append', index=False)

        # Commit changes and close the connection
        conn.commit()
        conn.close()

        print(f'Successfully updated {table_name} at {dt.datetime.now()}')
        print(f'Previous table length was {len(load_df)}')
        print(f'New length is {len(final_df)}')
        print('________________________________________________________')

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
