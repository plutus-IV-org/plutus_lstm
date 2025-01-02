import re
import os
import pandas as pd
import datetime as dt
from db_service.SQLite import tables_directory
from db_service.schemas.tables_schemas import ORDERS_SCHEMA
import sqlite3

def get_current_time_row(df):
    output = None
    current_time = dt.datetime.now().replace(minute=0, second=0, microsecond=0)
    adjusted_time = current_time - dt.timedelta(hours=1)
    if df.loc[adjusted_time].all():
        output = df.loc[adjusted_time]
    return output


def match_price_and_predictions(predictions_dict: dict, price_dict: dict, lags: int = 1):
    for asset_full_key in price_dict.keys():
        # Extract the asset_key from the full key
        asset_key = asset_full_key.split('-')[0]  # Extracting base key like 'XRP'
        # Create regex pattern to exclude interval like '1h'
        pattern = re.compile(fr'{re.escape(asset_key)}-USD.*simple-average.*_future_steps')
        # Find matching keys in predictions_dict
        matching_keys = [key for key in predictions_dict.keys() if pattern.search(key)]
        price_asset = price_dict[asset_full_key]
        prediction_asset = predictions_dict[matching_keys[0]]
        concat_table = pd.concat([price_asset, prediction_asset], axis=1)
        concat_table['Diff'] = concat_table[concat_table.columns[0]] - concat_table[concat_table.columns[lags]]
        # Add 'Target' column
        concat_table['Target'] = concat_table['Diff'].apply(lambda x: 1 if x > 0 else -1)
        concat_table['Verbal'] = concat_table['Diff'].apply(lambda x: 'BUY' if x > 0 else 'SELL')

        # Get the current row and current time
        current_row = get_current_time_row(concat_table)
        current_time = dt.datetime.now()

        # Add time and as_of time to the current row
        current_row['time'] = current_time
        current_row['as_of'] = current_time

        # Connect to SQLite database
        abs_path = os.path.join(tables_directory, 'orders')
        conn = sqlite3.connect(abs_path)

        # Create the table if it doesn't exist
        conn.execute(ORDERS_SCHEMA)

        # Insert the new row into the table
        conn.execute(
            f"""INSERT INTO orders (asset, diff, target, verbal, time, as_of)
                VALUES (?, ?, ?, ?, ?, ?)""",
            (
                'XRP-USD',
                current_row['Diff'],
                current_row['Target'],
                current_row['Verbal'],
                current_row['time'].strftime('%Y-%m-%d %H:%M:%S'),
                current_row['as_of'].strftime('%Y-%m-%d %H:%M:%S')
            )
        )

        # Commit changes and close the connection
        conn.commit()
        conn.close()
        q = 1
    pass


def add_entry_to_db():
    # Getting tables directory
    current_file_path = os.path.abspath(__file__)
    directory_path = os.path.dirname(current_file_path)
    tables_directory = os.path.join(directory_path, 'tables')

    def technical_data_add(table_name: str, table_to_add: pd.DataFrame, table_schema: str) -> None:
        # Get the full path to the database
        abs_path = os.path.join(tables_directory, table_name)