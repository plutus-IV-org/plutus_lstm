import re
import os
import math
import pandas as pd
import datetime as dt
from db_service.SQLite import tables_directory
from db_service.schemas.tables_schemas import HOURLY_ORDERS_SCHEMA, FUTURES_BALANCE_SCHEMA
from Const import HOURLY_ORDERS_DATA_TABLE, FUTURES_BALANCE_TABLE
import sqlite3


def get_current_hourly_time_row(df) -> dict:
    output = {}
    current_time = dt.datetime.now()
    current_hour = dt.datetime.now().replace(minute=0, second=0, microsecond=0)
    adjusted_time = current_hour - dt.timedelta(
        hours=2)  # 1st hour back because of timezone 2nd hour back due to past COMPLETED hour
    if df.loc[adjusted_time].all():
        selected_row = df.loc[adjusted_time]
        output['Asset'] = df.columns[0]
        output['Diff'] = selected_row.loc['Diff']
        output['Target'] = selected_row.loc['Target']
        output['Verbal'] = selected_row['Verbal']
        output['Time'] = current_time
        output['As of'] = current_hour
        output['Current price'] = selected_row.loc[df.columns[0]]
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

        return concat_table


def add_hourly_order_to_db(input: dict, testing: bool = False):
    # Connect to SQLite database
    try:
        abs_path = os.path.join(tables_directory, HOURLY_ORDERS_DATA_TABLE)
        conn = sqlite3.connect(abs_path)

        # Create the table if it doesn't exist
        conn.execute(HOURLY_ORDERS_SCHEMA)

        # Insert the new row into the table
        conn.execute(
            f"""INSERT INTO {HOURLY_ORDERS_DATA_TABLE.split('.')[0]} (asset, diff, target, verbal, time, as_of,quantity, leverage,
             take_profit, stop_loss, executed, testing)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                input['Asset'],
                input['Diff'],
                int(input['Target']),
                input['Verbal'],
                input['Time'].strftime('%Y-%m-%d %H:%M:%S'),
                input['As of'].strftime('%Y-%m-%d %H:%M:%S'),
                input['Quantity'],
                input['Leverage'],
                input['TP'],
                input['SL'],
                input['Executed'],
                testing
            )
        )

        # Commit changes and close the connection
        conn.commit()
        conn.close()
        print(f'{input}, has been successfully loaded to the db')
    except Exception as e:
        print(e)


def add_future_balance_to_db(input: dict, testing: bool = False):
    # Connect to SQLite database
    try:
        abs_path = os.path.join(tables_directory, FUTURES_BALANCE_TABLE)
        conn = sqlite3.connect(abs_path)

        # Create the table if it doesn't exist
        conn.execute(FUTURES_BALANCE_SCHEMA)

        # Insert the new row into the table
        conn.execute(
            f"""INSERT INTO {FUTURES_BALANCE_TABLE.split('.')[0]} (total_balance, available_balance, time, as_of, testing)
                VALUES (?, ?, ?, ?, ?)""",
            (
                input['Total Balance'],
                input['Available Balance'],
                input['Time'].strftime('%Y-%m-%d %H:%M:%S'),
                input['As of'].strftime('%Y-%m-%d %H:%M:%S'),
                testing
            )
        )

        # Commit changes and close the connection
        conn.commit()
        conn.close()
        print(f'{input}, has been successfully loaded to the db')
    except Exception as e:
        print(e)


def compute_quantity(price: float, funds: float, percentage: float = 1.0) -> int:
    quantity = math.floor(funds * percentage / price) - 1
    print(quantity)
    return quantity


def convert_ticker(ticker: str) -> str:
    if ticker == 'XRP-USD':
        return 'XRPUSDT'
    else:
        return ticker
