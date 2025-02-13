import re
import os
import math
import pandas as pd
from typing import Tuple
import datetime as dt
from db_service.SQLite import tables_directory
from db_service.schemas.tables_schemas import HOURLY_ORDERS_SCHEMA, FUTURES_BALANCE_SCHEMA, COMMISSIONS_SCHEMA, \
    TRADE_HISTORY_SCHEMA
from Const import HOURLY_ORDERS_DATA_TABLE, FUTURES_BALANCE_TABLE, COMMISSIONS_TABLE, TRADE_HISTORY_TABLE
import sqlite3


def get_current_hourly_time_row(df) -> dict:
    output = {}
    current_time = dt.datetime.now()
    current_hour = dt.datetime.now().replace(minute=0, second=0, microsecond=0)
    adjusted_time = current_hour - dt.timedelta(
        hours=2)  # 1st hour back because of timezone 2nd hour back due to past COMPLETED hour
    time_for_printing = current_hour - dt.timedelta(minutes=1)
    print(f'[INFO] Selected data at {time_for_printing} CET')
    print(f'[INFO] Close price is {round(df.loc[adjusted_time].iloc[0], 2)}')
    print(f'[INFO] Forecasted price is {round(df.loc[adjusted_time].iloc[1], 2)}')
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
        concat_table['Target'] = concat_table['Diff'].apply(lambda x: -1 if x > 0 else 1)
        concat_table['Verbal'] = concat_table['Diff'].apply(lambda x: 'SELL' if x > 0 else 'BUY')

        return concat_table


def add_hourly_order_to_db(input: dict, testing: bool = False):
    """
    Adds an hourly order to the SQLite database and prints a clean summary of the order.

    Args:
        input (dict): The order data to be inserted into the database.
        testing (bool): Whether the order is for testing purposes (default is False).
    """
    try:
        # Connect to SQLite database
        abs_path = os.path.join(tables_directory, HOURLY_ORDERS_DATA_TABLE)
        conn = sqlite3.connect(abs_path)

        # Create the table if it doesn't exist
        conn.execute(HOURLY_ORDERS_SCHEMA)

        # Insert the new row into the table
        conn.execute(
            f"""INSERT INTO {HOURLY_ORDERS_DATA_TABLE.split('.')[0]} (asset, diff, target, verbal, time, as_of, quantity, price, leverage,
             take_profit, stop_loss, executed, order_id, testing)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                input['Asset'],
                input['Diff'],
                int(input['Target']),
                input['Verbal'],
                input['Time'].strftime('%Y-%m-%d %H:%M:%S'),
                input['As of'].strftime('%Y-%m-%d %H:%M:%S'),
                input['Quantity'],
                input['Current price'],
                input['Leverage'],
                input['TP'],
                input['SL'],
                input['Executed'],
                input['OrderId'],
                testing
            )
        )

        # Commit changes and close the connection
        conn.commit()
        conn.close()

        # Beautiful print statement
        print("\n[INFO] Order successfully loaded to the database:")
        print("=" * 40)
        print(f"Asset:          {input['Asset']}")
        print(f"Verbal:         {input['Verbal']}")
        print(f"Time:           {input['Time'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"As of:          {input['As of'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Quantity:       {input['Quantity']}")
        print(f"Current Price:  {input['Current price']}")
        print(f"Leverage:       {input['Leverage']}")
        print(f"Take Profit:    {input['TP']}")
        print(f"Stop Loss:      {input['SL']}")
        print(f"Executed:       {input['Executed']}")
        print(f"Order ID:       {input['OrderId']}")
        print(f"Testing:        {testing}")
        print("=" * 40)

    except Exception as e:
        print(f"[ERROR] An error occurred: {e}")


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
        print(f'[INFO] Account balance has been successfully loaded to the db')
    except Exception as e:
        print(f'[ERROR] {e}')


def add_commissions_into_db(value: float, ticker: str) -> None:
    current_hour = dt.datetime.now().replace(minute=0, second=0, microsecond=0)
    # Connect to SQLite database
    try:
        abs_path = os.path.join(tables_directory, COMMISSIONS_TABLE)
        conn = sqlite3.connect(abs_path)

        # Create the table if it doesn't exist
        conn.execute(COMMISSIONS_SCHEMA)

        # Insert the new row into the table
        conn.execute(
            f"""INSERT INTO {COMMISSIONS_TABLE.split('.')[0]} ( time, ticker, commission)
                VALUES (?, ?, ?)""",
            (
                current_hour.strftime('%Y-%m-%d %H:%M:%S'),
                ticker,
                value,

            )
        )

        # Commit changes and close the connection
        conn.commit()
        conn.close()
        print(f'[INFO] commission of {value}, has been successfully loaded to the db')
    except Exception as e:
        print(e)


def add_previous_trade_to_db(input: dict, testing: bool):
    """
    Inserts a trade record into the trade_history table in the SQLite database.

    Args:
        input (Dict[str, Any]): Trade details to be inserted into the table.
    """
    # Connect to SQLite database
    try:
        abs_path = os.path.join(tables_directory, TRADE_HISTORY_TABLE)
        conn = sqlite3.connect(abs_path)

        # Create the table if it doesn't exist
        conn.execute(TRADE_HISTORY_SCHEMA)

        # Insert the new row into the table
        conn.execute(
            f"""INSERT INTO {TRADE_HISTORY_TABLE.split('.')[0]} (
                    ticker, open_direction, close_direction, initial_price, final_price,
                    quantity, price_diff, total_difference, commissions_total,
                    leverage_1, leverage_2, initial_time, final_time , testing
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,?)""",
            (
                input['ticker'],
                input['direction'],
                input['closing_direction'],
                input['initial_price'],
                input['final_price'],
                input['quantity'],
                input['price_diff'],
                input['total_difference'],
                input['commissions_total'],
                input['leverage_1'],
                input['leverage_2'],
                input['initial_time'],
                input['final_time'],
                testing
            )
        )

        # Commit changes and close the connection
        conn.commit()
        conn.close()
        print(f"[INFO] Trade {input['ticker']} has been successfully loaded to the DB.")
    except Exception as e:
        print(f"[Error] Adding trade to DB: {e}")


def compute_quantity(price: float, funds: float, percentage: float = 1.0) -> int:
    quantity = math.floor(funds * percentage / price) - 1
    return quantity


def calculate_quantity(price: float, funds: float, leverage: int, percentage: float = 1.0,
                       maintenance_margin_rate: float = 0.005) -> int:
    """
    Calculates the maximum tradable quantity, initial margin, and maintenance margin,
    ensuring sufficient funds for both.

    Args:
        price (float): Current price of the asset.
        funds (float): Available funds in the account.
        leverage (int): Leverage applied to the trade.
        percentage (float): Percentage of funds to use (default 1.0).
        maintenance_margin_rate (float): Minimum funds required to keep the trade
    Returns:
        int: Max allowed quantity
    """
    # Adjust funds based on the percentage to use
    usable_funds = funds * percentage

    # Calculate the maximum quantity iteratively
    max_quantity = 0
    while True:
        position_value = max_quantity * price
        initial_margin = position_value / leverage
        maintenance_margin = position_value * maintenance_margin_rate
        total_margin = initial_margin + maintenance_margin

        if total_margin > usable_funds:
            break

        max_quantity += 1

    # Subtract 1 to avoid exceeding funds
    max_quantity = max(0, max_quantity - 1)

    return max_quantity


def convert_ticker(ticker: str) -> str:
    ticker_split = ticker.split('-')[0]
    return ticker_split + "USDT"


def compute_sl_and_tp(price: float, direction: str, sl_coefficient: float = 0.02, tp_coefficient: float = 0.03) -> \
        Tuple[float, float]:
    """
    Compute the stop-loss (SL) and take-profit (TP) levels for a given price and trade direction.

    Parameters:
        price (float): The entry price of the trade.
        direction (str): The trade direction, either 'BUY' or 'SELL'.
        sl_coefficient (float, optional): The stop-loss coefficient as a percentage of the price. Default is 0.02 (2%).
        tp_coefficient (float, optional): The take-profit coefficient as a percentage of the price. Default is 0.03 (3%).

    Returns:
        Tuple[float, float]: A tuple containing the stop-loss and take-profit prices.

    Example:
        compute_sl_and_tp(100.0, 'BUY')
        (98.0, 103.0)

        compute_sl_and_tp(100.0, 'SELL')
        (102.0, 97.0)
    """
    if direction == 'BUY':
        sl = round((1 - sl_coefficient) * price, 2)
        tp = round((1 + tp_coefficient) * price, 2)
    elif direction == 'SELL':
        sl = round((1 + sl_coefficient) * price, 2)
        tp = round((1 - tp_coefficient) * price, 2)
    elif direction == 'HOLD':
        sl = round((1 + sl_coefficient) * price, 2)
        tp = round((1 - tp_coefficient) * price, 2)
    else:
        raise ValueError("Direction must be 'BUY' or 'SELL'")

    return sl, tp
