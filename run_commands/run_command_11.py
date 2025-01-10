"""Module for automatic trade on Binance"""

import os
import sys
import time

# Set the working directory to two levels up
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

from Const import BINANCE_API_KEY, BINANCE_SECRET_KEY, BINANCE_FOLDER_SCOPE
from utilities.binance_connector import BinanceTrader
from utilities.trading_services import *
from app_service.input_data_maker import generate_data


def run_session(sl_coeff, tp_coeff, leverage, capital_percentage, testing):
    bt = BinanceTrader(BINANCE_API_KEY, BINANCE_SECRET_KEY)
    last_direction = 0
    print('Good day Mr.Boloto! Let\'s get started!')

    while True:
        print("_" * 40)
        print('[INFO] New run started.')
        current_time = dt.datetime.now()
        print(f'[INFO] Current time is {current_time}')

        # Calculate wait time until the next hour
        next_run_time = current_time.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)
        wait_time = (next_run_time - current_time).total_seconds()

        if wait_time < 3480:
            print('The difference between current time and next trading hour is smaller than set time.')
            print(f"Sleeping for {wait_time / 60:.2f} minutes until the next hour...")
            time.sleep(wait_time)
            continue  # Restart the function from the beginning after sleeping

        # Account balance
        total_balance, available_balance = bt.get_account_balance()

        print('[INFO] Obtaining and computing data:')
        print("=" * 40)
        asset_prediction_dic, asset_price_dic, asset_names, interval = generate_data(BINANCE_FOLDER_SCOPE)
        print("=" * 40)

        if len(asset_prediction_dic) == 0 or len(asset_price_dic) == 0:
            bt.close_position('All')
            print('Could not obtain any prediction or price data')
            continue

        df = match_price_and_predictions(asset_prediction_dic, asset_price_dic)
        if df.empty:
            bt.close_position('All')
            print('Table data is empty')
            continue

        trading_dict = get_current_hourly_time_row(df)
        if trading_dict['Target'] == last_direction:
            print('Predicted new direction is the same as existing')
            continue

        last_direction = trading_dict['Target']
        print('[INFO] Algorithm indicates a new direction for trading')
        current_positions = bt.check_positions()
        if current_positions:
            bt.close_position('All')

        if len(trading_dict) == 0:
            print('Trading dict is empty')
            continue

        balance_dict = {'Total Balance': total_balance,
                        'Available Balance': available_balance,
                        'Time': trading_dict['Time'],
                        'As of': trading_dict['As of']}

        add_future_balance_to_db(balance_dict, testing)
        quantity = compute_quantity(trading_dict['Current price'], available_balance, capital_percentage)
        ticker = convert_ticker(trading_dict['Asset'])
        sl, tp = compute_sl_and_tp(trading_dict['Current price'], trading_dict['Verbal'], sl_coeff, tp_coeff)

        latest_trade_info = bt.get_diff_between_last_two_trades(ticker)
        add_previous_trade_to_db(latest_trade_info, testing)

        order_info = bt.create_market_order(
            ticker, trading_dict['Verbal'], quantity, leverage=leverage, take_profit=tp, stop_loss=sl
        )

        trading_dict['Quantity'] = quantity
        trading_dict['Leverage'] = leverage
        trading_dict['TP'] = tp
        trading_dict['SL'] = sl

        if order_info is not None:
            trading_dict['Executed'] = True
            trading_dict['OrderId'] = order_info['orderId']
        else:
            bt.close_position(ticker)
            trading_dict['Executed'] = False
            trading_dict['OrderId'] = 0

        com = bt.get_commissions_for_closed_position(ticker)
        add_commissions_into_db(com, ticker)
        add_hourly_order_to_db(trading_dict, testing)

        # Calculate wait time again at the end of the loop
        current_time = dt.datetime.now()
        next_run_time = current_time.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)
        wait_time = (next_run_time - current_time).total_seconds()

        if wait_time > 0:
            print(f"Sleeping for {wait_time / 60:.2f} minutes until the next hour...")
            time.sleep(wait_time)
            continue  # Restart the loop from the beginning after sleeping



if __name__ == '__main__':
    run_session(0.02, 0.03, 2, 1, False)
