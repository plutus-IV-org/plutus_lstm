"""Module for automatic trade on Binance"""

import os
import sys
import glob
import shutil
import datetime as dt

# Set the working directory to two levels up
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

from Const import BINANCE_SECRET_KEY, BINANCE_API_KEY, BINANCE_FOLDER_SCOPE
from utilities.binance_connector import BinanceTrader
from utilities.trading_services import match_price_and_predictions, get_current_hourly_time_row, add_hourly_order_to_db, \
    compute_quantity, convert_ticker, add_future_balance_to_db
from app_service.input_data_maker import generate_data

bt = BinanceTrader(BINANCE_API_KEY, BINANCE_SECRET_KEY)
current_positions = bt.check_positions()
open_orders = bt.check_open_orders()
total_balance, available_balance = bt.get_account_balance()

asset_prediction_dic, asset_price_dic, asset_names, interval = generate_data(BINANCE_FOLDER_SCOPE)
df = match_price_and_predictions(asset_prediction_dic, asset_price_dic)

# Get the current row and current time
trading_dict = get_current_hourly_time_row(df)

balance_dict = {'Total Balance': total_balance,
                'Available Balance': available_balance,
                'Time': trading_dict['Time'],
                'As of': trading_dict['As of']}

# Adding info into db
add_future_balance_to_db(balance_dict, True)


quantity = compute_quantity(trading_dict['Current price'], total_balance, 1)

ticker = convert_ticker(trading_dict['Asset'])
order_info = bt.create_market_order(ticker, trading_dict['Verbal'], quantity, leverage= 1, take_profit= 10 ,stop_loss= 0)

trading_dict['Quantity'] = quantity
trading_dict['Leverage'] = 2
trading_dict['TP'] = 10
trading_dict['SL'] = 1
trading_dict['Executed'] = True
add_hourly_order_to_db(trading_dict, True)
q = 1
