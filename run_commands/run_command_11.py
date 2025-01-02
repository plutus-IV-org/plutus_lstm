"""Module for automatic trade on Binance"""

import os
import sys
import glob
import shutil

# Set the working directory to two levels up
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

from Const import BINANCE_SECRET_KEY, BINANCE_API_KEY, BINANCE_FOLDER_SCOPE
from utilities.binance_connector import BinanceTrader
from utilities.prediction_table_creator import match_price_and_predictions
from app_service.input_data_maker import generate_data

bt = BinanceTrader(BINANCE_API_KEY, BINANCE_SECRET_KEY)
current_positions = bt.check_positions()
open_orders = bt.check_open_orders()
total_balance, available_balance = bt.get_account_balance()

asset_prediction_dic, asset_price_dic, asset_names, interval = generate_data(BINANCE_FOLDER_SCOPE)
match_price_and_predictions(asset_prediction_dic, asset_price_dic)



q=1