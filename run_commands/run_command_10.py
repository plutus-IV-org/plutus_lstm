"""
Module for running Volume analysis
"""
import os
import sys

# Set the working directory to two levels up
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

from researchers.volume_research import *
from Const import CRYPTO_TICKERS
import datetime as dt
import time

init_time = dt.datetime.now()
daily_lst = []
hourly_lst = []
daily_bool_lst = [True] * len(CRYPTO_TICKERS)
hourly_bool_lst = [True] * len(CRYPTO_TICKERS)
separator = '_____________________________________________________________________________'

while True:
    print(separator)
    t1 = dt.datetime.now()
    print(f'{t1}')
    print(f'RUNNING VOLUME ANALYSIS FOR {CRYPTO_TICKERS}')
    for ticker in CRYPTO_TICKERS:
        print(separator)
        print(f'______________________________ {ticker} ______________________________________')
        print('COLLECTING DATA')
        daily_stats_df, hourly_stats_df, daily_df, hourly_df = run_volume_analysis(ticker)
        print('RUNNING ANALYSIS')
        daily_diff = compare_daily_volume(daily_stats_df, daily_df)
        hourly_diff = compare_hourly_volume(hourly_stats_df, hourly_df)
        daily_lst.append(daily_diff)
        hourly_lst.append(hourly_diff)
        print(separator)
        # plot_daily_distribution(daily_stats_df, daily_df)
        # plot_hourly_distribution(hourly_stats_df, hourly_df)
    t2 = dt.datetime.now()
    daily_bool_lst = run_daily_volume_alert(CRYPTO_TICKERS, daily_lst, daily_bool_lst)
    init_time, hourly_bool_lst = reset_hourly_bool_lst(init_time, t1, hourly_bool_lst)
    hourly_bool_lst = run_hourly_volume_alert(CRYPTO_TICKERS, hourly_lst, hourly_bool_lst)
    print(f'End time {t2}')
    print(f'The whole analysis has been performed for {t2 - t1}')
    print('\n')
    time.sleep(600)
