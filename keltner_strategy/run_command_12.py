import os
import sys

# Set the working directory to two levels up
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

import json
from keltner_strategy.src.data_handler import *
from Const import BINANCE_API_KEY, BINANCE_SECRET_KEY, BINANCE_FOLDER_SCOPE
from utilities.binance_connector import BinanceTrader
from utilities.trading_services import *
import time
from utilities.operational_analysis import OperationalAnalysis


def time_till_next_hour():
    current_time = dt.datetime.now()
    # Calculate wait time until the next hour
    next_run_time = current_time.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)
    wait_time = (next_run_time - current_time).total_seconds()
    return wait_time


def run(sl_coeff, tp_coeff, leverage, capital_percentage, testing, look_back: int = 42):
    bt = BinanceTrader(BINANCE_API_KEY, BINANCE_SECRET_KEY)
    last_direction = 0
    latest_ticker = None
    latest_quantity = 0
    analysis = OperationalAnalysis()

    while True:
        print("_" * 60)
        print('[INFO] New run started.')
        current_time = dt.datetime.now()
        print(f'[INFO] Current time is {current_time}')

        current_hour = current_time.replace(minute=0, second=0, microsecond=0)
        look_back_time = current_hour - dt.timedelta(days=look_back)

        # Calculate wait time until the next hour
        next_run_time = current_time.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)
        wait_time = (next_run_time - current_time).total_seconds()

        formatted_current_hour = current_hour.strftime("%Y%m%d %H")
        formatted_look_back_hour = look_back_time.strftime("%Y%m%d %H")

        analysis.set_time_range_by_typing(formatted_look_back_hour, formatted_current_hour)

        config_path = os.path.join(os.getcwd(), "crypto_config.json")

        # Load the JSON file
        with open(config_path, "r") as file:
            config = json.load(file)

        # Extract asset details
        assets = config.get("assets", {})

        # Dictionary to store results
        asset_results = {}

        # if wait_time < 3480:
        #     print('[INFO]The difference between current time and next trading hour is smaller than set time.')
        #     print(f"Sleeping for {wait_time / 60:.2f} minutes until the next hour...")
        #     time.sleep(time_till_next_hour())
        #     continue

        print("\n[INFO] Obtaining and computing data:")
        print("=" * 60)

        for asset, params in assets.items():
            technical_data, data_before_start = analysis.download_asset_technical_data(asset, params['interval'])
            technical_data = data_pre_process(technical_data)
            data_before_start = data_pre_process(data_before_start)
            df = concatenate_tables(data_before_start, technical_data)

            df = df[~df.index.duplicated(keep='first')]

            df_channeled = compute_keltner_channel(df, params['ema'], params['atr'], params['multiplier'])[
                ['Close', 'Low', 'High', 'Lower_Band', 'Upper_Band']]

            df_channeled = df_channeled[(df.index > analysis.start) & (df.index < analysis.end)].tail(1000)

            apply_channel_trading_logic(df_channeled, params['delay_percentage'], params['interval_delay'])
            one_interval_back = current_hour - dt.timedelta(hours=2)

            commissions_df = apply_trading_commission(df_channeled, "taker")
            df_channeled['Adjusted_Returns'] = commissions_df['Channel_Target'].values * (
                        1 + df_channeled['Adjusted_Returns'])
            df_channeled['Cumulative_Returns'] = df_channeled['Adjusted_Returns'].cumprod()

            if one_interval_back in df_channeled.index:
                latest_available_info = df_channeled.loc[one_interval_back]

                # Store asset data in dictionary
                asset_results[asset] = {
                    "Close_Price": latest_available_info['Close'],
                    "Previous_Lower_Band": latest_available_info['Prev_Lower_Band'],
                    "Previous_Upper_Band": latest_available_info['Prev_Upper_Band'],
                    "Buy_Signal": latest_available_info['Buy_Signal'],
                    "Sell_Signal": latest_available_info['Sell_Signal'],
                    "Channel_Target": latest_available_info['Channel_Target'],
                    "Next_Close": latest_available_info['Next_Close'],
                    "Returns": latest_available_info['Returns'],
                    "Adjusted_Returns": latest_available_info['Adjusted_Returns'],
                    "Cumulative_Returns": latest_available_info['Cumulative_Returns']
                }

                # Print formatted output
                print(f"\n📌 Asset: {asset}")
                print("-" * 40)
                print(f"📊 Close Price         : {latest_available_info['Close']:.2f}")
                print(f"📉 Previous Lower Band : {latest_available_info['Prev_Lower_Band']:.2f}")
                print(f"📈 Previous Upper Band : {latest_available_info['Prev_Upper_Band']:.2f}")
                print(f"🟢 Buy Signal         : {latest_available_info['Buy_Signal']}")
                print(f"🔴 Sell Signal        : {latest_available_info['Sell_Signal']}")
                print(f"🎯 Channel Target     : {latest_available_info['Channel_Target']}")
                print(f"📈 Cum Returns for 1000 intervals with commissions : {latest_available_info['Cumulative_Returns']:.5f}")
                print("-" * 40)
            else:
                print(f"[WARNING] No available data for {asset} at {one_interval_back}")

        print("✅ [INFO] Data processing completed  ✅")
        print("=" * 60)

        try:
            current_positions = bt.check_positions()
        except Exception as e:
            print(e)
            print(f"Sleeping for {wait_time / 60:.2f} minutes until the next hour...")
            time.sleep(time_till_next_hour())
            continue

        for asset in asset_results.keys():
            trading_dict = {
                'Asset': asset,
                'Diff': 1,
                'Target': asset_results[asset]['Channel_Target'],
                'Verbal': 'BUY' if asset_results[asset]['Channel_Target'] == 1 else (
                    'SELL' if asset_results[asset]['Channel_Target'] == 0 else 'HOLD'),
                'Time': current_time,
                'As of': current_hour,
                'Current price': asset_results[asset]['Close_Price']
            }

            sl, tp = compute_sl_and_tp(trading_dict['Current price'], trading_dict['Verbal'], sl_coeff, tp_coeff)
            if trading_dict['Target'] == last_direction:
                print('[Reaffirmation] Predicted new direction is the same as existing')
                # if len(current_positions) > 0:
                #     bt.cancel_open_orders_for_symbol('All')
                # bt.create_market_order(
                #     latest_ticker, trading_dict['Verbal'], latest_quantity, leverage=leverage, take_profit=tp,
                #     stop_loss=sl, readjustment=True
                # )
                print(f"Sleeping for {wait_time / 60:.2f} minutes until the next hour...")
                time.sleep(time_till_next_hour())
                continue

            else:
                last_direction = trading_dict['Target']
                print('[INFO] New direction identified, therefore the algorythm will proceed.')
                bt.close_position('All', leverage=leverage)
                bt.cancel_open_orders_for_symbol('All')

                # Account balance
                total_balance, available_balance = bt.get_account_balance()
                balance_dict = {'Total Balance': total_balance,
                                'Available Balance': available_balance,
                                'Time': trading_dict['Time'],
                                'As of': trading_dict['As of']}

                add_future_balance_to_db(balance_dict, testing)

                quantity = calculate_quantity(trading_dict['Current price'], available_balance, leverage,
                                              capital_percentage)
                ticker = convert_ticker(asset)
                latest_ticker = ticker
                latest_quantity = quantity
                latest_trade_info = bt.get_diff_between_last_two_trades(ticker, leverage)
                add_previous_trade_to_db(latest_trade_info, testing)

                com = bt.get_commissions_for_closed_position(ticker)
                add_commissions_into_db(com, ticker)

                if trading_dict['Target'] == 0:
                    print(f'[INFO] The price for {asset} is in the channel, therefore canceling existing trades. HOLD.')
                    print(f"Sleeping for {wait_time / 60:.2f} minutes until the next hour...")
                    time.sleep(time_till_next_hour())
                    continue
                else:
                    print('[INFO] Algorithm indicates a new direction for trading')
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
                        bt.close_position(ticker, leverage=leverage)
                        trading_dict['Executed'] = False
                        trading_dict['OrderId'] = 0

                    add_hourly_order_to_db(trading_dict, testing)
                    print(f"Sleeping for {wait_time / 60:.2f} minutes until the next hour...")
                    time.sleep(time_till_next_hour())
                    continue


if __name__ == "__main__":
    run(0.7, 0.7, 2, 1, True, 42)
