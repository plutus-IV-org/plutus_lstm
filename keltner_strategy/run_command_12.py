import os
import sys
import json
import time
import datetime as dt

# Adjust path so we can import modules from one or two levels up as needed
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.getcwd())

# ----- Utilities / Imports -----
from keltner_strategy.src.data_handler import (
    data_pre_process,
    compute_keltner_channel,
    apply_channel_trading_logic,
    apply_trading_commission,
    concatenate_tables
)
from Const import BINANCE_API_KEY, BINANCE_SECRET_KEY, BINANCE_FOLDER_SCOPE
from utilities.binance_connector import BinanceTrader
from utilities.trading_services import (
    compute_sl_and_tp,
    calculate_quantity,
    convert_ticker,
    adjust_quantity,
    add_future_balance_to_db,
    add_previous_trade_to_db,
    add_hourly_order_to_db,
    add_commissions_into_db,
    get_close_side
)
from utilities.operational_analysis import OperationalAnalysis


def time_till_next_hour() -> float:
    """
    Returns how many seconds remain until the next full hour (local time).
    """
    current_time = dt.datetime.now()
    next_run_time = current_time.replace(minute=0, second=0, microsecond=0) + dt.timedelta(hours=1)
    wait_time = (next_run_time - current_time).total_seconds()
    return wait_time


def run(sl_coeff, tp_coeff, leverage, capital_percentage, testing, look_back=42):
    """
    Main trading loop, runs indefinitely:
      - Loads config for multiple assets
      - Computes signals for each
      - Rebalances positions across all signaled assets
      - Closes or flips positions by using SELL (if long) or BUY (if short)
      - Waits until next hour
    """
    # Create your Binance trading object and analysis object
    bt = BinanceTrader(BINANCE_API_KEY, BINANCE_SECRET_KEY)
    analysis = OperationalAnalysis()

    # Load config for assets
    config_path = os.path.join(os.getcwd(), "crypto_config.json")
    with open(config_path, "r") as file:
        config = json.load(file)
    assets_config = config.get("assets", {})

    # Keep track of last direction (+1=long, -1=short, 0=flat) for each asset
    last_directions = {asset: 0 for asset in assets_config}

    while True:
        print("_" * 60)
        current_time = dt.datetime.now()
        print(f"[INFO] New run started at {current_time}")

        # Identify the time windows for data retrieval
        current_hour = current_time.replace(minute=0, second=0, microsecond=0)
        look_back_time = current_hour - dt.timedelta(days=look_back)

        formatted_current_hour = current_hour.strftime("%Y%m%d %H")
        formatted_look_back_hour = look_back_time.strftime("%Y%m%d %H")

        # Set analysis range
        analysis.set_time_range_by_typing(formatted_look_back_hour, formatted_current_hour)

        wait_time = time_till_next_hour()
        if wait_time < 3480:
            print('[INFO]The difference between current time and next trading hour is smaller than set time.')
            print(f"Sleeping for {wait_time / 60:.2f} minutes until the next hour...")
            time.sleep(time_till_next_hour())
            continue

        # Gather signals for all assets first
        asset_signals = {}

        print("\n[INFO] Obtaining and computing data:")
        print("=" * 60)

        for asset, params in assets_config.items():
            interval = params['interval']
            try:
                # Download historical data
                technical_data, data_before_start = analysis.download_asset_technical_data(asset, interval)

                # Pre-process
                technical_data = data_pre_process(technical_data)
                data_before_start = data_pre_process(data_before_start)

                # Merge data
                df = concatenate_tables(data_before_start, technical_data)
                df = df[~df.index.duplicated(keep='first')]

                # Compute Keltner channels
                df_channeled = compute_keltner_channel(
                    df, params['ema'], params['atr'], params['multiplier']
                )[["Close", "Low", "High", "Lower_Band", "Upper_Band"]]

                # Filter the time range (take the last 1000 intervals to be safe)
                df_channeled = df_channeled[(df.index > analysis.start) & (df.index < analysis.end)].tail(1000)

                # Apply channel trading logic
                apply_channel_trading_logic(df_channeled, params['delay_percentage'], params['interval_delay'])

                # Apply commissions to returns
                commissions_df = apply_trading_commission(df_channeled, "taker")
                df_channeled['Adjusted_Returns'] = commissions_df['Channel_Target'].values * (
                        1 + df_channeled['Adjusted_Returns']
                )
                df_channeled['Cumulative_Returns'] = df_channeled['Adjusted_Returns'].cumprod()

                # We look one interval behind the current hour - 2 hours, consistent with your original code
                one_interval_back = current_hour - dt.timedelta(hours=2)

                if one_interval_back in df_channeled.index:
                    latest_available_info = df_channeled.loc[one_interval_back]

                    close_price = latest_available_info['Close']
                    channel_target = latest_available_info['Channel_Target']

                    print(f"\n📌 Asset: {asset}")
                    print("-" * 40)
                    print(f"Close Price         : {close_price:.2f}")
                    print(f"Prev Lower Band     : {latest_available_info['Prev_Lower_Band']:.2f}")
                    print(f"Prev Upper Band     : {latest_available_info['Prev_Upper_Band']:.2f}")
                    print(f"Buy Signal          : {latest_available_info['Buy_Signal']}")
                    print(f"Sell Signal         : {latest_available_info['Sell_Signal']}")
                    print(f"Channel Target      : {channel_target}")
                    print(f"Cumulative_Returns  : {latest_available_info['Cumulative_Returns']:.5f}")
                    print("-" * 40)

                    # Store the channel target and the close price
                    asset_signals[asset] = {
                        'channel_target': channel_target,
                        'close_price': close_price
                    }
                else:
                    print(f"[WARNING] No data for {asset} at {one_interval_back}, defaulting to HOLD")
                    asset_signals[asset] = {
                        'channel_target': 0,
                        'close_price': 0
                    }

            except Exception as e:
                print(f"[ERROR] Could not process {asset} due to: {e}")
                asset_signals[asset] = {
                    'channel_target': 0,
                    'close_price': 0
                }

        print("✅ [INFO] Data processing completed ✅")
        print("=" * 60)

        # (1) Check current positions from Binance
        try:
            current_positions = bt.check_positions()
        except Exception as e:
            print("[ERROR] Failed to check positions:", e)
            wait_time = time_till_next_hour()
            print(f"Sleeping {wait_time / 60:.2f} minutes until next hour...")
            time.sleep(wait_time)
            continue

        positions_dict = {}
        for p in current_positions:
            sym = p["symbol"]
            positions_dict[sym] = p

        # (2) Determine how many assets have nonzero signals
        nonzero_assets = [a for a in asset_signals if asset_signals[a]['channel_target'] != 0]
        num_nonzero = len(nonzero_assets)

        # Get account balances
        total_balance, available_balance = bt.get_account_balance()
        balance_dict = {
            'Total Balance': total_balance,
            'Available Balance': available_balance,
            'Time': dt.datetime.now(),
            'As of': current_hour
        }
        # Log the balance in DB
        add_future_balance_to_db(balance_dict, testing)

        if num_nonzero == 0:
            # No assets signaled => close all positions
            print("[INFO] All signals are 0. Closing all open positions.")
            for asset, pos_info in current_positions.items():
                old_dir = last_directions.get(asset, 0)
                current_size = pos_info.get('size', 0)
                if current_size > 0 and old_dir != 0:
                    # Figure out the side needed to close
                    closing_side = get_close_side(old_dir)
                    if closing_side:
                        print(f"[INFO] Closing {asset} with side={closing_side}, size={current_size}")
                        order_info = bt.create_market_order(
                            ticker=convert_ticker(asset),
                            side=closing_side,
                            quantity=current_size,
                            leverage=leverage
                        )
                        # DB logging
                        trading_dict = {
                            'Asset': asset,
                            'Diff': 1,
                            'Target': 0,
                            'Verbal': closing_side,
                            'Time': dt.datetime.now(),
                            'As of': current_hour,
                            'Current price': 0,  # or last known
                            'Quantity': current_size,
                            'Leverage': leverage,
                            'TP': 0,
                            'SL': 0,
                            'Executed': True if order_info else False,
                            'OrderId': order_info.get('orderId', 0) if order_info else 0
                        }
                        add_hourly_order_to_db(trading_dict, testing)

                        # Commissions, trade diffs
                        com = bt.get_commissions_for_closed_position(asset)
                        add_commissions_into_db(com, asset)
                        latest_trade_info = bt.get_diff_between_last_two_trades(asset, leverage)
                        add_previous_trade_to_db(latest_trade_info, testing)

                # Mark direction as 0 since we’re closed
                last_directions[asset] = 0

        else:
            # We do have some assets with signals
            portion_each = capital_percentage / num_nonzero
            print(f"[INFO] {num_nonzero} assets have nonzero signals, portion_each={portion_each:.2f}")

            # Loop through each asset to handle rebalancing
            for asset, sig_info in asset_signals.items():
                new_dir = sig_info['channel_target']
                current_price = sig_info['close_price']
                old_dir = last_directions.get(asset, 0)

                # If new_dir == 0 => close position if open
                if new_dir == 0:
                    # Close if there's a size
                    pos_size = current_positions.get(asset, {}).get('size', 0)
                    if pos_size > 0 and old_dir != 0:
                        closing_side = get_close_side(old_dir)
                        if closing_side:
                            print(f"[INFO] Closing {asset}, side={closing_side}, size={pos_size}")
                            order_info = bt.create_market_order(
                                ticker=convert_ticker(asset),
                                side=closing_side,
                                quantity=pos_size,
                                leverage=leverage
                            )
                            # DB logs
                            trading_dict = {
                                'Asset': asset,
                                'Diff': 1,
                                'Target': 0,
                                'Verbal': closing_side,
                                'Time': dt.datetime.now(),
                                'As of': current_hour,
                                'Current price': current_price,
                                'Quantity': pos_size,
                                'Leverage': leverage,
                                'TP': 0,
                                'SL': 0,
                                'Executed': True if order_info else False,
                                'OrderId': order_info.get('orderId', 0) if order_info else 0
                            }
                            add_hourly_order_to_db(trading_dict, testing)

                            # Commissions, trade diffs
                            com = bt.get_commissions_for_closed_position(asset)
                            add_commissions_into_db(com, asset)
                            latest_trade_info = bt.get_diff_between_last_two_trades(asset, leverage)
                            add_previous_trade_to_db(latest_trade_info, testing)

                    # Mark direction as 0
                    last_directions[asset] = 0
                    continue

                # If we get here, new_dir != 0 => we want a position
                # 1) Possibly close the old position if direction changed
                if old_dir != 0 and old_dir != new_dir:
                    # Close old
                    old_size = current_positions.get(asset, {}).get('size', 0)
                    if old_size > 0:
                        closing_side = get_close_side(old_dir)
                        print(
                            f"[INFO] Flipping {asset} from {old_dir} => {new_dir}. Closing old with side={closing_side} size={old_size}")
                        order_info_close = bt.create_market_order(
                            ticker=convert_ticker(asset),
                            side=closing_side,
                            quantity=old_size,
                            leverage=leverage
                        )
                        # DB logs for close
                        trading_dict_close = {
                            'Asset': asset,
                            'Diff': 1,
                            'Target': 0,
                            'Verbal': closing_side,
                            'Time': dt.datetime.now(),
                            'As of': current_hour,
                            'Current price': current_price,
                            'Quantity': old_size,
                            'Leverage': leverage,
                            'TP': 0,
                            'SL': 0,
                            'Executed': True if order_info_close else False,
                            'OrderId': order_info_close.get('orderId', 0) if order_info_close else 0
                        }
                        add_hourly_order_to_db(trading_dict_close, testing)

                        # Additional logs
                        com_close = bt.get_commissions_for_closed_position(asset)
                        add_commissions_into_db(com_close, asset)
                        last_trade_close = bt.get_diff_between_last_two_trades(asset, leverage)
                        add_previous_trade_to_db(last_trade_close, testing)

                # 2) Open (or adjust) the new position if needed
                # Decide quantity based on portion of capital
                if current_price == 0:
                    # Avoid dividing by zero. If data is missing, skip
                    print(f"[WARNING] {asset}: current_price=0. Skipping opening position.")
                    last_directions[asset] = 0
                    continue

                raw_quantity = calculate_quantity(current_price, total_balance, leverage, portion_each)
                quantity = adjust_quantity(raw_quantity, len(nonzero_assets))  # or adapt as needed

                # Check current size
                current_size = positions_dict.get(asset, {}).get('size', 0)
                if new_dir == old_dir and abs(current_size - quantity) > 1e-8:
                    # Reaffirm direction but adjust quantity
                    print(
                        f"[INFO] {asset}: Reaffirm direction {new_dir}, adjusting position from {current_size} => {quantity}")
                    # Close old
                    if current_size > 0:
                        closing_side = get_close_side(old_dir)
                        if closing_side:
                            bt.create_market_order(
                                ticker=convert_ticker(asset),
                                side=closing_side,
                                quantity=current_size,
                                leverage=leverage
                            )
                            # Log closing if needed, etc.
                            com_close = bt.get_commissions_for_closed_position(asset)
                            add_commissions_into_db(com_close, asset)
                            last_trade_close = bt.get_diff_between_last_two_trades(asset, leverage)
                            add_previous_trade_to_db(last_trade_close, testing)

                    # Now open new with correct size
                    opening_side = "BUY" if new_dir == 1 else "SELL"
                    sl, tp = compute_sl_and_tp(current_price, opening_side, sl_coeff, tp_coeff)
                    order_info = bt.create_market_order(
                        convert_ticker(asset),
                        opening_side,
                        quantity,
                        leverage=leverage,
                        take_profit=tp,
                        stop_loss=sl
                    )
                    # DB log
                    trading_dict = {
                        'Asset': asset,
                        'Diff': 1,
                        'Target': new_dir,
                        'Verbal': opening_side,
                        'Time': dt.datetime.now(),
                        'As of': current_hour,
                        'Current price': current_price,
                        'Quantity': quantity,
                        'Leverage': leverage,
                        'TP': tp,
                        'SL': sl,
                        'Executed': True if order_info else False,
                        'OrderId': order_info.get('orderId', 0) if order_info else 0
                    }
                    add_hourly_order_to_db(trading_dict, testing)

                elif new_dir != old_dir:
                    # We either just closed old_dir above or we had none
                    opening_side = "BUY" if new_dir == 1 else "SELL"
                    print(f"[INFO] Opening new {asset} direction={new_dir}, side={opening_side}, quantity={quantity}")
                    sl, tp = compute_sl_and_tp(current_price, opening_side, sl_coeff, tp_coeff)
                    order_info = bt.create_market_order(
                        convert_ticker(asset),
                        opening_side,
                        quantity,
                        leverage=leverage,
                        take_profit=tp,
                        stop_loss=sl
                    )
                    # DB log
                    trading_dict = {
                        'Asset': asset,
                        'Diff': 1,
                        'Target': new_dir,
                        'Verbal': opening_side,
                        'Time': dt.datetime.now(),
                        'As of': current_hour,
                        'Current price': current_price,
                        'Quantity': quantity,
                        'Leverage': leverage,
                        'TP': tp,
                        'SL': sl,
                        'Executed': True if order_info else False,
                        'OrderId': order_info.get('orderId', 0) if order_info else 0
                    }
                    add_hourly_order_to_db(trading_dict, testing)

                # Finally, store the new direction
                last_directions[asset] = new_dir

        # (3) Sleep until next hour
        wait_time = time_till_next_hour()
        print(f"[INFO] Run completed. Sleeping {wait_time / 60:.2f} minutes until next hour...\n")
        time.sleep(wait_time)


if __name__ == "__main__":
    # Example usage
    run(sl_coeff=0.7, tp_coeff=0.7, leverage=2, capital_percentage=1, testing=True, look_back=42)
