import pandas as pd


def data_pre_process(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the given DataFrame by computing percentage-based technical indicators.

    This function adds three new columns to the DataFrame:
    - `Pct_change`: The percentage change between the next closing price and the current closing price.
    - `Pct_high`: The percentage difference between the high and open price of the same period.
    - `Pct_low`: The percentage difference between the low and open price of the same period.

    Parameters:
    ----------
    df : pd.DataFrame
        A DataFrame containing historical price data with at least the following columns:
        - 'Open' : Opening price of the asset.
        - 'High' : Highest price of the asset during the period.
        - 'Low'  : Lowest price of the asset during the period.
        - 'Close': Closing price of the asset.

    Returns:
    -------
    pd.DataFrame
        The input DataFrame with three new calculated columns:
        - 'Pct_change'
        - 'Pct_high'
        - 'Pct_low'

    """
    # Calculating percentage changes and adding new columns
    df['Pct_change'] = (df['Close'].shift(-1) / df['Close']) - 1
    df['Pct_high'] = (df['High'] / df['Open']) - 1
    df['Pct_low'] = (df['Low'] / df['Open']) - 1

    return df


def concatenate_tables(df1, df2, tail_value: int = 200):
    return pd.DataFrame(pd.concat([
        df1[['Close', 'Low', 'High']].tail(tail_value),
        df2[['Close', 'Low', 'High']]
    ]))


def compute_keltner_channel(df, ema_period=65, atr_period=90, multiplier=1.5):
    """
    Computes a smoother Keltner Channel by increasing EMA and ATR periods, reducing sensitivity to market fluctuations.
    """
    if not {'Close', 'High', 'Low'}.issubset(df.columns):
        raise KeyError("The DataFrame must contain 'Close', 'High', and 'Low' columns.")
    df['EMA'] = df['Close'].ewm(span=ema_period, adjust=False).mean()
    df['ATR'] = (df['High'] - df['Low']).rolling(window=atr_period).mean()
    df['Upper_Band'] = df['EMA'] + (multiplier * df['ATR'])
    df['Lower_Band'] = df['EMA'] - (multiplier * df['ATR'])
    return df


def apply_channel_trading_logic(df, percentage_delay, interval_delay):
    """
    Updated logic:
    - Uses `percentage_delay` to adjust the buy/sell signal thresholds dynamically.
    - Uses `interval_delay` to delay trade execution after detecting a breakout.
    - If interval_delay > 0, trade only starts in the future IF price is still beyond the bands.
    """

    df['Prev_Lower_Band'] = df['Lower_Band'].shift(1) * (1 - percentage_delay)
    df['Prev_Upper_Band'] = df['Upper_Band'].shift(1) * (1 + percentage_delay)

    # ✅ Detect breakout signals at the moment they occur
    df['Buy_Signal'] = (df['Close'] > df['Prev_Upper_Band'])
    df['Sell_Signal'] = (df['Close'] < df['Prev_Lower_Band'])

    # ✅ Assign Channel Target (0 initially, delayed if needed)
    df['Channel_Target'] = 0
    df.loc[df['Buy_Signal'], 'Channel_Target'] = 1
    df.loc[df['Sell_Signal'], 'Channel_Target'] = -1

    # ✅ Apply interval delay: Remove target and only set it later if price is still out of bounds
    if interval_delay > 0:
        df['Channel_Target'] = df['Channel_Target'].shift(interval_delay).fillna(0)

        # ✅ Ensure that price is still out of bounds at the delayed interval
        df.loc[
            (df['Channel_Target'] == 1) & (df['Close'] <= df['Upper_Band']), 'Channel_Target'
        ] = 0  # Remove buy if price went back inside the bands

        df.loc[
            (df['Channel_Target'] == -1) & (df['Close'] >= df['Lower_Band']), 'Channel_Target'
        ] = 0  # Remove sell if price went back inside the bands

    # Compute next-period price for return calculation
    df['Next_Close'] = df['Close'].shift(-1)

    # Compute forward returns correctly
    df['Returns'] = (df['Next_Close'] - df['Close']) / df['Close']
    df['Adjusted_Returns'] = df['Returns'] * df['Channel_Target']

    # Compute cumulative returns
    df['Cumulative_Returns'] = (1 + df['Adjusted_Returns']).cumprod()

    return df


def apply_trading_commission(df, fee_type='taker'):
    """
    Applies trading commission only when the sign of a target column changes (excluding holding periods, i.e., 0 values).
    Returns a DataFrame where each target column contains 1 when no commission is applied and (1 - commission) when a trade occurs.

    Parameters:
    df (pd.DataFrame): Input DataFrame containing columns with 'target' in their names.
    fee_type (str): Type of commission to apply. Options:
                    - 'taker' (0.0005 or 0.05% for Binance Futures taker fee, applied twice for open/close)
                    - 'maker' (0.0002 or 0.02% for Binance Futures maker fee, applied twice for open/close)
                    - 'average' (0.00035 * 2 = 0.0007, assuming a mix of maker and taker trades)

    Returns:
    pd.DataFrame: A DataFrame with the same indices and target columns but with commission adjustments.
    """
    fees = {'taker': 0.0005 * 2, 'maker': 0.0002 * 2, 'average': 0.00035 * 2}  # Double fee for open/close
    commission = fees.get(fee_type, 0.0007)  # Default to 'average' if invalid option is given

    target_cols = [col for col in df.columns if 'arget' in col]

    commission_df = pd.DataFrame(1, index=df.index, columns=target_cols)  # Initialize with ones

    for col in target_cols:
        prev_values = df[col].shift(1).fillna(0)  # Shift to compare with previous value
        change_mask = (df[col] != prev_values) & (df[col] != 0)  # Detect changes, excluding holding periods
        commission_df.loc[change_mask, col] = 1 - commission  # Apply commission when change occurs

    return commission_df