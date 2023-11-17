import pandas as pd
from typing import List, Tuple


def detect_anomalies(df: pd.DataFrame, n_window: int, period: int = 20, zscore_lvl: int = 2) -> Tuple[
    List[pd.Timestamp], List[pd.Timestamp]]:
    """
    Detects anomalies in financial data based on z-scores of daily changes in closing prices.

    Parameters:
    df (pd.DataFrame): A DataFrame containing financial data. Must include a 'Close' column.
    n_window (int): The number of days to capture after each detected anomaly.
    period (int, optional): The rolling window period for calculating the mean and standard deviation. Defaults to 20.
    zscore_lvl (int, optional): The z-score level used as a threshold for detecting anomalies. Defaults to 2.

    Returns:
    Tuple[List[pd.Timestamp], List[pd.Timestamp]]: A tuple of two lists.
        The first list contains the indices of the detected anomalies.
        The second list contains indices for the days following each anomaly within the specified window.

    Raises:
    ValueError: If the DataFrame does not have a 'Close' column.
    """

    # Check if 'Close' column exists
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must have a 'Close' column")

    # Calculate daily changes as a ratio of closing prices
    df['Daily_change'] = df.Close / df.Close.shift()

    # Calculate rolling mean and standard deviation of daily changes
    df['rolling_mean'] = df['Daily_change'].rolling(window=period).mean()
    df['rolling_std'] = df['Daily_change'].rolling(window=period).std()

    # Calculate z-scores of daily changes
    df['z_score'] = abs(df['Daily_change'] - df['rolling_mean']) / df['rolling_std']

    # Detect anomalies using the specified z-score level
    anomalies = df[df['z_score'] > zscore_lvl].index

    # Data collector for indices post anomalies
    post_anomalies_periods = []

    # Iterate over detected anomalies to collect subsequent indices
    list_of_indexes = df.index.tolist()
    for anomaly in anomalies:
        anomaly_index_in_df = list_of_indexes.index(anomaly)
        first_to_capture = anomaly_index_in_df + 1
        last_to_capture = anomaly_index_in_df + n_window

        # Ensure indices are within DataFrame bounds
        if first_to_capture < len(list_of_indexes) and last_to_capture < len(list_of_indexes):
            periods_to_capture = list_of_indexes[first_to_capture:last_to_capture + 1]
        elif first_to_capture < len(list_of_indexes) <= last_to_capture:
            periods_to_capture = list_of_indexes[first_to_capture:]
        else:
            break

        # Add indices to collector
        for ind in periods_to_capture:
            post_anomalies_periods.append(ind)

    # Clear duplicates and sort indices
    post_anomalies_cleared = list(set(post_anomalies_periods))
    post_anomalies_cleared.sort()

    # Return anomalies and post-anomaly periods
    return anomalies.tolist(), post_anomalies_cleared


if __name__ == '__main__':
    import yfinance as yf
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    data = yf.download('ETH-USD')
    lst_1, lst_2 = detect_anomalies(data, 3, 21)
    # Plotting
    fig, ax = plt.subplots()

    # Plot 'Close' prices
    ax.plot(data.index, np.log(data['Close']), label='Log Close Price')

    # Highlight anomalies
    for timestamp in lst_1:
        ax.axvspan(timestamp - pd.DateOffset(days=1), timestamp, color='red', alpha=0.3)

    # Highlight post-anomaly periods
    for timestamp in lst_2:
        ax.axvspan(timestamp - pd.DateOffset(days=1), timestamp, color='grey', alpha=0.3)

    # Formatting the plot
    ax.xaxis.set_major_locator(mdates.MonthLocator())  # setting major ticks to be monthly
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))  # formatting ticks as 'Year-Month'
    plt.xticks(rotation=45)
    plt.xlabel('Date')
    plt.ylabel('Close Price')
    plt.title('Ethereum USD Closing Prices with Anomaly Detection')
    plt.legend()
    plt.tight_layout()

    # Display the plot
    plt.show()
