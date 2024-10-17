from typing import Optional, List, Tuple
from data_service.data_preparation import DataPreparation
from Const import TIME_INTERVALS, DAILY_ALERT_SOUND_PATH, HOURLY_ALERT_SOUND_PATH
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import kurtosis, skew
from playsound import playsound
import time
from datetime import datetime


def classify_distribution(skewness: float, kurt: float) -> tuple[str, Optional[float]]:
    """
    Classifies the distribution based on skewness and kurtosis.

    :param skewness: Skewness of the data
    :param kurt: Kurtosis of the data
    :return: A tuple containing the name of the distribution and degrees of freedom (for Chi-Square distribution)
    """
    if abs(skewness) < 0.5 and abs(kurt - 3) < 0.5:
        return "Normal Distribution", None
    elif skewness > 0.5 and kurt > 3:
        # Chi-square distribution, estimate degrees of freedom
        dof = 12 / kurt
        return "Chi-Square Distribution", dof
    elif skewness > 0.5 and kurt <= 3:
        return "Exponential Distribution", None
    elif abs(skewness) < 0.5 and kurt < 2:
        return "Uniform Distribution", None
    elif skewness < -0.5:
        return "Left-Skewed", None
    elif skewness > 0.5:
        return "Right-Skewed", None
    else:
        return "Unknown Distribution", None


def run_volume_analysis(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Perform analysis on both daily and hourly volume data for a given stock ticker.

    :param ticker: The stock ticker symbol
    :return: A tuple containing two DataFrames: daily and hourly statistics (mean, std, skew, kurtosis)
    """
    daily_int = TIME_INTERVALS[0]
    hourly_int = TIME_INTERVALS[1]

    # Download daily data
    daily_df = DataPreparation(ticker, 'Volume', 'B', daily_int, 100)._download_prices().tail(365)

    # Add weekday column to daily_df
    daily_df['Weekday'] = daily_df.index.to_series().dt.day_name()

    # Convert volume to millions for readability
    daily_df['Volume'] = daily_df['Volume'] / 1e6

    # Calculate statistics for each weekday (mean, std, skewness, kurtosis)
    daily_stats_df = daily_df.groupby('Weekday')['Volume'].agg(['mean', 'std', skew, kurtosis])

    # Download hourly data
    hourly_df = DataPreparation(ticker, 'Volume', 'B', hourly_int, 100)._download_prices().tail(365)

    # Add 'Hour' column to hourly_df
    hourly_df['Hour'] = hourly_df.index.hour  # Extract hour from the timestamp

    # Convert volume to millions for readability
    hourly_df['Volume'] = hourly_df['Volume'] / 1e6

    # Calculate statistics for each hour
    hourly_stats_df = hourly_df.groupby('Hour')['Volume'].agg(['mean', 'std', skew, kurtosis])

    return daily_stats_df, hourly_stats_df, daily_df, hourly_df


def plot_daily_distribution(stats_df: pd.DataFrame, daily_df: pd.DataFrame) -> None:
    """
    Plot the distribution of daily volume data.

    :param stats_df: DataFrame containing statistics (mean, std, skew, kurtosis) for each day
    :param daily_df: DataFrame containing daily volume data
    """
    # Classify the distribution type and estimate degrees of freedom for Chi-square
    stats_df[['Distribution', 'DoF']] = stats_df.apply(lambda x: classify_distribution(x['skew'], x['kurtosis']),
                                                       axis=1, result_type='expand')

    # Set seaborn style and color palette
    sns.set(style="whitegrid")
    plt.figure(figsize=(14, 8))  # Increase figure width to avoid crowding

    # Create a violin plot with a box plot overlay for volume distribution
    sns.violinplot(x='Weekday', y='Volume', data=daily_df,
                   order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                   inner=None, palette="muted", alpha=0.8, dodge=True)

    sns.boxplot(x='Weekday', y='Volume', data=daily_df,
                order=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'],
                width=0.2, color="k", boxprops=dict(alpha=0.3), dodge=True)

    # Set plot labels and title
    plt.xlabel('Weekday', fontsize=14, fontweight='bold')
    plt.ylabel('Volume (in Millions)', fontsize=14, fontweight='bold')
    plt.title('Volume Distribution per Weekday (in Millions)', fontsize=18, fontweight='bold')

    # Add distribution names, degrees of freedom, mean, and std as text annotations with staggered positions
    weekdays = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    for i, weekday in enumerate(weekdays):
        if weekday in stats_df.index:
            distribution = stats_df.loc[weekday, 'Distribution']
            dof = stats_df.loc[weekday, 'DoF']
            mean_val = stats_df.loc[weekday, 'mean']
            std_val = stats_df.loc[weekday, 'std']

            annotation = f"{distribution}"
            if dof is not None:
                annotation += f" (DoF: {dof:.2f})"
            annotation += f"\nMean: {mean_val:.2f}\nStd: {std_val:.2f}"

            # Stagger the vertical position based on even or odd index
            y_offset = 0.85 if i % 2 == 0 else 0.75

            # Dynamically adjust the position and add a background box to the text
            plt.text(i, daily_df['Volume'].max() * y_offset, annotation,
                     horizontalalignment='center',
                     fontsize=10, rotation=0, bbox=dict(facecolor='white', alpha=0.6))

    # Improve grid visibility
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add tight layout for better spacing
    plt.tight_layout()

    # Show the plot
    plt.show()


def plot_hourly_distribution(stats_df: pd.DataFrame, hourly_df: pd.DataFrame) -> None:
    """
    Plot the distribution of hourly volume data.

    :param stats_df: DataFrame containing statistics (mean, std, skew, kurtosis) for each hour
    :param hourly_df: DataFrame containing hourly volume data
    """

    # Classify the distribution type and estimate degrees of freedom for Chi-square
    stats_df[['Distribution', 'DoF']] = stats_df.apply(lambda x: classify_distribution(x['skew'], x['kurtosis']),
                                                       axis=1, result_type='expand')

    # Set seaborn style and color palette
    sns.set(style="whitegrid")

    # Increase figure size to ensure all annotations fit
    plt.figure(figsize=(18, 10))  # Increase figure size (width=18, height=10)

    # Create a violin plot with a box plot overlay for volume distribution by hour
    sns.violinplot(x='Hour', y='Volume', data=hourly_df,
                   inner=None, palette="muted", alpha=0.8, dodge=True)

    sns.boxplot(x='Hour', y='Volume', data=hourly_df,
                width=0.2, color="k", boxprops=dict(alpha=0.3), dodge=True)

    # Set plot labels and title
    plt.xlabel('Hour of Day', fontsize=14, fontweight='bold')
    plt.ylabel('Volume (in Millions)', fontsize=14, fontweight='bold')
    plt.title('Volume Distribution per Hour (in Millions)', fontsize=18, fontweight='bold')

    # Add distribution names, degrees of freedom, mean, and standard deviation as text annotations
    for i in range(24):
        if i in stats_df.index:
            distribution = stats_df.loc[i, 'Distribution']
            dof = stats_df.loc[i, 'DoF']
            mean = stats_df.loc[i, 'mean']
            std_dev = stats_df.loc[i, 'std']
            annotation = f"{distribution}\nMean: {mean:.2f}\nStd: {std_dev:.2f}"
            if dof is not None:
                annotation += f"\nDoF: {dof:.2f}"

            # Stagger the vertical position based on even or odd index to avoid overlapping
            y_offset = 0.85 if i % 2 == 0 else 0.75

            # Dynamically adjust the position and add a background box to the text
            plt.text(i, hourly_df['Volume'].max() * y_offset, annotation,
                     horizontalalignment='center',
                     fontsize=10, rotation=0, bbox=dict(facecolor='white', alpha=0.6))

    # Improve grid visibility
    plt.grid(True, linestyle='--', alpha=0.7)

    # Add tight layout for better spacing and adjust margins
    plt.tight_layout(pad=3.0)  # Adding padding between plots and the edge of the figure

    # Show the plot
    plt.show()


def compare_daily_volume(stats_df: pd.DataFrame, daily_df: pd.DataFrame) -> float:
    """
    Compare the latest daily volume to the mean daily volume for the current weekday.

    :param stats_df: DataFrame containing daily statistics (mean, std, etc.) for each weekday.
    :param daily_df: DataFrame containing the daily volume data.
    :return: None
    """
    latest_value = daily_df.iloc[-1, 1]  # Get the latest volume value from the daily data
    current_weekday = daily_df.iloc[-1, -1]  # Get the current weekday
    stat_mean = stats_df.loc[current_weekday, 'mean']  # Get the mean volume for the current weekday
    diff = latest_value / stat_mean  # Calculate the ratio between the latest value and the mean
    print(f'DAILY VOLUME RATIO IS {round(diff, 2)}')
    return diff


def compare_hourly_volume(stats_df: pd.DataFrame, hourly_df: pd.DataFrame) -> float:
    """
    Compare the latest hourly volume to the mean hourly volume for the current hour.

    :param stats_df: DataFrame containing hourly statistics (mean, std, etc.) for each hour.
    :param hourly_df: DataFrame containing the hourly volume data.
    :return: None
    """
    latest_value = hourly_df.iloc[-1, 1]  # Get the latest volume value from the hourly data
    current_hour = hourly_df.iloc[-1, -1]  # Get the current hour
    stat_mean = stats_df.loc[current_hour, 'mean']  # Get the mean volume for the current hour
    diff = latest_value / stat_mean  # Calculate the ratio between the latest value and the mean
    print(f'HOURLY VOLUME RATIO IS {round(diff, 2)}')
    return diff


def run_daily_volume_alert(tickers: List[str], ratio_lst: List[float], bool_lst: List[bool]) -> List[bool]:
    """
    Checks if tickers exceed their daily average trading volume and triggers an alert.

    Args:
        tickers (List[str]): List of ticker symbols to monitor.
        ratio_lst (List[float]): List of ratios, each representing the current volume divided by the daily average.
                                 A ratio > 1 means the ticker has exceeded its average daily volume.
        bool_lst (List[bool]): List of boolean flags that determine if the alert sound should be played for each ticker.
                               Once the sound is played, the flag is set to False to avoid repeat alerts.

    Returns:
        List[bool]: Updated bool list. It also prints alerts and plays sounds based on the conditions.
    """
    for ind in range(len(tickers)):
        if ratio_lst[ind] > 1:
            print(f'ATTENTION {tickers[ind]} HAS EXCEEDED DAILY AVERAGE VOLUME - {round(ratio_lst[ind], 2)}')
            if bool_lst[ind]:
                playsound(DAILY_ALERT_SOUND_PATH)
                bool_lst[ind] = False
                time.sleep(5)
    return bool_lst


def run_hourly_volume_alert(tickers: List[str], ratio_lst: List[float], bool_lst: List[bool]) -> List[bool]:
    """
    Checks if tickers exceed their hourly average trading volume and triggers an alert.

    Args:
        tickers (List[str]): List of ticker symbols to monitor.
        ratio_lst (List[float]): List of ratios, each representing the current volume divided by the daily average.
                                 A ratio > 1 means the ticker has exceeded its average hourly volume.
        bool_lst (List[bool]): List of boolean flags that determine if the alert sound should be played for each ticker.
                               Once the sound is played, the flag is set to False to avoid repeat alerts.

    Returns:
        List[bool]: Updated bool list. It also prints alerts and plays sounds based on the conditions.
    """
    for ind in range(len(tickers)):
        if ratio_lst[ind] > 1:
            print(f'ATTENTION {tickers[ind]} HAS EXCEEDED HOURLY AVERAGE VOLUME - {round(ratio_lst[ind], 2)}')
            if bool_lst[ind]:
                playsound(HOURLY_ALERT_SOUND_PATH)
                bool_lst[ind] = False
                time.sleep(5)

    return bool_lst


def reset_hourly_bool_lst(init_time: datetime, current_time: datetime, bool_list: List[bool]) -> Tuple[
    datetime, List[bool]]:
    """
    Resets a list of boolean values to True if the current hour is different from the initial hour.

    Parameters:
    init_time (datetime): The initial time to compare with.
    current_time (datetime): The current time.
    bool_list (List[bool]): A list of boolean values to reset.

    Returns:
    Tuple[datetime, List[bool]]: The updated initial time and the modified boolean list.
    """
    if current_time.hour != init_time.hour:
        init_time = current_time
        bool_list[:] = [True] * len(bool_list)  # Modify the list in place
    return init_time, bool_list


def run_momentum_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Perform momentum analysis on a given DataFrame with hourly financial data.

    The function calculates the momentum as the ratio of the 'Close' price to its
    previous value, drops any missing values, and computes statistical properties
    (mean, standard deviation, skewness, and kurtosis) grouped by the 'Hour' column.

    Parameters:
    -----------
    hour_df : pd.DataFrame
        DataFrame containing at least the columns 'Close' and 'Hour'. 'Close' represents
        the closing prices, and 'Hour' represents the hour of the day.

    Returns:
    --------
    tuple[pd.DataFrame, pd.DataFrame]
        - A DataFrame with the 'Momentum' column added and any NaN values removed.
        - A DataFrame with statistical properties (mean, standard deviation, skewness, and
          kurtosis) of the momentum for each hour.
    """
    hour_df = df.copy()
    hour_df['Momentum'] = hour_df['Close'] / hour_df['Close'].shift()
    hour_df.dropna(inplace=True)

    # Calculate statistics for each hour (mean, std, skewness, kurtosis)
    momentum_stats_df = hour_df.groupby('Hour')['Momentum'].agg(['mean', 'std', skew, kurtosis])

    return hour_df, momentum_stats_df


import pandas as pd


def compare_hourly_momentum(hour_df: pd.DataFrame, momentum_stats_df: pd.DataFrame) -> None:
    """
    Compare the most recent momentum value against statistical boundaries for the current hour.

    This function retrieves the most recent momentum value from the provided `hour_df` DataFrame
    and compares it to the statistical mean and standard deviation for the corresponding hour
    from the `momentum_stats_df` DataFrame. It prints a message indicating whether the current
    momentum is within one standard deviation of the mean, or if it exceeds the boundaries.

    Parameters:
    -----------
    hour_df : pd.DataFrame
        DataFrame containing hourly financial data, including a 'Momentum' column. The most
        recent data point is used for comparison.

    momentum_stats_df : pd.DataFrame
        DataFrame containing statistical properties (mean, standard deviation, etc.) of
        the momentum grouped by hour. This DataFrame should have a 'mean' and 'std' column.

    Returns:
    --------
    None
        The function prints the result of the comparison.
    """
    # Retrieve the most recent momentum value
    current_data = hour_df.iloc[-1].Momentum
    current_time = hour_df.index[-1].hour

    # Retrieve statistical mean and standard deviation for the current hour
    stat_mean = momentum_stats_df.loc[current_time]['mean']
    stat_std = momentum_stats_df.loc[current_time]['std']

    # Calculate the upper and lower boundaries
    top_boundary = stat_mean + stat_std
    bottom_boundary = stat_mean - stat_std

    # Compare current momentum with the boundaries
    if bottom_boundary < current_data < top_boundary:
        print(f'CURRENT MOMENTUM IS {current_data}')
    else:
        print(f'ATTENTION: CURRENT MOMENTUM {current_data} EXCEEDS STATISTICAL DATA {top_boundary, bottom_boundary}')
