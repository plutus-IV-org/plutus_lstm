import pandas as pd
from enum import Enum
import plotly.graph_objects as go
import numpy as np
from data_service.anomalies_detection import detect_anomalies
from utilities.directional_accuracy_services import save_directional_accuracy_score
from db_service.SQLite import directional_accuracy_history_load
from Const import DA_TABLE
from utilities.metrics import _directional_accuracy
from Const import Favorite, ShortsBatches, LongsBatches, ModelBatches
from app_service.candles_service import get_technical_data, create_candles_plot
from typing import List
import re


def is_crypto_asset(name):
    return '-USD' in name


def main_plot(data, asset_name, hover_or_click, anomalies_toggle, anomalies_window,
              rolling_period, zscore_lvl, candles_toggle):
    """
    Create a main price plot for a specified asset using Plotly.

    Args:
    - data (pd.DataFrame): DataFrame containing the price data.
    - asset_name (str): The name of the asset to plot.
    - hover_or_click (str): Set to 'Hover mode' for hover interactions, otherwise click interactions.

    Returns:
    - fig (go.Figure): The Plotly figure object.
    - df (pd.DataFrame): The updated DataFrame with extended dates.
    """
    if candles_toggle == 0:
        # Extracting asset-specific data from the provided DataFrame
        df = data[asset_name]
        x = df.index
        y = df.iloc[:, 0]
    else:
        name, interval = asset_name.split('_')
        if is_crypto_asset(asset_name):
            df = get_technical_data(name, interval, 'Binance').iloc[-128:, :]
        else:
            df = get_technical_data(name, interval).iloc[-128:, :]

    # Calculate time differences between consecutive entries to find the most common interval
    time_diffs = df.index.to_series().diff()
    most_common_diff = time_diffs.value_counts().idxmax()

    # Identify the granularity of the dataset (day, hour, minute, or second)
    if most_common_diff >= pd.Timedelta('1D'):
        most_common_diff = most_common_diff.days
        time_unit = 'D'
    elif most_common_diff >= pd.Timedelta('1H'):
        most_common_diff = most_common_diff.seconds // 3600
        time_unit = 'H'
    elif most_common_diff >= pd.Timedelta('1T'):
        most_common_diff = most_common_diff.seconds // 60
        time_unit = 'T'
    else:
        most_common_diff = most_common_diff.seconds
        time_unit = 'S'

    # Extending the time range of the data for future predictions or trend analysis
    max_date = df.index.max() + pd.Timedelta(most_common_diff * 20, unit=time_unit)

    # Generating new date range for the extended period
    most_common_diff = pd.Timedelta(most_common_diff, unit=time_unit)
    new_dates = pd.date_range(start=df.index.max() + most_common_diff, end=max_date, freq=most_common_diff)

    # Creating a new DataFrame with the additional dates
    df_new = pd.DataFrame(index=new_dates, columns=df.columns)
    df = pd.concat([df, df_new])

    # Configuring plot interactivity based on the hover_or_click argument
    if hover_or_click == 'Hover mode':
        hoverinfo_setting = 'none'
        hovertemplate_setting = None
    else:
        hoverinfo_setting = None
        hovertemplate_setting = 'Date: %{x} <br>Price: %{y:$.4f}<extra></extra>'

    if candles_toggle == 0:
        # Creating the plot with the configuration settings
        fig = go.Figure(data=go.Scatter(
            x=x,
            y=y,
            connectgaps=True,  # Connects gaps in the data with lines
            name="Price",
            mode='lines+markers',  # Combines line and marker styles
            hoverinfo=hoverinfo_setting,
            hovertemplate=hovertemplate_setting
        ))

        # Styling the plot - setting line color and axes titles
        fig.update_traces(line_color='white')
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_range=[df.index.min(), max_date],  # Sets the range of x-axis
            title_text=asset_name,
            uirevision="Don't change"  # Preserves the state of the plot (like zoom level) across updates
        )
    else:
        fig = create_candles_plot(df, asset_name)
    if anomalies_toggle == 1:
        copy_df = df.copy()
        copy_df.columns = ['Close']
        red_timestamps, grey_timestamps = detect_anomalies(copy_df, anomalies_window, rolling_period, zscore_lvl)

        if time_unit == 'D':
            tdelta = pd.Timedelta(days=1)
        elif time_unit == 'H':
            tdelta = pd.Timedelta(hours=1)
        elif time_unit == 'T':
            tdelta = pd.Timedelta(minutes=15)
        else:
            tdelta = pd.Timedelta(seconds=1)

            # Add red rectangles for red_days
        for timestamp in red_timestamps:
            fig.add_vrect(
                x0=timestamp - tdelta, x1=timestamp,
                fillcolor='red', opacity=0.5,
                layer='below', line_width=0,
            )

            # Add grey rectangles for grey_days
        for timestamp in grey_timestamps:
            fig.add_vrect(
                x0=timestamp - tdelta, x1=timestamp,
                fillcolor='grey', opacity=0.5,
                layer='below', line_width=0,
            )

    return fig, df


def dynamic_height(num_models):
    check_models = num_models * 2
    if 1 <= check_models <= 2:
        return 300 * (num_models * 2)
    elif 3 <= check_models <= 6:
        return 150 * (num_models * 2)
    else:
        return 100 * (num_models * 2)


def main_plot_formatting(fig):
    """
    Apply primary styling to a plotly figure.

    This function formats the primary plots with a specific style, making them suitable for main presentations.
    It sets a custom font, background colors, title, legend, and axes properties.

    Args:
    - fig (go.Figure): The Plotly figure object to be formatted.

    Returns:
    - go.Figure: The formatted Plotly figure object.
    """
    # Set a universal font for the plot
    font = "system-ui"

    # Update the layout of the figure with specific stylistic choices
    fig.update_layout(
        # Setting font and background properties
        font_family=font,
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent plot area background
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background

        # Setting title and legend properties
        title_font_size=20,
        title_x=0.1,
        showlegend=True,
        font_color='rgb(255, 255, 255)',

        # Setting x-axis properties
        xaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(150, 150, 150)',  # Grey axis line color
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family=font,
                size=12,
                color='rgb(255, 255, 255)',
            ),
        ),

        # Setting y-axis properties
        yaxis=dict(
            showline=True,
            showgrid=False,
            showticklabels=True,
            linecolor='rgb(150, 150, 150)',  # Grey axis line color
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family=font,
                size=12,
                color='rgb(255, 255, 255)',
            ),
        ),
    )

    # Adding grid lines to both x and y axes
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgb(150, 150, 150)')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgb(150, 150, 150)')
    return fig


def secondary_plot_formatting(fig):
    """
    Apply secondary styling to a plotly figure.

    This function is designed to format secondary or less prominent plots. It provides a simpler and cleaner style
    by hiding the x-axis details and using a minimalist approach for the y-axis.

    Args:
    - fig (go.Figure): The Plotly figure object to be formatted.

    Returns:
    - go.Figure: The formatted Plotly figure object.
    """
    font = "system-ui"

    fig.update_layout(
        # Setting font and background properties
        font_family=font,
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent plot area background
        plot_bgcolor='rgba(0,0,0,0)',  # Transparent plot background

        # Setting title and legend properties
        title_font_size=20,
        title_x=0.1,
        showlegend=False,  # Legend is not displayed in secondary plots
        font_color='rgb(255, 255, 255)',

        # Simplifying x-axis
        xaxis=dict(
            showline=False,
            showgrid=False,
            showticklabels=False,  # Hiding tick labels for a cleaner look
            linecolor='rgb(0, 0, 0)',
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family=font,
                size=12,
                color='rgb(255, 255, 255)',
            ),
        ),

        # Simplifying y-axis
        yaxis=dict(
            showline=False,
            showgrid=False,
            showticklabels=True,  # Keeping y-axis tick labels
            linecolor='rgb(255, 255, 255)',  # White line color for minimalism
            linewidth=2,
            ticks='outside',
            tickfont=dict(
                family=font,
                size=12,
                color='rgb(255, 255, 255)',
            ),
        ),
    )

    return fig


def crop_anomalies(df: pd.DataFrame, red_timestamps: List[pd.Timestamp],
                   grey_timestamps: List[pd.Timestamp]) -> pd.DataFrame:
    """
    Modifies a DataFrame by cropping values based on provided timestamps.

    This function takes a DataFrame and two lists of timestamps. For each timestamp in 'red_timestamps',
    it sets NaN in a cascading manner starting from the timestamp row. For each column, the range of NaNs
    increases by one step compared to the previous column. It then removes rows corresponding to
    'grey_timestamps' from the DataFrame.

    :param df: The DataFrame to be modified.
    :param red_timestamps: List of Timestamps where NaN cascading starts.
    :param grey_timestamps: List of Timestamps to be removed from the DataFrame.
    :return: A modified DataFrame with specific values cropped and certain rows removed.
    """
    steps = len(df.columns)
    # Cascading NaNs based on red_timestamps
    for red_t in red_timestamps:
        for t in range(steps):
            d = t + 1
            f = -d
            df_slice = df.loc[:red_t]
            ind = df_slice.index[f]
            # Setting NaNs in a cascading manner
            for x in range(d, steps):
                to_crop_column_name = x + 1
                df.loc[ind, to_crop_column_name] = np.nan

    # Removing rows corresponding to grey_timestamps
    df = df.loc[~df.index.isin(grey_timestamps)]

    return df


def auxiliary_dataframes(model, asset_prices, asset_predictions, asset_name, anomalies_toggle, anomalies_window,
                         rolling_period, zscore_lvl):
    """
    Create auxiliary dataframes for analysis of asset prices and predictions.

    Args:
    - model (str): The name of the model used for predictions.
    - asset_prices (dict): Historical prices of assets.
    - asset_predictions (dict): Predicted prices from the model.
    - asset_name (str): The name of the asset.

    Returns:
    - deviation_data (pd.DataFrame): The deviation of predictions from actual values.
    - combined_directions (pd.DataFrame): Combined direction of actual and predicted prices.
    """

    # Creating a DataFrame with asset prices
    historical_data = asset_prices[asset_name].copy()
    historical_data.columns = ['Price']
    prediction_data = asset_predictions[model]

    # Shifting and appending columns to calculate future deviations
    for shift_count in range(len(prediction_data.columns)):
        shifted_data = pd.DataFrame(historical_data['Price'].shift(-1 - shift_count)).rename(
            columns={'Price': -1 - shift_count})
        historical_data = pd.concat([historical_data, shifted_data], axis=1)

    auxiliary_df = historical_data.copy()
    auxiliary_df.dropna(inplace=True)
    historical_data.drop(columns=['Price'], inplace=True)
    historical_data.dropna(inplace=True)
    model_predictions = prediction_data.copy()

    # Matching indexes between actual and predicted data
    prediction_indexes = model_predictions.index.tolist()
    actual_indexes = auxiliary_df.index.tolist()
    matching_indexes = [date for date in prediction_indexes if date in actual_indexes]

    # Calculating deviations
    deviation_matrix = abs(
        model_predictions.loc[matching_indexes].values / historical_data.loc[matching_indexes].values - 1)
    deviation_data = pd.DataFrame(deviation_matrix.T, columns=historical_data.loc[matching_indexes].index)
    deviation_data.index = ['Lag ' + str(lag + 1) for lag in range(len(deviation_data.index))]

    # Preparing datasets for directional accuracy calculation
    model_predictions.columns = list(range(len(model_predictions.columns) + 1))[1:]
    model_predictions['Actual'] = auxiliary_df['Price']
    model_predictions = model_predictions[['Actual'] + [col for col in model_predictions.columns if col != 'Actual']]
    model_predictions.dropna(inplace=True)

    # Extracting attributes from model name
    model_attributes = model.split('_')
    future_days = model_attributes[2] if 'average' in model_attributes[1] else model_attributes[3]
    targeted = True if 'average' in model_attributes[1] else model_attributes[-1] == 'T'

    # Calculate differences in directions between actual and predicted prices
    direction_differences = model_predictions.copy()
    auxiliary_directions = auxiliary_df.copy()
    auxiliary_directions.columns = direction_differences.columns

    # Computing direction changes for prediction data
    for date in direction_differences.index:
        for column in reversed(direction_differences.columns):
            direction_differences.loc[date, column] = direction_differences.loc[date, column] - \
                                                      direction_differences.loc[date, 'Actual']
    direction_differences[direction_differences > 0] = 1
    direction_differences[direction_differences < 0] = -1

    # Removing the column representing the current day
    direction_differences = direction_differences[direction_differences.columns.tolist()[1:]]

    # Repeat the process for auxiliary data
    for date in auxiliary_directions.index:
        for column in reversed(auxiliary_directions.columns):
            auxiliary_directions.loc[date, column] = auxiliary_directions.loc[date, column] - auxiliary_directions.loc[
                date, 'Actual']
    auxiliary_directions[auxiliary_directions > 0] = 1
    auxiliary_directions[auxiliary_directions < 0] = -1

    # Removing the column representing the current day
    auxiliary_directions = auxiliary_directions[auxiliary_directions.columns.tolist()[1:]]

    if anomalies_toggle == 1:
        copy_df = asset_prices[asset_name].copy()
        copy_df.columns = ['Close']
        red_timestamps, grey_timestamps = detect_anomalies(copy_df, anomalies_window, rolling_period, zscore_lvl)
        direction_differences = crop_anomalies(direction_differences, red_timestamps, grey_timestamps)
        auxiliary_directions = crop_anomalies(auxiliary_directions, red_timestamps, grey_timestamps)
    # Calculate directional accuracy and other statistics
    result_df, trades_coverage_df = _directional_accuracy(auxiliary_directions, direction_differences,
                                                          {'future_days': future_days}, is_targeted=targeted,
                                                          reshape_required=False)

    # Combine direction data
    combined_directions = direction_differences + auxiliary_directions
    combined_directions[abs(combined_directions) == 1] = np.nan
    combined_directions[abs(combined_directions) == 2] = 1

    # Directional accuracy score
    directional_accuracy_score = result_df['6 months']

    # Calculating directional accuracy
    actual_percentage_change = auxiliary_df.loc[matching_indexes].T.pct_change().T.dropna(axis=1).values
    predicted_percentage_change = model_predictions.loc[matching_indexes].T.pct_change().T.dropna(axis=1).values
    accuracy_match = actual_percentage_change * predicted_percentage_change
    accuracy_match[accuracy_match > 0] = 1
    accuracy_match[accuracy_match < 0] = -1
    directional_accuracy = ((pd.DataFrame(accuracy_match) + 1) / 2).mean()

    # Print and save results
    if anomalies_toggle == 1:
        print('___DIRECTIONAL_ACCURACY_WITH_POST_ANOMALIES_TIMESTAMPS___')
    else:
        print('___DIRECTIONAL_ACCURACY_WITHOUT_POST_ANOMALIES_TIMESTAMPS___')
    print(f'Directional accuracy for model {model}')
    print(directional_accuracy_score.values)
    save_directional_accuracy_score(model, directional_accuracy_score)

    # Prepare and return data
    deviation_data_diff = pd.DataFrame(combined_directions.T, columns=auxiliary_df.loc[matching_indexes].index)
    deviation_data_diff.index = ['Lag ' + str(lag + 1) for lag in range(len(deviation_data_diff.index))]

    return deviation_data, deviation_data_diff


def select_dictionaries_by_type(full_dict: dict, key_word: str) -> dict:
    # Use dictionary comprehension to filter entries where the key contains key_word
    if key_word == 'all':
        return full_dict
    elif key_word == 'favorite':
        dict = {}
        for element in Favorite:
            name = element.value
            for key in full_dict.keys():
                if name in key:
                    dict[key] = full_dict[key]
        return dict
    else:
        return {key: value for key, value in full_dict.items() if key_word in key}


def select_dictionaries_by_interval(full_dict: dict, interval_length: str) -> dict:
    # Return the entire dictionary if it is empty or if interval_length is 'all'
    if len(full_dict) == 0 or interval_length == 'all':
        return full_dict
    else:
        # Initialize an empty dictionary to store the filtered entries
        output_dict = {}
        # Iterate through the dictionary keys
        for key in full_dict.keys():
            # Split the key into parts
            key_list = key.split('_')
            # Check if the interval_length is in the key parts
            if interval_length in key_list:
                # Add the key-value pair to the output dictionary
                output_dict[key] = full_dict[key]

        # If no matching items were found, return the entire dictionary
        if len(output_dict) == 0:
            return full_dict
        else:
            return output_dict


def group_by_history_score(df: pd.DataFrame) -> pd.DataFrame:
    df['date'] = pd.to_datetime(df['date']).dt.date
    output_df = df.groupby(['date', 'model_name']).last().reset_index()
    return output_df


def contains_enum_value(model_name: str, enum_class: Enum, nested: bool) -> bool:
    """
    Checks if the model name contains any of the values specified in the enum_class.

    :param model_name: Name of the model to check.
    :param enum_class: Enum class containing the values to check against.
    :param nested: A boolean flag indicating if the enum values are nested inside dictionaries.
    :return: True if the model name contains any of the enum values, False otherwise.
    """
    if nested:
        # Modified to handle enum values that are dictionaries
        for enum_value in enum_class:
            # Enum value is now a dict, so we iterate over its keys
            for key in enum_value.value.keys():
                if re.search(rf"{re.escape(key)}", model_name, re.IGNORECASE):
                    return True
    else:
        # The nested scenario remains unchanged
        for enum_value in enum_class:
            if re.search(rf"{re.escape(enum_value.value)}", model_name, re.IGNORECASE):
                return True
    return False


def filter_loaded_history_score(asset_interval: str, type: str) -> pd.DataFrame:
    """
    Filters loaded DataFrame based on model names matching certain criteria.

    :param asset_interval: A string representing the asset and interval, separated by '_'.
    :param type: The type of filtering to apply, based on specific characteristics like 'favorite'.
    :return: A filtered pandas DataFrame.
    """
    # Assuming directional_accuracy_history_load is defined elsewhere to load the DataFrame
    loaded_df = directional_accuracy_history_load(DA_TABLE)
    unique_names = set(loaded_df.model_name.values)
    asset, interval = asset_interval.split('_')
    matched_assets = [name for name in unique_names if name.split('_')[0] == asset]

    matched_interval = matched_assets if interval == 'All' else [
        model_name for model_name in matched_assets if
        re.search(rf"(?:^|_){re.escape(interval)}(?:$|_)", model_name, re.IGNORECASE)
    ]
    selected_df = loaded_df[loaded_df['model_name'].isin(matched_interval)]
    if type == 'all':
        output_df = selected_df.copy()
    elif type == 'favorite':
        output_df = selected_df[selected_df['model_name'].apply(lambda x: contains_enum_value(x, Favorite, False))]
    elif type == 'short-average':
        output_df = selected_df[selected_df['model_name'].apply(lambda x: contains_enum_value(x, ShortsBatches, True))]
    elif type == 'long-average':
        output_df = selected_df[selected_df['model_name'].apply(lambda x: contains_enum_value(x, LongsBatches, True))]
    elif type == 'top-average':
        output_df = selected_df[selected_df['model_name'].apply(lambda x: contains_enum_value(x, ModelBatches, True))]
    else:  # Case of simple-average
        output_df = selected_df[selected_df['model_name'].apply(lambda x: 'simple-average' in x.split('_'))]
    return output_df


def prepare_rating(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assigns rankings to models in a DataFrame based on 'da_score' for each date.

    The function sorts the DataFrame by 'date', groups by 'date',
    and assigns a rank to each model based on 'da_score' within each date group.
    The model with the highest 'da_score' for a date receives a rank of 1.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing 'model_name', 'da_score', and 'date'.

    Returns:
    pd.DataFrame: The DataFrame with an additional column 'rank' indicating the ranking.
    """

    # Ensure the DataFrame is sorted by 'date' for consistent ranking.
    df_sorted = df.sort_values(by='date')

    df_sorted.dropna(inplace=True)
    # Group by 'date' and rank 'da_score' within each group; highest score gets rank 1.
    df_sorted['rank'] = df_sorted.groupby('date')['da_score'].rank(method='first', ascending=False)

    # Convert the rank column to integers.
    df_sorted['rank'] = df_sorted['rank'].astype(int)

    return df_sorted


def filter_top_ranked_models(ranked_df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    """
    Filters the top N ranked models for each date from a DataFrame with ranked models.

    The function takes a DataFrame that contains model rankings and selects the rows
    where the rank is less than or equal to the specified top N rank.

    Parameters:
    ranked_df (pd.DataFrame): The DataFrame containing ranked models.
    top_n (int): The top rank level to include in the filtered DataFrame.

    Returns:
    pd.DataFrame: A DataFrame containing only the top N ranked models for each date.
    """

    # Ensure the input DataFrame is sorted by date and then by rank to maintain order.
    ranked_df_sorted = ranked_df.sort_values(by=['date', 'rank'])

    # Filter the DataFrame to only include rows where the rank is within the top N.
    top_ranked_df = ranked_df_sorted[ranked_df_sorted['rank'] <= top_n]

    return top_ranked_df


def get_intervals(asset_dict: dict) -> List:
    output_list = ['All']
    if len(asset_dict.keys())>0:
        for name in set(asset_dict.keys()):
            if 'average' in name:
                output_list.append(name.split('_')[2])
            else:
                output_list.append(name.split('_')[4])
    return list(set(output_list))

if __name__ == '__main__':
    df = filter_loaded_history_score('ETH-USD_All', 'simple-average')
    grouped_df = group_by_history_score(df)
    ranked_df = prepare_rating(grouped_df)
    filtered_df = filter_top_ranked_models(ranked_df, 2)
