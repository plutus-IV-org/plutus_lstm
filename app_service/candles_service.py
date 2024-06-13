"""
This module is dedicated for candles service which is an optional parameter in dash app.
"""

import pandas as pd
from data_service import data_preparation as dp
import plotly.graph_objects as go


def get_technical_data(asset: str, interval: str, source: str = 'Yahoo') -> pd.DataFrame:
    """
    Retrieves technical data like CLose,Open,Volume, High, Low for a particular asset
    :param asset: str
        Asset name
    :param interval: str
        Time interval
    :param source: str
        Optional source of data. Yahoo by default
    :return: pd.DataFrame
        Table containing mentioned components
    """
    DP = dp.DataPreparation(asset, 'Technical', source, interval, 1)
    df = DP._download_prices()
    return df


def create_candles_plot(df: pd.DataFrame , name: str = 'Candlestick Chart'):
    """
    Create a candlestick chart using Plotly for an asset's technical data.

    Args:
    - df (pd.DataFrame): DataFrame containing 'Open', 'High', 'Low', and 'Close' data for the asset.

    Returns:
    - fig (go.Figure): The Plotly figure object with the candlestick chart.
    """
    # Create the candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        increasing_line_color='green',  # Color for increasing candlesticks
        decreasing_line_color='white',
        showlegend= False   # Color for decreasing candlesticks
    )])

    # Update the layout to add titles and make the chart clearer
    fig.update_layout(
        title=name,
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False  # Disable the range slider
    )

    return fig



    # Update the layout to add titles and make the chart clearer
    fig.update_layout(
        title=name,
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,  # Disable the range slider
    )

    return fig
if __name__ == '__main__':
    data = get_technical_data('ETH-USD', '1d')
    fig = create_candles_plot(data, 'ETH-USD_1d')
    fig.show()
