"""
This module provides utilities for analyzing the "directional accuracy" (DA) score
decay of different machine learning models over time.

Functions:
    load_da_register: Loads the directional accuracy register from an Excel file into a DataFrame.
    show_decay: Generates and returns a line plot visualizing the DA score decay over time for models
               whose name contains a specified substring.
"""

import pandas as pd
import plotly.express as px
from PATH_CONFIG import _ROOT_PATH


def load_da_register() -> pd.DataFrame:
    """
    Load the directional accuracy (DA) register from an Excel file into a DataFrame.

    Returns:
        pd.DataFrame: A DataFrame containing the directional accuracy register.
    """
    abs_path = _ROOT_PATH() + '\\vaults\\data_vault\\directional_accuracy_register.xlsx'
    # Load the Excel file into a DataFrame
    df = pd.read_excel(abs_path)
    return df


def show_decay(name: str, register_df: pd.DataFrame):
    """
    Generate and return a line plot that visualizes the DA score decay over time for models whose name
    contains a specified substring.

    Parameters:
        name (str): The substring to search for in the model names.
        register_df (pd.DataFrame): The DataFrame containing the directional accuracy register.

    Returns:
        fig (matplotlib.figure.Figure): The matplotlib Figure object for the plot.
        ax (matplotlib.axes._subplots.AxesSubplot): The matplotlib Axes object for the plot.
    """
    df = register_df[['model_name', 'da_score', 'date']].copy()
    df['date'] = pd.to_datetime(df['date']).dt.date
    df = df.groupby(['model_name', 'date'])['da_score'].min().reset_index()
    df = df.set_index('date')
    filtered_df = df[df['model_name'].str.contains(name, case=False, na=False)]
    model_counts = filtered_df['model_name'].value_counts()
    models_to_keep = model_counts[model_counts > 5].index
    filtered_df = filtered_df[filtered_df['model_name'].isin(models_to_keep)]

    # Use Plotly to create the figure
    fig = px.line(filtered_df, x=filtered_df.index, y='da_score', color='model_name',
                  labels={'date': 'Date', 'da_score': 'DA Score'},
                  title='DA Score Decay Over Time')
    return fig
