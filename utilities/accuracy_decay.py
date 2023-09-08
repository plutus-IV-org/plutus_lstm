"""
This module provides utilities for analyzing the "directional accuracy" (DA) score
decay of different machine learning models over time.

Functions:
    load_da_register: Loads the directional accuracy register from an Excel file into a DataFrame.
    show_decay: Generates and returns a line plot visualizing the DA score decay over time for models
               whose name contains a specified substring.
"""

import pandas as pd
import matplotlib.pyplot as plt
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
    df = register_df[['model_name', 'da_score', 'date']].copy()  # Explicitly create a copy

    # Make sure the 'date' column is of datetime dtype
    df['date'] = pd.to_datetime(df['date'])

    # Drop the time component
    df['date'] = df['date'].dt.date

    # Handle duplicate indices by keeping only the row with the minimum 'da_score'
    idx_to_keep = df.groupby(['model_name', 'date'])['da_score'].idxmin()
    df = df.loc[idx_to_keep]

    # Set the modified 'date' column as index
    df = df.set_index('date')

    # Filter rows where 'model_name' contains the variable_name
    filtered_df = df[df['model_name'].str.contains(name, case=False, na=False)]

    # Create a Figure and Axes object
    fig, ax = plt.subplots(figsize=(18, 9))

    for model in filtered_df['model_name'].unique():
        subset = filtered_df[filtered_df['model_name'] == model]
        ax.plot(subset.index, subset['da_score'], label=model)

    ax.set_xlabel('Date')
    ax.set_ylabel('DA Score')
    ax.legend(title='Model Name', fontsize='small')
    ax.set_title('DA Score Decay Over Time')
    ax.grid(True)

    return fig, ax
