import datetime
from typing import Dict, Any, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.dates as mdates


def proces_bulk_data(file_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Processes an Excel file with multiple sheets, searching each sheet for rows containing the word "average,"
    and extracting tables below those rows. The function returns a dictionary with sheet names as keys and a dictionary
    of extracted tables as values.

    Parameters:
    ----------
    file_path : str
        The path to the Excel file to be processed.

    Returns:
    -------
    results : Dict[str, Dict[str, Any]]
        A dictionary where the keys are sheet names, and the values are dictionaries containing extracted tables.
        Each table is represented by a pandas DataFrame.
    """
    # Load the Excel file
    xls = pd.ExcelFile(file_path)

    # Initialize a dictionary to store the results
    results = {}

    # Iterate through each sheet in the Excel file
    for sheet_name in xls.sheet_names:
        # Read the sheet into a DataFrame and reset the index twice to simplify processing
        df = pd.read_excel(xls, sheet_name=sheet_name)
        df = df.T.reset_index().T.reset_index(drop=True)

        # Search for the row containing "average"
        average_list = []
        for idx, cell in enumerate(df[0].tolist()):
            if "average" in str(cell):
                average_list.append(idx)

        # Initialize a dictionary to store tables found in this sheet
        tables = {}

        if len(average_list) > 0:
            for avg_row in average_list:
                # Extract the tables below the found row
                table_start_index = avg_row + 1  # Starting from the next row
                initial_sliced_table = df.iloc[table_start_index:].T.set_index(table_start_index).T
                sliced_table = initial_sliced_table.copy()

                # Find the point where the table ends (first row with NaN in the first column)
                for i in range(len(sliced_table)):
                    value = sliced_table.iloc[i, 0]
                    if pd.isna(value):
                        sliced_table = sliced_table.iloc[:i]
                        break

                # Identify columns that contain date-like objects to split the tables
                tc = []
                for c in range(len(sliced_table.columns)):
                    if isinstance(sliced_table.iloc[1, c], pd.Timestamp) \
                            or isinstance(sliced_table.iloc[1, c], datetime.datetime) \
                            and not pd.isna(sliced_table.iloc[1, c]):
                        tc.append(c)

                # Extract individual tables based on identified columns
                table_1 = sliced_table.iloc[:, 0:tc[1] - 1]
                table_2 = sliced_table.iloc[:, tc[1]:tc[2] - 1]
                table_3 = sliced_table.iloc[:, tc[2]:tc[3] - 1]
                table_4 = sliced_table.iloc[:, tc[3]:tc[4] - 2]
                table_5 = sliced_table.iloc[:, tc[4]:]

                # Set the first column as the index for each table
                table_1.set_index(table_1.columns[0], inplace=True)
                table_2.set_index(table_2.columns[0], inplace=True)
                table_3.set_index(table_3.columns[0], inplace=True)
                table_4.set_index(table_4.columns[0], inplace=True)
                table_5.set_index(table_5.columns[0], inplace=True)

                # Store the tables in a dictionary with the number of columns as the key
                tables[str(len(table_1.columns))] = table_1, table_2, table_3, table_4, table_5

        # Store the tables dictionary in the results under the sheet name
        results[sheet_name] = tables

    return results


def apply_borders_to_all_cells(worksheet, start_row: int, start_col: int, num_rows: int, num_cols: int,
                               workbook) -> None:
    """
    Apply borders to all cells in the specified range (starting from start_row and start_col) with the given dimensions
    without modifying existing cell content or other formatting.
    """
    border_only_format = workbook.add_format({'border': 1})

    for row in range(start_row, start_row + num_rows):
        for col in range(start_col, start_col + num_cols):
            worksheet.conditional_format(row, col, row, col, {'type': 'no_blanks', 'format': border_only_format})
            worksheet.conditional_format(row, col, row, col, {'type': 'blanks', 'format': border_only_format})


def enrich_with_future_steps(df: pd.DataFrame, num_future_steps: int) -> pd.DataFrame:
    """
    Extends the DataFrame with future time steps (rows) and NaN values for future predictions.

    Parameters:
    ----------
    df : pd.DataFrame
        The original DataFrame with time index and price/Lag columns.
    num_future_steps : int
        The number of future steps to add with NaN values.

    Returns:
    -------
    pd.DataFrame
        The enriched DataFrame with future steps and NaN values for future rows.
    """
    last_index = df.index[-1]
    time_delta = df.index[1] - df.index[0]  # Detect time step (hours, days, etc.)

    # Generate future timestamps based on the detected frequency
    future_dates = pd.date_range(start=last_index + time_delta, periods=num_future_steps, freq=time_delta)

    # Create a DataFrame of NaN values for future dates
    future_df = pd.DataFrame(index=future_dates, columns=df.columns)

    # Append the future DataFrame to the original DataFrame
    enriched_df = pd.concat([df, future_df])

    return enriched_df


def extend_time_series(series: pd.Series, num_future_steps: int) -> pd.Series:
    """
    Extends the time series by adding future timestamps based on the detected time step,
    and adds NaN as values for the future time steps.

    Parameters:
    ----------
    series : pd.Series
        The original time series with DatetimeIndex.
    num_future_steps : int
        The number of future time steps to add.

    Returns:
    -------
    pd.Series
        The extended time series with future time steps and NaN values for future dates.
    """
    time_values = series.index
    time_delta = time_values[1] - time_values[0]  # Detect time step

    # Extend the index by the same frequency with num_future_steps
    future_dates = pd.date_range(start=time_values[-1] + time_delta, periods=num_future_steps, freq=time_delta)

    # Create a series with NaN values for the new future dates
    future_series = pd.Series([pd.NA] * num_future_steps, index=future_dates)

    # Append future NaN series to the original series
    extended_series = series.append(future_series)

    return extended_series


def create_enriched_scatter_plot(df: pd.DataFrame, asset_name: str, plot_path: str, da_list_val: List[str]) -> None:
    """
    Creates a scatter plot for price over time, enriched with rectangles showing the forecast directions based on Lag (L)
    and Initial (I) values, and saves the plot as an image.

    Parameters:
    ----------
    df : pd.DataFrame
        A DataFrame containing the Lag (L) and Initial (I) values for each day.
    asset_name : str
        The name of the asset for labeling the plot.
    plot_path : str
        The path where the plot image will be saved.
    da_list_val : List[str]
        A list of values representing the height multipliers for each lag rectangle.
    """
    num_future_steps: int = len(df.columns) - 1  # Deduct the first column
    df = enrich_with_future_steps(df, num_future_steps)

    price_values = df.iloc[:, 0]  # The first column contains the actual price values
    future_steps = df.shape[1] - 1  # Number of future Lags (columns excluding the first)
    extended_time_values = extend_time_series(price_values, num_future_steps=future_steps)

    non_nan_mask = extended_time_values.notna()  # Create a mask for non-NaN values
    filtered_time_values = extended_time_values.index[non_nan_mask]
    filtered_price_values = extended_time_values[non_nan_mask]

    fig, ax = plt.subplots(figsize=(18, 6))  # Set the width to 18 to make the X-axis much wider
    plt.scatter(filtered_time_values, filtered_price_values, color='black', s=20, label=f'Price of {asset_name}',
                zorder=3)

    plt.title(f'Price and Predictions for {asset_name} (with {num_future_steps} lags)', fontsize=16, weight='bold')
    plt.xlabel('Time', fontsize=12, weight='bold', color='#333333')
    plt.ylabel('Price', fontsize=12, weight='bold', color='#333333')
    plt.xticks(rotation=45, fontsize=10, weight='bold', color='#555555')
    plt.yticks(fontsize=10, weight='bold', color='#555555')
    plt.grid(True, which='both', linestyle='--', linewidth=0.7, color='lightgrey')

    # Calculate Y-axis limits based on all values including height multipliers
    all_values = pd.concat([df.iloc[:, 0]] + [df.iloc[:, i] for i in range(1, future_steps + 1)])
    min_y, max_y = all_values.min(), all_values.max()
    padding = 0.05 * (max_y - min_y)  # 5% padding for tighter Y-axis limits
    ax.set_ylim([min_y - padding, max_y + padding])

    # Define the width of each sub-segment (split a day into parts)
    segment_width = pd.Timedelta(df.index[1] - df.index[0]) / (future_steps + 1)

    # Draw rectangles for Lag predictions
    for i in range(len(df)):
        initial_value = df.iloc[i, 0]  # Known price (Initial I) on current date
        total_width = future_steps * segment_width
        center_start_position = df.index[i] - total_width / 2  # Centering the rectangles around the black dot

        # For each Lag, get the forecast made on earlier dates
        for lag in range(1, future_steps + 1):
            past_date_index = i - lag  # Go back `lag` days to get the forecast for the current day
            if past_date_index >= 0:
                lag_value = df.iloc[past_date_index, lag]  # Get the predicted Lag value
                start_value = df.iloc[past_date_index, 0]

                # Skip if both lag_value and start_value are NaN
                if pd.isna(lag_value) or pd.isna(start_value):
                    continue  # Avoid plotting scatter points on NaN

                color = '#2ca02c' if lag_value >= start_value else '#d62728'  # Use green for higher, red for lower
                start_position = center_start_position + (lag - 1) * segment_width
                height_multiplier = float(da_list_val[lag - 1])

                # Draw the rectangle based on height multiplier
                if color == '#2ca02c':
                    rect = Rectangle(
                        (start_position, min(lag_value, start_value)),
                        segment_width,
                        abs(lag_value - start_value) * height_multiplier,
                        color=color, alpha=0.6, edgecolor='black', linewidth=1, zorder=2
                    )
                else:
                    rect = Rectangle(
                        (start_position, abs(lag_value - start_value) * (1 - height_multiplier) + lag_value),
                        segment_width,
                        abs(lag_value - start_value) * height_multiplier,
                        color=color, alpha=0.6, edgecolor='black', linewidth=1, zorder=2
                    )
                ax.add_patch(rect)

                # Ensure the label is centered within the rectangle
                font_size = 7 if num_future_steps <= 5 else 4
                if color == '#2ca02c':
                    label_y_position = min(lag_value, start_value) + abs(
                        lag_value - start_value) * height_multiplier / 2
                else:
                    label_y_position = abs(lag_value - start_value) * (1 - height_multiplier) + lag_value + abs(
                        lag_value - start_value) * height_multiplier / 2

                if abs(lag_value - start_value) * height_multiplier < 0.02 * (max_y - min_y):
                    # If the rectangle is very small, offset the label slightly for readability
                    label_y_position += 0.02 * (max_y - min_y)
                ax.text(start_position + segment_width / 2, label_y_position, f'{lag}',
                        ha='center', va='center', fontsize=font_size, color='black', weight='bold', alpha=0.9, zorder=4)

    # Extend X-axis limits to ensure that all future rectangles are shown, including NaN dates
    ax.set_xlim([filtered_time_values[0], extended_time_values.index[-1] + pd.Timedelta(df.index[1] - df.index[0])])

    # Set specific tick labels for every date on X-axis, including NaN values
    ax.set_xticks(extended_time_values.index)

    # Dynamically adjust the date formatter based on the time frequency
    time_delta = df.index[1] - df.index[0]
    if time_delta >= pd.Timedelta(days=1):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    elif time_delta >= pd.Timedelta(hours=1):
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
    else:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))

    # Save the plot with better padding
    plt.tight_layout(pad=2)
    plt.savefig(plot_path, dpi=300)
    plt.close()


def prepare_output_report(file_path: str, stored_data: Dict[str, Dict[str, Any]], output_file_path: str) -> None:
    """
    Prepares an output report by adding new sheets with processed tables, conditional formatting, borders, and a scatter plot
    to an existing Excel file, saving the result to a new location. All 'Analysis' sheets will be moved to the beginning.
    """
    # Use XlsxWriter to enable adding plots and formatting
    with pd.ExcelWriter(output_file_path, engine='xlsxwriter') as writer:
        # Load the existing data into the writer
        original_sheets = pd.ExcelFile(file_path).sheet_names

        for sheet_name in original_sheets:
            df_existing = pd.read_excel(file_path, sheet_name=sheet_name)
            df_existing.to_excel(writer, sheet_name=sheet_name, index=True)  # Ensure index=True for indices

        # Collect 'Analysis' sheets to reorder later
        analysis_sheets = []

        # Process each key in stored_data and append tables sequentially in the sheet
        for key, key_data in stored_data.items():
            sheet_name = f'Analysis_{key}'
            analysis_sheets.append(sheet_name)  # Collect for reordering later
            start_row = 1  # Starting row for placing tables vertically
            first_column_max_len = 0  # To track the maximum length for the first column

            # Loop through each dataset (tab_1, tab_2, etc.) for the current key
            for short_key, (tab_1, tab_2, tab_3, tab_4, tab_5) in key_data.items():

                # Make sure tab_1 and tab_4 have the same columns for later use
                tab_1.columns = tab_4.columns

                # Concatenate tab_1 and tab_2 side by side
                combined_df = pd.concat([tab_2, tab_1], axis=1)

                # Write the combined DataFrame to the sheet starting from `start_row`
                combined_df.to_excel(writer, sheet_name=sheet_name, startrow=start_row, index=True, header=True)

                # Access the XlsxWriter objects to apply formatting
                workbook = writer.book
                worksheet = writer.sheets[sheet_name]

                # Define combined formats (with borders and colors)
                pale_yellow_border = workbook.add_format(
                    {'bg_color': '#FFFF85', 'border': 1})  # Light Yellow with Border
                soft_green_border = workbook.add_format({'bg_color': '#3BFF94', 'border': 1})  # Soft Green with Border
                light_red_border = workbook.add_format({'bg_color': '#FF7979', 'border': 1})  # Light Red with Border
                green_font_border = workbook.add_format({'font_color': 'green', 'border': 1})  # Green font with border
                red_font_border = workbook.add_format({'font_color': 'red', 'border': 1})  # Red font with border
                orange_font_border = workbook.add_format(
                    {'font_color': 'orange', 'border': 1})  # Orange font with border

                # Format for the blue cell
                blue_background_format = workbook.add_format({'bg_color': '#ADD8E6', 'border': 1})  # Light Blue

                # --- Auto-size the first column ---
                first_column_header = combined_df.index.name if combined_df.index.name else ''
                first_column_data = combined_df.index.astype(str).tolist()
                first_column_values = [first_column_header] + first_column_data
                first_column_max_len = max(first_column_max_len, max([len(str(val)) for val in first_column_values]))

                # Apply the color and border formatting to `tab_4` based on matching index values
                for row in combined_df.index:
                    if row in tab_4.index:
                        for col in tab_4.columns:
                            tab_4_value = tab_4.loc[row, col]
                            combined_value = combined_df.loc[row, col]
                            row_number = combined_df.index.tolist().index(row)
                            column_number = combined_df.columns.tolist().index(col)

                            # Apply formatting with borders based on conditions
                            if pd.isna(tab_4_value):
                                worksheet.write(start_row + row_number + 1, column_number + 1, combined_value,
                                                pale_yellow_border)
                            elif tab_4_value == 1:
                                worksheet.write(start_row + row_number + 1, column_number + 1, combined_value,
                                                soft_green_border)
                            elif tab_4_value == 0:
                                worksheet.write(start_row + row_number + 1, column_number + 1, combined_value,
                                                light_red_border)
                    else:
                        # Forecasting color formatting based on growth/fall predictions
                        for col in tab_4.columns.tolist():
                            lag = tab_4.columns.tolist().index(col) + 1
                            row_num = combined_df.index.tolist().index(row)
                            if row_num + lag <= len(combined_df) - 1:
                                forecasted_value = combined_df.loc[row, col]
                                future_value = combined_df.iloc[row_num + lag, 0]
                                current_value = combined_df.iloc[row_num, 0]
                                forecasted_diff = forecasted_value - current_value
                                real_diff = future_value - current_value

                                forecasted_diff_bool = np.sign(forecasted_diff)
                                real_diff_bool = np.sign(real_diff)

                                directional_value = real_diff_bool * forecasted_diff_bool
                                if directional_value == 1:
                                    worksheet.write(start_row + row_num + 1, lag + 1, forecasted_value,
                                                    green_font_border)
                                elif directional_value == -1:
                                    worksheet.write(start_row + row_num + 1, lag + 1, forecasted_value, red_font_border)
                                else:
                                    worksheet.write(start_row + row_num + 1, lag + 1, forecasted_value,
                                                    orange_font_border)

                # --- Insert additional statistics from tab_5 with borders ---
                list_str = tab_5['da_list'].iloc[-1]  # Get the string
                da_list_values = list_str[2:-2].split(',')  # Extract list of values

                # Insert each value from da_list under the corresponding Lag column with border
                for i, value in enumerate(da_list_values[:len(tab_4.columns)], start=1):
                    worksheet.write(start_row + len(combined_df) + 1, i + 1, round(float(value), 2), pale_yellow_border)

                # Insert da_score under non-Lag columns with border
                latest_da_score = tab_5['da_score'].iloc[-1]
                worksheet.write(start_row + len(combined_df) + 1, 0, "DA Score", pale_yellow_border)
                worksheet.write(start_row + len(combined_df) + 1, 1, round(float(latest_da_score), 2),
                                pale_yellow_border)

                # Apply borders to all cells without overwriting content
                apply_borders_to_all_cells(
                    worksheet=worksheet,
                    start_row=start_row,
                    start_col=0,
                    num_rows=len(combined_df) + 2,  # Include data rows and statistics row
                    num_cols=len(combined_df.columns) + 1,  # Include all columns (adjust for index)
                    workbook=workbook
                )

                # Highlight the first non-NaN value in the last row (DA Score row) with blue background
                last_row_idx = start_row + len(combined_df)  # The last row with data

                # Loop through columns B onwards to find the first non-NaN value and highlight it
                for col in range(1, len(combined_df.columns) + 1):
                    value = combined_df.iloc[-1, col]
                    if not pd.isna(value):  # If the value is not NaN, highlight it
                        worksheet.write(last_row_idx, col, value, blue_background_format)
                        break  # Stop after highlighting the first non-NaN value

                # Create scatter plot with time as X-axis and price as Y-axis, and enrich with forecast rectangles
                # Price is the first column
                asset_name = combined_df.columns[0]

                # Path to save the scatter plot image
                plot_path = f'{sheet_name}_{short_key}_enriched_scatter_plot.png'
                create_enriched_scatter_plot(combined_df, asset_name, plot_path, da_list_values)

                # Insert the enriched scatter plot image into the Excel sheet to the right of the table
                scatter_start_col = len(combined_df.columns) + 3  # Column "O" is around the 15th column
                worksheet.insert_image(start_row, scatter_start_col, plot_path)

                # Move `start_row` down for the next table and add buffer space
                start_row += len(combined_df) + 10

            # Set the width of the first column to the max length calculated
            worksheet.set_column(0, 0, first_column_max_len + 10)

        # Reorder sheets by moving the analysis sheets to the beginning
        workbook.worksheets_objs = sorted(
            workbook.worksheets_objs, key=lambda ws: 0 if ws.name in analysis_sheets else 1
        )

    print(f"Output report has been saved to {output_file_path}")