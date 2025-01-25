import sqlite3
import pandas as pd
from datetime import timedelta, datetime
from IPython.display import display  # Import display for Jupyter
import re
from typing import Optional
from data_service.vendors_data.binance_downloader import downloader


def get_table_path(table_name: str) -> str:
    """
    Returns the absolute path to the specified database table.

    Args:
        table_name (str): Name of the table file (e.g., 'commissions_table.db').

    Returns:
        str: Absolute path to the table file.
    """
    import os
    current_file_path = os.path.abspath(__file__)
    directory_path = os.path.dirname(os.path.dirname(current_file_path))
    tables_directory = os.path.join(directory_path, 'db_service', 'tables')
    return os.path.join(tables_directory, table_name)


class OperationalAnalysis:
    def __init__(self, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None):
        """
        Initializes the OperationalAnalysis class with optional start and end times.

        Args:
            start_time (Optional[datetime]): Start time of the analysis range.
            end_time (Optional[datetime]): End time of the analysis range.
        """
        self.start: Optional[datetime] = start_time
        self.end: Optional[datetime] = end_time
        self.dates_table: Optional[pd.DataFrame] = None

    def load_timestamps(self) -> None:
        """
        Loads timestamps from the database table and adjusts them to UK time (UTC - 1 hour).
        """
        abs_path = get_table_path('commissions_table.db')
        try:
            connection = sqlite3.connect(abs_path)
            print("Connection successful.")

            # Query all timestamps
            query = f"SELECT time FROM commissions_table"
            self.dates_table = pd.read_sql_query(query, connection)

            # Adjust times to UKT by subtracting 1 hour
            self.dates_table['time'] = pd.to_datetime(self.dates_table['time']) - timedelta(hours=1)
            self.dates_table.reset_index(drop=True, inplace=True)  # Remove redundant index
            self.dates_table.index.name = 'ID'  # Rename index to 'ID'

        except sqlite3.Error as e:
            print("Error connecting to database:", e)
        finally:
            if connection:
                connection.close()

    def load_table_from_db(self, table_tame: str, column_name: str) -> pd.DataFrame:
        """
        Loads data from the database between self.start and self.end times.
        Adjusts the timestamps from CET to UKT and returns a DataFrame with UKT indices.

        Args:
            table_tame (str): Name of the table to load data from.
            column_name (str): Name of the column in each table is different.

        Returns:
            pd.DataFrame: DataFrame containing the filtered data with UKT indices.
        """
        if self.start is None or self.end is None:
            raise ValueError("Start and end times must be defined before loading data.")

        # Convert self.start and self.end (UKT) to CET for querying the database (+1 hour)
        start_cet = (self.start + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')
        end_cet = (self.end + timedelta(hours=1)).strftime('%Y-%m-%d %H:%M:%S')

        table_name_db = table_tame + '.db'
        abs_path = get_table_path(table_name_db)

        try:
            connection = sqlite3.connect(abs_path)
            # Query data within the start and end time range (CET)
            # Use SQLite strftime to truncate fractional seconds for compatibility
            query = f"""
            SELECT *
            FROM {table_tame}
            WHERE strftime('%Y-%m-%d %H:%M:%S', "{column_name}") 
            BETWEEN '{start_cet}' AND '{end_cet}'
            """
            df = pd.read_sql_query(query, connection)

            # Convert timestamps from CET to UKT (-1 hour)
            df[column_name] = pd.to_datetime(df[column_name]) - timedelta(hours=1)

            # Set the column as the DataFrame index
            df.set_index(column_name, inplace=True)
            return df

        except sqlite3.Error as e:
            print("Error connecting to database:", e)
            return pd.DataFrame()  # Return an empty DataFrame in case of an error
        finally:
            if connection:
                connection.close()

    def display_timestamps(self) -> None:
        """
        Displays all available timestamps. Uses a table format in Jupyter and plain text otherwise.
        """
        if self.dates_table is not None:
            print("Available timestamps (adjusted to UKT):")
            pd.set_option('display.max_rows', None)  # Show all rows in Jupyter
            try:
                get_ipython  # This function exists only in Jupyter/IPython
                display(self.dates_table)  # Use Jupyter-friendly display
            except NameError:
                # Fallback for non-Jupyter environments (e.g., PyCharm)
                print(self.dates_table.to_string())  # Print all rows in plain text
        else:
            print("Timestamps are not loaded. Please call `load_timestamps()` first.")

    def select_time_range(self) -> None:
        """
        Prompts the user to select a start and end time by index. Validates the selections and sets
        the `start` and `end` attributes to `datetime` objects.
        """
        if self.dates_table is None:
            print("Timestamps are not loaded. Please call `load_timestamps()` first.")
            return

        try:
            # Prompt for start and end IDs
            start_id = int(input("Select start time ID: "))
            end_id = int(input("Select end time ID: "))

            # Validate the selected IDs
            if start_id < 0 or end_id >= len(self.dates_table) or start_id > end_id:
                raise ValueError("Invalid start or end time ID.")

            # Set start and end times (convert to datetime.datetime explicitly)
            self.start = self.dates_table.iloc[start_id]['time'].to_pydatetime()
            self.end = self.dates_table.iloc[end_id]['time'].to_pydatetime()
            print(f"Start time: {self.start}, End time: {self.end}")
        except ValueError as ve:
            print("Error:", ve)

    def set_time_range_by_typing(self, start_str: str, end_str: str) -> None:
        """
        Sets the start and end times by typing them directly in the format 'YYYYMMDD HH'.

        Args:
            start_str (str): The start time as a string in 'YYYYMMDD HH' format.
            end_str (str): The end time as a string in 'YYYYMMDD HH' format.

        Raises:
            ValueError: If the input format is incorrect or if the start time is after the end time.
        """

        def validate_and_convert(time_str: str) -> datetime:
            """
            Validates and converts a time string to a datetime object.

            Args:
                time_str (str): The time string in 'YYYYMMDD HH' format.

            Returns:
                datetime: The converted datetime object.

            Raises:
                ValueError: If the input format is incorrect.
            """
            pattern = r"^\d{8} \d{2}$"  # Format: YYYYMMDD HH
            if not re.match(pattern, time_str):
                raise ValueError(f"Invalid time format: {time_str}. Expected format is 'YYYYMMDD HH'.")
            return datetime.strptime(time_str, "%Y%m%d %H")

        try:
            # Validate and convert inputs
            start_time = validate_and_convert(start_str)
            end_time = validate_and_convert(end_str)

            # Ensure start time is earlier than or equal to end time
            if start_time > end_time:
                raise ValueError("Start time must be earlier than or equal to end time.")

            # Set start and end times
            self.start = start_time
            self.end = end_time
            print(f"Start time set to: {self.start}, End time set to: {self.end}")
        except ValueError as ve:
            print("Error:", ve)

    def download_asset_technical_data(self, asset: str, interval: str) -> pd.DataFrame:
        """
        Downloads asset data from Binance and filters it between self.start and self.end times.
        Adjusts the timestamps from CET (Binance) to UKT and returns the filtered data.

        Args:
            asset (str): The asset symbol (e.g., 'BTCUSDT').
            interval (str): The interval for the data (e.g., '1h', '4h', etc.).

        Returns:
            pd.DataFrame: DataFrame containing the filtered data with UKT indices.
        """
        if self.start is None or self.end is None:
            raise ValueError("Start and end times must be defined before downloading data.")

        # Download the data (assumes downloader returns a DataFrame with a 'time' column in CET)
        df = downloader(asset, interval)  # Replace this with the actual downloader logic
        df.reset_index(inplace=True)
        # Convert 'time' column to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['Time']):
            df['Time'] = pd.to_datetime(df['Time'])

        # Filter data between start and end times (CET)
        df_filtered = df[(df['Time'] >= self.start) & (df['Time'] <= self.end)].copy()

        # Set 'time' as the DataFrame index
        df_filtered.set_index('Time', inplace=True)

        return df_filtered


# Example Usage:
if __name__ == "__main__":
    # Instantiate the class but don't trigger prompts on import
    analysis = OperationalAnalysis()

    # Load timestamps and display them
    analysis.load_timestamps()
    analysis.display_timestamps()

    # When ready, call the method to select the time range
    analysis.select_time_range()
    analysis.download_asset_technical_data('XRP-USDT', '1h')
    q = 1
