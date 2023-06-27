from datetime import datetime
import pandas as pd
from meteostat import Point, Daily


def get_daily_meteo(most_dense: bool = True) -> pd.DataFrame:
    """
    This function fetches the daily average temperature data from Meteostat for two locations:
    Vancouver, BC and an arbitrary point that is "far from Vancouver". The data spans from
    1st Jan, 2010 to the current day.

    The function takes one optional argument, `most_dense`, which if set to True,
    will return the data in its most dense form i.e., without any missing values.
    If False, an Exception is raised indicating that only default cities are allowed.

    The result is a DataFrame where each column represents the average temperature of a city
    on a particular day.

    :param most_dense: boolean (optional), default is True.
                       If set to True, the function will return the most dense data.
                       If set to False, the function raises an exception as only default
                       cities are currently supported.

    :return: A DataFrame with daily average temperature data for the two locations.
             Each column represents a city with the column name in the format 'tavg in {city_name}'.
    """
    # Set time period
    start = datetime(2010, 1, 1)  # Start date is January 1, 2010
    end = datetime.now()  # End date is current day

    if most_dense:
        # These are the latitude, longitude and altitude for Vancouver, BC
        vanc = Point(49.2497, -123.1193, 0), 'Vancouver'
        pho = Point(33.46076014614866, -112.0869992615002), 'Phoenix'
        los_ang = Point(34.04849688640299, -118.25008238877551), 'Los Angeles'
        hou = Point(29.88064262016017, -95.92717225525821), 'Houston'
        chi = Point(41.89034725724475, -87.63206686339016), 'Chicago'
        cities_cordinates_lst = [vanc, pho, los_ang, hou, chi]
    else:
        raise Exception('Only default cities allowed so far.')

    # Prepare an empty DataFrame to hold the data for all cities
    full_df = pd.DataFrame()

    # Fetch the data for each city and append to the DataFrame
    for citi in cities_cordinates_lst:
        # Get daily data from start to end date for the specified location
        # The fetch function fetches the data from Meteostat and
        # the indexer at the end is used to filter only the 'tavg' (average temperature) column
        data = Daily(citi[0], start, end)
        data = data.fetch()[['tavg']] + 100

        # Rename the column to include the city name
        city_name = citi[1]
        data.rename(columns={'tavg': f'tavg in {city_name}'}, inplace=True)

        # Ensure that the data is of type float
        data = data.astype(float)

        # Append the data for this city to the full DataFrame
        full_df = pd.concat([full_df, data], axis=1)

    return full_df
