import pandas as pd
import numpy as np
import datetime as dt
import requests


def senti(asset):
    """
    1.Downloading the pickle with the news
    2.Creates 1min empty dataframe
    3.Enriching one min table with given news
    4.Forward fill & dropna
    5.Slices dataframe from the midnight of the next date from the start
    """
    all_news = pd.read_pickle(r"C:\Users\ilsbo\Downloads\news_dic.pkl")
    news_df = all_news[asset]
    news_ser = news_df['compound']
    news_ser = df3 = news_ser[~news_ser.index.duplicated(keep='last')]
    one_min_df = pd.DataFrame(index=pd.date_range(news_ser.index[0], news_ser.index[-1], freq='1min'))
    one_min_df[1] = np.empty(len(one_min_df.index)) * np.nan
    one_min_ser = one_min_df[1]
    for x in news_ser.index:
        one_min_ser.loc[x] = news_ser.loc[x]
    one_min_ser.fillna(method='ffill', inplace=True)
    one_min_ser.dropna(inplace=True)
    first_index_to_str = (one_min_ser.index[0] + dt.timedelta(1)).strftime("%Y-%m-%d")
    final_df = one_min_ser.loc[first_index_to_str:]
    return final_df


def download_btc_senti() -> pd.DataFrame:
    """
    Downloads Bitcoin sentiment scores from Senticrypt API.

    This function fetches sentiment data from the Senticrypt API, which provides
    real-time sentiment analysis based on Bitcoin-related social media data.
    The data includes multiple sentiment scores (score1, score2, score3, and mean).

    The function processes this data into a pandas DataFrame, renames the columns
    for better clarity, converts the index to Timestamps, and returns the selected sentiment columns.

    Source: https://senticrypt.com/docs.html

    :return: A pandas DataFrame with sentiment scores for Bitcoin.
             Columns returned:
             - 'Senti_BTC_1': First sentiment score
             - 'Senti_BTC_2': Second sentiment score
             - 'Senti_BTC_3': Third sentiment score
             - 'Senti_BTC_Mean': Mean sentiment score
    :rtype: pd.DataFrame
    :raises: requests.exceptions.RequestException if the API request fails.
    """
    api_url = 'https://api.senticrypt.com/v2/all.json'

    try:
        response = requests.get(api_url)
        response.raise_for_status()  # Raises HTTPError for bad responses
    except requests.exceptions.RequestException as e:
        raise SystemExit(f"Failed to fetch data from API: {e}")

    data = response.json()

    # Load data into a pandas DataFrame
    df = pd.DataFrame(data)

    # Set the 'date' column as the index and remove the name of the index
    df.set_index('date', drop=True, inplace=True)
    df.index.name = None

    # Convert index to Timestamps without specifying 'unit'
    df.index = pd.to_datetime(df.index)  # Assuming the date is in Unix timestamp format

    # Select relevant sentiment columns and rename them
    selected_df = df[['score1', 'score2', 'score3', 'mean']].copy()
    selected_df.rename(columns={'score1': 'Senti_BTC_1', 'score2': 'Senti_BTC_2',
                                'score3': 'Senti_BTC_3', 'mean': 'Senti_BTC_Mean'}, inplace=True)

    return selected_df + 1
