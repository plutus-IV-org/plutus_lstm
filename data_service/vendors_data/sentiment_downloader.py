import pandas as pd
import numpy as np
import datetime as dt

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
    news_ser= df3 = news_ser[~news_ser.index.duplicated(keep='last')]
    one_min_df=pd.DataFrame(index = pd.date_range(news_ser.index[0], news_ser.index[-1], freq = '1min'))
    one_min_df[1] = np.empty(len(one_min_df.index)) * np.nan
    one_min_ser=one_min_df[1]
    for x in news_ser.index:
        one_min_ser.loc[x] = news_ser.loc[x]
    one_min_ser.fillna(method = 'ffill' , inplace = True)
    one_min_ser.dropna(inplace =True)
    first_index_to_str = (one_min_ser.index[0] + dt.timedelta(1)).strftime("%Y-%m-%d")
    final_df = one_min_ser.loc[first_index_to_str:]
    return final_df

