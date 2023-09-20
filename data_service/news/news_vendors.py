import pandas as pd
from newsdataapi import NewsDataApiClient
from newsapi import NewsApiClient


def get_news_from_newsdata(name: str, language: str = 'en') -> pd.DataFrame:
    """
    WEB-PAGE: https://newsdata.io/search-news
    :param name: str Alias
    :param language: str Language
    :return: pd.DataFrame Imported news in df format.
    """
    api = NewsDataApiClient(apikey='pub_29707e0d8f9860030c42a95f47c7995691910')
    response = api.news_api(q=name, language=language)
    df = pd.DataFrame(response['results'])[['title', 'link', 'pubDate']].rename(
        columns={'title': 'Title', 'link': 'URL', 'pubDate': 'Time'})

    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)
    df.sort_index(ascending=False, inplace=True)
    return df.head(10)


def get_news_from_newsapi(name: str) -> pd.DataFrame:
    """
    WEB-PAGE: https://newsapi.org/
    :param name: str Alias
    :return: pd.DataFrame Imported news in df format.
    """
    api = NewsApiClient(api_key='bbeed1152068400cb92515a4b2b064c1')
    response = api.get_everything(q=name)
    df = pd.DataFrame(response['articles'])[['title', 'url', 'publishedAt']].rename(
        columns={'title': 'Title', 'url': 'URL', 'publishedAt': 'Time'})
    df['Time'] = pd.to_datetime(df['Time']).dt.tz_convert(None)
    df.set_index('Time', inplace=True)
    df.sort_index(ascending=False, inplace=True)
    return df.head(10)
