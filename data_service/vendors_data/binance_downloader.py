from binance.exceptions import BinanceAPIException
from time import sleep
from binance import Client, ThreadedWebsocketManager, ThreadedDepthCacheManager

api_key = "TJFbxUadj3OrleUMNU9VccxmAseDncRoq8ZPxZxBrNjgeYt8AKOQq75GKio6JJ8z"
api_secret = "VRE0XkDH1hkszXdm5ntYLI19GQE6Eu2dSwRNkqoZZY0sreWBEJLGZGGZWY9RCFRi"
client = Client(api_key, api_secret)
import pandas as pd
import datetime as dt


def downloader(asset, interval):
    # valid intervals: 1m,5m,15m,30m,1h,1d,5d,1wk,1mo
    try:
        ticker, cur = asset.split('-')
    except:
        raise ValueError('Binance downloader has received non crypto ticker, please amend init file!')
    symbol = ticker + 'USDT'

    if interval == '1m':
        now = dt.datetime.now().strftime("%d %B, %Y %H:%M:%S")
        back = (dt.datetime.now() - dt.timedelta(5)).strftime("%d %B, %Y")
        try:
            df = pd.DataFrame(
                client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, back, now))
        except BinanceAPIException as e:
            print(e)
            sleep(5)
            df = pd.DataFrame(
                client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MINUTE, back, now))
    elif interval == '5m':
        now = dt.datetime.now().strftime("%d %B, %Y %H:%M:%S")
        back = (dt.datetime.now() - dt.timedelta(20)).strftime("%d %B, %Y")
        try:
            df = pd.DataFrame(
                client.get_historical_klines(symbol, Client.KLINE_INTERVAL_5MINUTE, back, now))
        except BinanceAPIException as e:
            print(e)
            sleep(5)
            df = pd.DataFrame(
                client.get_historical_klines(symbol, Client.KLINE_INTERVAL_5MINUTE, back, now))
    elif interval == '15m':
        now = dt.datetime.now().strftime("%d %B, %Y %H:%M:%S")
        back = (dt.datetime.now() - dt.timedelta(60)).strftime("%d %B, %Y")
        try:
            df = pd.DataFrame(
                client.get_historical_klines(symbol, Client.KLINE_INTERVAL_15MINUTE, back, now))
        except BinanceAPIException as e:
            print(e)
            sleep(5)
            df = pd.DataFrame(
                client.get_historical_klines(symbol, Client.KLINE_INTERVAL_15MINUTE, back, now))
    elif interval == '30m':
        now = dt.datetime.now().strftime("%d %B, %Y %H:%M:%S")
        back = (dt.datetime.now() - dt.timedelta(120)).strftime("%d %B, %Y")
        try:
            df = pd.DataFrame(
                client.get_historical_klines(symbol, Client.KLINE_INTERVAL_30MINUTE, back, now))
        except BinanceAPIException as e:
            print(e)
            sleep(5)
            df = pd.DataFrame(
                client.get_historical_klines(symbol, Client.KLINE_INTERVAL_30MINUTE, back, now))
    elif interval == '1h':
        now = dt.datetime.now().strftime("%d %B, %Y %H:%M:%S")
        back = (dt.datetime.now() - dt.timedelta(240)).strftime("%d %B, %Y")
        try:
            df = pd.DataFrame(
                client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, back, now))
        except BinanceAPIException as e:
            print(e)
            sleep(5)
            df = pd.DataFrame(
                client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1HOUR, back, now))
    elif interval == '1d':
        now = dt.datetime.now().strftime("%d %B, %Y %H:%M:%S")
        back = (dt.datetime.now() - dt.timedelta(10000)).strftime("%d %B, %Y")
        try:
            df = pd.DataFrame(
                client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, back, now))
        except BinanceAPIException as e:
            print(e)
            sleep(5)
            df = pd.DataFrame(
                client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1DAY, back, now))
    elif interval == '1wk':
        now = dt.datetime.now().strftime("%d %B, %Y %H:%M:%S")
        back = (dt.datetime.now() - dt.timedelta(50000)).strftime("%d %B, %Y")
        try:
            df = pd.DataFrame(
                client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1WEEK, back, now))
        except BinanceAPIException as e:
            print(e)
            sleep(5)
            df = pd.DataFrame(
                client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1WEEK, back, now))
    elif interval == '1mo':
        now = dt.datetime.now().strftime("%d %B, %Y %H:%M:%S")
        back = (dt.datetime.now() - dt.timedelta(200000)).strftime("%d %B, %Y")
        try:
            df = pd.DataFrame(
                client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MONTH, back, now))
        except BinanceAPIException as e:
            print(e)
            sleep(5)
            df = pd.DataFrame(
                client.get_historical_klines(symbol, Client.KLINE_INTERVAL_1MONTH, back, now))
    else:
        raise Exception('Provided interval doesnt exist in Binance API.')

    df = df.iloc[:, :6]
    df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    df = df.set_index('Time')
    df.index = pd.to_datetime(df.index, unit='ms')
    #df.index = df.index.tz_localize('UTC').tz_convert('CET').tz_localize(None)
    df = df.astype(float)
    df.dropna(inplace=True)
    raw_data = df.copy()
    print(f'Initial length of downloaded data is {len(raw_data)}')
    min_accept_length = 2000
    cut_length = 3000
    if len(raw_data) < min_accept_length:
        raise ValueError('No enough data!')
    # if len(raw_data) > cut_length:
    #     raw_data = raw_data.tail(cut_length)
    print(f'Output length is {len(raw_data)}')
    return raw_data
