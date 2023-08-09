import pandas_ta as ta
import pandas as pd
import requests
from data_service.vendors_data import yahoo_downloader as yd
from data_service.vendors_data import binance_downloader as bd
from data_service.vendors_data.av_downloader import AVDownloader
from data_service.vendors_data.macro_downloader import mc_downloader as mc
from data_service.vendors_data.sentiment_downloader import senti
from data_service.vendors_data.meteo_downloader import get_daily_meteo


class DataPreparation:
    def __init__(self, asset, df_type, source, interval, input_length):
        self.asset = asset
        self.type = df_type
        self.past = input_length
        self.source = source
        self.interval = interval

    def _download_prices(self):
        if self.source == 'Yahoo' or self.source == 'Y':
            df = yd.downloader(self.asset, self.interval)
        if self.source == 'Binance' or self.source == 'B':
            df = bd.downloader(self.asset, self.interval)
        if self.source == 'AV' or self.source == 'A':
            data = AVDownloader()

            if self.asset[-2:] == "=X":
                asset_type = "Currency"
            else:
                asset_type = "Stock"

            if self.interval == "1d":
                raw_data_type = "Daily"
            else:
                raw_data_type = "Intraday"

            if asset_type == "Stock":
                if raw_data_type == "Daily":
                    df = data.get_stock_daily(self.asset)
                elif raw_data_type == "Intraday":
                    df = data.get_stock_intraday_data(self.asset, self.interval)
                else:
                    raise Exception('Please use "Daily" or "Intraday" data types for Stock asset type.')
            elif asset_type == "Currency":
                if raw_data_type == "Daily":
                    df = data.get_fx_daily(self.asset)
                else:
                    raise Exception('Please use only "Daily" data types for Currency asset type.')
            else:
                raise Exception('Please use one of the following asset types: Stock / Currency.')

        def close(df):
            tech_dataset = df.copy()
            close_prices = tech_dataset[['Close']]
            close_prices.index.name = None
            df = close_prices.copy()
            df.dropna(inplace=True)
            return df

        def volume(df):
            tech_dataset = df.copy()
            close_prices = tech_dataset[['Close', 'Volume']]
            close_prices.index.name = None
            df = close_prices.copy()
            df.dropna(inplace=True)
            return df

        def open(df):
            tech_dataset = df.copy()
            close_prices = tech_dataset[['Close', 'Open']]
            close_prices.index.name = None
            df = close_prices.copy()
            df.dropna(inplace=True)
            return df

        def technical(df):
            tech_dataset = df.copy()
            close_prices = tech_dataset[['Close', 'Open', 'High', 'Low', 'Volume']]
            close_prices.index.name = None
            df = close_prices.copy()
            df.dropna(inplace=True)
            return df

        def crypto_sentiment_daily():
            r = requests.get('https://api.alternative.me/fng/?limit=0')
            df = pd.DataFrame(r.json()['data'])
            df.value = df.value.astype(int)
            df.timestamp = pd.to_datetime(df.timestamp, unit='s')
            df.set_index('timestamp', inplace=True)
            df = df[::-1][['value']]
            df.rename(columns={'value': 'Crypto_Sentiment daily'}, inplace=True)
            return df

        def ranked_crypto_sentiment_daily():
            r = requests.get('https://api.alternative.me/fng/?limit=0')
            df = pd.DataFrame(r.json()['data'])
            df.value = df.value.astype(int)
            df.timestamp = pd.to_datetime(df.timestamp, unit='s')
            df.set_index('timestamp', inplace=True)
            df = df[::-1][['value']]
            df.rename(columns={'value': 'Crypto_Sentiment daily - ranked'}, inplace=True)
            df = df.where(df <= 50, 2)
            df = df.where(df == 2, 1)
            return df

        def indicators(df, input_length):
            """
            Calculate and add technical indicators to a given DataFrame.

            Parameters:
            df (pandas.DataFrame): Input DataFrame containing columns 'Close', 'Open', 'High', 'Low', and 'Volume'.
            input_length (int, str or list): Length used for calculating technical indicators. If a string or a list containing a single value is provided, the function will convert it to an integer.

            Returns:
            pandas.DataFrame: DataFrame containing original data with added technical indicators.
            """

            # Convert input_length to integer if it's a string or a list containing a single value
            if type(input_length) == str:
                l = int(input_length)
            else:
                l = int(input_length[0])

            l_s = int(l / 2)  # Calculate the shorter window (l_s) for SMA SMALL

            # Copy the input DataFrame
            tech_dataset = df.copy()
            close_prices = tech_dataset[['Close', 'Open', 'High', 'Low', 'Volume']]
            close_prices.index.name = None
            df = close_prices.copy()
            df.dropna(inplace=True)
            df_tech_ind = df.copy()

            # Calculate and add SMA SMALL and SMA BIG
            df_tech_ind['SMA SMALL'] = df_tech_ind['Close'].rolling(window=l_s).mean()
            df_tech_ind['SMA BIG'] = df_tech_ind['Close'].rolling(window=l).mean()

            # Calculate and add MACD
            df_tech_ind['MACD'] = df_tech_ind['SMA BIG'] / df_tech_ind['SMA SMALL']

            # Calculate and add RSI
            df_tech_ind['RSI'] = ta.rsi(df_tech_ind['Close'], length=l)

            # Calculate and add ADX
            adx = ta.adx(df_tech_ind['High'], df_tech_ind['Low'], df_tech_ind['Close'], length=l)
            df_tech_ind = pd.concat([df_tech_ind, adx], axis=1)

            # Calculate and add AROON
            aroon = ta.aroon(df_tech_ind['High'], df_tech_ind['Low'], length=l) + 100
            df_tech_ind = pd.concat([df_tech_ind, aroon[aroon.columns[-1]]], axis=1)

            # Remove rows with missing values
            df_tech_ind.dropna(inplace=True)

            return df_tech_ind

        def macro(df, interval):
            if interval in ['1d', '1wk']:
                tech_dataset = df.copy()
                close_prices = tech_dataset[['Close']]
                close_prices.index.name = None
                df = close_prices.copy()
                df.dropna(inplace=True)
                mc_tables = mc()
                for x in mc_tables.keys():
                    df = pd.concat([df, mc_tables[x]], axis=1)
                df.dropna(inplace=True)
                if len(df) < 2000:
                    raise Exception(
                        'After merging macro table the input length became smaller than required minimum')
            else:
                raise Exception('For macro data use only 1d or 1wk interval.')
            return df

        def meteo():
            return get_daily_meteo()

        def crypto_benchmark():
            return bd.downloader('BTC-USD', self.interval)

        def fundamentals(asset):
            """"
            bps = pd.read_pickle('bps800').loc['2011-01-01':, asset]
            eps = pd.read_pickle('eps800').loc['2011-01-01':, asset]
            dps = pd.read_pickle('dps800').loc['2011-01-01':, asset]
            cps = pd.read_pickle('cps800').loc['2011-01-01':, asset]
            tech_dataset = yf.download(asset, start='2011-01-01', end=dt.datetime.now(), progress=False)
            close_prices = tech_dataset[['Close']]
            close_prices.index.name = None
            df = close_prices.copy()
            df.dropna(inplace=True)
            con = pd.concat([df,bps,eps,dps,cps], axis=1)
            return con
            """
            pass

        if self.type == 'Close':
            df = close(df)
        elif self.type == 'Volume':
            df = volume(df)
        elif self.type == 'Open':
            df = open(df)
        elif self.type == 'Technical':
            df = technical(df)
        elif self.type == 'Indicators':
            df = indicators(df, self.past)
        elif self.type == 'Macro':
            df = macro(df, self.interval)
        elif self.type == 'Sentiment':
            df_close = close(df).tz_localize(None)
            df_senti = 100 + senti(self.asset).tz_localize(None)
            df = pd.concat([df_close, df_senti], axis=1).dropna().copy()
        elif self.type == 'Fundamentals':
            df = fundamentals(df)
        else:
            q1 = close(df)
            q2 = volume(df)
            q3 = open(df)
            q4 = technical(df)
            q5 = indicators(df, self.past)
            q6 = macro(df, self.interval)

            try:
                q7 = 100 + senti(self.asset)
            except:
                q7 = pd.DataFrame()
                print('Sentiment dictionary doesnt contain the asset')
            if self.asset[-4:] == '-USD':
                q8 = crypto_sentiment_daily()
                q9 = ranked_crypto_sentiment_daily()
                q10 = crypto_benchmark()[['Close']].rename(columns={'Close' : 'BTC-USD Close'})
                q1 = q1.merge(q8, how='left', left_index=True, right_index=True)
                q1 = q1.merge(q9, how='left', left_index=True, right_index=True)
                q1 = q1.merge(q10, how='left', left_index=True, right_index=True)
            q11 = meteo()
            q1 = q1.merge(q11, how='left', left_index=True, right_index=True)

            df = pd.concat([q1, q2, q3, q4, q5, q6, q7], axis=1)
            df = df.loc[:, ~df.columns.duplicated()]

        first_column = df.pop('Close')
        df.insert(0, 'Close', first_column)

        return df
