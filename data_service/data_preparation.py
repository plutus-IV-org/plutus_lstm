import pandas_ta as ta
import pandas as pd
import numpy as np
import requests
import datetime as dt
from data_service.vendors_data import yahoo_downloader as yd
from data_service.vendors_data import binance_downloader as bd
from data_service.vendors_data.av_downloader import AVDownloader
from data_service.vendors_data.macro_downloader import mc_downloader as mc
from data_service.vendors_data.sentiment_downloader import senti
from data_service.vendors_data.meteo_downloader import get_daily_meteo
from Const import CRYPTO_TICKERS, DAILY_CRYPTO_DATA_TABLE_LIST
from db_service.SQLite import technical_data_load
from Const import CUT_TAIL


class DataPreparation:
    def __init__(self, asset, df_type, source, interval, input_length, short=False):
        self.asset = asset
        self.type = df_type
        self.past = input_length
        self.source = source
        self.interval = interval
        self.is_short = short

    def _download_prices(self):
        if self.source == 'M' or self.source == 'Manual':
            import os
            file_name = self.asset + '_' + self.interval + '.csv'
            # Get the current working directory
            current_directory = os.getcwd()
            # Get the parent directory
            parent_directory = os.path.dirname(current_directory)
            file_directory = os.path.join(parent_directory, 'vaults', 'csv_source_vault', file_name)
            if os.path.exists(file_directory):
                print("The file exists.")
                df = pd.read_csv(file_directory)
                #TODO add pandas operations
            else:
                raise Exception("The file does not exist.")
        if self.source == 'Yahoo' or self.source == 'Y':
            df = yd.downloader(self.asset, self.interval)
        if self.source == 'Binance' or self.source == 'B':
            # if self.asset.split('_')[0] in CRYPTO_TICKERS and self.interval == '1d':
            #     table_index = CRYPTO_TICKERS.index(self.asset.split('_')[0])
            #     table_name = DAILY_CRYPTO_DATA_TABLE_LIST[table_index]
            #     print(f'For asset {self.asset} the daily data has been pulled from {table_name} ')
            #     df = technical_data_load(table_name)
            #     df['Time'] = df['Time'].astype('datetime64')
            #     df = df.set_index('Time')
            #     df = df.astype(float)
            #     df.dropna(inplace=True)
            #     if self.is_short:
            #         df = df.tail(CUT_TAIL)
            #     print(f'DB data has been cun by {CUT_TAIL} steps')
            # else:
            df = bd.downloader(self.asset, self.interval, self.is_short)
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

        def close_with_time(df, interval):
            # Assuming 'close(df)' is a predefined function
            close_df = close(df)

            # Extract the interval unit and number
            interval_unit = interval[-1]  # get the last character for unit (h, m, d, wk, M)
            interval_number = int(interval[:-1])  # get the number part of the interval

            # Define a dictionary for the interval unit conversion function
            conversion_function = {
                'h': lambda x: x.dt.hour + x.dt.minute / 60 + x.dt.second / 3600,
                'm': lambda x: x.dt.minute + x.dt.second / 60,
                'd': lambda x: x.dt.day,
                'w': lambda x: x.dt.isocalendar().week,
                'M': lambda x: x.dt.month
            }

            # Check if the interval unit is valid
            if interval_unit in conversion_function:
                # Apply the conversion function to the index
                time_component = conversion_function[interval_unit](close_df.index.to_series())
                # Scale based on the interval number and normalize
                close_df['timestamp_int'] = (time_component / (interval_number * 100)).astype(float)

            else:
                raise ValueError("Invalid interval unit. Please use 'h', 'm', 'd', 'wk', or 'mo'.")

            return close_df

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
            Source:
             1) https://github.com/Luchkata/Algorithmic_Trading_Machine_Learning/blob/main/Algorithmic_Trading_Machine_Learning_Quant_Strategies.ipynb

            Returns:
            pandas.DataFrame: DataFrame containing original data with added technical indicators.
            """

            # Convert input_length to integer if it's a string or a list containing a single value
            if type(input_length) == str:
                l = int(input_length)
            else:
                l = int(input_length[0])

            l_s = int(l / 2)  # Calculate the shorter window (l_s) for SMA SMALL
            l_ss = int(l / 4)
            l_sss = int(l / 8)

            # Copy the input DataFrame
            tech_dataset = df.copy()
            close_prices = tech_dataset[['Close', 'Open', 'High', 'Low', 'Volume']]
            close_prices.index.name = None
            df = close_prices.copy()
            df.dropna(inplace=True)
            df_tech_ind = df.copy()

            # Calculate and add SMA SMALL and SMA BIG
            df_tech_ind['SMA MICRO'] = df_tech_ind['Close'].rolling(window=l_sss).mean()
            df_tech_ind['SMA MINI'] = df_tech_ind['Close'].rolling(window=l_ss).mean()
            df_tech_ind['SMA SMALL'] = df_tech_ind['Close'].rolling(window=l_s).mean()
            df_tech_ind['SMA BIG'] = df_tech_ind['Close'].rolling(window=l).mean()

            # Calculate and add MACD
            df_tech_ind['MACD'] = df_tech_ind['SMA BIG'] / df_tech_ind['SMA SMALL']
            df_tech_ind['MACD_0.5'] = df_tech_ind['SMA SMALL'] / df_tech_ind['SMA MINI']
            df_tech_ind['MACD_0.25'] = df_tech_ind['SMA MINI'] / df_tech_ind['SMA MICRO']

            # Calculate and add RSI
            df_tech_ind['RSI'] = ta.rsi(df_tech_ind['Close'], length=l)
            df_tech_ind['RSI_0.5'] = ta.rsi(df_tech_ind['Close'], length=l_s)
            df_tech_ind['RSI_0.25'] = ta.rsi(df_tech_ind['Close'], length=l_ss)

            # Calculate and add ADX
            adx = ta.adx(df_tech_ind['High'], df_tech_ind['Low'], df_tech_ind['Close'], length=l)
            df_tech_ind = pd.concat([df_tech_ind, adx], axis=1)

            # Calculate and add AROON
            aroon = ta.aroon(df_tech_ind['High'], df_tech_ind['Low'], length=l) + 100
            df_tech_ind = pd.concat([df_tech_ind, aroon[aroon.columns[-1]]], axis=1)

            # Calculate German-Glass Volatility

            gks = ((np.log(df['High']) - np.log(df['Low'])) ** 2) / 2 - (2 * np.log(2) - 1) * (
                    (np.log(df['Close']) - np.log(df['Open'])) ** 2)
            gks_df = pd.DataFrame(gks, columns=['GKS'])

            if isinstance(input_length, list):
                il = input_length[0]
            else:
                il = input_length
            bb = ta.bbands(close=df['Close'], length=(int(il)))[
                [f'BBL_{str(il)}_2.0', f'BBM_{str(il)}_2.0', f'BBU_{str(il)}_2.0']]
            bb_halved = ta.bbands(close=df['Close'], length=int((int(il) / 2)))[
                [f'BBL_{str(int((int(il) / 2)))}_2.0', f'BBM_{str(int((int(il) / 2)))}_2.0',
                 f'BBU_{str(int((int(il) / 2)))}_2.0']]
            bb_quarter = ta.bbands(close=df['Close'], length=int((int(il) / 4)))[
                [f'BBL_{str(int((int(il) / 4)))}_2.0', f'BBM_{str(int((int(il) / 4)))}_2.0',
                 f'BBU_{str(int((int(il) / 4)))}_2.0']]
            atr = ta.atr(df['High'], df['Low'], df['Close'], length=int(il))
            atr_halved = ta.atr(df['High'], df['Low'], df['Close'], length=int(int(il) / 2))
            atr_quarter = ta.atr(df['High'], df['Low'], df['Close'], length=int(int(il) / 4))

            df_tech_ind = pd.concat([df_tech_ind, gks_df, bb, bb_halved, bb_quarter, atr, atr_halved, atr_quarter],
                                    axis=1)
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
                if len(df) < 2000 and self.is_short == False:
                    raise Exception(
                        'After merging macro table the input length became smaller than required minimum')
            else:
                raise Exception('For macro data use only 1d or 1wk interval.')
            return df

        def meteo():
            return get_daily_meteo()

        def crypto_benchmark():
            if self.interval == '1d':
                table_index = CRYPTO_TICKERS.index('BTC-USD')
                table_name = DAILY_CRYPTO_DATA_TABLE_LIST[table_index]
                print(f'For asset {self.asset} the daily data has been pulled from {table_name} ')
                df = technical_data_load(table_name)
                df['Time'] = df['Time'].astype('datetime64')
                df = df.set_index('Time')
                df = df.astype(float)
                df.dropna(inplace=True)
                if self.is_short:
                    df = df.tail(CUT_TAIL)
                print(f'DB data has been cun by {CUT_TAIL} steps')
            else:
                df = bd.downloader('BTC-USD', self.interval, self.is_short)
            return df

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
        else:
            df_dict = {}
            functions_to_call = {
                'close': close,
                'time': lambda df: close_with_time(df, self.interval),
                'volume': volume,
                'open': open,
                'technical': technical,
                'indicators': lambda df: indicators(df, self.past),
                'macro': lambda df: macro(df, self.interval),
                'senti': lambda: 100 + senti(self.asset),  # No 'df' passed here
                'crypto_sentiment_daily': crypto_sentiment_daily,
                'ranked_crypto_sentiment_daily': ranked_crypto_sentiment_daily,
                'crypto_benchmark': lambda: crypto_benchmark()[['Close']].rename(columns={'Close': 'BTC-USD Close'}),
            }

            # Iterate over the dictionary, try to call each function and store the result in df_dict
            for key, function in functions_to_call.items():
                try:
                    # For functions that don't take 'df' as an argument
                    if key in ['senti', 'crypto_sentiment_daily', 'ranked_crypto_sentiment_daily', 'crypto_benchmark']:
                        df_dict[key] = function()
                    else:  # For all other functions, pass 'df' as an argument
                        df_dict[key] = function(df)
                    print(f"Function {key} executed successfully.")
                except Exception as e:
                    print(f"An error occurred in function {key}: {e}")

            # If the asset is a cryptocurrency, merge additional data
            if self.asset[-4:] == '-USD':
                for key in ['crypto_sentiment_daily', 'ranked_crypto_sentiment_daily', 'crypto_benchmark']:
                    if key in df_dict:
                        df_dict['close'] = df_dict['close'].merge(df_dict[key], how='left', left_index=True,
                                                                  right_index=True)

            # Merge all dataframes from the dictionary that have been successfully created
            df_final = pd.DataFrame()
            for key, data in df_dict.items():
                if isinstance(data, pd.DataFrame):
                    df_final = df_final.join(data, how='outer', rsuffix=f'_{key}') if not df_final.empty else data
                else:
                    print(f"No DataFrame for key: {key}")

            # Remove duplicated columns after the join
            df = df_final.loc[:, ~df_final.columns.duplicated()]

        first_column = df.pop('Close')
        df.insert(0, 'Close', first_column)

        if self.type == 'Custom':
            df['Pct_Change/Volume'] = 10 - (df['Close'].pct_change() * 100 / np.log(df['Volume']))
        return df


if __name__ == '__main__':
    t1 = dt.datetime.now()
    df_long = DataPreparation('ETH-USD', 'Custom', 'B', '1d', 100)._download_prices()
    t2 = dt.datetime.now()
    delta_full = t2 - t1

    t1 = dt.datetime.now()
    df_short = DataPreparation('ETH-USD', 'Custom', 'B', '1d', 100, True)._download_prices()
    t2 = dt.datetime.now()
    delta_short = t2 - t1

