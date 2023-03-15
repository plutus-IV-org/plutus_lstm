import requests
import pandas as pd
import io
import time
from tqdm import tqdm

class AVDownloader:
    """
    A class used to download data from Alpha Vantage API

    ...

    Notes
    -----

    get_stock_intraday_data(ticker, interval="60min", number_of_month=1)
        returns dataframe with intraday prices of selected stock

    get_stock_daily
        returns dataframe with daily prices of selected stock

    get_fx_intraday_data
        returns dataframe with intraday prices of selected currency pair

    get_fx_daily
        returns dataframe with daily prices of selected currency pair

    get_crypto_daily
        returns dataframe with daily prices of selected digital currency pair

    get_ti
        returns dataframe with selected technical indicator

    """



    def __init__(self):
        """
        Parameters
        ----------
        api : str
            Alpha Vantage API
        min_accept_length : int
            minimal number of rows to be downloaded
        cut_length : int
            size of final dataframe
        max_rows_download : int
            maximum number of downloaded rows
        available_intervals : list
            list of intervals
        available_slices : list
            list of slices to be used in stock intraday data
            each slice = 30 days
        """

        self.api = "1L15VYNOL9JNN1NT"
        self.min_accept_length = 1500
        self.cut_length = 3000
        self.max_rows_download = 10000
        self.available_intervals = ["1m", "5m", "15m", "30m", "60m"]
        self.available_slices = ["year1month1", "year1month2", "year1month3", "year1month4", "year1month5",
                                 "year1month6", "year1month7", "year1month8", "year1month9", "year1month10",
                                 "year1month11", "year1month12"]

    def __raise_error(self):
        """Support private method that raises an error.
            As an argument method is using self.error_msg that is defined in each method
        """
        raise Exception(self.error_msg)

    def __check_intervals(self, interval):
        """Support private method that checks if provided interval name is correct.

        Raises
        ------
        NotImplementedError
            Provided interval is incorrect! Please use one of following:

        """
        if interval == "1m":
            interval = "1min"
            return interval
        elif interval == "5m":
            interval = "5min"
            return interval
        elif interval == "15m":
            interval = "15min"
            return interval
        elif interval == "30m":
            interval = "30min"
            return interval
        elif interval == "60m":
            interval = "60min"
            return interval
        # elif interval == "daily":
        #     pass
        # elif interval == "weekly":
        #     pass
        # elif interval == "monthly":
        #     pass
        else:
            self.error_msg = f"Provided interval is incorrect! Please use one of following: {' / '.join(self.available_intervals)} "
            self.__raise_error()

    def __check_num_of_rows(self, interval):
        """Support private method that checks a n umber of potential rows that will be downloaded.

            As parameters it takes interval name and number of months in range.
            If interval name is not correct it will raise an error

        Returns
        -------
        int
            number of rows that will be downloaded

        Raises
        ------
        NotImplementedError
            Provided interval is incorrect! Please use one of following:

        """
        if interval == "1m":
            number_of_month = 1 # returns 15300 rows
            return round(number_of_month)
        elif interval == "5m":
            number_of_month = 1 # returns 3600 rows
            return round(number_of_month)
        elif interval == "15m":
            number_of_month = self.cut_length / 1220
            return round(number_of_month) + 1
        elif interval == "30m":
            number_of_month = self.cut_length / 610
            return round(number_of_month) + 1
        elif interval == "60m":
            number_of_month = self.cut_length / 305
            return round(number_of_month) + 1
        else:
            self.error_msg = f"Provided interval for STOCK data type is incorrect! Please use one of following: {' / '.join(self.available_intervals)}"
            self.__raise_error()

    def __trim_data(self, raw_data):
        """Support private method that checks if number of rows and trims dataframe.

            As parameters it takes a raw dataframe, checks if number of rows is higher than self.min_accept_length
            and if true, proceeds by trimming a dataframe to lenght which is equal to self.cut_length.

            After it modifies column names to ['Open', 'High', 'Low', 'Close', 'Volume']
            If there is no Volume, it will ad empty column.

            !!!!!Currently works only with STOCK and CURRENCY

            Will raise 'No enough data!' if number of rows is below self.min_accept_length

        Returns
        -------
        pandas.DataFrame
            processed dataframe

        Raises
        ------
        NotImplementedError
            'No enough data!'

        """
        raw_data.dropna(inplace=True)
        print(f'Initial length of dowloaded data is {len(raw_data)}')
        if len(raw_data) < self.min_accept_length:
            self.error_msg = 'No enough data!'
            self.__raise_error()
        if len(raw_data) > self.cut_length:
            raw_data = raw_data.head(self.cut_length)
        print(f'Output length is {len(raw_data)}')
        raw_data = raw_data.iloc[::-1]
        return raw_data


    def __split_asset_name(self,asset):
        if len(asset) == 8:
            from_symbol = asset[0:3]
            to_symbol = asset[3:6]
            return from_symbol, to_symbol
        elif len(asset) == 5:
            from_symbol = "USD"
            to_symbol =  asset[0:3]
            return from_symbol, to_symbol
        else:
            raise Exception('Provided currency pairnames are incorrect. Please use format as: GBPUSD=X or EUR=X')


    def __reformat_stock_daily(self, processed_data):
        """Support private method that reformats dataframe.

        Returns
        -------
        pandas.DataFrame
            final dataframe
        """
        processed_data['timestamp'] = pd.DatetimeIndex(processed_data['timestamp'])
        processed_data = processed_data.set_index('timestamp')
        processed_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return processed_data

    def __reformat_stock_intraday(self, processed_data):
        """Support private method that reformats dataframe.

        Returns
        -------
        pandas.DataFrame
            final dataframe
        """
        processed_data['time'] = pd.DatetimeIndex(processed_data['time'])
        processed_data = processed_data.set_index('time')
        processed_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return processed_data

    def __reformat_fx_daily(self, processed_data):
        """Support private method that reformats dataframe.

        Returns
        -------
        pandas.DataFrame
            final dataframe
        """
        processed_data['timestamp'] = pd.DatetimeIndex(processed_data['timestamp'])
        processed_data = processed_data.set_index('timestamp')
        processed_data['Volume'] = 0
        processed_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return processed_data

    def __reformat_fx_intraday(self, processed_data):
        """Support private method that reformats dataframe.

        Returns
        -------
        pandas.DataFrame
            final dataframe
        """
        processed_data['timestamp'] = pd.DatetimeIndex(processed_data['timestamp'])
        processed_data = processed_data.set_index('timestamp')
        processed_data['Volume'] = 0
        processed_data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        return processed_data

    def get_stock_intraday_data(self, ticker, interval="60min"):
        """Method that returns historical intraday time series for the trailing 2 years,
        covering over 2 million data points per ticker

        Parameters
        ----------

        ticker : str
            The name of the equity of your choice. For example: symbol=IBM
        interval : str
            Time interval between two consecutive data points in the time series. The following values are supported:
            1min, 5min, 15min, 30min, 60min
        number_of_month : int

            This parameter is taking a number of the months of historical intraday data.

            !!! IMPORTANT NOTE:
            Actually this number defines how many slices we want to take into account. Each slice is a
            30-day window, with year1month1 being the most recent and year2month12 being the farthest from today.
            By default, slice=year1month1.

            To make it work with 2 or more months I had to implement for loop statement. Due to BASIC SUBSCRIPTION,
            there is a limitation of 5 requests a minute. IT IS POSSIBLE TO RUN THIS METHOD FOR 5 MONTH ONLY ONCE
            IN 1 MINUTE. Which may be not enough for analysis.

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        NotImplementedError
            'The number of rows you are trying to download exceeds value of max number o rows to download: '
        """

        number_of_month = self.__check_num_of_rows(interval)
        interval = self.__check_intervals(interval)

        empty_raw_df = pd.DataFrame()

        number_of_requests = 0

        for i in range(0,int(number_of_month)):

            if number_of_requests == 4:
                print("Limit of requests per minute was reached. Please wait 70 seconds.")
                # time.sleep(70)
                for k in tqdm(range(70)):
                    time.sleep(1)
                number_of_requests = 0


            url = "https://alpha-vantage.p.rapidapi.com/query"
            querystring = {"symbol": ticker,
                           "function": "TIME_SERIES_INTRADAY_EXTENDED",
                           "interval": interval,
                           "slice": self.available_slices[i],
                           "datatype": "csv"}

            headers = {
                "X-RapidAPI-Key": "5d05c549b4msh460a7b093e3c66ap175e4ajsne8304cd1be56",
                "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
            }
            response = requests.request("GET", url, headers=headers, params=querystring)
            r = response.content
            raw_data = pd.read_csv(io.StringIO(r.decode('utf-8')))
            empty_raw_df = pd.concat([empty_raw_df, raw_data])
            number_of_requests = number_of_requests + 1

        processed_data = self.__trim_data(empty_raw_df)
        final_data = self.__reformat_stock_intraday(processed_data)

        return final_data

    def get_stock_daily(self, ticker):
        """Method that returns raw (as-traded) daily time series (date, daily open, daily high, daily low, daily close,
        daily volume) of the global equity specified, covering 20+ years of historical data.

        Parameters
        ----------

        ticker : str
            The name of the equity of your choice. For example: symbol=IBM

        Returns
        -------
        pandas.DataFrame
        """
        url = "https://alpha-vantage.p.rapidapi.com/query"
        querystring = {"symbol": ticker,
                       "function": "TIME_SERIES_DAILY",
                       "outputsize": "full",
                       "datatype": "csv"}

        headers = {
            "X-RapidAPI-Key": "5d05c549b4msh460a7b093e3c66ap175e4ajsne8304cd1be56",
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }
        response = requests.request("GET", url, headers=headers, params=querystring)
        r = response.content
        raw_data = pd.read_csv(io.StringIO(r.decode('utf-8')))

        processed_data = self.__trim_data(raw_data)
        final_data = self.__reformat_stock_daily(processed_data)

        return final_data

    def get_fx_intraday_data(self, from_symbol, to_symbol, interval="60min"):
        """Method that returns intraday time series (timestamp, open, high, low, close)
        of the FX currency pair specified, updated realtime.

        Parameters
        ----------

        from_symbol : str
            A three-letter symbol from the forex currency list. For example: from_symbol=EUR
        to_symbol : str
            A three-letter symbol from the forex currency list. For example: to_symbol=USD
        interval : str
            Time interval between two consecutive data points in the time series. The following values are
            supported: 1min, 5min, 15min, 30min, 60min

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        NotImplementedError
            Provided interval is incorrect! Please use one of following:
        """
        self.__check_intervals(interval)

        url = "https://alpha-vantage.p.rapidapi.com/query"
        querystring = {"from_symbol": from_symbol,
                       "to_symbol": to_symbol,
                       "function": "FX_INTRADAY",
                       "interval": interval,
                       "outputsize": "full",
                       "datatype": "csv"}

        headers = {
            "X-RapidAPI-Key": "5d05c549b4msh460a7b093e3c66ap175e4ajsne8304cd1be56",
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }
        response = requests.request("GET", url, headers=headers, params=querystring)
        r = response.content
        raw_data = pd.read_csv(io.StringIO(r.decode('utf-8')))
        processed_data = self.__trim_data(raw_data)
        final_data = self.__reformat_fx_intraday(processed_data)

        return final_data

    def get_fx_daily(self, asset):
        """Method that returns the daily time series (timestamp, open, high, low, close) of the FX currency pair
        specified, updated realtime.

        Parameters
        ----------

        from_symbol : str
            A three-letter symbol from the forex currency list. For example: from_symbol=EUR
        to_symbol : str
            A three-letter symbol from the forex currency list. For example: to_symbol=USD

        Returns
        -------
        pandas.DataFrame
        """


        from_symbol, to_symbol = self.__split_asset_name(asset)



        url = "https://alpha-vantage.p.rapidapi.com/query"
        querystring = {"from_symbol": from_symbol,
                       "to_symbol": to_symbol,
                       "function": "FX_DAILY",
                       "outputsize": "full",
                       "datatype": "csv"}

        headers = {
            "X-RapidAPI-Key": "5d05c549b4msh460a7b093e3c66ap175e4ajsne8304cd1be56",
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }
        response = requests.request("GET", url, headers=headers, params=querystring)
        r = response.content
        raw_data = pd.read_csv(io.StringIO(r.decode('utf-8')))
        processed_data = self.__trim_data(raw_data)
        final_data = self.__reformat_fx_daily(processed_data)

        return final_data

    def get_crypto_intraday(self, symbol, interval, currency="USD"):
        """Method that returns intraday time series (timestamp, open, high, low, close, volume)
        of the cryptocurrency specified, updated realtime.

        Parameters
        ----------

        symbol : str
            The currency you would like to get the exchange rate for. It can either be a physical currency or
            digital/cryptocurrency. For example: from_currency=USD or from_currency=BTC.
        currency : str
            The destination currency for the exchange rate. It can either be a physical currency or
            digital/cryptocurrency. For example: to_currency=USD or to_currency=BTC.
        interval : str
            Time interval between two consecutive data points in the time series.
            The following values are supported: 1min, 5min, 15min, 30min, 60min

        Returns
        -------
        pandas.DataFrame
        """
        url = "https://alpha-vantage.p.rapidapi.com/query"
        querystring = {"symbol": symbol,
                       "market": currency,
                       "function": "CRYPTO_INTRADAY",
                       "interval": interval,
                       "outputsize": "full",
                       "datatype": "csv"}

        headers = {
            "X-RapidAPI-Key": "5d05c549b4msh460a7b093e3c66ap175e4ajsne8304cd1be56",
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }
        response = requests.request("GET", url, headers=headers, params=querystring)
        r = response.content
        raw_data = pd.read_csv(io.StringIO(r.decode('utf-8')))
        return self.__trim_data(raw_data)

    def get_ti(self, symbol, function, series_type, interval, time_period):
        """Method that downloads selected technical indicator for selected symbol Technical indicator APIs for a
        given equity or currency exchange pair, derived from the underlying time series based stock API and forex
        data. All indicators are calculated from adjusted time series data to eliminate artificial price/volume
        perturbations from historical split and dividend events.

        Parameters
        ----------

        symbol : str
            The name of the token of your choice. For example: symbol=IBM
        function : str
            The technical indicator of your choice. In this case, function=SMA
        series_type : str
            The desired price type in the time series. Four types are supported: close, open, high, low
        interval : str
            Time interval between two consecutive data points in the time series. The following values are supported: 1min, 5min, 15min, 30min, 60min, daily, weekly, monthly
        time_period : int
            Number of data points used to calculate each moving average value. Positive integers are accepted (e.g., time_period=60, time_period=200)

        Returns
        -------
        pandas.DataFrame

        Raises
        ------
        NotImplementedError
            Provided interval is incorrect! Please use one of following:
        """
        self.__check_intervals(interval)

        url = "https://alpha-vantage.p.rapidapi.com/query"
        querystring = {"symbol": symbol,
                       "function": function,
                       "series_type": series_type,
                       "interval": interval,
                       "time_period": time_period,
                       "datatype": "csv"}

        headers = {
            "X-RapidAPI-Key": "5d05c549b4msh460a7b093e3c66ap175e4ajsne8304cd1be56",
            "X-RapidAPI-Host": "alpha-vantage.p.rapidapi.com"
        }
        response = requests.request("GET", url, headers=headers, params=querystring)
        r = response.content
        raw_data = pd.read_csv(io.StringIO(r.decode('utf-8')))
        return self.__trim_data(raw_data)