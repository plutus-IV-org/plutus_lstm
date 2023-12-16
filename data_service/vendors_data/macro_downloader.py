import pandas_datareader as pdr
import pandas as pd
from datetime import date


def mc_downloader()-> dict:
    """
    The data is downloaded in CT (CST) time zone which usually a difference between 6-7 hours.
    That implies that theoretically your data is being tracked later than it was published. Why theoretically?
    Imagine a case where GDP data is published on 01.01.24 at 6 pm CT.
    This means that current time is smth like 1 am CET. The data index will be 01.01 however in reality its 02.01

    In my personal opinion, you dont need to spend to much time trying to align time zones for FRED,
    unless the data you want to collect is published daily or hourly.

    DFF -  Federal Funds Effective Rate (DAILY!)
    T10YIE -  10-Year Breakeven Inflation Rate (DAILY!)
    UNRATE - Unemployment Rate
    ICSA - Initial Claims
    PCEDG - Personal Consumption Expenditures: Durable Goods
    INDPRO - Industrial Production: Total Index
    DCOILWTICO - Crude Oil Prices: West Texas Intermediate (WTI) (DAILY!)
    GFDEBTN - Federal Debt (Public Debt)
    GVZCLS - CBOE Gold ETF Volatility Index (DAILY!)
    VIXCLS -  CBOE Volatility Index: VIX (DAILY!)
    MORTGAGE30US - 30-Year Fixed Rate Mortgage Average in the United States
    PCU221122221122 - Producer Price Index by Industry: Electric Power Distribution
    FDHBFRBN - Federal Debt Held by Federal Reserve Banks (FDHBFRBN)
    CSUSHPINSA- house holding prices
    :return:
        Macro dict
    """
    FRED_INDICATORS = ['GDP', 'CPIAUCSL', 'DFF', 'T10YIE', 'UNRATE', 'ICSA', 'PCEDG', 'INDPRO', 'DCOILWTICO', 'GFDEBTN',
                       'GVZCLS', 'VIXCLS', 'MORTGAGE30US', 'SP500', "PCU221122221122", 'CSUSHPINSA', "FDHBFRBN"]
    end = date.today()
    start = date(year=end.year - 35, month=end.month, day=end.day)
    macro_indicators = dict()
    tq_fred = FRED_INDICATORS
    for indicator in tq_fred:
        macro_indicators[indicator] = pdr.fred.FredReader(indicator, start=start, timeout=90).read()

    for ind in macro_indicators.keys():
        data_range = pd.date_range(macro_indicators[ind].index[0], date.today())
        df = pd.DataFrame(index=data_range)
        full_df = pd.concat([df, macro_indicators[ind]], axis=1).fillna(method='ffill').dropna()
        macro_indicators[ind] = full_df + 100
    return macro_indicators
