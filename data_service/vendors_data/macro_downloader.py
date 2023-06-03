import pandas_datareader as pdr
import pandas as pd
from datetime import date


def mc_downloader():
    FRED_INDICATORS = ['GDP', 'CPIAUCSL', 'DFF', 'T10YIE', 'UNRATE', 'ICSA', 'PCEDG', 'INDPRO', 'DCOILWTICO', 'GFDEBTN',
                       'GVZCLS', 'VIXCLS', 'MORTGAGE30US', 'SP500', "PCU221122221122"]
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
