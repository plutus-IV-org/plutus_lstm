import pandas as pd


def apply_target_to_close_price(df: pd.DataFrame, future: int) -> pd.DataFrame:
    """
    Applies target labels to the close price data in a DataFrame.
    :param df: A DataFrame containing the close price data.
    :return: The DataFrame with added target labels.
    """
    df['Tomorrow'] = df['Close'].shift(-1)
    df['Target'] = df['Tomorrow'] - df['Close']
    df['Target'][df['Target'] > 0] = 1
    df['Target'][df['Target'] <= 0] = 0
    df.drop(columns=['Tomorrow'], inplace=True)
    return df
