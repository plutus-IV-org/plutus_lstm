import pandas as pd


def apply_means(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates means and ratios such as  current/week, current/quarter etc.
    """
    for col in df.columns:
        week_name = 'Weekly_' + col
        quarter_name = 'Quarterly_' + col

        df[week_name] = df[col].rolling(7).mean()
        df[quarter_name] = df[col].rolling(90).mean()

        current_week_ratio = col + "/" + week_name
        current_quarter_ratio = col + "/" + quarter_name
        week_quarter_ratio = week_name + "/" + quarter_name

        df[current_week_ratio] = df[col] / df[week_name]
        df[current_quarter_ratio] = df[col] / df[quarter_name]
        df[week_quarter_ratio] = df[week_name] / df[quarter_name]

    return df
