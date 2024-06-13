import re
import pandas as pd
from app_service.app_utilities import main_plot, main_plot_formatting
import plotly.graph_objects as go


def calculate_future_dates(start_date, num_days, interval, is_crypto):
    if interval[-1] == 'm':
        end_date = start_date + pd.Timedelta(minutes=(int(interval[:-1]) * num_days))
    elif interval[-1] == 'h':
        end_date = start_date + pd.Timedelta(hours=(int(interval[:-1]) * num_days))
    elif interval[-1] == 'd':
        if is_crypto:
            end_date = start_date + pd.Timedelta(days=(int(interval[:-1]) * num_days))
        else:
            end_date = start_date + pd.offsets.BDay(num_days)
    elif interval[-1] == 'w':
        end_date = start_date + pd.Timedelta(weeks=(int(interval[:-1]) * num_days))

    if interval[-1] in ['h', 'd']:
        return pd.date_range(start=start_date, end=end_date, freq=interval)
    elif interval[-1] == 'm':
        interval_for_pandas = interval[:-1] + 'min'
        return pd.date_range(start=start_date, end=end_date, freq=interval_for_pandas)
    else:  # For business days and weeks
        return pd.bdate_range(start=start_date, end=end_date, freq=interval)


def is_crypto_asset(name):
    return '-USD' in name


def get_interval_period(model_key):
    return model_key.split('_')[5]


def add_prediction(asset_name, interaction_data, hover_or_click, asset_prices, asset_predictions, anomalies_toggle,
                   anomalies_window,
                   rolling_period, zscore_lvl, candles_toggle,fig,single_asset_price):

    # Handle the case when no interaction data is available
    if interaction_data is None:
        pass

    else:
        # Your existing logic when interaction_data is available
        selected_day = pd.to_datetime(interaction_data['points'][0]['x'])
        is_crypto = is_crypto_asset(asset_name)

        for model_key, predictions in asset_predictions.items():
            if re.search(asset_name.split("_")[0], model_key) and re.search(asset_name.split("_")[-1], model_key):
                num_days = len(predictions.columns)
                interval = asset_name.split("_")[-1]
                future_dates = calculate_future_dates(selected_day, num_days, interval, is_crypto)

                try:
                    prediction_row = predictions.loc[selected_day]
                    prediction_for_plot = prediction_row.to_frame().set_index(future_dates[1:])
                    # Adding actual price as the first day of prediction
                    prediction_for_plot.loc[selected_day] = asset_prices[asset_name].loc[selected_day][0]
                    prediction_for_plot.sort_index(inplace=True)
                    if candles_toggle ==1:
                        single_asset_price = single_asset_price[['Close']]

                    df_plot = pd.merge(single_asset_price, prediction_for_plot.sort_index(), left_index=True,
                                       right_index=True)
                    df_plot.columns = [f'{asset_name}_actual', f'{asset_name}_predicted']

                    fig.add_trace(go.Scatter(
                        x=df_plot.index,
                        y=df_plot[f'{asset_name}_predicted'],
                        mode='lines',
                        name=model_key,
                        connectgaps=True,
                        hovertemplate='Date: %{x} <br>Price: %{y:$.4f}<extra></extra>'
                    ))
                except KeyError:
                    # Log the error or handle it as per the requirements
                    pass

    fig = main_plot_formatting(fig)
    return fig
