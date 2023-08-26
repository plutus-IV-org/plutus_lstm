import plotly.graph_objects as go
from dash import Dash, html, dcc, Input, Output
import dash_bootstrap_components as dbc
from jupyter_dash import JupyterDash
import pandas as pd
import re
import plotly.subplots as sp
from pandas.tseries.offsets import BDay
from app_service.input_data_maker import generate_data
import numpy as np
import datetime as dt
from utilities.directional_accuracy_services import save_directional_accuracy_score
from flask import Flask
import logging

app = Flask(__name__)

log = logging.getLogger('werkzeug')
log.disabled = True
# other application initialization logic...


def main_plot_formatting(fig):
    # function to format plots
    font = "system-ui"
    fig.update_layout(font_family=font,
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      title_font_size=20, title_x=0.1, showlegend=True, font_color='rgb(255, 255, 255)',
                      xaxis=dict(
                          showline=True,
                          showgrid=False,
                          showticklabels=True,
                          linecolor='rgb(150, 150, 150)',
                          linewidth=2,
                          ticks='outside',
                          tickfont=dict(
                              family=font,
                              size=12,
                              color='rgb(255, 255, 255)',
                          )),
                      yaxis=dict(
                          showline=True,
                          showgrid=False,
                          showticklabels=True,
                          linecolor='rgb(150, 150, 150)',
                          linewidth=2,
                          ticks='outside',
                          tickfont=dict(
                              family=font,
                              size=12,
                              color='rgb(255, 255, 255)',
                          )), )
    fig.update_xaxes(showgrid=True, gridwidth=0.5, gridcolor='rgb(150, 150, 150)')
    fig.update_yaxes(showgrid=True, gridwidth=0.5, gridcolor='rgb(150, 150, 150)')
    return fig


def secondary_plot_formatting(fig):
    # function to format plots
    font = "system-ui"
    fig.update_layout(font_family=font,
                      paper_bgcolor='rgba(0,0,0,0)',
                      plot_bgcolor='rgba(0,0,0,0)',
                      title_font_size=20, title_x=0.1, showlegend=False, font_color='rgb(255, 255, 255)',
                      xaxis=dict(
                          showline=False,
                          showgrid=False,
                          showticklabels=False,
                          linecolor='rgb(0, 0, 0)',
                          linewidth=2,
                          ticks='outside',
                          tickfont=dict(
                              family=font,
                              size=12,
                              color='rgb(255, 255, 255)',
                          )),
                      yaxis=dict(
                          showline=False,
                          showgrid=False,
                          showticklabels=True,
                          linecolor='rgb(255, 255, 255)',
                          linewidth=2,
                          ticks='outside',
                          tickfont=dict(
                              family=font,
                              size=12,
                              color='rgb(255, 255, 255)',
                          )), )
    return fig


def main_plot(data, asset_name, max_length, hover_or_click):
    # function to create main price plot
    df = data[asset_name]
    x = df.index
    y = df.iloc[:, 0]
    # Get the difference between each timestamp
    time_diffs = df.index.to_series().diff()

    # Calculate the most common difference (in nanoseconds)
    most_common_diff = time_diffs.value_counts().idxmax()

    # Convert the most common difference to days, weeks or minutes as needed
    if most_common_diff >= pd.Timedelta('1D'):
        # Convert to days if the most common difference is 1 day or more
        most_common_diff = most_common_diff.days
        time_unit = 'D'
    elif most_common_diff >= pd.Timedelta('1H'):
        # Convert to hours if the most common difference is 1 hour or more
        most_common_diff = most_common_diff.seconds // 3600
        time_unit = 'H'
    elif most_common_diff >= pd.Timedelta('1T'):
        # Convert to minutes if the most common difference is 1 minute or more
        most_common_diff = most_common_diff.seconds // 60
        time_unit = 'T'
    else:
        # Otherwise, use seconds
        most_common_diff = most_common_diff.seconds
        time_unit = 'S'

    # Calculate the maximum future date to accommodate the maximum possible forecast range
    max_date = df.index.max() + pd.Timedelta(most_common_diff * 20, unit=time_unit)

    # Convert the most common difference back to Timedelta for pd.date_range
    most_common_diff = pd.Timedelta(most_common_diff, unit=time_unit)

    # Create new dates
    new_dates = pd.date_range(start=df.index.max() + most_common_diff, end=max_date, freq=most_common_diff)

    # Create a new DataFrame with these dates
    df_new = pd.DataFrame(index=new_dates, columns=df.columns)

    # Concatenate the old DataFrame with the new one
    df = pd.concat([df, df_new])

    if hover_or_click == 'Hover mode':
        fig = go.Figure(data=go.Scatter(x=x,
                                        y=y,
                                        connectgaps=True,
                                        name="Price",
                                        mode='lines+markers',
                                        hoverinfo='none'))
        # hovertemplate = 'Date: %{x} <br>Price: %{y:$.4f}<extra></extra>'))
    else:
        fig = go.Figure(data=go.Scatter(x=x,
                                        y=y,
                                        connectgaps=True,
                                        name="Price",
                                        mode='lines+markers',
                                        # hoverinfo='none'))
                                        hovertemplate='Date: %{x} <br>Price: %{y:$.4f}<extra></extra>'))

    fig.update_traces(line_color='white')
    fig.update_layout(xaxis_title="Date",
                      yaxis_title="Price",
                      # Add range to xaxis to keep it stable
                      xaxis_range=[df.index.min(), max_date])
    fig.update_layout(title_text=asset_name,
                      uirevision="Don't change")

    return fig, df


def generate_app(cluster):
    # building components of the app

    app = JupyterDash(external_stylesheets=[dbc.themes.DARKLY])
    # app = Dash(__name__)
    asset_predictions, asset_prices, asset_names, interval = generate_data(cluster)
    ap = asset_prices.copy()
    drop_down_assets = dcc.Dropdown(
        asset_names,
        multi=False,
        value=asset_names[0],
        id="assets_dropdown",
        style={'color': 'black',
               'width': '50%'}
    )
    hover_or_click = dcc.RadioItems(['Hover mode', 'Click mode'], 'Hover mode', id="hover_or_click")

    main_graphs = dbc.Card(
        [
            html.Div(
                [
                    dcc.Graph(id='main_graph'),
                ]
            ),
        ]
    )

    secondary_graphs = dbc.Card(
        [
            html.Div(
                [
                    dcc.Graph(id='secondary_graph'),
                ]
            ),
        ]
    )

    tab1_content = dbc.Container(
        [
            html.H1("Prediction results"),
            html.Hr(),

            dbc.Row(
                [
                    dbc.Col(drop_down_assets, width=8),
                ]),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(hover_or_click, width=4),
                ]),
            html.Br(),
            dbc.Row(
                [
                    dbc.Col(main_graphs),
                    dbc.Col(secondary_graphs),
                ],
                align="top",
            ),
        ],
        fluid=True,
    )

    tabs = dbc.Tabs(
        [
            dbc.Tab(tab1_content, label="Results"),
        ]
    )

    app.layout = tabs

    app.title = "Plutus Forecast App!"

    @app.callback(
        Output('main_graph', 'figure'),
        Input('assets_dropdown', 'value'),
        Input('main_graph', 'hoverData'),
        Input('main_graph', 'clickData'),
        Input('hover_or_click', 'value'),
        prevent_initial_call=False)
    def add_prediction(asset_name, hover_data, click_data, hover_or_click):

        if hover_or_click == 'Hover mode':

            if (hover_data is None or click_data is None):
                el = []
                for n in asset_predictions.keys():
                    el.append(int(len(asset_predictions[n])))
                max_length = max(el)
                fig, single_asset_price = main_plot(ap, asset_name, max_length, hover_or_click)
                fig = main_plot_formatting(fig)
                return fig
            # if you click/select a point of time...
            else:
                # get maximum length of the predictions like 126/200/300 etc.
                el = []
                for n in asset_predictions.keys():
                    el.append(int(len(asset_predictions[n])))
                max_length = max(el)
                fig, single_asset_price = main_plot(asset_prices, asset_name, max_length, hover_or_click)
                # The day you select on hover mode graph
                selected_day = pd.to_datetime(hover_data['points'][0]['x'])
                # Getting interval period ex. '15m' , '1d'
                for k in asset_predictions.keys():
                    name = k.split('_')[0]
                    interval = k.split('_')[5]
                    full_name = name + "_" + interval
                    if full_name == asset_name or name == asset_name:
                        ip = interval

                for key, val in asset_predictions.items():
                    # Next period from the selected timestamp
                    first_predicted_day = pd.to_datetime(hover_data['points'][0]['x']) + pd.to_timedelta(int(ip[:-1]),
                                                                                                         unit=ip[-1])
                    # if asset name is in asset prediction keys
                    if re.search(asset_name, key.split('_')[0] + "_" + key.split('_')[5]):
                        num_of_prediction_days = len(val.columns)
                        # N-future day from the selected day
                        # Prediction table copy
                        aux_df = val.copy()
                        # Real business n-future day
                        if '-USD' in name:
                            is_crypto = True
                        else:
                            is_crypto = False
                        # Creates n future days from the last index
                        if ip[-1] == 'm':
                            ld = val.index[-1] + dt.timedelta(minutes=(int(ip[:-1])*num_of_prediction_days))
                            add_days = pd.date_range(val.index[-1], ld, freq=(ip[:-1] + 'min'))[1:]
                        if ip[-1] == 'd' and is_crypto==False:
                            ld = val.index[-1] + BDay(num_of_prediction_days)
                            add_days = pd.bdate_range(aux_df.index[-1], ld)[1:]
                        if ip[-1] == 'd' and is_crypto == True:
                            ld = val.index[-1] + dt.timedelta(days=(int(ip[:-1]) * num_of_prediction_days))
                            add_days = pd.date_range(val.index[-1], ld, freq='D')[1:]
                        if ip[-1] == 'w':
                            ld = val.index[-1] + dt.timedelta(weeks=(int(ip[:-1])*num_of_prediction_days))
                            add_days = pd.bdate_range(aux_df.index[-1], ld, freq='W')[1:]

                        for x in add_days:
                            aux_df.loc[x] = np.zeros(num_of_prediction_days)
                        # Creates an array of indecies from selected day to N-future days
                        range_of_prediction_illlia = aux_df.loc[first_predicted_day:].index[:num_of_prediction_days]
                        try:
                            predictions_row = val.loc[selected_day]
                            prediction_for_plot = predictions_row.to_frame()
                        except:
                            pass
                        # Sets the indicies of prediction for the selected day
                        prediction_for_plot.set_index(range_of_prediction_illlia, inplace=True)
                        # adding actual price as first day of prediction to connect prediction lines with main line
                        prediction_for_plot.loc[selected_day] = ap[asset_name].loc[selected_day][0]
                        prediction_for_plot = prediction_for_plot.sort_index()
                        df_plot = pd.merge(single_asset_price, prediction_for_plot, left_index=True, right_index=True)
                        df_plot.columns = [asset_name + '_actual', asset_name + '_predicted']
                        x = df_plot.index
                        y = df_plot[asset_name + '_predicted']

                        fig.add_trace(go.Scatter(x=x,
                                                 y=y,
                                                 mode='lines',
                                                 name=key,
                                                 connectgaps=True,
                                                 hoverinfo='none'))
                        # hovertemplate='Date: %{x} <br>Price: %{y:$.4f}<extra></extra>'))
                    else:
                        pass

                fig = main_plot_formatting(fig)
                return fig
        else:

            if (hover_data is None or click_data is None):
                el = []
                for n in asset_predictions.keys():
                    el.append(int(len(asset_predictions[n])))
                max_length = max(el)
                fig, single_asset_price = main_plot(ap, asset_name, max_length, hover_or_click)
                fig = main_plot_formatting(fig)
                return fig

            else:
                el = []
                for n in asset_predictions.keys():
                    el.append(int(len(asset_predictions[n])))
                max_length = max(el)
                fig, single_asset_price = main_plot(asset_prices, asset_name, max_length, hover_or_click)

                selected_day = pd.to_datetime(click_data['points'][0]['x'])
                # interval period ex. '15m' , '1d'
                ip = next(iter(asset_predictions)).split('_')[5]

                for key, val in asset_predictions.items():
                    # Next period from the selected day
                    first_predicted_day = pd.to_datetime(hover_data['points'][0]['x']) + pd.to_timedelta(
                        int(ip[:-1]), unit=ip[-1])
                    # if asset name is in asset prediction keys
                    if re.search(asset_name, key.split('_')[0] + "_" + key.split('_')[5]):
                        num_of_prediction_days = len(val.columns)
                        # N-future day from the selected day
                        last_predicted_day = first_predicted_day + pd.to_timedelta(
                            (num_of_prediction_days * int(ip[:-1])), unit=ip[-1])
                        # Prediction table copy
                        aux_df = val.copy()
                        # Real business n-future day
                        if '-USD' in name:
                            is_crypto = True
                        else:
                            is_crypto = False
                        # Real business n-future day
                        # Creates n future days from the last index
                        if ip[-1] == 'm':
                            ld = val.index[-1] + dt.timedelta(minutes=int(ip[:-1]))
                            add_days = pd.date_range(val.index[-1], ld, freq=(ip[:-1] + 'min'))[1:]
                        if ip[-1] == 'd' and is_crypto == False:
                            ld = val.index[-1] + BDay(num_of_prediction_days)
                            add_days = pd.bdate_range(aux_df.index[-1], ld)[1:]
                        if ip[-1] == 'd' and is_crypto == True:
                            ld = val.index[-1] + dt.timedelta(days=(int(ip[:-1]) * num_of_prediction_days))
                            add_days = pd.date_range(val.index[-1], ld, freq='D')[1:]
                        if ip[-1] == 'w':
                            ld = val.index[-1] + dt.timedelta(weeks=int(ip[:-1]))
                            add_days = pd.bdate_range(aux_df.index[-1], ld, freq='W')[1:]
                        for x in add_days:
                            aux_df.loc[x] = np.zeros(num_of_prediction_days)
                        range_of_prediction_illlia = aux_df.loc[first_predicted_day:].index[:num_of_prediction_days]
                        range_of_prediction = pd.date_range(start=first_predicted_day, end=last_predicted_day, )
                        try:
                            predictions_row = val.loc[selected_day]
                        except Exception:
                            pass
                        try:
                            prediction_for_plot = predictions_row.to_frame()
                        except:
                            prediction_for_plot = predictions_row.copy()
                        prediction_for_plot.set_index(range_of_prediction_illlia, inplace=True)
                        # adding actual price as first day of prediction to connect prediction lines with main line
                        prediction_for_plot.loc[selected_day] = ap[asset_name].loc[selected_day][0]
                        prediction_for_plot = prediction_for_plot.sort_index()
                        df_plot = pd.merge(single_asset_price, prediction_for_plot, left_index=True, right_index=True)
                        df_plot.columns = [asset_name + '_actual', asset_name + '_predicted']
                        x = df_plot.index
                        y = df_plot[asset_name + '_predicted']

                        fig.add_trace(go.Scatter(x=x,
                                                 y=y,
                                                 mode='lines',
                                                 name=key,
                                                 connectgaps=True,
                                                 # hoverinfo='none'))
                                                 hovertemplate='Date: %{x} <br>Price: %{y:$.4f}<extra></extra>'))
                    else:
                        pass

                fig = main_plot_formatting(fig)
                return fig

    @app.callback(
        Output('secondary_graph', 'figure'),
        Input('assets_dropdown', 'value'),
        prevent_initial_call=False)
    def add_prediction_on_click(asset_name):

        relevant_models = []

        for key, val in asset_predictions.items():
            if re.search(asset_name, key.split('_')[0] + "_" + key.split('_')[5]):
                relevant_models.append(key)

        num_relevant_models = len(relevant_models)

        subplot_names = []
        for i in relevant_models:
            for ele in range(2):
                subplot_names.append(i)

        fig = sp.make_subplots(rows=num_relevant_models * 2,
                               cols=1,
                               shared_xaxes=True,
                               subplot_titles=(subplot_names))

        div_fig_row = 1
        diff_fig_row = 2

        for model in relevant_models:

            # Auxiliar graph
            dataset = asset_prices[asset_name].copy()
            dataset.columns = [0]
            asset_predictions_df = asset_predictions[model]

            for r in range(len(asset_predictions_df.columns)):
                to_add = pd.DataFrame(dataset[0].shift(-1 - r)).rename(columns={0: -1 - r})
                dataset = pd.concat([dataset, to_add], axis=1)
            aux_df = dataset.copy()
            aux_df.dropna(inplace=True)
            dataset.drop(columns=[0], inplace=True)
            dataset.dropna(inplace=True)
            dataset_pred = asset_predictions_df.copy()

            ind_pred = dataset_pred.index.tolist()
            ind_act = aux_df.index.tolist()
            match = [x for x in ind_pred if x in ind_act]
            pred_diviation = abs(dataset_pred.loc[match].values / dataset.loc[match].values - 1)
            data = pd.DataFrame(pred_diviation.T, columns=dataset.loc[match].index)
            ind = []
            for s in range(len(data.index)):
                string = 'Lag ' + str(s + 1)
                ind.append(string)
            data.index = ind

            dataset_pred.columns = list(range(len(dataset_pred.columns) + 1))[1:]
            dataset_pred[0] = aux_df[0]
            dataset_pred = dataset_pred[[0] + [col for col in dataset_pred.columns if col != 0]]
            dataset_pred.dropna(inplace=True)

            gg = dataset_pred.copy()
            gg2 = aux_df.copy()
            gg2.columns = gg.columns
            for x in gg.index:
                lst = gg.columns.tolist()
                lst.reverse()
                for y in lst:
                    gg.loc[x, y] = gg.loc[x, y] - gg.loc[x, 0]
            gg[gg > 0] = 1
            gg[gg < 0] = -1

            # drop 0 day column
            gg = gg[gg.columns.tolist()[1:]]

            for x in gg2.index:
                lst = gg2.columns.tolist()
                lst.reverse()
                for y in lst:
                    gg2.loc[x, y] = gg2.loc[x, y] - gg2.loc[x, 0]

            gg2[gg2 > 0] = 1
            gg2[gg2 < 0] = -1
            # drop 0 day column
            gg2 = gg2[gg2.columns.tolist()[1:]]

            gg3 = gg + gg2
            gg3[gg3 != 0] = 1
            gg4 = gg3.sum() / len(gg3)

            a1 = aux_df.loc[match].T.pct_change().T.dropna(axis=1).values
            a2 = dataset_pred.loc[match].T.pct_change().T.dropna(axis=1).values
            a1 = a1
            a2 = a2

            a3 = a1 * a2
            a3[a3 > 0] = 1
            a3[a3 < 0] = -1
            directional_accuracy = ((pd.DataFrame(a3) + 1) / 2).mean()

            print(f'Directional accuracy for model {model}')
            print(gg4.values)
            save_directional_accuracy_score(model,gg4)
            data_diff = pd.DataFrame(gg3.T, columns=aux_df.loc[match].index)
            data_diff.index = ind

            div_fig = go.Heatmap(z=data.values,
                                 x=data.columns,
                                 y=data.index,
                                 colorscale='Reds',
                                 showscale=False)

            diff_fig = go.Heatmap(z=data_diff.values,
                                  x=data_diff.columns,
                                  y=data_diff.index,
                                  colorscale='BuGn',
                                  showscale=False)

            fig.add_trace(div_fig, row=div_fig_row, col=1)
            fig.add_trace(diff_fig, row=diff_fig_row, col=1)

            div_fig_row += 2
            diff_fig_row += 2

        def dinamic_height(num_models):
            check_models = num_models * 2
            if 1 <= check_models <= 2:
                return 300 * (num_models * 2)
            elif 3 <= check_models <= 6:
                return 150 * (num_models * 2)
            else:
                return 100 * (num_models * 2)

        fig.update_layout(height=dinamic_height(num_relevant_models), title_text="Models performance")

        fig = secondary_plot_formatting(fig)

        return fig

    app.run_server(debug=False, port = np.random.randint(0, 1024))
