from dash import html, dcc, Input, Output
import dash_bootstrap_components as dbc
from jupyter_dash import JupyterDash
import re
import plotly.subplots as sp
from app_service.input_data_maker import generate_data
from flask import Flask
import logging
from app_service.visualise_predictions import add_prediction
from app_service.app_utilities import *

app = Flask(__name__)

log = logging.getLogger('werkzeug')
log.disabled = True


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
        [Input('assets_dropdown', 'value'),
         Input('main_graph', 'hoverData'),
         Input('main_graph', 'clickData'),
         Input('hover_or_click', 'value')],
        prevent_initial_call=False)
    def update_graph(asset_name, hover_data, click_data, hover_or_click):
        # Determine the interaction data based on the hover_or_click value
        if hover_data is None and click_data is None:
            fig, single_asset_price = main_plot(ap, asset_name, hover_or_click)
            fig = main_plot_formatting(fig)
            return fig

        else:
            interaction_data = hover_data if hover_or_click == "Hover mode" else click_data
            # Call your updated function
            fig = add_prediction(asset_name, interaction_data, hover_or_click, asset_prices, asset_predictions)
            return fig



    @app.callback(
        Output('secondary_graph', 'figure'),
        Input('assets_dropdown', 'value'),
        prevent_initial_call=False)
    def update_statistic_graph(asset_name):

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
            deviation_data, deviation_data_diff = auxiliary_dataframes(model, asset_prices, asset_predictions,
                                                                       asset_name)

            div_fig = go.Heatmap(z=deviation_data.values,
                                 x=deviation_data.columns,
                                 y=deviation_data.index,
                                 colorscale='Reds',
                                 showscale=False)

            diff_fig = go.Heatmap(z=deviation_data_diff.values,
                                  x=deviation_data_diff.columns,
                                  y=deviation_data_diff.index,
                                  colorscale='BuGn',
                                  showscale=False)

            fig.add_trace(div_fig, row=div_fig_row, col=1)
            fig.add_trace(diff_fig, row=diff_fig_row, col=1)

            div_fig_row += 2
            diff_fig_row += 2

        fig.update_layout(height=dynamic_height(num_relevant_models), title_text="Models performance")

        fig = secondary_plot_formatting(fig)

        return fig

    app.run_server(debug=False, port=np.random.randint(0, 1024))
