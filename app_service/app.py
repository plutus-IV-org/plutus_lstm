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
import socket
from contextlib import closing

app = Flask(__name__)

log = logging.getLogger('werkzeug')
log.disabled = True


def find_free_port(start_port, end_port):
    for port in range(start_port, end_port):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            if sock.connect_ex(('localhost', port)) != 0:
                return port
    raise IOError('No free ports')


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

    all_averages_toggle = dcc.RadioItems(
        options=[
            {'label': 'All ', 'value': 'all'},
            {'label': 'Average ', 'value': 'average'},
            {'label': 'Simple ', 'value': 'simple-average'},
            {'label': 'Top ', 'value': 'top-average'},
            {'label': 'Long ', 'value': 'long-average'},
            {'label': 'Short ', 'value': 'short-average'},
            {'label': 'Favorite ', 'value': 'favorite'},
        ],
        value='simple-average',
        id='all_averages_toggle'
    )
    anomalies_controls = html.Div([
        dbc.Row([
            # 'Apply Anomalies' radio items with reduced width to push them to the left
            dbc.Col([
                dbc.RadioItems(
                    options=[
                        {"label": "Anomalies_on", "value": 1},
                        {"label": "Anomalies_off", "value": 0},
                    ],
                    value=0,
                    id="anomalies_toggle",
                    inline=True,
                    style={'display': 'flex', 'alignItems': 'center', 'height': '30px'}
                ),
            ], width=3, lg=2),  # Adjust 'lg' for larger screens if needed

            # Spacer column to push input fields to the right
            dbc.Col(width=5, lg=6),

            # Input fields in one line with smaller boxes, aligned to the right
            dbc.Col([
                dbc.InputGroup(
                    [
                        dbc.InputGroupText("Window"),
                        dbc.Input(id="anomalies_window", type="number", value=2, size="sm"),
                        dbc.InputGroupText("Rolling"),
                        dbc.Input(id="rolling_period", type="number", value=20, size="sm"),
                        dbc.InputGroupText("Z-Score"),
                        dbc.Input(id="zscore_lvl", type="number", value=2, size="sm"),
                    ],
                    size="sm",
                    className="mb-2",
                    style={'flexWrap': 'nowrap', 'width': 'fit-content'}
                    # Use 'fit-content' to shrink wrap the input group
                ),
            ], width=4, lg=4, style={'display': 'flex', 'justifyContent': 'end'}),  # Adjust 'justifyContent' as needed
        ]),
    ], className="mb-4")

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
                    dbc.Col(all_averages_toggle, width=4)
                ]),
            html.Br(),

            anomalies_controls,

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
        [
            Input('assets_dropdown', 'value'),
            Input('main_graph', 'hoverData'),
            Input('main_graph', 'clickData'),
            Input('hover_or_click', 'value'),
            Input('anomalies_toggle', 'value'),
            Input('anomalies_window', 'value'),
            Input('rolling_period', 'value'),
            Input('zscore_lvl', 'value'),
            Input('all_averages_toggle', 'value')
        ],
        prevent_initial_call=False)
    def update_graph(asset_names, hover_data, click_data, hover_or_click, anomalies_toggle, anomalies_window,
                     rolling_period, zscore_lvl, all_averages_toggle):
        selected_dict = select_dictionaries(asset_predictions, all_averages_toggle)
        # Determine the interaction data based on the hover_or_click value
        if hover_data is None and click_data is None:
            fig, single_asset_price = main_plot(ap, asset_names, hover_or_click, anomalies_toggle, anomalies_window,
                                                rolling_period, zscore_lvl)
            fig = main_plot_formatting(fig)
            return fig

        else:
            interaction_data = hover_data if hover_or_click == "Hover mode" else click_data
            # Call your updated function
            fig = add_prediction(asset_names, interaction_data, hover_or_click, asset_prices, selected_dict,
                                 anomalies_toggle, anomalies_window,
                                 rolling_period, zscore_lvl)
            return fig

    @app.callback(
        Output('secondary_graph', 'figure'),
        [
            Input('assets_dropdown', 'value'),
            Input('anomalies_toggle', 'value'),
            Input('anomalies_window', 'value'),
            Input('rolling_period', 'value'),
            Input('zscore_lvl', 'value'),
            Input('all_averages_toggle', 'value')
        ],
        prevent_initial_call=False)
    def update_statistic_graph(asset_name, anomalies_toggle, anomalies_window, rolling_period, zscore_lvl,
                               all_averages_toggle):

        relevant_models = []
        selected_dict = select_dictionaries(asset_predictions, all_averages_toggle)
        for key, val in selected_dict.items():
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
            deviation_data, deviation_data_diff = auxiliary_dataframes(model, asset_prices, selected_dict,
                                                                       asset_name, anomalies_toggle, anomalies_window,
                                                                       rolling_period, zscore_lvl)

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

    # Find a free port in the range 1024 to 65535
    free_port = find_free_port(1024, 65535)

    app.run_server(debug=False, port=free_port)
