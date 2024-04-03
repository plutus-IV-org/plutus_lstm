from dash import html, dcc, Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc
from jupyter_dash import JupyterDash
import plotly.express as px
import plotly.subplots as sp
from app_service.input_data_maker import generate_data
from flask import Flask
import logging
from app_service.visualise_predictions import add_prediction
from app_service.app_utilities import *
import socket
from contextlib import closing
from playsound import playsound
from Const import DA_TABLE
import datetime as dt

app = Flask(__name__)

log = logging.getLogger('werkzeug')
log.disabled = True
sound_path = r"C:\Users\ilsbo\PycharmProjects\plutus_lstm\Notifications\Sound\run_command_report_generated.mp3"


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
    # TODO Additional tab functionalities
    # New tab content with additional functionalities
    tab_2_asset_names = asset_names.copy()
    tab_2_asset_names.append(asset_names[0].split('_')[0] + '_All')

    additional_content = dbc.Container(
        [
            html.H2("Directional accuracy score rating"),
            html.Hr(),
            dbc.Row(
                [
                    # Dropdown
                    dbc.Col(
                        dcc.Dropdown(
                            tab_2_asset_names,
                            multi=False,
                            value=tab_2_asset_names[-1] if asset_names else None,
                            id="tab_2_asset_names_dropdown",
                            style={'color': 'black'}
                        ),
                        width={"size": 3, "offset": 0},
                        style={'marginRight': 20},  # Add right margin for spacing
                    ),

                    # Input Group
                    dbc.Col(
                        dbc.InputGroup(
                            [
                                dbc.InputGroupText("Top"),
                                dbc.Input(id="top_amount", type="number", value=5, size="sm"),
                                dbc.InputGroupText("Cut"),
                                dbc.Input(id="unicorn_lvl", type="number", value=0.85, step=0.05, size="sm"),
                                dbc.InputGroupText("Length"),
                                dbc.Input(id="step_back", type="number", value=20, size="sm"),
                            ],
                            size="sm",
                            className="mb-2",
                            style={'flexWrap': 'nowrap'}
                        ),
                        width={"size": 3, "offset": 0},
                        style={'marginRight': 20, 'marginLeft': 20}  # Add margins for spacing
                    ),

                    # Radio Items
                    dbc.Col(
                        dcc.RadioItems(
                            options=[
                                {'label': 'All', 'value': 'all'},
                                {'label': 'Average', 'value': 'average'},
                                {'label': 'Simple', 'value': 'simple-average'},
                                {'label': 'Top', 'value': 'top-average'},
                                {'label': 'Long', 'value': 'long-average'},
                                {'label': 'Short', 'value': 'short-average'},
                                {'label': 'Favorite', 'value': 'favorite'},
                            ],
                            value='simple-average',
                            id='tab_2_all_averages_toggle',
                            style={'display': 'flex', 'flexDirection': 'row'}
                        ),
                        width={"size": 4, "offset": 0},
                        style={'marginLeft': 100}  # Add left margin for spacing
                    ),
                ],
                align="center",
                style={'marginLeft': 0, 'marginRight': 0},
            ),
            html.Br(),
            dcc.Graph(id='score-ratings-graph'),
            html.Br(),
            dcc.Graph(id='ranked_score-ratings-graph'),
        ],
        fluid=True,
    )

    tabs = dbc.Tabs(
        [
            dbc.Tab(tab1_content, label="Results"),
            dbc.Tab(additional_content, label="Score ratings")
        ]
    )

    app.layout = tabs
    app.title = "Plutus Forecast App!"

    # end
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
        [
            Output('score-ratings-graph', 'figure'),
            Output('ranked_score-ratings-graph', 'figure')
        ],
        [
            Input('tab_2_asset_names_dropdown', 'value'),
            Input('tab_2_all_averages_toggle', 'value'),
            Input('top_amount', 'value'),
            Input('unicorn_lvl', 'value'),
            Input('step_back', 'value')
        ],
        prevent_initial_call=False
    )
    def update_score_ratings_graph(asset_name: str, type: str, top: int, unicorn_lvl: float, length_step: int):
        if not asset_name or not type:
            raise PreventUpdate

        df = filter_loaded_history_score(asset_name, type)
        grouped_df = group_by_history_score(df)
        # remove overated records
        grouped_df = grouped_df[grouped_df['da_score'] < unicorn_lvl]
        # N-step back cut
        date_to = grouped_df.date.max()
        date_from = date_to - dt.timedelta(length_step)
        grouped_df = grouped_df[grouped_df['date'] > date_from]
        ranked_df = prepare_rating(grouped_df)
        filtered_df = filter_top_ranked_models(ranked_df, top)

        # Convert the 'date' column to datetime and sort
        filtered_df['date'] = pd.to_datetime(filtered_df['date'])
        filtered_df.sort_values('date', inplace=True)

        # Define a color palette (You may need more colors depending on the number of models)
        color_palette = px.colors.qualitative.Set1

        # Create a color map dictionary for model names to specific colors
        unique_model_names = filtered_df['model_name'].unique()
        color_map = {model: color_palette[i % len(color_palette)] for i, model in enumerate(unique_model_names)}

        # Add traces for each unique model name
        fig = px.scatter(
            filtered_df,
            x='date',
            y='da_score',
            color='model_name',  # Assign colors based on model_name which will be used in the legend
            title='Top Ranked Models DA Score by Date'
        )

        # Add the average line
        fig.add_hline(y=0.5, line_dash="dash", line_color="white")

        # Update the layout
        fig.update_layout(
            plot_bgcolor='#404040',  # Dark gray background
            paper_bgcolor='#404040',  # Dark gray paper background
            font=dict(color='white'),
            xaxis_title='Date',
            yaxis_title='DA Score',
            xaxis=dict(
                showgrid=False,
                tickangle=-45,
                type='date'  # Treat 'date' as a datetime type
            ),
            yaxis=dict(
                showgrid=False,
                range=[0, 1]  # DA score ranges from 0 to 1
            ),
            hovermode='x unified',
            legend_title_text='Models'
        )

        # Here's the important part: setting hovertemplate correctly
        # Loop over each trace and update the customdata and hovertemplate
        for trace, model_name in zip(fig.data, filtered_df['model_name'].unique()):
            # Filter the dataframe for the current model
            model_df = filtered_df[filtered_df['model_name'] == model_name]
            # Update the current trace
            trace.customdata = model_df['model_name']
            trace.hovertemplate = "<br>Model: %{customdata}<br>Score: %{y:.2f}<extra></extra>"

        # Create a scatter plot for ranking
        fig_vam = px.scatter(
            filtered_df,
            x='date',
            y='rank',
            color='model_name',  # Assign colors based on model_name which will be used in the legend
            title='Top Ranked Models Rank by Date',
            size_max=15  # Increase the marker size
        )

        # Update the layout for the ranking plot
        fig_vam.update_layout(
            plot_bgcolor='#404040',  # Dark gray background
            paper_bgcolor='#404040',  # Dark gray paper background
            font=dict(color='white'),
            xaxis_title='Date',
            yaxis_title='Rank',
            xaxis=dict(
                showgrid=False,
                tickangle=-45,
                type='date'  # Treat 'date' as a datetime type
            ),
            yaxis=dict(
                autorange='reversed',  # Reverse the y-axis
                showgrid=False,
                # Do not set a fixed range here as we want it to be dynamic based on the ranking
            ),
            hovermode='x unified',
            legend_title_text='Models'
        )

        # Increase the size of the dots
        fig_vam.update_traces(marker=dict(size=30))  # You can adjust the size value as needed

        # Here's the important part: setting hovertemplate correctly
        # Loop over each trace and update the customdata and hovertemplate
        for trace, model_name in zip(fig_vam.data, filtered_df['model_name'].unique()):
            # Filter the dataframe for the current model
            model_df = filtered_df[filtered_df['model_name'] == model_name]
            # Update the current trace
            trace.customdata = model_df['model_name']
            trace.hovertemplate = "<br>Model: %{customdata}<br>Score: %{y:.2f}<extra></extra>"

        return fig, fig_vam

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
                               shared_xaxes=False,
                               subplot_titles=(subplot_names))

        history_score_fig_row = 1
        diff_fig_row = 2

        for model in relevant_models:
            deviation_data, deviation_data_diff = auxiliary_dataframes(model, asset_prices, selected_dict,
                                                                       asset_name, anomalies_toggle, anomalies_window,
                                                                       rolling_period, zscore_lvl)
            df = directional_accuracy_history_load(DA_TABLE)
            selected_df = df[df['model_name'] == model]
            grouped_df = group_by_history_score(selected_df)
            # Load the DataFrame and extract unique model names
            # history_score_fig = go.Scatter(x=grouped_df['date'],
            #                                y=grouped_df['da_score'],
            #                                mode='lines',  # Use 'markers' for scatter plot
            #                                marker=dict(
            #                                    color=grouped_df['da_score'],
            #                                    # Color of markers based on da_score values
            #                                    colorscale='magma',  # You can choose any supported colorscale
            #
            #                                ))

            history_score_fig = go.Bar(
                x=grouped_df['date'],
                y=grouped_df['da_score'],
                marker=dict(
                    color=grouped_df['da_score'],
                    cmin=0,
                    cmax=1,  # Set the color of the bars based on the da_score
                    colorscale='Turbo',  # Or any other color scale that you find eye-friendly
                )
            )

            div_fig = go.Heatmap(z=deviation_data.values,
                                 x=deviation_data.columns,
                                 y=deviation_data.index,
                                 colorscale='Reds',
                                 showscale=False)
            # round(deviation_data_diff.mean(axis=1), 2).values.tolist()
            mean_values = deviation_data_diff.mean(axis=1).values.reshape(-1, 1)  # Reshape to (10, 1)
            ones_array = np.ones((len(deviation_data_diff.index), len(deviation_data_diff.columns)))
            broadcasted_array = ones_array * mean_values
            text_for_hover = broadcasted_array.astype(str)

            diff_fig = go.Heatmap(
                z=deviation_data_diff.values,
                x=deviation_data_diff.columns,
                y=deviation_data_diff.index,
                colorscale='BuGn',
                showscale=False,
                text=text_for_hover,
                hovertemplate='Date: %{x}<br>Lag: %{y} (%{z})<br>' +
                              'Lag mean: %{text}<extra></extra>'
            )

            # fig.add_trace(div_fig, row=div_fig_row, col=1)
            fig.add_trace(history_score_fig, row=history_score_fig_row, col=1)
            fig.add_trace(diff_fig, row=diff_fig_row, col=1)

            # div_fig_row += 2
            history_score_fig_row += 2
            diff_fig_row += 2

        fig.update_layout(height=dynamic_height(num_relevant_models), title_text="Models performance")

        fig = secondary_plot_formatting(fig)

        return fig

    # Find a free port in the range 1024 to 65535
    free_port = find_free_port(1024, 65535)

    playsound(sound_path)
    app.run_server(debug=False, port=free_port)
