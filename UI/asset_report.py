import dash
import dash_bootstrap_components as dbc
from dash import html, dcc
from dash import dash_table
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import timedelta


# Define the plotting function
def plot_selected_items(data: pd.DataFrame):
    selected_df = data
    last_month_date = selected_df.index.max() - timedelta(days=30)
    last_month_df = selected_df[selected_df.index >= last_month_date]

    num_rows = len(selected_df.columns)
    fig = make_subplots(rows=num_rows, cols=2, shared_xaxes=True, shared_yaxes=False,
                        vertical_spacing=0.025, horizontal_spacing=0.03, column_widths=[0.65, 0.35])
    for i, col in enumerate(selected_df.columns):
        fig.add_trace(go.Scatter(x=selected_df.index, y=np.log(selected_df[col]),
                                 mode='lines', name=f"{col} Full",
                                 hovertemplate='%{x|%d-%m-%Y}: %{y:.2f}'), row=i + 1, col=1)
        fig.add_trace(go.Scatter(x=last_month_df.index, y=np.log(last_month_df[col]),
                                 mode='lines', name=f"{col} Last Month",
                                 hovertemplate='%{x|%d-%m}: %{y:.2f}'), row=i + 1, col=2)
        min_y = np.log(last_month_df[col]).min()
        max_y = np.log(last_month_df[col]).max()
        fig.update_yaxes(range=[min_y - 0.1 * (max_y - min_y), max_y + 0.1 * (max_y - min_y)], row=i + 1, col=2)

    fig.update_layout(title="Normalised data distribution", hovermode='x unified',
                      plot_bgcolor='#2C2C2C', paper_bgcolor='#2C2C2C',
                      font=dict(color='#FFFFFF'),
                      height=200 * num_rows, width=1800)
    fig.update_xaxes(showline=True, linewidth=1, linecolor='white', mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor='white', mirror=True)
    return fig


def generate_full_report(data: pd.DataFrame, *source_dfs):
    # Concatenate all source DataFrames
    all_sources = pd.concat(source_dfs).reset_index()

    # Sort the concatenated DataFrame by Time
    all_sources['Time'] = pd.to_datetime(all_sources['Time'])
    all_sources.sort_values('Time', inplace=True)
    all_sources['Time'] = all_sources['Time'].dt.strftime('%d-%m %H:%M')

    # Initialize the Dash application with the DARKLY theme from DBC
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
    table_width = 1000
    url_col_width = int(0.05 * table_width)

    app.layout = html.Div([
        html.Div([
            html.H4("News line"),
            dash_table.DataTable(
                id='table_all_sources',
                columns=[{"name": i, "id": i} for i in all_sources.columns],
                data=all_sources.to_dict('records'),
                style_data_conditional=[{'if': {'column_id': 'URL'},
                                         'textDecoration': 'underline', 'cursor': 'pointer'}],
                style_cell_conditional=[{'if': {'column_id': 'URL'}, 'maxWidth': f'{url_col_width}px',
                                         'overflow': 'hidden', 'textOverflow': 'ellipsis'},
                                        {'if': {'column_id': 'Title'}, 'minWidth': '0px', 'width': 'auto',
                                         'maxWidth': 'none'}],
                style_table={'width': f'{table_width}px', 'tableLayout': 'fixed'},
                style_header={'backgroundColor': '#2C2C2C', 'fontWeight': 'bold', 'color': '#FFFFFF'},
                style_cell={'backgroundColor': '#1E1E1E', 'color': '#FFFFFF'},
            ),
        ]),
        dcc.Graph(figure=plot_selected_items(data))
    ])

    @app.callback(Output('table_all_sources', 'selected_row_ids'),
                  Input('table_all_sources', 'active_cell'))
    def click_to_select_row(active_cell):
        if active_cell:
            return [active_cell['row']]
        return []

    return app.run_server(debug=False)
