from dash import Dash, dcc, html, dash_table, ctx
from dash.dependencies import Input, Output, ALL, State, MATCH, ALLSMALLER
import plotly.express as px
import pandas as pd
import numpy as np
import dash_bootstrap_components as dbc
from dash.exceptions import PreventUpdate
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from insert_postgres import postgres
import plotly.graph_objects as go


def tranc_names(list_attr):
    """
    Truncates large strings in order to fit in plots.
    """
    attr = []
    for l in list_attr:
        if len(l) > 15:
            l = l[:11] + '...'
        attr.append(l)
    return attr


def unique_values(df):
    k = df.nunique()
    fig = go.Figure(go.Bar(
        x=list(k.values),
        y=list(k.index),
        orientation='h'))
    return fig


def corfig(corrMatrix):
    """
    Correlation Plot.
    """
    fig = px.imshow(corrMatrix, labels=dict(x="attr1", y='attr2', color='Correlation'), color_continuous_scale="PuBu",
                    text_auto=True)
    fig.update_yaxes(tickangle=-45, title=None,
                     ticktext=tranc_names(list(corrMatrix.index)),
                     tickvals=list(corrMatrix.index),
                     )
    fig.update_xaxes(tickangle=45, title=None,
                     ticktext=tranc_names(list(corrMatrix.index)),
                     tickvals=list(corrMatrix.index),
                     )
    return fig


def create_dash(csv_file_name, final_tables):
    """
    Creates a dashboard based on the processed csv file.
    """
    df_set = []
    df_attr = []
    metadata = []
    confidence = []
    footnotes = []
    df_attr_type = []
    attr_name_type = []
    cluster_id = []
    cluster_summary = []
    for i in final_tables:
        #df_set.append(pd.concat([pd.Series(final_tables[i]['clustering_results'][1], dtype=np.int64, name='Cluster'), final_tables[i]['table_data']], axis=1))
        df_set.append(final_tables[i]['table_data'])
        df_attr.append(list(final_tables[i]['attr_names'].keys()))
        #df_attr_type.append(['numeric'] + list(final_tables[i]['attr_names'].values()))
        df_attr_type.append(list(final_tables[i]['attr_names'].values()))
        attr_name_type.append(
            dict(zip(['Cluster'] + list(final_tables[i]['attr_names'].keys()),
                     ['numeric'] + list(final_tables[i]['attr_names'].values()))))
        '''attr_name_type.append(
            dict(zip( list(final_tables[i]['attr_names'].keys()),
                     list(final_tables[i]['attr_names'].values()))))'''
        metadata.append(final_tables[i]['metadata'])
        confidence.append(final_tables[i]['confidence'])
        footnotes.append(final_tables[i]['footnotes'])
        cluster_summary.append(final_tables[i]['clustering_results'][0])
        cluster_id.append(final_tables[i]['clustering_results'][1])

    app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], )  # suppress_callback_exceptions=True)

    ## Side bar menu for navigatation in discovered tables ##
    SIDEBAR_STYLE = {
        "position": "fixed",
        "top": 0,
        "left": 0,
        "bottom": 0,
        "width": "16rem",
        "padding": "2rem 1rem",
        "background-color": "#f8f9fa",
    }
    # padding for the page content
    CONTENT_STYLE = {
        "margin-left": "18rem",
        "margin-right": "2rem",
        "padding": "2rem 1rem",
    }
    nav_options = [dbc.NavLink(f"Table {i}", href=f"/{i}", active="exact") for i in range(len(df_set))]
    sidebar = html.Div([
        html.H2("Sidebar", className="display-4"),
        html.Hr(),
        html.P("Discovered Tables ", className="lead"),
        dbc.Nav(
            [dbc.NavLink("Home", href="/", active="exact")] + nav_options,
            vertical=True,
            pills=True, ), ],
        style=SIDEBAR_STYLE, )

    initial_content = html.Div([
        html.H1(csv_file_name, ),
        html.H3(f'Tables discovered:  {len(df_set)}')])

    content = html.Div(id='container', children=[], style=CONTENT_STYLE)

    app.layout = html.Div([
        dcc.Location(id="url"),
        sidebar,
        content
    ])

    @app.callback(
        Output('container', 'children'),
        Input("url", "pathname"), prevent_initial_call=True
    )
    def display_graphs(value):
        print(value)
        if value == '/':
            return initial_content
        else:
            df = df_set[int(value[1:])]
            df_cluster = cluster_summary[int(value[1:])]
            #cluster_id_ = cluster_id[int(value[1:])]
            df_tb = pd.concat([pd.Series(cluster_id[int(value[1:])], dtype=np.int64, name='Cluster'),df], axis=1)

            # df['id'] = df.index      # for subheader indication

            # Metadata
            str_metadata = ''
            if metadata[int(value[1:])]:
                for m in metadata[int(value[1:])]:
                    str_metadata = str_metadata + '#### ' + m + '\n'

            # Footnotes
            str_footnotes = ''
            if footnotes[int(value[1:])]:

                for m in footnotes[int(value[1:])]:
                    str_footnotes = str_footnotes + '###### ' + m + '\n'

            # Pytheas confidence scores
            conf_markdown = f'#### **Pytheas conficende score**\n ' \
                            f"#### Table start : {np.round(confidence[int(value[1:])]['body_start'], 3)}\n" \
                            f"#### Table end   : {np.round(confidence[int(value[1:])]['body_end'], 3)}"

            corrMatrix = np.round(df.corr(), 2)

            new_child = dbc.Container(
                # style={'width': '45%', 'display': 'inline-block', 'outline': 'thin lightgrey solid', 'padding': 10},
                children=[
                    dbc.Row(dbc.Col(html.Div(html.H1('Dashboard', className='text-center')))),
                    dbc.Row(
                        [dbc.Col(html.Div(dcc.Markdown(str_metadata), className='rounded bg-light p-1'), width='auto'),
                         dbc.Col(html.Div(dcc.Markdown(conf_markdown), className='rounded bg-danger p-1'),
                                 width='auto')],
                        justify="between", ),  # className='pb-3 pt-3'),
                    dbc.Row([
                        dbc.Col(html.Div([html.Button("Download CSV", id="btn_csv"),
                                          html.Button("Insert POSTGRES", id="insert-postgres"),
                                          # html.Button("Insert MONGO", id="insert-mongo"),
                                          dcc.Download(id="download-dataframe-csv"),
                                          ]), width="auto"),
                        dbc.Col(html.Div(html.Button("Reset table", id="reset-table")), width="auto")],
                        justify='between', className='mt-3 mb-1'),
                    dbc.Row(  # It is appeared if POSTGRES button get pushed
                        dbc.Modal([
                            dbc.ModalHeader("Postgres"),
                            dbc.ModalBody(
                                dbc.Form([
                                    html.Div([
                                        html.Div([
                                            dbc.Label("Username", className="mt-2 mb-0"),
                                            dbc.Input(type="text", id='input-user', placeholder="Enter username",
                                                      value='postgres'), ],
                                            className="me-2", style={'width': '49%', 'display': 'inline-block', }, ),
                                        html.Div([
                                            dbc.Label("Password", className="mt-2 mb-0"),
                                            dbc.Input(type="password", placeholder="Enter password",
                                                      id='input-password', value='12345a'), ],
                                            # className="mb-3",
                                            style={'width': '49%', 'display': 'inline-block'}, )], ),
                                    html.Div([
                                        dbc.Label("Database path", className="mt-2 mb-0"),
                                        dbc.Input(type="text", placeholder="Enter postgress path", id='input-path',
                                                  value='C:/Program Files/PostgreSQL/13/'), ],
                                        style={'width': '99%'}, ),
                                    html.Div([
                                        html.Div([
                                            dbc.Label("Port", className="mt-2 mb-0"),
                                            dbc.Input(type="number", id='input-port', placeholder="Enter port",
                                                      value=5432), ],
                                            className="me-2",
                                            style={'width': '49%', 'display': 'inline-block', }, ),
                                        html.Div([
                                            dbc.Label("Database", className="mt-2 mb-0"),
                                            dbc.Input(type="text", id='input-database',
                                                      placeholder="Enter database name", value='test'), ],
                                            className="mb-3",
                                            style={'width': '49%', 'display': 'inline-block', }, ), ], ),
                                    dbc.Button("Submit", id='submit', color="primary"), ], )),
                            dbc.ModalFooter(dbc.Button("Close", id="close", className="ml-auto")), ],
                            id="modal",
                            is_open=False,  # True, False
                            size="lg",  # "sm", "lg", "xl"
                            backdrop=True,  # True, False or Static for modal to not be closed by clicking on backdrop
                            scrollable=True,  # False or True if modal has a lot of text
                            centered=True,  # True, False
                            fade=True  # True, False
                        )),
                    dbc.Row(  # Data Table of csv file
                        dbc.Col(
                            html.Div([
                                html.H4(csv_file_name[:-4] + '_table' + value[1:], className='text-center border', ),
                                dash_table.DataTable(id='my_table',
                                                     columns=[
                                                         {'name': i, 'id': i,
                                                          'type': attr_name_type[int(value[1:])][i] if
                                                          attr_name_type[int(value[1:])][i] not in ['time',
                                                                                                    'date'] else 'datetime',
                                                          'deletable': True,
                                                          # 'renamable': True,
                                                          'selectable': True,
                                                          } for i in df_tb.columns if i != 'id'],
                                                     data=df_tb.to_dict('records'),
                                                     filter_action='native',
                                                     row_deletable=True,
                                                     column_selectable="multi",
                                                     sort_action="native",
                                                     fixed_rows={'headers': True},
                                                     style_table={'height': 600, 'overflowY': 'auto',
                                                                  'overflowX': 'auto'},
                                                     tooltip_header={
                                                         c: c + ' (type : ' + attr_name_type[int(value[1:])][c] + ')'for
                                                         c in df_tb.columns},
                                                     # df_attr_type[int(value[1:])][i] + ')' for
                                                     style_data={
                                                         # 'width': '{}%'.format(100. / len(df.columns)),
                                                         'minWidth': '160px', 'width': '160px',
                                                         'maxWidth': '160px',
                                                         'textOverflow': 'hidden',
                                                         'whiteSpace': 'normal',
                                                     },
                                                     style_header={
                                                         'minWidth': '160px', 'width': '160px',
                                                         'maxWidth': '160px',
                                                         'textOverflow': 'ellipsis',
                                                         'overflow': 'hidden',
                                                     },
                                                     css=[{"selector": ".Select-menu-outer",
                                                           "rule": "display: block !important"}], )],
                                className="shadow-lg bg-white rounded p-3")), ),
                    dbc.Row(html.Div(dcc.Markdown(str_footnotes, ), style={'width': 'auto'}),
                            className='justify-content-end mb-3 '),
                    dbc.Row(
                        [dbc.Col(
                            html.Div([
                                html.Div(
                                    dcc.RadioItems(
                                        id={'type': 'dynamic-radio', 'index': value[-1]},
                                        options=[{'label': k, 'value': k} for k in df.columns if
                                                 attr_name_type[int(value[1:])][k] != 'text'],
                                        value=[k for k in df.columns if attr_name_type[int(value[1:])][k] != 'text'][0],
                                        inputStyle={"margin-left": "15px"}), ),
                                # style = {'width':'400px', 'border':'1px solid black'}
                                dcc.Graph(id={'type': 'dynamic-graph2', 'index': value[-1]
                                              }, figure={})
                            ]), width=6),
                            dbc.Col(
                                html.Div(
                                    dcc.Graph(id='count-unique', figure=unique_values(df))),
                                width=6)]),
                    dbc.Row([
                        dbc.Col(
                            html.Div(
                                dcc.Graph(id='static_fig', figure=corfig(corrMatrix))), align="center", width=6),
                        dbc.Col(
                            html.Div([
                                dcc.Graph(id={'type': 'scatter-plots', 'index': value[-1]}),
                                dbc.Label("Absolute MIN Thresshold"),
                                dcc.Slider(0.1, 0.9, step=0.1, value=0.7,
                                           id={'type': 'corr-slider', 'index': value[-1]},
                                           # marks={str(year): str(year) for year in df['year'].unique()},
                                           )]), width=6), ]),
                    dbc.Row([
                        dbc.Col(
                            dbc.Card([
                                html.Div([
                                    dbc.Label("X variable"),
                                    dcc.Dropdown(
                                        id={'type': 'x-variable', 'index': value[-1]},
                                        options=[{"label": col, "value": col} for col in df.columns],
                                        value=df.columns[0], ), ]),
                                html.Div([
                                    dbc.Label("Y variable"),
                                    dcc.Dropdown(id={'type': 'y-variable', 'index': value[-1]},
                                                 # options=[{"label": 'None', "value": 'None'}] + [{"label": col, "value": col} for col in df.columns],
                                                 options=[{"label": 'None', "value": 'None'}] + [
                                                     {'label': k, 'value': k} for k in df.columns if
                                                     attr_name_type[int(value[1:])][k] == 'numeric'],
                                                 value='None', ), ]),
                                html.Div([
                                    dbc.Label("Color Variable"),
                                    dcc.Dropdown(
                                        id={'type': 'color-variable', 'index': value[-1]},
                                        # options=[{"label": 'None', "value": 'None'}] + [{"label": col, "value": col} for col in df.columns],
                                        options=[{"label": 'None', "value": 'None'}] + [{'label': k, 'value': k} for k
                                                                                        in df.columns if
                                                                                        attr_name_type[int(value[1:])][
                                                                                            k] != 'numeric'],
                                        value='None', )]), ],
                                body=True, )),
                        dbc.Col(dcc.Graph(id={'type': 'indicator-graphic', 'index': value[-1]}, figure={}), md=9), ],
                        align="center",
                    ),
                    dbc.Row([
                        dbc.Col(
                            dbc.Card([
                                html.Div([
                                    dbc.Label("X variable"),
                                    dcc.Dropdown(
                                        id={'type': '3d-x-variable', 'index': value[-1]},
                                        options=[{"label": col, "value": col} for col in df_cluster.columns[1:]],
                                        value=df_cluster.columns[1], ), ]),
                                html.Div([
                                    dbc.Label("Y variable"),
                                    dcc.Dropdown(
                                        id={'type': '3d-y-variable', 'index': value[-1]},
                                        options=[{"label": 'None', "value": 'None'}] + [{"label": col, "value": col} for
                                                                                        col in df_cluster.columns[1:]],
                                        value=df_cluster.columns[2], ), ]),
                                html.Div([
                                    dbc.Label("Z Variable"), dcc.Dropdown(
                                        id={'type': '3d-z-variable', 'index': value[-1]},
                                        options=[{"label": 'None', "value": 'None'}] + [{"label": col, "value": col} for
                                                                                        col in df_cluster.columns[1:]],
                                        value=df_cluster.columns[3], )]), ],
                                body=True, )),
                        dbc.Col(dcc.Graph(id={'type': '3d-scatterplots', 'index': value[-1]}, figure={}), md=9), ],
                        align="center", ), ],
                fluid=True)
            return new_child

    # Button - Download table to csv  #
    @app.callback(
        Output("download-dataframe-csv", "data"),
        Input("btn_csv", "n_clicks"),
        Input("url", "pathname"),
        Input('my_table', 'selected_columns'),
        Input('my_table', 'derived_virtual_data'),
        prevent_initial_call=True,
    )
    def func(n_clicks, value, sel_col, data):
        if ctx.triggered_id != 'btn_csv':
            raise PreventUpdate
        if not data:
            dff = None
        else:
            if sel_col:
                for d in data:
                    for key in list(d):
                        if key not in sel_col:
                            del d[key]
            dff = pd.DataFrame(data)
        return dcc.send_data_frame(dff.to_csv, "mydf.csv", index=False)

    # Button - Insert table to POSTGRES Database System #
    @app.callback(
        Output("modal", "is_open"),
        [Input("insert-postgres", "n_clicks"), Input("close", "n_clicks"), Input('submit', 'n_clicks'),
         Input('input-user', 'value'), Input('input-password', 'value'), Input('input-port', 'value'),
         Input('input-database', 'value'), Input('input-path', 'value'),
         Input('my_table', 'derived_virtual_data'), Input('my_table', 'selected_columns'),
         Input("url", "pathname"), ],
        [State("modal", "is_open")],
    )
    def toggle_modal(nopen, nclose, nsubmit, user, passw, port, database, path, data, sel_col, value, is_open):
        if ctx.triggered_id in ["close", 'insert-postgres']:
            return not is_open
        elif ctx.triggered_id == 'submit':
            table_name = csv_file_name[:-4] + '_table' + value[1:]
            if sel_col:
                for d in data:
                    for key in list(d):
                        if key not in sel_col:
                            del d[key]
            header = {key: attr_name_type[int(value[1:])][key] for key in data[0].keys()}
            postgres(data, user, passw, port, database, path, table_name, header)
            return not is_open
        else:
            raise PreventUpdate
        return is_open

    # Button - Revert the changes applied on the table like filters, sorts, deletions etc. #
    @app.callback(
        Output("my_table", "data"), Output("my_table", "columns"),Output("my_table", "filter_query"),
        Input("reset-table", "n_clicks"), Input("url", "pathname"),)# Input('my_table', 'columns')])

    def reset_table(reset_click, value):
        if reset_click in [0, None]:
            raise PreventUpdate
        else:
            print (ctx.triggered_id )
        #if ctx.triggered_id == 'reset-table':
            #df = df_set[int(value[1:])]
            df = pd.concat([pd.Series(cluster_id[int(value[1:])], dtype=np.int64, name='Cluster'), df_set[int(value[1:])]], axis=1)
            #data = df.to_records() #
            data = df.to_dict('records')
            columns = [
                {'name': i, 'id': i,
                 'type': attr_name_type[int(value[1:])][i] if
                 attr_name_type[int(value[1:])][i] not in ['time',
                                                           'date'] else 'datetime',
                 'deletable': True,
                 # 'renamable': True,
                 'selectable': True,
                 } for i in df.columns if i != 'id']
            filters = ''
            return data ,columns, filters
            #return dash_table.DataTable(data=data, columns=columns, filter_query=filters)
        #else:
        #    raise PreventUpdate

    # 3D scatter plots - Clustered data #
    @app.callback(
        Output({'type': '3d-scatterplots', 'index': MATCH}, 'figure'),
        [Input(component_id={'type': '3d-x-variable', 'index': MATCH}, component_property='value'),
         Input(component_id={'type': '3d-y-variable', 'index': MATCH}, component_property='value'),
         Input(component_id={'type': '3d-z-variable', 'index': MATCH}, component_property='value'),
         Input("url", "pathname")])
    def update_graph(xaxis, yaxis, zaxis, value):
        if value == '/':
            raise PreventUpdate
        dff = df_set[int(value[1:])]
        cluster_id_ = cluster_id[int(value[1:])]
        fig = px.scatter_3d(dff, x=xaxis, y=yaxis, z=zaxis, color=cluster_id_, height=800)
        return fig

    # x, y, z must be different #
    def filter_options_cluster(v, c, value):
        """Disable option v"""
        if value == '/':
            raise PreventUpdate
        return [{"label": col, "value": col, "disabled": col == v or col == c} for col in
                df_set[int(value[1:])].columns]
        # functionality is the same for both dropdowns, so we reuse filter_options

    app.callback(Output(component_id={'type': '3d-x-variable', 'index': MATCH}, component_property='options'),
                 [Input(component_id={'type': '3d-y-variable', 'index': MATCH}, component_property='value'),
                  Input(component_id={'type': '3d-z-variable', 'index': MATCH}, component_property='value'),
                  Input("url", "pathname")], prevent_initial_call=True, )(filter_options_cluster)
    app.callback(Output(component_id={'type': '3d-y-variable', 'index': MATCH}, component_property='options'),
                 [Input(component_id={'type': '3d-z-variable', 'index': MATCH}, component_property='value'),
                  Input(component_id={'type': '3d-x-variable', 'index': MATCH}, component_property='value'),
                  Input("url", "pathname")], prevent_initial_call=True, )(filter_options_cluster)
    app.callback(Output(component_id={'type': '3d-z-variable', 'index': MATCH}, component_property='options'),
                 [Input(component_id={'type': '3d-y-variable', 'index': MATCH}, component_property='value'),
                  Input(component_id={'type': '3d-x-variable', 'index': MATCH}, component_property='value'),
                  Input("url", "pathname")], prevent_initial_call=True, )(filter_options_cluster)

    # Histogram - Bar Plot #
    # If y variable is None -> Histogram of variable x (y=count) #
    @app.callback(
        Output({'type': 'indicator-graphic', 'index': MATCH}, 'figure'),
        [Input(component_id={'type': 'x-variable', 'index': MATCH}, component_property='value'),
         Input(component_id={'type': 'y-variable', 'index': MATCH}, component_property='value'),
         Input(component_id={'type': 'color-variable', 'index': MATCH}, component_property='value'),
         Input("url", "pathname")])
    def update_graph(xaxis_column_name, yaxis_column_name, color_name, value):
        if value == '/':
            raise PreventUpdate
        dff = df_set[int(value[1:])]
        if yaxis_column_name == 'None':
            yaxis_column_name = None
        if color_name == 'None':
            color_name = None
        fig = px.histogram(dff, x=xaxis_column_name, y=yaxis_column_name, color=color_name)
        if attr_name_type[int(value[1:])][xaxis_column_name] == 'text':
            fig.update_xaxes(
                ticktext=tranc_names(list(np.unique(dff[xaxis_column_name][~dff[xaxis_column_name].isna()]))),
                tickvals=list(np.unique(dff[xaxis_column_name][~dff[xaxis_column_name].isna()])))
        if yaxis_column_name:
            if attr_name_type[int(value[1:])][yaxis_column_name] == 'text':
                fig.update_yaxes(
                    ticktext=tranc_names(list(np.unique(dff[yaxis_column_name][~dff[yaxis_column_name].isna()]))),
                    tickvals=list(np.unique(dff[yaxis_column_name][~dff[yaxis_column_name].isna()])))
                print('error')
        return fig

    # x, y, color must be different #
    def filter_options(v, c, value):
        """Disable option v"""
        if value == '/':
            raise PreventUpdate
        if ctx.outputs_list['id']['type'] == 'x-variable':
            return [{"label": col, "value": col, "disabled": col == v or col == c} for col in
                    df_set[int(value[1:])].columns]
        elif ctx.outputs_list['id']['type'] == 'y-variable':
            # print ('yvar')
            return [{"label": 'None', "value": 'None'}] + [
                {"label": col, "value": col, "disabled": col == v or col == c}
                for col in df_set[int(value[1:])].columns if attr_name_type[int(value[1:])][col] == 'numeric']
        else:
            # print ('cvar')
            return [{"label": 'None', "value": 'None'}] + [
                {"label": col, "value": col, "disabled": col == v or col == c}
                for col in df_set[int(value[1:])].columns if attr_name_type[int(value[1:])][col] != 'numeric']

    # functionality is the same for both dropdowns, so we reuse filter_options
    app.callback(Output(component_id={'type': 'x-variable', 'index': MATCH}, component_property='options'),
                 [Input(component_id={'type': 'y-variable', 'index': MATCH}, component_property='value'),
                  Input(component_id={'type': 'color-variable', 'index': MATCH}, component_property='value'),
                  Input("url", "pathname")], prevent_initial_call=True, )(filter_options)
    app.callback(Output(component_id={'type': 'y-variable', 'index': MATCH}, component_property='options'),
                 [Input(component_id={'type': 'x-variable', 'index': MATCH}, component_property='value'),
                  Input(component_id={'type': 'color-variable', 'index': MATCH}, component_property='value'),
                  Input("url", "pathname")], prevent_initial_call=True, )(filter_options)
    app.callback(Output(component_id={'type': 'color-variable', 'index': MATCH}, component_property='options'),
                 [Input(component_id={'type': 'x-variable', 'index': MATCH}, component_property='value'),
                  Input(component_id={'type': 'y-variable', 'index': MATCH}, component_property='value'),
                  Input("url", "pathname")], prevent_initial_call=True, )(filter_options)

    # Ecdf plot #
    @app.callback(
        [Output({'type': 'dynamic-graph2', 'index': MATCH}, 'figure'),
         Output({'type': 'dynamic-radio', 'index': MATCH}, 'value'),
         Output({'type': 'dynamic-radio', 'index': MATCH}, 'options')],
        [Input(component_id={'type': 'dynamic-radio', 'index': MATCH}, component_property='value'),
         Input("url", "pathname"),
         Input('my_table', 'derived_virtual_data')], prevent_initial_call=True)
    def update_graph2(ctg_value, value, data, ):
        if value == '/':
            raise PreventUpdate
        else:
            if not data:
                print('NOne Data!')
                print(ctx.triggered_id)
                fig = {}
            else:
                dff = pd.DataFrame(data)
                if ctg_value not in dff.columns:
                    print('not in df.columns')
                    ctg_value = [k for k in dff.columns if attr_name_type[int(value[1:])][k] != 'text'][0]
                fig = px.ecdf(dff, ctg_value, hover_data={ctg_value: False, ctg_value + '_': dff[ctg_value], })
                if attr_name_type[int(value[1:])][ctg_value] == 'text':
                    fig.update_xaxes(ticktext=tranc_names(list(np.unique(dff[ctg_value][~dff[ctg_value].isna()]))),
                                     tickvals=list(np.unique(dff[ctg_value][~dff[ctg_value].isna()]))),
                print(ctx.triggered_id)
                numeric_existed = [k for k in dff.columns if attr_name_type[int(value[1:])][k] != 'text']
                col = dff[numeric_existed].columns
            return fig, ctg_value, col

    # Scatter plots of variables with correlation larger than user defined threshold #
    @app.callback(
        Output({'type': 'scatter-plots', 'index': MATCH}, 'figure'),
        [Input(component_id={'type': 'corr-slider', 'index': MATCH}, component_property='value'),
         Input("url", "pathname"),
         Input('my_table', 'derived_virtual_data'), ])
    def update_graph3(thres, value, data):
        if value == '/':
            raise PreventUpdate
        else:
            df = df_set[int(value[1:])]
            df_corr = np.round(df.corr(), 2)
            attr1 = []
            attr2 = []
            for i in range(len(df_corr.columns)):
                for j in range(i + 1, len(df_corr.index)):
                    val = (df_corr.iloc[i, j])
                    if abs(val) > thres:
                        attr1.append(df_corr.columns[i])
                        attr2.append(df_corr.index[j])
            total_plots = len(attr1)
            if total_plots == 0:
                fig = {}
                return fig
            if total_plots < 15:
                cols = 3
            else:
                cols = 4
            rows = total_plots // cols + min(total_plots % cols, 1)
            fig = make_subplots(rows=rows,
                                cols=cols,
                                subplot_titles=(['Correlation: ' + str(df_corr.loc[attr1[ind], attr2[ind]])
                                                 for ind in range(len(attr1))]), )
            for ind in range(total_plots):
                fig.add_trace(
                    go.Scatter(
                        x=df[attr1[ind]],
                        y=df[attr2[ind]],
                        mode="markers"  # +text",
                        # text=["Text A", "Text B", "Text C"],
                        # textposition="bottom center"
                    ), row=ind // cols + 1, col=ind % cols + 1)
                fig.update_xaxes(title_text=attr1[ind], row=ind // cols + 1, col=ind % cols + 1, title_standoff=0.1)
                fig.update_yaxes(title_text=attr2[ind], row=ind // cols + 1, col=ind % cols + 1, title_standoff=0.1)
            fig.update_layout(title_text="Scatter Plots", height=rows * (300 - rows * 30),
                              showlegend=False)  # height=rows*230,
            return fig

    app.run_server(debug=True, )  # dev_tools_ui=False, dev_tools_props_check=False)


if __name__ == '__main__':
    print('Main')
    csv_file_name = ''
    df_set = ''
    create_dash(csv_file_name, df_set)
