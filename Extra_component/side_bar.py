from dash import Dash, dcc, html
import pandas as pd
from plotly import express as px
from dash.dependencies import Input, Output, State, MATCH, ALL
import dash_bootstrap_components as dbc


app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

df_bar =[
    pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
}),
pd.DataFrame({
    "Fruit2": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount2": [4, 1, 2, 2, 4, 5],
    "City2": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]}),

    ]

df_attr =[list(df_bar[0].columns), list(df_bar[1].columns)]
app.layout = html.Div([
    dbc.Col(html.Div(
    dcc.RadioItems( options = ['name'+str(i) for i in range(len(df_bar))],
        value='name0',
        id='table-radio',
    )),width=2),
    dbc.Col(html.Div(id='container'))
])

@app.callback(
    Output('container','children'),
    [Input('table-radio','value')]
)

def display_graphs(value):
    new_child = html.Div([
        dcc.Graph(id={
                    'type': 'dynamic-graph',
                    'index': value[-1]
                },
                figure={}),
        dcc.Dropdown(
            id={
                'type': 'dynamic-dpn-ctg',
                'index': value[-1]
            },
            options=df_attr[0],
            value=df_attr[0][0],#       clearable=False
        ),
    ])
    return new_child

@app.callback(
    Output({'type': 'dynamic-graph', 'index': MATCH}, 'figure'),
     Input(component_id={'type': 'dynamic-dpn-ctg', 'index': MATCH}, component_property='value'),
)
def update_graph(ctg_value):
    print(s_value)
    dff = df[df['state'].isin(s_value)]

    if chart_choice == 'bar':
        dff = dff.groupby([ctg_value], as_index=False)[['detenues', 'under trial', 'convicts', 'others']].sum()
        fig = px.bar(dff, x=ctg_value, y=num_value)
        return fig
    elif chart_choice == 'line':
        if len(s_value) == 0:
            return {}
        else:
            dff = dff.groupby([ctg_value, 'year'], as_index=False)[['detenues', 'under trial', 'convicts', 'others']].sum()
            fig = px.line(dff, x='year', y=num_value, color=ctg_value)
            return fig
    elif chart_choice == 'pie':
        fig = px.pie(dff, names=ctg_value, values=num_value)
        return fig



if __name__ == "__main__":
    app.run_server(debug=True)