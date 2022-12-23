import warnings
warnings.filterwarnings('ignore')

from dash import Dash, dcc, html, Input, Output
from mlpipeline import Pipline


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)
app.title = "Stellar Classification"

app.layout = html.Div(children=[
    html.Meta(charSet='UTF-8'),
    html.Meta(name='viewport', content='width=device-width, initial-scale=1.0'),
    html.Div([html.H4(children=app.title)], 
        style={'textAlign': 'center', 'backgroundColor': '#D6EAF8'}
    ),
    html.Div(children=[
        html.Div(children=[
            html.Div(children='Ascension angle', style={'fontSize': 15}),
            dcc.Slider(
                id='alpha',
                min=0,
                max=360,
                value=150,
                marks={
                    0: {'label': '0', 'style': {'color': '#17202A'}},
                    360: {'label': '360', 'style': {'color': '#17202A'}}
                },
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Div(children='Declination angle', style={'fontSize': 15}),
            dcc.Slider(
                id='delta',
                min=0,
                max=360,
                value=150,
                marks={
                    0: {'label': '0', 'style': {'color': '#17202A'}},
                    360: {'label': '360', 'style': {'color': '#17202A'}}
                },
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Div(children='Ultraviolet', style={'fontSize': 15}),
            dcc.Slider(
                id='u',
                min=0,
                max=30,
                value=23,
                marks={
                    0: {'label': '0', 'style': {'color': '#17202A'}},
                    30: {'label': '30', 'style': {'color': '#17202A'}}
                },
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Div(children='Green', style={'fontSize': 15}),
            dcc.Slider(
                id='g',
                min=0,
                max=30,
                value=10,
                marks={
                    0: {'label': '0', 'style': {'color': '#17202A'}},
                    30: {'label': '30', 'style': {'color': '#17202A'}}
                },
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Div(children='Red', style={'fontSize': 15}),
            dcc.Slider(
                id='r',
                min=0,
                max=30,
                value=18,
                marks={
                    0: {'label': '0', 'style': {'color': '#17202A'}},
                    30: {'label': '30', 'style': {'color': '#17202A'}}
                },
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Div(children='Infrared (I)', style={'fontSize': 15}),
            dcc.Slider(
                id='i',
                min=0,
                max=30,
                value=18,
                marks={
                    0: {'label': '0', 'style': {'color': '#17202A'}},
                    30: {'label': '30', 'style': {'color': '#17202A'}}
                },
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Div(children='Infrared (Z)', style={'fontSize': 15}),
            dcc.Slider(
                id='z',
                min=0,
                max=30,
                value=18,
                marks={
                    0: {'label': '0', 'style': {'color': '#17202A'}},
                    30: {'label': '30', 'style': {'color': '#17202A'}}
                },
                tooltip={"placement": "bottom", "always_visible": True}
            ),
            html.Div(children='Redshift', style={'fontSize': 15}),
            dcc.Slider(
                id='redshift',
                min=0,
                max=10,
                value=2.5,
                marks={
                    0: {'label': '0', 'style': {'color': '#17202A'}},
                    10: {'label': '10', 'style': {'color': '#17202A'}}
                },
                tooltip={"placement": "bottom", "always_visible": True}
            ),
        ], className='six columns'),
        html.Div(id='image-output-container', className='six columns'),
    ], className='row'),
    html.Br(),
    html.Div(children=[
        html.Div(
            id='slider-output-container',
            style={'textAlign': 'center', 'fontSize': 16},
            className='six columns'
        )
    ], className='row'),
    html.Br(),
    html.Div([html.P(children='Stellar Classification')], 
        style={'textAlign': 'center', 'backgroundColor': '#D6EAF8', 'color': '#D6EAF8'}
    ),
])

@app.callback(
    [
        Output(component_id='slider-output-container', component_property='children'),
        Output(component_id='image-output-container', component_property='children')
    ],
    [
        Input(component_id='alpha', component_property='value'),
        Input(component_id='delta', component_property='value'),
        Input(component_id='u', component_property='value'),
        Input(component_id='g', component_property='value'),
        Input(component_id='r', component_property='value'),
        Input(component_id='i', component_property='value'),
        Input(component_id='z', component_property='value'),
        Input(component_id='redshift', component_property='value')
    ]
)
def ml_application(alpha, delta, u, g, r, i, z, redshift):
    data = [[alpha, delta, u, g, r, i, z, redshift]]
    pipe = Pipline(data=data)
    conclusion, fig = pipe.pipeline()
    return conclusion, dcc.Graph(figure=fig)


if __name__ == '__main__':
    app.run_server(debug=True)
