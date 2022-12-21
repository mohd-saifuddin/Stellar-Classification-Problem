import warnings
warnings.filterwarnings('ignore')

from dash import Dash, dcc, html, Input, Output
from mlpipeline import Pipline as pipe


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=None)

app.layout = html.Div(children=[
    dcc.Slider(id='alpha', min=0, max=300, value=150,
        marks={
            0: {'label': '0', 'style': {'color': '#77b0b1'}},
            300: {'label': '300', 'style': {'color': '#f50'}}
        }
    ),
    dcc.Slider(id='delta', min=0, max=300, value=150,
        marks={
            0: {'label': '0', 'style': {'color': '#77b0b1'}},
            300: {'label': '300', 'style': {'color': '#f50'}}
        }
    ),
    dcc.Slider(id='u', min=0, max=30, value=15,
        marks={
            0: {'label': '0', 'style': {'color': '#77b0b1'}},
            30: {'label': '30', 'style': {'color': '#f50'}}
        }
    ),
    dcc.Slider(id='g', min=0, max=30, value=10,
        marks={
            0: {'label': '0', 'style': {'color': '#77b0b1'}},
            30: {'label': '30', 'style': {'color': '#f50'}}
        }
    ),
    dcc.Slider(id='r', min=0, max=30, value=10,
        marks={
            0: {'label': '0', 'style': {'color': '#77b0b1'}},
            30: {'label': '30', 'style': {'color': '#f50'}}
        }
    ),
    dcc.Slider(id='i', min=0, max=30, value=10,
        marks={
            0: {'label': '0', 'style': {'color': '#77b0b1'}},
            30: {'label': '30', 'style': {'color': '#f50'}}
        }
    ),
    dcc.Slider(id='z', min=0, max=30, value=10,
        marks={
            0: {'label': '0', 'style': {'color': '#77b0b1'}},
            30: {'label': '30', 'style': {'color': '#f50'}}
        }
    ),
    dcc.Slider(id='redshift', min=0, max=0.5, value=0.1,
        marks={
            0: {'label': '0', 'style': {'color': '#77b0b1'}},
            0.5: {'label': '0.5', 'style': {'color': '#f50'}}
        }
    ),
    html.Div(id='slider-output-container')
])

@app.callback(
    Output(component_id='slider-output-container', component_property='children'),
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
    ml = pipe(data=data)
    conclusion = ml.ml_pipeline()
    return conclusion


if __name__ == '__main__':
    app.run_server(debug=True)
