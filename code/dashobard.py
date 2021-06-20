import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.express as px
import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
PATH = '../data/train.csv'

train = pd.read_csv(PATH,index_col=0)


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame({
    "Fruit": ["Apples", "Oranges", "Bananas", "Apples", "Oranges", "Bananas"],
    "Amount": [4, 1, 2, 2, 4, 5],
    "City": ["SF", "SF", "SF", "Montreal", "Montreal", "Montreal"]
})

#df = train[['YearBuilt','SalePrice']].groupby('YearBuilt')['SalePrice'].count()
fig = px.bar(df, x="Fruit", y="Amount", color="City", barmode="group")
#fig = px.bar(df)

app.layout = html.Div(children=[
    html.H1(children='Hello world from Dash Framework'),

    html.Div(children='''
        Dash: A web application really framework for Python.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)