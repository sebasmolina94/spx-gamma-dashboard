
import dash
from dash import dcc, html, Input, Output
import yfinance as yf
import plotly.graph_objs as go
import pandas as pd
from datetime import datetime, timedelta

app = dash.Dash(__name__)
server = app.server

def fetch_options_data(ticker='^SPX'):
    data = yf.Ticker(ticker)
    expirations = data.options
    if not expirations:
        return pd.DataFrame(), [], None

    # Focus on 0DTE (nearest expiration)
    expiry = expirations[0]
    opt_chain = data.option_chain(expiry)
    calls = opt_chain.calls
    puts = opt_chain.puts

    calls['type'] = 'call'
    puts['type'] = 'put'

    options = pd.concat([calls, puts])
    options['gamma_exposure'] = options['openInterest'] * options['impliedVolatility']
    options['strike'] = options['strike'].astype(float)
    return options, data.history(period='1d', interval='1m'), expiry

def plot_dashboard(options, price_data, expiry):
    fig = go.Figure()

    # Plot historical price
    fig.add_trace(go.Scatter(
        x=price_data.index,
        y=price_data['Close'],
        mode='lines',
        name='SPX Price'
    ))

    # Plot gamma exposure by strike
    if not options.empty:
        grouped = options.groupby('strike').agg({
            'gamma_exposure': 'sum',
            'volume': 'sum'
        }).reset_index()

        fig.add_trace(go.Bar(
            x=grouped['strike'],
            y=grouped['gamma_exposure'],
            name='Gamma Exposure',
            yaxis='y2',
            marker_color='orange',
            opacity=0.6
        ))

        fig.add_trace(go.Bar(
            x=grouped['strike'],
            y=grouped['volume'],
            name='Volume per Strike',
            yaxis='y2',
            marker_color='blue',
            opacity=0.4
        ))

    # Configure axes
    fig.update_layout(
        title=f"SPX Gamma Exposure and Volume by Strike (Expiry: {expiry})",
        xaxis_title="Strike Price / Time",
        yaxis=dict(title="SPX Price", side='left'),
        yaxis2=dict(title="Exposure / Volume", overlaying='y', side='right', showgrid=False),
        legend=dict(x=0, y=1),
        margin=dict(l=40, r=40, t=80, b=40),
        height=600
    )
    return fig

@app.callback(
    Output('live-graph', 'figure'),
    Input('interval-component', 'n_intervals')
)
def update_graph(n):
    options, price_data, expiry = fetch_options_data()
    return plot_dashboard(options, price_data, expiry)

app.layout = html.Div([
    html.H1("Real-Time SPX Gamma Exposure Dashboard"),
    dcc.Graph(id='live-graph'),
    dcc.Interval(id='interval-component', interval=60*1000, n_intervals=0)
])

if __name__ == '__main__':
    app.run_server(debug=True)
