#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   confidence_plot.py
@Time    :   2024/03/22 16:17:42
@Author  :   George Trenins
@Contact :   gstrenin@gmail.com
@Desc    :   None
'''


from __future__ import print_function, division, absolute_import
import argparse
import plotly.graph_objs as go
import plotly.express as px
from dash import Dash, dcc, html
from rpmdgle.visualisation._freesocket import find_free_port
import pandas as pd
import numpy as np
from dash.dependencies import Input, Output
import plotly.io as pio

parser = argparse.ArgumentParser(description="Plot data from one or more files showing the confidence intervals.")

parser.add_argument('-x', '--xcol', type=int, default=0, help="Index of the column containing the x-data (zero-based).")
parser.add_argument('-y', '--ycol', type=str, default="1:", help="Index of the column containing the y-data (zero-based).")
parser.add_argument('-s', '--separator', type=str, default='\s+', help="Separator character")
parser.add_argument('-H', '--header', type=int, default=None, help="Header line")
parser.add_argument('data', nargs='+', help="Data files.")

def string_to_slice(s: str) -> slice:
    if s == ":":
        slc = slice(None)
    else:
        try:
            slc = int(s)
        except ValueError:
            slc = slice(*[int(s) if s else None for s in s.split(":")])
    return slc


def main():

    args = parser.parse_args()
    n = len(args.data)
    vals = np.linspace(0.1, 0.9, n)
    clist = px.colors.sample_colorscale('viridis', list(vals))
    go_list = []
    slc = string_to_slice(args.ycol)
    for c,d in zip(clist,args.data):
        df = pd.read_csv(
            d, sep=args.separator, 
            header=args.header,
            index_col=False,
            comment='#')
        ncols = df.shape[1]
        if isinstance(slc, int):
            cols = [slc]
        else:
            cols = range(ncols)[slc] 
        for y in cols:
            go_list.append(
                go.Scatter(
                    # hoverinfo='text',
                    # hovertext=d,
                    x=df.iloc[:,args.xcol],
                    y=df.iloc[:,y],
                    mode='lines',
                    #line=dict(color='rgb(31, 119, 180)'),
                    line=dict(color=c),
                    name=f"{d}.{y}",
                    # showlegend=False
                )
            )

    fig = go.Figure(go_list)
    fig.update_layout(
        margin=dict(l=100, r=0, b=60, t=40, pad=4),
        yaxis = dict( tickfont = dict(size=32)),
        xaxis = dict( tickfont = dict(size=32))
    )
    
    app = Dash(__name__)

    app.layout = html.Div([
        dcc.Graph(id="graph", figure=fig, style={'width': '100%', 'height': '95vh'}),
        html.Button("Save as EPS", id='save-button')
    ])

    # Define a callback function to handle button clicks
    @app.callback(
        Output('save-button', 'n_clicks'),
        Input('save-button', 'n_clicks'),
        prevent_initial_call=True
    )
    def save_graph_as_eps(n_clicks):
        if n_clicks is not None:
            # Define the filename for the EPS file
            filename = "plot.eps"

            # Convert the plotly figure to an EPS file
            pio.write_image(fig, filename, format='eps')

            # Reset the button's click count
            return None


    
    app.run(debug=True, port=find_free_port())


if __name__ == "__main__":
    main()