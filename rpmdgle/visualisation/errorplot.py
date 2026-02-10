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
import dash
from dash import Dash, dcc, html
from dash.dependencies import Input, Output, State
from rpmdgle.visualisation._freesocket import find_free_port, get_local_ip
import pandas as pd
import numpy as np
import os
import json

parser = argparse.ArgumentParser(description="Plot data from one or more files showing the confidence intervals.")

parser.add_argument('-x', '--xcol', type=int, default=0, help="Index of the column containing the x-data (zero-based).")
parser.add_argument('-y', '--ycol', type=int, default=1, help="Index of the column containing the y-data (zero-based).")
parser.add_argument('-e', '--ecol', type=int, default=2, help="Index of the column containing the standard deviations of the y-data (zero-based).")
parser.add_argument('--localnet', action="store_true", help="Deploy Dash App in the local network")
parser.add_argument('-s', '--separator', type=str, default='\s+', help="Separator character")
parser.add_argument('-H', '--header', type=int, default=None, help="Header line")
parser.add_argument('data', nargs='+', help="Data files.")

def main():

    args = parser.parse_args()
    n = len(args.data)
    vals = np.linspace(0.1, 0.9, n)
    clist = px.colors.sample_colorscale('viridis', list(vals))
    options = []
    dfs = []
    ymin, ymax = None, None
    xmin, xmax = None, None
    for c,d in zip(clist,args.data):
        df = pd.read_csv(
            d, sep=args.separator, 
            header=args.header,
            index_col=False,
            comment='#')
        dfs.append(df)
        ynt, yxt = np.min(df[args.ycol]), np.max(df[args.ycol])
        xnt, xxt = np.min(df[args.xcol]), np.max(df[args.xcol])
        ymin = ynt if ymin is None else min(ynt, ymin)
        ymax = yxt if ymax is None else max(yxt, ymax)
        xmin = xnt if xmin is None else min(xnt, xmin)
        xmax = xxt if xmax is None else max(xxt, xmax)
        options.append(
            {
                "label": html.Div([d], style={'color': c, 'font-size': 20, 'padding-left':'0.5em'}),
                "value": d,
            },
        )
    yspan = max(np.finfo(np.float64).tiny, ymax-ymin)
    pad = 0.1
    yrange_dflt = [ymin-yspan*(pad/2), ymax+yspan*(pad/2)]
    xrange_dflt = [xmin, xmax]

    app = Dash(__name__)

    # Define app layout
    app.layout = html.Div([
        html.Div([
            html.Div([
                dcc.Slider(
                    id='interval-slider',
                    min=0,
                    max=4,
                    step=0.25,
                    value=2,
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": True}
                    )
                ],  style={'padding-top':'2vh', 'padding-right':'15%', 'width': '75%', 'margin': '20px auto', 'horizontalAlign': 'left'}),
            dcc.Graph(
                id='graph', 
                style={
                    'display': 'inline-block',
                    'width': '80%', 'height': '85vh',
                    'verticalAlign': 'top'}),
            html.Div(id='hidden-div', style={'display': 'none'}),
            html.Div([
                dcc.Checklist(
                    id='series-checklist',
                    options=options,
                    value=[], # initially no confidence intervals are shown
                    inline=True,
                    labelStyle={"display": "flex", "align-items": "center"},
                )
            ], style={
                'display': 'inline-block', 
                'width': '15%', 
                'verticalAlign': 'top',
                'padding-left':'2%',
                'padding-top':'5vh'
                })
        ])
    ])

    @app.callback(
        Output('graph', 'figure'),
        [Input('series-checklist', 'value'),
         Input('interval-slider', 'value'),
         Input('graph', 'relayoutData')]
    )
    def update_graph(selected_series, interval_scale, relayout_data):
        ctx = dash.callback_context
        triggered_id = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
        if triggered_id in {'interval-slider','series-checklist'} or (
            triggered_id is None and relayout_data is None  # initial call
        ):
            # Preserve current x and y axis ranges
            try:
                x_range = [relayout_data[f'xaxis.range[{i}]'] for i in (0,1)]
            except:
                x_range = None
            try:
                y_range = [relayout_data[f'yaxis.range[{i}]'] for i in (0,1)]
            except:
                y_range = None
            # Default figure layout
            layout = go.Layout(
                xaxis=dict(
                    tickfont=dict(size=20), range=xrange_dflt  
                ),
                yaxis=dict(
                    tickfont=dict(size=20), range=yrange_dflt
                ),
                margin=dict(l=100, r=0, b=60, t=40, pad=4),
                plot_bgcolor='rgba(0,0,225,0.05)',  # Set background color of plotting area
            )
            # Restore previous x and y axis ranges
            if x_range is not None:
                layout['xaxis']['range'] = x_range
            if y_range is not None:
                layout['yaxis']['range'] = y_range

            # Produce plots
            go_list = []
            for c,df,f in zip(clist,dfs,args.data):
                go_list.extend([
                    go.Scatter(
                        x=df[args.xcol],
                        y=df[args.ycol],
                        mode='lines',
                        line=dict(color=c),
                        showlegend=False,
                        name=f"{d}"
                    ),
                go.Scatter(
                    x=df[args.xcol],
                    y=df[args.ycol]+interval_scale*df[args.ecol],
                    mode='lines',
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    showlegend=False,
                    visible=f in selected_series,
                    name=f"{d}: upper"
                ),  
                go.Scatter(
                    x=df[args.xcol],
                    y=df[args.ycol]-interval_scale*df[args.ecol],
                    mode='lines',
                    marker=dict(color="#444"),
                    line=dict(width=0),
                    fillcolor='rgba(0, 88, 255, 0.2)',
                    fill='tonexty',
                    showlegend=False,
                    visible=f in selected_series,
                    name=f"{d}: lower"
                )]  
                )

            return {'data': go_list, 'layout': layout}

        else:
            raise dash.exceptions.PreventUpdate

    if args.localnet:    
        ip='0.0.0.0'
        host = get_local_ip()
        port = find_free_port(ip=ip)
        if os.getenv("WERKZEUG_RUN_MAIN") != "true":
            app.logger.info(f"Other users on local network can connect via http://{host}:{port}/")
        app.run(debug=True, host=ip, port=port)
    else:
        app.run(debug=True, port=find_free_port())

 
if __name__ == "__main__":
    main()