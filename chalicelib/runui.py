# AAdhikari: 8/25/2018

import os 
import sys 
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as datatable
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime
from datetime import date
import plotly.tools as tls
from threading import Thread, current_thread
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import simplejson as json
import uuid
from collections import defaultdict
import requests
from flask import jsonify
import re

sys.path.append('.')
from uiapp import app
from uiforecasting import *
from uianalytics import *
from uiheatmap import *
from uidynamicpricing import *


def serve_layout():
	
	return html.Div(children=[
		
		html.Div(className="banner", children=[
			html.Div(className='container scalable', children=[
				html.H2(html.A(
					'Gozo Analytics & AI/ML Dashboard',					
					style={'text-decoration': 'none', 'color': 'inherit'}
				)),
				html.A(
					html.Img(
						src="https://www.gozocabs.com/images/logo2_new.png?v1.3", 
						style={
							'height':'8%', 'width':'8%', 'align': 'center', 'marginTop' : 15
						}
					),
					href='https://www.gozocabs.com/',					
				)
			]),
		]),
		dcc.Tabs(
			id="tabs", 
			value='dynamic-pricing', 
			children=[
				dcc.Tab(
					label='Forecasting Playground', 
					value='forecasting-playground',
				),
				dcc.Tab(
					label='Analytics Playground', 
					value='analytics-playground'
				),
				dcc.Tab(
					label='Heatmap Playground', 
					value='heatmap-playground'
				),
				dcc.Tab(
					label='Dynamic Pricing', 
					value='dynamic-pricing'
				),				
			],
			#vertical='vertical',
			style={
					'borderRight': 'thin lightgrey solid',
					'textAlign': 'left',					
				}			
		),
		html.Div(id='tabs-content'),
		html.Div(datatable.DataTable(rows=[{}]), style={'display': 'none'})
	])

app.layout = serve_layout


@app.callback(
	dash.dependencies.Output('tabs-content', 'children'),
	[
		dash.dependencies.Input('tabs', 'value')
	])
def render_content(tab):
	if tab == 'forecasting-playground':
		return forcasting_page_layout()
	elif tab == 'analytics-playground':
		return analytics_page_layout()
	elif tab == 'heatmap-playground':
		return heatmap_page_layout()
	elif tab == 'dynamic-pricing':
		return dynamicpricing_page_layout()
	elif tab == 'hawkeye-client':
		return price_monitor_layout()

external_css = [
    "https://cdnjs.cloudflare.com/ajax/libs/normalize/7.0.0/normalize.min.css",
    "https://fonts.googleapis.com/css?family=Open+Sans|Roboto",  # Fonts
    "https://maxcdn.bootstrapcdn.com/font-awesome/4.7.0/css/font-awesome.min.css",
    # Base Stylesheet
    "https://cdn.rawgit.com/xhlulu/9a6e89f418ee40d02b637a429a876aa9/raw/base-styles.css",
    # Custom Stylesheet
    "https://cdn.rawgit.com/plotly/dash-regression/98b5a541/custom-styles.css"
]
		
my_css_url="https://codepen.io/chriddyp/pen/bWLwgP.css"
#my_css_url="https://codepen.io/chriddyp/pen/brPBPO.css"
app.css.append_css({
	"external_url": external_css
})
	
if __name__ == '__main__':
	#app.run_server(debug=True, host='10.22.154.230')
	app.run_server(debug=True, host='127.0.0.1', port=8095)
