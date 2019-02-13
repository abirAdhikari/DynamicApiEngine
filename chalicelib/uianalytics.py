# Author: AAdhikari - 06/09/2018 - first cut
# AAdhikari - 06/13/2018 - Added 1) URL column for quick validation 2) Added 'Show Only Lowest' optional
# AAdhikari - 06/14/2018 - Added 1) Hooked up Refresh, made other other option change serializable in jason
# AAdhikari - 06/15/2018 - Added 1) Hooked up date based query
 
import os 
import sys 
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as dt
import json
import pandas as pd
import numpy as np
import plotly
import logging
import sqlite3
from datetime import datetime, timedelta, date
import uuid
import requests

sys.path.append('.')
from uiapp import *
from debuglogger import *


def analytics_page_layout():
	
	analytics_session_id = str('ANALYTICS_SESSION')
	return html.Div(children=[
		html.H1(''),
		html.Div(children=
			[
				html.H6('Date Filter'),
				dcc.DatePickerRange(
					id='input_date_range',
					min_date_allowed=datetime(2015, 7, 1),
					max_date_allowed=datetime (2019, 12, 31),
					initial_visible_month=datetime(2018, 9, 30),
					start_date=datetime(2018, 1, 1),
					end_date=datetime(2018, 9, 30)
				),					
			], style={'width': '20%', 'height': '100%', 'display': 'inline-block'}
		),
		html.Div(children=
			[	
				html.Label('Hit refresh after making selections'),
				html.Div(
					
					html.Button('Refresh', 
						id='refresh-button',
						style={'backgroundColor' : 'gainsboro'}
					),
					style={'align': 'center', 'color' : 'blue', 'display': 'inline-block'}
				),
				
			], style={'width': '10%', 'height': '100%', 'display': 'inline-block'}
		),
		html.Div(children=[
				dt.DataTable(
					rows=[{}],
					row_selectable=True,
					filterable=True,
					sortable=True,
					min_height=600,
					#min_width=1500,
					selected_row_indices=[],
					id='analytics-datatable',
				), 
				html.Div(id='selected-indexes'),
				dcc.Graph(
					id='analytics-graph'
				),
			],
			style={'align': 'right', 'width': '100%', 'height': '100%', 'resizable': 'True'}
		),
		
		html.Div(id='intermediate-value-refresh', style={'display': 'none'}),
		html.Div(analytics_session_id, id='analytics-session-id', style={'display': 'none'}),
	])
	
@app.callback(
	dash.dependencies.Output('intermediate-value-refresh', 'children'),
	[
		dash.dependencies.Input('analytics-session-id', 'children'),
		dash.dependencies.Input('input_date_range', 'start_date'),
		dash.dependencies.Input('input_date_range', 'end_date'),
	])
def update_controls(
					in_analytics_session_id,					  
					in_start_date,
					in_end_date,
					):
	_controls_to_json = json.dumps(
		{
			'start_date'				: in_start_date,
			'end_date'					: in_end_date, 
		}
	)
	logging.log(logging.INFO, 'UIAnalytics::.update_controls(): Enter')
	global_session_dict[in_analytics_session_id] = _controls_to_json
	logging.log(logging.INFO, 'UIAnalytics::.update_controls(): Exit')
	return _controls_to_json

	
@app.callback(
	dash.dependencies.Output('analytics-datatable', 'rows'),
	[
		dash.dependencies.Input('analytics-session-id', 'children'),
		dash.dependencies.Input('refresh-button', 'n_clicks'),
	]
)
def update_from_database(
		in_analytics_session_id, 
		n_clicks
	):
	
	if n_clicks == None:
		return
		
	logging.log(logging.INFO, "UIAnalytics::update_from_database(): Enter")	
	_json_data = global_session_dict[in_analytics_session_id]
	if _json_data == None or _json_data == '':
		logging.log(logging.INFO, "UIAnalytics::update_from_database(): controls date not initialized")
		rows=[{}]		
		return rows
	
	print_seperator_start('')
	headers = {"content-type": "application/json", "Authorization": "None" }
	url = aws_endpoint_url + 'analytics?query=' + _json_data
	print (url)
	r = requests.get(url, headers=headers)
	print (r.status_code)
	if r.status_code == 403 or r.status_code == 405:
		return graphs	
	print(r)
	
	
	results = json.loads(r.text)
	analytics_data = results["analytics_data"]	
	df_json = json.loads(analytics_data)
	df = pd.read_json(df_json['df'], convert_dates=['bkg_create_date', 'bkg_pickup_date'])
	print_dataframe(df, 'Analytics Dataframe', False)
	logging.log(logging.INFO, "UIAnalytics::update_from_database()")
	if len(df) == 0:
		rows=[{}]
		logging.log(logging.INFO, "UIAnalytics::update_from_database(): Exit - Empty Rows")
		return rows
	logging.log(logging.INFO, "UIAnalytics::update_from_database(): Exit")
	return df.to_dict('records')

@app.callback(
	dash.dependencies.Output('analytics-graph', 'figure'),
	[
		dash.dependencies.Input('analytics-session-id', 'children'),
		dash.dependencies.Input('analytics-datatable', 'rows'),
		dash.dependencies.Input('analytics-datatable', 'selected_row_indices')
	]
)
def update_figure(in_analytics_session_id, in_rows, in_selected_row_indices):
	logging.log(logging.INFO, "UIAnalytics::update_figure(): Enter")
	
	_json_data = global_session_dict[in_analytics_session_id]
	if _json_data == None or _json_data == '':
		logging.log(logging.INFO, "UIAnalytics::update_from_database(): controls date not initialized")
		return None
		
	filtered_df = pd.DataFrame(in_rows)	
	
	if len(filtered_df) == 0:
		logging.log(logging.INFO, "UIAnalytics::update_figure(): Exit - Empty DataFrame")
		return None
	
	filtered_df.insert(0, 'Count', 1)
	df_count	= filtered_df.groupby('Date', as_index=False).count()
	df_sum		= filtered_df.groupby('Date', as_index=False).sum()	
	
	fig = plotly.tools.make_subplots(
		rows=3, cols=1,
		subplot_titles=('Booking Trend', 'Total Revenue Trend', 'Gozo Revenue Trend'),
		shared_xaxes=True)
	marker = {'color': ['#0074D9']*len(filtered_df)}
	for i in (in_selected_row_indices or []):
		marker['color'][i] = '#FF851B'
	
	fig.append_trace({
		'x': df_count['Date'],
		'y': df_count['Count'],
		'type': 'bar',
		'marker': marker
	}, 1, 1)
	
	fig.append_trace({
		'x': df_sum['Date'],
		'y': df_sum['bkg_total_amount'],
		'type': 'bar',
		'marker': marker
	}, 2, 1)
	
	fig.append_trace({
		'x': df_sum['Date'],
		'y': df_sum['bkg_gozo_amount'],
		'type': 'bar',
		'marker': marker
	}, 3, 1)
	fig['layout']['showlegend'] = False
	fig['layout']['height'] = 800
	fig['layout']['margin'] = {
		'l': 40,
		'r': 10,
		't': 60,
		'b': 200
	}
	fig['layout']['yaxis3']['type'] = 'log'
	logging.log(logging.INFO, "UIAnalytics::update_figure(): Exit")
	return fig

def get_dummy_figure():
	fig = plotly.tools.make_subplots(
		rows=0, cols=0)
	fig['layout']['display'] = None
	return fig
		

'''
Removing this for now - Bad performance, chrome gets stuck with dataset of 100K

@app.callback(
	dash.dependencies.Output('analytics-datatable', 'selected_row_indices'),
	[
		dash.dependencies.Input('analytics-session-id', 'children'),
		dash.dependencies.Input('analytics-graph', 'clickData')
	],
	[
		dash.dependencies.State('analytics-datatable', 'selected_row_indices')
	]
)
def update_selected_row_indices(
		in_analytics_session_id, 
		in_click_data, 
		in_selected_row_indices):
	
	logging.log(logging.INFO, "UIAnalytics::update_selected_row_indices(): Enter")
	
	_json_data = global_session_dict[in_analytics_session_id]
	if _json_data == None or _json_data == '':
		logging.log(logging.INFO, "UIAnalytics::update_selected_row_indices(): controls date not initialized")
		rows=[{}]		
		return 
		
	if in_click_data:
		for point in in_click_data['points']:
			if point['pointNumber'] in in_selected_row_indices:
				in_selected_row_indices.remove(point['pointNumber'])
			else:
				in_selected_row_indices.append(point['pointNumber'])
	logging.log(logging.INFO, "UIAnalytics::update_selected_row_indices(): Exit")
	return in_selected_row_indices
'''
