# Author: AAdhikari - 06/24/2018 - first cut
 
import os 
import sys 
import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import dash_table_experiments as datatable
import json
import pandas as pd
import numpy as np
import plotly
import logging
import sqlite3
from datetime import datetime, timedelta, date
from collections import defaultdict
import uuid
import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import requests

sys.path.append('.')
from uiapp import *
from heatmappipeline import *


def heatmap_page_layout():
	
	heatmap_session_id = str('HEATMAP_SESSION')
	return html.Div(children=[
		html.Div(children=[	
			html.Div(children=
				[
					html.H6('Date Filter'),
					dcc.DatePickerRange(
						id='input_date_range',
						min_date_allowed=datetime(2015, 1, 1),
						max_date_allowed=(2019, 12, 12),
						initial_visible_month=datetime(2018, 9, 30),
						start_date=datetime(2018, 9, 15),
						end_date=datetime(2018, 9, 30)
					),					
				], style={'width': '20%', 'height': '50%', 'display': 'inline-block'}
			),
			html.Div(children=
				[
					html.H6(''),
									
				], style={'width': '1%', 'height': '50%', 'display': 'inline-block'}
			),
			html.Div(children=
				[
					html.H6('Input Field'),
					dcc.Dropdown(
						id='primary_input_field',
						options=[{'label': i, 'value': i} for i in ['Oneway', 'Roundtrips', 'Oneway Returns']],
						value='Oneway',						 
					),
				], style={'width': '20%', 'height': '50%', 'display': 'inline-block'}
			),
			html.Div(children=
				[
					html.H6(''),
									
				], style={'width': '1%', 'height': '50%', 'display': 'inline-block'}
			),
			html.Div(children=
				[
					html.H6('View Types'), 
					dcc.Dropdown(
						id='view_type',
						options=[{'label': i, 'value': i} for i in ['Show Bubbels', 'Show Routes', 'Show Both']],
						value='Show Both',
					),
				], style={'width': '20%', 'height': '50%', 'display': 'inline-block'}
			),
			html.Div(children=
				[
					html.H6(''),
									
				], style={'width': '10%', 'height': '50%', 'display': 'inline-block'}
			),
			html.Div(children=
				[	
					html.Label('Hit Show after making selections'),
					html.Div(
						html.Button('Show', 
							id='show_button',
							style={'backgroundColor' : 'gainsboro'}
						),						
					),
				], style={'width': '10%', 'height': '50%', 'display': 'inline-block'}
			),
			dcc.Graph(
				id='heatmap_graph'
			),					
			html.Div(children=[
					datatable.DataTable(
						rows=[{}],
						row_selectable=True,
						filterable=True,
						sortable=True,
						min_height=600,
						selected_row_indices=[],
						id='heatmap_datatable',
					), 
					html.Div(id='selected_indexes'),					
				],
				style={'align': 'right', 'width': '100%', 'height': '100%', 'resizable': 'True'}
			),
		]),
		html.Div(id='intermediate_show', style={'display': 'none'}),
		html.Div(heatmap_session_id, id='heatmap_session_id', style={'display': 'none'}),
	])
	

@app.callback(
	dash.dependencies.Output('intermediate_show', 'children'),
	[
		dash.dependencies.Input('heatmap_session_id', 'children'),
		dash.dependencies.Input('primary_input_field', 'value'),
		dash.dependencies.Input('view_type', 'value'),
		dash.dependencies.Input('input_date_range', 'start_date'),
		dash.dependencies.Input('input_date_range', 'end_date'),
	])
def update_controls(
					in_heatmap_session_id,
					in_primary_input_field,
					in_view_type,
					in_start_date,
					in_end_date,
					):
	_controls_to_json = json.dumps(
		{
			'primary_input_field': in_primary_input_field, 
			'view_type'		: in_view_type, 
			'start_date'	: in_start_date,
			'end_date'		: in_end_date, 			
		}
	)
	print ("UIHeatmap::update_controls() - Enter")
	logging.log(logging.INFO, 'UIHeatmap::.update_controls(): Enter')
	global_session_dict[in_heatmap_session_id] = _controls_to_json
	logging.log(logging.INFO, 'UIHeatmap::.update_controls(): Exit')
	print ("UIHeatmap::update_controls() - Enter")
	return _controls_to_json


	
@app.callback(
	dash.dependencies.Output('heatmap_datatable', 'rows'),
	[
		dash.dependencies.Input('heatmap_session_id', 'children'),
		dash.dependencies.Input('show_button', 'n_clicks'),
	]
)
def update_from_database(in_heatmap_session_id, n_clicks):
	'''
	if n_clicks == None:
		return
	_json_data = global_session_dict[in_heatmap_session_id]
	if _json_data == None or _json_data == '':
		rows=[{}]	
		logging.log(logging.INFO, "UIHeatmap::update_from_database(): Uninitialized controls")	
		return rows
	
	logging.log(logging.INFO, "UIHeatmap::update_from_database(): Enter")	
	df = get_heatmap_json_wrapper(_json_data)
	logging.log(logging.INFO, "UIHeatmap::update_from_database(): Dataframe Length: %s", len(df))
	
	if len(df) == 0:
		rows=[{}]
		return rows
		
	logging.log(logging.INFO, "UIHeatmap::update_from_database(): Exit")	
	return df.to_dict('records')
	'''
	if n_clicks == None:
		return
		
	logging.log(logging.INFO, "UIHeatmap::update_from_database(): Enter")	
	_json_data = global_session_dict[in_heatmap_session_id]
	if _json_data == None or _json_data == '':
		logging.log(logging.INFO, "UIHeatmap::update_from_database(): controls date not initialized")
		rows=[{}]		
		return rows	
	
	headers = {"content-type": "application/json", "Authorization": "None" }
	url = aws_endpoint_url + 'heatmap?query=' + _json_data
	print (url)
	r = requests.get(url, headers=headers)
	print (r.status_code)
	if r.status_code == 403 or r.status_code == 405:
		return 	
	print(r)	
	
	results = json.loads(r.text)
	heatmap_data = results["heatmap_data"]	
	df_json = json.loads(heatmap_data)
	df = pd.read_json(df_json['df'], convert_dates=['bkg_create_date', 'bkg_pickup_date'])
	print_dataframe(df, 'Heatmap Dataframe', False)
	logging.log(logging.INFO, "UIHeatmap::update_from_database()")
	if len(df) == 0:
		rows=[{}]
		logging.log(logging.INFO, "UIHeatmap::update_from_database(): Exit - Empty Rows")
		return rows
	logging.log(logging.INFO, "UIHeatmap::update_from_database(): Exit")
	return df.to_dict('records')

@app.callback(
	dash.dependencies.Output('heatmap_graph', 'figure'),
	[
		dash.dependencies.Input('heatmap_session_id', 'children'),
		dash.dependencies.Input('show_button', 'n_clicks'),
		dash.dependencies.Input('heatmap_datatable', 'rows'),
		dash.dependencies.Input('heatmap_datatable', 'selected_row_indices')
	]
)
def update_figure(in_heatmap_session_id, n_clicks, in_rows, in_selected_row_indices):

	_json_data = global_session_dict[in_heatmap_session_id]
	if _json_data == None or _json_data == '':
		rows=[{}]	
		logging.log(logging.INFO, "UIHeatmap::update_figure(): Uninitialized controls")	
		return rows
	
	logging.log(logging.INFO, "UIHeatmap::update_figure(): Enter")
	_json_to_controls=json.loads(_json_data)
	
	df = pd.DataFrame(in_rows)		
	df.insert(0, 'Count', 1)
	view_type = _json_to_controls['view_type']

	cities = []
	#if view_type == 'Show All' or view_type == 'Show Bubbels':
	df_cities	= df.groupby(['bkg_from_city_latitude', 'bkg_from_city_longitude', 'bkg_from_city'], 
		as_index=False).count().sort_values(['Count'], ascending=False)
	df_cities['text'] = df_cities['bkg_from_city'] + ': ' + df_cities['Count'].apply(lambda x: str(x))
	
	scale = df_cities['Count'].max()
	
	limits = [(0,5),(6,15),(16,35),(36,75),(76,3000)]
	colors = ["rgb(255,65,54)", "rgb(0,116,217)", "rgb(133,20,75)", "rgb(255,133,27)", "lightgrey"]
	
	cities = []
	for i in range(len(limits)):
		lim = limits[i]
		df_sub = df_cities[lim[0]:lim[1]]
		city = dict(
			type = 'scattergeo',
			locationmode = 'india',
			lon = df_sub['bkg_from_city_longitude'],
			lat = df_sub['bkg_from_city_latitude'],
			text = df_sub['text'],
			mode = 'markers',
			marker = dict(
				reversescale = True,
				size = df_sub['Count']/scale * (500 / (i+1)),
				width = 1,
				color = colors[i],
				line = dict(width=0.5, color='rgb(40,40,40)'),
				sizemode = 'area'
			),
			name = '{0} - {1}'.format(lim[0],lim[1]) 
		)
		cities.append(city)
	routes = []
	#if view_type == 'Show All' or view_type == 'Show Routes':	
	df_routes	= df.groupby(['bkg_from_city_latitude', 'bkg_from_city_longitude', 'bkg_from_city', 
			'bkg_to_city_latitude', 'bkg_to_city_longitude', 'bkg_to_city'], 
			as_index=False).count()
	df_routes['text'] = 'From: ' + df_routes['bkg_from_city']\
			+ ', To: ' + df_routes['bkg_to_city'] +\
			', Count: '+ df_routes['Count'].apply(lambda x: str(x))
	
	
	df_routes = df_routes.sort_values(by='Count', ascending=False)
	
	for i in range( len( df_routes ) ):
		routes.append(
			dict(
				type = 'scattergeo',
				locationmode = 'india',
				lon = [df_routes['bkg_from_city_longitude'][i], df_routes['bkg_to_city_longitude'][i]],
				lat = [df_routes['bkg_from_city_latitude'][i], df_routes['bkg_to_city_latitude'][i]],
				text = df_routes['text'][i],
				mode = 'lines',
				line = dict(
					reversescale = True,
					width = 2,
					color = 'red',
				),
				opacity = float(df_routes['Count'][i])/float(df_routes['Count'].max()),
				name = '{0} - {1} : {2}'.format(
					df_routes['bkg_from_city'][i],df_routes['bkg_to_city'][i], df_routes['Count'][i])
			)
		)
	

	title = _json_to_controls['primary_input_field']
	
	layout = dict(
        autosize=False,
		width=1500,
		height=1200,
        title = title + ' Heatmap',
		geo = dict(
		    scope='india',
            #projection=dict( type = 'Mercator' ),
			projection=dict( type = 'azimuthal equal area' ),
            showland = True,
            landcolor = "rgb(250, 250, 250)",
            subunitcolor = "rgb(217, 217, 217)",
            countrycolor = "rgb(217, 217, 217)",
            countrywidth = 0.5,
            subunitwidth = 0.5,
			lonaxis = dict( range= [ 63.5, 97.25 ] ),
			lataxis = dict( range= [ 6.44, 35.30 ] ),
        ),
	)


	fig = dict( data=cities+routes, layout=layout )
	logging.log(logging.INFO, "UIHeatmap::update_figure(): Exit")

	return fig
	

