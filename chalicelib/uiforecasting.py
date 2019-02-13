

import os 
import sys 
import dash
import dash_core_components as dcc
import dash_html_components as html
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
import requests
from flask import jsonify
import re

sys.path.append('.')
from uiapp import *
from forecastingpipeline import *
from prophetmodeller import *

use_rest_api = True
cache_dataframe = None

df_base = BaseDataframe()
df_gozo = df_base.gozo_cities_df
cities = df_gozo['cty_name'].tolist()


def forcasting_page_layout():
	
	forecast_session_id = str('FORECASTING_SESSION')
	
	return html.Div(children=[
		html.Div(children=[		
			html.Div(children=
				[
					html.Div(children=
						[
							html.H6('Primary Input Field'),
							dcc.Dropdown(
								id='primary_input_field',
								options=[{'label': i, 'value': i} for i in ['Booking Count', 'Total Amount', 'Gozo Amount']],
								value='Booking Count',						 
							),
						], style={'width': '75%', 'height': '50%', 'display': 'inline-block'}
					),
					
					html.Div(children=
						[
							html.H6('Date Range To Train Forecasting Model'),
							dcc.DatePickerRange(
								id='input_date_range',
								min_date_allowed=datetime(2015, 7, 1),
								max_date_allowed=datetime (2019, 12, 31),
								initial_visible_month=datetime(2018, 9, 30),
								start_date=datetime(2015, 7, 1),
								end_date=datetime(2018, 9, 30)
							),
						], style={'width': '75%', 'height': '50%', 'display': 'inline-block'}
					),
			
					html.Div(children=
						[
							html.H6('Click Submit after making all selections'),
							html.Div(children=
								[
									html.Div(
										html.Button('Submit', 
											id='button',
											style={'backgroundColor' : 'gainsboro'}
										)									
									),
								], style={'width': '75%', 'height': '50%', 'display': 'inline-block'}
							 ),
						], style={'width': '75%', 'height': '50%', 'display': 'inline-block'}
					),					
					
				], style={'width': '25%', 'height': '50%', 'display': 'inline-block'}
			),
			
			
			html.Div(children=
				[
					html.Div(children=
						[
							html.H6('Built In Queries'),
							html.Label('For Custom Queries - Select \'Custom*\' in \'Built In Queries\' and then select bkg_from_city and/or bkg_to_city'),
							dcc.Dropdown(
								id='query_type',
								options=[{'label': i, 'value': i} for i in 
											[
											'All_India', 'Custom', 'Custom_Aggregate_From_Cities', 'Custom_Aggregate_To_Cities', 'Custom_Aggregate_From_And_To_Cities', 'Custom_Only_From_Cities', 'Custom_Only_To_Cities', 
											'5_Best_Routes', '5_Best_From_City_Routes', '5_Best_To_City_Routes', '5_Worst_Routes', '5_Worst_To_City_Routes', '5_Worst_From_City_Routes',
											'10_Best_Routes', '10_Best_From_City_Routes', '10_Best_To_City_Routes', '10_Worst_Routes', '10_Worst_From_City_Routes', '10_Worst_To_City_Routes',
											]],
								value='All_India',						 
							),
						], style={'width': '75%', 'height': '50%', 'display': 'inline-block'}
					),
					
					html.Div(children=
						[
							html.H6('bkg_from_city'),
							dcc.Dropdown(
								id='from_cities',
								options=[{'label': i, 'value': i} for i in cities],
								value='',	
								multi=True
							),
						], style={'width': '75%', 'height': '50%', 'display': 'inline-block'}
					),
					
					html.Div(children=
						[
							html.H6('bkg_to_city'),
							dcc.Dropdown(
								id='to_cities',
								options=[{'label': i, 'value': i} for i in cities],
								value='',	
								multi=True
							),
						], style={'width': '75%', 'height': '50%', 'display': 'inline-block'}
					),
					
				], style={'width': '20%', 'height': '50%', 'display': 'inline-block'}
			),
			
			html.Div(children=
				[
					html.Div(children=
						[
							html.H6('Modeling Types'), 
							dcc.Dropdown(
								id='model_type',
								options=[{'label': i, 'value': i} for i in ['Log', 'Linear']],
								value='Linear',
							),
						], style={'width': '100%', 'height': '50%', 'display': 'inline-block'}
					),
					
					html.Div(children=
						[
							html.H6('Forecast Type'),
							dcc.Dropdown(
								id='forecast_and_validate',
								options=[{'label': i, 'value': i} for i in ['Forecast', 'Regress, Forecast and Validate']],
								value='Forecast',
							),
						], style={'width': '100%', 'height': '50%', 'display': 'inline-block'}
					),
					
					html.Div(children=
						[
							html.H6('Forecast Period'),
							dcc.Dropdown(
								id='forecast_period',
								options=[{'label': i, 'value': i} for i in ['Weekly', 'Bi-Weekly', 'Monthly', 'Bi-Monthly', '3-Months']],
								value='Monthly',
							)
						], style={'width': '100%', 'height': '50%', 'display': 'inline-block'}
					),
					
					html.Div(children=
						[
							html.H6('Trends Type'),
							dcc.Dropdown(
								id='plot_components',
								options=[{'label': i, 'value': i} for i in ['Forecast Trends', 'Forecast, Weekly, Annual Trends']],
								value='Forecast Trends',
							)
						], style={'width': '100%', 'height': '50%', 'display': 'inline-block'}
					),					
				], 	style={'width': '15%', 'height': '50%', 'display': 'inline-block'}
			),
			
			html.Div(children=
						[
							html.H6(''),							
							
						], style={'width': '2%', 'height': '100%', 'display': 'inline-block'}
					),
			
			html.Div(children=
				[
					html.Div(children=
						[
							html.H6('Seasonal Surge (Induced in Forecast)'),
							dcc.Dropdown(
								id='seasonal_surge',
								options=[{'label': i, 'value': i} for i in 
											['0.0','0.1','0.125','0.15','0.20','0.25']],
								value='0.0',								
							),
							
						], style={'width': '100%', 'height': '100%', 'display': 'inline-block', }
					),
					html.Div(children=
						[
							html.H6('Weekend Surge (Induced in Forecast)'),
							dcc.Dropdown(
								id='weekend_surge',
								options=[{'label': i, 'value': i} for i in 
											['0.0','0.1','0.125','0.15','0.20','0.25']],
								value='0.0',								
							),
							
						], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
					),
				],
				
				style={'width': '20%', 'height': '50%', 'display': 'inline-block', 'display': 'none'}
			),			
		]),
			
		html.Div(id='forecast_graph_dynamic'),
		
		# Hidden div inside the app that stores the intermediate value
		html.Div(id='intermediate-value', style={'display': 'none'}),
		html.Div(forecast_session_id, id='forecast_session_id', style={'display': 'none'}),
		
		
	])

@app.callback(
	dash.dependencies.Output('intermediate-value', 'children'),
	[
		dash.dependencies.Input('forecast_session_id', 'children'),
		dash.dependencies.Input('model_type', 'value'),
		dash.dependencies.Input('forecast_and_validate', 'value'),
		dash.dependencies.Input('forecast_period', 'value'),
		dash.dependencies.Input('query_type', 'value'),
		dash.dependencies.Input('from_cities', 'value'),
		dash.dependencies.Input('to_cities', 'value'),
		dash.dependencies.Input('primary_input_field', 'value'),
		dash.dependencies.Input('input_date_range', 'start_date'),
		dash.dependencies.Input('input_date_range', 'end_date'),
		dash.dependencies.Input('plot_components', 'value'),
		dash.dependencies.Input('seasonal_surge', 'value'),
		dash.dependencies.Input('weekend_surge', 'value'),
	])
def update_controls(
					in_forecast_session_id,
					in_model_type,
					in_forecast_and_validate,
					in_forecast_period,
					in_query_type,
					in_from_cities,
					in_to_cities,
					in_primary_input_field,
					in_start_date,
					in_end_date,
					in_plot_components,
					in_seasonal_surge,
					in_weekend_surge,
					):
	
	validate_existing	= _is_forecast_and_validate(in_forecast_and_validate)
	forecast_period		= _get_forecast_period(in_forecast_period)
	primary_input_field = _get_primary_input_field(in_primary_input_field)
	start_date	= datetime.strptime(in_start_date, '%Y-%m-%d')
	end_date	= datetime.strptime(in_end_date, '%Y-%m-%d')
	plot_components		= _get_plot_components(in_plot_components)
	
	_controls_to_json = json.dumps(
			{
				'model_type'				: in_model_type, 
				'forecast_and_validate'		: validate_existing,
				'forecast_period'			: forecast_period, 
				'query_type'				: in_query_type,
				'from_cities'				: in_from_cities, 
				'to_cities'					: in_to_cities,
				'primary_input_field'		: primary_input_field, 
				'start_date'				: str(start_date),
				'end_date'					: str(end_date), 
				'plot_components'			: plot_components,
				'seasonal_surge'			: in_seasonal_surge,
				'weekend_surge'				: in_weekend_surge,
				'rest_call'					: use_rest_api,
			}
		)
	print ("UIForecasting::update_controls() - Enter")
	print (_controls_to_json)
	global_session_dict[in_forecast_session_id] = _controls_to_json
	print ("UIForecasting::update_controls() - Exit")
	return _controls_to_json
					
@app.callback(
	dash.dependencies.Output('forecast_graph_dynamic', 'children'),
	[
		dash.dependencies.Input('forecast_session_id', 'children'),
		dash.dependencies.Input('button', 'n_clicks'),
	])
def update_dynamic_graphs(
					in_forecast_session_id,
					n_clicks,
					):
	if n_clicks == None:
		return
	print ("UIForecasting::update_figure() - Enter")
	_json_data = global_session_dict[in_forecast_session_id]
	
	if _json_data == None or _json_data == '':
		print ("UIForecasting::update_figure() - controls date not initialized")
		return
	
	if use_rest_api:	
		return create_graphs_using_rest_api(_json_data)
	else:
		return create_graphs_using_json_api(_json_data)
	
				
def create_graphs(_json_data):
		
	_json_to_controls=json.loads(_json_data)	
	model_type = _json_to_controls["model_type"]
	validate_existing = _json_to_controls["forecast_and_validate"]
	forecast_period = _json_to_controls["forecast_period"]
	query_type = _json_to_controls["query_type"]
	from_cities = _json_to_controls["from_cities"]
	to_cities = _json_to_controls["to_cities"]
	primary_input_field = _json_to_controls["primary_input_field"]
	start_date	= datetime.strptime(
			_json_to_controls["start_date"].split()[0], '%Y-%m-%d')
	end_date	= datetime.strptime(
			_json_to_controls["end_date"].split()[0], '%Y-%m-%d')
	
	graphs = []
	graphs_data = []
	print_seperator_start('')
	
	log_transformation = model_type == 'Log'
	graphs_data = get_forecasting_graphs(
		yhat=primary_input_field,
		future_periods=forecast_period,
		do_log_transform=log_transformation,
		query_type=query_type,
		start_date=start_date, 
		end_date=end_date,
		from_cities=from_cities,
		to_cities=to_cities,
		set_capacity=False,
		rest_call=use_rest_api,
		debug=False
		)
	training_days = (end_date - start_date).days
	print(graphs_data)
	for label, graph_data in graphs_data.items():
		data = ProphetUtils.get_comparison_graph(graph_data, forecast_period, training_days)
		graphs.append(dcc.Graph(
				id=label,
				figure={
					'data': data,
					'layout': go.Layout(
						xaxis={'title': 'Time Period [' + label + ']'},
						yaxis={'title': primary_input_field},
					)
				}
			))
			
	print_seperator_end('')
	return graphs		 

def create_graphs_using_rest_api(_json_data):
	colored_graphs = []
	
	print_seperator_start('')
	headers = {"content-type": "application/json", "Authorization": "None" }
	url = aws_endpoint_url + 'forecast?query=' + _json_data
	print (url)
	r = requests.get(url, headers=headers)
	print (r.status_code)
	if r.status_code == 403 or r.status_code == 405:
		return graphs	
	print(r)
	_json_to_controls=json.loads(_json_data)	
	in_start_date = _json_to_controls["start_date"]
	in_end_date = _json_to_controls["end_date"]
	start_date	= datetime.strptime(in_start_date.split()[0], '%Y-%m-%d')
	end_date	= datetime.strptime(in_end_date.split()[0], '%Y-%m-%d')
	forecast_period = _json_to_controls["forecast_period"]
	primary_input_field = _json_to_controls["primary_input_field"]
	
	trends = json.loads(r.text)
	trends_data = trends["trends_data"]
	
	for trend_data in trends_data:
		graphs = []
	
		trend_result = json.loads(trend_data)
		label = trend_result["label"]		
		df = pd.read_json(trend_result['df'], convert_dates=['ds'])
		
		training_days = (end_date - start_date).days
		data = ProphetUtils.get_comparison_graph(df, forecast_period, training_days)
		meta = 'Training: ' + str(training_days) + ', Forecast: ' + str(forecast_period)
		print_dataframe(data, meta, False)
		
		graphs.append(
			html.H4("************************************************************************************",
				style={
					'textAlign': 'center', 'color' : 'grey'
				}
			)
		)
		graphs.append(
			html.H4(label,
				style={
					'textAlign': 'center', 'color' : 'orange'
				}
			)
		)
		
		graphs.append(dcc.Graph(
				id=label,
				figure={
					'data': data,
					'layout': go.Layout(
						xaxis={'title': 'Time Period [' + label + ']'},
						yaxis={'title': primary_input_field},
					)
				}
			))
		graphs.append(
			html.H4("************************************************************************************",
				style={
					'textAlign': 'center', 'color' : 'grey'
				}
			)
		)
		colored_graphs.append(
				html.Div(
					children=graphs,
					style={
						'textAlign': 'center', 'backgroundColor' : 'gainsboro'
					}
				)
			)
			
	print_seperator_end('')
	return colored_graphs			
			
def create_graphs_using_json_api(_json_data):
	graphs = []
	graphs_data = []
	print_seperator_start('')
	
	
	_json_to_controls=json.loads(_json_data)	
	in_start_date = _json_to_controls["start_date"]
	in_end_date = _json_to_controls["end_date"]
	start_date	= datetime.strptime(in_start_date.split()[0], '%Y-%m-%d')
	end_date	= datetime.strptime(in_end_date.split()[0], '%Y-%m-%d')
	forecast_period = _json_to_controls["forecast_period"]
	primary_input_field = _json_to_controls["primary_input_field"]
	training_days = (end_date - start_date).days
	
	graphs_data = get_forecasting_trends_json_wrapper(_json_data)
	
	for label, graph_data in graphs_data.items():
		data = ProphetUtils.get_comparison_graph(graph_data, forecast_period, training_days)
		graphs.append(dcc.Graph(
				id=label,
				figure={
					'data': data,
					'layout': go.Layout(
						xaxis={'title': 'Time Period [' + label + ']'},
						yaxis={'title': primary_input_field},
					)
				}
			))
			
	print_seperator_end('')
	return graphs
		
		
def print_seperator_start(meta):
	print("")
	print("--------------------------------------------------" +meta+ "--------------------------------------------------")

def print_seperator_end(meta):
	print("--------------------------------------------------" +meta+ "--------------------------------------------------")
	print("")
	
def _get_primary_input_field(input):
	if input == 'Booking Count':
		return 'bkg_user_id'
	elif input == 'Total Amount':
		return 'bkg_total_amount'
	elif input == 'Gozo Amount':
		return 'bkg_gozo_amount'

def _get_forecast_period(input):
	if input == 'Weekly':
		return 7
	elif input == 'Bi-Weekly':
		return 14
	elif input == 'Monthly':
		return 30
	elif input == 'Bi-Monthly':
		return 60
	elif input == '3-Months':
		return 90

def _is_forecast_and_validate(input):
	if input == 'Regress, Forecast and Validate':
		return True
	return False

def _get_plot_components(input):
	if input == 'Forecast, Weekly, Annual Trends':
		return True
	else:
		return False
