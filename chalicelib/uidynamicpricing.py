

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
import requests
from flask import jsonify
import re
from collections import defaultdict

sys.path.append('.')
from uiapp import *
from forecastingpipeline import *
from prophetmodeller import *

df_base = BaseDataframe()
df_gozo_cities = df_base.gozo_cities_df
cities = df_gozo_cities['cty_name'].tolist()
df_gozo_zones = df_base.gozo_zones_df
zones = df_gozo_zones['zon_name'].tolist()


pricing_options = [200* n for n in range(5,35+1)]
pricing_options.insert(0,1) # sepcial case to compute surge factor

MAX_GRAPH_PER_PAGE = 10
use_rest_api = True
cache_dataframes = {}
cache_colored_graphs = []
current_visible_graphs = {'start':0, 'end':0}

today = datetime.today()
def dynamicpricing_page_layout():
	
	dyn_pricing_session_id = str('DYNAMIC_PRICING_SESSION')
	
	return html.Div(children=[
		html.Div(children=[	
			html.Div(children=
				[
					html.H6(''),											
				], style={'width': '1%', 'height': '50%', 'display': 'inline-block'}
			),
			html.Div(children=
				[
					html.Div(children=
						[
							html.H6('Primary Input Field'),
							dcc.Dropdown(
								id='primary_input_field',
								options=[{'label': i, 'value': i} for i in ['Booking Count', 'Total Amount', 'Gozo Amount']],
								value='Booking Count',	
								disabled=True
							),
						], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
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
								end_date=datetime(datetime.today().year, datetime.today().month, datetime.today().day)
							),
						], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
					),
			
					html.Div(children=
						[
							html.H6('Click Submit after making all selections'),
							html.Div(children=
								[
									html.Div(
										html.Button('Submit', 
											id='submit_button',  n_clicks_timestamp='0',
											style={'backgroundColor' : 'gainsboro'}
										)									
									),
								], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
							 ),
						], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
					),
					

					html.Div(children=
						[
							html.H6('Click Publish after satisfactory modeling'),
							html.Div(children=
								[
									html.Div(
										html.Button('Publish', 
											id='publish_button',
											style={'backgroundColor' : 'gainsboro'}
										)									
									),
								], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
							 ),
						], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
					),
					
					html.Div(children=
						[
							html.H6('Auto Publish'),
							dcc.Dropdown(
								id='auto_publish',
								options=[{'label': i, 'value': i} for i in ['No', 'Yes']],
								value='No',
							),
						], style={'width': '50%', 'height': '100%', 'display': 'inline-block'}
					),
					
				], style={'width': '20%', 'height': '100%', 'display': 'inline-block'}
			),
			html.Div(children=
				[
					html.H6(''),
									
				], style={'width': '1%', 'height': '50%', 'display': 'inline-block'}
			),
			
			html.Div(children=
				[ 	
					html.Div(children=
						[
							html.H6('Zone Based or Route Based'),
							dcc.Dropdown(
								id='route_or_zone_based',
								options=[{'label': i, 'value': i} for i in ['Route_Based', 'Zone_Based']],
								value='Route_Based',
							),
						], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
					),
					
					html.Div(children=
						[
							html.H6('Built In Queries'),
							html.Label('For Custom Queries - Select \'Custom*\' in \'Built In Queries\' and then select bkg_from_city and/or bkg_to_city'),
							dcc.Dropdown(
								id='query_type',
								options=[{'label': i, 'value': i} for i in 
											[
											'All_India', 
											'Custom', 'Custom_Aggregate_From_Cities', 'Custom_Aggregate_To_Cities', 'Custom_Aggregate_From_And_To_Cities', 'Custom_Only_From_Cities', 'Custom_Only_To_Cities', 
											'5_Best_Routes', '5_Best_From_City_Routes', '5_Best_To_City_Routes', '5_Worst_Routes', '5_Worst_To_City_Routes', '5_Worst_From_City_Routes',
											'10_Best_Routes', '10_Best_From_City_Routes', '10_Best_To_City_Routes', '10_Worst_Routes', '10_Worst_From_City_Routes', '10_Worst_To_City_Routes',
											'ALL_HIGH_PRIORITY', 'ALL_MODERATE_PRIORITY', 'ALL_LOW_PRIORITY'
											]],
								value='Custom',
								#disabled=True,
							),
							html.Label(''),
						], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
					),
					
					html.Div(children=
						[
							html.H6('bkg_from_city'),
							dcc.Dropdown(
								id='from_cities',
								options=[{'label': i, 'value': i} for i in cities],
								value=['Delhi'],	
								multi=True
							),
						], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
					),
					
					html.Div(children=
						[
							html.H6('bkg_to_city'),
							dcc.Dropdown(
								id='to_cities',
								options=[{'label': i, 'value': i} for i in cities],
								value=['Delhi'],	
								multi=True
							),
						], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
					),
					
					html.Div(children=
						[
							html.H6('bkg_from_zone'),
							dcc.Dropdown(
								id='from_zones',
								options=[{'label': i, 'value': i} for i in zones],
								value=['Z-DELHI-NCR'],	
								multi=True
							),
						], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
					),
					
					html.Div(children=
						[
							html.H6('bkg_to_zone'),
							dcc.Dropdown(
								id='to_zones',
								options=[{'label': i, 'value': i} for i in zones],
								value=['Z-DELHI-NCR'],	
								multi=True
							),
						], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
					),
					
				], style={'width': '15%', 'height': '100%', 'display': 'inline-block'}
			),
			html.Div(children=
				[
					html.H6(''),
									
				], style={'width': '1%', 'height': '50%', 'display': 'inline-block'}
			),
			html.Div(children=
				[
					html.Div(children=
						[
							html.H6('Metric Types'), 
							dcc.Dropdown(
								id='metric_type',
								options=[{'label': i, 'value': i} for i in ['Mean', 'Median', 'Avg. Mean+Median', 'Trimmed Mean', 'Mean of IQR']],
								value='Median',								
							),
						], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
					),
					
					html.Div(children=
						[
							html.H6('Modeling Types'), 
							dcc.Dropdown(
								id='model_type',
								options=[{'label': i, 'value': i} for i in ['Log', 'Linear']],
								value='Linear',								
							),
						], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
					),
					
					html.Div(children=
						[
							html.H6('Forecast Type'),
							dcc.Dropdown(
								id='forecast_and_validate',
								options=[{'label': i, 'value': i} for i in ['Forecast', 'Regress, Forecast and Validate']],
								value='Forecast',								
							),
						], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
					),
					
					html.Div(children=
						[
							html.H6('Forecast Period'),
							dcc.Dropdown(
								id='forecast_period',
								options=[{'label': i, 'value': i} for i in ['Weekly', 'Bi-Weekly', 'Monthly', 'Bi-Monthly', '3-Months', '6-Months']],
								value='3-Months',
							)
						], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
					),
					
					html.Div(children=
						[
							html.H6('Trends Type'),
							dcc.Dropdown(
								id='plot_components',
								options=[{'label': i, 'value': i} for i in ['Forecast Trends', 'Forecast, Weekly, Annual Trends']],
								value='Forecast Trends',
							)
						], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
					),
				],
				style={'width': '15%', 'height': '100%', 'display': 'inline-block'}
			),
			html.Div(children=
				[
					html.H6(''),
									
				], style={'width': '1%', 'height': '50%', 'display': 'inline-block'}
			),
			
			html.Div(children=
				[ 	
					html.Div(children=
						[
							html.Div(children=
								[
									html.H6('Base Price'),
									dcc.Dropdown(
										id='base_price',
										options=[{'label': i, 'value': i} for i in pricing_options],
										value='1',								
									),
								], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
							),
							html.Div(children=
								[
									html.H6('D/P Elasticity'),
									dcc.Dropdown(
										id='elasticity',
										options=[{'label': i, 'value': i} for i in 
													['-1.0','-1.5','-2.0','-2.5','-3.0']],
										value='-2.0',								
									),
									
								], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
							),					
						], style={'width': '49%', 'height': '100%', 'display': 'inline-block'}
					),
					html.Div(children=
						[
							html.H6(''),
											
						], style={'width': '1%', 'height': '50%', 'display': 'inline-block'}
					),
					html.Div(children=
						[
							
							html.Div(children=
								[
									html.H6('Seasonal Surge'),
									dcc.Dropdown(
										id='seasonal_surge',
										options=[{'label': i, 'value': i} for i in 
													['0.0','0.1','0.125','0.15','0.20','0.25']],
										value='0.0',	
									),
									
								], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
							),
							html.Div(children=
								[
									html.H6('Weekend Surge'),
									dcc.Dropdown(
										id='weekend_surge',
										options=[{'label': i, 'value': i} for i in 
													['0.0','0.1','0.125','0.15','0.20','0.25']],
										value='0.0',							
									),
									
								], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
							),
						], style={'width': '49%', 'height': '100%', 'display': 'inline-block'}
					),
					html.Div(children=
						[
							html.H6(''),
											
						], style={'width': '1%', 'height': '50%', 'display': 'inline-block'}
					),
					html.Div(children=
						[		
							html.Div(children=
								[
									html.H6('Base Capacity - Absolute'),
									dcc.Dropdown(
										id='base_capacity_absolute',
										options=[{'label': i, 'value': i} for i in 
													['ForecastedValue', 'MedianOfForecastedValue', 
													'MedianOfLast3Months', 'MedianOfLast6Months', 'MedianOfLast9Months', 'MedianOfLast12Months', 
													'MedianOfLast3MonthsAndForecastedValue', 'MedianOfLast6MonthsAndForecastedValue', 'MedianOfLast9MonthsAndForecastedValue', 'MedianOfLast12MonthsAndForecastedValue' ]],
										value='MedianOfLast6MonthsAndForecastedValue',	
									),
									
								], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
							),
							html.Div(children=
								[
									html.H6('Base Capacity - Distribution'),
									dcc.Dropdown(
										id='base_capacity_distribution',
										options=[{'label': i, 'value': i} for i in 
													['Median', 'WeeklyMedian']],
										value='WeeklyMedian',	
									),
									
								], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
							),
							
						], style={'width': '49%', 'height': '100%', 'display': 'inline-block'}
					),
					html.Div(children=
						[
							html.Div(children=
								[
									html.H6('Lower Surge Factor'),
									dcc.Dropdown(
										id='base_price_low',
										options=[{'label': i, 'value': i} for i in 
													['0.90', '1.00', '1.10', '1.20', '1.30', '1.40', '1.50', '1.60', '1.70', '1.80', '1.90', 
													         '2.00', '2.10', '2.20', '2.30', '2.40', '2.50', '2.60', '2.70', '2.80', '2.90', 
															 '3.00', '3.10', '3.20', '3.30', '3.40', '3.50', '3.60', '3.70', '3.80', '3.90'] ],
										value=['0.90', '1.00', '1.10', '1.20', '1.30', '1.40', '1.50', '1.60', '1.70', '1.80', '1.90', '2.00', '2.20', '2.40', '2.70', '3.00', '3.50'],
										multi=True
									),
								], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
							),
							html.Div(children=
								[
									html.H6('Upper Surge Factor'),
									dcc.Dropdown(
										id='base_price_high',
										options=[{'label': i, 'value': i} for i in 
													['1.00', '1.10', '1.20', '1.30', '1.40', '1.50', '1.60', '1.70', '1.80', '1.90', 
													 '2.00', '2.10', '2.20', '2.30', '2.40', '2.50', '2.60', '2.70', '2.80', '2.90', 
													 '3.00', '3.10', '3.20', '3.30', '3.40', '3.50', '3.60', '3.70', '3.80', '3.90', '4.00'] ],
										value=['1.00', '1.10', '1.20', '1.30', '1.40', '1.50', '1.60', '1.70', '1.80', '1.90', '2.00', '2.20', '2.40', '2.70', '3.00', '3.50', '4.00'],
										multi=True
									),
								], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
							),
							html.Div(children=
								[
									html.H6('Capacities'),
									dcc.Dropdown(
										id='booking_capacities',
										options=[{'label': i, 'value': i} for i in 
													['M-000', 'M-010', 'M-020', 'M-030', 'M-040', 'M-050', 'M-060', 'M-070', 'M-080', 'M-090', 'M-100', 
														         'M-110', 'M-120', 'M-130', 'M-140', 'M-150', 'M-160', 'M-170', 'M-180', 'M-190', 'M-200',
																 'M-210', 'M-220', 'M-230', 'M-240', 'M-250', 'M-260', 'M-270', 'M-280', 'M-290', 'M-300' ]
												],
										#value=['1.00', '1.10', '1.20', '1.30', '1.40', '1.50', '1.60', '1.70', '1.80', '1.90', '2.00', '2.20', '2.40', '2.70', '3.00', '3.50', '4.00'],
										value=['M-000', 'M-010', 'M-020', 'M-030', 'M-040', 'M-050', 'M-060', 'M-070', 'M-080', 'M-090', 'M-100', 'M-120', 'M-140', 'M-170', 'M-200', 'M-250', 'M-300' ],
										multi=True
									),
								], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
							),
													
						], style={'width': '100%', 'height': '100%', 'display': 'inline-block'}
					),
					
				], style={'width': '40%', 'height': '100%', 'display': 'inline-block'}
			),
		]),
		
		html.Div(children=
			[
				html.H6(''),
								
			], style={'width': '1%', 'height': '50%', 'display': 'inline-block'}
		),
		
		html.Div(children=
			[
				html.Div(children=
					[
						html.Div(
							html.Button('Prev 10', 
								id='top_previous', n_clicks_timestamp='0',
								style={'backgroundColor' : 'gainsboro'}
							),									
							style={'width': '50%', 'height': '100%', 'display': 'inline-block'}
						),									
						html.Div(
							html.Button('Next 10', 
								id='top_next', n_clicks_timestamp='0',
								style={'backgroundColor' : 'gainsboro'}
							),									
							style={'width': '50%', 'height': '100%', 'display': 'inline-block'}									
						),
					], style={'width': '15%', 'height': '100%', 'display': 'inline-block'}
				 ),
			], style={'width': '100%', 'height': '100%', 'display': 'inline-block', 'textAlign': 'center'}
		),
		
		html.Div(children=
			[
				html.H6(''),
								
			], style={'width': '1%', 'height': '50%', 'display': 'inline-block'}
		),
		
		html.Div(id='dyn_pricing_graph_dynamic'),
		
		html.Div(children=
			[
				html.H6(''),
								
			], style={'width': '1%', 'height': '50%', 'display': 'inline-block'}
		),
		
		html.Div(children=
			[
				html.Div(children=
					[
						html.Div(
							html.Button('Prev 10', 
								id='bottom_previous', n_clicks_timestamp='0',
								style={'backgroundColor' : 'gainsboro'}
							),									
							style={'width': '50%', 'height': '100%', 'display': 'inline-block'}
						),									
						html.Div(
							html.Button('Next 10', 
								id='bottom_next', n_clicks_timestamp='0',
								style={'backgroundColor' : 'gainsboro'}
							),									
							style={'width': '50%', 'height': '100%', 'display': 'inline-block'}									
						),
					], style={'width': '15%', 'height': '100%', 'display': 'inline-block'}
				 ),
			], style={'width': '100%', 'height': '100%', 'display': 'inline-block', 'textAlign': 'center'}
		),
		# Hidden div inside the app that stores the intermediate value
		html.Div(id='dyn_pricing_intermediate_value', style={'display': 'none'}),
		html.Div(id='publish_intermediate_value', style={'display': 'none'}),
		html.Div(dyn_pricing_session_id, id='dyn_pricing_session_id', style={'display': 'none'}),	
		
	])


@app.callback(
	dash.dependencies.Output('query_type', 'value'),
	[
		dash.dependencies.Input('dyn_pricing_session_id', 'children'),
		dash.dependencies.Input('route_or_zone_based', 'value'),
	])
def set_query_type(in_dyn_pricing_session_id, in_route_or_zone_based):
	if in_route_or_zone_based == 'Zone_Based': 
		return 'Custom_Aggregate_From_And_To_Cities'
	else:
		_json_data = global_session_dict[in_dyn_pricing_session_id]
		if _json_data == None or len(_json_data) == 0:
			return 'Custom'
		#print ("UIDynamicPricing::set_query_type() _json_data ", _json_data)
		_json_to_controls=json.loads(_json_data)
		query = _json_to_controls["query_type"]
		#print ("UIDynamicPricing::set_query_type() ", query)
		if query == 'Custom_Aggregate_From_And_To_Cities':
			return 'Custom'
		else:
			return query


# start - disable query_type, from_cities, to_cities		
def disable_from_cities(in_query_type):
	disabled = disable_from_cities_and_to_cities(in_query_type)
	if disabled == True:
		return disabled
	else:
		if 'Custom_Only_To_Cities' == in_query_type:
			return True
		elif '_To_City_Routes' in in_query_type:
			return True
		else:
			return False
	
	
def disable_to_cities(in_query_type):
	disabled = disable_from_cities_and_to_cities(in_query_type)
	if disabled == True:
		return disabled
	else:
		if 'Custom_Only_From_Cities' == in_query_type:
			return True
		elif '_From_City_Routes' in in_query_type:
			return True
		else:
			return False

def disable_from_cities_and_to_cities(in_query_type):
	if 'PRIORITY' in in_query_type:
		return True
	elif 'All_India' in in_query_type:
		return True	
	elif '_Best_Routes' in in_query_type:
		return True	
	elif '_Worst_Routes' in in_query_type:
		return True		
	return False
		
@app.callback(
	dash.dependencies.Output('query_type', 'disabled'),
	[
		dash.dependencies.Input('route_or_zone_based', 'value'),		
	])
def set_button_enabled_state(in_route_or_zone_based):
	if in_route_or_zone_based == 'Zone_Based': 
		return True
	return False
@app.callback(
	dash.dependencies.Output('from_cities', 'disabled'),
	[
		dash.dependencies.Input('route_or_zone_based', 'value'),	
		dash.dependencies.Input('query_type', 'value'),			
	])
def set_button_enabled_state(in_route_or_zone_based, in_query_type):
	if in_route_or_zone_based == 'Zone_Based': 
		return True
	else:
		return disable_from_cities(in_query_type)
		
@app.callback(
	dash.dependencies.Output('to_cities', 'disabled'),
	[
		dash.dependencies.Input('route_or_zone_based', 'value'),	
		dash.dependencies.Input('query_type', 'value'),	
	])
def set_button_enabled_state(in_route_or_zone_based, in_query_type):
	if in_route_or_zone_based == 'Zone_Based': 
		return True
	else:
		return disable_to_cities(in_query_type)
# end - disable query_type, from_cities, to_cities		
	

# start - change color for query_type, from_cities, to_cities
style_grey={'backgroundColor' : 'gainsboro'}
style_default={'backgroundColor' : 'white'}

@app.callback(
	dash.dependencies.Output('query_type', 'style'),
	[
		dash.dependencies.Input('route_or_zone_based', 'value'),		
	])
def set_button_enabled_state(in_route_or_zone_based):
	if in_route_or_zone_based == 'Zone_Based': 
		return style_grey
	return style_default

'''
@app.callback(
	dash.dependencies.Output('from_cities', 'style'),
	[
		dash.dependencies.Input('route_or_zone_based', 'value'),	
		dash.dependencies.Input('query_type', 'value')
	])
def set_button_enabled_state_v2(in_route_or_zone_based, in_query_type):
	if in_route_or_zone_based == 'Zone_Based': 
		print ("UIDynamicPricing::set_button_enabled_state() ", in_route_or_zone_based, style_grey)
		return style_grey
	else:
		ret = disable_from_cities(in_query_type)
		if ret == True:
			print ("UIDynamicPricing::set_button_enabled_state() ", in_route_or_zone_based, style_grey)
			return style_grey
		else:
			print ("UIDynamicPricing::set_button_enabled_state() ", in_route_or_zone_based, style_default)
			return style_default
			
@app.callback(
	dash.dependencies.Output('to_cities', 'style'),
	[
		dash.dependencies.Input('route_or_zone_based', 'value'),	
		dash.dependencies.Input('query_type', 'value'),	
	])
def set_button_enabled_state_v2(in_route_or_zone_based, in_query_type):
	if in_route_or_zone_based == 'Zone_Based': 
		return style_grey
	else:
		if disable_to_cities(in_query_type):
			return style_grey
		else:
			return style_default
# end - change color for query_type, from_cities, to_cities		
'''


# start - disable from_zones and to_zones		
@app.callback(
	dash.dependencies.Output('from_zones', 'disabled'),
	[
		dash.dependencies.Input('route_or_zone_based', 'value'),		
	])
def set_button_enabled_state(in_route_or_zone_based):
	if in_route_or_zone_based == 'Zone_Based': 
		return False
	return True
@app.callback(
	dash.dependencies.Output('to_zones', 'disabled'),
	[
		dash.dependencies.Input('route_or_zone_based', 'value'),		
	])
def set_button_enabled_state(in_route_or_zone_based):
	if in_route_or_zone_based == 'Zone_Based': 
		return False
	return True
# end - disable from_zones and to_zones


# start - change color for from_zones, to_zones
@app.callback(
	dash.dependencies.Output('from_zones', 'style'),
	[
		dash.dependencies.Input('route_or_zone_based', 'value'),		
	])
def set_button_enabled_state(in_route_or_zone_based):
	if in_route_or_zone_based == 'Zone_Based': 
		return style_default
	return style_grey
@app.callback(
	dash.dependencies.Output('to_zones', 'style'),
	[
		dash.dependencies.Input('route_or_zone_based', 'value'),		
	])
def set_button_enabled_state(in_route_or_zone_based):
	if in_route_or_zone_based == 'Zone_Based': 
		return style_default
	return style_grey
# end - change color for from_zones, to_zones

@app.callback(
	dash.dependencies.Output('dyn_pricing_intermediate_value', 'children'),
	[
		dash.dependencies.Input('dyn_pricing_session_id', 'children'),
		dash.dependencies.Input('model_type', 'value'),
		dash.dependencies.Input('forecast_and_validate', 'value'),
		dash.dependencies.Input('forecast_period', 'value'),
		dash.dependencies.Input('route_or_zone_based', 'value'),
		dash.dependencies.Input('from_zones', 'value'),
		dash.dependencies.Input('to_zones', 'value'),
		dash.dependencies.Input('query_type', 'value'),
		dash.dependencies.Input('from_cities', 'value'),
		dash.dependencies.Input('to_cities', 'value'),
		dash.dependencies.Input('primary_input_field', 'value'),
		dash.dependencies.Input('input_date_range', 'start_date'),
		dash.dependencies.Input('input_date_range', 'end_date'),
		dash.dependencies.Input('plot_components', 'value'),
		dash.dependencies.Input('base_price', 'value'),
		dash.dependencies.Input('base_price_low', 'value'),
		dash.dependencies.Input('base_price_high', 'value'),
		dash.dependencies.Input('elasticity', 'value'),
		dash.dependencies.Input('seasonal_surge', 'value'),
		dash.dependencies.Input('weekend_surge', 'value'),
		dash.dependencies.Input('booking_capacities', 'value'),
		dash.dependencies.Input('base_capacity_absolute', 'value'),
		dash.dependencies.Input('base_capacity_distribution', 'value'),
		dash.dependencies.Input('auto_publish', 'value')		
	])
def update_controls(
					in_dyn_pricing_session_id,
					in_model_type,
					in_forecast_and_validate,
					in_forecast_period,
					in_route_or_zone_based,
					in_from_zones,
					in_to_zones,
					in_query_type,
					in_from_cities,
					in_to_cities,
					in_primary_input_field,
					in_start_date,
					in_end_date,
					in_plot_components,
					in_base_price,
					in_base_price_low,
					in_base_price_high,
					in_elasticity,
					in_seasonal_surge,
					in_weekend_surge,
					in_booking_capacities,
					in_base_capacity_absolute,
					in_base_capacity_distribution,
					in_auto_publish
					):
	
	validate_existing	= _is_forecast_and_validate(in_forecast_and_validate)
	forecast_period		= _get_forecast_period(in_forecast_period)
	primary_input_field = _get_primary_input_field(in_primary_input_field)
	start_date	= datetime.strptime(in_start_date, '%Y-%m-%d')
	end_date	= datetime.strptime(in_end_date, '%Y-%m-%d')
	plot_components		= _get_plot_components(in_plot_components)
	
	from_cities = in_from_cities
	to_cities = in_to_cities
	if in_route_or_zone_based == 'Zone_Based':
		from_cities.clear()
		to_cities.clear()
		in_query_type = 'Custom_Aggregate_From_And_To_Cities'		
		for from_zone in in_from_zones:	
			for cities in df_gozo_zones['cty_names'][df_gozo_zones.zon_name == from_zone]:
				for city in cities.split(','):
					from_cities.append(city.lstrip().rstrip())
		for to_zone in in_to_zones:		
			for cities in df_gozo_zones['cty_names'][df_gozo_zones.zon_name == to_zone]:
				for city in cities.split(','):
					to_cities.append(city.lstrip().rstrip())
		
		
	_controls_to_json = json.dumps(
			{
				'model_type'				: in_model_type, 
				'forecast_and_validate'		: validate_existing,
				'forecast_period'			: forecast_period, 
				'route_or_zone_based'		: in_route_or_zone_based,
				'from_zones'				: in_from_zones, 
				'to_zones'					: in_to_zones,
				'query_type'				: in_query_type,
				'from_cities'				: from_cities, 
				'to_cities'					: to_cities,
				'primary_input_field'		: primary_input_field, 
				'start_date'				: str(start_date),
				'end_date'					: str(end_date), 
				'auto_publish'				: in_auto_publish, 
				'plot_components'			: plot_components,				
				'base_price'				: in_base_price,
				'base_price_low'			: in_base_price_low,
				'base_price_high'			: in_base_price_high,
				'elasticity'				: in_elasticity,
				'seasonal_surge'			: in_seasonal_surge,
				'weekend_surge'				: in_weekend_surge,
				'booking_capacities'		: in_booking_capacities,
				'base_capacity_absolute'	: in_base_capacity_absolute,
				'base_capacity_distribution': in_base_capacity_distribution,
				'rest_call'					: use_rest_api,
			}
		)
	print ("UIDynamicPricing::update_controls() - Enter")
	print (_controls_to_json)
	global_session_dict[in_dyn_pricing_session_id] = _controls_to_json
	print ("UIDynamicPricing::update_controls() - Exit")
	return _controls_to_json
					
@app.callback(
	dash.dependencies.Output('dyn_pricing_graph_dynamic', 'children'),
	[
		dash.dependencies.Input('dyn_pricing_session_id', 'children'),	
		dash.dependencies.Input('submit_button', 'n_clicks_timestamp'),		
		dash.dependencies.Input('top_previous', 'n_clicks_timestamp'),        
        dash.dependencies.Input('top_next', 'n_clicks_timestamp'),
		dash.dependencies.Input('bottom_previous', 'n_clicks_timestamp'),        
        dash.dependencies.Input('bottom_next', 'n_clicks_timestamp')
	])
def update_dynamic_graphs(
					in_dyn_pricing_session_id,	
					in_submit_ts,					
					in_top_previous_ts,
					in_top_next_ts,
					in_bottom_previous_ts,
					in_bottom_next_ts
					):
	btn_state = [int(in_submit_ts), int(in_top_previous_ts), int(in_top_next_ts), int(in_bottom_previous_ts), int(in_bottom_next_ts)]
	if all(v == 0 for v in btn_state):	
		return
	 
	print ("UIDynamicPricing::update_figure() - Enter")
	_json_data = global_session_dict[in_dyn_pricing_session_id]
	
	if _json_data == None or _json_data == '':
		print ("UIDynamicPricing::update_figure() - controls data not initialized")
		return
	
	if use_rest_api:	
		return create_graphs_using_rest_api(_json_data, btn_state)
	else:
		return create_graphs_using_json_api(_json_data, btn_state)
	print ("UIDynamicPricing::update_figure() - Exit")
	
				
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
	#print(graphs_data)
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

def create_graphs_using_rest_api(_json_data, btn_state):
	global cache_colored_graphs
	global current_visible_graphs
	global cache_dataframes
	
	max_index = btn_state.index(max(i for i in btn_state if i is not None))
	
	# submit button
	if max_index == 0: 
		cache_dataframes.clear()	
		cache_colored_graphs.clear()	
		colored_graphs = []
		
		print_seperator_start('')
		headers = {"content-type": "application/json", "Authorization": "None" }
		url = aws_endpoint_url + 'dynamicprice?query=' + _json_data
		print (url)
		r = requests.get(url, headers=headers, timeout=None)
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
		booking_capacities = _json_to_controls["booking_capacities"]
		
		
		dynamic_prices = json.loads(r.text)
		dynamic_prices_data = dynamic_prices["dynamic_price_data"]
		
		#print('\ndynamic_prices_data-Start\n', dynamic_prices_data, '\ndynamic_prices_data-End\n')
		
		
		for dynamic_price_data in dynamic_prices_data:		
			graphs = []	
			print_dataframe(dynamic_price_data, "Dynamic Pricing Data Set", False)		
			dynamic_price_result = json.loads(dynamic_price_data)
			
			label = dynamic_price_result["label"]
			df_trends_data = pd.read_json(
				dynamic_price_result["trends_data"], convert_dates=['ds'])		
			df_dynamic_price_table = pd.read_json(
				dynamic_price_result["dynamic_price_table"], convert_dates=['ds'])
			
			cache_dataframes[label] = df_dynamic_price_table
			
			print_dataframe(df_trends_data, 
				"Dynamic Pricing Data Set - Trend Data", False)				
			print_dataframe(df_dynamic_price_table, 
				"Dynamic Pricing Data Set - Dynamic Price Datatable", False)
			# show overall trend
			training_days = (end_date - start_date).days
			data = ProphetUtils.get_comparison_graph(
				df_trends_data, forecast_period, training_days)
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
							title='Current Rate (Red Lines) vs. Projected Rate (Blue Lines)',
							xaxis={'title': 'Time Period [' + label + ']'},
							yaxis={'title': primary_input_field},
						)
					}
				))
			
			# show forecast trend
			# show dynamic price trend
			df_table = df_dynamic_price_table.copy()
			dow_map = { 6:'Sun', 0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat'}
			df_table['Date'] = df_table.apply(
				lambda row: row['Weekday']+" "+str(row['Date'].date()),axis=1)
			df_table.index = df_table['Date']
			trim_factor = len(df_table)
			
			print ('Booking Capacities', booking_capacities)
			price_levels = []
			for capacity in booking_capacities:
				#cap = int(''.join(filter(str.isdigit, capacity)))
				#y_label = str('Capacity left : %s' % capacity)
				y_label = capacity
				#name_lable = str('Capacity Remaining : %s vechiles' % round(float(capacity)))
				#name_lable = str('Capacity Remaining : %s vechiles' % capacity)
				name_lable = capacity
				price_levels.append(
							go.Scatter(x=df_table.head(
										trim_factor).index,
								   y=df_table.head(
										trim_factor)[y_label],
								   name=name_lable)
							)
			
			graphs.append(dcc.Graph(
					id=label+'Dynamic Price',
					figure={
						'data': price_levels,
						'layout': go.Layout(
							title='Rate vs Reservation Date and Current Capacity Levels',
							xaxis={'title': 'Days [' + label + ']'},
							yaxis={'title': 'Rate (Rs)'},
						)
					}
				))
			# line break
			graphs.append(html.H5(" "))
			
			# show dynamic price table
			graphs.append(datatable.DataTable(
				id=label+'Dynamic Price Table',
				rows=df_dynamic_price_table.to_dict('records'),			
				))
			
			
			
			total_dynamic_price = int(df_dynamic_price_table['Total DP'].sum())
			total_static_price = int(df_dynamic_price_table['Total SP'].sum())
			percentage = round(float(((total_dynamic_price - total_static_price)/total_static_price)*100),2)
			dp_color = 'green'
			sp_color = 'red'
			increase = '% Increase'
			if total_static_price > total_dynamic_price:
				dp_color = 'red'
				sp_color = 'green'
				increase = '% Drop'
			dp_label = str('Yield from Dynamic Price for %s days: %s (%s %s)' % 
				(forecast_period, f"{total_dynamic_price:,d}", percentage, increase))
			sp_label = str('Yield from Static Price for %s days: %s' % 
				(forecast_period, f"{total_static_price:,d}"))
			graphs.append(html.H5(dp_label,
					style={
						'textAlign': 'center', 'color' : dp_color
					}
				))
			graphs.append(html.H5(sp_label,
					style={
						'textAlign': 'center', 'color' : sp_color
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
		cache_colored_graphs = colored_graphs		
		current_visible_graphs = {'start':0, 'end':MAX_GRAPH_PER_PAGE}
		
	# Show previous	
	elif max_index == 1 or max_index == 3:
		prev_visible_graphs = current_visible_graphs
		start = prev_visible_graphs['start']
		end = prev_visible_graphs['end']		
		
		start = start - MAX_GRAPH_PER_PAGE
		end = start - MAX_GRAPH_PER_PAGE
		if start < 0:
			start = 0
		if end < 0:
			end = MAX_GRAPH_PER_PAGE
		if end > len(cache_colored_graphs):
			end = len(cache_colored_graphs)
			
		current_visible_graphs = {'start':start, 'end':end}		
		
	# show next	
	elif max_index == 2 or max_index == 4:
		prev_visible_graphs = current_visible_graphs
		start = prev_visible_graphs['start']
		end = prev_visible_graphs['end']		
		
		start = start + MAX_GRAPH_PER_PAGE
		end = start + MAX_GRAPH_PER_PAGE
		
		if start > len(cache_colored_graphs): 
			start = 0
		if end > len(cache_colored_graphs):
			end = len(cache_colored_graphs)
			
		current_visible_graphs = {'start':start, 'end':end}
	
	print (current_visible_graphs, len(cache_colored_graphs))	
	return cache_colored_graphs[current_visible_graphs['start']:current_visible_graphs['end']]
	


	
def create_graphs_using_json_api(_json_data, btn_state):
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

@app.callback(
	dash.dependencies.Output('publish_intermediate_value', 'children'),
	[
		dash.dependencies.Input('dyn_pricing_session_id', 'children'),
		dash.dependencies.Input('publish_button', 'n_clicks'),
	])
def publish_dynamic_prices(
					in_dyn_pricing_session_id,
					n_clicks,
					):
	if n_clicks == None:
		return
	print ("UIDynamicPricing::publish_dynamic_prices() - Enter")
	_json_data = global_session_dict[in_dyn_pricing_session_id]
	
	if _json_data == None or _json_data == '':
		print ("UIDynamicPricing::publish_dynamic_prices() - controls date not initialized")
		return
	
	for label, dataframe in cache_dataframes.items():
		name = label
		name = name.replace('bkg_from_city:','').replace('bkg_to_city:', '')
		name = name.replace('bkg_from_zone:','').replace('bkg_to_zone:', '')
		name = 'dynprice_' +  name
		
		filename = name + '.csv'
		filename = '.\\resources\\dynamic_prices_client\\'+filename
		
		db_dataframe = dataframe.copy()
		csv_dataframe = dataframe.copy()
		csv_dataframe['Date'] = pd.to_datetime(
			csv_dataframe['Date'], infer_datetime_format=True)
		csv_dataframe.to_csv(filename, index=False, date_format='%Y-%m-%d')
		db_dataframe.index = db_dataframe['Date'].dt.date
		print ("UIDynamicPricing::publish_dynamic_prices() - update to db")
		SqlLiteDynPriceAdapter.getInstance().update(name, dataframe)
		
	
	print ("UIDynamicPricing::publish_dynamic_prices() - Exit")
	

	
def _get_primary_input_field(input):
	if input == 'Booking Count':
		return 'bkg_id'
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
	elif input == '6-Months':
		return 180

def _is_forecast_and_validate(input):
	if input == 'Regress, Forecast and Validate':
		return True
	return False

def _get_plot_components(input):
	if input == 'Forecast, Weekly, Annual Trends':
		return True
	else:
		return False
