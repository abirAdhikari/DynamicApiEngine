# AAdhikari: 9/3/2018

import os 
import sys 
from scipy import stats
import plotly.plotly as py
import plotly.graph_objs as go
from fbprophet import Prophet
import numpy as np
import datetime as dt
import time
import matplotlib.pyplot as plt
from collections import defaultdict
import json

#profiler
#import numba
#import cProfile

sys.path.append('./chalicelib')
sys.path.append('.')
from prophetmodeller import *
from genericdataframe import *
from debuglogger import *

class ForcastingPipelineData:
	def __init__(self):
		self.yhat = None
		self.future_periods = 0
		self.do_log_transform = False
		self.query_type = None
		self.route_or_zone_based = None
		self.start_date = None
		self.end_date = None
		self.from_cities = []
		self.to_cities = []
		self.from_zones = []
		self.to_zones = []
		self.validate_existing = False
		self.set_capacity = False
		self.rest_call=False
		self.debug = False
		self.booking_data = []
		self.forecast_data = []
		self.weekend_surge_factor = 0.0
		self.seasonal_surge_factor = 0.0


class ForecastingPipelineExecuter:
	def __init__(self):
		self.pipeline_map = {}
		
	def add(self, pipeline):
		name = pipeline.pipeline_name
		if name not in self.pipeline_map:
			self.pipeline_map[name] = pipeline
	
	def execute(self):
		for pipeline in self.pipeline_map.values():
			pipeline.execute_pipeline()	
			
class GenericForecastingPipeline:
	def __init__(self, name, pipeline_data):
		self.pipeline_name = name
		self.pipeline_data = pipeline_data
	
	def execute_pipeline(self):
		print('Executing pipeline: %s'% self.pipeline_name)



class ProphetForecastingPipeline(GenericForecastingPipeline):
	def __init__(self, pipeline_data):
		super().__init__('ProphetForecastingPipeline', pipeline_data)
	
	def execute_pipeline(self):
		self.setup()
		self.generate()	
	
	def setup(self):
		
		pricing = PricingDataframe()
		df = pricing.get_dataframe()		
		
		all_inf_or_nan = df.isin([np.inf, -np.inf, np.nan])
		df = df[~all_inf_or_nan]
		
		print (self.pipeline_data.start_date, self.pipeline_data.end_date)
		df = df[df['Date'] >= self.pipeline_data.start_date]
		df = df[df['Date'] <= self.pipeline_data.end_date]
		
		self.pipeline_data.master_dataframe = df
		
	
	def remove_inf_nan(self, df):
		all_inf_or_nan = df.isin([np.inf, -np.inf, np.nan])
		df = df[~all_inf_or_nan]
		return df
	
	def generate(self):
	
		query = self.pipeline_data.query_type.lower()
		if 'best' in query or 'worst' in query:
			self.generate_for_best_and_worst_routes()
		elif 'priority' in query:
			self.generate_for_priority_routes()
		else:		
			self.generate_for_custom_routes()
	
	def generate_for_best_and_worst_routes(self):		
		
		booking_data = []
		forecast_data = []
		df = self.pipeline_data.master_dataframe.copy()
		# decorator_df to extract, bkg_to_city, bkg_from_city information
		decorator_df = self.pipeline_data.master_dataframe.copy()
		query = self.pipeline_data.query_type.lower()
		
		count = int(''.join(filter(str.isdigit, query)))
		column = ''
		if 'to_city_routes' in query:
			column = ['bkg_to_city']
		elif 'from_city_routes' in query:
			column = ['bkg_from_city']
		else:
			column = ['bkg_from_city', 'bkg_to_city']
		
		if 'amount' in self.pipeline_data.yhat: 
			if 'best' in query:
				decorator_df = decorator_df.groupby(column)[self.pipeline_data.yhat].sum().nlargest(count)
			else:
				decorator_df = decorator_df.groupby(column)[self.pipeline_data.yhat].sum().nsmallest(count)
		else:
			if 'best' in query:
				decorator_df = decorator_df.groupby(column)[self.pipeline_data.yhat].count().nlargest(count)
			else:
				decorator_df = decorator_df.groupby(column)[self.pipeline_data.yhat].count().nsmallest(count)
	
		decorator_df = decorator_df.reset_index()
		print('Generating %s' % query)
		decorator_df = decorator_df.drop(labels=self.pipeline_data.yhat, axis=1)
		
		
		counter = 1
		if 'to_city_routes' in query:
			to_cities = decorator_df['bkg_to_city'].tolist()
			for to_city in to_cities:
				label = ("#%s: bkg_to_city: %s" % (counter, to_city))
				to_df = df[df['bkg_to_city'] == to_city]
				booking_data.append(self.generate_booking_data(label, to_df))
				forecast_data.append(self.generate_forecast_data(label, to_df))
				counter += 1
			
		elif 'from_city_routes' in query:
			from_cities = decorator_df['bkg_from_city'].tolist()
			for from_city in from_cities:
				label = ("#%s: bkg_from_city: %s" % (counter, from_city))
				from_df = df[df['bkg_from_city'] == from_city]
				booking_data.append(self.generate_booking_data(label, from_df))
				forecast_data.append(self.generate_forecast_data(label, from_df))
				counter += 1
		else:
			from_cities = decorator_df['bkg_from_city'].tolist()
			to_cities = decorator_df['bkg_to_city'].tolist()
			for cities in zip(from_cities, to_cities):
				from_city_str = cities[0]
				to_city_str = cities[1]
				from_city = df[df['bkg_from_city'] == from_city_str]
				from_to_df = from_city[from_city['bkg_to_city'] == to_city_str]
				label = ("#%s: bkg_from_city:%s___bkg_to_city:%s" % (counter, from_city_str, to_city_str))
				booking_data.append(self.generate_booking_data(label, from_to_df))
				forecast_data.append(self.generate_forecast_data(label, from_to_df))
				counter += 1
			
		self.pipeline_data.booking_data = booking_data
		self.pipeline_data.forecast_data = forecast_data		
	
	def generate_for_priority_routes(self):		
		
		booking_data = []
		forecast_data = []
		df = self.pipeline_data.master_dataframe.copy()
		# decorator_df to extract, bkg_to_city, bkg_from_city information
		decorator_df = self.pipeline_data.master_dataframe.copy()
		query = self.pipeline_data.query_type.lower()

		
		#resource_file = os.getcwd() + '\\chalicelib\\resources\\catalogs\\priority_routes.csv'
		resource_file = os.getcwd() + '/chalicelib/resources/catalogs/priority_routes_formatted.csv'
		#resource_file = os.getcwd() + '\\chalicelib\\resources\\catalogs\\priority_routes_formatted_stripped.csv'
		
		routes_df = pd.read_csv(resource_file, low_memory=False)
		
		#routes_df = routes_df.filter(['INDEX', 'SOURCE','DESTINATION','POLL_PRIORITY'], axis=1)
		routes_df = routes_df.filter(['INDEX', 'NORMALIZED_SOURCE','NORMALIZED_DESTINATION','POLL_PRIORITY'], axis=1)
		
		print_dataframe(routes_df, "routes_df", False)
		if query == 'ALL_HIGH_PRIORITY'.lower():
			routes_df = routes_df.loc[routes_df['POLL_PRIORITY'] == 'HIGH_PRIORITY']
		elif query == 'ALL_MODERATE_PRIORITY'.lower():
			routes_df = routes_df.loc[routes_df['POLL_PRIORITY'] == 'MODERATE_PRIORITY']
		elif query == 'ALL_LOW_PRIORITY'.lower():
			routes_df = routes_df.loc[routes_df['POLL_PRIORITY'] == 'LOW_PRIORITY']
		
		from_cities = routes_df['NORMALIZED_SOURCE'].tolist()
		to_cities = routes_df['NORMALIZED_DESTINATION'].tolist()
		
		print ('Query = %s, From Cities - %s, To Cities - %s' % (query, from_cities, to_cities))
		
		for cities in zip(from_cities, to_cities):
			from_city = cities[0]
			to_city = cities[1]
			
			'''
			if from_city == "Bengaluru":
				from_city = "Bengaluru (Bangalore)"
			if to_city == "Bengaluru":
				to_city = "Bengaluru (Bangalore)"
			
			if from_city == "New Jalpaiguri Siliguri":
				from_city = "New Jalpaiguri"
			if to_city == "New Jalpaiguri Siliguri":
				to_city = "New Jalpaiguri"
			'''
			
			label = ("bkg_from_city:%s___bkg_to_city:%s" % (from_city, to_city))
			from_df = df[df['bkg_from_city'] == from_city]
			print('PRIORITY: ', query, label)
			from_to_df = from_df[from_df['bkg_to_city'] == to_city]			
			print('Custom: ', query, label, ', dataframe length: ',len(from_to_df))
			try:
				bd = self.generate_booking_data(label, from_to_df)
				fd = self.generate_forecast_data(label, from_to_df)
				booking_data.append(bd)
				forecast_data.append(fd)
			except ValueError as err:
				print ('#####################################################')
				print('PRIORITY: ', query, label, err)
				print ('#####################################################')
				pass			
		
		self.pipeline_data.booking_data = booking_data
		self.pipeline_data.forecast_data = forecast_data

	def generate_for_custom_routes(self):		
		
		booking_data = []
		forecast_data = []
		df = self.pipeline_data.master_dataframe.copy()
		# decorator_df to extract, bkg_to_city, bkg_from_city information
		decorator_df = self.pipeline_data.master_dataframe.copy()
		query = self.pipeline_data.query_type.lower()

		from_cities = self.pipeline_data.from_cities
		to_cities = self.pipeline_data.to_cities
		
		print ('Query = %s, From Cities - %s, To Cities - %s' % (query, from_cities, to_cities))
	
		if query == 'All_India'.lower():
			booking_data.append(self.generate_booking_data('All India', df))
			forecast_data.append(self.generate_forecast_data('All India', df))
			
		elif query == 'Custom_Only_To_Cities'.lower():
			for to_city in to_cities:
				label = ("bkg_to_city: %s" % to_city)
				print(query, label)
				to_df = df[df['bkg_to_city'] == to_city]
				booking_data.append(self.generate_booking_data(label, to_df))
				forecast_data.append(self.generate_forecast_data(label, to_df))
				
		elif query == 'Custom_Only_From_Cities'.lower():
			for from_city in from_cities:
				label = ("bkg_from_city: %s" % from_city)
				print(query, label)
				from_df = df[df['bkg_from_city'] == from_city]
				booking_data.append(self.generate_booking_data(label, from_df))
				forecast_data.append(self.generate_forecast_data(label, from_df))
				
		elif query == 'Custom'.lower():
			for from_city in from_cities:
				from_df = df[df['bkg_from_city'] == from_city]
				for to_city in to_cities:
					label = ("bkg_from_city:%s___bkg_to_city:%s" % (from_city, to_city))
					print('Custom: ', query, label)
					from_to_df = from_df[from_df['bkg_to_city'] == to_city]
					print('Custom: ', query, label, ', dataframe length: ',len(from_to_df))
					booking_data.append(self.generate_booking_data(label, from_to_df))
					forecast_data.append(self.generate_forecast_data(label, from_to_df))
		
		elif query == 'Custom_Aggregate_From_Cities'.lower():
			bool_true_from_df = df.bkg_from_city.isin(from_cities)
			from_df = df[bool_true_from_df]
			for to_city in to_cities:
				label = ("bkg_from_city:%s___bkg_to_city:%s" % (from_cities, to_city))
				from_to_df = from_df[from_df['bkg_to_city'] == to_city]
				print('Custom_Aggregate_From_Cities: ', query, label, ', dataframe length: ',len(from_to_df))
				booking_data.append(self.generate_booking_data(label, from_to_df))
				forecast_data.append(self.generate_forecast_data(label, from_to_df))
					
		elif query == 'Custom_Aggregate_To_Cities'.lower():
			bool_true_to_df = df.bkg_to_city.isin(to_cities)
			to_df = df[bool_true_to_df]				
			for from_city in from_cities:
				from_to_df = to_df[to_df['bkg_from_city'] == from_city]					
				label = ("bkg_from_city:%s___bkg_to_city:%s" % (from_city, to_cities))
				print('Custom_Aggregate_To_Cities: ', query, label, ', dataframe length: ',len(from_to_df))
				booking_data.append(self.generate_booking_data(label, from_to_df))
				forecast_data.append(self.generate_forecast_data(label, from_to_df))
					
		elif query == 'Custom_Aggregate_From_And_To_Cities'.lower():
			bool_true_from_df = df.bkg_from_city.isin(from_cities)
			bool_true_to_df = df.bkg_to_city.isin(to_cities)
			ddf = df[bool_true_from_df & bool_true_to_df]	
			if self.pipeline_data.route_or_zone_based == 'Zone_Based':
				if len(self.pipeline_data.from_zones) == 1 and len(self.pipeline_data.to_zones) == 1:
					label = ("bkg_from_zone:%s___bkg_to_zone:%s" % 
						(self.pipeline_data.from_zones[0], self.pipeline_data.to_zones[0]))							
				else:
					label = ("bkg_from_zone:%s___bkg_to_zone:%s" % 
						(self.pipeline_data.from_zones, self.pipeline_data.to_zones))	
			else:
				label = ("bkg_from_city:%s___bkg_to_city:%s" % (from_cities, to_cities))				
			print('Custom_Aggregate_From_And_To_Cities: ', query, label, ', dataframe length: ',len(ddf))
			booking_data.append(self.generate_booking_data(label, ddf))
			forecast_data.append(self.generate_forecast_data(label, ddf))
				
		self.pipeline_data.booking_data = booking_data
		self.pipeline_data.forecast_data = forecast_data

	def generate_booking_data(self, label, df):
		df = self.remove_inf_nan(df)
		
		if 'amount' in self.pipeline_data.yhat:
			df = df.groupby('Date', as_index=False).sum()
		else:
			df = df.groupby('Date', as_index=False).count()
		
		df = df.filter(['Date', self.pipeline_data.yhat], axis=1)
		df.columns = ['ds', 'y']
		
		df_to_json = df.to_json(orient='records')
		return json.dumps({"label": label, "df": df_to_json})
		
	def generate_forecast_data(self, label, df):
			
		df = self.remove_inf_nan(df);
		
		if 'amount' in self.pipeline_data.yhat:
			df = df.groupby('Date', as_index=False).sum()
		else:
			df = df.groupby('Date', as_index=False).count()
		
		df = df.filter(['Date', self.pipeline_data.yhat], axis=1)
		df.columns = ['ds', 'y']
		
		print_dataframe (df, 'Prophet input dataframe: ', True)
		
		prophet = ProphetModeller(df, 
			self.pipeline_data.future_periods, 
			self.pipeline_data.do_log_transform, 
			self.pipeline_data.validate_existing)
	
		# set capacity to twice the mean value
		if self.pipeline_data.set_capacity is True:
			capacity['cap'] = df.groupby('y').mean() * 2
			prophet.set_capacity(capacity)
		
		prophet.predict()
		
		print_dataframe(prophet._forecast, 'Prophet Forecast Data: '+ self.pipeline_data.yhat, True)
		cmp_df = ProphetUtils.get_comparison_dataframe(df, prophet._forecast)
		print_dataframe(cmp_df, 'Prophet Forecast Data', True)
		
		training_days = (self.pipeline_data.end_date - self.pipeline_data.start_date).days + 2 # + 2 for start and end dates
		
		print ('training_days', training_days)
		cmp_df = cmp_df.reset_index()
		
		df_to_json = cmp_df.to_json(orient='records')
									
		if self.pipeline_data.debug:
			plotly_graph =	ProphetUtils.get_comparison_graph(
										cmp_df, 
										self.pipeline_data.future_periods, 
										training_days)
			'''
			print_seperator_start('prophet._forecast start')
			print(prophet._forecast)
			print_seperator_end('prophet._forecast end')
			print_seperator_start('prophet.components_figure start')
			print(prophet.main_figure)
			print_seperator_end('prophet.components_figure end')
			'''
			plt.show()
		
		if self.pipeline_data.validate_existing:
			#target_len = len(self.pipeline_data.future_periods)
			print_dataframe(df, 'Target Dataframe', False)
			target_df = df.tail(self.pipeline_data.future_periods)
			if len(target_df) > 0:
				# We will apply the surge and compute errors in prediction with that value
				surge_factor = self.pipeline_data.weekend_surge_factor
				if surge_factor > 0.0:
					target_df_y = target_df['y'] # We are preserving the index here
					print ("######### target_df_y: ", len(target_df_y))
					# reindex target_df on date for multiplication
					target_df = target_df[['ds','y']]
					target_df.index = target_df['ds'].values
					target_df.drop(['ds'],axis=1,inplace=True)
					print_dataframe(target_df, 'Original Dataframe', False)
										
					# surge on Sat, half surge on Friday and Sunday (i.e day before and after)
					dow_map = { 6:'Sun', 0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat'}
					dow_surge_map = { 6:(1+surge_factor/2), 0:1, 1:1, 2:1, 3:1, 4:(1+surge_factor/2), 5:(1+surge_factor)}
					
					df_surge = pd.DataFrame(columns=['Date', 'Surge'])
					df_surge['Date'] = df.tail(self.pipeline_data.future_periods)['ds']
					df_surge['Surge'] = df_surge['Date'].apply(lambda x: dow_surge_map[x.weekday()])
					# reindex on date again
					df_surge.index = df_surge['Date'].values
					df_surge.drop(['Date'],axis=1,inplace=True)
					print_dataframe(df_surge, 'Weekend Surge', False)
					surge_factor += 1
					
					print_dataframe(target_df,	'Before Applying Surge', False)
					target_df = target_df.mul(df_surge['Surge'], axis="index")
					# reindex to original value
					target_df.index = target_df_y.index
					target_df = target_df['y']
					print_dataframe(target_df,	'After Applying Surge', False)
					
					rmse = prophet.rmse(target_df)
					mape = prophet.mean_absolute_percentage_error(target_df)
					smape = prophet.smape(target_df)
					errors = " (RMSE: " + str(rmse) + ", MAPE: " + str(mape) +\
						", SMAPE: " + str(smape) + ", Surge Factor: " + str(surge_factor-1) + ")"
					label += errors
				else:
					target_df = target_df['y']
					rmse = prophet.rmse(target_df)
					mape = prophet.mean_absolute_percentage_error(target_df)
					smape = prophet.smape(target_df)
					errors = " (RMSE: " + str(rmse) + ", MAPE: " + str(mape) + ", SMAPE: " + str(smape) + ")"
					label += errors
					
		return json.dumps({"label": label, "df": df_to_json})
		

def get_forecasting_trends(
	yhat,
	future_periods,
	do_log_transform,
	query_type,
	start_date,
	end_date,
	from_cities,
	to_cities,
	validate_existing,
	set_capacity,
	rest_call,
	debug
	):

	prophetWrapper = ProphetForecastingPipeline(
		yhat,
		future_periods,
		do_log_transform,
		query_type,
		start_date,
		end_date,
		from_cities,
		to_cities,
		validate_existing,
		set_capacity,
		rest_call,
		debug
		)
	
	prophetWrapper.setup()
	
	forecast_data = prophetWrapper.generate()	
	
	return forecast_data
			

def get_forecasting_trends_json_wrapper(_json_data):

	_json_to_controls=json.loads(_json_data)
	print (_json_to_controls)	
	
	start_date = datetime.strptime(
			(_json_to_controls["start_date"]).split()[0], '%Y-%m-%d')
	end_date = datetime.strptime(
			(_json_to_controls["end_date"]).split()[0], '%Y-%m-%d')
	
	print (_json_to_controls["start_date"], _json_to_controls["end_date"])
	print (start_date, end_date)
	
	pipedata = ForcastingPipelineData()
	pipedata.yhat = _json_to_controls["primary_input_field"]
	pipedata.future_periods = _json_to_controls["forecast_period"]
	pipedata.do_log_transform = _json_to_controls["model_type"] == 'Log'
	pipedata.query_type = _json_to_controls["query_type"]
	pipedata.route_or_zone_based = _json_to_controls["route_or_zone_based"]
	pipedata.start_date = start_date
	pipedata.end_date = end_date
	pipedata.from_cities = _json_to_controls["from_cities"]
	pipedata.to_cities = _json_to_controls["to_cities"]
	pipedata.from_zones = _json_to_controls["from_zones"]
	pipedata.to_zones = _json_to_controls["to_zones"]
	pipedata.validate_existing = _json_to_controls["forecast_and_validate"]
	pipedata.set_capacity = False
	pipedata.rest_call=_json_to_controls["rest_call"]
	pipedata.debug = False
	pipedata.weekend_surge_factor = float(_json_to_controls["weekend_surge"])
	pipedata.seasonal_surge_factor = float(_json_to_controls["seasonal_surge"])	
		
	
	prophetPipeline = ProphetForecastingPipeline(pipedata)
	executer = ForecastingPipelineExecuter()
	executer.add(prophetPipeline)
	executer.execute()
	
	return pipedata

	
#unit test
def run_unit_test():
	'''
	
	# All India
	json_data = {	
		"model_type": "Linear", 
		"forecast_and_validate": "false", 
		"forecast_period": 30, 
		"query_type": "All_India", 
		"from_cities": "", 
		"to_cities": "", 
		"primary_input_field": "bkg_user_id", 
		"start_date": "2015-07-01 00:00:00", 
		"end_date": "2018-04-15 00:00:00", 
		"plot_components": "false", 
		"rest_call": "true"
	}
	'''
	
	# Custom
	json_data = {
		"model_type": "Linear", 
		"forecast_and_validate": "false", 
		"forecast_period": 30, 
		"query_type": "Custom", 
		"from_cities": ["Delhi"], 
		"to_cities": ["Agra"], 
		"primary_input_field": "bkg_user_id", 
		"start_date": "2015-07-01 00:00:00", 
		"end_date": "2018-04-15 00:00:00", 
		"plot_components": "false", 
		"rest_call": "true"
	}	
	
	return get_forecasting_trends_json_wrapper(json.dumps(json_data))
'''	
def profile_unit_test():
	cProfile.run('run_unit_test()', sort='time')

@numba.jit
def profile_numba_unit_test():
	cProfile.run('run_unit_test()', sort='time')

@numba.jit
def numba_run_unit_test():
	print(datetime.now(), 'Started the test')
	run_unit_test()
	print(datetime.now(), 'Ended the test')
	
#run_unit_test()
#profile_unit_test()
#profile_numba_unit_test()
#numba_run_unit_test()
'''
