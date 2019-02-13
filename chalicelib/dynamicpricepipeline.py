
import pandas as pd
import numpy as np
import scipy

import math as mt

import os
import sys
import psycopg2
import matplotlib
import matplotlib.pyplot as plt
from numpy.random import normal
import calendar
from scipy.optimize import curve_fit
import scipy.optimize as optimize

#%matplotlib inline
plt.rcParams['figure.figsize'] = (16,8)
import warnings
warnings.filterwarnings('ignore')

#import plotly
import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

sys.path.append('./chalicelib')
sys.path.append('.')
from forecastingpipeline import *

MINIMUM_MEDIAN_VAL = 20

class DyPricePipelineData:
	def __init__(self):
		''' Common pipeline data '''
		self.city_src = None
		self.city_dst = None
		self.date_start = None
		self.date_end = None
		self.time_start = None
		self.time_end = None
		self.forecasting_days = None
		self.forecast_and_validate = False
		
		''' ForecastingDyPricePipeline data '''
		self.df_booking = pd.DataFrame()
		self.df_forecasting = pd.DataFrame()
		self.df_base_capacity = pd.DataFrame()
		self.elasticity = -2.0
		self.base_price_start = 4000
		self.base_capacity_absolute = 'ForecastedValue'
		self.base_capacity_distribution = 'Median'
		self.base_price_low = []
		self.base_price_high = []
		self.capacities = []
		self.capacities_col = {}
		
		''' WeekendDyPricePipeline data '''
		self.weekend_surge_factor = 0.0
		''' SeasonalDyPricePipeline data '''
		self.seasonal_surge_factor = 0.0
		''' Other pipeline data '''
		''' Output held in below '''
		self.df_dynamicprice = pd.DataFrame()

'''
Pipeline execution interface
1. Create all different pipelines
2. Add them in order you want
3. Execute them serially after they are added
'''	
class DyPricePipelineExecuter:
	def __init__(self):
		self.pipeline_map = {}
		
	def add(self, pipeline):
		name = pipeline.pipeline_name
		if name not in self.pipeline_map:
			self.pipeline_map[name] = pipeline
	
	def execute(self):
		for pipeline in self.pipeline_map.values():
			pipeline.execute_pipeline()	

'''
Generic pipeline interface to provide the execution wrapper
'''			
class GenericDyPricePipeline:
	def __init__(self, name, pipeline_data):
		self.pipeline_name = name
		self.name = name
		self.pipeline_data = pipeline_data
	
	def execute_pipeline(self):
		print('Executing pipeline: %s'% self.pipeline_name)


'''
1. Create ForecastingDyPricePipeline instance
2. Creating forecasting period dataframe with ds, yhat
	a. use FB prophet to get the forecast data with yhat
	b. use simulate_demand() to create it if none provided
3. Create demand/price elasticity 
'''
		
class ForecastingDyPricePipeline(GenericDyPricePipeline):
	def __init__(self, pipeline_data):
		super().__init__('ForecastingDyPricePipeline', pipeline_data)
				
	def execute_pipeline(self):
		super().execute_pipeline()
		if len(self.pipeline_data.df_forecasting) == 0:
			# if no forecasting data is provided then create your own
			print("Simulate pipeline")
			self.pipeline_data.df_forecasting = self.simulate_demand(
					self.pipeline_data.date_start,
					self.pipeline_data.date_end,						
					baseline_booking_level=50, 
					noise_level=2.0, 
					amplitude=5)
		self.construct_forecast_median_dataframe(self.pipeline_data)
		self.run_optimization()
		print_dataframe(self.pipeline_data.df_forecasting, 'Forecasting dataframe', False)
	
	# Below function procuces array of size 'forecast_period' with random 
	# gausian/probability distribution around the baseline_booking_level +- noise_level
	def simulate_demand(self, 
			date_start,
			date_end,
			seasonal_period=7.0, 
			baseline_booking_level=35.0, 
			noise_level=5.0,
			amplitude=3.0, 
			phase=2.0*mt.pi/7.0, 
			trend=0.06
		):
		"""Simulate a demand curve over a forecast period.

		Parameters
		----------

		forecast_period (integer):
		   Length of the forecast period (in days)

		seasonal_period (float):
		   Length of the seasonal component. Typically 7 days for weekly
		   fluctuations.

		baseline_booking_level (float):
		   Baseline booking level

		noise_level (float):
		   Noise level (additive) of the booking numbers. This number
		   corresponds to the standard deviation of the normal distribution.

		amplitude (float):
		   Amplitude of the periodic signal

		phase (float):
		   phase shift of the signal (in radians)

		trend (float):
		   slope of the linear trend term. In units of bookings/day. Set to 0 if you
		   do not want to include a linear trend.

		Returns
		-------
		A numpy.ndarray (1D) containing the forecast numbers.
		"""

		forecast_days = (pd.to_datetime(date_end) - pd.to_datetime(date_start)).days
		
		forecast = np.linspace(1,forecast_days,forecast_days)

		# noise level: Draw random samples from a normal (Gaussian) distribution.
		noise = noise_level * normal(size=len(forecast))

		# Demand signal:
		demand = baseline_booking_level + trend * forecast + amplitude * \
				 np.cos(2.0 * mt.pi * forecast / seasonal_period + phase) + noise

		df_demand = pd.DataFrame(demand, 
			index=pd.date_range(
				start=date_start, end=date_end,closed='left'
			),columns=['yhat'])
		
		return df_demand
	
	
	def demand_price_elasticity(
			self, 
			price, 
			nominal_demand, 
			elasticity=-2.0, 
			nominal_price=5000.0
		):
		
		"""Returns demand given a value for the elasticity, nominal demand and nominal price.

		Parameters
		----------

		price (numpy.ndarray):
			one-dimensional price array. The length of that array should correspond to the
			length of the forecast period.

		nominal_demand (numpy.ndarray):
			one-dimensional forecasted booking array. The length of that array should
			correspond to the length of the forecast period.

		elasticity (float):
			value of the elasticity between price and demand. A value of e=-2 is reasonable.
			The above formula usually yields a negative value, due to the inverse nature of the 
			relationship between price and quantity demanded, as described by the "law of demand".
			e_=frac {d}D/D	/	{d}P/P
			For example, 
			(a) if the price increases by 5% and demand decreases by 5%, 
			then the elasticity at the initial price and quantity = −5%(D)/5(P)% = −1
			
			
		nominal_price (float):
			booking rate for which the forecast was computed.

		Returns
		-------

		A numpy.ndarray of expected demand.
		"""

		return nominal_demand * ( price / nominal_price ) ** (elasticity)

	# Objective function:
	def objective(
			self, 
			p_t, 
			nominal_demand=np.array([50,40,30,20]),
			elasticity=-2.0, 
			nominal_price=5000.0
		):
		
		"""
		Definition of the objective function. This is the function that want to minimize.
		(minus sign in front)

		Parameters
		----------

		p_t (numpy.ndarray):
			one-dimensional price array. The length of that array should correspond to the
			length of the forecast period.

		nominal_demand (numpy.ndarray):
			one-dimensional forecasted booking array. The length of that array should
			correspond to the length of the forecast period.

		elasticity (float):
			value of the elasticity between price and demand. A value of e=-2 is
			reasonable.

		nominal price (float):
			booking rate for which the forecast was computed.

		Returns
		-------

		Value of the objective function (float).

		Note: here we're trying to minimize the objective function. That's where the
		minus sign comes_in.

		"""
		pd_elasticity = self.demand_price_elasticity(p_t, nominal_demand=nominal_demand,
															elasticity=elasticity,
															nominal_price=nominal_price)
		result = (-1.0 * np.sum( p_t *  pd_elasticity))
		
		log_this = False
		iteration = 0
		if log_this:
			iteration = iteration + 1
			print('Iteration : ', iteration)
			print('\n*******************************************************************************')
			print('\nRunning Objective function : p_t\n', p_t)
			print('\nRunning Objective function : pd_elasticity\n', pd_elasticity)
			print('\nRunning Objective function : result= np.sum( p_t *  pd_elasticity)\n', result)
			print('*******************************************************************************\n')
			
		return result / 100
															
	# Constraints:
	def constraint_1(
			self, 
			p_t
		):
		""" This constraint ensures that the prices are positive."""
		return p_t


	def constraint_2(
			self, 
			p_t, 
			capacity=20, 
			forecasted_demand=35.0,
			elasticity=-2.0, 
			nominal_price=5000.0
		):
		""" This constraint ensures that the demand does not exceed	capacity.

		Parameters
		----------

		p_t (float):
			Room price

		capacity (integer):
			Capacity of the routes (in bookinds/day).

		forecasted_demand (float):
			Forecasted demand (in bookings) for that day

		elasticity (float):
			slope of the

		nominal_price (float):
			The price for which the forecasted_demand was computed.

		Returns
		-------
		Returns an array of excess capacity.

		"""
		return capacity - self.demand_price_elasticity(
				p_t, 
				nominal_demand=forecasted_demand,
				elasticity=elasticity,
				nominal_price=nominal_price
		)
	
	def construct_forecast_median_dataframe(self, pipedata):
		original_df_forecast = pipedata.df_forecasting
		original_df_booking = pipedata.df_booking
		
		print_dataframe(original_df_booking, 'original_df_booking: ', False)
		print_dataframe(original_df_forecast, 'original_df_forecast: ', False)
		
		# reindex with time as index
		df_booking = original_df_booking.copy()
		df_booking.index = df_booking.ds
		df_booking = df_booking.rename_axis('')
		
		# reindex with time as index
		df_forecast = original_df_forecast.copy()
		df_forecast.index = df_forecast.ds
		df_forecast['yhat'] = df_forecast['yhat'].apply(lambda x: int(x))
		print_dataframe(df_forecast,'df_forecast', False)
		
		median_val = 0
		if pipedata.base_capacity_absolute == 'ForecastedValue':
			pass
		else:
			df_base_capacity = pd.DataFrame(columns=['base_capacity'])
			if pipedata.base_capacity_absolute == 'MedianOfForecastedValue':
				df_base_capacity['base_capacity'] = df_forecast['yhat']
			elif pipedata.base_capacity_absolute == 'MedianOfLast3Months':
				df_base_capacity['base_capacity'] = df_booking.tail(90)['y']	
			elif pipedata.base_capacity_absolute == 'MedianOfLast6Months':
				df_base_capacity['base_capacity'] = df_booking.tail(180)['y']
			elif pipedata.base_capacity_absolute == 'MedianOfLast9Months':
				df_base_capacity['base_capacity'] = df_booking.tail(270)['y']		
			elif pipedata.base_capacity_absolute == 'MedianOfLast12Months':
				df_base_capacity['base_capacity'] = df_booking.tail(365)['y']	
				
				# NOTE: ABCDE
				# We have not included frame for future period with median values for MedianOfLastXXMonths, 
				# we will do it later, check NOTE ABCDE later
			else:
				if pipedata.base_capacity_absolute == 'MedianOfLast3MonthsAndForecastedValue':
					df_base_capacity['base_capacity'] = df_booking.tail(90)['y']
				elif pipedata.base_capacity_absolute == 'MedianOfLast6MonthsAndForecastedValue':
					df_base_capacity['base_capacity'] = df_booking.tail(180)['y']				
				elif pipedata.base_capacity_absolute == 'MedianOfLast9MonthsAndForecastedValue':
					df_base_capacity['base_capacity'] = df_booking.tail(270)['y']				
				elif pipedata.base_capacity_absolute == 'MedianOfLast12MonthsAndForecastedValue':
					df_base_capacity['base_capacity'] = df_booking.tail(365)['y']			
			
				print (pipedata.base_capacity_absolute)
				forecasting = pd.DataFrame(columns=['base_capacity'])
				forecasting['base_capacity'] = df_forecast['yhat']
				print_dataframe(df_base_capacity,'booking', False)			
				print_dataframe(forecasting,'forecasting', False)
				df_base_capacity = df_base_capacity.append(forecasting)
				
			print_dataframe(df_base_capacity,'df_base_capacity', False)
			
			dow_map = { 6:'Sun', 0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat'}
			df_base_capacity['Date'] = df_base_capacity['base_capacity'].index
			df_base_capacity['Weekday'] = df_base_capacity['Date'].apply(lambda x: dow_map[x.weekday()])
			
			log_weekly_median = False
			if log_weekly_median:
				grouped_df = df_base_capacity.groupby('Weekday')
				for key, values in grouped_df:
					weekly_df = grouped_df.get_group(key)
					weekly_median = weekly_df['base_capacity'].median()
					weekly_mean = weekly_df['base_capacity'].mean()
					label = 'df_base_capacity: ' + str(key) + ', base_capacity_meadian: ' + str(weekly_median) + ', base_capacity_mean: ' +  str(weekly_mean)
					print_dataframe(weekly_df.sort_values(by=['base_capacity']), label, log_weekly_median)
					
					
			groupby_dow = df_base_capacity.groupby('Weekday')['base_capacity'].median()
			print ('groupby_dow\n', groupby_dow)
			df_base_capacity['base_capacity'] = df_base_capacity['Date'].apply(lambda x: groupby_dow.loc[dow_map[x.weekday()]])
			
			if 'ForecastedValue' not in pipedata.base_capacity_absolute:
				# NOTE: ABCDE
				forecasting = pd.DataFrame(columns=['Weekday','base_capacity'])
				forecasting['Weekday'] = df_forecast['ds'].apply(lambda x: dow_map[x.weekday()])
				forecasting['base_capacity'] = df_forecast['ds'].apply(lambda x: groupby_dow.loc[dow_map[x.weekday()]])				
				df_base_capacity = df_base_capacity.append(forecasting)
				
			print_dataframe(df_base_capacity,'df_base_capacity_with_weekdays', False)
		
		def set_median(x):
			if x > MINIMUM_MEDIAN_VAL:
				return int(x)
			else:
				return MINIMUM_MEDIAN_VAL
				
		# applying 15% business growth surge factor
		business_growth_factor = 0.15 * (len(df_forecast)/365) #15% YoY
		print ('business_growth_factor', business_growth_factor)
		if pipedata.base_capacity_distribution == 'Median':	
			median_val = df_base_capacity['base_capacity'].median()
			median_val = int(median_val + (median_val * business_growth_factor))
			if median_val < MINIMUM_MEDIAN_VAL:
				median_val = MINIMUM_MEDIAN_VAL
			df_forecast['base_capacity'] = df_forecast['yhat'].apply(lambda x: set_median(median_val))
		else:		
			df_base_capacity = df_base_capacity.tail(len(df_forecast))
			df_base_capacity = df_base_capacity.rename_axis('')
			df_forecast = df_forecast.rename_axis('')			
			df_forecast['base_capacity'] = df_base_capacity['base_capacity'].apply(lambda x: set_median(x + (x * business_growth_factor)))
			
		pipedata.df_base_capacity = df_forecast
		print_dataframe(pipedata.df_base_capacity,'pipedata.df_base_capacity', False)
		
	def run_optimization(self):
		# Let's run the optimization algorithm for all segments
		# Median, Median+10%, Median+20%, Median+30%, Median+40% ... Median+300%

		capacities = self.pipeline_data.capacities
		base_price_lows = self.pipeline_data.base_price_low
		base_price_highs = self.pipeline_data.base_price_high

		# Forecasted demand : Median value
		yhat_demand = self.pipeline_data.df_base_capacity['base_capacity'].values
		
		optimization_results = {}
		for capacity, base_price_low, base_price_high in zip(capacities, base_price_lows, base_price_highs):

			# Nominal price associated with forecasted demand:
			nominal_price = self.pipeline_data.base_price_start
			# Forecasted demand:
			
			nominal_demand = yhat_demand
			
			#print ('Capacity : ', capacity)
			#print ('Nominal Demand : ', nominal_demand)
			# Assumed price elasticity:
			# !!!elasticity = -2.0

			# Starting values:
			p_start = self.pipeline_data.base_price_start * np.ones(len(nominal_demand))

			# bounds on the prices. Let's stick with reasonable values.
			# One could be more sophisticated here and apply constraints
			# that limit the prices to be in range of what competitors
			# are charging, for example.
			bounds = tuple(
				(
					base_price_low, 
					base_price_high
				) 
				for p in p_start
			)

			# Constraints:
			constraints = (
				{
					'type': 'ineq', 'fun':  
					lambda x:  self.constraint_1(x)
				},
				{
					'type': 'ineq', 'fun':  
					lambda x, 
						capacity=capacity,
						forecasted_demand=nominal_demand,
						elasticity=self.pipeline_data.elasticity,
						nominal_price=nominal_price: self.constraint_2
						(
							x,
							capacity=capacity,
							forecasted_demand=nominal_demand,
							elasticity=self.pipeline_data.elasticity,
							nominal_price=nominal_price
						)
				}
			)
			opt_results = optimize.minimize(
				self.objective, 
				p_start, 
				args=(
					nominal_demand,
					self.pipeline_data.elasticity,
					nominal_price
				),
				method='SLSQP', bounds=bounds,
				constraints=constraints
			)

			optimization_results[capacity] = opt_results
		'''
		for k, v in optimization_results.items():
			print ('optimization_results key:', k)
			print ('optimization_results val:', v)
		'''
		
		# Plotting the resulting rates vs dates.
		time_array = np.linspace(1,len(nominal_demand),len(nominal_demand))
		dynamicprice_df = pd.DataFrame(index=time_array)
		
		for capacity in optimization_results.keys():
			dynamicprice_df = pd.concat(
					[
						dynamicprice_df,
						pd.DataFrame(
							optimization_results[capacity]['x'],
							columns=['{}'.format(capacity)],
							index=time_array)
					],
					axis=1
			)

		print_dataframe(dynamicprice_df, 'run_optimization->dynamicprice_df', False)
		'''
		datelist = pd.date_range(
			start=self.pipeline_data.date_start, 
			end=self.pipeline_data.date_end, closed='right').tolist() # closed=None includes both start and end date 
	
		# if last booking date for some route is less than 
		
		if self.pipeline_data.forecast_and_validate:
			shift_date = self.pipeline_data.forecasting_days + 1 # plus 1 to include end date
			dynamicprice_df.index = [ (x.date() - timedelta(shift_date)) for x in datelist]
		else:
			dynamicprice_df.index = [ x.date() for x in datelist]
		'''
		
		datelist = self.pipeline_data.df_forecasting['ds'].tolist()
		dynamicprice_df.index = [ x.date() for x in datelist]
		
		self.pipeline_data.df_dynamicprice = dynamicprice_df
		
		return dynamicprice_df
	
	def publish(
			self, 
			dynamicprice_df
		):
		# wite it to database if needed
		dynamicprice_df.to_csv('rates.csv')
		
class WeekendDyPricePipeline(GenericDyPricePipeline):
	def __init__(self, pipeline_data):
		super().__init__('WeekeendDyPricePipeline', pipeline_data)
		
	def execute_pipeline(self):
		super().execute_pipeline()
		surge_factor = self.pipeline_data.weekend_surge_factor
		df = self.pipeline_data.df_dynamicprice
		print (self.pipeline_name, surge_factor, len(df))
		# surge on Sat, half surge on Friday and Sunday (i.e day before and after)
		df_surge = pd.DataFrame(columns=['Date', 'Surge'])
		dow_map = { 6:'Sun', 0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat'}
		dow_surge_map = { 6:(1+surge_factor/2), 0:1, 1:1, 2:1, 3:1, 4:(1+surge_factor/2), 5:(1+surge_factor)}
		df_surge['Date'] = self.pipeline_data.df_dynamicprice.index
		df_surge['Surge'] = df_surge['Date'].apply(lambda x: dow_surge_map[x.weekday()])
		# reindex on date again
		df_surge.index = df_surge['Date'].values
		df_surge.drop(['Date'],axis=1,inplace=True)
		print_dataframe(df_surge, 'Weekend Surge', False)
		if surge_factor > 0.0 and len(df) > 0:
			surge_factor += 1
			print (self.pipeline_name, surge_factor)
			print_dataframe(
				df.tail(14),	self.name + ' - Before Applying Surge', False)
			#df = df[df.columns].multiply(df_surge['Surge'], axis="index")
			df = df.mul(df_surge['Surge'], axis="index")
			print_dataframe(
				df.tail(14),	self.name + ' - After Applying Surge', False)
			self.pipeline_data.df_dynamicprice = df

		
class SeasonalDyPricePipeline(GenericDyPricePipeline):
	def __init__(self, pipeline_data):
		super().__init__('FestivalDyPricePipeline', pipeline_data)
		
	def execute_pipeline(self):
		super().execute_pipeline()
		surge_factor = self.pipeline_data.seasonal_surge_factor
		df = self.pipeline_data.df_dynamicprice
		print (self.pipeline_name, surge_factor, len(df))
		if surge_factor > 0.0 and len(df) > 0:
			surge_factor += 1
			print (self.pipeline_name, surge_factor)
			print_dataframe(
				df.tail(5),	self.name + ' - Before Applying Surge', False)
			df = df * surge_factor
			print_dataframe(
				df.tail(5),	self.name + ' - After Applying Surge', False)
			print ('\After\n', df.tail(5))
			self.pipeline_data.df_dynamicprice = df
			
def plot_booking_forecast(
		forecasted_demand, 
		plotgraph=True
	):
	# Let's plot the booking forecast using plotly:

	booking = [
				go.Scatter(
					x=forecasted_demand.index, y=forecasted_demand['yhat'],
					name='Forecasted Vehicle Bookings for the next 90 days'
				)
			]

	layout_occ = go.Layout(
					title='Forecasted Vehicle Bookings -- Each trip is for Rs 5000',
					xaxis={'title':'Day'},
					yaxis={'title':'Number of Vehicles Booked'},
					shapes=[
						{
							'type':'line',
							'x0':'2018-09-30',
							'x1':'2018-10-30',
							'y0':80.0, 'y1':80.0,
							'line': 
							{
								'color': 'rgb(50, 171, 96)',
								'width': 4, 'dash':'dashdot'
							}
						}
					]
				)
	fig = go.Figure(data=booking, layout=layout_occ)
	if plotgraph:
		py.sign_in('abir', 'RpuARUVZC7oTDiAGtu2H')
		full_file_name = os.getcwd() + '/occ_ts.png'
		
		plot(
			fig, 
			show_link=False, 
			filename=full_file_name, 
			image = 'png', 
			image_width=2500, 
			image_height=1250
		)
		#iplot(fig, filename='occ_ts')
	return fig


		
def construct_forecast_datatable(pipedata):
	
	df_forecast = pipedata.df_forecasting
	df_dynamicprice = pipedata.df_dynamicprice
	capacities_col = pipedata.capacities_col
	base_price = pipedata.base_price_start
	df_base_capacity = pipedata.df_base_capacity
	forecast_and_validate = pipedata.forecast_and_validate
		
	df_to_show = df_dynamicprice.copy()
	#df_to_show = df_to_show[np.sort(np.asarray(df_to_show.columns))]
	#df_to_show = df_to_show[np.asarray(df_to_show.columns)]
	
	
	# Renaming the columns:
	if forecast_and_validate:
		df_total = pd.DataFrame(columns=['Realized Booking', 'Total DP', 'Total SP', 'Yield','Manual Count Quotation', 'Manual Count Booking','Count Quotation', 'Count Booking'])
		df_total['Realized Booking'] = df_forecast['y']
		df_total.index = df_forecast['ds'].values
	else:
		df_total = pd.DataFrame(columns=['Total DP', 'Total SP', 'Yield','Manual Count Quotation', 'Manual Count Booking','Count Quotation', 'Count Booking'])
	
	print_dataframe(df_forecast,'df_forecast', False)
	print_dataframe(df_base_capacity,'df_base_capacity', False)
	print_dataframe(df_dynamicprice,'df_dynamicprice', False)
	
	
	capacity_cols = df_to_show.select_dtypes(include=[np.number]).columns.tolist()
	#find the capacity delta b/n 2 columns
	capacities = capacity_cols.copy()
	# add 0 to compute detals b/n 2 consecutive columns
	capacities.insert(0,'0.0') 
	
	capaciated_percentage_increase = {}
	for i in range(0, len(capacities) - 1):
		percentage_increase = float(capacities[i+1]) - float(capacities[i])
		#print('%s - %s = %s' % (array[i+1], array[i], abs(val)))
		capaciated_percentage_increase[capacities[i+1]] = abs(percentage_increase)
	
	print_dataframe(df_to_show,'df_to_show', False)
	print_dataframe(df_total,'df_total', False)
	for date_index, prices in df_to_show.iterrows():
		#print (date_index, prices)
		dynamic_price_sum = 0
		static_price_sum = 0
		forecasted_booking_count = df_base_capacity.loc[date_index, 'yhat']	
		base_capacity = df_base_capacity.loc[date_index, 'base_capacity']
		#print ('forecasted_booking_count: ', forecasted_booking_count, ', base_capacity: ', base_capacity)		
		if forecast_and_validate:
			# compute actual reaslized price
			forecasted_booking_count = df_total.loc[date_index, 'Realized Booking']	
			
		current_cap_index = 0
		capacity_percentage = int(capacity_cols[current_cap_index])
		
		if forecasted_booking_count <= base_capacity:
			dynamic_price_sum = (forecasted_booking_count * prices[current_cap_index])		
		else:
			log_df = pd.DataFrame(columns=[
					'Index', 'Total #', 'Prev %', 'Curr %', 'Diff %', 'Surge Factor', 
					'Curr Bucket #', 'Loop Total #', 'Loop Price Rs', 'Loop Dy. Price', 'Remaining #'])
			current_count = base_capacity
			remaining_count = forecasted_booking_count - current_count	
			dynamic_price_sum = current_count * prices[current_cap_index]
			while (remaining_count > 0):
				current_cap_index += 1
				previous_percentage = int(capacity_cols[current_cap_index-1])	
				current_percentage = int(capacity_cols[current_cap_index])	
				diff_percentage = current_percentage - previous_percentage
				current_bucket_count = (base_capacity * diff_percentage)/100
				current_count += current_bucket_count
				if current_count > forecasted_booking_count:
					current_bucket_partial_count = abs(forecasted_booking_count - current_count)
					current_bucket_total_price = current_bucket_partial_count * prices[current_cap_index]
				else:
					current_bucket_total_price = current_bucket_count * prices[current_cap_index]
				
				dynamic_price_sum += current_bucket_total_price  
				remaining_count = remaining_count - current_bucket_count
				
				abs_current_bucket_count = 0
				if current_count > forecasted_booking_count:
					abs_current_bucket_count = abs(forecasted_booking_count - current_count)
				else:
					abs_current_bucket_count = current_bucket_count
				
				d = {
						'Index' 		: [current_cap_index], 
						'Total #' 		: [forecasted_booking_count],
						'Prev %' 		: [previous_percentage],
						'Curr %' 		: [current_percentage],
						'Diff %' 		: [diff_percentage],
						'Surge Factor' 	: [prices[current_cap_index]],
						'Curr Bucket #'	: [abs_current_bucket_count],
						'Loop Total #' 	: [current_count],
						'Loop Price Rs'	: [round(current_bucket_total_price,2)],
						'Loop Dy. Price': [round(dynamic_price_sum,2)],
						'Remaining #' 	: [remaining_count]
					}
				temp_df = pd.DataFrame(data=d)
				log_df = log_df.append(temp_df)
				
				'''
				print (
					"  Index: ", format(current_cap_index, '03d'), 
					", Total #: ", forecasted_booking_count,
					", Prev %: ", format(previous_percentage, '03d') ,
					", Curr %: ", format(current_percentage, '03d'),
					", Diff %: ", format(diff_percentage, '03d'),
					", Surge Factor: ", prices[current_cap_index],
					", Curr Bucket #: ", abs_current_bucket_count,
					", Loop Total #: ", current_count,
					", Loop Price Rs: ", round(current_bucket_total_price,2),
					", Loop Dy. Price: ", round(dynamic_price_sum,2),
					", Remaining #: ", remaining_count
				)
				'''
			
			#if len(log_df) > 0:
			#	print_dataframe (log_df, '###### Dynamic Pricing Datatable #####', True)	
			
		static_price_sum = forecasted_booking_count * base_price
		
		if dynamic_price_sum > 0 and static_price_sum > 0:
			df_total.loc[date_index, 'Total DP'] =  (dynamic_price_sum) 
			df_total.loc[date_index, 'Total SP'] = (static_price_sum)
			percent = round(float(((dynamic_price_sum - static_price_sum)/static_price_sum)*100), 2)
			if dynamic_price_sum > static_price_sum:
				df_total.loc[date_index, 'Yield'] = str('%s%s' % (percent, '% Increase'))
			else:
				df_total.loc[date_index, 'Yield'] = str('%s%s' % (percent, '% Drop'))

	'''
	if len(capacities_col) > 0:
		df_to_show.columns = ['Capacity left : {}'.format(capacities_col[x]) for x in df_to_show.columns]
	else:
		df_to_show.columns = ['Capacity left : {}'.format(x) for x in df_to_show.columns]
	'''
	capacities.pop(0)
	df_to_show.columns = ['{}'.format(capacities_col[int(x)]) for x in df_to_show.columns]
	
	df_to_show['Forecast-Act.'] = df_base_capacity['yhat']	
	
	if pipedata.base_capacity_absolute =='ForecastedValue':
		df_to_show['Base Capacity'] = df_base_capacity['yhat']			
	else:
		df_to_show['Base Capacity'] = df_base_capacity['base_capacity']
		
	
	
	#print (df_to_show)
	# Rounding the numbers:
	for col in df_to_show.columns:
		df_to_show[col] = df_to_show[col].apply(lambda x: round(x,4))

	
	dow_map = { 6:'Sun', 0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat'}
	df_to_show['Date'] = df_to_show.index
	df_to_show['Weekday'] = df_to_show['Date'].apply(
			lambda x: dow_map[x.weekday()])
	
	df_to_show['Date'] = df_to_show.apply(
			lambda row: row['Weekday']+" "+str(row['Date']), axis=1)
	df_to_show.index = df_to_show['Date'].values
	
	print_dataframe(df_to_show,'df_to_show', False)
	print_dataframe(df_total,'df_total', False)
		
	#df_to_show.drop(['Date','Weekday'],axis=1,inplace=True)
	df_total.index = df_to_show['Date'].values
	if forecast_and_validate:
		df_to_show['Realized Booking'] = df_total['Realized Booking']	
	df_to_show['Total DP'] = df_total['Total DP']
	df_to_show['Total SP'] = df_total['Total SP']
	df_to_show['Yield'] = df_total['Yield']	
	df_to_show['Count Quotation'] = df_total['Count Quotation']
	df_to_show['Count Booking'] = df_total['Count Booking']	
	df_to_show['Manual Count Quotation'] = df_total['Manual Count Quotation']
	df_to_show['Manual Count Booking'] = df_total['Manual Count Booking']	
	df_to_show['Additional Surge'] = 1
	
	#count_list=['0','0','0','0','0','0','0','0','0','0']
	count_list=[0,0,0,0,0,0,0,0,0,0]
	for index, row in df_to_show.iterrows():
		df_to_show.at[index, 'Count Quotation'] = str(count_list)
		df_to_show.at[index, 'Count Booking'] = str(count_list)	
		df_to_show.at[index, 'Manual Count Quotation'] = str(count_list)
		df_to_show.at[index, 'Manual Count Booking'] = str(count_list)	
	# reverse the columns
	print_dataframe(df_to_show,'df_to_show (before return)', False)	
	return df_to_show
	
	
def plot_capacitated_booking_forecast(dynamicprice_df, plotgraph=True):
	# Plotting the booking rate time series.
	# Let's focus on a single week cycle.

	price_levels = [go.Scatter(x=dynamicprice_df.head(7).index,
							   y=dynamicprice_df.head(7)['Capacity left : 20.0'],
							   name='Capacity Remaining : 20 vehicles'),
					go.Scatter(x=dynamicprice_df.head(7).index,
							   y=dynamicprice_df.head(7)['Capacity left : 40.0'],
							   name='Capacity Remaining : 40 vehicles'),
					go.Scatter(x=dynamicprice_df.head(7).index,
							   y=dynamicprice_df.head(7)['Capacity left : 60.0'],
							   name='Capacity Remaining : 60 vehicles'),
					go.Scatter(x=dynamicprice_df.head(7).index,
							   y=dynamicprice_df.head(7)['Capacity left : 80.0'],
							   name='Capacity Remaining : 80 vehicles')]

	layout_prices = go.Layout(title='Rate vs Reservation Date and Current Capacity Levels',
						   xaxis={'title':'Day'}, yaxis={'title':'Rate (Rs)'})

	fig = go.Figure(data=price_levels, layout=layout_prices)
	
	if plotgraph:
		py.sign_in('abir', 'RpuAoTDiAGtHC7oTDiAGtu2H')
		full_file_name = os.getcwd() + '\\price_levels_ts.png'
		plot(fig, show_link=False, filename=full_file_name, image = 'png', image_width=2500, image_height=1250)
		#iplot(fig, filename='price_levels_ts')
	return fig

def get_dynamic_price_json_wrapper(_json_data):

	_json_to_controls=json.loads(_json_data)
	print (_json_to_controls)	
	forecast_period = _json_to_controls["forecast_period"]
	from_cities = _json_to_controls["from_cities"]
	to_cities = _json_to_controls["to_cities"]
	
	start_date = datetime.strptime(
			(_json_to_controls["end_date"]).split()[0], '%Y-%m-%d')
	end_date = start_date + timedelta(forecast_period)
	
	forecasting_pipedata = get_forecasting_trends_json_wrapper(_json_data)
	booking_data_list = forecasting_pipedata.booking_data
	forecast_data_list = forecasting_pipedata.forecast_data
	forecast_and_validate = _json_to_controls["forecast_and_validate"]
	auto_publish = _json_to_controls["auto_publish"] == 'Yes'
	empty_df = pd.DataFrame()
	output_data = []
	output_data_map = {}
	for data in zip(booking_data_list, forecast_data_list):
		
		booking_result = json.loads(data[0])
		df_booking_result = booking_result['df']
		df_booking = pd.read_json(df_booking_result, convert_dates=['ds'])
		
		trend_result = json.loads(data[1])
		label = trend_result["label"]		
		df_trend_result = trend_result['df']
		df_forecast = pd.read_json(df_trend_result, convert_dates=['ds'])
		
		try:		
			print ('#########################')
			print_dataframe(df_booking, 'Booking Dataframe', False)
			print_dataframe(df_forecast, 'Forecast Dataframe', False)
			
			
			# get only forecast data, trim trend data
			df_forecast_only = df_forecast.tail(forecast_period)
			
			pipedata = DyPricePipelineData()
			pipedata.date_start = start_date.strftime('%Y-%m-%d')
			pipedata.date_end = end_date.strftime('%Y-%m-%d')
			pipedata.city_src = _json_to_controls["from_cities"][0]
			pipedata.city_dst = _json_to_controls["to_cities"][0]
			pipedata.forecasting_days = _json_to_controls["forecast_period"]
			pipedata.df_booking = df_booking
			pipedata.df_forecasting = df_forecast_only
			pipedata.elasticity = float(_json_to_controls["elasticity"])
			pipedata.base_capacity_absolute = _json_to_controls["base_capacity_absolute"]
			pipedata.base_capacity_distribution = _json_to_controls["base_capacity_distribution"]
					
			lower_factors = _json_to_controls["base_price_low"]
			upper_factors = _json_to_controls["base_price_high"]
			base_start = int(_json_to_controls["base_price"])
			pipedata.base_price_start = base_start
			
			for base_low in lower_factors:
				pipedata.base_price_low.append(float(base_low) * base_start)
			
			for base_high in upper_factors:
				pipedata.base_price_high.append(float(base_high) * base_start)
			
			for capacity in _json_to_controls["booking_capacities"]:
				cap = int(''.join(filter(str.isdigit, capacity)))
				pipedata.capacities.append(int(cap))
				pipedata.capacities_col[cap] = capacity
			pipedata.df_dynamicprice = None
			
			pipedata.weekend_surge_factor = float(_json_to_controls["weekend_surge"])
			pipedata.seasonal_surge_factor = float(_json_to_controls["seasonal_surge"])	
			pipedata.forecast_and_validate = forecast_and_validate
			
			forecastpipeline = ForecastingDyPricePipeline(pipedata)
			weekendpipeline = WeekendDyPricePipeline(pipedata)
			seasonalpipeline = SeasonalDyPricePipeline(pipedata)
			
			executer = DyPricePipelineExecuter()
			executer.add(forecastpipeline)
			executer.add(weekendpipeline)
			executer.add(seasonalpipeline)
			executer.execute()
			
			df_dynamicprice_table = construct_forecast_datatable(pipedata)
			
			# pack everything to json
			results = json.dumps(
				{
					"label": label, 
					"trends_data": df_trend_result,					
					"dynamic_price_table": df_dynamicprice_table.to_json(orient='records'),
				}
			)
			
			output_data.append(results)
			
			if auto_publish:
				output_data_map[label] =  df_dynamicprice_table
				
		except ValueError as err:
			print ('#####################################################')
			print('ERROR In COMPUTING DYNAMIC PRICE: ', label, err)
			print ('#####################################################')
			pass
			#raise
			
	if auto_publish:
		publish_dynamic_prices(output_data_map)
		
	return output_data

def publish_dynamic_prices(output_data_map):

	try:
		SqlLiteDynPriceAdapter.getInstance().connect()
		for label, dataframe in output_data_map.items():
			name = label
			name = name.replace('bkg_from_city:','').replace('bkg_to_city:', '')
			name = name.replace('bkg_from_zone:','').replace('bkg_to_zone:', '')
			name = 'dynprice_' +  name
			
			filename = name + '.csv'
			filename = './chalicelib/resources/dynamic_prices_server/'+filename
			dataframe.to_csv(filename)
			db_dataframe = dataframe.copy()
			print_dataframe(db_dataframe, '##### Publish Dynamic Prices #####', False)
			#db_dataframe.index = db_dataframe['Date'].dt.date
			SqlLiteDynPriceAdapter.getInstance().continous_update(name, dataframe)
			print ("Published dynamic price table for ", name)		
		SqlLiteDynPriceAdapter.getInstance().disconnect()
	except ValueError as err:
		print('ERROR publish_dynamic_prices: ', err)
		SqlLiteDynPriceAdapter.getInstance().disconnect()
	
	
def forecast_demand_unit_test():

	'''
		{
			"model_type": "Linear", 
			"forecast_and_validate": false, 
			"forecast_period": 30, 
			"query_type": "Custom", 
			"from_cities": ["Delhi"], 
			"to_cities": ["Chandigarh", "Delhi"], 
			"primary_input_field": "bkg_user_id", 
			"start_date": "2015-07-01 00:00:00", 
			"end_date": "2018-04-15 00:00:00", 
			"plot_components": false, 
			"base_price": "3000", 
			"base_price_low": "0.8", 
			"base_price_high": "2.0", 
			"elasticity": "-2.0", 
			"seasonal_surge": "0.0", 
			"weekend_surge": "0.0", 
			"rest_call": true
		}
	'''
	
	pipedata = DyPricePipelineData()
	pipedata.date_start = '2018-09-30'
	pipedata.date_end = '2018-10-30'
	pipedata.city_src = 'Delhi'
	pipedata.city_dst = 'Agra'
	
	pipedata.forecasting_days = (
		pd.to_datetime(pipedata.date_end) - pd.to_datetime(pipedata.date_start)).days
	pipedata.df_forecasting = None
	pipedata.elasticity = -2.0
	pipedata.base_price_start = 4000
	pipedata.base_price_low = pipedata.base_price_start * 0.8
	pipedata.base_price_high = pipedata.base_price_start * 1.5
	pipedata.capacities = [20.0, 40.0, 60.0, 80.0]
	pipedata.df_dynamicprice = None
	
	dynamicPricing = ForecastingDyPricePipeline(pipedata)
	
	executer = DyPricePipelineExecuter()
	executer.add(dynamicPricing)
	executer.execute()
	
	
	# plot plotly graph
	plot_booking_forecast(pipedata.df_forecasting)
	
	df_to_show = construct_forecast_datatable(pipedata)
	
	#print ('E\n', df_to_show)
	plot_capacitated_booking_forecast(df_to_show)

#forecast_demand_unit_test()
