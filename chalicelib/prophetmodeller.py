# AAdhikari: 8/30/2018

import os, sys

import pandas as pd
import numpy as np
import fbprophet as fbpro
import sklearn.metrics as skm
import math
import datetime as dt
import plotly.graph_objs as go

sys.path.append('./chalicelib')
sys.path.append('.')

class ProphetModeller(object):
	"""
		This class is a wrapper over TS prediction algorithms provided by Prophet
		Prophet is Facebook-developed open-source lib for producing high quality forecasts for
		time series data that has multiple seasonality with linear or non-linear growth
		(https://facebook.github.io/prophet/)

		Set of product documentation on Prophet is available at
		https://facebook.github.io/prophet/docs/quick_start.html#python-api
	"""

	# TODO list:
	# 1. Add capability to set change point dates manually: m = Prophet(changepoints=['2014-01-01'])
	# 2. Add ability to configure outlier dates/date intervals per the method in
	#	 https://facebook.github.io/prophet/docs/outliers.html
	# 3. Add ability to manipulate with uncertainty intervals and do highly computationally-intencive Markov Chain
	#	 sampling per https://facebook.github.io/prophet/docs/uncertainty_intervals.html

	def __init__(self, df_data, future_periods, do_log_transform, validate_existing):
		""" Constructor for the class
			:param df_data a pandas data frame with TS data - it must have exactly two columns with
				   constant names - ['ds', 'y']
			:param future_periods - the int value indicating the number of TS data points to forecast ahead
			:param do_log_transform - 0/1 switch indicating if df_data has to be log transformed before prediction
		"""
		self._df_data = df_data
		self._future_periods = future_periods
		self._do_log_transform = do_log_transform
		self._validate_existing = validate_existing

		# fake empty placeholder forecast DF, to be properly populated at predict time
		self._forecast = pd.DataFrame(columns = ['ds', 'yhat', 'yhat_lower', 'yhat_upper'])

		# fake empty placeholder holidays DF, to be set separately via respective property
		# before predict() call, if needed
		# see more details on holidays at https://facebook.github.io/prophet/docs/seasonality,_holiday_effects,_and_regressors.html
		self._holidays = pd.DataFrame(columns = ['holiday', 'ds', 'lower_window', 'upper_window'])
		self._holidays_set = 0	# if changed to 1 in the holidays setter, it will trigger self.model setup with holidays

		self._capacity_used = 0 # if changed to 1 in set_capacity, it will result in "logistic" growth set in the self.model

		# below are default values for other class members that can be later altered via read-write properties

		self._turn_negative_forecasts_to_0 = 1 # this is 0/1 flag
		
		#https://facebook.github.io/prophet/docs/trend_changepoints.html
		self._changepoint_prior_scale = 1   # default value set by Facebook team is 0.05
		self._changepoint_range=0.95 # default value set by Facebook team is 0.8
		
		'''
		self._n_changepoints = 25 # default value set by Facebook team is 25
		changepoints = []
		#dow_map = { 6:'Sun', 0:'Mon', 1:'Tue', 2:'Wed', 3:'Thu', 4:'Fri', 5:'Sat'}
		def saturdays_changepoints(x):
			if x.weekday() == 5:
				changepoints.append(str(x))
		tmp_df = pd.DataFrame(columns = ['tmp'])
		tmp_df['tmp'] = df_data['ds'].dt.date.apply(lambda x: saturdays_changepoints(x))
		self._changepoints=changepoints
		#print (self._changepoints)
		'''

		# default value set by Facebook taam (see https://facebookincubator.github.io/prophet/docs/holiday_effects.html
		self._holidays_prior_scale = 10.0

		self._seasonality_prior_scale = 10.0 #default value set by Facebook team

		self._yearly_seasonality = True # set False via property if the TS does not have yearly seasonality
		self._weekly_seasonality = True # set False via property if the TS does not have weekly seasonality
		self._daily_seasonality = 'auto' # LOCAL SETTING

		# freq: Any for pd.date_range, such as 'D' or 'M'
		# see http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases for more info
		self._frequency = 'D' # default set by Facebook team


	def set_capacity(self, list_of_capacity_modifiers):
		""" this will set capacity modifier, assuming the y values in the df_data could be bigger if all
			potential capacity (like market size, physical limit of generated power, etc. ) could be exhausted
			We must specify the carrying capacity in a column cap.
			It would usually be set using data or expertise about the market size or other limits related to y values.

			Note: if you like to set capacity modifiers, it should be done before you call predict()
		"""
		self._df_data['cap'] = list_of_capacity_modifiers
		self._capacity_used = 1

	def predict(self):
		""" This is the major prediction utility """
		debug = 0

		if np.mean(self._df_data['y'].values) == 0:
			print("[ProphetModeller.predict] Waring: Input for Prophet is ts with zero values - no forecast will be creaed")

		else:
			df = self._df_data.copy() # make a local copy of input data ...
			if self._validate_existing:
				df = df[:-self._future_periods + 2]

			if self._do_log_transform:
				df['y'] = np.log(df['y'])
			if self._capacity_used:
				if self._holidays_set:
					self.model = fbpro.Prophet(growth='logistic',
									  changepoint_prior_scale = self._changepoint_prior_scale,
									  changepoint_range = self._changepoint_range,
									  #n_changepoints = self._n_changepoints,
									  #changepoints=self._changepoints,
									  holidays = self._holidays,
									  holidays_prior_scale = self._holidays_prior_scale,
									  yearly_seasonality = self._yearly_seasonality,
									  weekly_seasonality = self._weekly_seasonality,
									  daily_seasonality = self._daily_seasonality,
									  seasonality_prior_scale = self._seasonality_prior_scale)
				else:
					self.model = fbpro.Prophet(growth='logistic',
									  changepoint_prior_scale = self._changepoint_prior_scale,
									  changepoint_range = self._changepoint_range,
									  #n_changepoints = self._n_changepoints,
									  #changepoints=self._changepoints,
									  yearly_seasonality = self._yearly_seasonality,
									  weekly_seasonality = self._weekly_seasonality,
									  daily_seasonality = self._daily_seasonality,
									  seasonality_prior_scale = self._seasonality_prior_scale)
			else:
				if self._holidays_set:
					self.model = fbpro.Prophet(changepoint_prior_scale = self._changepoint_prior_scale,
									  changepoint_range = self._changepoint_range,
									  #n_changepoints = self._n_changepoints,
									  #changepoints=self._changepoints,
									  holidays = self._holidays,
									  holidays_prior_scale=self._holidays_prior_scale,
									  yearly_seasonality = self._yearly_seasonality,
									  weekly_seasonality = self._weekly_seasonality,
									  daily_seasonality = self._daily_seasonality,
									  seasonality_prior_scale = self._seasonality_prior_scale)
				else:
					self.model = fbpro.Prophet(changepoint_prior_scale=self._changepoint_prior_scale,
									  changepoint_range = self._changepoint_range,
									  #n_changepoints = self._n_changepoints,
									  #changepoints=self._changepoints,
									  yearly_seasonality = self._yearly_seasonality,
									  weekly_seasonality = self._weekly_seasonality,
									  daily_seasonality = self._daily_seasonality,
									  seasonality_prior_scale = self._seasonality_prior_scale)

			self.model.fit(df)
			future = self.model.make_future_dataframe(periods=self._future_periods, freq=self._frequency)

			# this will return data for columns ['ds', 'yhat', 'yhat_lower', 'yhat_upper'] always
			# ds - datetime stamp of the point in observations
			# yhat - prediction
			# 'yhat_lower' and 'yhat_upper' - uncertainty intervals
			#
			# optionally, additional cols could be added with the impact of each holiday season, if holidays configured
			# in this case, value of 'holiday' col in each holiday df row will be the name of the impact col for such
			# a holiday
			self._forecast = self.model.predict(future)

			if self._turn_negative_forecasts_to_0:
				self._forecast.loc[self._forecast.yhat < 0, 'yhat'] = 0
				self._forecast.loc[self._forecast.yhat_lower < 0, 'yhat_lower'] = 0
				self._forecast.loc[self._forecast.yhat_upper < 0, 'yhat_upper'] = 0

			if self._do_log_transform:
				self._forecast['yhat'] = np.exp(self._forecast['yhat'])
				self._forecast['yhat_lower'] = np.exp(self._forecast['yhat_lower'])
				self._forecast['yhat_upper'] = np.exp(self._forecast['yhat_upper'])

			if debug:
				print("Forecasted values:")
				print(self._forecast.tail())
			
			#self.main_figure = self.model.plot(self.forecast)
			#self.components_figure = self.model.plot_components(self.forecast)

	def get_forecast_only(self):
		""" this will return the subset of self._forecast without historic data """
		df_forecast_only = self._forecast.iloc[
				(self._forecast.shape[0] - self._future_periods):self._forecast.shape[0], ]
		return df_forecast_only

	# validation methods / metrics

	def rmse(self, targets):
		""" This method validates RMSE of the predicted values vs. the targets from the validation set

		:param targets - a list of target (true) values from the validation set

		:return calculated RMSE or -1 in case the length of targets list does not equal to self._future_periods
		"""

		rmse = -1

		if len(targets) != self._future_periods:
			print("[ProphetModeller.rmse] invalid target length: ", len(targets),
				  ", expected length: ", self._future_periods)
		else:
			y_pred = self.get_forecast_only()['yhat']

			rmse = math.sqrt(skm.mean_squared_error(targets, y_pred))

		return rmse

	def mean_absolute_percentage_error(self, targets):
		""" This method validates MAPE of the predicted values vs. the targets from the validation set

			:param targets - a list of target (true) values from the validation set

			:return calculated MAPE or -1 in case the length of targets list does not equal to self._future_periods
		"""
		mape = -1

		if len(targets) != self._future_periods:
			print("[ProphetModeller.mean_absolute_percentage_error] invalid target length: ", len(targets),
				  ", expected length: ", self._future_periods)

		else:
			y_pred = self.get_forecast_only()['yhat']
			y_true = targets

			## Note: does not handle mix 1d representation
			# if _is_1d(y_true):
			#	 y_true, y_pred = _check_1d_array(y_true, y_pred)
			mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
		return mape

	def smape(self, targets):
		""" This method validates SMAPE of the predicted values vs. the targets from the validation set

			:param targets - a list of target (true) values from the validation set

			:return calculated SMAPE or -1 in case the length of targets list does not equal to self._future_periods
		"""
		smape = -1
		if len(targets) != self._future_periods:
			print("[ProphetModeller.smape] invalid target length: ", len(targets),
				  ", expected length: ", self._future_periods)

		else:
			y_pred = self.get_forecast_only()['yhat']
			y_true = targets

			denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
			diff = np.abs(y_true - y_pred) / denominator
			diff[denominator == 0] = 0.0
			smape = np.mean(diff)

		return smape

	# properties

	@property
	def forecast(self):
		# this returns the entire forecast df, which contains the origianl TS plus self._future_periods forecasted
		# periods in future - if you need to get the forecasted new values, there is a separate method for that above
		return self._forecast

	@property
	def turn_negative_forecasts_to_0(self):
		return self._turn_negative_forecasts_to_0

	@turn_negative_forecasts_to_0.setter
	def turn_negative_forecasts_to_0(self, value):
		if value != 1:
			self._turn_negative_forecasts_to_0 = 0
		else:
			self._turn_negative_forecasts_to_0 = 1

	# Parameter modulating the flexibility of the automatic changepoint selection. Large values will allow many
	# changepoints, small values will allow few changepoints.
	# see more on the statistical meaning of this parameter at
	# https://facebookincubator.github.io/prophet/docs/trend_changepoints.html, Adjusting trend flexibility
	@property
	def changepoint_prior_scale(self):
		return self._changepoint_prior_scale

	@changepoint_prior_scale.setter
	def changepoint_prior_scale(self, value):
		self._changepoint_prior_scale = value # should be float value between 0 and 1

	@property
	def holidays(self):
		return self._holidays

	@holidays.setter
	def holidays(self, df_holidays):
		# df_holidays must be created in adherance to the convention explained at
		# https://facebookincubator.github.io/prophet/docs/holiday_effects.html, Modelling holidays
		self._holidays = df_holidays
		self._holidays_set = 1	# if changed to 1 in the holidays setter, it will trigger self.model setup with holidays

	# Parameter modulating the strength of the holiday components self.model.
	@property
	def holidays_prior_scale(self):
		return self._holidays_prior_scale

	@holidays_prior_scale.setter
	def holidays_prior_scale(self, value):
		# value is non-negative float
		# detailed statistical meaning of this parameter is explained in
		# https://facebookincubator.github.io/prophet/docs/holiday_effects.html,
		# 'Prior scale for holidays and seasonality' section
		self._holidays_prior_scale = value

	# seasonality flags
	@property
	def yearly_seasonality(self):
		return self._yearly_seasonality
	@yearly_seasonality.setter
	def yearly_seasonality(self, boolean_value):
		self._yearly_seasonality = boolean_value

	@property
	def weekly_seasonality(self):
		return self._weekly_seasonality

	@weekly_seasonality.setter
	def weekly_seasonality(self, boolean_value):
		self._weekly_seasonality = boolean_value

	# Parameter modulating the strength of the
	# seasonality self.model. Larger values allow the self.model to fit larger seasonal
	# fluctuations, smaller values dampen the seasonality.
	@property
	def seasonality_prior_scale(self):
		return self._seasonality_prior_scale

	@seasonality_prior_scale.setter
	def seasonality_prior_scale(self, value):
		self._seasonality_prior_scale = value

	# Any valid frequency for pd.date_range, such as 'D' or 'M'.
	# Full list of valid frequencies (or offset aliases) is available at
	# http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases
	@property
	def frequency(self):
		return self._frequency

	@frequency.setter
	def frequency(self, value):
		self._frequency = value
	
	

class ProphetUtils:
	def get_comparison_dataframe(training_df, forecast_df):
		"""
		Join the history with the forecast.
		The resulting dataset will contain 
		columns 'yhat', 'yhat_lower', 'yhat_upper' and 'y'.
		"""
		return forecast_df.set_index('ds')\
			[['yhat', 'yhat_lower', 'yhat_upper']].join(training_df.set_index('ds'))


	def get_comparison_graph(cmp_df, forecast_period, num_values):
		def create_go(name, column, num, **kwargs):
			points = cmp_df.tail(num)
			args = dict(
				name=name, 
				x=points['ds'], 
				y=points[column], 
				mode='lines')
			args.update(kwargs)
			return go.Scatter(**args)
		
		lower_bound = create_go(
			'Lower Bound', 
			'yhat_lower', 
			forecast_period,
			line=dict(width=0),
			marker=dict(color='rgb(68,68,68)'))
			
		upper_bound = create_go(
			'Upper Bound', 
			'yhat_upper', 
			forecast_period,
			line=dict(width=0),
			marker=dict(color='rgb(68,68,68)'),
			fillcolor='rgba(68, 68, 68, 0.3)', 
			fill='tonexty')
			
		forecast = create_go(
			'Forecast', 
			'yhat', 
			forecast_period,
			line=dict(color='rgb(31, 119, 180)'))
		
		actual = create_go(
			'Actual', 
			'y', 
			num_values,
			marker=dict(color="red"))
		
		cmp_graph = [lower_bound, upper_bound, forecast, actual]
		
		return cmp_graph
		
	
		
