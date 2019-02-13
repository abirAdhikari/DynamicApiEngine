#http://kourentzes.com/forecasting/2017/07/29/benchmarking-facebooks-prophet/
#http://ucanalytics.com/blogs/wp-content/uploads/2017/08/ARIMA-TimeSeries-Analysis-of-Tractor-Sales.html
# AAdhikari: 8/30/2018

import os, sys

import pandas as pd
import numpy as np
import math
import datetime as dt
import plotly.graph_objs as go

sys.path.append('./chalicelib')
sys.path.append('.')

class AutoArimaModeller(object):
	
	def __init__(self, df_data, future_periods, do_log_transform, validate_existing):

		self._df_data = df_data
		self._future_periods = future_periods
		self._do_log_transform = do_log_transform
		self._validate_existing = validate_existing

		# fake empty placeholder forecast DF, to be properly populated at predict time
		self._forecast = pd.DataFrame(columns = ['ds', 'yhat', 'yhat_lower', 'yhat_upper'])



	def predict(self):
		""" This is the major prediction utility """
		debug = 0

		if np.mean(self._df_data['y'].values) == 0:
			print("[AutoArimaModeller.predict] Waring: Input for Prophet is ts with zero values - no forecast will be creaed")

		else:
			df = self._df_data.copy() # make a local copy of input data ...
			if self._validate_existing:
				df = df[:-self._future_periods+1]

			if self._do_log_transform:
				df['y'] = np.log(df['y'])
			
			if self._do_log_transform:
				self._forecast['yhat'] = np.exp(self._forecast['yhat'])
				self._forecast['yhat_lower'] = np.exp(self._forecast['yhat_lower'])
				self._forecast['yhat_upper'] = np.exp(self._forecast['yhat_upper'])

			if debug:
				print("Forecasted values:")
				print(self._forecast.tail())
			
			
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
			print("[AutoArimaModeller.rmse] invalid target length: ", len(targets),
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
			print("[AutoArimaModeller.mean_absolute_percentage_error] invalid target length: ", len(targets),
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
			print("[AutoArimaModeller.smape] invalid target length: ", len(targets),
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
		return self._forecast

	
	
