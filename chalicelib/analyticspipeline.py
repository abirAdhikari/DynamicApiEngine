
import pandas as pd
import numpy as np
import scipy

import math as mt

import pickle as pkl
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
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
from IPython.display import Image

sys.path.append('./chalicelib')
sys.path.append('.')
from genericdataframe import *

class AnalyticsPipelineData:
	def __init__(self):
		''' Common pipeline data '''
		self.date_start = None
		self.date_end = None
		self.df_analytics = None
		
class AnalyticsPipelineExecuter:
	def __init__(self):
		self.pipeline_map = {}
		
	def add(self, pipeline):
		name = pipeline.pipeline_name
		if name not in self.pipeline_map:
			self.pipeline_map[name] = pipeline
	
	def execute(self):
		for pipeline in self.pipeline_map.values():
			pipeline.execute_pipeline()	

class GenericAnalyticsPipeline:
	def __init__(self, name, pipeline_data):
		self.pipeline_name = name
		self.name = name
		self.pipeline_data = pipeline_data
	
	def execute_pipeline(self):
		print('Executing pipeline: %s'% self.pipeline_name)
		
class AnalyticsPipeline(GenericAnalyticsPipeline):
	def __init__(self, pipeline_data):
		super().__init__('AnalyticsPipeline', pipeline_data)
				
	def execute_pipeline(self):
		super().execute_pipeline()
		
		df_analytics = AnalyticsDataframe()
		df = df_analytics.get_dataframe()
		print(df.dtypes)
		df = df[df['Date'] > self.pipeline_data.date_start]
		df = df[df['Date'] < self.pipeline_data.date_end]
		df.drop(['bkg_user_id'], axis=1, inplace=True)
		self.pipeline_data.df_analytics = df
		
		
		
def get_analytics_data_json_wrapper(_json_data):

	_json_to_controls=json.loads(_json_data)
	print (_json_to_controls)	
	
	start_date = datetime.strptime(	
			(_json_to_controls["start_date"]).split()[0], '%Y-%m-%d')
	end_date = datetime.strptime(
			(_json_to_controls["end_date"]).split()[0], '%Y-%m-%d')
	
	pipedata = AnalyticsPipelineData()
	pipedata.date_start = start_date.strftime('%Y-%m-%d')
	pipedata.date_end = end_date.strftime('%Y-%m-%d')
		
	analyticspipeline = AnalyticsPipeline(pipedata)
			
	executer = AnalyticsPipelineExecuter()
	executer.add(analyticspipeline)
	executer.execute()
	
	df_to_json = pipedata.df_analytics.to_json(orient='records')
	return json.dumps({"label": 'analytics', "df": df_to_json})
		
	
def alaytics_unit_test():
	
	pass
	
	
	

#forecast_demand_unit_test()
