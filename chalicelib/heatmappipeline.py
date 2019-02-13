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
from shutil import copyfile
from collections import defaultdict
import json

sys.path.append('./chalicelib')
sys.path.append('.')
from prophetmodeller import *
from genericdataframe import *
from debuglogger import *

class HeatmapPipelineData:
	def __init__(self):
		''' Common pipeline data '''
		self.date_start = None
		self.date_end = None
		self.df_input = None
		self.df_heatmap = None
		self.primary_field = None
		self.oneway_return_discount_factor = 20
		
class HeatmapPipelineExecuter:
	def __init__(self):
		self.pipeline_map = {}
		
	def add(self, pipeline):
		name = pipeline.pipeline_name
		if name not in self.pipeline_map:
			self.pipeline_map[name] = pipeline
	
	def execute(self):
		for pipeline in self.pipeline_map.values():
			pipeline.execute_pipeline()	

class GenericHeatmapPipeline:
	def __init__(self, name, pipeline_data):
		self.pipeline_name = name
		self.name = name
		self.pipeline_data = pipeline_data
	
	def execute_pipeline(self):
		print('Executing pipeline: %s'% self.pipeline_name)
		
class HeatmapPipeline(GenericHeatmapPipeline):
	def __init__(self, pipeline_data):
		super().__init__('HeatmapPipeline', pipeline_data)
				
	def execute_pipeline(self):
		super().execute_pipeline()
		
		df = self.pipeline_data.df_input
		print(df.dtypes)
		df = df[df['Date'] > self.pipeline_data.date_start]
		df = df[df['Date'] < self.pipeline_data.date_end]
		
		if self.pipeline_data.primary_field == 'Oneway':
			df = df.drop(df[df['bkg_to_city'] == df['bkg_from_city']].index)
		elif self.pipeline_data.primary_field == 'Roundtrips':
			df = df.drop(df[df['bkg_to_city'] != df['bkg_from_city']].index)
		elif self.pipeline_data.primary_field == 'Oneway Returns':
			df = df.drop(df[df['bkg_to_city'] == df['bkg_from_city']].index)
			# switch To and From columns
			swapped_col_list = self.swap_columns(df.columns)
			df.columns = swapped_col_list
			df['Date'] = df['Date'] + timedelta(days=1)
			
		self.pipeline_data.df_heatmap = df
		
	def swap_columns(self, col_list):
		swapped_col_list = []
		for name in col_list:
			if 'bkg_to_city' == name:
				swapped_col_list.append('bkg_from_city')
			elif 'bkg_from_city' == name:
				swapped_col_list.append('bkg_to_city')
			elif 'bkg_to_city_id' == name:
				swapped_col_list.append('bkg_from_city_id')
			elif 'bkg_from_city_id' == name:
				swapped_col_list.append('bkg_to_city_id')
			elif 'bkg_to_city_latitude' == name:
				swapped_col_list.append('bkg_from_city_latitude')
			elif 'bkg_from_city_latitude' == name:
				swapped_col_list.append('bkg_to_city_latitude')
			elif 'bkg_to_city_longitude' == name:
				swapped_col_list.append('bkg_from_city_longitude')
			elif 'bkg_from_city_longitude' == name:
				swapped_col_list.append('bkg_to_city_longitude')
			else:
				swapped_col_list.append(name)
		print ("Swap Column Input: ", col_list)
		print ("Swap Column Output: ", swapped_col_list)
		return swapped_col_list

class OnewayReturnDiscountPipeline(GenericHeatmapPipeline):
	def __init__(self, pipeline_data):
		super().__init__('OnewayReturnDiscountPipeline', pipeline_data)
				
	def execute_pipeline(self):
		super().execute_pipeline()
		
		df = self.pipeline_data.df_heatmap		
		print_dataframe(df, 'OnewayReturnDiscountPipeline', False)
		discount = self.pipeline_data.oneway_return_discount_factor
		df['bkg_total_amount'] = df['bkg_total_amount'].astype(np.int64)
		df['Total Mean'] = df.groupby(['bkg_from_city', 'bkg_to_city', 'bkg_vehicle_type_id'])['bkg_total_amount'].transform('mean')
		df['Discounted Price'] = df['Total Mean'] * ((100 - discount)/100)
		df['Avail. Count'] = df.groupby(['bkg_from_city', 'bkg_to_city', 'bkg_vehicle_type_id'])['bkg_user_id'].transform('count')
		self.pipeline_data.df_heatmap = df		
		
	
		
def get_heatmap_json_wrapper(_json_data):
	_json_to_controls=json.loads(_json_data)
	print (_json_to_controls)	
	
	start_date = datetime.strptime(
			(_json_to_controls["start_date"]).split()[0], '%Y-%m-%d')
	end_date = datetime.strptime(
			(_json_to_controls["end_date"]).split()[0], '%Y-%m-%d')
	primary_input_field = _json_to_controls["primary_input_field"]
	
	pipedata = HeatmapPipelineData()
	pipedata.date_start = start_date.strftime('%Y-%m-%d')
	pipedata.date_end = end_date.strftime('%Y-%m-%d')
	pipedata.primary_field = primary_input_field	
	pipedata.df_input = HeatmapDataframe().get_dataframe()
		
	heatmappipeline = HeatmapPipeline(pipedata)
			
	executer = HeatmapPipelineExecuter()
	executer.add(heatmappipeline)
	if primary_input_field == 'Oneway Returns':
		returndiscountpipeline = OnewayReturnDiscountPipeline(pipedata)
		executer.add(returndiscountpipeline)
	executer.execute()
	
	
	
	print_dataframe(pipedata.df_heatmap, 'Heatmap Dataframe', False)
	pipedata.df_heatmap.sort_values(by=['bkg_from_city', 'bkg_to_city', 'bkg_vehicle_type_id'])
	df_to_json = pipedata.df_heatmap.to_json(orient='records')
	return json.dumps({"label": 'heatmap', "df": df_to_json})
	

def should_add_to_heatmap(field):
	if field == 'Oneway':
		return False
	if field == 'Roundtrips':
		return False
	if field == 'Oneway Returns':
		return False
	return True

#https://en.wikipedia.org/wiki/Haversine_formula
from math import cos, asin, sqrt
def distance(lat1, lon1, lat2, lon2):
    p = 0.017453292519943295     #Pi/180
    a = 0.5 - cos((lat2 - lat1) * p)/2 + cos(lat1 * p) * cos(lat2 * p) * (1 - cos((lon2 - lon1) * p)) / 2
    return 12742 * asin(sqrt(a)) #2*R*asin...

#unit test
def run_unit_test():
	pass
		
#run_unit_test()
	
