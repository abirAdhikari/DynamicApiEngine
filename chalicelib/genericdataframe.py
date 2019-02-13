# AAdhikari: 9/1/2018

import sys
import os
import pandas as pd
import logging
import json
from pandas.io.json import json_normalize
import re
import numpy as np

sys.path.append('./chalicelib')
sys.path.append('.')
from databaseadapter import *
from debuglogger import *

use_absolute_path = True

class BaseDataframe:
	def __init__(self):
		self.columns = None
		self.refresh_interval = 30 #in minutes
		self.last_refresh_time = None
		self._dataframe = None
		#self.dbAdapter = SqlLiteAdapter.getInstance()
		self.dbAdapter = MySqlAdapter.getInstance()
		self.gozo_cities_df	= self.initialize_gozo_cities_dataframe()
		self.gozo_zones_df	= self.initialize_gozo_zones_dataframe()
		self.mmt_cities_df	= self.initialize_mmt_cities_dataframe()
		
		self.remove_time_from_datetime_columns = True
		
		# for MySqlAdapter don't do any customization on the datatime fields
		self.infer_datetime_format = False


	def set_columns(self, columns):
		self.columns = columns
		
	def get_columns(self):
		return self.columns
	
	def pull_all_columns(self):
		return False
	
	def need_to_refresh(self):
		if self._dataframe is None:
			return True
		if self.dbAdapter.need_to_refresh() is True:
			return True
		return False
		
	def get_dataframe(self):
		if self.need_to_refresh() is True:
			if self.pull_all_columns() is False:
				self.dbAdapter.set_columns(self.columns)
			df = self.dbAdapter.get_latest_dataframe()
			df = self.filter_columns(df)
			df = self.normalize_dataframe(df)
			self._dataframe = df			
		return self._dataframe

	def filter_columns(self, df):
		logutil(__class__, self.columns)
		return df.filter(self.columns, axis=1)
		
	def normalize_dataframe(self, df):
		raise Exception('Not Implemented')
	
	def initialize_gozo_cities_dataframe(self):
		if use_absolute_path:
			current_dir = '/home/abir/Downloads/RestApiEngine'
		else:
			current_dir = os.getcwd()
		if 'chalicelib' not in current_dir:
			current_dir += '/chalicelib'
		gozo_cities_csv = current_dir + '/resources/catalogs/cities.csv'	
		#print (gozo_cities_csv)
		gozo_df = pd.read_csv(gozo_cities_csv)
		gozo_df = gozo_df.drop_duplicates()
		gozo_df = gozo_df.set_index('cty_id')
		print_dataframe(gozo_df, "Gozo - Parsed the CSV", False)
		return gozo_df
	
	def initialize_gozo_zones_dataframe(self):
		#https://www.gozocabs.com/lookup/citylist
		if use_absolute_path:
			current_dir = '/home/abir/Downloads/RestApiEngine'
		else:
			current_dir = os.getcwd()
		if 'chalicelib' not in current_dir:
			current_dir += '/chalicelib'
		gozo_cities_csv = current_dir + '/resources/catalogs/zones_to_cities.csv'	
		#print (gozo_cities_csv)
		gozo_df = pd.read_csv(gozo_cities_csv)
		gozo_df = gozo_df.drop_duplicates()
		gozo_df = gozo_df.set_index('zon_id')
		print_dataframe(gozo_df, "Gozo - Parsed the CSV", False)
		return gozo_df
		
	def initialize_mmt_cities_dataframe(self):
		if use_absolute_path:
			current_dir = '/home/abir/Downloads/RestApiEngine'
		else:
			current_dir = os.getcwd()

		if 'chalicelib' not in current_dir:
			current_dir += '/chalicelib'
		mmt_input_json = current_dir + '/resources/catalogs/makemytrip_catalog.json'
		with open(mmt_input_json) as data_file: 
			content = data_file.read()
		content = '[' + content + ']'
		content = json.loads(content)			
		mmt_df = pd.DataFrame(json_normalize(content))
		mmt_df = mmt_df.set_index('city_name')
		print_dataframe(mmt_df, "MakeMyTrip - Parsed the JSON", False)
		return mmt_df
		
	def remove_nan_and_return_filtered(self, df):
		df.dropna(axis=0,inplace=True)
		all_inf_or_nan = df.isin([np.inf, -np.inf, np.nan])
		df = df[~all_inf_or_nan]
		return df
		

class PricingDataframe(BaseDataframe):
	def __init__(self):
		super().__init__()
		self.columns = [
			'bkg_id',
			'bkg_create_date',
			'bkg_pickup_date',
			'bkg_from_city_id',
			'bkg_to_city_id',
			#'bkg_total_amount',
			#'bkg_gozo_amount',
			'bkg_vehicle_type_id',				
			]
		self.set_primary_date('bkg_pickup_date')
	
	def set_primary_date(self, datecol):
		self.primarydate = datecol
		
	def normalize_dataframe(self, df):		
		logutil(__class__, 'Before calling convert')		
		print (df.dtypes)
		print_dataframe(df, 'DataFrame', False)
		df.fillna(0, inplace=True)
		# pd.to_datetime() yelids in m/d/YYY (4/16/2018) format 
		# 5X to 10X faster
		if self.infer_datetime_format:
			df['bkg_create_date'] = pd.to_datetime(
				df['bkg_create_date'], infer_datetime_format=True)			
			df['bkg_pickup_date']	= pd.to_datetime(
				df['bkg_pickup_date'], infer_datetime_format=True)
		else:
			df['bkg_create_date'] = pd.to_datetime(df['bkg_create_date'])			
			df['bkg_pickup_date']	= pd.to_datetime(df['bkg_pickup_date'])
		
		'''		
		df['bkg_total_amount']	= pd.to_numeric(
			df['bkg_total_amount'], errors='coerce')
		df['bkg_gozo_amount']	= pd.to_numeric(
			df['bkg_gozo_amount'], errors='coerce')
		'''
		# df[i].dt.date date in YYYY-MM-DD (2012-09-14)
		df['Date']	= df[self.primarydate].dt.date	
		
		if self.remove_time_from_datetime_columns:
			df['bkg_pickup_date'] = df['bkg_pickup_date'].dt.date
			df['bkg_create_date'] = df['bkg_create_date'].dt.date
		
		if self.infer_datetime_format:		
			df['Date'] = pd.to_datetime(df['Date'], infer_datetime_format=True)
		else:
			df['Date'] = pd.to_datetime(df['Date'])
		df['bkg_from_city']		= df['bkg_from_city_id']
		df['bkg_to_city']		= df['bkg_to_city_id']		
		
		
		
		print_dataframe(df, 'Pricing dataframe', False)
		print_dataframe(self.gozo_cities_df, 'Gozo cities dataframe', False)
		
		cities_dict_df = self.gozo_cities_df[['cty_name']]
		print_dataframe(cities_dict_df, 'cities_dict_df', False)
		cities_dict = cities_dict_df.to_dict('index')
		
		unknowns_list = []
		def find_city_name(x):
			if x in cities_dict:
				city_info = cities_dict[x]
				return city_info['cty_name']
			else:
				nonlocal unknowns_list
				unknowns_list.append(x)
				return x		
		
		df['bkg_from_city'] = df['bkg_from_city'].apply(lambda x: find_city_name(x))
		print ('bkg_from_city unknowns_list: ', unknowns_list)
		unknowns_list = []
		df['bkg_to_city'] = df['bkg_to_city'].apply(lambda x: find_city_name(x))
		print ('bkg_to_city unknowns_list: ', unknowns_list)
		print_dataframe(df, 'Pricing dataframe', False)				
		
		return df
		
class HeatmapDataframe(BaseDataframe):
	def __init__(self):
		super().__init__()
		self.heatmap = []		
		self.columns = [
			'bkg_user_id',
			'bkg_create_date',
			'bkg_pickup_date',
			'bkg_from_city_id',
			'bkg_to_city_id',
			'bkg_total_amount',
			'bkg_gozo_amount',
			'bkg_vehicle_type_id',		
			]
		self.set_primary_date('bkg_pickup_date')
	
	def set_primary_date(self, datecol):
		self.primarydate = datecol
	
	def add_heatmap_column(self, heatmapcol):
		if heatmapcol not in self.heatmap:
			self.heatmap.append(heatmapcol)
		if heatmapcol not in self.columns:
			self.columns.append(heatmapcol)
			
	
	def normalize_dataframe(self, df):		
		logutil(__class__, 'Before calling convert')		
		df.fillna(0, inplace=True)
		#df.drop(['bkg_id'], axis=1, inplace=True)
		# pd.to_datetime() yelids in m/d/YYY (4/16/2018) format 
		# 5X to 10X faster
		df['bkg_create_date'] = pd.to_datetime(
			df['bkg_create_date'], infer_datetime_format=True)
		df['bkg_pickup_date']	= pd.to_datetime(
			df['bkg_pickup_date'], infer_datetime_format=True)
		df['Date']			= df[self.primarydate].dt.date	
		df['bkg_from_city']		= df['bkg_from_city_id']
		df['bkg_to_city']		= df['bkg_to_city_id']		
		
		if self.remove_time_from_datetime_columns:
			df['bkg_pickup_date'] = df['bkg_pickup_date'].dt.date
			df['bkg_create_date'] = df['bkg_create_date'].dt.date
			
		df['Date'] = pd.to_datetime(
			df['Date'], infer_datetime_format=True)
		
		cities_dict_df = self.gozo_cities_df[['cty_name', 'cty_lat', 'cty_long']]
		print_dataframe(cities_dict_df, 'cities_dict_df', False)
		cities_dict = cities_dict_df.to_dict('index')
		
		def find_city_name(x):
			if x in cities_dict:
				city_info = cities_dict[x]
				return city_info['cty_name']
			else:
				return x

		def find_coordinates(x, type):
			if x in cities_dict:
				city_coordinates = cities_dict[x]
				return city_coordinates[type]
			else:
				return None	
				

		df['bkg_from_city'] = df['bkg_from_city_id'].apply(lambda x: find_city_name(x))
		df['bkg_to_city'] = df['bkg_to_city_id'].apply(lambda x: find_city_name(x))
		
		df['bkg_from_city_latitude'] = df['bkg_from_city_id'].apply(lambda x: find_coordinates(x, 'cty_lat'))
		df['bkg_from_city_longitude'] = df['bkg_from_city_id'].apply(lambda x: find_coordinates(x, 'cty_long'))
				
		df['bkg_to_city_latitude'] = df['bkg_to_city_id'].apply(lambda x: find_coordinates(x, 'cty_lat'))
		df['bkg_to_city_longitude'] = df['bkg_to_city_id'].apply(lambda x: find_coordinates(x, 'cty_long'))		
		
		print_dataframe(df, '*Heatmap Dataframe', False)
		return df		
	
		
class AnalyticsDataframe(PricingDataframe):
	def __init__(self):
		super().__init__()
	
def genericdataframe_unit_test():	

	test_pricing_df = False
	test_heatmap_df = True
	if test_pricing_df:
		pricingdf = PricingDataframe()
		logutil('UnitTest::PricingDataframe', 'Before getting dataframe')
		df = pricingdf.get_dataframe()
		logutil('UnitTest::PricingDataframe', df.dtypes)
		logutil('UnitTest::PricingDataframe', ('Length = %s' % len(df)))
		# df.dropna(inplace=True) returns None, so doing twice
		df_filtered = df.dropna(axis=0)
		df_droppped = df.drop(df_filtered.index)
		logutil('UnitTest::PricingDataframe', (
			'Length with NaN and INF removed = %s' % len(df_filtered)))
		#print_dataframe(df, 'UnitTest::PricingDataframe - Filtered Dataframe', False)
		#print_dataframe(df_dropped, 'UnitTest::PricingDataframe - Dropped Dataframe', False)
		logutil('UnitTest::PricingDataframe', 'After getting dataframe')
	
	if test_heatmap_df:
		heatmapdf = HeatmapDataframe()
		logutil('UnitTest::HeatmapDataframe', 'Before getting dataframe')
		df = heatmapdf.get_dataframe()
		logutil('UnitTest::HeatmapDataframe', df.dtypes)
		logutil('UnitTest::HeatmapDataframe', ('Length = %s' % len(df)))
		df_filtered = df.dropna(axis=0)
		df_droppped = df.drop(df_filtered.index)
		logutil('UnitTest::HeatmapDataframe', (
			'Length with NaN and INF removed = %s' % len(df_filtered)))
		#print_dataframe(df_filtered, 'UnitTest::HeatmapDataframe - Filtered Dataframe', False)
		print_dataframe(df_droppped, 'UnitTest::HeatmapDataframe - Dropped Dataframe', False)
		df_from_unknown = df_droppped[df_droppped['bkg_from_city_latitude'].isna()]
		print_dataframe(df_from_unknown['bkg_from_city'], 
			'UnitTest::HeatmapDataframe - Unknown bkg_from_city Dataframe', False)
		df_to_unknown = df_droppped[df_droppped['bkg_to_city_latitude'].isna()]
		print_dataframe(df_to_unknown['bkg_to_city'], 
			'UnitTest::HeatmapDataframe - Unknown bkg_to_city Dataframe', False)
		
		logutil('UnitTest::HeatmapDataframe', 'After getting dataframe')
		
#genericdataframe_unit_test()
