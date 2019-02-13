# database adapter layer
import os 
import sys
# AAdhikari: 8/28/2018
import sys 
import sqlite3
from datetime import datetime, timedelta, date
import logging
import pandas as pd

sys.path.append('./chalicelib')
from debuglogger import *

use_absolute_path = True

class DatabaseAdapter:

	def __init__(self):
		self._master_dataframe = None
		self.connection = None
		self.columns = None
		self.refresh_interval = timedelta(minutes=480) #in minutes
		self.last_refresh_time = None
	
	def need_to_refresh(self):
		if self._master_dataframe is None:
			return True
		else:
			current = datetime.now()
			time_diff = current - self.last_refresh_time
			time_diff_mins = timedelta(minutes=time_diff.total_seconds()/60)
			if time_diff_mins > self.refresh_interval:
				return True
		return False
	
	def get_latest_dataframe(self):
		if self.need_to_refresh() is True:
			print ('Refreshing from DB [Last Poll Time: %s]' % (self.last_refresh_time))
			self._master_dataframe = self.get_all_from_database()
			self.last_refresh_time = datetime.now()
			print ('Refreshed from DB [Last Poll Time: %s]' % (self.last_refresh_time))
		else:
			print ('Using last polled data [Last Poll Time: %s]' % (self.last_refresh_time))
			
		return self._master_dataframe
		
	def get_dataframe(self):
		if self._master_dataframe is None:
			self._master_dataframe = self.get_all_from_database()
			self.last_refresh_time = datetime.now()
		return self._master_dataframe
		
	
	def connect(self):
		raise Exception('Not Implemented')

	def disconnect(self):
		raise Exception('Not Implemented')

		
	def set_columns(self, cols):
		query_cols = []
		for x in cols:
			query_cols.append('\"' + x + '\", ')		
		size = len(query_cols)
		if size > 0:
			last_col = query_cols[size-1]
			last_col = last_col[:len(last_col)-2]
			query_cols[size-1] = last_col
		self.columns = ''.join(query_cols)
		
	def get_all_from_database(self):
		raise Exception('Not Implemented')
		
	def get_from_database(self, in_start_date, in_end_date):
		raise Exception('Not Implemented')

	
class SqlLiteAdapter(DatabaseAdapter):

	# Here will be the singleton instance stored.
	__instance = None

	def __init__(self):
		""" Virtually private constructor, can only be accessed by getInstance() """
		if SqlLiteAdapter.__instance != None:
			raise Exception("This class is a singleton!")
		else:
			SqlLiteAdapter.__instance = self
			super().__init__()
			self.columns = '*'
			
			if use_absolute_path:
				current_dir = '/home/abir/Downloads/RestApiEngine'
			else:
				current_dir = os.getcwd()
			
			if 'chalicelib' not in current_dir:
				current_dir += '/chalicelib'
			self.set_database(current_dir\
				+ '/resources/database/booking_database_v2.sqlite')	
	
	@staticmethod
	def getInstance():
		""" Static access method. """
		if SqlLiteAdapter.__instance == None:
			SqlLiteAdapter()
		return SqlLiteAdapter.__instance 

			
	def set_database(self, db_location):
		self.db_location = db_location
		
	def connect(self):
		self.connection = sqlite3.connect(self.db_location)

	def disconnect(self):
		self.connection.close()

	def get_all_from_database(self):
		return self.get_from_database(in_start_date=None, in_end_date=None)
		
	def get_from_database(self, in_start_date, in_end_date):
		self.connect()
		query = ''
		
		if in_start_date == None or in_end_date == None:
			query = 'select %s from booking_data;' % self.columns
		else:
			query = 'select %s from booking_data where \
				\'bkg_pickup_date\' >= \' %s \' and \'bkg_pickup_date\' <= \' %s \';' \
				% (self.columns, str(in_start_date), str(in_end_date))
		self._master_dataframe = pd.read_sql_query(query, self.connection)
		self.disconnect()
		logutil(__class__, query)
		#logutil(__class__, self._master_dataframe.tail())
		
		return self._master_dataframe

class SqlLiteDynPriceAdapter(DatabaseAdapter):

	__instance = None

	def __init__(self):
		""" Virtually private constructor, can only be accessed by getInstance() """
		if SqlLiteDynPriceAdapter.__instance != None:
			raise Exception("This class is a singleton!")
		else:
			SqlLiteDynPriceAdapter.__instance = self
			super().__init__()
			if use_absolute_path:
				current_dir = '/home/abir/Downloads/RestApiEngine'
			else:
				current_dir = os.getcwd()
			
			if 'chalicelib' not in current_dir:
				current_dir += '/chalicelib'
			self.set_database(current_dir\
				+ '/resources/database/dynamic_price.sqlite')	
	
	@staticmethod
	def getInstance():
		""" Static access method. """
		if SqlLiteDynPriceAdapter.__instance == None:
			SqlLiteDynPriceAdapter()
		return SqlLiteDynPriceAdapter.__instance 

			
	def set_database(self, db_location):
		self.db_location = db_location
		
	def connect(self):
		self.connection = sqlite3.connect(self.db_location)

	def disconnect(self):
		self.connection.close()

	def get_record(self, table_name, date):
		self.connect()
		print('SqlLiteDynPriceAdapter: get_record() start')
		query = 'select * from \'%s\' where Date == datetime(\'%s\');' % (table_name, date)
		print (query)
		result = pd.read_sql_query(query, self.connection)
		print('SqlLiteDynPriceAdapter: get_record() end [result=%s]' % result)
		self.disconnect()
		return result
	
	def update(self, table_name, df, if_exists='replace'):
		self.connect()
		print('SqlLiteDynPriceAdapter: Update start')
		df.to_sql(table_name, self.connection, if_exists=if_exists, index=False)
		print('SqlLiteDynPriceAdapter: Update end')
		self.disconnect()
	
	def continous_update(self, table_name, df, if_exists='replace'):
		
		print('SqlLiteDynPriceAdapter: Update start')
		df.to_sql(table_name, self.connection, if_exists=if_exists, index=False)
		print('SqlLiteDynPriceAdapter: Update end')
		

import mysql.connector

class MySqlAdapter(DatabaseAdapter):

	# Here will be the singleton instance stored.
	__instance = None

	def __init__(self):
		""" Virtually private constructor, can only be accessed by getInstance() """
		print ('MySqlAdapter: Initializing the database connection')
		if MySqlAdapter.__instance != None:
			raise Exception("This class is a singleton!")
		else:
			MySqlAdapter.__instance = self
			super().__init__()
			self.columns = '*'		
			self.host="gozo-repl.c3lsfej9lead.ap-south-1.rds.amazonaws.com"
			self.user="gzrdsuser"
			self.passwd="G0z0Rd$321"
			self.database="gozodb"
			print ('Host: ', self.host, ', User: ', self.user, ', Passwd: ', self.passwd, ', Database: ', self.database)
	@staticmethod
	def getInstance():
		""" Static access method. """
		if MySqlAdapter.__instance == None:
			MySqlAdapter()
		return MySqlAdapter.__instance 

			
	def set_database(self, db_location):
		self.db_location = db_location
		
	def connect(self):
		self.connection = mysql.connector.connect(
			  host=self.host,
			  user=self.user,
			  passwd=self.passwd,
			  database=self.database
			)		
		
	def disconnect(self):
		self.connection.close()

	def get_all_from_database(self):
		return self.get_from_database(in_start_date=None, in_end_date=None)
		
	def get_from_database(self, in_start_date, in_end_date):
		print ('MySqlAdapter Start: ', datetime.now())
		self.connect()
		query = ''
		if in_start_date == None or in_end_date == None:
			query = 'select %s from booking;' % "*"
		else:
			query = 'select %s from booking where \
				\'bkg_pickup_date\' >= \' %s \' and \'bkg_pickup_date\' <= \' %s \';' \
				% ("*", str(in_start_date), str(in_end_date))
		print (query)
		print (self.columns)
		df = pd.read_sql_query(query, self.connection)
		print (df.columns)
		self._master_dataframe = df[["bkg_id", "bkg_create_date", "bkg_pickup_date", "bkg_from_city_id", "bkg_to_city_id", "bkg_vehicle_type_id"]]
		print_dataframe(self._master_dataframe, 'master_dataframe', False)
		self.disconnect()
		print ('MySqlAdapter End: ', datetime.now())
		return self._master_dataframe		
	

def databaseadapter_unit_test():
	sqliteAdapter = SqlLiteAdapter()
	logutil(__class__, 'Before getting dataframe')
	columns = [
			'bkg_id',
			'bkg_create_date',
			'bkg_pickup_date',
			'bkg_from_city_id',
			'bkg_to_city_id',	
			]
	sqliteAdapter.set_columns(columns)
	df = sqliteAdapter.get_all_from_database()
	logutil(__class__, ('Length = %s' % len(df)))
	logutil(__class__, 'After getting dataframe')

#databaseadapter_unit_test()
