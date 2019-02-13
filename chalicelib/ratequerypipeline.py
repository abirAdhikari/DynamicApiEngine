
import pandas as pd
import os, sys
from datetime import datetime

sys.path.append('./chalicelib')
from databaseadapter import SqlLiteDynPriceAdapter

class RateQueryPipelineData:
	def __init__(self):
		''' Common pipeline data '''
		self.city_src = None
		self.city_dst = None
		self.current_count = None
		self.date_start = None
		self.date_end = None
		self.rate = None

class RateQueryPipelineExecuter:
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
class GenericRateQueryPipeline:
	def __init__(self, name, pipeline_data):
		self.pipeline_name = name
		self.name = name
		self.pipeline_data = pipeline_data
	
	def execute_pipeline(self):
		print('Executing pipeline: %s'% self.pipeline_name)


		
class OnewayRateQueryPipeline(GenericRateQueryPipeline):
	def __init__(self, pipeline_data):
		super().__init__('OnewayRateQueryPipeline', pipeline_data)
				
	def execute_pipeline(self):
		super().execute_pipeline()
		table_name = self.pipeline_data.city_src + '-' + self.pipeline_data.city_dst
		start = self.pipeline_data.date_start + ' 00:00:00'
		try:
			df_result = SqlLiteDynPriceAdapter.getInstance().get_record(table_name, start)
		except:
			print('Dynamic price not available for %s', table_name)
			return
			
		current_count = self.pipeline_data.current_count
		if len(df_result) > 0:
			result = 0.0
			capacities = []
			for column in df_result.columns:
				if 'Capacity' in column:
					col_capacity = float("".join(filter(str.isdigit, column)))
					# divide by 10 as columns are stored as 20.0, 40.0, 
					# the above filter gives output 200, 400 
					capacities.append(col_capacity/10)
			capacities = sorted(capacities)
			max_capacity = capacities[len(capacities)-1]
			remaining_capacity = float(max_capacity - current_count)
			
			
			print('Capacities=%s, Max Capacity=%s, Remaing Capacity= %s, Current Count=%s' % 
				(capacities, max_capacity, remaining_capacity, current_count))
			
			for capacity in capacities:
				if remaining_capacity > capacity:
					continue
				else:
					capacity_col = str('Capacity left : %s' % capacity)
					result = df_result[[capacity_col]].values.tolist()
					# result is pandas series
					print (result)
					self.pipeline_data.rate = result[0][0]
					break
			
class OnewayPickupTimePipeline(GenericRateQueryPipeline):
	def __init__(self, pipeline_data):
		super().__init__('OnewayPickupTimePipeline', pipeline_data)
				
	def execute_pipeline(self):
		super().execute_pipeline()
		rate = self.pipeline_data.rate
		
		# HACK for POC for now, eventually we need to compare the 
		# query date+time vs. booking date+time
		#if self.pipeline_data.time_start != '09:00':
		#	rate = rate * 1.2 # 20% surge
		print('OnewayPickupTimePipeline Start Time:%s, Rate:%s' % \
			(self.pipeline_data.time_start, self.pipeline_data.rate))
			
		start_date_time = datetime.strptime(
			self.pipeline_data.date_start +	' ' + self.pipeline_data.time_start,
			'%Y-%m-%d %H:%M')
		# strip seconds and microseconds from current time
		now_date_time = datetime.now().replace(second=0, microsecond=0)
		delta = start_date_time - now_date_time
		days, hours, minutes = delta.days, delta.seconds // 3600, delta.seconds % 3600 / 60.0
		print ('OnewayPickupTimePipeline Booking Time: %s, Query Time: %s, Delta: %s (%s, %s, %s)'\
			% (start_date_time, now_date_time, delta, days, hours, minutes))
		if days <= 0:
			if hours <= 3:
				rate = rate * 1.5
			elif hours <= 6:
				rate = rate * 1.3
			elif hours <= 8:
				rate = rate * 1.2
		self.pipeline_data.rate = rate
		print('OnewayPickupTimePipeline Rate:%s' % (rate))
		

def api_get_oneway_rate(
	source, destination, 
	date_start, date_end,
	time_start, time_end,
	current_count):
	
	pipedata = RateQueryPipelineData()
	pipedata.city_src = source
	pipedata.city_dst = destination
	pipedata.date_start = date_start
	pipedata.date_end = date_end
	pipedata.time_start = time_start
	pipedata.time_end = time_end
	pipedata.current_count = float(current_count)
	
	
	onewayrate = OnewayRateQueryPipeline(pipedata)
	pickuptime = OnewayPickupTimePipeline(pipedata)
	executer = RateQueryPipelineExecuter()
	executer.add(onewayrate)
	executer.add(pickuptime)
	executer.execute()
		
	if pipedata.rate == None:
		return 'Not Available'
	else:
		return pipedata.rate
