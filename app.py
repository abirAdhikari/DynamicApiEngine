from chalice import Chalice
from chalicelib.forecastingpipeline import get_forecasting_trends_json_wrapper
from chalicelib.analyticspipeline import get_analytics_data_json_wrapper
from chalicelib.heatmappipeline import get_heatmap_json_wrapper
from chalicelib.dynamicpricepipeline import get_dynamic_price_json_wrapper
from chalicelib.ratequerypipeline import api_get_oneway_rate

app = Chalice(app_name='RestApiEngine')
app.debug = True


@app.route('/checkstatus')
def index():
    return {'server': 'successful'}	
'''
Web:
http://127.0.0.1:8000/forecast?query={"model_type": "Linear", "forecast_and_validate": false, "forecast_period": 30, "query_type": "All_India", "from_cities": "", "to_cities": "", "primary_input_field": "bkg_user_id", "start_date": "2015-07-01 00:00:00", "end_date": "2018-04-15 00:00:00", "plot_components": false, "seasonal_surge": "0.0", "weekend_surge": "0.0", "rest_call": true}
CLI:
curl -G -i http://127.0.0.1:8000/forecast?query=%7B%22model_type%22:%20%22Linear%22,%20%22forecast_and_validate%22:%20false,%20%22forecast_period%22:%2030,%20%22query_type%22:%20%22All_India%22,%20%22from_cities%22:%20%22%22,%20%22to_cities%22:%20%22%22,%20%22primary_input_field%22:%20%22bkg_user_id%22,%20%22start_date%22:%20%222015-07-01%2000:00:00%22,%20%22end_date%22:%20%222018-04-15%2000:00:00%22,%20%22plot_components%22:%20false,%20%22rest_call%22:%20true%7D
'''
@app.route('/forecast', content_types=['application/json'], methods=['GET'])
def chalice_api_get_forecast():
	request = app.current_request
	#print(request.__dict__)
	json_args = request.query_params['query']
	print (json_args)
	pipedata = get_forecasting_trends_json_wrapper(request.query_params['query'])
	return {'trends_data': pipedata.forecast_data}


	
'''
Web:
http://127.0.0.1:8000/analytics?query={"start_date": "2018-01-01", "end_date": "2018-04-15"}
CLI:
curl -G -i http://127.0.0.1:8000/analytics?query=%7B%22start_date%22:%20%222018-01-01%22,%20%22end_date%22:%20%222018-04-15%22%7D
'''
@app.route('/analytics', methods=['GET'])
def chalice_api_get_analytics():
	request = app.current_request
	#print(request.__dict__)
	json_args = request.query_params['query']
	print (json_args)
	analytics_data = get_analytics_data_json_wrapper(request.query_params['query'])
	return {'analytics_data': analytics_data}
	
	
'''	
Web:
http://127.0.0.1:8000/heatmap?query={"primary_input_field": "Oneway Returns", "view_type": "Show Both", "start_date": "2018-04-01", "end_date": "2018-04-15"}
CLI:
curl -G -i http://127.0.0.1:8000/heatmap?query=%7B%22primary_input_field%22:%20%22Oneway%22,%20%22view_type%22:%20%22Show%20Both%22,%20%22start_date%22:%20%222018-04-01%22,%20%22end_date%22:%20%222018-04-15%22%7D
'''
@app.route('/heatmap', methods=['GET'])
def chalice_api_get_heatmap():
	request = app.current_request
	#print(request.__dict__)
	json_args = request.query_params['query']
	print (json_args)
	heatmap_data = get_heatmap_json_wrapper(request.query_params['query'])
	return {'heatmap_data': heatmap_data}
	
	
'''
Web:
http://127.0.0.1:8000/dynamicprice?query={"model_type": "Linear", "forecast_and_validate": false, "forecast_period": 30, "query_type": "Custom", "from_cities": ["Delhi"], "to_cities": ["Delhi"], "primary_input_field": "bkg_user_id", "start_date": "2015-07-01 00:00:00", "end_date": "2018-04-15 00:00:00", "plot_components": false, "base_price": "3000", "base_price_low": "0.9", "base_price_high": "1.5", "elasticity": "-2.0", "seasonal_surge": "0.0", "weekend_surge": "0.0", "booking_capacities": ["20.0", "40.0", "60.0", "80.0", "100.0"], "rest_call": true}
CLI: Gives error because of . in capacities
curl -G -i http://127.0.0.1:8000/dynamicprice?query=%7B%22model_type%22:%20%22Linear%22,%20%22forecast_and_validate%22:%20false,%20%22forecast_period%22:%2030,%20%22query_type%22:%20%22Custom%22,%20%22from_cities%22:%20[%22Delhi%22],%20%22to_cities%22:%20[%22Delhi%22],%20%22primary_input_field%22:%20%22bkg_user_id%22,%20%22start_date%22:%20%222015-07-01%2000:00:00%22,%20%22end_date%22:%20%222018-04-15%2000:00:00%22,%20%22plot_components%22:%20false,%20%22base_price%22:%20%223000%22,%20%22base_price_low%22:%20%220.9%22,%20%22base_price_high%22:%20%221.5%22,%20%22elasticity%22:%20%22-2.0%22,%20%22seasonal_surge%22:%20%220.0%22,%20%22weekend_surge%22:%20%220.0%22,%20%22booking_capacities%22:%20[%2220.0%22,%20%2240.0%22,%20%2260.0%22,%20%2280.0%22,%20%22100.0%22],%20%22rest_call%22:%20true%7D
'''
@app.route('/dynamicprice', methods=['GET'])
def chalice_api_get_dynamicprice():
	request = app.current_request
	#print(request.__dict__)
	json_args = request.query_params['query']
	print (json_args)
	dynamic_price_data = get_dynamic_price_json_wrapper(request.query_params['query'])
	return {'dynamic_price_data': dynamic_price_data}
	
	
'''
Web:
http://127.0.0.1:8000/ratequery/oneway/Delhi/Delhi/2018-04-25/2018-04-26/09:00/09:00/40
CLI:
curl -G -i http://127.0.0.1:8000/ratequery/oneway/Delhi/Delhi/2018-04-25/2018-04-26/09:00/09:00/40
'''
@app.route('/ratequery/oneway/{source}/{destination}/{date_start}/{date_end}/{time_start}/{time_end}/{current_capacity}', methods=['GET'])
def chalice_api_ratequery_oneway(
	source, destination, 
	date_start, date_end,
	time_start, time_end,
	current_capacity):
	
	
	rate = api_get_oneway_rate(
		source, destination, 
		date_start, date_end,
		time_start, time_end,
		current_capacity)
	
	return {
		'source'		: source, 
		'destination'	: destination, 
		'date_start' 	: date_start,
		'date_end'		: date_end, 
		'date_start' 	: date_start,
		'time_start'	: time_start, 
		'time_end'		: time_end, 
		'currnet_capacity' : current_capacity,
		'rate'			: rate
	}

