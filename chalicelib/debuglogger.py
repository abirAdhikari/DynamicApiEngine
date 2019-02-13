# AAdhikari: 8/30/2018

from datetime import datetime, timedelta, date
import inspect
import sys
import traceback
	
def logutil(i_classname, i_message):
	if True:
		stack = traceback.extract_stack()
		filename, codeline, funcName, text = stack[-2]
		print("%s %s::%s() %s" % (datetime.now(), i_classname, funcName, i_message))
	pass

	
def print_seperator_start(meta=''):
	print("")
	print("--------------------------------------------------" +meta+ "--------------------------------------------------")

def print_seperator_end(meta=''):
	print("--------------------------------------------------" +meta+ "--------------------------------------------------")
	print("")

def print_dataframe(df, meta='', p=False):
	if p:
		print_seperator_start('')
		print('Dataframe Length: ', len(df))
		print('Start Dataframe : ' + meta)
		print(df)
		print('End Dataframe : ' + meta)
		print_seperator_end()
		
	
