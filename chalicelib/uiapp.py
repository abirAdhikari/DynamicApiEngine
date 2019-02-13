import os 
import sys 
import dash
from collections import defaultdict

app = dash.Dash()

app.scripts.config.serve_locally=True
app.config['suppress_callback_exceptions'] = True

global_session_dict = defaultdict(str)

aws_endpoint_url = 'http://127.0.0.1:8000/'
#aws_endpoint_url = 'https://dtrlderllg.execute-api.us-east-1.amazonaws.com/api/'
