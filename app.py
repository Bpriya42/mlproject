import pickle
from flask import Flask , request, render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler

application = Flask(__name__)

app = application

@app.route('/')

def index():
    return render_template('index.html')

@app.route('/predictdata', methods = ['GET', 'POST'])
def predict_datapoint():
    '''Get and predict data'''

    if request.method == 'GET':
        return render_template('home.html') # simple input data fields to do the predictions
    else:
        # capture data
        # standard scaling or feature scaling
        
        pass