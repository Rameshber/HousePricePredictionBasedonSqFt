# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 11:41:01 2022

@author: Anush Goswami
"""

import numpy as np
from flask import Flask, request, jsonify, render_template

import pickle


app = Flask(__name__)
model = pickle.load(open('linearregression-1.pkl','rb')) 


@app.route('/')
def home():
  
    return render_template("index.html")
  
@app.route('/predict',methods=['GET'])
def predict():
    
    
    '''
    For rendering results on HTML GUI
    '''
    exp = float(request.args.get('exp'))
    
    prediction = model.predict([[exp]])
    
        
    return render_template('index.html', prediction_text='Regression Model  has predicted House Price On Basis Of SqFt is : {}'.format(prediction))

if __name__ == "__main__":
  app.run(debug = True)
