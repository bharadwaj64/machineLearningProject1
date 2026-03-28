import pandas as pd 
import numpy as np 
from flask import Flask,render_template,request,jsonify 
from sklearn.linear_model import Ridge 
from sklearn.preprocessing import StandardScaler 
import pickle

scalar = pickle.load(open('scalar.pkl','rb'))
ridge = pickle.load(open('ridge.pkl','rb'))

application = Flask(__name__)
app = application

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/predict',methods = ['GET','POST'])
def predict():
    if request.method == 'POST':
        Temperature = float(request.form['Temperature'])
        RH = float(request.form['RH'])
        Ws = float(request.form['Ws'])
        Rain = float(request.form['Rain'])
        FFMC = float(request.form['FFMC'])
        DMC = float(request.form['DMC'])
        ISI = float(request.form['ISI'])
        Classes = float(request.form['Classes'])
        Region = float(request.form['Region'])

        scalar_values = scalar.transform([[Temperature,RH,Ws,Rain,FFMC,DMC,ISI,Classes,Region]])
        predicted_value = ridge.predict(scalar_values)

        return render_template('result.html',results = predicted_value[0])

    else:
        return render_template('predict.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
