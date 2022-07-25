import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd 
app = Flask(__name__)
model = pickle.load(open('randomforest_model.pkl', 'rb'))
flights = pd.read_csv("data.csv", low_memory = False)

ip_feat = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean','fractal_dimension_mean',
                 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
                 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst','fractal_dimension_worst']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    cancaer_ip_data = [float(x) for x in request.form.values()]
    final_features = [np.array(cancaer_ip_data, dtype = int)]
    prediction = model.predict(final_features)
    print(" prediction: ", prediction)

    if prediction <= 0.5:
        x = 'Benign'
        return render_template('index11.html', prediction_text='The patient is diagnosied with {}.'.format(x))
    else:
        x = 'Malignant'
    return render_template('index11.html', prediction_text='The patient is diagnosied with {}.'.format(x))

if __name__ == "__main__":
    app.run(debug=True)