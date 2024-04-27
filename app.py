from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load("student_mark_predictor.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            study_hours = float(request.form['study_hours'])
            prediction = model.predict([[study_hours]])[0][0].round(2)
            return render_template('result.html', study_hours=study_hours, prediction=prediction)
        except ValueError:
            return render_template('error.html', message='Please enter a valid numeric value for study hours.')

if __name__ == '__main__':
    app.run(debug=True)
