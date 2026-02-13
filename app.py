
from flask import Flask, render_template, request, make_response
import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load model and preprocessor
MODEL_PATH = 'artifacts/diabetes_risk_model.keras'
PREPROCESSOR_PATH = 'artifacts/preprocessor.pkl'

if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(PREPROCESSOR_PATH, 'rb') as f:
        preprocessor = pickle.load(f)
    print("Model and preprocessor loaded.")
else:
    print("Error: Model or preprocessor artifacts not found.")
    exit(1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        form_data = request.form.to_dict()
        
        # Convert numeric fields
        numeric_fields = [
            'Age', 'alcohol_consumption_per_week', 'physical_activity_minutes_per_week',
            'diet_score', 'sleep_hours_per_day', 'screen_time_hours_per_day',
            'family_history_diabetes', 'hypertension_history', 'cardiovascular_history',
            'bmi', 'waist_to_hip_ratio', 'systolic_bp', 'diastolic_bp', 'heart_rate',
            'cholesterol_total', 'hdl_cholesterol', 'ldl_cholesterol', 'triglycerides',
            'glucose_fasting', 'glucose_postprandial', 'insulin_level', 'hba1c'
        ]
        
        input_data = {}
        for key, value in form_data.items():
            if key in numeric_fields:
                input_data[key] = float(value)
            else:
                input_data[key] = value
        
        # Create DataFrame for preprocessing
        input_df = pd.DataFrame([input_data])
        
        # Preprocess
        processed_data = preprocessor.transform(input_df)
        
        # Predict
        prediction = model.predict(processed_data)
        risk_score = float(prediction[0][0])
        
        return render_template('index.html', prediction=f"{risk_score:.2f}", input_data=input_data)

    except Exception as e:
        return render_template('index.html', error=str(e), input_data=form_data)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
