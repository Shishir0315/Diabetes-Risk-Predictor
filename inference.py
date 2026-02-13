
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import os

# Load model and preprocessor
try:
    model_path = 'artifacts/diabetes_risk_model.keras'
    preprocessor_path = 'artifacts/preprocessor.pkl'
    
    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        raise FileNotFoundError("Model or preprocessor not found. Please run train_model_py.")

    model = tf.keras.models.load_model(model_path)
    with open(preprocessor_path, 'rb') as f:
        preprocessor = pickle.load(f)
        
    print("Model and preprocessor loaded successfully.")
    
except Exception as e:
    print(f"Error loading model/preprocessor: {e}")
    exit(1)

def predict_diabetes_risk(input_data):
    """
    Predict diabetes risk score from input data.
    
    Args:
        input_data (dict or pd.DataFrame): Input features.
        
    Returns:
        float: Predicted diabetes risk score.
    """
    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data
    else:
        raise ValueError("Input data must be a dictionary or DataFrame.")
        
    # Preprocess the input
    try:
        processed_data = preprocessor.transform(input_df)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        print("Please check input feature names and types.")
        return None

    # Predict
    prediction = model.predict(processed_data, verbose=0)
    return prediction[0][0]

if __name__ == "__main__":
    # Sample input data (from the dataset, e.g., row 2)
    sample_input = {
        'Age': 58,
        'gender': 'Male',
        'ethnicity': 'Asian',
        'education_level': 'Highschool',
        'income_level': 'Lower-Middle',
        'employment_status': 'Employed',
        'smoking_status': 'Never',
        'alcohol_consumption_per_week': 0,
        'physical_activity_minutes_per_week': 215,
        'diet_score': 5.7,
        'sleep_hours_per_day': 7.9,
        'screen_time_hours_per_day': 7.9,
        'family_history_diabetes': 0,
        'hypertension_history': 0,
        'cardiovascular_history': 0,
        'bmi': 30.5,
        'waist_to_hip_ratio': 0.89,
        'systolic_bp': 134,
        'diastolic_bp': 78,
        'heart_rate': 68,
        'cholesterol_total': 239,
        'hdl_cholesterol': 41,
        'ldl_cholesterol': 160,
        'triglycerides': 145,
        'glucose_fasting': 136,
        'glucose_postprandial': 236,
        'insulin_level': 6.36,
        'hba1c': 8.18
        # diabetes_risk_score was 29.6 for this row
    }
    
    print("\n--- Running Inference on Sample Data ---")
    print(f"Input Data: {sample_input}")
    
    risk_score = predict_diabetes_risk(sample_input)
    
    if risk_score is not None:
        print(f"\nPredicted Diabetes Risk Score: {risk_score:.2f}")
        print(f"Actual Risk Score (from dataset): 29.6") 
