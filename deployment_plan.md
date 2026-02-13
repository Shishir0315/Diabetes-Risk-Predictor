# Deploy Diabetes Risk Model with Flask

## Goal
Deploy the trained Keras regression model as a web application using Flask.

## Prerequisites
- Trained model at `artifacts/diabetes_risk_model.keras`
- Preprocessor at `artifacts/preprocessor.pkl`

## Implementation Steps

### 1. Update Dependencies
- Add `flask` to `requirements.txt`.

### 2. Backend (Flask App)
- File: `app.py`
- Load model and preprocessor on startup.
- Create a route `/` to serve the HTML form.
- Create a route `/predict` (POST) to handle form submissions, preprocess data, and return the prediction.

### 3. Frontend (HTML/CSS)
- File: `templates/index.html`
- A clean, modern form to input all features (Age, Gender, BMI, etc.).
- Use simple CSS for styling.

### 4. Run Script
- Update `run_app.bat` to launch the Flask server.
