<<<<<<< HEAD

# Diabetes Risk Prediction Model

This project predicts diabetes risk score based on lifestyle metrics using a Neural Network.

## Prerequisites
- Windows OS with Python installed (via Microsoft Store or python.org).
- Ensure execution aliases for Python are enabled if using `python` command, or use `py` launcher.

## Running the Project
1. **Install Dependencies**:
   Open a terminal in this folder and run:
   ```cmd
   py -m pip install -r requirements.txt
   ```
   Or simply run `run_inference.bat`.

2. **Run Inference**:
   To predict risk for a sample input:
   ```cmd
   py inference.py
   ```

3. **Train Model** (Optional):
   If you want to retrain the model:
   ```cmd
   py train_model.py
   ```

## Files
- `data_preprocessing.py`: Cleans and prepares data.
- `train_model.py`: Trains the Neural Network.
- `inference.py`: Loads the model and predicts.
- `run_inference.bat`: Helper script to run inference.
=======
# Diabetes-Risk-Predictor
An end-to-end Deep Learning web app for predicting diabetes risk. Built with TensorFlow/Keras neural networks and deployed via Flask. Features a complete preprocessing pipeline and a modern, responsive UI for real-time health risk assessment based on lifestyle metrics like BMI and glucose.
>>>>>>> d74067194d26de28d7bf8273b84c27ba16542704
