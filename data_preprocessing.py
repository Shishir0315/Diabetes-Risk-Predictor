
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import pickle
import os

# Create artifacts directory if it doesn't exist
os.makedirs('artifacts', exist_ok=True)

# Load dataset
file_path = 'd:/CNN Projects/regression dataset/Diabetes_and_LifeStyle_Dataset .csv'
df = pd.read_csv(file_path)

# Drop unnecessary columns
# diagnosed_diabetes and diabetes_stage are droped to prevent data leakage (since we want to predict risk)
drop_cols = ['diagnosed_diabetes', 'diabetes_stage']
df = df.drop(columns=drop_cols)

# Separate features and target
target = 'diabetes_risk_score'
X = df.drop(columns=[target])
y = df[target]

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Preprocessing for numerical data
numerical_transformer = StandardScaler()

# Preprocessing for categorical data
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ]
)

# Apply preprocessing
X_processed = preprocessor.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Save the preprocessor
with open('artifacts/preprocessor.pkl', 'wb') as f:
    pickle.dump(preprocessor, f)

# Save processed data
np.save('artifacts/X_train.npy', X_train)
np.save('artifacts/X_test.npy', X_test)
np.save('artifacts/y_train.npy', y_train)
np.save('artifacts/y_test.npy', y_test)

print("Preprocessing complete. Artifacts saved.")
