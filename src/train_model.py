# src/train_model.py

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_preprocessing import load_data, preprocess_data, split_data
from src.utils import evaluate_model
from sklearn.ensemble import RandomForestClassifier

def train_and_evaluate():
    # Load the dataset
    file_path = os.path.join('data', 'banana_quality.csv')
    df = load_data(file_path)

    # Preprocess the data
    target_column = 'Quality'  # Ganti dengan 'Quality'
    X, y = preprocess_data(df, target_column)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Train the model using Random Forest
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict the target on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model
    evaluate_model(y_test, y_pred)
