# src/data_preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Load the banana quality dataset.
    """
    return pd.read_csv(file_path)

def preprocess_data(df, target_column):
    """
    Preprocess the data: Handle missing values, split features and target, and scale numerical data.
    """
    # Handling missing values (if any)
    df = df.dropna()

    # Splitting features (X) and target (y)
    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # Scaling the features (numerical columns)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split the data into training and testing sets.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
