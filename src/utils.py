# src/utils.py

from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(y_test, y_pred):
    """
    Evaluate the machine learning model using accuracy and classification report.
    """
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.2f}")
    print("Classification Report:")
    print(report)
