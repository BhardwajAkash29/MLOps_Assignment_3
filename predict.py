# predict.py
import joblib
import os
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

def verify_prediction():
    model_path = os.path.join('models', 'linear_regression_model.joblib')
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Make sure train.py runs first.")
        exit(1)
    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    housing = fetch_california_housing(as_frame=True)
    X, y = housing.data, housing.target
    _, X_test, _, _ = train_test_split(X, y, test_size=0.01, random_state=42)
    if not X_test.empty:
        sample_prediction = model.predict(X_test.head(1))
        print(f"Sample prediction: {sample_prediction}")
        print("Model prediction verified successfully!")
    else:
        print("No data available for sample prediction.")
        exit(1)

if __name__ == "__main__":
    verify_prediction()