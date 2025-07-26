# train.py
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

def train_model():
    housing = fetch_california_housing(as_frame=True)
    X, y = housing.data, housing.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    os.makedirs('models', exist_ok=True)
    model_path = os.path.join('models', 'linear_regression_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model trained and saved to {model_path}")
    print(f"R^2 Score on test set: {model.score(X_test, y_test):.4f}")

if __name__ == "__main__":
    train_model()