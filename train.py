# train.py
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import os

def train_model():
    # Load the California Housing dataset
    housing = fetch_california_housing(as_frame=True)
    X, y = housing.data, housing.target

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the Linear Regression model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Create a directory for models if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Save the trained model
    model_path = os.path.join('models', 'linear_regression_model.joblib')
    joblib.dump(model, model_path)
    print(f"Model trained and saved to {model_path}")

    # You can optionally print R^2 score for verification
    r2_score = model.score(X_test, y_test)
    print(f"R^2 Score on test set: {r2_score}")

if __name__ == "__main__":
    train_model()