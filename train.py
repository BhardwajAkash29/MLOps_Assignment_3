import joblib
import os
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load the California Housing dataset
housing = fetch_california_housing(as_frame=True)
X = housing.data
y = housing.target

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"Model trained and saved to models/linear_regression_model.joblib")
print(f"R^2 Score on test set: {r2:.4f}")

# --- NEW ADDITIONS BELOW ---

# Create the 'models' directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Save the trained model
model_path = os.path.join('models', 'linear_regression_model.joblib')
joblib.dump(model, model_path)

# Save a small portion of the test data for prediction verification
# Using X_test from the split. You might save a smaller sample if X_test is very large.
test_data_path = os.path.join('models', 'test_data.joblib')
joblib.dump(X_test.head(5), test_data_path) # Saving first 5 rows of X_test as an example
# Or, if you need the full X_test:
# joblib.dump(X_test, test_data_path)