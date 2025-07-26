# Use a slim Python base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy requirements.txt and install dependencies
# This is done first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create the src directory inside the container
RUN mkdir -p src

# Create the models directory inside the container (if train.py doesn't create it reliably within the image build)
# It's good practice to ensure the directory exists for copies.
RUN mkdir -p models

# Copy the rest of your application code
# train.py is expected in /app (root of WORKDIR)
COPY train.py .
# predict.py is expected in /app/src/
COPY predict.py src/

# --- NEW CRITICAL ADDITION BELOW ---
# Copy the models directory from the runner's workspace into the Docker image
# This directory should contain linear_regression_model.joblib and test_data.joblib
COPY models/ models/

# Command to run when the container starts (optional, for default behavior)
# This will be overridden by the `docker run ... python src/predict.py` in CI
CMD ["python", "src/predict.py"]