# Use a slim Python base image for smaller image size
FROM python:3.9-slim-buster

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt first to leverage Docker layer caching (if dependencies don't change often)
COPY requirements.txt .
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create necessary directories inside the container
# predict.py will be copied into src/, and models/ will hold trained models and test data
RUN mkdir -p src
RUN mkdir -p models

# Copy your application code into the container
# train.py is expected in the root of the container's WORKDIR
COPY train.py .
# predict.py is copied into the src/ directory inside the container
COPY predict.py src/

# CRITICAL: Copy the 'models/' directory (containing the trained model and test data)
# from the GitHub Actions runner's workspace into the Docker image.
# This ensures predict.py can find them inside the container.
COPY models/ models/

# Define the default command to run when the container starts.
# This serves as a default but will be overridden by the `docker run ... python src/predict.py`
# command in your GitHub Actions workflow for verification.
CMD ["python", "src/predict.py"]