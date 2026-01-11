# Use Python 3.12 slim image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy minimal requirements for inference
COPY requirements-inference.txt /app/requirements-inference.txt

# Install Python dependencies (only what's needed for inference)
RUN pip install --no-cache-dir -r requirements-inference.txt

# Copy only necessary source code for inference
COPY src/predict.py /app/src/predict.py

# Create __init__.py if it doesn't exist
RUN touch /app/src/__init__.py

# Create models directory
# Model must be provided via volume mount: -v $(pwd)/models:/app/models
RUN mkdir -p /app/models/baseline

# Set Python path
ENV PYTHONPATH=/app

# Set entrypoint
ENTRYPOINT ["python", "-m", "src.predict"]

# Default arguments (can be overridden)
CMD ["--input_path", "/data/input.csv", "--output_path", "/data/predictions.csv"]
