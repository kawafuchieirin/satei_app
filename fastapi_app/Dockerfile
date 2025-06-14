FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY fastapi_app/requirements-ml.txt .
RUN pip install --no-cache-dir -r requirements-ml.txt

# Copy FastAPI application
COPY fastapi_app/ .

# Copy valuation_core from parent directory
COPY valuation_core /app/valuation_core

# Copy ML model files from model-creation
COPY model-creation/models/*.joblib /app/models/

# Create models directory if it doesn't exist
RUN mkdir -p /app/models

# Expose port
EXPOSE 8001

# Run FastAPI server with shared MLPredictor
CMD ["uvicorn", "main_shared:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]