FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY django_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Django application
COPY django_app/ .

# Copy valuation_core from parent directory
COPY valuation_core /app/valuation_core

# Create static files directory
RUN mkdir -p /app/static

# Collect static files
RUN python manage.py collectstatic --noinput

# Expose port
EXPOSE 8000

# Run Django development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]