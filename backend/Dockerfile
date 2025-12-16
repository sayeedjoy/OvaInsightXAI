FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Python packages that need compilation
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Ensure model directory exists and remove any old incompatible model
RUN mkdir -p /app/app/model && \
    rm -f /app/app/model/model.pkl

# Copy and run model retraining script to ensure model compatibility with installed sklearn version
COPY retrain_model.py .
RUN python retrain_model.py && \
    rm retrain_model.py && \
    test -f /app/app/model/model.pkl || (echo "ERROR: Model file not created!" && exit 1) && \
    ls -lh /app/app/model/model.pkl && \
    chmod 644 /app/app/model/model.pkl

# Create model directory if MODEL_PATH points to a different location
# This allows the model to be mounted or placed in /opt/models
RUN mkdir -p /opt/models

# Create non-root user for security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /opt/models && \
    # Verify model file exists and is accessible
    test -f /app/app/model/model.pkl || (echo "ERROR: Model file missing after chown!" && exit 1) && \
    ls -lh /app/app/model/model.pkl
USER appuser

# Expose port
EXPOSE 8000

# Run the application
# Use PORT environment variable if set, otherwise default to 8000
# Add --log-level info for better debugging
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info"]

