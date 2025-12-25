FROM python:3.11-slim

WORKDIR /app

# Build version - increment to force rebuild with new training data
# v2: Balanced classes for better predictions
ARG BUILD_VERSION=v2

# Install system dependencies for Python packages that need compilation
# SHAP requires libgomp1, libgcc-s1, and g++ for C++ extensions
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libgomp1 \
    libgcc-s1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Ensure model directory exists and remove any old incompatible model BEFORE copying app code
RUN mkdir -p /app/app/model && \
    rm -rf /app/app/model/model.pkl

# Copy application code (this may include old model.pkl, but we'll remove it and retrain)
COPY app/ ./app/

# Force remove any old model that might have been copied with app code
RUN rm -f /app/app/model/model.pkl && \
    test ! -f /app/app/model/model.pkl || (echo "ERROR: Old model still exists!" && exit 1)

# Copy and run model retraining script to ensure model compatibility with installed sklearn version
# BUILD_VERSION=$BUILD_VERSION forces cache invalidation when training data changes
COPY retrain_model.py .
RUN echo "Building with version: $BUILD_VERSION" && \
    python retrain_model.py && \
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

