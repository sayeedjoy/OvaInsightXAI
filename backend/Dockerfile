FROM python:3.11-slim

WORKDIR /app

# Build version - increment to force rebuild with new training data
# v3: Cloud-optimized build with better caching and health checks
ARG BUILD_VERSION=v3

# Set environment variables for Python optimization
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    # Reduce memory usage for ML frameworks
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    # PyTorch CPU optimization
    TORCH_NUM_THREADS=2

# Install system dependencies for Python packages that need compilation
# SHAP requires libgomp1, libgcc-s1, and g++ for C++ extensions
# OpenCV headless needs libgl1-mesa-glx for some operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    libgcc-s1 \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements and install Python dependencies
# Using a multi-stage approach to leverage Docker layer caching
COPY requirements.txt .

# Install dependencies with increased pip timeout for cloud builds
RUN pip install --no-cache-dir --timeout 300 -r requirements.txt \
    && pip cache purge

# Ensure model directory exists and remove any old incompatible model BEFORE copying app code
RUN mkdir -p /app/app/model && \
    rm -rf /app/app/model/model.pkl

# Copy application code (this may include old model.pkl, but we'll remove it and retrain)
COPY app/ ./app/

# Force remove any old ovarian model that might have been copied with app code
# Other models (Hepatitis, PCOS, Brain Tumor) are kept as-is
RUN rm -f /app/app/model/model.pkl && \
    test ! -f /app/app/model/model.pkl || (echo "ERROR: Old ovarian model still exists!" && exit 1)

# Copy and run model retraining script to ensure model compatibility with installed sklearn version
# BUILD_VERSION=$BUILD_VERSION forces cache invalidation when training data changes
COPY retrain_model.py .
RUN echo "Building with version: $BUILD_VERSION" && \
    python retrain_model.py && \
    rm retrain_model.py && \
    test -f /app/app/model/model.pkl || (echo "ERROR: Ovarian model file not created!" && exit 1) && \
    ls -lh /app/app/model/model.pkl && \
    chmod 644 /app/app/model/model.pkl

# Verify all model files exist
RUN echo "Verifying model files..." && \
    ls -la /app/app/model/ && \
    test -f /app/app/model/model.pkl && echo "✓ Ovarian model exists" && \
    test -f /app/app/model/Hepatitis_B.pkl && echo "✓ Hepatitis B model exists" && \
    test -f /app/app/model/pcos.pkl && echo "✓ PCOS model exists" && \
    test -f /app/app/model/model_PTH.pth && echo "✓ Brain tumor model exists" && \
    echo "All model files verified!"

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

# Health check to ensure the application is responding
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Run the application
# Use PORT environment variable if set, otherwise default to 8000
# Add --log-level info for better debugging
# Increase timeout for large model loading
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --log-level info --timeout-keep-alive 120"]
