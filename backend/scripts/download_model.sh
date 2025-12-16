#!/bin/bash
# Script to download model file during deployment
# Usage: MODEL_URL=https://your-url.com/model.pkl ./scripts/download_model.sh

set -e

MODEL_DIR="app/model"
MODEL_FILE="$MODEL_DIR/model.pkl"

# Check if MODEL_URL is set
if [ -z "$MODEL_URL" ]; then
    echo "Error: MODEL_URL environment variable is not set"
    echo "Set it to the URL where your model.pkl file is hosted"
    exit 1
fi

# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Download model file
echo "Downloading model from $MODEL_URL..."
if command -v curl &> /dev/null; then
    curl -L "$MODEL_URL" -o "$MODEL_FILE"
elif command -v wget &> /dev/null; then
    wget "$MODEL_URL" -O "$MODEL_FILE"
else
    echo "Error: Neither curl nor wget is available"
    exit 1
fi

# Verify file was downloaded
if [ -f "$MODEL_FILE" ]; then
    FILE_SIZE=$(stat -f%z "$MODEL_FILE" 2>/dev/null || stat -c%s "$MODEL_FILE" 2>/dev/null || echo "0")
    echo "Model downloaded successfully (size: $FILE_SIZE bytes)"
else
    echo "Error: Model file was not downloaded"
    exit 1
fi

