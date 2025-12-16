#!/usr/bin/env python3
"""Python script to download model file during deployment.
Usage: MODEL_URL=https://your-url.com/model.pkl python scripts/download_model.py
"""

import os
import sys
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

MODEL_DIR = Path("app/model")
MODEL_FILE = MODEL_DIR / "model.pkl"


def main():
    model_url = os.getenv("MODEL_URL")
    
    if not model_url:
        print("Error: MODEL_URL environment variable is not set", file=sys.stderr)
        print("Set it to the URL where your model.pkl file is hosted", file=sys.stderr)
        sys.exit(1)
    
    # Create model directory if it doesn't exist
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download model file
    print(f"Downloading model from {model_url}...")
    try:
        urlretrieve(model_url, MODEL_FILE)
    except URLError as e:
        print(f"Error downloading model: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Verify file was downloaded
    if MODEL_FILE.exists():
        file_size = MODEL_FILE.stat().st_size
        print(f"Model downloaded successfully (size: {file_size} bytes)")
    else:
        print("Error: Model file was not downloaded", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

