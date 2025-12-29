"""FastAPI entrypoint for the ML prediction service.

Run locally with: uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import traceback
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.model import predictor
from app.routes import predict
from app.utils.config import ALLOWED_ORIGINS, MODEL_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# S3 URL for model download
MODEL_URL = "https://cdn.rumorscanner.net/files/model.pkl"


def download_model_if_needed() -> bool:
    """
    Download the model from S3 URL if it doesn't exist at any configured path.
    Returns True if download was successful, False otherwise.
    """
    # If MODEL_PATH was explicitly provided by env var, don't overwrite it by default.
    # This prevents silently downloading an old/incompatible model into a custom mount path.
    if "MODEL_PATH" in os.environ and os.getenv("ALLOW_MODEL_DOWNLOAD", "0") != "1":
        if not MODEL_PATH.exists():
            logger.warning(
                "MODEL_PATH is set to %s but file does not exist. "
                "Skipping auto-download because ALLOW_MODEL_DOWNLOAD!=1.",
                MODEL_PATH,
            )
        return MODEL_PATH.exists()

    # Check if model already exists
    if MODEL_PATH.exists():
        logger.info(f"Model already exists at {MODEL_PATH}, skipping download")
        return True
    
    # Check alternative locations
    alternative_paths = [
        Path("/opt/models/model.pkl"),
        Path("/app/app/model/model.pkl"),
    ]
    
    for alt_path in alternative_paths:
        if alt_path.exists():
            logger.info(f"Model found at {alt_path}, skipping download")
            return True
    
    # Model not found, attempt download
    logger.info(f"Model not found, attempting to download from {MODEL_URL}")
    
    try:
        # Ensure the target directory exists
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Download the model
        logger.info(f"Downloading model to {MODEL_PATH}...")
        urlretrieve(MODEL_URL, MODEL_PATH)
        
        # Verify download
        if MODEL_PATH.exists():
            file_size = MODEL_PATH.stat().st_size
            logger.info(f"Model downloaded successfully (size: {file_size} bytes)")
            return True
        else:
            logger.error("Model download completed but file not found")
            return False
            
    except URLError as exc:
        logger.warning(f"Failed to download model from {MODEL_URL}: {exc}")
        logger.warning("Application will continue, but model loading may fail")
        return False
    except Exception as exc:
        logger.error(f"Unexpected error during model download: {exc}", exc_info=True)
        logger.warning("Application will continue, but model loading may fail")
        return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure the model is loaded at startup so first request is fast."""
    logger.info("Starting application...")
    
    # Download model from S3 if needed
    logger.info("Checking for model file...")
    download_model_if_needed()
    
    # Attempt to load the model
    logger.info("Attempting to load model...")
    # Call warmup - it won't raise exceptions anymore
    predictor.warmup()
    logger.info("Application startup complete")
    yield
    logger.info("Application shutting down...")


app = FastAPI(
    title="ML Prediction API",
    description="Serves predictions for the 12-feature medical model.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(predict.router)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Global exception handler to catch all unhandled exceptions."""
    logger.error(
        "Unhandled exception occurred",
        exc_info=True,
        extra={
            "path": request.url.path,
            "method": request.method,
            "exception_type": type(exc).__name__,
        }
    )
    
    # Return a proper error response
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": f"Internal server error: {str(exc)}",
            "error_type": type(exc).__name__,
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """Handle request validation errors with detailed information."""
    logger.warning("Request validation error: %s", exc.errors())
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": exc.errors()}
    )


@app.get("/")
async def root():
    """Root endpoint for health checks and API information."""
    return {
        "status": "ok",
        "message": "ML Prediction API is running",
        "version": "2.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

