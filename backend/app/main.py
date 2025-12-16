"""FastAPI entrypoint for the ML prediction service.

Run locally with: uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.model import predictor
from app.routes import predict
from app.utils.config import ALLOWED_ORIGINS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure the model is loaded at startup so first request is fast."""
    logger.info("Starting application...")
    logger.info("Attempting to load model...")
    # Call warmup - it won't raise exceptions anymore
    predictor.warmup()
    logger.info("Application startup complete")
    yield
    logger.info("Application shutting down...")


app = FastAPI(
    title="ML Prediction API",
    description="Serves predictions for the 12-feature medical model.",
    version="0.1.0",
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
        "version": "0.1.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

