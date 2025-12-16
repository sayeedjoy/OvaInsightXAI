"""FastAPI entrypoint for the ML prediction service.

Run locally with: uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.model import predictor
from app.routes import predict
from app.utils.config import ALLOWED_ORIGINS


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Ensure the model is loaded at startup so first request is fast."""
    predictor.warmup()
    yield


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


