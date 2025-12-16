"""Pydantic schemas shared across the API."""

from typing import Union

from pydantic import BaseModel, ConfigDict, Field


class PredictionRequest(BaseModel):
    """Represents the 12-feature payload expected by the model."""

    model_config = ConfigDict(extra="forbid")

    age: float = Field(..., description="Age in years")
    alb: float = Field(..., description="Albumin level")
    alp: float = Field(..., description="Alkaline phosphatase")
    bun: float = Field(..., description="Blood urea nitrogen")
    ca125: float = Field(..., description="Cancer antigen 125")
    eo_abs: float = Field(..., description="Eosinophil absolute count")
    ggt: float = Field(..., description="Gamma-glutamyl transferase")
    he4: float = Field(..., description="Human epididymis protein 4")
    mch: float = Field(..., description="Mean corpuscular hemoglobin")
    mono_abs: float = Field(..., description="Monocyte absolute count")
    na: float = Field(..., description="Sodium level")
    pdw: float = Field(..., description="Platelet distribution width")


class PredictionResponse(BaseModel):
    """Response body returned by the /predict endpoint."""

    prediction: Union[int, float]
    confidence: float | None = None


class HealthResponse(BaseModel):
    """Simple health check payload."""

    status: str


