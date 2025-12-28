"""Registry of available models and their feature schemas."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

from app.utils.config import APP_DIR, FEATURE_ORDER as OVARIAN_FEATURE_ORDER

MODEL_DIR = APP_DIR / "model"


@dataclass(frozen=True)
class ModelConfig:
    key: str
    path: Path
    feature_order: List[str]


# Feature orders are expressed using the Pydantic field names (snake_case) used by
# the corresponding request schemas. These map cleanly to JSON via field aliases.
MODEL_REGISTRY: dict[str, ModelConfig] = {
    "ovarian": ModelConfig(
        key="ovarian",
        path=MODEL_DIR / "model.pkl",
        feature_order=OVARIAN_FEATURE_ORDER,
    ),
    "hepatitis_b": ModelConfig(
        key="hepatitis_b",
        path=MODEL_DIR / "Hepatitis_B.pkl",
        # Column order must match the training CSV and HepatitisBRequest aliases.
        feature_order=[
            "Age",
            "Sex",
            "Fatigue",
            "Malaise",
            "Liver_big",
            "Spleen_palpable",
            "Spiders",
            "Ascites",
            "Varices",
            "Bilirubin",
            "Alk_phosphate",
            "Sgot",
            "Albumin",
            "Protime",
            "Histology",
        ],
    ),
    "pcos": ModelConfig(
        key="pcos",
        path=MODEL_DIR / "pcos.pkl",
        feature_order=[
            "Marraige Status (Yrs)",
            "Cycle(R/I)",
            "Pulse rate(bpm)",
            "FSH(mIU/mL)",
            "Age (yrs)",
            "Follicle No. (L)",
            "BMI",
            "Skin darkening (Y/N)",
            "II beta-HCG(mIU/mL)",
            "BP _Diastolic (mmHg)",
            "hair growth(Y/N)",
            "Avg. F size (L) (mm)",
            "Avg. F size (R) (mm)",
            "Waist:Hip Ratio",
            "Weight (Kg)",
            "Weight gain(Y/N)",
            "LH(mIU/mL)",
            "Follicle No. (R)",
            "Hip(inch)",
            "Waist(inch)",
        ],
    ),
    "brain_tumor": ModelConfig(
        key="brain_tumor",
        path=MODEL_DIR / "model_PTH.pth",
        feature_order=[],  # Image-based model, no feature order needed
    ),
}

