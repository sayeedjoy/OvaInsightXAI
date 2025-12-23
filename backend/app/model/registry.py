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
        feature_order=[
            "age",
            "sex",
            "fatigue",
            "malaise",
            "liver_big",
            "spleen_palpable",
            "spiders",
            "ascites",
            "varices",
            "bilirubin",
            "alk_phosphate",
            "sgot",
            "albumin",
            "protime",
            "histology",
        ],
    ),
    "pcos": ModelConfig(
        key="pcos",
        path=MODEL_DIR / "pcos.pkl",
        feature_order=[
            "age_years",
            "marriage_status_years",
            "cycle_regular_irregular",
            "pulse_rate_bpm",
            "bp_diastolic_mmhg",
            "weight_kg",
            "weight_gain_yn",
            "height_bmi_related",
            "bmi",
            "waist_inch",
            "hip_inch",
            "waist_hip_ratio",
            "hair_growth_yn",
            "skin_darkening_yn",
            "fsh_miu_ml",
            "lh_miu_ml",
            "ii_beta_hcg_miu_ml",
            "follicle_no_left",
            "follicle_no_right",
            "avg_f_size_left_mm",
            "avg_f_size_right_mm",
        ],
    ),
}

