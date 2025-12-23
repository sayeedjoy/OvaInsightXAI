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


class HepatitisBRequest(BaseModel):
    """Payload for the hepatitis B model (15 features, ordered to match training columns)."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    age: float = Field(..., alias="Age", description="Age in years")
    sex: float = Field(..., alias="Sex", description="Sex (encoded numeric)")
    fatigue: float = Field(..., alias="Fatigue", description="Fatigue indicator")
    malaise: float = Field(..., alias="Malaise", description="Malaise indicator")
    liver_big: float = Field(..., alias="Liver_big", description="Liver enlargement indicator")
    spleen_palpable: float = Field(..., alias="Spleen_palpable", description="Palpable spleen indicator")
    spiders: float = Field(..., alias="Spiders", description="Spider angioma indicator")
    ascites: float = Field(..., alias="Ascites", description="Ascites indicator")
    varices: float = Field(..., alias="Varices", description="Varices indicator")
    bilirubin: float = Field(..., alias="Bilirubin", description="Bilirubin level")
    alk_phosphate: float = Field(..., alias="Alk_phosphate", description="Alkaline phosphatase")
    sgot: float = Field(..., alias="Sgot", description="SGOT level")
    albumin: float = Field(..., alias="Albumin", description="Serum albumin level")
    protime: float = Field(..., alias="Protime", description="Prothrombin time")
    histology: float = Field(..., alias="Histology", description="Histology indicator")


class PcosRequest(BaseModel):
    """Payload for the PCOS model."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    marriage_status_years: float = Field(..., alias="Marraige Status (Yrs)", description="Years married")
    cycle_regular_irregular: float = Field(..., alias="Cycle(R/I)", description="Cycle regularity encoded")
    pulse_rate_bpm: float = Field(..., alias="Pulse rate(bpm)", description="Pulse rate")
    fsh_miu_ml: float = Field(..., alias="FSH(mIU/mL)", description="FSH level")
    age_years: float = Field(..., alias="Age (yrs)", description="Age in years")
    follicle_no_left: float = Field(..., alias="Follicle No. (L)", description="Follicle count left")
    bmi: float = Field(..., alias="BMI", description="Body Mass Index")
    skin_darkening_yn: float = Field(..., alias="Skin darkening (Y/N)", description="Skin darkening indicator")
    ii_beta_hcg_miu_ml: float = Field(..., alias="II beta-HCG(mIU/mL)", description="II beta-HCG level")
    bp_diastolic_mmhg: float = Field(..., alias="BP _Diastolic (mmHg)", description="Diastolic BP")
    hair_growth_yn: float = Field(..., alias="hair growth(Y/N)", description="Hair growth indicator")
    avg_f_size_left_mm: float = Field(..., alias="Avg. F size (L) (mm)", description="Average follicle size left")
    avg_f_size_right_mm: float = Field(..., alias="Avg. F size (R) (mm)", description="Average follicle size right")
    waist_hip_ratio: float = Field(..., alias="Waist:Hip Ratio", description="Waist-to-hip ratio")
    weight_kg: float = Field(..., alias="Weight (Kg)", description="Weight in kg")
    weight_gain_yn: float = Field(..., alias="Weight gain(Y/N)", description="Weight gain indicator")
    lh_miu_ml: float = Field(..., alias="LH(mIU/mL)", description="LH level")
    follicle_no_right: float = Field(..., alias="Follicle No. (R)", description="Follicle count right")
    hip_inch: float = Field(..., alias="Hip(inch)", description="Hip in inches")
    waist_inch: float = Field(..., alias="Waist(inch)", description="Waist in inches")

