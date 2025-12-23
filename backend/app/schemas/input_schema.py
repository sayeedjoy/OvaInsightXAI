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
    """Payload for the hepatitis B model."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    age: float = Field(..., description="Age in years")
    sex: float = Field(..., description="Sex (encoded numeric)")
    fatigue: float = Field(..., description="Fatigue indicator")
    malaise: float = Field(..., description="Malaise indicator")
    liver_big: float = Field(..., description="Liver enlargement indicator")
    spleen_palpable: float = Field(..., description="Palpable spleen indicator")
    spiders: float = Field(..., description="Spider angioma indicator")
    ascites: float = Field(..., description="Ascites indicator")
    varices: float = Field(..., description="Varices indicator")
    bilirubin: float = Field(..., description="Bilirubin level")
    alk_phosphate: float = Field(..., description="Alkaline phosphatase")
    sgot: float = Field(..., description="SGOT level")
    albumin: float = Field(..., description="Albumin level")
    protime: float = Field(..., description="Prothrombin time")
    histology: float = Field(..., description="Histology indicator")


class PcosRequest(BaseModel):
    """Payload for the PCOS model."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    age_years: float = Field(..., alias="Age (yrs)", description="Age in years")
    marriage_status_years: float = Field(..., alias="Marraige Status (Yrs)", description="Years married")
    cycle_regular_irregular: float = Field(..., alias="Cycle(R/I)", description="Cycle regularity encoded")
    pulse_rate_bpm: float = Field(..., alias="Pulse rate (bpm)", description="Pulse rate")
    bp_diastolic_mmhg: float = Field(..., alias="BP _Diastolic (mmHg)", description="Diastolic BP")
    weight_kg: float = Field(..., alias="Weight (Kg)", description="Weight in kg")
    weight_gain_yn: float = Field(..., alias="Weight gain (Y/N)", description="Weight gain indicator")
    height_bmi_related: float = Field(..., alias="Height / BMI related", description="Height/BMI related feature")
    bmi: float = Field(..., alias="BMI", description="Body Mass Index")
    waist_inch: float = Field(..., alias="Waist(inch)", description="Waist in inches")
    hip_inch: float = Field(..., alias="Hip(inch)", description="Hip in inches")
    waist_hip_ratio: float = Field(..., alias="Waist:Hip Ratio", description="Waist-to-hip ratio")
    hair_growth_yn: float = Field(..., alias="Hair growth (Y/N)", description="Hair growth indicator")
    skin_darkening_yn: float = Field(..., alias="Skin darkening (Y/N)", description="Skin darkening indicator")
    fsh_miu_ml: float = Field(..., alias="FSH (mIU/mL)", description="FSH level")
    lh_miu_ml: float = Field(..., alias="LH (mIU/mL)", description="LH level")
    ii_beta_hcg_miu_ml: float = Field(..., alias="II beta-HCG (mIU/mL)", description="II beta-HCG level")
    follicle_no_left: float = Field(..., alias="Follicle No. (L)", description="Follicle count left")
    follicle_no_right: float = Field(..., alias="Follicle No. (R)", description="Follicle count right")
    avg_f_size_left_mm: float = Field(..., alias="Avg. F size (L) (mm)", description="Average follicle size left")
    avg_f_size_right_mm: float = Field(..., alias="Avg. F size (R) (mm)", description="Average follicle size right")

