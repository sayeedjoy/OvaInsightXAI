# MultiDx Clinical AI - Multi-Disease Prediction System

A comprehensive full-stack web application for predicting multiple medical conditions using machine learning. This system provides early risk assessment for **Ovarian Cancer**, **PCOS (Polycystic Ovary Syndrome)**, and **Hepatitis B**, helping healthcare professionals make informed decisions.

## Overview

This application combines a FastAPI backend with a Next.js frontend to deliver an intuitive interface for multiple disease predictions. The system uses trained machine learning models to analyze patient data and provide predictions with confidence scores for three different medical conditions.

## Features

- **Multi-Disease Support**: Predictions for Ovarian Cancer, PCOS, and Hepatitis B
- **Ovarian Cancer Analysis**: Input 12 critical biomarkers including CA125, HE4, and other clinical indicators
- **PCOS Analysis**: Input 20 clinical features including hormonal levels, BMI, and follicle measurements
- **Hepatitis B Analysis**: Input 15 clinical and laboratory features including liver function tests
- **Real-time Prediction**: Get instant predictions with confidence scores
- **Test Case Generation**: Built-in test case generators for positive and negative scenarios
- **User-friendly Interface**: Clean, responsive design that works on all devices
- **Visual Results**: Color-coded results with detailed confidence metrics
- **Model Registry**: Centralized model management system supporting multiple ML models
- **Medical Disclaimer**: Built-in disclaimers emphasizing the tool is for research purposes

## Tech Stack

### Backend
- **FastAPI** - Modern, fast web framework for building APIs
- **Python** - Core programming language
- **scikit-learn** - Machine learning model
- **Pydantic** - Data validation
- **Uvicorn** - ASGI server

### Frontend
- **Next.js 15** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first CSS framework
- **Lucide Icons** - Modern icon library
- **React Hook Form** - Form state management

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application entry point
│   │   ├── routes/
│   │   │   └── predict.py       # Prediction endpoints for all models
│   │   ├── schemas/
│   │   │   └── input_schema.py  # Pydantic models for validation (all models)
│   │   ├── model/
│   │   │   ├── model.pkl        # Trained Ovarian Cancer model
│   │   │   ├── pcos.pkl         # Trained PCOS model
│   │   │   ├── Hepatitis_B.pkl  # Trained Hepatitis B model
│   │   │   ├── predictor.py     # Model loading and inference
│   │   │   └── registry.py      # Model registry and configuration
│   │   ├── services/
│   │   │   └── preprocessing.py # Data preprocessing utilities
│   │   └── utils/
│   │       └── config.py        # Configuration constants
│   ├── requirements.txt         # Python dependencies
│   └── retrain_model.py         # Model retraining script
│
├── frontend/
│   ├── src/
│   │   ├── app/
│   │   │   ├── ovarian/         # Ovarian Cancer prediction page
│   │   │   ├── pcos/            # PCOS prediction page
│   │   │   ├── hepatitis/       # Hepatitis B prediction page
│   │   │   └── api/             # API routes
│   │   ├── components/
│   │   │   ├── prediction-components/  # Shared prediction UI components
│   │   │   ├── pcos-components/        # PCOS-specific components
│   │   │   └── hepatitis-components/    # Hepatitis-specific components
│   │   ├── lib/
│   │   │   ├── test-case-generator.ts      # Ovarian test case generator
│   │   │   ├── pcos-test-case-generator.ts # PCOS test case generator
│   │   │   └── hepatitis-test-case-generator.ts # Hepatitis test case generator
│   │   └── styles/              # Global styles
│   ├── package.json
│   └── next.config.ts
│
└── README.md
```

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+** (for backend)
- **Node.js 18+** (for frontend)
- **pnpm** or **npm** (package manager)

## Installation

# For local use scikit-learn>=1.6.0
# For Server end scikit-learn==1.5.1

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - **Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **macOS/Linux:**
     ```bash
     source venv/bin/activate
     ```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
pnpm install
# or
npm install
```

## Running the Application

### Start the Backend Server

From the `backend` directory:

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### Start the Frontend Development Server

From the `frontend` directory:

```bash
pnpm dev
# or
npm run dev
```

The application will be available at `http://localhost:3000`

### Access the Application

- **Frontend**: http://localhost:3000
- **Ovarian Cancer Prediction**: http://localhost:3000/ovarian
- **PCOS Prediction**: http://localhost:3000/pcos
- **Hepatitis B Prediction**: http://localhost:3000/hepatitis
- **API Documentation**: http://localhost:8000/docs (FastAPI Swagger UI)

## API Endpoints

### Prediction Endpoints

#### POST /predict (Ovarian Cancer)

Submit biomarker data for ovarian cancer prediction. This is an alias for `/predict/ovarian`.

**Request Body:**
```json
{
  "age": 45.0,
  "alb": 4.2,
  "alp": 85.0,
  "bun": 15.0,
  "ca125": 35.0,
  "eo_abs": 0.2,
  "ggt": 25.0,
  "he4": 50.0,
  "mch": 28.0,
  "mono_abs": 0.5,
  "na": 140.0,
  "pdw": 12.0
}
```

**Response:**
```json
{
  "prediction": 1,
  "confidence": 0.85
}
```

- `prediction`: 0 = No Ovarian Cancer, 1 = Possible Ovarian Cancer
- `confidence`: Confidence score between 0 and 1

#### POST /predict/ovarian

Same as `/predict` - ovarian cancer prediction endpoint.

#### POST /predict/pcos

Submit clinical data for PCOS prediction.

**Request Body:** (20 features)
```json
{
  "Marraige Status (Yrs)": 5.0,
  "Cycle(R/I)": 1.0,
  "Pulse rate(bpm)": 72.0,
  "FSH(mIU/mL)": 6.5,
  "Age (yrs)": 28.0,
  "Follicle No. (L)": 12.0,
  "BMI": 25.5,
  "Skin darkening (Y/N)": 1.0,
  "II beta-HCG(mIU/mL)": 2.5,
  "BP _Diastolic (mmHg)": 80.0,
  "hair growth(Y/N)": 1.0,
  "Avg. F size (L) (mm)": 8.5,
  "Avg. F size (R) (mm)": 8.2,
  "Waist:Hip Ratio": 0.85,
  "Weight (Kg)": 65.0,
  "Weight gain(Y/N)": 1.0,
  "LH(mIU/mL)": 12.5,
  "Follicle No. (R)": 10.0,
  "Hip(inch)": 38.0,
  "Waist(inch)": 32.0
}
```

**Response:**
```json
{
  "prediction": 1,
  "confidence": 0.78
}
```

#### POST /predict/hepatitis_b

Submit clinical and laboratory data for Hepatitis B prediction.

**Request Body:** (15 features)
```json
{
  "Age": 45.0,
  "Sex": 1.0,
  "Fatigue": 1.0,
  "Malaise": 0.0,
  "Liver_big": 1.0,
  "Spleen_palpable": 0.0,
  "Spiders": 0.0,
  "Ascites": 0.0,
  "Varices": 0.0,
  "Bilirubin": 1.2,
  "Alk_phosphate": 85.0,
  "Sgot": 45.0,
  "Albumin": 4.0,
  "Protime": 12.0,
  "Histology": 1.0
}
```

**Response:**
```json
{
  "prediction": 1,
  "confidence": 0.82
}
```

### Utility Endpoints

#### GET /health

Health check endpoint to verify the API is running and models are loaded.

**Response:**
```json
{
  "status": "ok"
}
```

#### GET /model-info

Get information about a specific model or all models.

**Query Parameters:**
- `model` (optional): Model key (`ovarian`, `pcos`, or `hepatitis_b`). If omitted, returns info for ovarian model.

**Response:**
```json
{
  "model_key": "ovarian",
  "model_path": "/path/to/model.pkl",
  "feature_count": 12,
  "loaded": true
}
```

#### GET /test-case/negative

Generate a negative (healthy/normal) test case for ovarian cancer model. Useful for testing and form auto-fill.

**Response:**
```json
{
  "age": 35.0,
  "alb": 4.5,
  ...
}
```

#### GET /test-case/positive

Generate a positive (disease-indicating) test case for ovarian cancer model. Useful for testing and form auto-fill.

**Response:**
```json
{
  "age": 55.0,
  "ca125": 150.0,
  ...
}
```

## Usage

### Ovarian Cancer Prediction

1. Navigate to the **Ovarian Cancer Prediction** page (`/ovarian`)
2. Fill in all 12 biomarker fields:
   - **Age**: Patient age
   - **ALB**: Albumin level
   - **ALP**: Alkaline phosphatase
   - **BUN**: Blood urea nitrogen
   - **CA125**: Cancer antigen 125
   - **EO#**: Absolute eosinophil count
   - **GGT**: Gamma-glutamyl transferase
   - **HE4**: Human epididymis protein 4
   - **MCH**: Mean corpuscular hemoglobin
   - **MONO#**: Absolute monocyte count
   - **Na**: Sodium level
   - **PDW**: Platelet distribution width
3. Optionally use **Fill Test Case** buttons to auto-populate with positive or negative test cases
4. Click **Submit** to get the prediction
5. Review the results, which include:
   - Prediction status (Possible Ovarian Cancer / No Ovarian Cancer)
   - Confidence level with visual indicator
   - Risk assessment badge

### PCOS Prediction

1. Navigate to the **PCOS Prediction** page (`/pcos`)
2. Fill in all 20 clinical features including:
   - Demographics: Age, Marriage Status, BMI, Weight
   - Hormonal levels: FSH, LH, II beta-HCG
   - Physical measurements: Waist, Hip, Waist:Hip Ratio
   - Clinical indicators: Cycle regularity, Pulse rate, Blood pressure
   - Symptoms: Skin darkening, Hair growth, Weight gain
   - Ultrasound data: Follicle counts and sizes (left and right)
3. Optionally use **Fill Test Case** buttons to auto-populate with positive or negative test cases
4. Click **Submit** to get the prediction
5. Review the results with confidence scores

### Hepatitis B Prediction

1. Navigate to the **Hepatitis B Prediction** page (`/hepatitis`)
2. Fill in all 15 clinical and laboratory features including:
   - Demographics: Age, Sex
   - Symptoms: Fatigue, Malaise
   - Physical examination: Liver size, Spleen palpability, Spiders, Ascites, Varices
   - Laboratory tests: Bilirubin, Alkaline phosphatase, SGOT, Albumin, Prothrombin time
   - Histology: Histology indicator
3. Optionally use **Fill Test Case** buttons to auto-populate with positive or negative test cases
4. Click **Submit** to get the prediction
5. Review the results with confidence scores

## Development

### Backend Development

The backend uses FastAPI with automatic API documentation. After starting the server, visit `http://localhost:8000/docs` to explore the interactive API documentation.

### Frontend Development

The frontend uses Next.js with TypeScript. The prediction components are modularized:
- **Shared components**: `src/components/prediction-components/` - Reusable prediction UI components
- **PCOS components**: `src/components/pcos-components/` - PCOS-specific form and UI
- **Hepatitis components**: `src/components/hepatitis-components/` - Hepatitis-specific form and UI

Each prediction type has its own page route and dedicated components for optimal user experience.

### Model Retraining

To retrain models with new data:

```bash
cd backend
python retrain_model.py
```

This will generate new model files (`model.pkl`, `pcos.pkl`, `Hepatitis_B.pkl`) in `backend/app/model/` based on the retraining script configuration.

### Model Registry

The application uses a centralized model registry system (`backend/app/model/registry.py`) that manages:
- Model file paths
- Feature order for each model
- Model configuration

To add a new model, update the `MODEL_REGISTRY` dictionary in `registry.py` and create the corresponding request schema in `schemas/input_schema.py`.

## Important Notes

⚠️ **Medical Disclaimer**: This tool is designed for research and educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

## Environment Variables

### Backend

No environment variables are required for basic operation. CORS origins are configured in `backend/app/utils/config.py`.

### Frontend

Create a `.env.local` file in the `frontend` directory if you need to configure the backend URL:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Troubleshooting

### Backend Issues

- **Model not found**: Ensure model files exist in `backend/app/model/`:
  - `model.pkl` (Ovarian Cancer)
  - `pcos.pkl` (PCOS)
  - `Hepatitis_B.pkl` (Hepatitis B)
- **Port already in use**: Change the port in the uvicorn command: `--port 8001`
- **Model loading errors**: Check that model files are compatible with the installed scikit-learn version

### Frontend Issues

- **API connection errors**: Verify the backend is running and CORS is properly configured
- **Build errors**: Clear `.next` directory and reinstall dependencies

## Supported Models

### Ovarian Cancer Model
- **Features**: 12 biomarkers
- **Model File**: `model.pkl`
- **Endpoint**: `/predict` or `/predict/ovarian`

### PCOS Model
- **Features**: 20 clinical features
- **Model File**: `pcos.pkl`
- **Endpoint**: `/predict/pcos`

### Hepatitis B Model
- **Features**: 15 clinical and laboratory features
- **Model File**: `Hepatitis_B.pkl`
- **Endpoint**: `/predict/hepatitis_b`

## License

This project is for educational and research purposes.

## Contributing

Contributions are welcome! Please ensure your code follows the existing style and includes appropriate tests. When adding new models:
1. Add the model file to `backend/app/model/`
2. Create a Pydantic schema in `backend/app/schemas/input_schema.py`
3. Register the model in `backend/app/model/registry.py`
4. Create frontend components and pages following the existing pattern
5. Add test case generators if applicable

---

**Built with ❤️ for healthcare research**
