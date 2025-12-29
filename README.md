# OvaInsightXAI - Multi-Disease Clinical AI Prediction System

A comprehensive full-stack web application for predicting multiple medical conditions using machine learning. This system provides early risk assessment for **Ovarian Cancer**, **PCOS (Polycystic Ovary Syndrome)**, and **Hepatitis B**, helping healthcare professionals make informed decisions with explainable AI insights.

## Overview

This application combines a FastAPI backend with a Next.js frontend to deliver an intuitive interface for multiple disease predictions. The system uses trained machine learning models to analyze patient data and provide predictions with confidence scores for three different medical conditions, enhanced with comprehensive explainable AI (XAI) visualizations.

## Features

### Core Prediction Features
- **Multi-Disease Support**: Predictions for Ovarian Cancer, PCOS, Hepatitis B, and Brain Tumor
- **Brain Tumor Analysis**: Image-based analysis of MRI scans using deep learning (PyTorch/PVTv2)
- **Ovarian Cancer Analysis**: Input 12 critical biomarkers including CA125, HE4, and other clinical indicators
- **PCOS Analysis**: Input 20 clinical features including hormonal levels, BMI, and follicle measurements
- **Hepatitis B Analysis**: Input 15 clinical and laboratory features including liver function tests
- **Real-time Prediction**: Get instant predictions with confidence scores
- **Test Case Generation**: Built-in test case generators for tabular models

### Explainable AI (XAI) Features
- **SHAP (SHapley Additive exPlanations)**: Feature importance analysis showing how each feature contributes to the prediction
- **LIME (Local Interpretable Model-agnostic Explanations)**: Local explanations for individual predictions
- **PDP (Partial Dependence Plots)**: Visualize the marginal effect of features on predictions
- **ICE (Individual Conditional Expectation)**: Show how predictions change as features vary
- **ALE (Accumulated Local Effects)**: Alternative to PDP for handling correlated features
- **Interactive Visualizations**: Dynamic charts and graphs for all XAI methods using ECharts

### User Experience Features
- **User-friendly Interface**: Clean, responsive design that works on all devices
- **Visual Results**: Color-coded results with detailed confidence metrics
- **Dark Mode Support**: Full dark/light theme toggle
- **FAQ Section**: Comprehensive answers to common questions
- **Team Section**: Display of research team members
- **Modern Navigation**: Responsive navbar with dropdown menus

### Technical Features
- **Model Registry**: Centralized model management system supporting multiple ML models
- **Data Validation**: Pydantic schemas for robust input validation
- **API Documentation**: Interactive Swagger UI for all endpoints
- **Health Checks**: Built-in health monitoring endpoints
- **CORS Support**: Configurable cross-origin resource sharing
- **Medical Disclaimer**: Built-in disclaimers emphasizing the tool is for research purposes

## Tech Stack

### Backend
- **FastAPI** - Modern, fast web framework for building APIs
- **Python 3.10+** - Core programming language
- **PyTorch / TIMM** - Deep learning for brain tumor image analysis
- **scikit-learn** - Machine learning models for tabular data
- **SHAP / LIME** - Explainable AI (XAI) for all models
- **Pydantic** - Data validation
- **Uvicorn** - ASGI server
- **NumPy/Pandas / PIL** - Data and image processing

### Frontend
- **Next.js 15** - React framework with App Router
- **TypeScript** - Type-safe development
- **Tailwind CSS** - Utility-first CSS framework
- **shadcn/ui** - High-quality component library
- **ECharts** - Interactive data visualization
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
│   │   │   ├── model_PTH.pth    # Trained Brain Tumor model (PyTorch)
│   │   │   ├── predictor.py     # Model loading and inference
│   │   │   └── registry.py      # Model registry and configuration
│   │   ├── services/
│   │   │   ├── preprocessing.py # Data preprocessing utilities
│   │   │   └── xai/             # Explainable AI services
│   │   │       ├── shap_explainer.py
│   │   │       ├── lime_explainer.py
│   │   │       ├── pdp_explainer.py
│   │   │       ├── ice_explainer.py
│   │   │       ├── ale_explainer.py
│   │   │       └── utils.py
│   │   └── utils/
│   │       └── config.py        # Configuration constants
│   ├── Dockerfile               # Docker configuration
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
│   │   │   ├── hepatitis-components/    # Hepatitis-specific components
│   │   │   ├── xai/                    # XAI visualization components
│   │   │   └── layout/                 # Layout components (navbar, etc.)
│   │   ├── lib/
│   │   │   ├── test-case-generator.ts      # Ovarian test case generator
│   │   │   ├── pcos-test-case-generator.ts # PCOS test case generator
│   │   │   └── hepatitis-test-case-generator.ts # Hepatitis test case generator
│   │   └── styles/              # Global styles
│   ├── package.json
│   └── next.config.ts
│
├── DOKPLOY_SETUP.md            # Dokploy deployment guide
├── HostGuide.md                # Comprehensive hosting guide
└── README.md
```

## Prerequisites

Before you begin, ensure you have the following installed:

- **Python 3.10+** (for backend)
- **Node.js 18+** (for frontend)
- **pnpm** or **npm** (package manager)

## Installation

> **Note**: For local development use `scikit-learn>=1.6.0`. For server deployment use `scikit-learn==1.5.1`.

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
  "confidence": 0.85,
  "xai": {
    "shap": {...},
    "lime": {...},
    "pdp_1d": {...},
    "ice_1d": {...},
    "ale_1d": {...}
  }
}
```

- `prediction`: 0 = No Ovarian Cancer, 1 = Possible Ovarian Cancer
- `confidence`: Confidence score between 0 and 1
- `xai`: Explainable AI explanations (optional, can be disabled)

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
  "confidence": 0.78,
  "xai": {...}
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

#### POST /predict/brain_tumor

Submit an MRI image for brain tumor classification and XAI analysis.

**Form Data:**
- `file`: MRI image file (JPEG/PNG)
- `include_xai`: (optional, default: true)

**Response:**
```json
{
  "prediction": "Glioma",
  "confidence": 0.94,
  "xai": {
    "shap": {"heatmap_image": "base64...", ...},
    "lime": {"visualization_image": "base64...", ...},
    "pdp_1d": {...},
    "ice_1d": {...},
    "ale_1d": {...}
  }
}
```

- `prediction`: Tumor class (Glioma, Meningioma, No Tumor, Pituitary)
- `confidence`: Confidence score
- `xai`: Comprehensive image-based XAI visualizations

**Query Parameters:**
- `include_xai` (optional, default: `true`): Set to `false` to disable XAI explanations for faster responses

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
   - **XAI Visualizations**: Interactive charts showing SHAP, LIME, PDP, ICE, and ALE explanations

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
5. Review the results with confidence scores and XAI visualizations

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
5. Review the results with confidence scores and XAI visualizations

## Development

### Backend Development

The backend uses FastAPI with automatic API documentation. After starting the server, visit `http://localhost:8000/docs` to explore the interactive API documentation.

### Frontend Development

The frontend uses Next.js with TypeScript. The prediction components are modularized:
- **Shared components**: `src/components/prediction-components/` - Reusable prediction UI components
- **PCOS components**: `src/components/pcos-components/` - PCOS-specific form and UI
- **Hepatitis components**: `src/components/hepatitis-components/` - Hepatitis-specific form and UI
- **XAI components**: `src/components/xai/` - Explainable AI visualization components

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

## Deployment

### Quick Deployment Guide

This project consists of two main components:
1. **Frontend**: Next.js 15 application (deploy to Vercel, Netlify, or Railway)
2. **Backend**: FastAPI application (deploy to Dokploy, Railway, Render, Fly.io, or DigitalOcean)

### Frontend Deployment

#### Vercel (Recommended)

1. Go to [vercel.com](https://vercel.com) and sign in
2. Click **"Add New Project"** and import your Git repository
3. Configure:
   - **Root Directory**: `frontend`
   - **Framework Preset**: Next.js (auto-detected)
   - **Build Command**: `pnpm build` (or `npm run build`)
4. Add environment variables:
   ```env
   NEXT_PUBLIC_API_URL=https://your-backend-url.com
   BACKEND_URL=https://your-backend-url.com
   ```
5. Click **"Deploy"**

#### Netlify

1. Go to [netlify.com](https://netlify.com) and sign in
2. Click **"Add new site"** → **"Import an existing project"**
3. Configure:
   - **Base directory**: `frontend`
   - **Build command**: `pnpm build` (or `npm run build`)
   - **Publish directory**: `frontend/.next`
4. Add environment variables (same as Vercel)

### Backend Deployment

#### Dokploy (Recommended for Self-Hosted)

⚠️ **CRITICAL: Use Dockerfile Build Method**

**You MUST select "Dockerfile" as the build method in Dokploy dashboard to use Docker instead of Nixpacks.**

**Steps:**

1. **Create New Application** in Dokploy
2. **Connect your Git repository**
3. **Set Root Directory**: `backend` (IMPORTANT: Must be exactly `backend`, not `backend/backend`)
4. **Select Build Method**: Choose **"Dockerfile"** (NOT "Nixpacks" or "Auto-detect")
5. **Build Command**: Leave empty (Dockerfile handles the build)
6. **Start Command**: Leave empty (Dockerfile CMD handles startup)
7. **Dockerfile Path**: Should auto-detect as `Dockerfile` (since Root Directory is `backend`)

**Environment Variables:**
```env
PORT=8000
PYTHON_VERSION=3.10
```

**Why Dockerfile is Required:**
- Nixpacks has issues with Python virtual environments and start commands
- Dockerfile provides consistent, reliable builds
- Better control over the build process and dependencies
- Supports volume mounting for the model file

#### Railway

1. Go to [railway.app](https://railway.app) and sign in
2. Click **"New Project"** → **"Deploy from GitHub repo"**
3. Set **Root Directory** to `backend`
4. Configure **Start Command**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```
5. Add environment variables:
   ```env
   PORT=8000
   PYTHON_VERSION=3.11
   ```

#### Render

1. Go to [render.com](https://render.com) and sign in
2. Click **"New +"** → **"Web Service"**
3. Configure:
   - **Root Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
4. Add environment variables (same as Railway)

#### Fly.io

1. Install Fly CLI and login:
   ```bash
   fly auth login
   cd backend
   fly launch
   ```
2. Configure `fly.toml` (see HostGuide.md for details)
3. Deploy:
   ```bash
   fly deploy
   ```

### Model File Deployment

Since model files (130+ MB) exceed GitHub's 100 MB file limit, they're excluded from the repository. You need to deploy model files separately:

**Option 1: Manual Upload (Recommended for Quick Setup)**
- After deployment, upload model files via platform's file system access
- Upload to `backend/app/model/` directory:
  - `model.pkl` (Ovarian Cancer)
  - `pcos.pkl` (PCOS)
  - `Hepatitis_B.pkl` (Hepatitis B)

**Option 2: Cloud Storage Download**
- Store models in AWS S3, Google Cloud Storage, etc.
- Download during build process using environment variable `MODEL_URL`
- See HostGuide.md for detailed implementation

**Option 3: Git LFS**
- Use Git Large File Storage to track model files
- Install Git LFS and track `app/model/*.pkl` files

See `HostGuide.md` for comprehensive deployment options and detailed instructions.

### Environment Variables

#### Frontend

Create `frontend/.env.production`:

```env
# Backend API URL
NEXT_PUBLIC_API_URL=https://your-backend-api.com
BACKEND_URL=https://your-backend-api.com

# App URL (for auth callbacks, etc.)
NEXT_PUBLIC_APP_URL=https://your-frontend-domain.com
```

#### Backend

Create `backend/.env` or set in platform dashboard:

```env
# Server Configuration
PORT=8000
HOST=0.0.0.0

# CORS Origins (comma-separated)
ALLOWED_ORIGINS=https://your-frontend-domain.com,https://www.your-frontend-domain.com

# XAI Performance Optimization (Critical for Brain Tumor)
XAI_ENABLED=true                # Master switch for XAI
XAI_PARALLEL=false              # Set false to prevent CPU spikes (sequential)
XAI_ESSENTIAL_ONLY=false        # Set true to compute only SHAP/LIME for speed

# Brain Tumor Specific XAI Tuning
SHAP_IMAGE_PATCH_SIZE=48        # Larger = faster computation
SHAP_IMAGE_STRIDE=24            # Larger = fewer grid points
LIME_IMAGE_NUM_SAMPLES=50       # Lower = faster computation
```

### CORS Configuration

Update `backend/app/utils/config.py` to read from environment:

```python
import os

# Read from environment or use default
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000"
).split(",")
```

For production, ensure your frontend domain is included in `ALLOWED_ORIGINS`.

## Troubleshooting

### Backend Issues

- **Model not found**: Ensure model files exist in `backend/app/model/`:
  - `model.pkl` (Ovarian Cancer)
  - `pcos.pkl` (PCOS)
  - `Hepatitis_B.pkl` (Hepatitis B)
  - Model files are NOT in Git repository (exceed 100 MB limit)
  - You must deploy the model files separately (see Model File Deployment section)

- **Port already in use**: Use `$PORT` environment variable (platforms set this automatically)

- **Model loading errors**: Check that model files are compatible with the installed scikit-learn version
  - Local: `scikit-learn>=1.6.0`
  - Server: `scikit-learn==1.5.1`

- **CORS errors**: Update `ALLOWED_ORIGINS` in `backend/app/utils/config.py` to include your frontend domain

- **XAI computation timeouts**: Adjust `XAI_TIMEOUT_SECONDS` environment variable or disable XAI with `include_xai=false` query parameter

### Frontend Issues

- **API connection errors**: Verify the backend is running and CORS is properly configured
- **Build errors**: Clear `.next` directory and reinstall dependencies
- **Environment variables not working**: Use `NEXT_PUBLIC_` prefix for client-side variables

### Deployment Issues

- **Dokploy Nixpacks errors**: Select **"Dockerfile"** as build method in Dokploy dashboard
- **Root directory issues**: Ensure Root Directory is set to `backend` (not `backend/backend`)
- **Model file missing**: Upload model files after deployment (see Model File Deployment section)

## Supported Models

### Ovarian Cancer Model
- **Features**: 12 biomarkers
- **Model File**: `model.pkl`
- **Endpoint**: `/predict` or `/predict/ovarian`
- **XAI Support**: Full support for all XAI methods

### PCOS Model
- **Features**: 20 clinical features
- **Model File**: `pcos.pkl`
- **Endpoint**: `/predict/pcos`
- **XAI Support**: Full support for all XAI methods

### Hepatitis B Model
- **Features**: 15 clinical and laboratory features
- **Model File**: `Hepatitis_B.pkl`
- **Endpoint**: `/predict/hepatitis_b`
- **XAI Support**: Full support for all XAI methods

### Brain Tumor Model
- **Type**: Deep Learning (Image-based)
- **Architecture**: PVTv2-B1 with custom classifier
- **Model File**: `model_PTH.pth`
- **Endpoint**: `/predict/brain_tumor`
- **XAI Support**: Full image-based support (Patch-based PDP/ICE/ALE/SHAP/LIME)

## Important Notes

⚠️ **Medical Disclaimer**: This tool is designed for research and educational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals for medical decisions.

## Security Considerations

1. **Environment Variables**: Never commit `.env` files to Git
2. **CORS**: Only allow trusted origins in production
3. **HTTPS**: Always use HTTPS in production
4. **Input Validation**: Backend validates inputs via Pydantic schemas
5. **Data Privacy**: Input data is processed securely and not stored permanently

## License

This project is for educational and research purposes.

## Contributing

Contributions are welcome! Please ensure your code follows the existing style and includes appropriate tests. When adding new models:

1. Add the model file to `backend/app/model/`
2. Create a Pydantic schema in `backend/app/schemas/input_schema.py`
3. Register the model in `backend/app/model/registry.py`
4. Create frontend components and pages following the existing pattern
5. Add test case generators if applicable
6. Update XAI services if needed

## Additional Resources

- **Dokploy Setup Guide**: See `DOKPLOY_SETUP.md` for detailed Dokploy deployment instructions
- **Hosting Guide**: See `HostGuide.md` for comprehensive deployment options and troubleshooting

---

**Built with ❤️ for healthcare research**
