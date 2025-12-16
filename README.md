# Ovarian Cancer Prediction System

A full-stack web application for predicting ovarian cancer risk using machine learning. This system analyzes 12 key biomarkers to provide early risk assessment, helping healthcare professionals make informed decisions.

## Overview

This application combines a FastAPI backend with a Next.js frontend to deliver an intuitive interface for ovarian cancer prediction. The system uses a trained machine learning model to analyze patient biomarkers and provide predictions with confidence scores.

## Features

- **Biomarker Analysis**: Input 12 critical biomarkers including CA125, HE4, and other clinical indicators
- **Real-time Prediction**: Get instant predictions with confidence scores
- **User-friendly Interface**: Clean, responsive design that works on all devices
- **Visual Results**: Color-coded results with detailed confidence metrics
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
│   │   │   └── predict.py       # Prediction endpoint
│   │   ├── schemas/
│   │   │   └── input_schema.py  # Pydantic models for validation
│   │   ├── model/
│   │   │   ├── model.pkl        # Trained ML model
│   │   │   └── predictor.py    # Model loading and inference
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
│   │   │   ├── predict/         # Prediction page
│   │   │   └── api/             # API routes
│   │   ├── components/
│   │   │   └── prediction-components/  # Prediction UI components
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
- **Prediction Page**: http://localhost:3000/predict
- **API Documentation**: http://localhost:8000/docs (FastAPI Swagger UI)

## API Endpoints

### POST /predict

Submit biomarker data for prediction.

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

### GET /health

Health check endpoint to verify the API is running.

**Response:**
```json
{
  "status": "ok"
}
```

## Usage

1. Navigate to the **Prediction Test** page from the navigation menu
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

3. Click **Submit** to get the prediction
4. Review the results, which include:
   - Prediction status (Possible Ovarian Cancer / No Ovarian Cancer)
   - Confidence level with visual indicator
   - Risk assessment badge

## Development

### Backend Development

The backend uses FastAPI with automatic API documentation. After starting the server, visit `http://localhost:8000/docs` to explore the interactive API documentation.

### Frontend Development

The frontend uses Next.js with TypeScript. The prediction components are modularized in `src/components/prediction-components/` for easy maintenance.

### Model Retraining

To retrain the model with new data:

```bash
cd backend
python retrain_model.py
```

This will generate a new `model.pkl` file in `backend/app/model/`.

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

- **Model not found**: Ensure `model.pkl` exists in `backend/app/model/`
- **Port already in use**: Change the port in the uvicorn command: `--port 8001`

### Frontend Issues

- **API connection errors**: Verify the backend is running and CORS is properly configured
- **Build errors**: Clear `.next` directory and reinstall dependencies

## License

This project is for educational and research purposes.

## Contributing

Contributions are welcome! Please ensure your code follows the existing style and includes appropriate tests.

---

**Built with ❤️ for healthcare research**
