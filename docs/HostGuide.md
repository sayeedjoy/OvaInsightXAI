# Deployment & Hosting Guide

Complete guide for deploying the Ovarian Cancer Prediction System to various hosting platforms.

## Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Frontend Deployment](#frontend-deployment)
  - [Vercel (Recommended)](#vercel-recommended)
  - [Netlify](#netlify)
  - [Railway](#railway-frontend)
- [Backend Deployment](#backend-deployment)
  - [Dokploy](#dokploy)
  - [Railway](#railway-backend)
  - [Render](#render)
  - [Fly.io](#flyio)
  - [DigitalOcean App Platform](#digitalocean-app-platform)
- [Docker Deployment](#docker-deployment)
- [Environment Variables](#environment-variables)
- [CORS Configuration](#cors-configuration)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project consists of two main components:

1. **Frontend**: Next.js 15 application (deploy to Vercel, Netlify, or Railway)
2. **Backend**: FastAPI application (deploy to Dokploy, Railway, Render, Fly.io, or DigitalOcean)

The frontend communicates with the backend via API calls. Ensure both services are properly configured with correct URLs and CORS settings.

---

## Prerequisites

Before deploying, ensure you have:

- ✅ Git repository with your code
- ✅ GitHub/GitLab/Bitbucket account (for most platforms)
- ✅ Domain name (optional, but recommended)
- ✅ Environment variables ready (see [Environment Variables](#environment-variables))

---

## Frontend Deployment

### Vercel (Recommended)

Vercel is the recommended platform for Next.js applications due to seamless integration and optimal performance.

#### Step 1: Prepare Your Repository

1. Push your code to GitHub/GitLab/Bitbucket
2. Ensure `frontend/` directory contains all necessary files

#### Step 2: Deploy to Vercel

1. Go to [vercel.com](https://vercel.com) and sign in
2. Click **"Add New Project"**
3. Import your Git repository
4. Configure the project:
   - **Root Directory**: Set to `frontend`
   - **Framework Preset**: Next.js (auto-detected)
   - **Build Command**: `pnpm build` (or `npm run build`)
   - **Output Directory**: `.next` (auto-detected)
   - **Install Command**: `pnpm install` (or `npm install`)

#### Step 3: Environment Variables

Add the following environment variables in Vercel dashboard:

```env
NEXT_PUBLIC_API_URL=https://your-backend-url.com
BACKEND_URL=https://your-backend-url.com
```

#### Step 4: Deploy

1. Click **"Deploy"**
2. Wait for the build to complete
3. Your app will be live at `https://your-project.vercel.app`

#### Custom Domain (Optional)

1. Go to **Settings** → **Domains**
2. Add your custom domain
3. Follow DNS configuration instructions

---

### Netlify

#### Step 1: Deploy via Netlify Dashboard

1. Go to [netlify.com](https://netlify.com) and sign in
2. Click **"Add new site"** → **"Import an existing project"**
3. Connect your Git repository
4. Configure build settings:
   - **Base directory**: `frontend`
   - **Build command**: `pnpm build` (or `npm run build`)
   - **Publish directory**: `frontend/.next`

#### Step 2: Environment Variables

Add environment variables in **Site settings** → **Environment variables**:

```env
NEXT_PUBLIC_API_URL=https://your-backend-url.com
BACKEND_URL=https://your-backend-url.com
```

#### Step 3: Deploy

Netlify will automatically deploy on every push to your main branch.

---

### Railway (Frontend)

#### Step 1: Create Railway Project

1. Go to [railway.app](https://railway.app) and sign in
2. Click **"New Project"** → **"Deploy from GitHub repo"**
3. Select your repository

#### Step 2: Configure Service

1. Click on the service → **Settings**
2. Set **Root Directory** to `frontend`
3. Configure build:
   - **Build Command**: `pnpm build`
   - **Start Command**: `pnpm start`

#### Step 3: Environment Variables

Add in **Variables** tab:

```env
NEXT_PUBLIC_API_URL=https://your-backend-url.com
BACKEND_URL=https://your-backend-url.com
NODE_ENV=production
```

#### Step 4: Deploy

Railway will automatically build and deploy your application.

---

## Backend Deployment

### Dokploy

Dokploy is a self-hosted deployment platform similar to Vercel but for any application type.

#### Step 1: Install Dokploy

1. Follow [Dokploy installation guide](https://dokploy.com/docs/installation)
2. Access your Dokploy dashboard

#### Step 2: Create New Application

1. Click **"New Application"**
2. Select **"Git Repository"**
3. Connect your repository

#### Step 3: Configure Build

1. Set **Root Directory** to `backend`
2. **Build Command**: (leave empty or use `pip install -r requirements.txt`)
3. **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

#### Step 4: Environment Variables

Add in **Environment Variables** section:

```env
PORT=8000
PYTHON_VERSION=3.11
```

#### Step 5: Deploy

1. Click **"Deploy"**
2. Dokploy will build and deploy your FastAPI application
3. Access your API at the provided URL

---

### Railway (Backend)

#### Step 1: Create New Service

1. In your Railway project, click **"New"** → **"Service"**
2. Select **"GitHub Repo"** → Choose your repository

#### Step 2: Configure Service

1. Set **Root Directory** to `backend`
2. Railway will auto-detect Python
3. Configure **Start Command**:
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```

#### Step 3: Environment Variables

Add in **Variables** tab:

```env
PORT=8000
PYTHON_VERSION=3.11
```

#### Step 4: Deploy

Railway will automatically:
- Install dependencies from `requirements.txt`
- Build your application
- Deploy and provide a public URL

---

### Render

#### Step 1: Create Web Service

1. Go to [render.com](https://render.com) and sign in
2. Click **"New +"** → **"Web Service"**
3. Connect your Git repository

#### Step 2: Configure Service

- **Name**: `ovarian-cancer-api` (or your preferred name)
- **Environment**: Python 3
- **Root Directory**: `backend`
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

#### Step 3: Environment Variables

Add in **Environment** section:

```env
PORT=8000
PYTHON_VERSION=3.11
```

#### Step 4: Deploy

1. Click **"Create Web Service"**
2. Render will build and deploy your application
3. Your API will be available at `https://your-service.onrender.com`

**Note**: Free tier services on Render spin down after inactivity. Consider upgrading for production use.

---

### Fly.io

#### Step 1: Install Fly CLI

```bash
# macOS/Linux
curl -L https://fly.io/install.sh | sh

# Windows (PowerShell)
iwr https://fly.io/install.ps1 -useb | iex
```

#### Step 2: Login and Initialize

```bash
fly auth login
cd backend
fly launch
```

#### Step 3: Configure `fly.toml`

Create or update `backend/fly.toml`:

```toml
app = "your-app-name"
primary_region = "iad"

[build]

[env]
  PORT = "8000"

[[services]]
  internal_port = 8000
  protocol = "tcp"

  [[services.ports]]
    port = 80
    handlers = ["http"]
    force_https = true

  [[services.ports]]
    port = 443
    handlers = ["tls", "http"]

  [services.concurrency]
    type = "connections"
    hard_limit = 25
    soft_limit = 20

  [[services.http_checks]]
    interval = "10s"
    timeout = "2s"
    grace_period = "5s"
    method = "GET"
    path = "/health"
```

#### Step 4: Deploy

```bash
fly deploy
```

Your API will be available at `https://your-app-name.fly.dev`

---

### DigitalOcean App Platform

#### Step 1: Create App

1. Go to [DigitalOcean App Platform](https://cloud.digitalocean.com/apps)
2. Click **"Create App"**
3. Connect your Git repository

#### Step 2: Configure Component

1. Add **"Web Service"**
2. Configure:
   - **Source Directory**: `backend`
   - **Build Command**: `pip install -r requirements.txt`
   - **Run Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

#### Step 3: Environment Variables

Add in **App-Level Environment Variables**:

```env
PORT=8000
PYTHON_VERSION=3.11
```

#### Step 4: Deploy

1. Click **"Create Resources"**
2. DigitalOcean will build and deploy your application

---

## Docker Deployment

Docker provides a consistent deployment method across all platforms.

### Create Dockerfile for Backend

Create `backend/Dockerfile`:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Create .dockerignore

Create `backend/.dockerignore`:

```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
venv/
env/
.venv
.git
.gitignore
README.md
```

### Build and Run Locally

```bash
cd backend
docker build -t ovarian-cancer-api .
docker run -p 8000:8000 ovarian-cancer-api
```

### Deploy to Docker Hub / Container Registry

1. Build and tag:
   ```bash
   docker build -t yourusername/ovarian-cancer-api:latest .
   ```

2. Push to registry:
   ```bash
   docker push yourusername/ovarian-cancer-api:latest
   ```

3. Deploy on any platform that supports Docker (Railway, Render, Fly.io, etc.)

---

## Model File Deployment

Since `model.pkl` (130+ MB) exceeds GitHub's 100 MB file limit, it's excluded from the repository via `.gitignore`. You need to deploy the model file separately. Here are several approaches:

### Option 1: Manual Upload via Platform File System (Recommended for Most Platforms)

#### Railway
1. After initial deployment, use Railway's **Volume** feature:
   - Go to your service → **Settings** → **Volumes**
   - Create a volume and mount it to `/app/app/model`
   - Upload `model.pkl` to the volume via Railway CLI or dashboard

2. **Or** use Railway's file upload:
   ```bash
   railway run bash
   # Then upload model.pkl to backend/app/model/
   ```

#### Render
1. After deployment, SSH into your service:
   ```bash
   # Render provides SSH access in dashboard
   ```
2. Upload model file via SCP or use Render's file browser:
   ```bash
   scp backend/app/model/model.pkl user@your-service.onrender.com:/opt/render/project/src/app/model/
   ```

#### Fly.io
1. Use Fly's volumes or upload during deployment:
   ```bash
   fly volumes create model_data --size 1
   ```
2. Mount in `fly.toml`:
   ```toml
   [[mounts]]
     source = "model_data"
     destination = "/app/app/model"
   ```
3. Upload model file:
   ```bash
   fly ssh console
   # Then copy model.pkl to /app/app/model/
   ```

### Option 2: Download from Cloud Storage During Build

Store your model in cloud storage (AWS S3, Google Cloud Storage, etc.) and download during deployment.

#### Step 1: Upload Model to Cloud Storage

**AWS S3 Example:**
```bash
aws s3 cp backend/app/model/model.pkl s3://your-bucket/models/model.pkl
```

**Google Cloud Storage Example:**
```bash
gsutil cp backend/app/model/model.pkl gs://your-bucket/models/model.pkl
```

#### Step 2: Create Download Script

Create `backend/scripts/download_model.sh`:
```bash
#!/bin/bash
set -e

MODEL_DIR="app/model"
MODEL_URL="${MODEL_URL:-https://your-storage-url.com/models/model.pkl}"

mkdir -p "$MODEL_DIR"
curl -L "$MODEL_URL" -o "$MODEL_DIR/model.pkl"

echo "Model downloaded successfully"
```

#### Step 3: Update Build Process

**For Railway/Render:**
- Add to **Build Command**:
  ```bash
  chmod +x scripts/download_model.sh && scripts/download_model.sh && pip install -r requirements.txt
  ```

**For Docker:**
Update `backend/Dockerfile`:
```dockerfile
# Add before CMD
RUN chmod +x scripts/download_model.sh
RUN scripts/download_model.sh
```

#### Step 4: Set Environment Variable

Add to your platform's environment variables:
```env
MODEL_URL=https://your-storage-url.com/models/model.pkl
```

### Option 3: Git LFS (Git Large File Storage)

If you want to keep the model in Git:

#### Step 1: Install Git LFS
```bash
# Windows (via Chocolatey)
choco install git-lfs

# Or download from https://git-lfs.github.com/
```

#### Step 2: Initialize Git LFS
```bash
cd backend
git lfs install
git lfs track "app/model/*.pkl"
git add .gitattributes
git commit -m "Track model files with Git LFS"
```

#### Step 3: Add Model File
```bash
git add app/model/model.pkl
git commit -m "Add model file via Git LFS"
git push
```

**Note**: Git LFS has storage quotas on free GitHub accounts (1 GB storage, 1 GB bandwidth/month).

### Option 4: CI/CD Script Download

Create a deployment script that downloads the model as part of CI/CD:

Create `backend/.github/workflows/deploy.yml` (if using GitHub Actions):
```yaml
name: Deploy Backend

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Download model file
        run: |
          mkdir -p backend/app/model
          curl -L "${{ secrets.MODEL_URL }}" -o backend/app/model/model.pkl
      
      - name: Deploy to platform
        # Add your platform-specific deployment steps
```

### Option 5: Platform-Specific File Upload

Some platforms allow direct file uploads:

#### Dokploy
- Use Dokploy's file manager to upload `model.pkl` to `backend/app/model/` after deployment

#### DigitalOcean App Platform
- Use **Settings** → **App-Level Settings** → **File Uploads** to upload the model file

### Recommended Approach

**For Production:**
- **Option 2 (Cloud Storage)** - Most reliable and scalable
- Store model in S3/GCS with public or signed URL
- Download during build process
- No Git repository bloat

**For Quick Deployment:**
- **Option 1 (Manual Upload)** - Fastest for initial setup
- Deploy without model first
- Upload model file via platform's file system access
- Works well for Railway, Render, Fly.io

**For Development/Testing:**
- **Option 3 (Git LFS)** - If you need version control for model files
- Good for small teams
- Watch storage quotas

### Verification

After deployment, verify the model loads correctly:

```bash
# Check health endpoint
curl https://your-api-url.com/health

# Test prediction endpoint
curl -X POST https://your-api-url.com/predict \
  -H "Content-Type: application/json" \
  -d '{"age": 45, "alb": 4.2, "alp": 85, "bun": 15, "ca125": 35, "eo_abs": 0.2, "ggt": 25, "he4": 50, "mch": 28, "mono_abs": 0.5, "na": 140, "pdw": 12}'
```

If the model file is missing, you'll see:
```json
{
  "detail": "Model file not found at /app/app/model/model.pkl"
}
```

---

## Environment Variables

### Frontend Environment Variables

Create `frontend/.env.production`:

```env
# Backend API URL
NEXT_PUBLIC_API_URL=https://your-backend-api.com
BACKEND_URL=https://your-backend-api.com

# App URL (for auth callbacks, etc.)
NEXT_PUBLIC_APP_URL=https://your-frontend-domain.com

# Optional: Database (if using)
DATABASE_URL=postgresql://user:password@host:5432/dbname

# Optional: Auth (if using)
BETTER_AUTH_SECRET=your-secret-key
BETTER_AUTH_URL=https://your-frontend-domain.com
```

### Backend Environment Variables

Create `backend/.env` (or set in platform dashboard):

```env
# Server Configuration
PORT=8000
HOST=0.0.0.0

# CORS Origins (comma-separated)
ALLOWED_ORIGINS=https://your-frontend-domain.com,https://www.your-frontend-domain.com

# Optional: Logging
LOG_LEVEL=info
```

### Update Backend CORS Configuration

Update `backend/app/utils/config.py` to read from environment:

```python
import os

# Read from environment or use default
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000"
).split(",")
```

---

## CORS Configuration

### Update Backend CORS Settings

Ensure your backend allows requests from your frontend domain.

**Option 1: Environment Variable (Recommended)**

Update `backend/app/utils/config.py`:

```python
import os
from typing import List

def get_allowed_origins() -> List[str]:
    """Get allowed origins from environment or default to localhost."""
    origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000")
    return [origin.strip() for origin in origins_str.split(",")]

ALLOWED_ORIGINS = get_allowed_origins()
```

**Option 2: Direct Configuration**

For production, update `backend/app/utils/config.py`:

```python
ALLOWED_ORIGINS = [
    "https://your-frontend-domain.com",
    "https://www.your-frontend-domain.com",
]
```

---

## Troubleshooting

### Frontend Issues

#### Build Fails on Vercel

- **Issue**: Build command fails
- **Solution**: 
  - Ensure `package.json` has correct build script
  - Check Node.js version (should be 18+)
  - Verify all dependencies are in `package.json`

#### API Calls Fail

- **Issue**: CORS errors or connection refused
- **Solution**:
  - Verify `NEXT_PUBLIC_API_URL` is set correctly
  - Check backend CORS configuration includes frontend domain
  - Ensure backend is running and accessible

#### Environment Variables Not Working

- **Issue**: Variables not available in browser
- **Solution**:
  - Use `NEXT_PUBLIC_` prefix for client-side variables
  - Restart deployment after adding variables
  - Check variable names match exactly

### Backend Issues

#### Model File Not Found

- **Issue**: `FileNotFoundError: model.pkl`
- **Solution**:
  - **Model file is NOT in Git repository** (exceeds 100 MB limit)
  - You must deploy the model file separately (see [Model File Deployment](#model-file-deployment) section)
  - Verify path in `backend/app/utils/config.py` (should be `app/model/model.pkl`)
  - Check file exists in `backend/app/model/` directory on deployed server
  - For cloud storage approach: Verify `MODEL_URL` environment variable is set correctly
  - For manual upload: Ensure file was uploaded to correct path on server

#### Port Already in Use

- **Issue**: Port binding error
- **Solution**:
  - Use `$PORT` environment variable (platforms set this automatically)
  - Update start command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

#### Dependencies Installation Fails

- **Issue**: `pip install` fails
- **Solution**:
  - Check Python version (3.10+ required)
  - Verify `requirements.txt` is in `backend/` directory
  - Some platforms require specifying Python version explicitly

#### CORS Errors

- **Issue**: Frontend can't access backend API
- **Solution**:
  - Update `ALLOWED_ORIGINS` in `backend/app/utils/config.py`
  - Include both `https://` and `https://www.` variants
  - Restart backend after configuration changes

### General Issues

#### Slow Response Times

- **Issue**: API responses are slow
- **Solution**:
  - Check platform resource limits (upgrade if needed)
  - Verify model loading happens at startup (not per request)
  - Consider adding caching for predictions

#### Deployment Fails

- **Issue**: Build or deployment fails
- **Solution**:
  - Check platform logs for specific errors
  - Verify all required files are in repository
  - Ensure build commands are correct
  - Check platform-specific requirements

---

## Recommended Setup

### Production Setup

**Frontend**: Vercel (optimal Next.js support)
**Backend**: Railway or Render (good Python support, reasonable pricing)

### Development Setup

**Frontend**: Local development with `pnpm dev`
**Backend**: Local development with `uvicorn app.main:app --reload`

### Cost-Effective Setup

**Frontend**: Vercel (free tier available)
**Backend**: Railway (free tier) or Render (free tier with limitations)

---

## Security Considerations

1. **Environment Variables**: Never commit `.env` files to Git
2. **CORS**: Only allow trusted origins in production
3. **API Keys**: Use secure environment variable storage
4. **HTTPS**: Always use HTTPS in production
5. **Rate Limiting**: Consider adding rate limiting to API endpoints
6. **Input Validation**: Backend already validates inputs via Pydantic

---

## Monitoring & Logs

### Vercel Logs

- Access logs in Vercel dashboard → **Deployments** → **Functions** → **View Function Logs**

### Backend Logs

- **Railway**: Dashboard → **Deployments** → **View Logs**
- **Render**: Dashboard → **Logs** tab
- **Fly.io**: `fly logs` command

### Health Checks

Your backend includes a `/health` endpoint. Configure monitoring services to check:
- `GET https://your-backend-url.com/health`

---

## Next Steps

1. ✅ Deploy frontend to Vercel
2. ✅ Deploy backend to Railway/Render
3. ✅ Configure environment variables
4. ✅ Update CORS settings
5. ✅ Test end-to-end functionality
6. ✅ Set up custom domains (optional)
7. ✅ Configure monitoring and alerts

---

**Need Help?** Check platform-specific documentation or open an issue in your repository.

