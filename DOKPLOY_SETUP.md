# Dokploy Deployment Configuration

## Important: Root Directory Configuration

When deploying to Dokploy, you **MUST** set the **Root Directory** to `backend` in the Dokploy dashboard.

### Steps to Configure Dokploy:

1. **Create New Application** in Dokploy
2. **Connect your Git repository**
3. **Set Root Directory**: `backend`
4. **Build Command**: Leave empty (or use `pip install -r requirements.txt`)
5. **Start Command**: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

### Why This is Important:

- Dokploy/Nixpacks will auto-detect the frontend directory if Root Directory is not set
- The frontend requires `pnpm-lock.yaml` which may not be present
- Setting Root Directory to `backend` ensures only the Python backend is built

### Alternative: Use Dockerfile

If you prefer to use Docker instead of Nixpacks:

1. In Dokploy, select **"Dockerfile"** as the build method
2. Set **Root Directory** to `backend`
3. Dokploy will use `backend/Dockerfile` automatically

### Environment Variables:

Add these in Dokploy's Environment Variables section:

```env
PORT=8000
PYTHON_VERSION=3.10
```

### Troubleshooting:

If you see errors about `pnpm-lock.yaml`:
- ✅ **Solution**: Set Root Directory to `backend` in Dokploy settings
- ✅ **Alternative**: Use the Dockerfile build method instead of Nixpacks

