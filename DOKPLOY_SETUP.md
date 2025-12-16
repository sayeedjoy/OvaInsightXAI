# Dokploy Deployment Configuration

## ⚠️ CRITICAL: Use Dockerfile Build Method

**You MUST select "Dockerfile" as the build method in Dokploy dashboard to use Docker instead of Nixpacks.**

### Steps to Configure Dokploy (Dockerfile Method):

1. **Create New Application** in Dokploy
2. **Connect your Git repository**
3. **Set Root Directory**: `backend` (IMPORTANT: Must be exactly `backend`, not `backend/backend`)
4. **Select Build Method**: Choose **"Dockerfile"** (NOT "Nixpacks" or "Auto-detect")
5. **Build Command**: Leave empty (Dockerfile handles the build)
6. **Start Command**: Leave empty (Dockerfile CMD handles startup)
7. **Dockerfile Path**: Should auto-detect as `Dockerfile` (since Root Directory is `backend`)

### Why Dockerfile is Required:

- Nixpacks has issues with Python virtual environments and start commands
- Dockerfile provides consistent, reliable builds
- Better control over the build process and dependencies
- Supports volume mounting for the model file

### If You See Nixpacks Errors:

- ✅ **Solution 1**: In Dokploy dashboard, explicitly select **"Dockerfile"** as build method
- ✅ **Solution 2**: Ensure Root Directory is set to `backend` (not `backend/backend`)
- ✅ **Solution 3**: Check that `backend/Dockerfile` exists in your repository

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

