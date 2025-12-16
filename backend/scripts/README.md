# Deployment Scripts

Scripts to help with backend deployment, particularly for handling the model file.

## download_model.sh / download_model.py

Downloads the model file from a URL during deployment.

### Usage

Set the `MODEL_URL` environment variable to point to your hosted model file:

```bash
export MODEL_URL=https://your-storage-url.com/models/model.pkl
./scripts/download_model.sh
```

Or with Python:

```bash
export MODEL_URL=https://your-storage-url.com/models/model.pkl
python scripts/download_model.py
```

### Integration with Deployment Platforms

#### Railway / Render
Add to your **Build Command**:
```bash
chmod +x scripts/download_model.sh && scripts/download_model.sh && pip install -r requirements.txt
```

#### Docker
Add to your `Dockerfile`:
```dockerfile
# Download model before copying app code
RUN chmod +x scripts/download_model.sh
RUN scripts/download_model.sh

# Or use Python version
RUN python scripts/download_model.py
```

### Hosting Your Model File

You can host your `model.pkl` file on:
- **AWS S3** (with public or signed URL)
- **Google Cloud Storage**
- **GitHub Releases** (as an asset)
- **Dropbox** (with direct download link)
- **Any web server** with direct file access

### Security Note

For production, consider:
- Using signed URLs with expiration
- Restricting access by IP
- Using environment variables for credentials
- Not committing `MODEL_URL` to version control

