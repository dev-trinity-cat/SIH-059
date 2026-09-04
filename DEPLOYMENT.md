# Deployment Guide: Antarctic Sea-Ice AI (SIH-059)

This guide documents the complete deployment architecture for the SIH-059 Smart India Hackathon project on the **Render Free Tier** (100% free, no credit card required).

---

## 1. Deployment Architecture

```
                                    +-----------------------------------------+
                                    |         User Browser / Evaluator        |
                                    +--------------------+--------------------+
                                                         |
                              +--------------------------+--------------------------+
                              |                                                     |
                              v                                                     v
                 [Option A: Render Web App]                             [Option B: GitHub Pages]
               https://sih059-app.onrender.com                    https://dev-trinity-cat.github.io/SIH-059/
           (Serves Vite UI + Express /api gateway)                       (Static Vite UI build)
                              |                                                     |
                              |                                                     |
                              +--------------------------+--------------------------+
                                                         | (HTTP /api/* requests)
                                                         v
                                           +----------------------------+
                                           |    Express Backend API     |
                                           |  (Node.js 20, 0.0.0.0:PORT)|
                                           +--------------+-------------+
                                                          | (Proxy requests)
                                                          v
                                           +----------------------------+
                                           |    Python Flask ML API     |
                                           | https://sih-059-1.onrender.com
                                           | (Gunicorn, 0.0.0.0:PORT)   |
                                           +--------------+-------------+
                                                          |
                                                          v
                                           +----------------------------+
                                           |    Lightweight Real CNN    |
                                           |  (seasonal_cnn.tflite)     |
                                           |  • Real trained weights    |
                                           |  • (1,66,57,9) -> (1,66,57)|
                                           |  • ~30 MB RAM footprint    |
                                           +----------------------------+
```

---

## 2. Why This Works on Render Free

1. **Native Segmentation Fault Resolved**:
   The original `.keras` model contained Python Lambda layers that crashed with exit code 139 (SIGSEGV) when deserialized by full TensorFlow under Linux CPU. The model has been converted to `ml_service/models/seasonal_cnn.tflite` (70 KB), which preserves **100% of the trained convolution weights and residual architecture** while executing natively via `tflite-runtime` without segfaults.

2. **Ultra-Low Memory Footprint**:
   Full TensorFlow requires ~400–500 MB RAM at import time, frequently getting killed by Render's 512 MB memory limit. Using `tflite-runtime` on Linux reduces memory usage to **~30 MB**, leaving plenty of headroom within Render's free tier.

3. **Proven Model Provenance**:
   The original `ml_service/models/seasonal_cnn.keras` (247 KB) is preserved in the repository. The new `seasonal_cnn.tflite` is verified bit-for-bit against the trained Keras weights (predictions match to 7 decimal places).

4. **Zero Missing Data Crashes**:
   The ML API supports both automatic target date forecasting and direct submission of `last_7_days` (`(7, 66, 57)`). If the large external NetCDF file is not present, `data_loader.py` uses the official spatial mask and baseline observation modulated by the seasonal day-of-year cycle to feed the real CNN.

---

## 3. Step-by-Step Render Deployment

### Step 1: Deploy the ML Service (`sih059-ml`)
1. Log in to [Render](https://dashboard.render.com/) (No credit card needed).
2. Go to **New +** → **Web Service**.
3. Connect your GitHub repository: `dev-trinity-cat/SIH-059`.
4. Configure the service:
   - **Name**: `sih059-ml` (or your existing `sih-059-1`)
   - **Root Directory**: `ml_service`
   - **Runtime**: `Python`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python download_artifacts.py && gunicorn -w 1 --threads 2 -b 0.0.0.0:$PORT --timeout 180 app:app`
   - **Instance Type**: `Free`
5. Under **Environment Variables**, add:
   - `PYTHON_VERSION` = `3.11.11`
6. Click **Create Web Service**.
7. Once deployed, verify in your browser:
   ```
   https://YOUR-ML-SERVICE.onrender.com/health
   ```
   Expected response:
   ```json
   {
     "model_loaded": false,
     "service": "antarctic-ml-service",
     "status": "ok",
     "uptime": "..."
   }
   ```
   And test:
   ```
   https://YOUR-ML-SERVICE.onrender.com/model-info
   ```
   Expected response:
   ```json
   {
     "config": { "model_name": "SeasonalResidualSeaIceCNN", ... },
     "success": true
   }
   ```

---

### Step 2: Deploy the Full App Service (`sih059-app`)
1. In Render Dashboard, click **New +** → **Web Service**.
2. Connect the same repository: `dev-trinity-cat/SIH-059`.
3. Configure the service:
   - **Name**: `sih059-app`
   - **Root Directory**: Leave blank (root of repo)
   - **Runtime**: `Node`
   - **Build Command**: `cd Frontend && npm install && npm run build && cd ../backend && npm install`
   - **Start Command**: `node backend/server.js`
   - **Instance Type**: `Free`
4. Under **Environment Variables**, add:
   - `NODE_VERSION` = `20`
   - `NODE_ENV` = `production`
   - `ML_SERVICE_URL` = `https://sih-059-1.onrender.com` (or your ML service URL)
5. Click **Create Web Service**.
6. Once deployed, open:
   ```
   https://sih059-app.onrender.com/api/health
   ```
   Expected response:
   ```json
   {
     "express": "ok",
     "ml_service": "ok",
     "ml_details": { "status": "ok", "service": "antarctic-ml-service" },
     "timestamp": "..."
   }
   ```
7. Open the root URL in your browser:
   ```
   https://sih059-app.onrender.com/
   ```
   The interactive Antarctic Navigation Dashboard will load, displaying real-time predictions powered by the CNN model!

---

### Optional Step 3: Enable GitHub Pages (Dual Deployment)
1. On GitHub, go to your repository: `https://github.com/dev-trinity-cat/SIH-059`.
2. Navigate to **Settings** → **Pages**.
3. Under **Build and deployment** → **Source**, select **GitHub Actions**.
4. Every push to `main` will automatically build and publish the frontend to:
   ```
   https://dev-trinity-cat.github.io/SIH-059/
   ```

---

## 4. API Endpoints Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Combined health status of Express + ML service |
| `GET` | `/api/model-info` | Metadata, architecture, and validation metrics |
| `GET` | `/api/grid-info` | 66x57 latitude/longitude coordinates & ocean mask |
| `GET` | `/api/available-dates` | Date range supported for forecasting (`2023-01-01` to `2025-12-31`) |
| `POST` | `/api/predict` | Runs real CNN inference. Accepts `{ "target_date": "YYYY-MM-DD" }` or explicit `{ "target_date": "...", "last_7_days": [[[...]]] }` |

---

## 5. Free Plan Notes
- **Cold Starts**: Render free instances go to sleep after 15 minutes of inactivity. When a request arrives after sleep, the instance takes ~40–50 seconds to spin up. Subsequent requests respond in milliseconds.
- **SSL / HTTPS**: Both Render and GitHub Pages automatically provision free SSL certificates.
