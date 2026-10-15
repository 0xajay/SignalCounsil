# SignalCouncil: Production Cloud Run & Extension Deployment Guide

This guide provides step-by-step instructions for deploying the **SignalCouncil** backend to **Google Cloud Run** and configuring the **Chrome Extension** for production.

---

## Architecture & Production Readiness Summary

1. **Dynamic Cloud Run Port Binding:**
   The backend FastAPI application reads the dynamic `PORT` environment variable provided by Cloud Run (default: `8080`) and binds to `0.0.0.0:${PORT:-8080}` via `uvicorn`.
2. **Stateless Service Design:**
   SignalCouncil operates purely statelessly:
   - Real-time/historical OHLCV data is acquired dynamically via public market data endpoints.
   - Evidence graphs, dependency matrices, walk-forward testing, and stress-test calibrations execute purely in-memory.
   - LLM synthesis is generated on-demand via Gemini Flash.
   - *No local SQLite database or sticky file storage is used.* Instances can autoscale from 0 to N without persistent storage hazards or cold-state drift.
3. **Environment & Secrets Management:**
   Secrets (e.g. `GEMINI_API_KEY`) and runtime configurations (`PORT`, `HOST`, `ENVIRONMENT`, `CORS_ORIGINS`, `LOG_LEVEL`) are read strictly from environment variables or Google Cloud Secret Manager. No local paths or `.env` files are required in production containers.
4. **Health Check Probes:**
   Dedicated `/health`, `/healthz`, and `/api/v1/health` endpoints return service status without exposing keys or internal architecture details.
5. **Flexible Frontend API Configuration:**
   The Chrome Extension supports both build-time configuration (`VITE_API_URL`) and runtime user overrides via Chrome extension storage (`chrome.storage.sync`), allowing users to point their extension to their Cloud Run service URL right from the UI without rebuilding.

---

## 1. Backend Configuration & Environment Variables

| Variable | Required | Default | Description |
| :--- | :---: | :---: | :--- |
| `GEMINI_API_KEY` | **Yes** | — | Google Gemini API key (recommended via Secret Manager). |
| `PORT` | Auto | `8080` | Port for the container HTTP server. Provided dynamically by Cloud Run. |
| `HOST` | No | `0.0.0.0` | IP binding address. Must be `0.0.0.0` in containers. |
| `ENVIRONMENT` | No | `production` | Deployment stage (`production`, `staging`, `development`). |
| `CORS_ORIGINS` | No | `*` | Comma-separated allowed origins or `*` to allow Chrome extensions. |
| `LOG_LEVEL` | No | `INFO` | Python log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`). |
| `DEFAULT_GEMINI_MODEL`| No | `gemini-2.5-flash` | Gemini model name used for AI synthesis and chart vision. |

---

## 2. Deploying Backend to Google Cloud Run

### Prerequisites
- [Google Cloud SDK (`gcloud`)](https://cloud.google.com/sdk/docs/install) installed and authenticated.
- A Google Cloud project with billing enabled.
- Cloud Run and Cloud Build APIs enabled:
  ```bash
  gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com
  ```

### Step 2.1: Store Gemini API Key in Secret Manager
Store your Gemini API key securely so Cloud Run mounts it safely at runtime:
```bash
# Create the secret
gcloud secrets create signal-gemini-key --replication-policy="automatic"

# Add the secret payload (paste your Gemini API key when prompted, or pass via echo)
echo -n "YOUR_ACTUAL_GEMINI_API_KEY" | gcloud secrets versions add signal-gemini-key --data-file=-
```

### Step 2.2: Deploy Directly with Cloud Build & Cloud Run
From the `signal-council` repository root or `backend/` directory:

#### Option A: Deploy from `backend/` directory (Recommended)
```bash
cd backend

gcloud run deploy signal-council-api \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --set-secrets GEMINI_API_KEY=signal-gemini-key:latest \
  --set-env-vars ENVIRONMENT=production,CORS_ORIGINS="*" \
  --min-instances 0 \
  --max-instances 10 \
  --memory 1Gi \
  --cpu 1
```

#### Option B: Deploy from Repository Root
```bash
gcloud run deploy signal-council-api \
  --source . \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --set-secrets GEMINI_API_KEY=signal-gemini-key:latest \
  --set-env-vars ENVIRONMENT=production,CORS_ORIGINS="*" \
  --min-instances 0 \
  --max-instances 10 \
  --memory 1Gi \
  --cpu 1
```

Once deployment completes, Cloud Run will output your service URL:
```text
Service [signal-council-api] revision [signal-council-api-00001-xxx] has been deployed and is serving 100 percent of traffic.
Service URL: https://signal-council-api-xxxxxxxxxx-uc.a.run.app
```

---

## 3. Verifying Backend Health & Status

Verify that your Cloud Run container has started and responds:

```bash
# 1. Check health endpoint (returns HTTP 200)
curl -s https://<YOUR_CLOUD_RUN_URL>/health

# Example output:
# {"status":"healthy","service":"signal-council-api","version":"1.0.0","environment":"production"}

# 2. Run a smoke test on a market symbol
curl -s -X POST https://<YOUR_CLOUD_RUN_URL>/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol": "AAPL", "timeframe": "1d"}' | jq .verdict
```

---

## 4. Local Container Testing (Optional)

You can build and test the container locally with Docker:

```bash
# Build the container image
docker build -t signal-council-backend -f backend/Dockerfile backend/

# Run the container listening on port 8080
docker run -d \
  --name signal-council \
  -e PORT=8080 \
  -e GEMINI_API_KEY="YOUR_GEMINI_API_KEY" \
  -e ENVIRONMENT=development \
  -p 8080:8080 \
  signal-council-backend

# Check health
curl http://localhost:8080/health
```

---

## 5. Building & Configuring the Chrome Extension

The frontend Chrome extension connects to your Cloud Run backend.

### Option 1: Configure at Build Time (Via `.env`)
1. Create `extension/.env` from `.env.example`:
   ```bash
   cd extension
   echo "VITE_API_URL=https://<YOUR_CLOUD_RUN_URL>" > .env
   ```
2. Build the extension bundle:
   ```bash
   npm run build
   ```
   The production-ready unpacked extension will be generated in `extension/dist/`.

### Option 2: Configure at Runtime in Extension UI (No Rebuild Needed)
If you already built or installed the extension:
1. Open the SignalCouncil extension side panel or popup.
2. Click the **API Server** badge or **Settings** gear icon in the header.
3. Paste your Cloud Run Service URL:
   `https://signal-council-api-xxxxxxxxxx-uc.a.run.app`
4. Click **Test Connection** to verify live connectivity with your `/health` endpoint.
5. Click **Save**. The extension will store this URL in `chrome.storage.sync` and immediately route all subsequent stress-test analyses to your deployed Cloud Run backend!

---

## 6. Installing Extension into Google Chrome

1. Open Google Chrome and navigate to: `chrome://extensions/`
2. In the top right corner, toggle **Developer mode** to **ON**.
3. In the top left corner, click **Load unpacked**.
4. Select the `extension/dist` folder inside this repository.
5. Click the Extensions puzzle icon in Chrome's toolbar and pin **SignalCouncil**.
6. Open any financial chart on **TradingView** (e.g. `tradingview.com/chart`), click the SignalCouncil icon to open the side panel, and click **Stress Test Active Chart**.

---

## 7. Troubleshooting & FAQs

### Q: Does Cloud Run require persistent disk storage for SignalCouncil?
**No.** All calculations (OHLCV indicators, evidence graph matrices, backtests, and stress calibrations) are purely functional and computed on-demand in-memory. State is not saved locally, making the container completely stateless and resilient to instance recycling and autoscaling.

### Q: Why do we use `exec uvicorn ... --port ${PORT:-8080}` in Dockerfile?
Cloud Run assigns an arbitrary port via the `$PORT` environment variable during container initialization. Running uvicorn with shell variable expansion `${PORT:-8080}` guarantees the server listens on Cloud Run's required port, while `exec` replaces the shell process so container shutdown signals (SIGTERM) are handled immediately.

### Q: Are my API keys exposed in the health check or extension?
**No.** The `/health` endpoint only reports service health and whether Gemini is configured (`status: healthy`), never displaying the raw key. In the frontend, the extension speaks exclusively to your backend API, so the `GEMINI_API_KEY` remains securely kept inside Cloud Run / Secret Manager.
