import json
import os
from datetime import datetime, timezone
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

os.makedirs("documentation_work/evidence/api_responses", exist_ok=True)

# 1. Health endpoint response
res_health = client.get("/health")
health_data = {
    "metadata": {
        "endpoint": "/health",
        "method": "GET",
        "capture_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status_code": res_health.status_code,
    },
    "response": res_health.json()
}
with open("documentation_work/evidence/api_responses/api_health.json", "w") as f:
    json.dump(health_data, f, indent=2)
print("Saved api_health.json")

# 2. AAPL analysis request (US market)
print("Requesting /api/v1/analyze for AAPL...")
res_aapl = client.post("/api/v1/analyze", json={
    "symbol": "AAPL",
    "market": "US",
    "timeframe": "1d"
})
print("AAPL Status:", res_aapl.status_code)
if res_aapl.status_code == 200:
    data_aapl = {
        "metadata": {
            "endpoint": "/api/v1/analyze",
            "method": "POST",
            "request_payload": {"symbol": "AAPL", "market": "US", "timeframe": "1d"},
            "capture_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status_code": 200,
        },
        "response": res_aapl.json()
    }
    with open("documentation_work/evidence/api_responses/api_analyze_aapl.json", "w") as f:
        json.dump(data_aapl, f, indent=2)
    print("Saved api_analyze_aapl.json")

# 3. NSEI analysis request (Indian market)
print("Requesting /api/v1/analyze for ^NSEI...")
res_nsei = client.post("/api/v1/analyze", json={
    "symbol": "^NSEI",
    "market": "IN",
    "timeframe": "1d"
})
print("NSEI Status:", res_nsei.status_code)
if res_nsei.status_code == 200:
    data_nsei = {
        "metadata": {
            "endpoint": "/api/v1/analyze",
            "method": "POST",
            "request_payload": {"symbol": "^NSEI", "market": "IN", "timeframe": "1d"},
            "capture_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "status_code": 200,
        },
        "response": res_nsei.json()
    }
    with open("documentation_work/evidence/api_responses/api_analyze_nsei.json", "w") as f:
        json.dump(data_nsei, f, indent=2)
    print("Saved api_analyze_nsei.json")
