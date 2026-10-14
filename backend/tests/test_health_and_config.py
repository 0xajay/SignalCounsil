from fastapi.testclient import TestClient
from main import app
from app.config import Settings

client = TestClient(app)

def test_health_endpoints():
    """Verify lightweight health endpoints for Cloud Run / k8s liveness probes."""
    for path in ["/health", "/healthz", "/api/v1/health", "/"]:
        response = client.get(path)
        assert response.status_code == 200
        data = response.json()
        assert data.get("status") in ("healthy", "ok")
        assert data.get("service") == "signal-council-api"
        # Verify NO credentials or secret keys are exposed
        assert "key" not in data
        assert "secret" not in data
        assert "gemini_api_key" not in data

def test_settings_default_environment(monkeypatch):
    """Verify settings parses PORT, HOST, CORS, etc correctly from env."""
    monkeypatch.setenv("PORT", "9090")
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("CORS_ORIGINS", "https://example.com, https://app.example.com")
    monkeypatch.setenv("ENVIRONMENT", "staging")
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key_123")
    
    custom_settings = Settings()
    assert custom_settings.port == 9090
    assert custom_settings.host == "127.0.0.1"
    assert custom_settings.environment == "staging"
    assert custom_settings.cors_origins == ["https://example.com", "https://app.example.com"]
    assert custom_settings.is_gemini_configured is True
    assert custom_settings.is_production is False
