import os
import logging
from typing import List

logger = logging.getLogger(__name__)

# Attempt to load local .env file in development if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class Settings:
    """Production runtime settings sourced strictly from environment variables.
    
    Cloud Run injects PORT (defaults to 8080) and secrets via Secret Manager
    or container environment variables.
    """
    def __init__(self):
        self.gemini_api_key: str = os.environ.get("GEMINI_API_KEY", "").strip()
        self.port: int = int(os.environ.get("PORT", "8080"))
        self.host: str = os.environ.get("HOST", "0.0.0.0")
        self.environment: str = os.environ.get("ENVIRONMENT", "production").lower()
        self.log_level: str = os.environ.get("LOG_LEVEL", "INFO").upper()
        self.default_gemini_model: str = os.environ.get("DEFAULT_GEMINI_MODEL", "gemini-2.5-flash")
        
        # Parse CORS origins from comma-separated string or allow all
        raw_cors = os.environ.get("CORS_ORIGINS", "*").strip()
        if raw_cors == "*" or not raw_cors:
            self.cors_origins: List[str] = ["*"]
        else:
            self.cors_origins: List[str] = [origin.strip() for origin in raw_cors.split(",") if origin.strip()]

    @property
    def is_gemini_configured(self) -> bool:
        """Returns True if a non-empty Gemini API key is present in environment."""
        return bool(self.gemini_api_key)

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


settings = Settings()
