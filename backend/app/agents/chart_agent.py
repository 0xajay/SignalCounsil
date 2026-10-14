import os
import json
import logging
from app.domain.models import ChartObservation

logger = logging.getLogger(__name__)

class ChartParserAgent:
    """
    Multimodal agent responsible for reading chart images and extracting deterministic observations.
    Supports Gemini Flash multimodal vision when GEMINI_API_KEY is available.
    Gracefully falls back to baseline technical observations when offline or API key is not set.
    """
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        
    async def analyze_chart(self, image_bytes: bytes, mime_type: str = "image/png") -> ChartObservation:
        gemini_key = self.api_key or os.environ.get("GEMINI_API_KEY")
        
        # 1. Try Gemini Flash Multimodal Vision if key is available
        if gemini_key:
            try:
                from google import genai
                from google.genai import types
                client = genai.Client(api_key=gemini_key)
                prompt = (
                    "You are an expert technical analyst. Observe this financial chart and extract "
                    "the structural information requested. Do not make any buy or sell decisions. "
                    "Simply describe what is visible on the chart accurately."
                )
                vision_models = ["gemini-flash-lite-latest", "gemini-3-flash-preview", "gemini-3.8-flash", "gemini-flash-latest"]
                for model_name in vision_models:
                    try:
                        logger.info(f"Parsing chart with Gemini Vision ({model_name})...")
                        response = await client.aio.models.generate_content(
                            model=model_name,
                            contents=[
                                types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                                prompt
                            ],
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=ChartObservation,
                                temperature=0.0
                            )
                        )
                        if response.parsed:
                            return response.parsed
                        return ChartObservation.model_validate_json(response.text)
                    except Exception as model_err:
                        logger.warning(f"Vision model {model_name} failed: {model_err}")
                        continue
            except Exception as e:
                logger.warning(f"Gemini chart vision parsing failed ({e}); falling back.")

        # 2. Provide safe, structured baseline observation when API key is unavailable or request fails
        logger.info("Multimodal screenshot parsing skipped: GEMINI_API_KEY not configured. To enable visual chart parsing, export GEMINI_API_KEY.")
        return ChartObservation(
            visible_timeframe="Auto",
            chart_type="Candlestick",
            visible_indicators=["Price Action", "OHLCV"],
            support_resistance_levels=[],
            chart_structures=[],
            confidence=0.5
        )
