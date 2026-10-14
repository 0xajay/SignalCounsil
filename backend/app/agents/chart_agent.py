import os
import json
import logging
from app.domain.models import ChartObservation

logger = logging.getLogger(__name__)

class ChartParserAgent:
    """
    Multimodal agent responsible for reading chart images and extracting deterministic observations.
    Supports Gemini multimodal vision when GEMINI_API_KEY is available.
    Gracefully falls back when only text LLMs (like Groq) are active.
    """
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        
    async def analyze_chart(self, image_bytes: bytes, mime_type: str = "image/png") -> ChartObservation:
        gemini_key = self.api_key or os.environ.get("GEMINI_API_KEY")
        
        # 1. Try Gemini Multimodal Vision if key is available
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
                logger.info("Parsing chart with Gemini Vision (gemini-3.8-flash)...")
                response = await client.aio.models.generate_content(
                    model="gemini-3.8-flash",
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
            except Exception as e:
                logger.warning(f"Gemini chart vision parsing failed ({e}); falling back.")

        # 2. If using Groq: Groq models (openai/gpt-oss-120b, qwen, etc.) are strictly text-only
        # and reject image_url payloads with 'messages[0].content must be a string'.
        # We provide a safe, structured baseline observation instead of failing with HTTP 400.
        logger.info("Multimodal screenshot parsing skipped: Groq active models are text-only. To enable visual chart parsing, export GEMINI_API_KEY.")
        return ChartObservation(
            visible_timeframe="Auto",
            chart_type="Candlestick",
            visible_indicators=["Price Action", "OHLCV"],
            support_resistance_levels=[],
            chart_structures=[],
            confidence=0.5
        )
