import pytest
from unittest.mock import patch, MagicMock, AsyncMock
from app.agents.chart_agent import ChartParserAgent
from app.domain.models import ChartObservation

@pytest.mark.anyio
@patch("google.genai.Client")
async def test_analyze_chart_gemini_vision(mock_client_class, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "DUMMY_KEY")
    mock_client = MagicMock()
    mock_client_class.return_value = mock_client
    
    expected_obs = ChartObservation(
        visible_timeframe="1D",
        chart_type="Candlestick",
        visible_indicators=["RSI", "MACD", "Volume"],
        support_resistance_levels=["Horizontal support at 150"],
        chart_structures=["Double Bottom"],
        confidence=0.9
    )
    mock_response = MagicMock()
    mock_response.parsed = expected_obs
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    
    agent = ChartParserAgent(api_key="DUMMY_KEY")
    observation = await agent.analyze_chart(b"fake_image_bytes")
    
    assert isinstance(observation, ChartObservation)
    assert observation.visible_timeframe == "1D"
    assert observation.chart_type == "Candlestick"
    assert "RSI" in observation.visible_indicators
    assert observation.confidence == 0.9

@pytest.mark.anyio
async def test_analyze_chart_text_only_fallback(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    agent = ChartParserAgent(api_key=None)
    observation = await agent.analyze_chart(b"fake_image_bytes")
    assert isinstance(observation, ChartObservation)
    assert observation.chart_type == "Candlestick"
