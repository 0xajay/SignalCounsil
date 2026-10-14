import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from app.agents.council import SignalCouncilOrchestrator, AdversarialReviewOutput
from app.domain.models import (
    TradeThesisStressTestResult, DirectionalBias, EvidenceStrength,
    SetupStatus, BacktestStatus, ThesisStatus, HistoricalSetupResult,
    CanonicalAnalysisSummary, InstrumentMetadata, DataQualityReport
)

@pytest.mark.anyio
async def test_prompt_agent_gemini_flash_success(monkeypatch):
    """Ensure SignalCouncilOrchestrator calls Gemini Flash model when GEMINI_API_KEY is present."""
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")
    
    orchestrator = SignalCouncilOrchestrator()
    mock_client = MagicMock()
    
    mock_response = MagicMock()
    mock_response.parsed = AdversarialReviewOutput(
        bull_case="Bull case citing [EV-TREN-1]",
        bear_case="Bear case citing [EV-MOME-1]",
        explanation="Explanation citing [EV-TREN-1]"
    )
    mock_response.text = '{"bull_case": "...", "bear_case": "...", "explanation": "..."}'
    
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    orchestrator._client = mock_client
    
    result = await orchestrator._prompt_agent(
        identity="Test Identity",
        prompt="Test Prompt",
        schema=AdversarialReviewOutput
    )
    
    assert isinstance(result, AdversarialReviewOutput)
    assert result.bull_case == "Bull case citing [EV-TREN-1]"
    
    # Verify the model called was a Gemini Flash model
    mock_client.aio.models.generate_content.assert_called_once()
    call_kwargs = mock_client.aio.models.generate_content.call_args.kwargs
    assert "flash" in call_kwargs["model"].lower()
    assert "gemini" in call_kwargs["model"].lower()
    assert "pro" not in call_kwargs["model"].lower()
    assert "groq" not in call_kwargs["model"].lower()


@pytest.mark.anyio
async def test_prompt_agent_gemini_flash_fallback(monkeypatch):
    """Ensure SignalCouncilOrchestrator tries next Gemini Flash model on failure."""
    monkeypatch.setenv("GEMINI_API_KEY", "test_gemini_key")
    
    orchestrator = SignalCouncilOrchestrator()
    mock_client = MagicMock()
    
    mock_success_response = MagicMock()
    mock_success_response.parsed = AdversarialReviewOutput(
        bull_case="Fallback bull case [EV-TREN-1]",
        bear_case="Fallback bear case [EV-MOME-1]",
        explanation="Fallback explanation [EV-TREN-1]"
    )
    
    # First model call raises error, next flash model succeeds
    mock_client.aio.models.generate_content = AsyncMock(
        side_effect=[Exception("Quota exceeded on primary"), mock_success_response]
    )
    orchestrator._client = mock_client
    
    result = await orchestrator._prompt_agent(
        identity="Test Identity",
        prompt="Test Prompt",
        schema=AdversarialReviewOutput
    )
    
    assert isinstance(result, AdversarialReviewOutput)
    assert result.bull_case == "Fallback bull case [EV-TREN-1]"
    assert mock_client.aio.models.generate_content.call_count == 2
    
    # Verify both attempted models were Gemini Flash models
    for call in mock_client.aio.models.generate_content.call_args_list:
        model_used = call.kwargs["model"].lower()
        assert "flash" in model_used
        assert "gemini" in model_used
        assert "pro" not in model_used


@pytest.mark.anyio
async def test_stress_test_trade_thesis_without_gemini_key_falls_back_cleanly(monkeypatch):
    """When GEMINI_API_KEY is not set, stress_test_trade_thesis falls back gracefully to deterministic synthesis."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    
    orchestrator = SignalCouncilOrchestrator(api_key=None)
    
    dummy_summary = CanonicalAnalysisSummary(
        symbol="TEST",
        timeframe="1d",
        directional_bias="BULLISH",
        evidence_strength="MODERATE",
        setup_status="CONFIRMED",
        strategy_backtest_status="PASS",
        walk_forward_status="PASS",
        directional_family_alignment=75,
        directional_coverage_excl_risk="75/100",
        ema_relationship="Bullish",
        ema_20=105.0,
        ema_50=100.0,
        pattern_sample_count=20,
        pattern_win_rate_pct=60.0,
        pattern_profit_factor=1.8,
        pattern_max_drawdown_pct=5.0,
        walk_forward_win_rate_pct=55.0,
        walk_forward_profit_factor=1.4,
        walk_forward_compounded_return_pct=12.0,
        walk_forward_max_drawdown_pct=8.0,
        portfolio_compounded_return_pct=10.0,
        portfolio_max_drawdown_pct=6.0,
        risk_limit_max_drawdown_breached=False,
        benchmark_name="SPY",
        benchmark_return_pct=8.0,
        benchmark_alpha_pct=4.0,
        benchmark_alpha_status="ALPHA_POSITIVE",
        final_validated_edge_verdict="VALIDATED EDGE",
        selection_bias_provenance="Walk-Forward",
        sensitivity_analysis_summary="Robust"
    )
    
    dummy_input = TradeThesisStressTestResult(
        symbol="TEST",
        timeframe="1d",
        directional_bias=DirectionalBias.BULLISH,
        evidence_strength=EvidenceStrength.MODERATE,
        setup_status=SetupStatus.CONFIRMED,
        backtest_status=BacktestStatus.PASS,
        thesis_status=ThesisStatus.INTACT,
        historical_setup_results=HistoricalSetupResult(
            setup_definition="Test Setup",
            sample_count=20,
            win_rate=0.60,
            profit_factor=1.8,
            max_drawdown=0.05,
            net_return=0.15
        ),
        explanation="Deterministic baseline explanation citing [EV-TREN-1].",
        canonical_summary=dummy_summary,
        instrument=InstrumentMetadata(symbol="TEST", canonical_symbol="TEST"),
        data_quality=DataQualityReport()
    )
    
    result = await orchestrator.stress_test_trade_thesis(dummy_input)
    assert result is not None
    assert result.symbol == "TEST"
    assert result.explanation is not None
