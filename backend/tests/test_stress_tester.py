import pytest
from app.domain.models import (
    StrategyResult, Signal, DirectionalBias, SetupStatus,
    ThesisStatus, BacktestStatus, EvidenceStrength
)
from app.domain.stress_tester import TradeThesisStressTester
from tests.fixtures.market_data import get_ohlcv_models

def test_stress_test_bullish_case():
    data = get_ohlcv_models()
    tester = TradeThesisStressTester(data, symbol="AAPL", timeframe="1d")

    # Consistent bullish signals across independent families
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.BUY, confidence=0.8, metadata={"ema_20": 110, "ema_50": 100}),
        StrategyResult(strategy_name="Breakout", signal=Signal.BUY, confidence=0.7, metadata={"donchian_upper": 108}),
        StrategyResult(strategy_name="Momentum", signal=Signal.BUY, confidence=0.8, metadata={"rsi": 28}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.BUY, confidence=0.7, metadata={"higher_highs": True}),
        StrategyResult(strategy_name="MeanReversion", signal=Signal.NEUTRAL, confidence=0.5, metadata={}),
        StrategyResult(strategy_name="Risk", signal=Signal.NEUTRAL, confidence=0.8, metadata={"volatility_pct": 2.0}),
    ]

    result = tester.run_stress_test(strat_results)

    assert result.setup_status == SetupStatus.CONFIRMED
    assert len(result.supporting_evidence) >= 3
    assert len(result.confirmation_conditions) > 0
    assert len(result.invalidation_conditions) > 0
    assert result.historical_setup_results.sample_count > 0

def test_stress_test_bearish_case():
    data = get_ohlcv_models()
    tester = TradeThesisStressTester(data, symbol="AAPL", timeframe="1d")

    # Consistent bearish signals across independent families
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.8, metadata={"ema_20": 90, "ema_50": 100}),
        StrategyResult(strategy_name="Breakout", signal=Signal.SELL, confidence=0.7, metadata={"donchian_lower": 92}),
        StrategyResult(strategy_name="Momentum", signal=Signal.SELL, confidence=0.8, metadata={"rsi": 75}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.SELL, confidence=0.7, metadata={"lower_lows": True}),
        StrategyResult(strategy_name="MeanReversion", signal=Signal.NEUTRAL, confidence=0.5, metadata={}),
        StrategyResult(strategy_name="Risk", signal=Signal.NEUTRAL, confidence=0.8, metadata={"volatility_pct": 2.0}),
    ]

    result = tester.run_stress_test(strat_results)

    assert result.setup_status == SetupStatus.CONFIRMED
    assert len(result.supporting_evidence) >= 3
    assert len(result.confirmation_conditions) > 0
    assert len(result.invalidation_conditions) > 0

def test_stress_test_sideways_case():
    data = get_ohlcv_models()
    tester = TradeThesisStressTester(data, symbol="SPY", timeframe="1d")

    # All neutral / indeterminate signals
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
        StrategyResult(strategy_name="Breakout", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
        StrategyResult(strategy_name="Momentum", signal=Signal.NEUTRAL, confidence=0.5, metadata={"rsi": 50}),
        StrategyResult(strategy_name="MeanReversion", signal=Signal.NEUTRAL, confidence=0.5, metadata={}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
        StrategyResult(strategy_name="Risk", signal=Signal.NEUTRAL, confidence=0.8, metadata={"volatility_pct": 1.5}),
    ]

    result = tester.run_stress_test(strat_results)

    assert result.directional_bias == DirectionalBias.NEUTRAL
    assert result.setup_status == SetupStatus.UNCONFIRMED
    assert result.thesis_status == ThesisStatus.NOT_ESTABLISHED
    assert "NO RELIABLE EDGE ESTABLISHED" in result.explanation

def test_stress_test_conflicted_case():
    data = get_ohlcv_models()
    tester = TradeThesisStressTester(data, symbol="TSLA", timeframe="1d")

    # Trend says BUY, but Momentum & MeanReversion say SELL
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.BUY, confidence=0.8, metadata={}),
        StrategyResult(strategy_name="Breakout", signal=Signal.BUY, confidence=0.7, metadata={}),
        StrategyResult(strategy_name="Momentum", signal=Signal.SELL, confidence=0.8, metadata={"rsi": 78}),
        StrategyResult(strategy_name="MeanReversion", signal=Signal.SELL, confidence=0.8, metadata={}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
        StrategyResult(strategy_name="Risk", signal=Signal.NEUTRAL, confidence=0.8, metadata={}),
    ]

    result = tester.run_stress_test(strat_results)

    # Conflicted evidence must force NEUTRAL and NOT_ESTABLISHED
    assert result.setup_status == SetupStatus.CONFLICTED
    assert result.directional_bias == DirectionalBias.NEUTRAL
    assert result.thesis_status == ThesisStatus.NOT_ESTABLISHED
    assert any("Conflicted Evidence" in w for w in result.data_quality_warnings)
    assert "NO RELIABLE EDGE ESTABLISHED" in result.explanation

def test_stress_test_invalid_data_case():
    data = get_ohlcv_models()[:10] # Only 10 candles (< 25 critical minimum)
    tester = TradeThesisStressTester(data, symbol="INVALID", timeframe="1d")

    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.BUY, confidence=0.8, metadata={}),
    ]

    result = tester.run_stress_test(strat_results)

    assert result.setup_status == SetupStatus.INVALID_DATA
    assert result.thesis_status == ThesisStatus.NOT_ESTABLISHED
    assert result.evidence_strength == EvidenceStrength.INSUFFICIENT
    assert len(result.data_quality_warnings) > 0
