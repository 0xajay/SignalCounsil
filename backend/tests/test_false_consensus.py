import pytest
from app.domain.models import (
    OHLCVData, StrategyResult, Signal, DirectionalBias, EvidenceFamily
)
from app.domain.evidence import EvidenceLedgerBuilder
from tests.fixtures.market_data import get_ohlcv_models

def test_false_consensus_collinear_detection():
    data = get_ohlcv_models()
    market_data = OHLCVData(data=data)

    # 2 strategies in the TREND family both vote BUY (collinear)
    # 1 strategy in MOMENTUM votes BUY
    # 1 strategy in MEAN_REVERSION votes NEUTRAL
    results = [
        StrategyResult(strategy_name="Trend", signal=Signal.BUY, confidence=0.8, metadata={"ema_20": 105, "ema_50": 100}),
        StrategyResult(strategy_name="Breakout", signal=Signal.BUY, confidence=0.7, metadata={"donchian_upper": 104}),
        StrategyResult(strategy_name="Momentum", signal=Signal.BUY, confidence=0.6, metadata={"rsi": 25}),
        StrategyResult(strategy_name="MeanReversion", signal=Signal.NEUTRAL, confidence=0.5, metadata={}),
    ]

    builder = EvidenceLedgerBuilder(market_data, results)
    claims = builder.build_claims()
    assert len(claims) == 4

    dependencies, raw_votes, indep_votes, collinear_detected, dep_explanation = builder.detect_false_consensus(claims)

    # Raw votes should count 3 BUYs
    assert raw_votes["BUY"] == 3

    # But independent family votes should only count 2 BULLISH families (TREND and MOMENTUM)
    assert indep_votes["BULLISH"] == 2
    assert indep_votes["NEUTRAL"] == 1

    # Collinear overlap must be detected because raw count (3) > independent count (2)
    assert collinear_detected is True
    assert "False Consensus Alert" in dep_explanation
    assert "Trend" in dep_explanation

def test_independent_family_internal_conflict():
    data = get_ohlcv_models()
    market_data = OHLCVData(data=data)

    # Conflict within TREND family: Trend says BUY, Breakout says SELL
    results = [
        StrategyResult(strategy_name="Trend", signal=Signal.BUY, confidence=0.8, metadata={}),
        StrategyResult(strategy_name="Breakout", signal=Signal.SELL, confidence=0.8, metadata={}),
    ]

    builder = EvidenceLedgerBuilder(market_data, results)
    claims = builder.build_claims()
    dependencies, raw_votes, indep_votes, collinear_detected, dep_explanation = builder.detect_false_consensus(claims)

    assert indep_votes["CONFLICTED"] == 1
    trend_dep = [d for d in dependencies if d.family == EvidenceFamily.TREND][0]
    assert "Contradiction within TREND" in trend_dep.collinearity_risk
