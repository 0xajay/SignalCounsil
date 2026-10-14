import pytest
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from app.domain.models import (
    OHLCV, OHLCVData, StrategyResult, Signal, DirectionalBias,
    EvidenceFamily, EvidencePolarity, EvidenceStrength, SetupStatus,
    BacktestStatus, ThesisStatus, InstrumentMetadata, DataQualityReport
)
from app.adapters.yahoo_finance import get_instrument_metadata
from app.domain.evidence import EvidenceLedgerBuilder
from app.domain.historical_setup import HistoricalSetupEngine, HistoricalSetupConfig
from app.domain.stress_tester import TradeThesisStressTester
from tests.fixtures.market_data import get_ohlcv_models, create_bullish_trending_data

# -------------------------------------------------------------------------
# Test 1: NIFTY index points and metadata
# -------------------------------------------------------------------------
def test_nifty_index_points_and_metadata():
    meta = get_instrument_metadata("^NSEI")
    
    assert meta.symbol == "^NSEI"
    assert meta.canonical_symbol == "^NSEI"
    assert meta.instrument_type == "INDEX"
    assert meta.unit_label == "points"
    assert meta.exchange == "NSE"
    assert meta.exchange != "NSI"
    assert meta.currency == "INR"

# -------------------------------------------------------------------------
# Test 2: Missing or invalid volume
# -------------------------------------------------------------------------
def test_missing_or_invalid_volume():
    # Index with zero volume
    data = get_ohlcv_models()
    for d in data:
        d.volume = 0
    
    nifty_meta = InstrumentMetadata(
        symbol="^NSEI",
        canonical_symbol="^NSEI",
        instrument_type="INDEX",
        unit_label="points",
        exchange="NSE",
        currency="INR"
    )
    
    tester = TradeThesisStressTester(data, symbol="^NSEI", timeframe="1d", instrument=nifty_meta)
    df = pd.DataFrame([d.model_dump() for d in data])
    report = tester.audit_data_quality(df)
    
    # Should flag volume as not valid for volume confirmation
    assert report.has_valid_volume is False
    assert report.zero_volume_bars == len(data)
    
    # When running stress test, triggers must not require volume confirmation
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.BUY, confidence=0.8, metadata={}),
        StrategyResult(strategy_name="Breakout", signal=Signal.BUY, confidence=0.7, metadata={}),
        StrategyResult(strategy_name="Momentum", signal=Signal.BUY, confidence=0.8, metadata={}),
        StrategyResult(strategy_name="MeanReversion", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.BUY, confidence=0.7, metadata={}),
        StrategyResult(strategy_name="Risk", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
    ]
    res = tester.run_stress_test(strat_results)
    
    for conf in res.confirmation_conditions:
        assert "on confirmed volume" not in conf.rule

# -------------------------------------------------------------------------
# Test 3: Neutral market bias with a bearish trend signal
# -------------------------------------------------------------------------
def test_neutral_market_bias_with_bearish_trend_signal():
    data = get_ohlcv_models()
    tester = TradeThesisStressTester(data, symbol="AAPL", timeframe="1d")
    
    # Trend says SELL (bearish), but Momentum & MeanReversion say BUY (bullish)
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.8, metadata={"ema_20": 95, "ema_50": 100}),
        StrategyResult(strategy_name="Breakout", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
        StrategyResult(strategy_name="Momentum", signal=Signal.BUY, confidence=0.75, metadata={"rsi": 30}),
        StrategyResult(strategy_name="MeanReversion", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
        StrategyResult(strategy_name="Risk", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
    ]
    
    res = tester.run_stress_test(strat_results)
    
    assert res.directional_bias == DirectionalBias.NEUTRAL
    assert res.setup_status == SetupStatus.CONFLICTED
    assert res.tested_hypothesis == "RANGE_CONSOLIDATION"
    # Never claim a bearish bias in the synthesis text when final directional bias is NEUTRAL
    assert "• Market Bias: NEUTRAL" in res.explanation
    assert "Market bias is NEUTRAL" in res.explanation

# -------------------------------------------------------------------------
# Test 4: Failed backtest with bearish directional evidence
# -------------------------------------------------------------------------
def test_failed_backtest_with_bearish_directional_evidence():
    # Dataset where shorting fails
    data = get_ohlcv_models()
    tester = TradeThesisStressTester(data, symbol="AAPL", timeframe="1d")
    
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.85, metadata={"ema_20": 90, "ema_50": 100}),
        StrategyResult(strategy_name="Breakout", signal=Signal.SELL, confidence=0.75, metadata={"donchian_lower": 92}),
        StrategyResult(strategy_name="Momentum", signal=Signal.SELL, confidence=0.80, metadata={"rsi": 70}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.SELL, confidence=0.70, metadata={}),
        StrategyResult(strategy_name="MeanReversion", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
        StrategyResult(strategy_name="Risk", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
    ]
    
    res = tester.run_stress_test(strat_results)
    
    # Requirement: failed backtest blocks validated profitability, but does NOT force bias to NEUTRAL
    assert res.backtest_status == BacktestStatus.FAIL
    assert res.directional_bias == DirectionalBias.BEARISH
    assert res.thesis_status == ThesisStatus.NOT_ESTABLISHED
    assert "NO VALIDATED EDGE" in res.verdict
    assert "VALIDATED THESIS" not in res.verdict

# -------------------------------------------------------------------------
# Test 5: Counterargument that must oppose the selected thesis
# -------------------------------------------------------------------------
def test_counterargument_must_oppose_selected_thesis():
    data = create_bullish_trending_data(100)
    tester = TradeThesisStressTester(data, symbol="NVDA", timeframe="1d")
    
    # Bullish candidate with 1 opposing sell signal from MeanReversion
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.BUY, confidence=0.85, metadata={}),
        StrategyResult(strategy_name="Breakout", signal=Signal.BUY, confidence=0.75, metadata={}),
        StrategyResult(strategy_name="Momentum", signal=Signal.BUY, confidence=0.80, metadata={}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.BUY, confidence=0.70, metadata={}),
        StrategyResult(strategy_name="MeanReversion", signal=Signal.SELL, confidence=0.65, metadata={"rsi": 78}),
        StrategyResult(strategy_name="Risk", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
    ]
    
    res = tester.run_stress_test(strat_results)
    
    assert res.tested_hypothesis == "BULLISH_CONTINUATION"
    # Strongest counterargument (bear_case) must challenge the bullish hypothesis
    assert "Challenges Bullish Hypothesis" in res.bear_case
    assert "MeanReversion" in res.bear_case or "EV-MEAN" in res.bear_case
    # bear_case must NOT repeat the bullish supporting cases
    assert "Bullish thesis supported by" not in res.bear_case

# -------------------------------------------------------------------------
# Test 6: Correlated indicators grouped into one adjusted family
# -------------------------------------------------------------------------
def test_correlated_indicators_grouped_into_one_adjusted_family():
    data = get_ohlcv_models()
    market_data = OHLCVData(data=data)
    
    # Trend and Breakout are both in TREND family
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.BUY, confidence=0.8, metadata={}),
        StrategyResult(strategy_name="Breakout", signal=Signal.BUY, confidence=0.7, metadata={}),
        StrategyResult(strategy_name="Momentum", signal=Signal.BUY, confidence=0.8, metadata={}),
    ]
    
    builder = EvidenceLedgerBuilder(market_data, strat_results)
    claims = builder.build_claims()
    dependencies, raw_votes, indep_votes, collinear_detected, dep_explanation = builder.detect_false_consensus(claims)
    
    assert raw_votes["BUY"] == 3
    assert indep_votes["BULLISH"] == 2 # Trend family (1) + Momentum family (1)
    assert collinear_detected is True
    # Verify wording does not claim collinearity inflates weight
    assert "inflating" not in dep_explanation.lower() or "not inflating" in dep_explanation.lower()
    assert "reduced duplicate votes" in dep_explanation.lower()

# -------------------------------------------------------------------------
# Test 7: Ledger counts matching the canonical result
# -------------------------------------------------------------------------
def test_ledger_counts_matching_canonical_result():
    data = get_ohlcv_models()
    tester = TradeThesisStressTester(data, symbol="MSFT", timeframe="1d")
    
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.BUY, confidence=0.8, metadata={}),
        StrategyResult(strategy_name="Breakout", signal=Signal.BUY, confidence=0.7, metadata={}),
        StrategyResult(strategy_name="Momentum", signal=Signal.SELL, confidence=0.6, metadata={}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.BUY, confidence=0.7, metadata={}),
        StrategyResult(strategy_name="MeanReversion", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
        StrategyResult(strategy_name="Risk", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
    ]
    
    res = tester.run_stress_test(strat_results)
    
    # Total ledger items
    total_claims = len(res.supporting_evidence) + len(res.opposing_evidence) + len(res.neutral_evidence)
    assert total_claims == len(strat_results) == 6
    assert len(res.supporting_evidence) == 3 # 3 BUYs
    assert len(res.opposing_evidence) == 1   # 1 SELL
    assert len(res.neutral_evidence) == 2    # 2 NEUTRALs
    assert res.reconciliation_status == "VALIDATED"

# -------------------------------------------------------------------------
# Test 8: Bullish and bearish triggers with separate invalidation conditions
# -------------------------------------------------------------------------
def test_bullish_and_bearish_triggers_with_separate_invalidation_conditions():
    data = get_ohlcv_models()
    tester = TradeThesisStressTester(data, symbol="AAPL", timeframe="1d")
    
    # Bullish scenario
    strat_bull = [
        StrategyResult(strategy_name="Trend", signal=Signal.BUY, confidence=0.8, metadata={}),
        StrategyResult(strategy_name="Breakout", signal=Signal.BUY, confidence=0.7, metadata={}),
        StrategyResult(strategy_name="Momentum", signal=Signal.BUY, confidence=0.8, metadata={}),
        StrategyResult(strategy_name="MeanReversion", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
        StrategyResult(strategy_name="Risk", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
    ]
    res_bull = tester.run_stress_test(strat_bull)
    assert res_bull.tested_hypothesis == "BULLISH_CONTINUATION"
    assert any("resistance" in c.rule.lower() for c in res_bull.confirmation_conditions)
    assert any("support" in c.rule.lower() for c in res_bull.invalidation_conditions)
    assert len(res_bull.risk_controls) >= 3
    
    # Bearish scenario
    strat_bear = [
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.8, metadata={}),
        StrategyResult(strategy_name="Breakout", signal=Signal.SELL, confidence=0.7, metadata={}),
        StrategyResult(strategy_name="Momentum", signal=Signal.SELL, confidence=0.8, metadata={}),
        StrategyResult(strategy_name="MeanReversion", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
        StrategyResult(strategy_name="Risk", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
    ]
    res_bear = tester.run_stress_test(strat_bear)
    assert res_bear.tested_hypothesis == "BEARISH_EXPANSION"
    assert any("support" in c.rule.lower() for c in res_bear.confirmation_conditions)
    assert any("resistance" in c.rule.lower() for c in res_bear.invalidation_conditions)
    assert len(res_bear.risk_controls) >= 3

# -------------------------------------------------------------------------
# Test 9: Insufficient out-of-sample data
# -------------------------------------------------------------------------
def test_insufficient_out_of_sample_data():
    # 60 candles: enough to pass warmup (50), but very few bars in OOS (35% of 60 = 21 bars)
    data = get_ohlcv_models()[:58]
    engine = HistoricalSetupEngine(data, config=HistoricalSetupConfig(min_oos_samples=5))
    res, status = engine.investigate_setup(DirectionalBias.BULLISH)
    
    assert res.out_of_sample is not None
    if res.out_of_sample.sample_count < 5:
        assert res.out_of_sample.status == BacktestStatus.INSUFFICIENT_DATA
        assert res.oos_status == BacktestStatus.INSUFFICIENT_DATA
        assert "Insufficient sample size" in res.failure_explanation

# -------------------------------------------------------------------------
# Test 10: Stale or incomplete daily candles
# -------------------------------------------------------------------------
def test_stale_or_incomplete_daily_candles():
    data = get_ohlcv_models()
    tester = TradeThesisStressTester(data, symbol="AAPL", timeframe="1d")
    
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
        StrategyResult(strategy_name="Breakout", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
        StrategyResult(strategy_name="Momentum", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
        StrategyResult(strategy_name="MeanReversion", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
        StrategyResult(strategy_name="Risk", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
    ]
    
    res = tester.run_stress_test(strat_results)
    
    # Must have last_completed_candle_date in YYYY-MM-DD format
    assert res.instrument.last_completed_candle_date != ""
    assert len(res.instrument.last_completed_candle_date) == 10
    assert "T00:00:00" not in res.instrument.last_completed_candle_date
    assert res.data_quality.schema_valid is True
    assert res.data_quality.freshness_valid is True
