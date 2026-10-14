import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from app.domain.models import (
    OHLCV, StrategyResult, Signal, DirectionalBias, EvidenceStrength,
    SetupStatus, BacktestStatus, ThesisStatus, InstrumentMetadata,
    DataQualityReport, HistoricalSetupConfig
)
from app.domain.stress_tester import TradeThesisStressTester
from app.domain.historical_setup import HistoricalSetupEngine
from tests.fixtures.market_data import get_ohlcv_models

def create_bullish_trending_data(num_bars: int = 120) -> list[OHLCV]:
    """Generate steadily upward trending market data where bullish setups pass historical tests."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(num_bars)]
    close = 100.0 + np.arange(num_bars) * 0.8 + np.sin(np.arange(num_bars)) * 0.5
    high = close + 1.2
    low = close - 0.8
    open_p = close - 0.3
    volume = [2500] * num_bars
    return [
        OHLCV(
            timestamp=dates[i],
            open=float(open_p[i]),
            high=float(high[i]),
            low=float(low[i]),
            close=float(close[i]),
            volume=int(volume[i])
        )
        for i in range(num_bars)
    ]

# -------------------------------------------------------------------------
# Acceptance Test 1: Bullish evidence with a valid historical test
# -------------------------------------------------------------------------
def test_acceptance_1_bullish_evidence_with_valid_historical_test():
    data = create_bullish_trending_data(120)
    tester = TradeThesisStressTester(data, symbol="AAPL", timeframe="1d")

    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.BUY, confidence=0.8, metadata={"ema_20": 115, "ema_50": 105}),
        StrategyResult(strategy_name="Breakout", signal=Signal.BUY, confidence=0.7, metadata={"donchian_upper": 114}),
        StrategyResult(strategy_name="Momentum", signal=Signal.BUY, confidence=0.8, metadata={"rsi": 32}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.BUY, confidence=0.7, metadata={"higher_highs": True}),
        StrategyResult(strategy_name="MeanReversion", signal=Signal.NEUTRAL, confidence=0.5, metadata={}),
        StrategyResult(strategy_name="Risk", signal=Signal.NEUTRAL, confidence=0.8, metadata={"volatility_pct": 2.0}),
    ]

    result = tester.run_stress_test(strat_results)

    # 1. Thesis and evidence confirmation
    assert result.setup_status == SetupStatus.CONFIRMED
    assert result.directional_bias == DirectionalBias.BULLISH
    assert result.backtest_status in (BacktestStatus.PASS, BacktestStatus.WARNING)
    assert result.thesis_status == ThesisStatus.INTACT
    assert result.verdict.startswith("VALIDATED THESIS: BULLISH")
    assert result.reconciliation_status == "VALIDATED"

    # 2. Reconciled counts: 4 BUY signals match 4 supporting claims
    assert len(result.supporting_evidence) == 4
    assert len(result.opposing_evidence) == 0
    assert len(result.neutral_evidence) == 2

    # 3. Direction-aware long triggers
    assert any(c.condition_id == "CONF-1" and "above swing resistance" in c.rule for c in result.confirmation_conditions)
    assert any(c.condition_id == "INVAL-1" and "below" in c.rule for c in result.invalidation_conditions)

# -------------------------------------------------------------------------
# Acceptance Test 2: Bearish indication with a failed historical test
# -------------------------------------------------------------------------
def test_acceptance_2_bearish_indication_with_failed_historical_test():
    # Synthetic random walk with upward drift where shorting fails backtest
    data = get_ohlcv_models()
    tester = TradeThesisStressTester(data, symbol="AAPL", timeframe="1d")

    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.8, metadata={"ema_20": 90, "ema_50": 100}),
        StrategyResult(strategy_name="Breakout", signal=Signal.SELL, confidence=0.7, metadata={"donchian_lower": 92}),
        StrategyResult(strategy_name="Momentum", signal=Signal.SELL, confidence=0.8, metadata={"rsi": 75}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.SELL, confidence=0.7, metadata={"lower_lows": True}),
        StrategyResult(strategy_name="MeanReversion", signal=Signal.NEUTRAL, confidence=0.5, metadata={}),
        StrategyResult(strategy_name="Risk", signal=Signal.NEUTRAL, confidence=0.8, metadata={"volatility_pct": 2.0}),
    ]

    result = tester.run_stress_test(strat_results)

    # Deterministic backtest failed -> Profitability is blocked, but directional bias remains BEARISH
    assert result.backtest_status == BacktestStatus.FAIL
    assert result.directional_bias == DirectionalBias.BEARISH
    assert result.thesis_status == ThesisStatus.NOT_ESTABLISHED
    assert "NO VALIDATED EDGE" in result.verdict
    assert any("Backtest Gate" in w for w in result.audit_warnings)
    assert result.reconciliation_status == "VALIDATED"
    # Even though gated to neutral, candidate supporting evidence (4 SELLs) remains recorded
    assert len(result.supporting_evidence) == 4

# -------------------------------------------------------------------------
# Acceptance Test 3: Conflicting strategy signals
# -------------------------------------------------------------------------
def test_acceptance_3_conflicting_strategy_signals():
    data = get_ohlcv_models()
    tester = TradeThesisStressTester(data, symbol="TSLA", timeframe="1d")

    # Contradicting signals: Trend & Breakout say BUY, Momentum & MeanReversion say SELL
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.BUY, confidence=0.8, metadata={}),
        StrategyResult(strategy_name="Breakout", signal=Signal.BUY, confidence=0.7, metadata={}),
        StrategyResult(strategy_name="Momentum", signal=Signal.SELL, confidence=0.8, metadata={"rsi": 78}),
        StrategyResult(strategy_name="MeanReversion", signal=Signal.SELL, confidence=0.8, metadata={}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
        StrategyResult(strategy_name="Risk", signal=Signal.NEUTRAL, confidence=0.8, metadata={}),
    ]

    result = tester.run_stress_test(strat_results)

    assert result.setup_status == SetupStatus.CONFLICTED
    assert result.directional_bias == DirectionalBias.NEUTRAL
    assert result.thesis_status == ThesisStatus.NOT_ESTABLISHED
    assert result.verdict == "NO VALIDATED EDGE: EVIDENCE CONFLICTED"
    assert any("Conflicted Evidence" in w for w in result.audit_warnings)
    assert result.reconciliation_status == "VALIDATED"

# -------------------------------------------------------------------------
# Acceptance Test 4: Duplicated / correlated evidence (False Consensus)
# -------------------------------------------------------------------------
def test_acceptance_4_duplicated_correlated_evidence():
    data = get_ohlcv_models()
    tester = TradeThesisStressTester(data, symbol="NVDA", timeframe="1d")

    # 2 indicators vote BUY, but both belong to the TREND family (Trend and Breakout)
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.BUY, confidence=0.8, metadata={}),
        StrategyResult(strategy_name="Breakout", signal=Signal.BUY, confidence=0.7, metadata={}),
        StrategyResult(strategy_name="Momentum", signal=Signal.NEUTRAL, confidence=0.5, metadata={}),
        StrategyResult(strategy_name="MeanReversion", signal=Signal.NEUTRAL, confidence=0.5, metadata={}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.NEUTRAL, confidence=0.0, metadata={}),
        StrategyResult(strategy_name="Risk", signal=Signal.NEUTRAL, confidence=0.8, metadata={}),
    ]

    result = tester.run_stress_test(strat_results)

    # Raw votes = 2 BUYs, but independent family votes = only 1 BULLISH family
    trend_family = [d for d in result.independent_evidence_families if d.family.value == "TREND"][0]
    assert len(trend_family.indicators) == 2
    assert "Trend" in trend_family.indicators and "Breakout" in trend_family.indicators
    assert trend_family.effective_weight == 1.0

    # Because only 1 independent family supported the trade, setup is UNCONFIRMED
    assert result.setup_status == SetupStatus.UNCONFIRMED
    assert "False Consensus Alert" in result.dependency_explanation

# -------------------------------------------------------------------------
# Acceptance Test 5: Insufficient out-of-sample trades (2-trade OOS rule)
# -------------------------------------------------------------------------
def test_acceptance_5_insufficient_out_of_sample_trades():
    # Test partition logic directly: a 2-trade OOS sample MUST be flagged as INSUFFICIENT_DATA
    engine = HistoricalSetupEngine([], config=HistoricalSetupConfig(min_oos_samples=5))
    
    # 2 trades with 100% win rate
    trades_2 = [
        {"bar_idx": 80, "net_ret": 0.04, "is_oos": True},
        {"bar_idx": 85, "net_ret": 0.03, "is_oos": True}
    ]

    oos_partition = engine._calc_partition(trades_2, "Out-Of-Sample", min_samples=5)

    assert oos_partition.sample_count == 2
    assert oos_partition.win_rate == 1.0
    # Strict gate: 2 trades < 5 required -> INSUFFICIENT_DATA
    assert oos_partition.status == BacktestStatus.INSUFFICIENT_DATA

# -------------------------------------------------------------------------
# Acceptance Test 6: Invalid or unavailable volume (Equity vs Index semantics)
# -------------------------------------------------------------------------
def test_acceptance_6_invalid_or_unavailable_volume():
    data = create_bullish_trending_data(60)
    # Zero out volume for 30% of bars
    for i in range(20):
        data[i] = OHLCV(
            timestamp=data[i].timestamp,
            open=data[i].open,
            high=data[i].high,
            low=data[i].low,
            close=data[i].close,
            volume=0
        )

    # 6a. Cash Index feed (e.g. ^NSEI) - Zero volume is normal and valid
    index_tester = TradeThesisStressTester(
        data, 
        symbol="^NSEI", 
        instrument=InstrumentMetadata(symbol="^NSEI", canonical_symbol="^NSEI", instrument_type="INDEX")
    )
    df_index = pd.DataFrame([c.model_dump() for c in data])
    index_quality = index_tester.audit_data_quality(df_index)
    assert index_quality.is_valid is True
    assert not any("zero volume recorded for equity" in w for w in index_quality.warnings)

    # 6b. Equity feed (e.g. AAPL) - Zero volume is an anomaly
    equity_tester = TradeThesisStressTester(
        data, 
        symbol="AAPL", 
        instrument=InstrumentMetadata(symbol="AAPL", canonical_symbol="AAPL", instrument_type="EQUITY")
    )
    df_equity = pd.DataFrame([c.model_dump() for c in data])
    equity_quality = equity_tester.audit_data_quality(df_equity)
    assert any("Data anomaly: 20 of 60 bars have zero volume recorded for equity" in w for w in equity_quality.warnings)

# -------------------------------------------------------------------------
# Acceptance Test 7: Mismatched consensus and ledger values
# -------------------------------------------------------------------------
def test_acceptance_7_mismatched_consensus_and_ledger_values():
    data = get_ohlcv_models()
    tester = TradeThesisStressTester(data, symbol="AAPL", timeframe="1d")

    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.BUY, confidence=0.8, metadata={}),
        StrategyResult(strategy_name="Breakout", signal=Signal.BUY, confidence=0.7, metadata={}),
    ]

    # Deliberately mismatched ledger counts
    reconciled, errors = tester.audit_reconciliation(
        strategy_results=strat_results,
        raw_votes={"BUY": 2, "SELL": 0, "NEUTRAL": 0},
        indep_votes={"BULLISH": 1, "BEARISH": 0, "NEUTRAL": 0, "CONFLICTED": 0},
        supporting=[], # Mismatched: expected 2
        opposing=[],
        neutral=[],
        candidate_bias=DirectionalBias.BULLISH,
        final_bias=DirectionalBias.BULLISH,
        backtest_status=BacktestStatus.PASS,
        setup_status=SetupStatus.CONFIRMED
    )

    assert reconciled is False
    assert any("Internal Reconciliation Error" in err for err in errors)

# -------------------------------------------------------------------------
# Acceptance Test 8: Missing, stale or incorrect market data
# -------------------------------------------------------------------------
def test_acceptance_8_missing_stale_or_incorrect_market_data():
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.BUY, confidence=0.8, metadata={}),
    ]

    # 8a: Stale / missing data (< 25 candles)
    data_short = get_ohlcv_models()[:15]
    tester_short = TradeThesisStressTester(data_short, symbol="STALE", timeframe="1d")
    result_short = tester_short.run_stress_test(strat_results)

    assert result_short.setup_status == SetupStatus.INVALID_DATA
    assert result_short.data_quality.is_valid is False
    assert result_short.verdict == "DATA BLOCKED: Insufficient or corrupted market data"
    assert result_short.reconciliation_status == "FAILED"

    # 8b: Corrupted negative price data
    data_corrupt = list(get_ohlcv_models()[:55])
    data_corrupt[-1] = OHLCV(
        timestamp=data_corrupt[-1].timestamp,
        open=-10.0,
        high=100.0,
        low=-20.0,
        close=-15.0,
        volume=1000
    )
    tester_corrupt = TradeThesisStressTester(data_corrupt, symbol="CORRUPT", timeframe="1d")
    result_corrupt = tester_corrupt.run_stress_test(strat_results)

    assert result_corrupt.data_quality.is_valid is False
    assert result_corrupt.data_quality.negative_price_detected is True
    assert result_corrupt.verdict == "DATA BLOCKED: Insufficient or corrupted market data"
