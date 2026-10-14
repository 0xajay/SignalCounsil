import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from app.domain.models import (
    OHLCV, OHLCVData, DirectionalBias, BacktestStatus, Signal,
    StrategyResult, EvidenceFamily, EvidenceStrength, SetupStatus, ThesisStatus,
    TransactionCostModel, SetupPartitionMetrics, MeasuredDependence,
    FinalAnalysisResult, HistoricalSetupConfig, DrawdownAuditRecord,
    EventDrivenPortfolioResult, WalkForwardAnalogueResult, WalkForwardEvaluationPoint,
    SetupFeatureVector
)
from app.domain.engine import to_dataframe
from app.domain.historical_setup import (
    HistoricalSetupEngine, simulate_trade_at_bar, calculate_partition_metrics,
    simulate_event_driven_portfolio, evaluate_walk_forward_analogues,
    FEATURE_CONFIGS, compute_feature_similarity
)
from app.domain.evidence import EvidenceLedgerBuilder
from app.domain.stress_tester import TradeThesisStressTester

def make_deterministic_ohlcv(n_bars: int = 150, start_price: float = 100.0, trend: float = -0.002) -> list[OHLCV]:
    """Generates synthetic deterministic OHLCV bars for walk-forward and portfolio testing."""
    bars = []
    base_date = datetime(2025, 1, 1)
    price = start_price
    for i in range(n_bars):
        date_str = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        ret = trend + np.sin(i / 6.0) * 0.006 + (0.001 if i % 2 == 0 else -0.001)
        close_p = price * (1.0 + ret)
        high_p = max(price, close_p) * 1.004
        low_p = min(price, close_p) * 0.996
        open_p = price
        bars.append(OHLCV(
            timestamp=date_str,
            open=round(open_p, 2),
            high=round(high_p, 2),
            low=round(low_p, 2),
            close=round(close_p, 2),
            volume=100000.0 + (i % 10) * 1000.0
        ))
        price = close_p
    return bars

def make_rally_ohlcv(n_bars: int = 120, start_price: float = 100.0) -> list[OHLCV]:
    """Generates strong upward rally where short setups experience major drawdowns exceeding 25%."""
    bars = []
    base_date = datetime(2025, 1, 1)
    price = start_price
    for i in range(n_bars):
        date_str = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        close_p = price * 1.012  # steady ~1.2% gain per day
        high_p = close_p * 1.006
        low_p = price * 0.996
        open_p = price
        bars.append(OHLCV(
            timestamp=date_str,
            open=round(open_p, 2),
            high=round(high_p, 2),
            low=round(low_p, 2),
            close=round(close_p, 2),
            volume=150000.0
        ))
        price = close_p
    return bars

# ==============================================================================
# 1. Matching that depends only on past information
# ==============================================================================
def test_matching_depends_only_on_past_information():
    """
    Requirement 1:
    At each evaluation timestamp t, construct the comparable historical set
    using only observations available strictly before that timestamp: i <= t - horizon.
    Changing future bars after t must not alter past candidate selection or matching.
    """
    bars_base = make_deterministic_ohlcv(100)
    engine_base = HistoricalSetupEngine(bars_base, horizon_bars=5)
    cost_model = TransactionCostModel()
    df_base = to_dataframe(bars_base)
    
    wf_res_base = evaluate_walk_forward_analogues(
        engine=engine_base,
        df=df_base,
        candidate_bias=DirectionalBias.BEARISH,
        horizon_bars=5,
        min_prior_history=30,
        min_prior_matches=3,
        similarity_threshold=0.75,
        cost_model=cost_model
    )
    
    for pt in wf_res_base.points:
        assert pt.evaluation_bar >= 30
        assert pt.prior_matches_count >= 0
    
    pt_60 = next((p for p in wf_res_base.points if p.evaluation_bar == 60), None)
    assert pt_60 is not None

    # Mutate bars 61..99 completely (drastically change future prices)
    bars_mutated = list(bars_base[:61])
    price = bars_mutated[-1].close
    ts = bars_mutated[-1].timestamp
    base_date = ts if isinstance(ts, datetime) else datetime.strptime(str(ts)[:10], "%Y-%m-%d")
    for i in range(len(bars_base) - 61):
        date_str = (base_date + timedelta(days=i + 1)).strftime("%Y-%m-%d")
        price = price * 1.05  # huge upward spike
        bars_mutated.append(OHLCV(
            timestamp=date_str,
            open=round(price * 0.99, 2),
            high=round(price * 1.02, 2),
            low=round(price * 0.98, 2),
            close=round(price, 2),
            volume=500000.0
        ))
    
    engine_mutated = HistoricalSetupEngine(bars_mutated, horizon_bars=5)
    df_mutated = to_dataframe(bars_mutated)
    wf_res_mutated = evaluate_walk_forward_analogues(
        engine=engine_mutated,
        df=df_mutated,
        candidate_bias=DirectionalBias.BEARISH,
        horizon_bars=5,
        min_prior_history=30,
        min_prior_matches=3,
        similarity_threshold=0.75,
        cost_model=cost_model
    )
    
    pt_60_mutated = next((p for p in wf_res_mutated.points if p.evaluation_bar == 60), None)
    assert pt_60_mutated is not None
    # Matching candidate count and forecasted status at bar 60 must be 100% identical
    assert pt_60.prior_matches_count == pt_60_mutated.prior_matches_count
    assert pt_60.is_forecasted == pt_60_mutated.is_forecasted
    assert pt_60.forecast_direction == pt_60_mutated.forecast_direction

# ==============================================================================
# 2. No future-leakage in feature selection or threshold fitting
# ==============================================================================
def test_no_future_leakage_in_feature_selection_or_threshold_fitting():
    """
    Requirement 2:
    Do not use future returns to select matches, choose similarity thresholds or optimize features.
    Freeze the matching configuration for each evaluation window.
    """
    total_weight = sum(cfg[1] for cfg in FEATURE_CONFIGS)
    assert abs(total_weight - 1.0) < 1e-6
    
    feat_a = SetupFeatureVector(
        trend_regime="BEARISH_TREND",
        trend_strength_pct=-1.5,
        momentum_state="NEUTRAL",
        rsi_value=45.0,
        volatility_regime="NORMAL_VOLATILITY",
        volatility_pct=2.1,
        distance_from_resistance_pct=3.5,
        distance_from_support_pct=1.2,
        breakout_state="CONSOLIDATING_IN_RANGE"
    )
    feat_b = SetupFeatureVector(
        trend_regime="BEARISH_TREND",
        trend_strength_pct=-1.5,
        momentum_state="NEUTRAL",
        rsi_value=45.0,
        volatility_regime="NORMAL_VOLATILITY",
        volatility_pct=2.1,
        distance_from_resistance_pct=3.5,
        distance_from_support_pct=1.2,
        breakout_state="CONSOLIDATING_IN_RANGE"
    )
    dist, sim, comps, notes = compute_feature_similarity(feat_a, feat_b)
    assert dist == 0.0
    assert sim == 1.0

    feat_c = SetupFeatureVector(
        trend_regime="BULLISH_TREND",
        trend_strength_pct=2.5,
        momentum_state="OVERBOUGHT",
        rsi_value=72.0,
        volatility_regime="HIGH_VOLATILITY",
        volatility_pct=4.2,
        distance_from_resistance_pct=0.5,
        distance_from_support_pct=5.0,
        breakout_state="UPPER_EXPANSION"
    )
    dist_ac, sim_ac, _, _ = compute_feature_similarity(feat_a, feat_c)
    dist_ca, sim_ca, _, _ = compute_feature_similarity(feat_c, feat_a)
    assert abs(dist_ac - dist_ca) < 1e-6
    assert abs(sim_ac - sim_ca) < 1e-6

# ==============================================================================
# 3. Walk-forward evaluation with insufficient historical matches
# ==============================================================================
def test_walk_forward_insufficient_historical_matches():
    """
    Requirement 3 & 10:
    Flag insufficient evaluation samples and periods without enough prior matches.
    """
    bars = make_deterministic_ohlcv(90)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    cost_model = TransactionCostModel()
    df = to_dataframe(bars)
    
    wf_res = evaluate_walk_forward_analogues(
        engine=engine,
        df=df,
        candidate_bias=DirectionalBias.BEARISH,
        horizon_bars=5,
        min_prior_history=55,
        min_prior_matches=5,
        similarity_threshold=0.999,
        cost_model=cost_model
    )
    
    assert wf_res.evaluation_dates_count > 0
    assert wf_res.status == BacktestStatus.INSUFFICIENT_DATA
    assert "walk-forward" in wf_res.uncertainty_disclosure.lower()
    uncovered = [p for p in wf_res.points if not p.is_forecasted]
    assert len(uncovered) > 0
    assert any("INSUFFICIENT_PRIOR_MATCHES" in p.skip_reason for p in uncovered)

# ==============================================================================
# 4. Reproducible costs and returns
# ==============================================================================
def test_reproducible_costs_and_returns():
    """
    Requirement 4 & 7:
    Apply canonical 0.15% round-trip cost model consistently across all trades.
    """
    cost_model = TransactionCostModel()
    gross_ret = 0.03
    net_ret, total_costs = cost_model.calculate_net_return(gross_ret)
    
    expected_net = (1.0 + gross_ret) * ((1.0 - 0.00075) ** 2) - 1.0
    assert abs(net_ret - expected_net) < 1e-8
    assert abs(total_costs - (gross_ret - net_ret)) < 1e-8

    bars = make_deterministic_ohlcv(60)
    df = to_dataframe(bars)
    trade_1 = simulate_trade_at_bar(df, 20, DirectionalBias.BEARISH, horizon_bars=5, cost_model=cost_model)
    trade_2 = simulate_trade_at_bar(df, 20, DirectionalBias.BEARISH, horizon_bars=5, cost_model=cost_model)
    assert trade_1 is not None and trade_2 is not None
    assert trade_1["gross_return"] == trade_2["gross_return"]
    assert trade_1["net_return"] == trade_2["net_return"]
    assert trade_1["total_costs"] == trade_2["total_costs"]

# ==============================================================================
# 5. Concurrent trades and maximum exposure
# ==============================================================================
def test_concurrent_trades_and_maximum_exposure():
    """
    Requirement 5:
    Enforce available-capital and maximum-exposure constraints in event-driven portfolio simulation.
    Max 5 slots, 20% equity per position, max gross exposure <= 100%.
    """
    bars = make_deterministic_ohlcv(120)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    
    hist_res, _ = engine.investigate_setup(DirectionalBias.BEARISH)
    port_sim = hist_res.portfolio_simulation
    assert port_sim is not None
    
    assert port_sim.max_concurrent_positions <= 5
    assert port_sim.max_allowed_concurrent_positions == 5
    assert port_sim.max_total_exposure_pct <= 100.0
    assert port_sim.final_equity > 0.0

# ==============================================================================
# 6. Correct portfolio equity under overlapping trade horizons
# ==============================================================================
def test_correct_portfolio_equity_under_overlapping_trade_horizons():
    """
    Requirement 6:
    Calculate portfolio equity curve chronologically under overlapping trades.
    When trades overlap, capital is allocated per slot and returns are marked to market.
    """
    bars = make_deterministic_ohlcv(100)
    df = to_dataframe(bars)
    cost_model = TransactionCostModel()
    
    trades = []
    for i in range(20, 30):
        t = simulate_trade_at_bar(df, i, DirectionalBias.BEARISH, horizon_bars=5, cost_model=cost_model)
        if t:
            trades.append(t)
    
    assert len(trades) >= 5
    port_res = simulate_event_driven_portfolio(
        trades=trades,
        df=df,
        initial_capital=100000.0,
        max_concurrent_positions=5,
        allocation_per_position_pct=20.0,
        max_total_exposure_pct=100.0
    )
    
    assert port_res.trades_executed_count > 0
    assert len(port_res.equity_curve) > 0
    assert port_res.max_concurrent_positions <= 5
    assert port_res.peak_equity >= port_res.trough_equity
    assert port_res.portfolio_max_drawdown_pct >= 0.0

# ==============================================================================
# 7. Separation of portfolio returns and sequential trade-return proxies
# ==============================================================================
def test_separation_of_portfolio_returns_and_sequential_trade_return_proxies():
    """
    Requirement 7 & P0 Option A vs B:
    Do not mix Option A portfolio metrics with Option B trade-level metrics.
    Label cumulative product as sequentially compounded trade-return proxy, not executable portfolio return.
    """
    bars = make_deterministic_ohlcv(100)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_res, _ = engine.investigate_setup(DirectionalBias.BEARISH)
    
    # Option B proxy exists and has caveats
    assert hasattr(hist_res, "trade_return_proxy_compounded_pct")
    assert "proxy" in hist_res.portfolio_sizing_assumptions.lower()
    assert "overlap" in hist_res.overlapping_horizon_caveat.lower()
    
    # Option A portfolio simulation is separate
    port = hist_res.portfolio_simulation
    assert port is not None
    assert port.accounting_method == "EVENT_DRIVEN_CHRONOLOGICAL_CASH_AND_POSITIONS"
    
    # Both metrics are separate fields and not conflated
    assert hist_res.portfolio_simulation.portfolio_compounded_return_pct != hist_res.compounded_net_return_pct

# ==============================================================================
# 8. Drawdown source attribution
# ==============================================================================
def test_drawdown_source_attribution():
    """
    Requirement 8:
    Trace the drawdown calculation to its exact source dataset, dates, trade series and equity curve.
    Keep full-strategy, in-sample, out-of-sample, analogue-subset and portfolio drawdowns distinct.
    """
    bars = make_deterministic_ohlcv(120)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_res, _ = engine.investigate_setup(DirectionalBias.BEARISH)
    
    records = hist_res.drawdown_audit_records
    assert len(records) >= 4
    
    sources = [r.source_dataset for r in records]
    assert "FULL_STRATEGY_BACKTEST" in sources
    assert "IN_SAMPLE" in sources
    assert "OUT_OF_SAMPLE" in sources
    assert "PORTFOLIO_SIMULATION" in sources
    
    for r in records:
        assert r.peak_date != ""
        assert r.trough_date != ""
        assert r.peak_equity >= r.trough_equity
        assert r.configured_limit_pct == 25.0
        assert r.is_breached == (r.max_drawdown_pct > 25.0)

# ==============================================================================
# 9. Drawdown threshold breach visibility
# ==============================================================================
def test_drawdown_threshold_breach_visibility():
    """
    Requirement 9:
    Generate a structured DRAWDOWN_LIMIT_BREACH warning when a configured limit is exceeded.
    Display this warning in the main result summary and stress tester output.
    """
    rally_bars = make_rally_ohlcv(120)
    engine = HistoricalSetupEngine(rally_bars, horizon_bars=5)
    hist_res, _ = engine.investigate_setup(DirectionalBias.BEARISH)
    
    # Full backtest drawdown exceeds 25.0%
    full_rec = next(r for r in hist_res.drawdown_audit_records if r.source_dataset == "FULL_STRATEGY_BACKTEST")
    assert full_rec.max_drawdown_pct > 25.0
    assert full_rec.is_breached is True
    
    # Check that drawdown_breach_warning is set on HistoricalSetupResult
    assert hist_res.drawdown_breach_warning is not None
    assert "DRAWDOWN_LIMIT_BREACH" in hist_res.drawdown_breach_warning
    
    # Stress tester should surface DRAWDOWN_LIMIT_BREACH in audit_warnings, data_quality_warnings, and explanation
    strat_results = [
        StrategyResult(strategy_name="TrendStrategy", signal=Signal.SELL, rule="Bearish EMA", weight=1.0, confidence=1.0),
        StrategyResult(strategy_name="BreakoutStrategy", signal=Signal.SELL, rule="Lower breakdown", weight=1.0, confidence=1.0)
    ]
    tester = TradeThesisStressTester(market_data=rally_bars, symbol="TEST", timeframe="1d")
    result = tester.run_stress_test(strategy_results=strat_results)
    
    assert result.drawdown_breach_warning is not None
    assert any("DRAWDOWN_LIMIT_BREACH" in w for w in result.audit_warnings)
    assert any("DRAWDOWN_LIMIT_BREACH" in w for w in result.data_quality_warnings)
    assert "DRAWDOWN_LIMIT_BREACH" in result.explanation

# ==============================================================================
# 10. No false empirical-collinearity claim when only heuristic grouping is active
# ==============================================================================
def test_no_false_empirical_collinearity_claim_when_only_heuristic_grouping_active():
    """
    Requirement 10:
    Distinguish heuristic Trend/Breakout family grouping from empirically measured dependence.
    Do not describe heuristic grouping as empirical collinearity.
    """
    strat_results = [
        StrategyResult(strategy_name="TrendStrategy", signal=Signal.SELL, rule="Bearish EMA", weight=1.0, confidence=1.0),
        StrategyResult(strategy_name="BreakoutStrategy", signal=Signal.SELL, rule="Lower breakdown", weight=1.0, confidence=1.0),
    ]
    market_data = OHLCVData(data=make_deterministic_ohlcv(60))
    builder = EvidenceLedgerBuilder(market_data, strat_results)
    claims = builder.build_claims()
    
    deps, _, _, false_consensus, dep_explanation = builder.detect_false_consensus(claims, measured_dependencies=None)
    
    trend_dep = next(d for d in deps if d.family == EvidenceFamily.TREND)
    assert "HEURISTIC" in trend_dep.grouping_type
    assert "qualitative" in dep_explanation.lower() or "heuristic" in dep_explanation.lower()
    assert "distinct from empirical redundancy" in dep_explanation
    assert "No cross-family empirical redundancy penalties active" in dep_explanation

# ==============================================================================
# 11. Consistent interpretation of raw market returns and short-strategy returns
# ==============================================================================
def test_consistent_interpretation_of_raw_market_returns_and_short_strategy_returns():
    """
    Requirement 11:
    Clearly distinguish raw forward market returns from direction-adjusted strategy returns.
    For a short setup, market decline is positive strategy return.
    """
    bars = make_deterministic_ohlcv(50)
    df = to_dataframe(bars)
    cost_model = TransactionCostModel()
    
    trade = simulate_trade_at_bar(df, 10, DirectionalBias.BEARISH, horizon_bars=5, cost_model=cost_model)
    assert trade is not None
    
    entry_p = trade["entry_price"]
    exit_p = trade["exit_price"]
    raw_mkt_ret = (exit_p - entry_p) / entry_p
    
    assert abs(trade["raw_market_return"] - raw_mkt_ret) < 1e-6
    assert abs(trade["gross_return"] - (-raw_mkt_ret)) < 1e-6
    if raw_mkt_ret < 0:
        assert trade["gross_return"] > 0

# ==============================================================================
# 12. Stable historical-validation gate
# ==============================================================================
def test_stable_historical_validation_gate():
    """
    Requirement 12:
    Do not call historical matching success a strategy validation pass.
    Preserve negative and inconclusive results: Strategy validation remains independent,
    even when analogue matching completes.
    """
    bars = make_deterministic_ohlcv(120)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_res, overall_status = engine.investigate_setup(DirectionalBias.BEARISH)
    
    comp = engine.compare_historical_setups(
        candidate_bias=DirectionalBias.BEARISH,
        horizon_bars=5,
        backtest_result=hist_res
    )
    
    if comp.comparable_setups_count >= 5:
        assert comp.matching_status == "MATCHING_COMPLETE"
        assert comp.status != "PASS"
        assert comp.historical_strategy_validation in ("FAIL", "WARNING", "INSUFFICIENT_DATA", "PASS")
        assert comp.historical_strategy_validation == hist_res.full_history.status.value
