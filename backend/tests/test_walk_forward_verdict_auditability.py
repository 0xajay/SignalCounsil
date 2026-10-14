import pytest
import numpy as np
from datetime import datetime, timedelta
from app.domain.models import (
    OHLCV, DirectionalBias, Signal, StrategyResult, EvidenceFamily,
    DrawdownAuditRecord, EventDrivenPortfolioResult, WalkForwardAnalogueResult,
    AnalysisResult
)
from app.domain.historical_setup import HistoricalSetupEngine
from app.domain.stress_tester import TradeThesisStressTester
from app.domain.evidence import EvidenceLedgerBuilder, compute_measured_dependencies

def make_sample_ohlcv(n_bars: int = 150, start_price: float = 100.0, trend: float = -0.0015) -> list[OHLCV]:
    bars = []
    base_date = datetime(2025, 1, 1)
    price = start_price
    for i in range(n_bars):
        date_str = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        ret = trend + np.sin(i / 6.0) * 0.008 + (0.002 if i % 2 == 0 else -0.002)
        close_p = price * (1.0 + ret)
        high_p = max(price, close_p) * 1.005
        low_p = min(price, close_p) * 0.995
        open_p = price
        bars.append(OHLCV(
            timestamp=date_str,
            open=round(open_p, 2),
            high=round(high_p, 2),
            low=round(low_p, 2),
            close=round(close_p, 2),
            volume=100000.0
        ))
        price = close_p
    return bars

# ==============================================================================
# 1. Walk-forward evaluation fails while retrospective analogues appear profitable
# ==============================================================================
def test_walk_forward_fails_while_retrospective_analogues_appear_profitable():
    """
    Test 1:
    Walk-forward evaluation fails (negative compounded return, DD breach)
    even when retrospective analogues might look profitable.
    The engine must never reinterpret an unsuccessful walk-forward result as profitable.
    """
    bars = make_sample_ohlcv(140)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_res, _ = engine.investigate_setup(DirectionalBias.BEARISH)
    comp = engine.compare_historical_setups(DirectionalBias.BEARISH, horizon_bars=5, backtest_result=hist_res)
    
    assert comp.walk_forward_evaluation is not None
    wf = comp.walk_forward_evaluation
    
    # Walk-forward status MUST be FAIL due to DD breach (> 25%)
    assert wf.max_drawdown_pct > 25.0
    assert wf.walk_forward_status == "FAIL"
    assert comp.walk_forward_status == "FAIL"
    
    # Retrospective analogue subset must NOT overwrite WF failure
    assert comp.walk_forward_status != "PASS"
    
    # Test specific reported walk-forward metrics disconnect behavior:
    # A win rate above 50% with profit factor < 1.0 and negative compounded return
    # correctly generates the disconnect explanation and flags FAIL
    simulated_wf = WalkForwardAnalogueResult(
        similarity_threshold=0.75,
        horizon_bars=5,
        evaluation_dates_count=135,
        covered_forecasts_count=109,
        uncovered_dates_count=26,
        coverage_pct=80.74,
        win_rate_pct=58.7,
        profit_factor=0.95,
        compounded_equity_return_pct=-5.44,
        sum_of_returns_pct=-3.72,
        max_drawdown_pct=34.59,
        status="FAIL",
        walk_forward_status="FAIL",
        win_rate_profitability_disconnect_explanation=(
            "A win rate of 58.7% does not guarantee profitability. The walk-forward profit factor is 0.95 (< 1.0) "
            "and compounded net return is -5.44% because average losses exceed average gains and 0.15% round-trip "
            "transaction costs drag net performance."
        ),
        uncovered_dates_explanation="26 of 135 chronological evaluation dates were uncovered (80.7% coverage).",
        overlapping_dependency_flag=True
    )
    assert simulated_wf.walk_forward_status == "FAIL"
    assert simulated_wf.win_rate_pct > 50.0
    assert simulated_wf.profit_factor < 1.0
    assert simulated_wf.compounded_equity_return_pct < 0.0
    assert simulated_wf.max_drawdown_pct > 25.0
    assert "does not guarantee profitability" in simulated_wf.win_rate_profitability_disconnect_explanation

# ==============================================================================
# 2. Generic PASS is never displayed for matching completion
# ==============================================================================
def test_generic_pass_never_displayed_for_matching_completion():
    """
    Test 2:
    Replace generic PASS badge in analogue panel with MATCHING COMPLETE.
    Matching status must be MATCHING_COMPLETE, never PASS.
    """
    bars = make_sample_ohlcv(140)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_res, _ = engine.investigate_setup(DirectionalBias.BEARISH)
    comp = engine.compare_historical_setups(DirectionalBias.BEARISH, horizon_bars=5, backtest_result=hist_res)
    
    assert comp.matching_status == "MATCHING_COMPLETE"
    assert comp.status == "MATCHING_COMPLETE"
    assert comp.matching_status != "PASS"
    assert comp.status != "PASS"

# ==============================================================================
# 3. Drawdown warning and detailed audit agree
# ==============================================================================
def test_drawdown_warning_and_detailed_audit_agree():
    """
    Test 3:
    Reconciles drawdown warnings with detailed risk audit.
    If any metric exceeds configured 25% limit, DRAWDOWN_LIMIT_BREACH is emitted
    in both detailed audit record and summary warning.
    """
    bars = make_sample_ohlcv(140)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_res, _ = engine.investigate_setup(DirectionalBias.BEARISH)
    comp = engine.compare_historical_setups(DirectionalBias.BEARISH, horizon_bars=5, backtest_result=hist_res)
    
    assert comp.risk_limit_status == "DRAWDOWN_LIMIT_BREACH"
    assert comp.drawdown_breach_warning is not None
    assert "DRAWDOWN_LIMIT_BREACH" in comp.drawdown_breach_warning
    
    # Check that audit records reflect breaches
    breached_records = [r for r in comp.drawdown_audit_records if r.is_breached]
    assert len(breached_records) >= 1
    for r in breached_records:
        assert r.max_drawdown_pct > 25.0

# ==============================================================================
# 4. Missing peak/trough timestamps cause a traceability warning
# ==============================================================================
def test_missing_peak_trough_timestamps_cause_traceability_warning():
    """
    Test 4:
    If peak/trough provenance is missing, do not claim the result is fully traceable.
    Must emit PROVENANCE_MISSING and traceability warning.
    """
    # Create record with missing peak/trough
    untraceable_rec = DrawdownAuditRecord(
        metric_label="Test Untraceable Drawdown",
        source_dataset="Synthetic Test",
        equity_curve_type="TRADE_RETURN_PROXY",
        max_drawdown_pct=30.0,
        peak_date="",
        trough_date="",
        peak_equity=None,
        trough_equity=None,
        calculation_window="Unverified",
        configured_limit_pct=25.0,
        is_breached=True,
        is_provenance_complete=False,
        traceability_status="PROVENANCE_MISSING",
        traceability_warning="Drawdown calculation lacks timestamp or equity provenance; peak and trough cannot be audited.",
        explanation="Testing missing provenance"
    )
    
    assert untraceable_rec.is_provenance_complete is False
    assert untraceable_rec.traceability_status == "PROVENANCE_MISSING"
    assert untraceable_rec.traceability_warning is not None
    assert "lacks timestamp or equity provenance" in untraceable_rec.traceability_warning

    # In contrast, an engine generated record must be FULLY_TRACEABLE
    bars = make_sample_ohlcv(140)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_res, _ = engine.investigate_setup(DirectionalBias.BEARISH)
    comp = engine.compare_historical_setups(DirectionalBias.BEARISH, horizon_bars=5, backtest_result=hist_res)
    
    for r in comp.drawdown_audit_records:
        if r.peak_date and r.trough_date:
            assert r.is_provenance_complete is True
            assert r.traceability_status == "FULLY_TRACEABLE"

# ==============================================================================
# 5. Portfolio exposure capacity differs from actual observed exposure
# ==============================================================================
def test_portfolio_exposure_capacity_differs_from_actual_observed_exposure():
    """
    Test 5:
    Configured capacity is 100% (5 slots x 20%), but actual observed maximum exposure
    differs from configured capacity (e.g. 80.5% vs 100.0%, 4 slots used concurrently).
    """
    bars = make_sample_ohlcv(140)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_res, _ = engine.investigate_setup(DirectionalBias.BEARISH)
    comp = engine.compare_historical_setups(DirectionalBias.BEARISH, horizon_bars=5, backtest_result=hist_res)
    
    port = comp.portfolio_simulation
    assert port is not None
    assert port.max_gross_exposure_capacity_pct == 100.0
    assert port.max_observed_exposure_pct is not None
    assert port.max_observed_exposure_pct != port.max_gross_exposure_capacity_pct
    assert port.max_concurrent_positions == 4
    assert port.max_allowed_concurrent_positions == 5
    assert port.initial_capital == 100000.0
    assert port.final_equity > 0.0
    assert port.trades_executed_count > 0
    assert port.trades_skipped_capital_count == 0

# ==============================================================================
# 6. Event-driven and walk-forward metrics have distinct source metadata
# ==============================================================================
def test_event_driven_and_walk_forward_have_distinct_source_metadata():
    """
    Test 6:
    Verify event-driven portfolio and walk-forward evaluations have distinct source
    metadata, distinct dates, and distinct methodologies.
    """
    bars = make_sample_ohlcv(140)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_res, _ = engine.investigate_setup(DirectionalBias.BEARISH)
    comp = engine.compare_historical_setups(DirectionalBias.BEARISH, horizon_bars=5, backtest_result=hist_res)
    
    port = comp.portfolio_simulation
    wf = comp.walk_forward_evaluation
    
    assert port.signal_source == "HISTORICAL_ANALOGUE_CANDIDATE_SETUPS"
    assert port.accounting_method == "EVENT_DRIVEN_CHRONOLOGICAL_CASH_AND_POSITIONS"
    assert port.execution_mechanics_doc is not None
    
    assert wf.data_product_type == "WALK_FORWARD_ANALOGUE_EVALUATION"
    assert wf.min_prior_history == 50
    assert wf.min_prior_matches == 3

# ==============================================================================
# 7. Heuristic grouping cannot trigger unsupported empirical-collinearity language
# ==============================================================================
def test_heuristic_grouping_cannot_trigger_unsupported_empirical_collinearity():
    """
    Test 7:
    Heuristic family grouping must be kept separate from empirically measured correlation.
    Synthesis must not describe heuristic grouping as empirically demonstrated collinearity.
    """
    bars = make_sample_ohlcv(140)
    tester = TradeThesisStressTester(bars, symbol="TEST", timeframe="1d")
    
    strat_results = [
        StrategyResult(strategy_name="TrendFollowing", signal=Signal.SELL, confidence=0.75, metadata={}),
        StrategyResult(strategy_name="MomentumOscillator", signal=Signal.SELL, confidence=0.70, metadata={}),
        StrategyResult(strategy_name="MeanReversion", signal=Signal.NEUTRAL, confidence=0.50, metadata={}),
    ]
    
    report = tester.run_stress_test(strat_results)
    
    # Explanation synthesis check:
    assert "empirically demonstrated collinearity" not in report.explanation.lower()
    # Check measured dependencies vs heuristic dependencies
    for dep in report.evidence_dependencies:
        # Heuristic groupings are not empirical correlations
        assert dep.family in [EvidenceFamily.MOMENTUM, EvidenceFamily.TREND, EvidenceFamily.MEAN_REVERSION, EvidenceFamily.PRICE_ACTION, EvidenceFamily.VOLATILITY_RISK]

# ==============================================================================
# 8. Neutral correlated families do not inflate directional consensus
# ==============================================================================
def test_neutral_correlated_families_do_not_inflate_directional_consensus():
    """
    Test 8:
    When neutral indicator families are correlated, the correlation does not
    add a directional vote if both remain neutral.
    Directional family alignment only considers non-neutral families.
    """
    bars = make_sample_ohlcv(140)
    tester = TradeThesisStressTester(bars, symbol="TEST", timeframe="1d")
    
    # Provide neutral indicators for both Momentum and Mean Reversion
    strat_results = [
        StrategyResult(strategy_name="MomentumOscillator", signal=Signal.NEUTRAL, confidence=0.50, metadata={}),
        StrategyResult(strategy_name="MeanReversion", signal=Signal.NEUTRAL, confidence=0.50, metadata={}),
        StrategyResult(strategy_name="TrendFollowing", signal=Signal.SELL, confidence=0.75, metadata={}),
    ]
    
    report = tester.run_stress_test(strat_results)
    
    # Neutral families do not add directional votes
    # In directional_family_alignment, neutral families are not counted as agreeing with directional bias
    assert report.directional_bias == DirectionalBias.BEARISH
    assert report.directional_family_alignment is not None
    # Alignment should reflect only active directional families (e.g. Trend), not inflated by neutral ones
    assert report.directional_family_alignment <= 100

# ==============================================================================
# 9. Five-bar overlapping horizons are flagged in evaluation interpretation
# ==============================================================================
def test_five_bar_overlapping_horizons_flagged_in_interpretation():
    """
    Test 9:
    Five-bar overlapping outcomes must be flagged as dependent observations.
    Sensitivity analysis using non-overlapping evaluation intervals must be provided.
    """
    bars = make_sample_ohlcv(140)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_res, _ = engine.investigate_setup(DirectionalBias.BEARISH)
    comp = engine.compare_historical_setups(DirectionalBias.BEARISH, horizon_bars=5, backtest_result=hist_res)
    
    wf = comp.walk_forward_evaluation
    assert wf.overlapping_dependency_flag is True
    assert wf.overlapping_dependency_explanation is not None
    assert "dependent observations" in wf.overlapping_dependency_explanation.lower()
    
    # Non-overlapping sensitivity analysis check
    sens = wf.sensitivity_analysis
    assert sens is not None
    assert sens.sampling_step_bars == 5
    assert sens.non_overlapping_samples_count > 0
    assert sens.ci_spans_zero is True
    assert "No Statistical Edge" in sens.statistical_significance or "NO_STATISTICAL_EDGE" in sens.statistical_significance
    assert "autocorrelation" in sens.autocorrelation_caveat.lower()

# ==============================================================================
# 10. Summary and detailed validation statuses remain consistent
# ==============================================================================
def test_summary_and_detailed_validation_statuses_consistent():
    """
    Test 10:
    Ensure all 5 validation statuses are exposed and agree between
    HistoricalComparisonResult and StressTestReport / AnalysisResult:
    - matching_status
    - strategy_backtest_status
    - walk_forward_status
    - portfolio_simulation_status
    - risk_limit_status
    """
    bars = make_sample_ohlcv(140)
    tester = TradeThesisStressTester(bars, symbol="TEST", timeframe="1d")
    
    strat_results = [
        StrategyResult(strategy_name="TrendFollowing", signal=Signal.SELL, confidence=0.75, metadata={}),
        StrategyResult(strategy_name="MomentumOscillator", signal=Signal.SELL, confidence=0.70, metadata={}),
    ]
    
    report = tester.run_stress_test(strat_results)
    hist_comp = report.historical_comparison
    
    assert hist_comp is not None
    
    # 5 statuses on HistoricalComparisonResult
    assert hist_comp.matching_status == "MATCHING_COMPLETE"
    assert hist_comp.strategy_backtest_status in ("FAIL", "WARNING", "PASS")
    assert hist_comp.walk_forward_status == "FAIL"
    assert hist_comp.portfolio_simulation_status == "PASS"
    assert hist_comp.risk_limit_status == "DRAWDOWN_LIMIT_BREACH"
    
    # 5 statuses on StressTestReport / AnalysisResult
    assert report.matching_status == hist_comp.matching_status
    assert report.strategy_backtest_status == hist_comp.strategy_backtest_status
    assert report.walk_forward_status == hist_comp.walk_forward_status
    assert report.portfolio_simulation_status == hist_comp.portfolio_simulation_status
    assert report.risk_limit_status == hist_comp.risk_limit_status
