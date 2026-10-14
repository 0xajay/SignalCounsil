import pytest
import numpy as np
from datetime import datetime, timedelta
from app.domain.models import (
    OHLCV, DirectionalBias, Signal, StrategyResult, EvidenceFamily,
    PortfolioEntryRecord, BenchmarkComparisonResult, OffsetSensitivityRun,
    EventDrivenPortfolioResult, WalkForwardAnalogueResult, WalkForwardSensitivityAnalysis,
    AnalysisResult
)
from app.domain.historical_setup import (
    HistoricalSetupEngine, verify_signal_causality,
    evaluate_walk_forward_analogues, simulate_event_driven_portfolio
)
from app.domain.stress_tester import TradeThesisStressTester
from app.domain.evidence import EvidenceLedgerBuilder

def make_sample_ohlcv(n_bars: int = 150, start_price: float = 100.0, trend: float = 0.001) -> list[OHLCV]:
    bars = []
    base_date = datetime(2025, 1, 1)
    price = start_price
    for i in range(n_bars):
        date_str = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        ret = trend + np.sin(i / 5.0) * 0.01 + (0.002 if i % 2 == 0 else -0.002)
        close_p = price * (1.0 + ret)
        high_p = max(price, close_p) * 1.008
        low_p = min(price, close_p) * 0.992
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
# 1. Future-leakage injection test
# ==============================================================================
def test_future_leakage_injection_detected():
    """
    Test 1: Intentionally inject future leakage into PortfolioEntryRecord entries
    and prove that verify_signal_causality detects and flags it.
    """
    clean_records = [
        PortfolioEntryRecord(
            entry_id="rec_1",
            decision_timestamp="2025-02-01",
            decision_bar_idx=20,
            entry_timestamp="2025-02-02",
            entry_bar_idx=21,
            exit_timestamp="2025-02-07",
            exit_bar_idx=26,
            holding_bars=5,
            feature_snapshot={"trend": 1.2, "rsi": 55.0},
            eligible_historical_memory=20,
            matching_threshold=0.75,
            selected_analogue_ids=["a1", "a2"],
            generated_signal="LONG",
            parameter_version="v1.0",
            causality_status="PROVEN_CAUSAL_ZERO_LOOKAHEAD",
            lookahead_detected=False
        )
    ]
    res_clean = verify_signal_causality(clean_records, horizon_bars=5)
    assert res_clean["is_causal"] is True
    assert res_clean["causality_status"] == "PROVEN_CAUSAL_ZERO_LOOKAHEAD"
    assert len(res_clean["violations"]) == 0

    # Leakage Injection A: Memory includes future bars (memory > decision_bar_idx)
    leak_records_a = [
        clean_records[0].model_copy(update={
            "eligible_historical_memory": 35  # > decision_bar_idx (20)
        })
    ]
    res_a = verify_signal_causality(leak_records_a, horizon_bars=5)
    assert res_a["is_causal"] is False
    assert res_a["causality_status"] == "FUTURE_LEAKAGE_DETECTED"
    assert len(res_a["violations"]) > 0

    # Leakage Injection B: Decision made after entry bar (timing causality breach)
    leak_records_b = [
        clean_records[0].model_copy(update={
            "decision_bar_idx": 25,
            "entry_bar_idx": 21
        })
    ]
    res_b = verify_signal_causality(leak_records_b, horizon_bars=5)
    assert res_b["is_causal"] is False
    assert res_b["causality_status"] == "FUTURE_LEAKAGE_DETECTED"
    assert len(res_b["violations"]) > 0

    # Leakage Injection C: Future feature timestamp leakage
    leak_records_c = [
        clean_records[0].model_copy(update={
            "feature_snapshot": {"timestamp": "2025-02-15", "trend": 1.2, "uses_future_data": True}
        })
    ]
    res_c = verify_signal_causality(leak_records_c, horizon_bars=5)
    assert res_c["is_causal"] is False
    assert res_c["causality_status"] == "FUTURE_LEAKAGE_DETECTED"
    assert len(res_c["violations"]) > 0

# ==============================================================================
# 2. Retrospective candidates vs causal walk-forward-generated signals
# ==============================================================================
def test_retrospective_candidates_vs_causal_walk_forward():
    """
    Test 2: Verify that simulation distinguishes retrospective candidate selection
    from causal walk-forward signal generation. Retrospective candidate selection
    must be labeled EXPLORATORY_HISTORICAL_SIMULATION.
    """
    import pandas as pd
    bars = make_sample_ohlcv(140)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_res, _ = engine.investigate_setup(DirectionalBias.BULLISH)
    comp = engine.compare_historical_setups(DirectionalBias.BULLISH, horizon_bars=5, backtest_result=hist_res)
    
    assert comp.portfolio_simulation is not None
    port = comp.portfolio_simulation
    
    # Mode must explicitly disclose retrospective analogue selection
    assert port.signal_causality_mode == "RETROSPECTIVE_ANALOGUE_SELECTION"
    assert port.simulation_execution_status == "EXPLORATORY_HISTORICAL_SIMULATION"
    assert "retrospective" in port.explanation.lower()
    
    # Causal Walk-Forward Evaluation must remain distinct
    assert comp.walk_forward_evaluation is not None
    assert comp.walk_forward_evaluation.data_product_type == "WALK_FORWARD_ANALOGUE_EVALUATION"
    
    # Causal walk-forward generation mode label test
    df = pd.DataFrame([b.model_dump() for b in bars])
    sample_causal_trades = [
        {
            "setup_id": "c1",
            "bar_idx": 10,
            "entry_bar_idx": 11,
            "exit_bar_idx": 16,
            "net_return": 0.02,
            "net_return_pct": 2.0
        }
    ]
    causal_port = simulate_event_driven_portfolio(
        trades=sample_causal_trades,
        df=df,
        signal_causality_mode="CAUSAL_WALK_FORWARD_SIGNALS"
    )
    assert causal_port.signal_causality_mode == "CAUSAL_WALK_FORWARD_SIGNALS"
    assert causal_port.simulation_execution_status == "SUCCESSFUL_EXECUTION"

# ==============================================================================
# 3. Simulation execution success with negative profitability
# ==============================================================================
def test_simulation_execution_success_with_negative_profitability():
    """
    Test 3: Successful simulation execution must NOT imply profitability.
    Verify that an execution can be SUCCESSFUL/EXPLORATORY while profitability is UNPROFITABLE.
    """
    import pandas as pd
    loss_trades = [
        {
            "setup_id": "loss_1",
            "bar_idx": 10,
            "entry_bar_idx": 11,
            "exit_bar_idx": 16,
            "decision_date": "2025-01-11",
            "net_return": -0.045,
            "net_return_pct": -4.5
        },
        {
            "setup_id": "loss_2",
            "bar_idx": 25,
            "entry_bar_idx": 26,
            "exit_bar_idx": 31,
            "decision_date": "2025-01-26",
            "net_return": -0.032,
            "net_return_pct": -3.2
        }
    ]
    bars = make_sample_ohlcv(80)
    df = pd.DataFrame([b.model_dump() for b in bars])
    port = simulate_event_driven_portfolio(
        trades=loss_trades,
        df=df,
        signal_causality_mode="CAUSAL_WALK_FORWARD_SIGNALS"
    )
    
    # Simulation execution was successful without accounting crashes
    assert port.simulation_execution_status == "SUCCESSFUL_EXECUTION"
    assert port.cash_reconciliation_verified is True
    
    # But profitability status is UNPROFITABLE
    assert port.portfolio_compounded_return_pct < 0.0
    assert port.profitability_status == "UNPROFITABLE"

# ==============================================================================
# 4. Portfolio PASS criteria with risk-limit breaches
# ==============================================================================
def test_portfolio_pass_criteria_with_risk_limit_breaches():
    """
    Test 4: Portfolio simulation execution passing must never mask risk-limit breaches.
    When max drawdown > 25%, risk_limit_status must be DRAWDOWN_LIMIT_BREACH.
    """
    bars = make_sample_ohlcv(140, trend=-0.002)  # downtrend generates high drawdown
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_res, _ = engine.investigate_setup(DirectionalBias.BEARISH)
    comp = engine.compare_historical_setups(DirectionalBias.BEARISH, horizon_bars=5, backtest_result=hist_res)
    
    # Risk limit status must reflect the drawdown breach independently
    assert comp.walk_forward_evaluation is not None
    assert comp.walk_forward_evaluation.max_drawdown_pct > 25.0
    assert comp.risk_limit_status == "DRAWDOWN_LIMIT_BREACH"
    assert comp.walk_forward_status == "FAIL"

# ==============================================================================
# 5. Non-overlapping block starting-offset sensitivity
# ==============================================================================
def test_non_overlapping_block_starting_offset_sensitivity():
    """
    Test 5: Evaluate 5 starting offsets (k in [0..4]) for non-overlapping block sampling.
    Verify that offset sensitivity runs are calculated and ranges populated.
    """
    bars = make_sample_ohlcv(150)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    comp = engine.compare_historical_setups(DirectionalBias.BULLISH, horizon_bars=5)
    
    assert comp.walk_forward_evaluation is not None
    wf = comp.walk_forward_evaluation
    assert wf.sensitivity_analysis is not None
    sa = wf.sensitivity_analysis
    assert sa.method == "NON_OVERLAPPING_BLOCK_SAMPLING"
    assert sa.sampling_step_bars == 5
    
    # 5 offsets must be evaluated
    assert len(sa.offset_sensitivity_runs) == 5
    offsets = [r.offset_k for r in sa.offset_sensitivity_runs]
    assert offsets == [0, 1, 2, 3, 4]
    
    # Parameter sensitivity ranges must be populated
    assert len(sa.mean_return_range) == 2
    assert len(sa.win_rate_range) == 2
    assert len(sa.profit_factor_range) == 2
    assert sa.mean_return_range[0] <= sa.mean_return_range[1]
    assert sa.win_rate_range[0] <= sa.win_rate_range[1]
    assert sa.profit_factor_range[0] <= sa.profit_factor_range[1]
    
    # Terminology must use 'non-overlapping' rather than claiming statistical independence
    assert "non_overlapping" in sa.method.lower()

# ==============================================================================
# 6. Confidence-interval calculation verification
# ==============================================================================
def test_confidence_interval_calculation():
    """
    Test 6: Verify confidence interval calculation formula [mean +/- 1.96 * SE],
    observations used, zero-span detection, and documentation.
    """
    bars = make_sample_ohlcv(150)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    comp = engine.compare_historical_setups(DirectionalBias.BULLISH, horizon_bars=5)
    wf = comp.walk_forward_evaluation
    assert wf is not None
    sa = wf.sensitivity_analysis
    assert sa is not None
    
    # Formula check: CI = mean +/- 1.96 * SE
    expected_lower = sa.mean_net_return_pct - 1.96 * sa.standard_error_pct
    expected_upper = sa.mean_net_return_pct + 1.96 * sa.standard_error_pct
    assert np.isclose(sa.ci_95_lower_pct, round(expected_lower, 2), atol=0.05)
    assert np.isclose(sa.ci_95_upper_pct, round(expected_upper, 2), atol=0.05)
    
    # Zero span check
    if sa.ci_95_lower_pct <= 0.0 <= sa.ci_95_upper_pct:
        assert sa.ci_spans_zero is True
        assert "NO_STATISTICAL_EDGE" in sa.statistical_significance
    
    # Documentation check
    assert "mean_return +/- 1.96 * SE" in sa.confidence_interval_documentation
    assert "insufficient evidence" in sa.confidence_interval_documentation.lower()

# ==============================================================================
# 7. Benchmark alignment and excess-return calculation
# ==============================================================================
def test_benchmark_alignment_and_excess_return():
    """
    Test 7: Benchmark comparison for identical historical dates and capital convention.
    Compares strategy with Buy-and-Hold underlying and Cash risk-free rate (6.5%).
    """
    bars = make_sample_ohlcv(140)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_res, _ = engine.investigate_setup(DirectionalBias.BULLISH)
    comp = engine.compare_historical_setups(DirectionalBias.BULLISH, horizon_bars=5, backtest_result=hist_res)
    
    assert comp.portfolio_simulation is not None
    bm = comp.portfolio_simulation.benchmark_comparison
    assert bm is not None
    
    # Benchmark alignment check
    assert bm.start_date != ""
    assert bm.end_date != ""
    assert bm.start_price > 0.0
    assert bm.end_price > 0.0
    assert bm.holding_days > 0
    
    # Buy-and-Hold return calculation check
    expected_asset_ret = round(((bm.end_price - bm.start_price) / bm.start_price) * 100.0, 2)
    assert np.isclose(bm.benchmark_return_pct, expected_asset_ret, atol=0.05)
    
    # Cash return check (6.5% p.a.)
    years = bm.holding_days / 365.25
    expected_cash_ret = round(((1.0 + 0.065) ** years - 1.0) * 100.0, 2)
    assert np.isclose(bm.cash_benchmark_return_pct, expected_cash_ret, atol=0.05)
    
    # Excess return calculation check
    assert np.isclose(bm.excess_return_vs_asset_pct, bm.strategy_net_return_pct - bm.benchmark_return_pct, atol=0.05)
    assert np.isclose(bm.excess_return_vs_cash_pct, bm.strategy_net_return_pct - bm.cash_benchmark_return_pct, atol=0.05)
    
    # Dividend adjustment disclosure
    assert "dividend reinvestment adjustments" in bm.dividend_adjustment_note

# ==============================================================================
# 8. Portfolio cash, exposure and drawdown reconciliation
# ==============================================================================
def test_portfolio_cash_exposure_and_drawdown_reconciliation():
    """
    Test 8: Verify bar-by-bar cash reconciliation (cash + invested == equity),
    maximum observed exposure separated from capacity, and consistent equity series.
    """
    bars = make_sample_ohlcv(140)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_res, _ = engine.investigate_setup(DirectionalBias.BULLISH)
    comp = engine.compare_historical_setups(DirectionalBias.BULLISH, horizon_bars=5, backtest_result=hist_res)
    
    port = comp.portfolio_simulation
    assert port is not None
    
    # Cash reconciliation check
    assert port.cash_reconciliation_verified is True
    assert port.max_cash_discrepancy < 1e-4
    
    # Bar-by-bar equity curve consistency
    for pt in port.equity_curve:
        cash = pt["cash"]
        invested = pt["invested"]
        equity = pt["equity"]
        assert np.isclose(cash + invested, equity, atol=1e-3)
    
    # Max observed exposure separated from capacity (100%)
    assert port.max_gross_exposure_capacity_pct == 100.0
    assert port.max_observed_exposure_pct <= port.max_gross_exposure_capacity_pct
    assert np.isclose(port.max_observed_exposure_pct, port.max_concurrent_positions * port.allocation_per_position_pct, atol=5.0)
    
    # Return and drawdown derived strictly from equity curve
    assert port.equity_curve[0]["equity"] == port.initial_capital
    assert np.isclose(port.equity_curve[-1]["equity"], port.final_equity, atol=1e-3)
    expected_ret = round(((port.final_equity - port.initial_capital) / port.initial_capital) * 100.0, 2)
    assert np.isclose(port.portfolio_compounded_return_pct, expected_ret, atol=0.05)

# ==============================================================================
# 9. Consistent top-level verdict when walk-forward validation fails
# ==============================================================================
def test_consistent_top_level_verdict_when_walk_forward_fails():
    """
    Test 9: When walk-forward validation fails and risk limits are breached,
    the top-level analysis verdict must NOT claim a validated edge or confirmed bullish thesis.
    """
    bars = make_sample_ohlcv(140, trend=-0.001)
    tester = TradeThesisStressTester(bars, symbol="^NSEI", timeframe="1d")
    
    strat_results = [
        StrategyResult(strategy_name="TrendFollowing", signal=Signal.SELL, confidence=0.75, metadata={}),
        StrategyResult(strategy_name="MomentumOscillator", signal=Signal.SELL, confidence=0.70, metadata={}),
    ]
    
    result = tester.run_stress_test(strat_results)
    
    # All 6 structured status fields must be present
    assert hasattr(result, "simulation_execution_status")
    assert hasattr(result, "profitability_status")
    assert hasattr(result, "historical_validation_status")
    assert hasattr(result, "walk_forward_status")
    assert hasattr(result, "risk_limit_status")
    assert hasattr(result, "benchmark_comparison_status")
    
    # When walk-forward status is FAIL, thesis cannot be CONFIRMED or VALIDATED
    if result.walk_forward_status == "FAIL":
        assert result.setup_status.value != "CONFIRMED"
        assert result.thesis_status.value != "INTACT"
        assert "VALIDATED THESIS" not in result.verdict
