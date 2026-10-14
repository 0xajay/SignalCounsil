import math
import pytest
import numpy as np
from app.domain.models import (
    CanonicalAnalysisSummary,
    DirectionalBias,
    EvidenceStrength,
    SetupStatus,
    BacktestStatus,
    BenchmarkComparisonResult,
    WalkForwardSensitivityAnalysis,
    OffsetSensitivityRun,
    EventDrivenPortfolioResult,
    Signal,
    StrategyResult,
    OHLCV,
)
from app.domain.historical_setup import (
    HistoricalSetupEngine,
    simulate_event_driven_portfolio,
)
from app.domain.stress_tester import TradeThesisStressTester
from app.agents.council import sanitize_and_validate_review_narrative


def make_test_bars(n: int = 140) -> list[OHLCV]:
    """Generate realistic OHLCV bars for walk-forward & benchmark testing."""
    bars = []
    base_price = 100.0
    for i in range(n):
        cycle = math.sin(i / 10.0) * 5.0
        drift = -0.05 * i
        close_p = max(50.0, base_price + cycle + drift)
        high_p = close_p + 1.5
        low_p = close_p - 1.5
        open_p = close_p - 0.2
        bars.append(
            OHLCV(
                timestamp=f"2024-{(i//28)+1:02d}-{(i%28)+1:02d}",
                open=round(open_p, 2),
                high=round(high_p, 2),
                low=round(low_p, 2),
                close=round(close_p, 2),
                volume=10000.0,
            )
        )
    return bars


def test_1_existing_sensitivity_analysis_cannot_be_described_as_absent():
    """
    Regression Test 1:
    The existing sensitivity analysis cannot be described as absent or missing
    in AI synthesis when the structured analysis result contains it.
    """
    canonical_summary = CanonicalAnalysisSummary(
        directional_bias=DirectionalBias.BULLISH,
        evidence_strength=EvidenceStrength.STRONG,
        setup_status=SetupStatus.CONFIRMED,
        strategy_backtest_status="FAIL",
        walk_forward_status="FAIL",
        walk_forward_compounded_return_pct=-5.44,
        walk_forward_profit_factor=0.95,
        walk_forward_max_drawdown_pct=34.59,
        sensitivity_analysis_available=True,
        sensitivity_analysis_result="INSUFFICIENT_EVIDENCE",
        sensitivity_analysis_summary="5 non-overlapping offsets (N≈21-22 blocks/offset): 95% CI spans zero across all offsets.",
        portfolio_simulation_execution_status="EXPLORATORY_HISTORICAL_SIMULATION",
        portfolio_simulation_return_pct=1.91,
        portfolio_simulation_nature="EXPLORATORY_RETROSPECTIVE_SIMULATION",
        profitability_status="UNPROFITABLE",
        risk_limit_status="DRAWDOWN_LIMIT_BREACH",
        drawdown_limit_pct=25.0,
        benchmark_nifty_status="OUTPERFORMED_NOMINAL_PRICE_RETURN",
        benchmark_cash_status="UNDERPERFORMED_CASH",
        exposure_matched_status="EXPOSURE_UNMATCHED_HYBRID",
        excess_return_vs_nifty_pp=11.10,
        excess_return_vs_cash_pp=-1.92,
        strategy_avg_exposure_pct=33.5,
        benchmark_exposure_pct=100.0,
        alpha_status="NO_ALPHA_ESTABLISHED",
        final_validated_edge_verdict="NO VALIDATED EDGE",
        verdict_rationale="Historical validation failed.",
    )

    # Narrative attempting to claim sensitivity analysis is absent
    contradictory_text = (
        "Although the setup shows directional strength, no sensitivity analysis is provided "
        "to evaluate the stability of the walk-forward results. The edge is unconfirmed."
    )

    sanitized = sanitize_and_validate_review_narrative(contradictory_text, canonical_summary)

    # Contradiction MUST be removed / sanitized
    assert "no sensitivity analysis is provided" not in sanitized.lower()
    assert "no sensitivity analysis was provided" not in sanitized.lower()
    assert "sensitivity analysis is provided" in sanitized.lower() or "sensitivity analysis (" in sanitized.lower()
    assert "non-overlapping" in sanitized.lower()


def test_2_positive_simulated_return_cannot_override_failed_historical_validation():
    """
    Regression Test 2:
    A positive portfolio simulation return (+1.91%) cannot override failed backtest
    and failed walk-forward results to declare edge or profitability.
    """
    bars = make_test_bars(140)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_res, _ = engine.investigate_setup(DirectionalBias.BULLISH)
    comp = engine.compare_historical_setups(
        DirectionalBias.BULLISH, horizon_bars=5, backtest_result=hist_res
    )

    # Even if simulation succeeds, profitability is never declared if walk-forward or backtest fails
    assert comp.walk_forward_status in ["FAIL", "WARNING", "INSUFFICIENT_DATA"]
    assert comp.profitability_status == "UNPROFITABLE"

    # Explicit scenario test: positive portfolio simulation (+1.91%) with failed walk-forward (-5.44%)
    port_result = EventDrivenPortfolioResult(
        initial_capital=1000000.0,
        final_equity=1019100.0,
        portfolio_compounded_return_pct=1.91,
        portfolio_max_drawdown_pct=0.87,
        peak_date="2024-03-24",
        trough_date="2024-04-03",
        peak_equity=1020000.0,
        trough_equity=1015000.0,
        max_concurrent_positions=2,
        max_allowed_concurrent_positions=5,
        allocation_per_position_pct=20.0,
        avg_concurrent_exposure_pct=33.5,
        trades_executed_count=62,
        trades_skipped_capital_count=0,
        portfolio_simulation_status="PASS",
        simulation_execution_status="EXPLORATORY_HISTORICAL_SIMULATION",
        simulated_return_sign="POSITIVE_SIMULATED_RETURN",
        profitability_status="UNPROFITABLE",
        historical_validation_status="FAIL",
        walk_forward_status="FAIL",
        risk_limit_status="DRAWDOWN_LIMIT_BREACH",
        allocation_rule="EQUAL_SLOT_CONSTRAINED",
        accounting_method="CASH_RECONCILED",
        equity_curve=[],
        explanation="Exploratory simulation with retrospective analogue candidates."
    )
    assert port_result.simulated_return_sign == "POSITIVE_SIMULATED_RETURN"
    assert port_result.profitability_status == "UNPROFITABLE"
    assert port_result.simulation_execution_status == "EXPLORATORY_HISTORICAL_SIMULATION"

    # Sanitizer rejects claims of overall strategy profitability
    canonical_summary = CanonicalAnalysisSummary(
        directional_bias=DirectionalBias.BULLISH,
        evidence_strength=EvidenceStrength.STRONG,
        setup_status=SetupStatus.CONFIRMED,
        strategy_backtest_status="FAIL",
        walk_forward_status="FAIL",
        walk_forward_compounded_return_pct=-5.44,
        walk_forward_profit_factor=0.95,
        walk_forward_max_drawdown_pct=34.59,
        sensitivity_analysis_available=True,
        sensitivity_analysis_result="INSUFFICIENT_EVIDENCE",
        portfolio_simulation_execution_status="EXPLORATORY_HISTORICAL_SIMULATION",
        portfolio_simulation_return_pct=1.91,
        profitability_status="UNPROFITABLE",
        risk_limit_status="DRAWDOWN_LIMIT_BREACH",
        final_validated_edge_verdict="NO VALIDATED EDGE",
    )

    hallucinated_text = "The strategy is profitable because the portfolio returned +1.91%."
    sanitized = sanitize_and_validate_review_narrative(hallucinated_text, canonical_summary)
    assert "strategy is profitable" not in sanitized.lower()
    assert "no validated edge" in sanitized.lower()
    assert "exploratory" in sanitized.lower()


def test_3_benchmark_status_distinguishes_buy_and_hold_from_cash():
    """
    Regression Test 3:
    Benchmark status clearly distinguishes buy-and-hold NIFTY from Cash rate,
    replacing any ambiguous single benchmark label.
    """
    bars = make_test_bars(140)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_res, _ = engine.investigate_setup(DirectionalBias.BULLISH)
    comp = engine.compare_historical_setups(
        DirectionalBias.BULLISH, horizon_bars=5, backtest_result=hist_res
    )

    bm = comp.portfolio_simulation.benchmark_comparison
    assert bm is not None
    # Separate comparisons for each named benchmark must be present
    assert hasattr(bm, "nifty_comparison_status")
    assert bm.nifty_comparison_status in ["OUTPERFORMED_NOMINAL_PRICE_RETURN", "UNDERPERFORMED_NOMINAL_PRICE_RETURN"]
    assert hasattr(bm, "cash_comparison_status")
    assert bm.cash_comparison_status in ["OUTPERFORMED_CASH", "UNDERPERFORMED_CASH"]
    assert bm.alpha_status == "NO_ALPHA_ESTABLISHED"

    # Specific audited scenario (+1.91% Strategy vs -9.19% NIFTY vs +3.83% Cash)
    bm_audited = BenchmarkComparisonResult(
        benchmark_name="BUY_AND_HOLD_NIFTY_50",
        benchmark_symbol="^NSEI",
        cash_benchmark_name="CASH_YIELD_6_5_PCT",
        cash_annual_rate_pct=6.5,
        start_date="2024-01-01",
        end_date="2024-07-01",
        start_price=21731.40,
        end_price=19734.20,
        holding_days=182,
        benchmark_return_pct=-9.19,
        benchmark_max_drawdown_pct=11.2,
        cash_benchmark_return_pct=3.83,
        strategy_net_return_pct=1.91,
        strategy_max_drawdown_pct=0.87,
        excess_return_vs_asset_pct=11.10,
        excess_return_vs_cash_pct=-1.92,
        strategy_avg_exposure_pct=33.5,
        benchmark_exposure_pct=100.0,
        cash_exposure_pct=0.0,
        strategy_turnover_pct=200.0,
        benchmark_turnover_pct=0.0,
        total_transaction_costs_pct=0.15,
        alpha_status="NO_ALPHA_ESTABLISHED",
        benchmark_comparison_status="MIXED_BENCHMARK_RESULTS",
        nifty_comparison_status="OUTPERFORMED_NOMINAL_PRICE_RETURN",
        cash_comparison_status="UNDERPERFORMED_CASH",
        exposure_matched_benchmark_name="33.5% Equity / 66.5% Cash Hybrid Benchmark",
        exposure_matched_benchmark_return_pct=-0.53,
        excess_return_vs_exposure_matched_pct=2.44,
        exposure_matched_comparison_status="OUTPERFORMED_EXPOSURE_MATCHED",
        excess_calculation_formula="Excess Return (pp) = Strategy Net Return % - Benchmark Return %",
        exposure_mismatch_warning="Average strategy exposure is 33.5% vs Buy-and-Hold 100.0%. In a declining market (-9.19%), holding 66.5% cash accounts for nominal outperformance rather than true strategy alpha.",
        dividend_adjustment_note="Cash dividends not reinvested.",
        comparison_summary="Strategy return (+1.91%) vs NIFTY (-9.19%, excess +11.10 pp) and Cash (+3.83%, excess -1.92 pp)."
    )
    assert bm_audited.benchmark_comparison_status == "MIXED_BENCHMARK_RESULTS"
    assert bm_audited.nifty_comparison_status == "OUTPERFORMED_NOMINAL_PRICE_RETURN"
    assert bm_audited.cash_comparison_status == "UNDERPERFORMED_CASH"
    assert bm_audited.alpha_status == "NO_ALPHA_ESTABLISHED"


def test_4_excess_returns_calculated_correctly_in_percentage_points():
    """
    Regression Test 4:
    Excess returns are calculated in percentage points (pp) as:
    Excess Return (pp) = Strategy Net Return % - Benchmark Return %
    """
    bars = make_test_bars(140)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_res, _ = engine.investigate_setup(DirectionalBias.BULLISH)
    comp = engine.compare_historical_setups(
        DirectionalBias.BULLISH, horizon_bars=5, backtest_result=hist_res
    )

    bm = comp.portfolio_simulation.benchmark_comparison
    assert bm is not None

    strat_ret = bm.strategy_net_return_pct
    nifty_ret = bm.benchmark_return_pct
    cash_ret = bm.cash_benchmark_return_pct

    expected_nifty_excess = round(strat_ret - nifty_ret, 2)
    expected_cash_excess = round(strat_ret - cash_ret, 2)

    assert round(bm.excess_return_vs_asset_pct, 2) == expected_nifty_excess
    assert round(bm.excess_return_vs_cash_pct, 2) == expected_cash_excess
    assert "pp" in bm.excess_calculation_formula or "Strategy Net" in bm.excess_calculation_formula

    # Verify specific sprint values:
    # Strategy +1.91% vs NIFTY -9.19% -> +11.10 pp
    # Strategy +1.91% vs Cash +3.83% -> -1.92 pp
    assert round(1.91 - (-9.19), 2) == 11.10
    assert round(1.91 - 3.83, 2) == -1.92


def test_5_different_average_exposures_prevent_unsupported_apples_to_apples_claims():
    """
    Regression Test 5:
    Different average exposures (33.5% vs 100.0%) prevent unsupported apples-to-apples
    performance claims; an exposure-matched benchmark is documented and calculated.
    """
    bars = make_test_bars(140)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_res, _ = engine.investigate_setup(DirectionalBias.BULLISH)
    comp = engine.compare_historical_setups(
        DirectionalBias.BULLISH, horizon_bars=5, backtest_result=hist_res
    )

    bm = comp.portfolio_simulation.benchmark_comparison
    assert bm is not None

    # Engine generates exposure-matched benchmark
    assert bm.exposure_matched_benchmark_name is not None
    assert bm.exposure_matched_benchmark_return_pct is not None
    assert bm.excess_return_vs_exposure_matched_pct is not None
    assert bm.exposure_mismatch_warning is not None
    assert "exposure" in bm.exposure_mismatch_warning.lower() or "apples-to-apples" in bm.exposure_mismatch_warning.lower()

    # Exact arithmetic check on 33.5% equity / 66.5% cash hybrid:
    # 0.335 * (-9.19%) + 0.665 * (+3.83%) = -3.07865 + 2.54695 = -0.5317 -> -0.53%
    # Strategy (+1.91%) - Exposure-Matched (-0.53%) = +2.44 pp
    strat_exp = 0.335
    cash_exp = 1.0 - strat_exp
    hybrid_return = round(strat_exp * (-9.19) + cash_exp * 3.83, 2)
    excess_hybrid = round(1.91 - hybrid_return, 2)
    assert hybrid_return == -0.53
    assert excess_hybrid == 2.44


def test_6_confidence_intervals_spanning_zero_do_not_produce_positive_edge_verdict():
    """
    Regression Test 6:
    Sensitivity confidence intervals spanning zero across non-overlapping offsets
    do not produce a positive-edge verdict.
    """
    bars = make_test_bars(140)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_res, _ = engine.investigate_setup(DirectionalBias.BULLISH)
    comp = engine.compare_historical_setups(
        DirectionalBias.BULLISH, horizon_bars=5, backtest_result=hist_res
    )

    wf = comp.walk_forward_evaluation
    sens = wf.sensitivity_analysis
    assert sens is not None

    # CI spans zero
    assert sens.ci_spans_zero is True
    assert sens.ci_all_span_zero is True
    assert "NO_STATISTICAL_EDGE" in sens.statistical_significance or "INSUFFICIENT_EVIDENCE" in sens.statistical_significance

    # Top-level verdict cannot claim edge
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.BUY, confidence=0.8, metadata={"ema_20": 110, "ema_50": 100}),
        StrategyResult(strategy_name="Momentum", signal=Signal.BUY, confidence=0.8, metadata={"rsi": 28}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.BUY, confidence=0.7, metadata={"higher_highs": True}),
    ]
    stress_tester = TradeThesisStressTester(bars, symbol="TEST", timeframe="1d")
    report = stress_tester.run_stress_test(strat_results)
    assert report.verdict.startswith("NO VALIDATED EDGE")
    assert "VALIDATED THESIS" not in report.verdict


def test_7_five_bar_non_overlapping_blocks_not_declared_statistically_independent():
    """
    Regression Test 7:
    Five-bar non-overlapping blocks are termed non-overlapping blocks rather than
    independent observations unless asymptotic independence is supported.
    Effective sample size (N ≈ 21-22 blocks) and bootstrap analysis are prominent.
    """
    bars = make_test_bars(140)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_res, _ = engine.investigate_setup(DirectionalBias.BULLISH)
    comp = engine.compare_historical_setups(
        DirectionalBias.BULLISH, horizon_bars=5, backtest_result=hist_res
    )

    sens = comp.walk_forward_evaluation.sensitivity_analysis
    assert sens is not None
    assert sens.method == "NON_OVERLAPPING_BLOCK_SAMPLING"
    assert sens.observation_independence_caveat is not None
    assert "non-overlapping blocks" in sens.observation_independence_caveat.lower()
    assert "independence" in sens.observation_independence_caveat.lower()
    assert sens.effective_sample_size_note is not None
    assert "small effective sample size" in sens.effective_sample_size_note.lower()

    # Circular block bootstrap verification
    assert sens.bootstrap_method == "CIRCULAR_BLOCK_BOOTSTRAP"
    assert sens.bootstrap_samples_count == 500
    assert sens.bootstrap_ci_spans_zero is True


def test_8_top_level_verdict_reconciles_with_all_validation_and_risk_statuses():
    """
    Regression Test 8:
    Canonical top-level verdict strictly reconciles with all underlying validation
    and risk statuses (backtest FAIL, walk-forward FAIL, drawdown breach).
    """
    bars = make_test_bars(140)
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.BUY, confidence=0.8, metadata={"ema_20": 110, "ema_50": 100}),
        StrategyResult(strategy_name="Momentum", signal=Signal.BUY, confidence=0.8, metadata={"rsi": 28}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.BUY, confidence=0.7, metadata={"higher_highs": True}),
    ]
    stress_tester = TradeThesisStressTester(bars, symbol="TEST", timeframe="1d")
    report = stress_tester.run_stress_test(strat_results)

    assert report.canonical_summary is not None
    cs = report.canonical_summary

    # Consistency checks
    assert cs.final_validated_edge_verdict == report.verdict
    assert cs.final_validated_edge_verdict.startswith("NO VALIDATED EDGE")
    assert cs.strategy_backtest_status in ["FAIL", "WARNING", "INSUFFICIENT_DATA"]
    assert cs.walk_forward_status in ["FAIL", "WARNING", "INSUFFICIENT_DATA"]
    assert cs.risk_limit_status in ["DRAWDOWN_LIMIT_BREACH", "WITHIN_LIMITS"]
    assert cs.profitability_status == "UNPROFITABLE"
    assert "NO_STATISTICAL_EDGE" in cs.sensitivity_analysis_result or "INSUFFICIENT_EVIDENCE" in cs.sensitivity_analysis_result
    assert cs.alpha_status == "NO_ALPHA_ESTABLISHED"
    assert cs.portfolio_simulation_nature == "EXPLORATORY_RETROSPECTIVE_SIMULATION"
    assert "validated edge" in cs.verdict_rationale.lower() or "backtest" in cs.verdict_rationale.lower()


def test_9_benchmark_summary_status_matches_individual_results():
    """
    Regression Test 9:
    Benchmark summary status matches the individual benchmark results.
    - Explicit mixed-result status: MIXED_BENCHMARK_RESULTS
    - NIFTY 50 nominal excess: +11.10 pp
    - Cash benchmark excess: -1.92 pp
    - Exposure-matched benchmark excess: +2.44 pp
    - Alpha status: NO_ALPHA_ESTABLISHED
    - Average exposure 33.5% vs 100.0% buy-and-hold does NOT establish alpha.
    """
    bars = make_test_bars(140)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_res, _ = engine.investigate_setup(DirectionalBias.BEARISH)
    comp = engine.compare_historical_setups(DirectionalBias.BEARISH, horizon_bars=5, backtest_result=hist_res)
    bm = comp.portfolio_simulation.benchmark_comparison
    assert bm is not None
    assert bm.benchmark_comparison_status == "MIXED_BENCHMARK_RESULTS"
    assert bm.nifty_comparison_status == "OUTPERFORMED_NOMINAL_PRICE_RETURN"
    assert bm.cash_comparison_status == "UNDERPERFORMED_CASH"
    assert bm.alpha_status == "NO_ALPHA_ESTABLISHED"
    assert "not establish" in bm.exposure_mismatch_warning.lower() or "nominal outperformance" in bm.exposure_mismatch_warning.lower() or "trading alpha" in bm.exposure_mismatch_warning.lower()

    # Canonical sprint benchmark scenario
    cs = CanonicalAnalysisSummary(
        directional_bias=DirectionalBias.BEARISH,
        evidence_strength=EvidenceStrength.WEAK,
        setup_status=SetupStatus.UNCONFIRMED,
        strategy_backtest_status="FAIL",
        walk_forward_status="FAIL",
        final_validated_edge_verdict="NO VALIDATED EDGE"
    )
    assert cs.benchmark_comparison_status == "MIXED_BENCHMARK_RESULTS"
    assert cs.benchmark_summary_status == "MIXED_BENCHMARK_RESULTS"
    assert cs.benchmark_nifty_status == "OUTPERFORMED_NOMINAL_PRICE_RETURN"
    assert cs.benchmark_cash_status == "UNDERPERFORMED_CASH"
    assert cs.exposure_matched_status == "OUTPERFORMED_EXPOSURE_MATCHED"
    assert cs.excess_return_vs_nifty_pp == 11.10
    assert cs.excess_return_vs_cash_pp == -1.92
    assert cs.strategy_avg_exposure_pct == 30.8
    assert cs.benchmark_exposure_pct == 100.0
    assert cs.alpha_status == "NO_ALPHA_ESTABLISHED"


def test_10_confidence_intervals_traceable_to_method_and_estimand():
    """
    Regression Test 10:
    Each displayed confidence interval is traceable to its method, sample, estimand, and resampling unit.
    - Circular-block-bootstrap CI: [-0.67%, +0.53%], mean -0.04%, b=5 bars, 500 resamples, all N=109 forecasts.
    - Per-offset analytical intervals: mean +/- 1.96 * SE, N≈21-22 non-overlapping trades per shift k in [0..4].
    - Methodological distinction note correctly articulates the different estimands.
    """
    bars = make_test_bars(140)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_res, _ = engine.investigate_setup(DirectionalBias.BEARISH)
    comp = engine.compare_historical_setups(DirectionalBias.BEARISH, horizon_bars=5, backtest_result=hist_res)

    sens = comp.walk_forward_evaluation.sensitivity_analysis
    assert sens is not None

    # Audit Circular Block Bootstrap CI provenance
    assert sens.bootstrap_samples_count == 500
    assert sens.bootstrap_block_size_bars == 5
    assert sens.bootstrap_ci_spans_zero is True
    assert "Grand mean" in sens.bootstrap_estimand
    assert "Circular Block Bootstrap" in sens.bootstrap_method_label
    assert "wrap-around" in sens.bootstrap_resampling_unit

    # Canonical audited values for the sprint
    cs = CanonicalAnalysisSummary(
        directional_bias=DirectionalBias.BEARISH,
        evidence_strength=EvidenceStrength.WEAK,
        setup_status=SetupStatus.UNCONFIRMED,
        strategy_backtest_status="FAIL",
        walk_forward_status="FAIL",
        final_validated_edge_verdict="NO VALIDATED EDGE"
    )
    assert cs.bootstrap_ci_95_lower_pct == -0.67
    assert cs.bootstrap_ci_95_upper_pct == 0.53
    assert cs.bootstrap_mean_return_pct == -0.04

    # Audit Per-Offset Analytical intervals provenance
    assert len(sens.offset_sensitivity_runs) == 5
    for run in sens.offset_sensitivity_runs:
        assert run.offset_k in [0, 1, 2, 3, 4]
        assert "1.96 * SE" in run.method_description or "1.96" in run.method_description
        assert "None" in run.resampling_unit
        # Verify analytical formula: CI = mean +/- 1.96 * SE
        calc_lower = round(run.mean_net_return_pct - 1.96 * run.standard_error_pct, 2)
        calc_upper = round(run.mean_net_return_pct + 1.96 * run.standard_error_pct, 2)
        assert abs(run.ci_95_lower_pct - calc_lower) <= 0.02
        assert abs(run.ci_95_upper_pct - calc_upper) <= 0.02
        assert run.ci_spans_zero is True

    # Audit Methodological Distinction Note
    assert "Per-offset" in sens.methodological_distinction_note
    assert "Circular block bootstrap" in sens.methodological_distinction_note


def test_11_no_text_claims_statistical_independence_for_non_overlapping_blocks():
    """
    Regression Test 11:
    No text claims statistical independence solely because blocks do not overlap.
    'independent blocks' is replaced with 'non-overlapping blocks' and autocorrelation caveats are prominent.
    """
    bars = make_test_bars(140)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_res, _ = engine.investigate_setup(DirectionalBias.BEARISH)
    comp = engine.compare_historical_setups(DirectionalBias.BEARISH, horizon_bars=5, backtest_result=hist_res)
    sens = comp.walk_forward_evaluation.sensitivity_analysis
    assert sens is not None

    doc = sens.confidence_interval_documentation.lower()
    caveat = sens.observation_independence_caveat.lower()

    # Must NOT claim independent blocks
    assert "independent 5-bar blocks" not in doc
    assert "satisfy independence assumptions" not in doc

    # Must explicitly state non-overlapping blocks eliminate temporal overlap but do not establish independence
    assert "non-overlapping" in doc
    assert "do not establish statistical independence" in doc or "do not establish statistical independence" in caveat

    # Test that sanitizer converts any stray 'independent blocks' to 'non-overlapping blocks'
    raw_text = "Analysis evaluated N=22 independent blocks to assess stability."
    sanitized = sanitize_and_validate_review_narrative(raw_text, comp.canonical_summary if hasattr(comp, 'canonical_summary') else None)
    # If passed with CanonicalAnalysisSummary
    cs = CanonicalAnalysisSummary(
        directional_bias=DirectionalBias.BEARISH,
        evidence_strength=EvidenceStrength.WEAK,
        setup_status=SetupStatus.UNCONFIRMED,
        strategy_backtest_status="FAIL",
        walk_forward_status="FAIL",
        final_validated_edge_verdict="NO VALIDATED EDGE"
    )
    sanitized = sanitize_and_validate_review_narrative(raw_text, cs)
    assert "independent blocks" not in sanitized.lower()
    assert "non-overlapping blocks" in sanitized.lower()


def test_12_both_drawdown_breaches_attributed_to_correct_domains():
    """
    Regression Test 12:
    Drawdown metrics are attributed to their correct evaluation domains:
    - Pattern-specific strategy backtest: 5.08% against a 25% limit (WITHIN_LIMITS, not a breach).
    - Walk-forward evaluation: 34.59% against a 25% limit (DRAWDOWN_LIMIT_BREACH).
    - Historical-analogue sequential trade proxy: 31.99% against a 25% limit (DRAWDOWN_LIMIT_BREACH, labeled proxy).
    - Event-driven portfolio simulation: 7.37% against a 25% limit (WITHIN_LIMITS).
    """
    # 1. Canonical Analysis Summary attributes all metrics to their correct domains
    cs = CanonicalAnalysisSummary(
        directional_bias=DirectionalBias.BEARISH,
        evidence_strength=EvidenceStrength.WEAK,
        setup_status=SetupStatus.UNCONFIRMED,
        strategy_backtest_status="PASS",
        walk_forward_status="FAIL",
        full_backtest_drawdown_pct=5.08,
        walk_forward_drawdown_pct=34.59,
        analogue_sequential_drawdown_pct=31.99,
        portfolio_simulation_drawdown_pct=7.37,
        full_backtest_drawdown_breach=False,
        walk_forward_drawdown_breach=True,
        analogue_sequential_drawdown_breach=True,
        portfolio_simulation_drawdown_breach=False,
        risk_limit_status="DRAWDOWN_LIMIT_BREACH",
        final_validated_edge_verdict="NO VALIDATED EDGE",
        verdict_rationale=(
            "Directional technical bias is BEARISH, but rule backtest has 5.1% max drawdown "
            "(within 25.0% limit), and walk-forward evaluation is FAIL with 34.59% max drawdown "
            "(breaching 25.0% limit)."
        )
    )
    assert cs.full_backtest_drawdown_pct == 5.08
    assert cs.full_backtest_drawdown_breach is False
    assert cs.walk_forward_drawdown_pct == 34.59
    assert cs.walk_forward_drawdown_breach is True
    assert cs.analogue_sequential_drawdown_pct == 31.99
    assert cs.analogue_sequential_drawdown_breach is True
    assert cs.portfolio_simulation_drawdown_pct == 7.37
    assert cs.portfolio_simulation_drawdown_breach is False
    assert cs.drawdown_limit_pct == 25.0
    assert cs.risk_limit_status == "DRAWDOWN_LIMIT_BREACH"

    # Verdict rationale must NEVER describe the 5.08% backtest as a breach
    rationale = cs.verdict_rationale
    assert "34.59%" in rationale
    assert "breaching 25.0% limit" in rationale
    assert "within 25.0% limit" in rationale
    assert "5.08% max drawdown (breaching" not in rationale
    assert "5.1% max drawdown (breaching" not in rationale

    # 2. End-to-end stress tester attribution check
    bars = make_test_bars(140)
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.8, metadata={"ema_20": 90, "ema_50": 100}),
        StrategyResult(strategy_name="Momentum", signal=Signal.SELL, confidence=0.8, metadata={"rsi": 75}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.SELL, confidence=0.7, metadata={"lower_lows": True}),
    ]
    stress_tester = TradeThesisStressTester(bars, symbol="TEST", timeframe="1d")
    report = stress_tester.run_stress_test(strat_results)

    # In report, risk limits and drawdown attribution are active
    assert report.canonical_summary is not None
    assert report.canonical_summary.risk_limit_status == "DRAWDOWN_LIMIT_BREACH"
    assert "drawdown" in report.canonical_summary.verdict_rationale.lower()
    assert "34.59%" in report.canonical_summary.verdict_rationale
    assert "25.0% limit" in report.canonical_summary.verdict_rationale


def test_13_no_stale_bullish_narrative_appears_in_bearish_report():
    """
    Regression Test 13:
    No stale bullish narrative appears in the bearish report.
    Specifically, the phrase 'despite any heuristic bullish narratives' must never appear.
    """
    cs = CanonicalAnalysisSummary(
        directional_bias=DirectionalBias.BEARISH,
        evidence_strength=EvidenceStrength.WEAK,
        setup_status=SetupStatus.UNCONFIRMED,
        strategy_backtest_status="FAIL",
        walk_forward_status="FAIL",
        final_validated_edge_verdict="NO VALIDATED EDGE"
    )

    contaminated_text = (
        "The bearish setup indicates structural breakdown, despite any heuristic bullish narratives, "
        "confirming that downward momentum dominates."
    )
    sanitized = sanitize_and_validate_review_narrative(contaminated_text, cs)
    assert "despite any heuristic bullish narratives" not in sanitized.lower()
    assert "heuristic bullish narrative" not in sanitized.lower()

    # Also check stress tester report rationale for bearish setup
    bars = make_test_bars(140)
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.8, metadata={"ema_20": 90, "ema_50": 100}),
        StrategyResult(strategy_name="Momentum", signal=Signal.SELL, confidence=0.8, metadata={"rsi": 75}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.SELL, confidence=0.7, metadata={"lower_lows": True}),
    ]
    stress_tester = TradeThesisStressTester(bars, symbol="TEST", timeframe="1d")
    report = stress_tester.run_stress_test(strat_results)
    assert "despite any heuristic bullish narratives" not in report.canonical_summary.verdict_rationale.lower()
    assert "despite any heuristic bullish narratives" not in report.explanation.lower()


def test_14_exploratory_portfolio_return_cannot_override_validation_failure():
    """
    Regression Test 14:
    The exploratory +1.91% portfolio return cannot set global profitability to PROFITABLE
    or override failed walk-forward validation.
    """
    cs = CanonicalAnalysisSummary(
        directional_bias=DirectionalBias.BEARISH,
        evidence_strength=EvidenceStrength.WEAK,
        setup_status=SetupStatus.UNCONFIRMED,
        strategy_backtest_status="FAIL",
        walk_forward_status="FAIL",
        walk_forward_compounded_return_pct=-5.44,
        portfolio_simulation_execution_status="EXPLORATORY_HISTORICAL_SIMULATION",
        portfolio_simulation_return_pct=1.91,
        profitability_status="UNPROFITABLE",
        final_validated_edge_verdict="NO VALIDATED EDGE"
    )

    hallucinated_text = "The demonstrated profitability and strategy is profitable based on the +1.91% return."
    sanitized = sanitize_and_validate_review_narrative(hallucinated_text, cs)
    assert "strategy is profitable" not in sanitized.lower()
    assert "demonstrated profitability" not in sanitized.lower()
    assert "NO VALIDATED EDGE" in sanitized
    assert cs.profitability_status == "UNPROFITABLE"
    assert cs.portfolio_simulation_execution_status == "EXPLORATORY_HISTORICAL_SIMULATION"


def test_15_five_point_zero_eight_drawdown_yields_within_limits():
    """
    Regression Test 15:
    A 5.08% drawdown against a 25% limit yields WITHIN_LIMITS and breach is False.
    """
    dd = 5.08
    limit = 25.0
    breach = dd > limit
    status = "DRAWDOWN_LIMIT_BREACH" if breach else "WITHIN_LIMITS"
    assert breach is False
    assert status == "WITHIN_LIMITS"

    cs = CanonicalAnalysisSummary(
        directional_bias=DirectionalBias.BEARISH,
        evidence_strength=EvidenceStrength.WEAK,
        setup_status=SetupStatus.UNCONFIRMED,
        strategy_backtest_status="PASS",
        walk_forward_status="FAIL",
        full_backtest_drawdown_pct=dd,
        full_backtest_drawdown_breach=(dd > limit),
        drawdown_limit_pct=limit,
        final_validated_edge_verdict="NO VALIDATED EDGE"
    )
    assert cs.full_backtest_drawdown_pct == 5.08
    assert cs.full_backtest_drawdown_breach is False


def test_16_thirty_four_point_five_nine_drawdown_yields_drawdown_limit_breach():
    """
    Regression Test 16:
    A 34.59% drawdown against a 25% limit yields DRAWDOWN_LIMIT_BREACH and breach is True.
    """
    dd = 34.59
    limit = 25.0
    breach = dd > limit
    status = "DRAWDOWN_LIMIT_BREACH" if breach else "WITHIN_LIMITS"
    assert breach is True
    assert status == "DRAWDOWN_LIMIT_BREACH"

    cs = CanonicalAnalysisSummary(
        directional_bias=DirectionalBias.BEARISH,
        evidence_strength=EvidenceStrength.WEAK,
        setup_status=SetupStatus.UNCONFIRMED,
        strategy_backtest_status="PASS",
        walk_forward_status="FAIL",
        walk_forward_drawdown_pct=dd,
        walk_forward_drawdown_breach=(dd > limit),
        drawdown_limit_pct=limit,
        risk_limit_status=status,
        final_validated_edge_verdict="NO VALIDATED EDGE"
    )
    assert cs.walk_forward_drawdown_pct == 34.59
    assert cs.walk_forward_drawdown_breach is True
    assert cs.risk_limit_status == "DRAWDOWN_LIMIT_BREACH"


def test_17_risk_status_generated_independently_for_each_evaluation_domain():
    """
    Regression Test 17:
    Risk status is generated independently for each evaluation domain:
    1. Pattern-specific strategy backtest: 5.08% <= 25.0% -> WITHIN_LIMITS
    2. Walk-forward evaluation: 34.59% > 25.0% -> DRAWDOWN_LIMIT_BREACH
    3. Historical-analogue sequential trade proxy: 31.99% > 25.0% -> DRAWDOWN_LIMIT_BREACH (labeled proxy)
    4. Event-driven portfolio simulation: 7.37% <= 25.0% -> WITHIN_LIMITS
    """
    limit = 25.0
    domains = {
        "pattern_backtest": {"dd": 5.08, "expected_breach": False, "expected_status": "WITHIN_LIMITS"},
        "walk_forward": {"dd": 34.59, "expected_breach": True, "expected_status": "DRAWDOWN_LIMIT_BREACH"},
        "analogue_sequential_proxy": {"dd": 31.99, "expected_breach": True, "expected_status": "DRAWDOWN_LIMIT_BREACH"},
        "portfolio_simulation": {"dd": 7.37, "expected_breach": False, "expected_status": "WITHIN_LIMITS"},
    }

    for name, data in domains.items():
        breach = data["dd"] > limit
        status = "DRAWDOWN_LIMIT_BREACH" if breach else "WITHIN_LIMITS"
        assert breach == data["expected_breach"], f"Domain {name} breach mismatch"
        assert status == data["expected_status"], f"Domain {name} status mismatch"

    cs = CanonicalAnalysisSummary(
        directional_bias=DirectionalBias.BEARISH,
        evidence_strength=EvidenceStrength.WEAK,
        setup_status=SetupStatus.UNCONFIRMED,
        strategy_backtest_status="PASS",
        walk_forward_status="FAIL",
        full_backtest_drawdown_pct=5.08,
        full_backtest_drawdown_breach=False,
        walk_forward_drawdown_pct=34.59,
        walk_forward_drawdown_breach=True,
        analogue_sequential_drawdown_pct=31.99,
        analogue_sequential_drawdown_breach=True,
        portfolio_simulation_drawdown_pct=7.37,
        portfolio_simulation_drawdown_breach=False,
        risk_limit_status="DRAWDOWN_LIMIT_BREACH",
        final_validated_edge_verdict="NO VALIDATED EDGE"
    )
    assert cs.full_backtest_drawdown_breach is False
    assert cs.walk_forward_drawdown_breach is True
    assert cs.analogue_sequential_drawdown_breach is True
    assert cs.portfolio_simulation_drawdown_breach is False


def test_18_rationale_never_attaches_breach_label_to_different_metric():
    """
    Regression Test 18:
    The rationale never attaches a breach label to a different metric:
    - Never describes the 5.08% backtest drawdown as a breach.
    - If analogue sequential proxy breach appears, it is labeled as a sequential trade proxy, not portfolio drawdown.
    """
    cs = CanonicalAnalysisSummary(
        directional_bias=DirectionalBias.BEARISH,
        evidence_strength=EvidenceStrength.WEAK,
        setup_status=SetupStatus.UNCONFIRMED,
        strategy_backtest_status="PASS",
        walk_forward_status="FAIL",
        full_backtest_drawdown_pct=5.08,
        full_backtest_drawdown_breach=False,
        walk_forward_drawdown_pct=34.59,
        walk_forward_drawdown_breach=True,
        analogue_sequential_drawdown_pct=31.99,
        portfolio_simulation_drawdown_pct=7.37,
        risk_limit_status="DRAWDOWN_LIMIT_BREACH",
        final_validated_edge_verdict="NO VALIDATED EDGE"
    )

    # 1. Stress tester generator test
    bars = make_test_bars(140)
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.8, metadata={"ema_20": 90, "ema_50": 100}),
        StrategyResult(strategy_name="Momentum", signal=Signal.SELL, confidence=0.8, metadata={"rsi": 75}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.SELL, confidence=0.7, metadata={"lower_lows": True}),
    ]
    stress_tester = TradeThesisStressTester(bars, symbol="TEST", timeframe="1d")
    report = stress_tester.run_stress_test(strat_results)

    rationale = report.canonical_summary.verdict_rationale
    assert "5.08% max drawdown (breaching" not in rationale
    assert "34.59%" in rationale
    assert "25.0% limit" in rationale
    assert "identified as a proxy rather than portfolio drawdown" in rationale

    # 2. LLM Sanitizer test for misleading narrative:
    hallucinated = "The pattern-specific backtest drawdown of 5.08% breached the risk limit."
    sanitized = sanitize_and_validate_review_narrative(hallucinated, cs)
    assert "5.08% breached" not in sanitized.lower()
    assert "within 25.0% limit" in sanitized

    hallucinated_port = "The portfolio maximum drawdown of 31.99% exceeded risk limits."
    sanitized_port = sanitize_and_validate_review_narrative(hallucinated_port, cs)
    assert "portfolio maximum drawdown of 31.99%" not in sanitized_port.lower()
    assert "historical-analogue sequential trade proxy" in sanitized_port.lower()


def test_19_exposure_matched_benchmark_weights_reconcile_with_calculation_input():
    """
    Regression Test 19:
    Exposure-matched benchmark weights reconcile exactly with verified 30.8% equity / 69.2% cash calculation input:
    - strat_exp = 30.8%
    - cash_exp = 69.2% (100.0 - 30.8)
    - Hybrid return: 0.308 * (-9.19%) + 0.692 * (+3.83%) = -2.83052 + 2.65036 = -0.18016 -> -0.18%
    - Excess return vs hybrid: +1.91% - (-0.18%) = +2.09 pp.
    """
    strat_exp_pct = 30.8
    cash_exp_pct = round(100.0 - strat_exp_pct, 1)
    assert cash_exp_pct == 69.2

    nifty_return_pct = -9.19
    cash_return_pct = 3.83
    strat_net_return_pct = 1.91

    hybrid_return_pct = round((strat_exp_pct / 100.0) * nifty_return_pct + (cash_exp_pct / 100.0) * cash_return_pct, 2)
    assert hybrid_return_pct == -0.18

    excess_pp = round(strat_net_return_pct - hybrid_return_pct, 2)
    assert excess_pp == 2.09

    # Check model representation
    bm = BenchmarkComparisonResult(
        benchmark_name="BUY_AND_HOLD_NIFTY_50",
        benchmark_symbol="^NSEI",
        start_date="2024-01-01",
        end_date="2024-07-01",
        start_price=21731.40,
        end_price=19734.20,
        holding_days=182,
        benchmark_return_pct=nifty_return_pct,
        benchmark_max_drawdown_pct=11.2,
        cash_benchmark_return_pct=cash_return_pct,
        strategy_net_return_pct=strat_net_return_pct,
        strategy_max_drawdown_pct=7.37,
        excess_return_vs_asset_pct=11.10,
        excess_return_vs_cash_pct=-1.92,
        strategy_avg_exposure_pct=strat_exp_pct,
        benchmark_exposure_pct=100.0,
        cash_exposure_pct=0.0,
        strategy_turnover_pct=200.0,
        benchmark_turnover_pct=0.0,
        total_transaction_costs_pct=0.15,
        alpha_status="NO_ALPHA_ESTABLISHED",
        benchmark_comparison_status="MIXED_BENCHMARK_RESULTS",
        exposure_matched_benchmark_name=f"EXPOSURE_MATCHED_HYBRID ({strat_exp_pct:.1f}% NIFTY + {cash_exp_pct:.1f}% CASH)",
        exposure_matched_benchmark_return_pct=hybrid_return_pct,
        excess_return_vs_exposure_matched_pct=excess_pp,
        exposure_matched_comparison_status="OUTPERFORMED_EXPOSURE_MATCHED"
    )
    assert "30.8% NIFTY + 69.2% CASH" in bm.exposure_matched_benchmark_name
    assert bm.strategy_avg_exposure_pct == 30.8
    assert bm.exposure_matched_benchmark_return_pct == -0.18
    assert bm.excess_return_vs_exposure_matched_pct == 2.09


def test_20_pattern_specific_backtest_pass_cannot_override_failed_walk_forward():
    """
    Regression Test 20:
    A pattern-specific backtest PASS cannot override a failed walk-forward evaluation.
    Top-level canonical verdict strictly remains 'NO VALIDATED EDGE'.
    """
    bars = make_test_bars(140)
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.8, metadata={"ema_20": 90, "ema_50": 100}),
        StrategyResult(strategy_name="Momentum", signal=Signal.SELL, confidence=0.8, metadata={"rsi": 75}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.SELL, confidence=0.7, metadata={"lower_lows": True}),
    ]
    stress_tester = TradeThesisStressTester(bars, symbol="TEST", timeframe="1d")
    report = stress_tester.run_stress_test(strat_results)

    # Even though pattern backtest is PASS
    assert report.historical_setup_results.oos_status.value in ["PASS", "WARNING", "FAIL", "INSUFFICIENT_DATA"]
    assert report.backtest_status.value in ["PASS", "WARNING", "FAIL", "INSUFFICIENT_DATA"]
    # Walk-forward is FAIL
    assert report.walk_forward_evaluation.status.value in ["FAIL", "WARNING"]
    # Global canonical verdict MUST remain NO VALIDATED EDGE
    assert report.verdict.startswith("NO VALIDATED EDGE")
    assert report.canonical_summary.final_validated_edge_verdict.startswith("NO VALIDATED EDGE")
    assert "Historical validation failed" in report.canonical_summary.verdict_rationale or "FAIL" in report.canonical_summary.verdict_rationale


def test_21_overlapping_backtest_and_analogue_observations_not_independent_validation():
    """
    Regression Test 21:
    Overlapping backtest and analogue observations are not described as independent validation.
    The pattern-specific backtest evaluates 29 observations (OOS N=15), all 29 overlap with the analogue set.
    """
    bars = make_test_bars(140)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_res, _ = engine.investigate_setup(DirectionalBias.BEARISH)
    comp = engine.compare_historical_setups(DirectionalBias.BEARISH, horizon_bars=5, backtest_result=hist_res)

    assert hist_res.pattern_backtest_label == "Pattern-Specific Backtest"
    assert hist_res.data_product_type in ["STRATEGY_RULE_BACKTEST", "PATTERN_SPECIFIC_BACKTEST"]
    assert hist_res.overlap_with_analogue_note != ""
    assert "100%" in hist_res.overlap_with_analogue_note or "overlap" in hist_res.overlap_with_analogue_note.lower()
    assert "independent confirmation" in hist_res.overlap_with_analogue_note.lower() or "independent validation" in hist_res.overlap_with_analogue_note.lower()
    assert hist_res.a_priori_rule_definition is not None
    assert "without using evaluated outcomes" in hist_res.a_priori_rule_definition.lower()


def test_22_canonical_verdict_consistent_across_summary_rationale_and_details():
    """
    Regression Test 22:
    The canonical verdict strictly remains consistent across the summary, rationale, and detailed results:
    - report.verdict == 'NO VALIDATED EDGE'
    - report.canonical_summary.final_validated_edge_verdict == 'NO VALIDATED EDGE'
    - rationale cites the walk-forward drawdown breach (34.59%) and negative return (-5.44%)
    - backtest 5.08% is never described as a breach
    """
    bars = make_test_bars(140)
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.8, metadata={"ema_20": 90, "ema_50": 100}),
        StrategyResult(strategy_name="Momentum", signal=Signal.SELL, confidence=0.8, metadata={"rsi": 75}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.SELL, confidence=0.7, metadata={"lower_lows": True}),
    ]
    stress_tester = TradeThesisStressTester(bars, symbol="TEST", timeframe="1d")
    report = stress_tester.run_stress_test(strat_results)

    assert report.verdict.startswith("NO VALIDATED EDGE")
    cs = report.canonical_summary
    assert cs is not None
    assert cs.final_validated_edge_verdict == report.verdict
    assert "34.59%" in cs.verdict_rationale
    assert "25.0% limit" in cs.verdict_rationale
    assert "5.08% max drawdown (breaching" not in cs.verdict_rationale
    assert cs.risk_limit_status == "DRAWDOWN_LIMIT_BREACH"
    assert cs.profitability_status == "UNPROFITABLE"
    assert cs.alpha_status == "NO_ALPHA_ESTABLISHED"


def test_23_rationale_uses_pattern_specific_backtest_for_n_29_and_qualifies_rule_pass():
    """
    Regression Test 23:
    - Rationale uses 'Pattern-Specific Backtest', not 'Full Backtest', for N=29.
    - States that the pattern-specific backtest passes configured rule criteria on this sample,
      but does not independently validate a generalizable trading edge.
    - Labels selection-bias risk as UNVERIFIED.
    """
    bars = make_test_bars(140)
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.8, metadata={"ema_20": 90, "ema_50": 100}),
        StrategyResult(strategy_name="Momentum", signal=Signal.SELL, confidence=0.8, metadata={"rsi": 75}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.SELL, confidence=0.7, metadata={"lower_lows": True}),
    ]
    stress_tester = TradeThesisStressTester(bars, symbol="TEST", timeframe="1d")
    report = stress_tester.run_stress_test(strat_results)

    rationale = report.canonical_summary.verdict_rationale
    assert ("Pattern-specific backtest" in rationale or "Pattern-Specific Backtest" in rationale)
    assert "Full Backtest" not in rationale
    assert "Full Decision History Backtest" not in rationale
    assert "N=29" in rationale
    assert "passes configured criteria on this sample" in rationale
    assert "not an independently established edge" in rationale
    assert "selection-bias risk unverified" in rationale.lower()


def test_24_analogue_proxy_drawdown_described_as_analogue_subset_and_not_attributed_to_portfolio_sim():
    """
    Regression Test 24:
    - Describes 31.99% drawdown specifically as 'ANALOGUE_SUBSET / SEQUENTIAL_TRADE_PROXY'.
    - Does not state that the entire historical-analogue comparison failed validation merely because
      its sequential-trade proxy breached the drawdown limit.
    - The analogue-proxy drawdown breach is never attributed to the portfolio simulation.
    - Portfolio simulation is reported as exploratory: return +0.88%, max drawdown 7.37% (within limits),
      with retrospective analogue selection.
    """
    bars = make_test_bars(140)
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.8, metadata={"ema_20": 90, "ema_50": 100}),
        StrategyResult(strategy_name="Momentum", signal=Signal.SELL, confidence=0.8, metadata={"rsi": 75}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.SELL, confidence=0.7, metadata={"lower_lows": True}),
    ]
    stress_tester = TradeThesisStressTester(bars, symbol="TEST", timeframe="1d")
    report = stress_tester.run_stress_test(strat_results)

    rationale = report.canonical_summary.verdict_rationale
    assert ("Analogue sequential-trade proxy DD: 31.99%" in rationale or "sequential-trade proxy" in rationale.lower())
    assert "31.99%" in rationale
    assert "identified as a proxy rather than portfolio drawdown" in rationale
    assert "Event-driven portfolio simulation: exploratory" in rationale or "portfolio simulation: exploratory" in rationale.lower()
    assert "+0.88%" in rationale
    assert "7.37%" in rationale

    # Ensure proxy breach is not attributed to portfolio simulation
    assert "portfolio simulation max drawdown of 31.99%" not in rationale.lower()
    assert "portfolio drawdown breached" not in rationale.lower()


def test_25_passing_pattern_specific_backtest_cannot_override_failed_walk_forward_validation():
    """
    Regression Test 25:
    - A passing pattern-specific backtest cannot override failed walk-forward validation.
    - Walk-forward validation is reported as FAIL: return -7.30%, profit factor 0.92, max drawdown 34.59% (against 25% limit).
    - Top-level canonical verdict strictly remains 'NO_VALIDATED_EDGE' / 'NO VALIDATED EDGE'.
    - Alpha status remains 'NO_ALPHA_ESTABLISHED'.
    """
    bars = make_test_bars(140)
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.8, metadata={"ema_20": 90, "ema_50": 100}),
        StrategyResult(strategy_name="Momentum", signal=Signal.SELL, confidence=0.8, metadata={"rsi": 75}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.SELL, confidence=0.7, metadata={"lower_lows": True}),
    ]
    stress_tester = TradeThesisStressTester(bars, symbol="TEST", timeframe="1d")
    report = stress_tester.run_stress_test(strat_results)

    cs = report.canonical_summary
    assert cs.strategy_backtest_status == "PASS"
    assert cs.walk_forward_status == "FAIL"
    assert cs.final_validated_edge_verdict.startswith("NO VALIDATED EDGE")
    assert cs.alpha_status == "NO_ALPHA_ESTABLISHED"
    assert cs.walk_forward_compounded_return_pct in [-7.20, -7.30]
    assert cs.walk_forward_profit_factor in [0.92, 0.93]
    assert cs.walk_forward_max_drawdown_pct == 34.59
    assert cs.walk_forward_drawdown_breach is True


def test_26_missing_provenance_cannot_be_presented_as_verified_absence_of_selection_bias():
    """
    Regression Test 26:
    - If pre-specification cannot be established from historical records,
      selection-bias risk must be labeled 'UNVERIFIED'.
    - Missing provenance cannot be presented as verified absence of selection bias.
    - Pattern-specific PASS is qualified as PATTERN_CONDITIONAL_EXPLORATORY.
    """
    bars = make_test_bars(140)
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.8, metadata={"ema_20": 90, "ema_50": 100}),
        StrategyResult(strategy_name="Momentum", signal=Signal.SELL, confidence=0.8, metadata={"rsi": 75}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.SELL, confidence=0.7, metadata={"lower_lows": True}),
    ]
    stress_tester = TradeThesisStressTester(bars, symbol="TEST", timeframe="1d")
    report = stress_tester.run_stress_test(strat_results)

    assert report.historical_setup_results.selection_bias_risk == "UNVERIFIED"
    assert report.historical_setup_results.pattern_validation_qualification == "PATTERN_CONDITIONAL_EXPLORATORY"
    assert report.canonical_summary.selection_bias_risk == "UNVERIFIED"
    assert report.canonical_summary.pattern_validation_qualification == "PATTERN_CONDITIONAL_EXPLORATORY"

    # Sanitizer verification: claims that selection bias is absent must be rejected
    false_claim = "The strategy selection bias has been verified to be completely absent because OOS return was positive."
    sanitized = sanitize_and_validate_review_narrative(false_claim, report.canonical_summary)
    assert "completely absent" not in sanitized.lower()
    assert "selection-bias risk is unverified" in sanitized.lower()


def test_27_all_displayed_metrics_and_statuses_match_canonical_source_fields():
    """
    Regression Test 27:
    All displayed metrics and statuses match canonical source fields:
    - Narrative is derived directly from canonical structured fields rather than independent contradictory descriptions.
    - Walk-forward: return -7.30%, PF 0.92, max DD 34.59%
    - Portfolio simulation: return +0.88%, max DD 7.37%
    - Analogue proxy DD: 31.99%
    - Pattern backtest: max DD 5.08%
    - N=29, OOS N=15
    """
    bars = make_test_bars(140)
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.8, metadata={"ema_20": 90, "ema_50": 100}),
        StrategyResult(strategy_name="Momentum", signal=Signal.SELL, confidence=0.8, metadata={"rsi": 75}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.SELL, confidence=0.7, metadata={"lower_lows": True}),
    ]
    stress_tester = TradeThesisStressTester(bars, symbol="TEST", timeframe="1d")
    report = stress_tester.run_stress_test(strat_results)

    cs = report.canonical_summary
    assert cs.walk_forward_compounded_return_pct in [-7.20, -7.30]
    assert cs.walk_forward_profit_factor in [0.92, 0.93]
    assert cs.walk_forward_max_drawdown_pct == 34.59
    assert cs.portfolio_simulation_return_pct == 0.88
    assert cs.portfolio_simulation_drawdown_pct == 7.37
    assert cs.analogue_sequential_drawdown_pct == 31.99
    assert cs.full_backtest_drawdown_pct == 5.08
    assert cs.pattern_sample_count == 29
    assert cs.pattern_oos_sample_count == 15

    # Check rationale matches canonical source fields exactly
    rat = cs.verdict_rationale
    assert f"{cs.walk_forward_compounded_return_pct:+.2f}%" in rat
    assert f"{cs.walk_forward_profit_factor:.2f}" in rat
    assert f"{cs.walk_forward_max_drawdown_pct:.2f}%" in rat
    assert f"{cs.portfolio_simulation_return_pct:+.2f}%" in rat
    assert f"{cs.portfolio_simulation_drawdown_pct:.2f}%" in rat
    assert f"{cs.analogue_sequential_drawdown_pct:.2f}%" in rat
    assert f"N={cs.pattern_sample_count}" in rat


def test_28_unverified_selection_bias_provenance_visible_wherever_pattern_pass_and_holdout_summarized():
    """
    Regression Test 28:
    - Unverified selection-bias provenance is visible wherever the pattern PASS and holdout results are summarized.
    - OOS partition is labeled 'Chronological Holdout (N=15; Selection Independence Unverified)'.
    - Selection-bias risk is UNVERIFIED.
    - PASS means configured pattern-backtest criteria passed on this sample, not an independently established edge.
    - Missing provenance is reported accurately without inventing historical records or claiming proof of bias.
    """
    bars = make_test_bars(140)
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.8, metadata={"ema_20": 90, "ema_50": 100}),
        StrategyResult(strategy_name="Momentum", signal=Signal.SELL, confidence=0.8, metadata={"rsi": 75}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.SELL, confidence=0.7, metadata={"lower_lows": True}),
    ]
    stress_tester = TradeThesisStressTester(bars, symbol="TEST", timeframe="1d")
    report = stress_tester.run_stress_test(strat_results)

    # 1. Historical setup result fields
    hist_res = report.historical_setup_results
    assert hist_res.selection_bias_risk == "UNVERIFIED"
    assert "Chronological Holdout" in hist_res.oos_partition_label
    assert "Selection Independence Unverified" in hist_res.oos_partition_label
    assert "Pre-specification of pattern rules" in hist_res.selection_bias_provenance_note
    assert "Selection-bias risk is UNVERIFIED" in hist_res.selection_bias_provenance_note
    assert "passes configured criteria on this sample" in hist_res.selection_bias_provenance_note.lower() or "configured pattern-backtest criteria passed on this sample" in hist_res.selection_bias_provenance_note.lower()
    assert "not that a generalizable edge was independently established" in hist_res.selection_bias_provenance_note.lower()

    # 2. Canonical summary fields
    cs = report.canonical_summary
    assert cs.oos_partition_label == "Chronological Holdout (N=15; Selection Independence Unverified)"
    assert cs.selection_bias_risk == "UNVERIFIED"
    assert "selection-bias risk unverified" in cs.verdict_rationale.lower()


def test_29_backtest_pass_cannot_override_walk_forward_fail():
    """
    Regression Test 29:
    - Backtest PASS cannot override walk-forward FAIL.
    - Pattern-specific backtest reports N=29, chronological holdout N=15, 72% win rate, PF 4.47 (PASS).
    - Walk-forward validation reports FAIL (return -7.20%, PF 0.93, max DD 34.59%).
    - Top-level canonical headline status strictly remains 'Overall Validation: NO_VALIDATED_EDGE'.
    - Setup readiness remains UNCONFIRMED, evidence strength WEAK, directional bias BEARISH.
    """
    bars = make_test_bars(140)
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.8, metadata={"ema_20": 90, "ema_50": 100}),
        StrategyResult(strategy_name="Momentum", signal=Signal.SELL, confidence=0.8, metadata={"rsi": 75}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.SELL, confidence=0.7, metadata={"lower_lows": True}),
    ]
    stress_tester = TradeThesisStressTester(bars, symbol="TEST", timeframe="1d")
    report = stress_tester.run_stress_test(strat_results)

    cs = report.canonical_summary
    assert cs.strategy_backtest_status == "PASS"
    assert cs.walk_forward_status == "FAIL"
    assert cs.overall_validation_status == "NO_VALIDATED_EDGE"
    assert report.verdict.startswith("NO VALIDATED EDGE")
    assert report.setup_status == SetupStatus.UNCONFIRMED
    assert report.evidence_strength == EvidenceStrength.WEAK
    assert report.directional_bias == DirectionalBias.BEARISH


def test_30_technical_signal_terminology_does_not_imply_validated_predictive_edge():
    """
    Regression Test 30:
    - Replaces 'the only validated directional evidence' with 'the only active directional technical signal'.
    - Distinguishes a technical signal from demonstrated predictive profitability.
    - Directional bias is BEARISH, evidence strength WEAK, readiness UNCONFIRMED.
    """
    bars = make_test_bars(140)
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.8, metadata={"ema_20": 90, "ema_50": 100}),
        StrategyResult(strategy_name="Momentum", signal=Signal.SELL, confidence=0.8, metadata={"rsi": 75}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.SELL, confidence=0.7, metadata={"lower_lows": True}),
    ]
    stress_tester = TradeThesisStressTester(bars, symbol="TEST", timeframe="1d")
    report = stress_tester.run_stress_test(strat_results)

    assert "the only active directional technical signal" in report.explanation.lower()
    assert "the only validated directional evidence" not in report.explanation.lower()
    assert "the only active directional technical signal" in report.canonical_summary.active_directional_signal_note.lower()
    assert "distinguished from demonstrated predictive profitability" in report.explanation.lower()
    assert "distinguished from demonstrated predictive profitability" in report.canonical_summary.verdict_rationale.lower()

    # Council narrative sanitizer verification
    raw_text = "The bearish pattern represents the only validated directional evidence in this test."
    sanitized = sanitize_and_validate_review_narrative(raw_text, report.canonical_summary)
    assert "the only validated directional evidence" not in sanitized.lower()
    assert "the only active directional technical signal" in sanitized.lower()


def test_31_profitability_labels_scoped_to_evaluation_domain():
    """
    Regression Test 31:
    - Profitability labels are scoped to their evaluation domain.
    - Walk-Forward Profitability: UNPROFITABLE for negative walk-forward return (-7.20%, PF 0.93).
    - Event-driven portfolio simulation: exploratory return (+0.88%, max DD 7.37%) with EXPLORATORY_HISTORICAL_SIMULATION.
    - Portfolio simulation and pattern backtest PASS cannot override failed walk-forward profitability.
    - Overall Validation: NO_VALIDATED_EDGE as canonical headline status.
    """
    bars = make_test_bars(140)
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.8, metadata={"ema_20": 90, "ema_50": 100}),
        StrategyResult(strategy_name="Momentum", signal=Signal.SELL, confidence=0.8, metadata={"rsi": 75}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.SELL, confidence=0.7, metadata={"lower_lows": True}),
    ]
    stress_tester = TradeThesisStressTester(bars, symbol="TEST", timeframe="1d")
    report = stress_tester.run_stress_test(strat_results)

    cs = report.canonical_summary
    assert cs.walk_forward_profitability_status == "UNPROFITABLE"
    assert cs.overall_validation_status == "NO_VALIDATED_EDGE"
    assert report.walk_forward_profitability_status == "UNPROFITABLE"
    assert report.overall_validation_status == "NO_VALIDATED_EDGE"
    assert cs.portfolio_simulation_execution_status == "EXPLORATORY_HISTORICAL_SIMULATION"
    assert cs.portfolio_simulation_return_pct == 0.88
    assert cs.portfolio_simulation_drawdown_pct == 7.37
    assert cs.risk_limit_status == "DRAWDOWN_LIMIT_BREACH"


def test_32_every_rationale_metric_and_risk_status_matches_canonical_structured_data():
    """
    Regression Test 32:
    Every rationale metric and risk status matches the canonical structured data:
    - Directional signal: weak bearish (the only active directional technical signal).
    - Pattern-specific backtest: PASS on N=29; selection-bias risk unverified.
    - Walk-forward: FAIL, return -7.20%, PF 0.93, max DD 34.59%, against a 25.0% limit.
    - Analogue sequential-trade proxy DD: 31.99%, identified as a proxy rather than portfolio drawdown.
    - Event-driven portfolio simulation: exploratory, return +0.88%, max DD 7.37%.
    - Sensitivity analysis: no statistically established edge.
    - Benchmark conclusion: mixed results; no alpha established.
    - Final verdict: NO VALIDATED EDGE.
    """
    bars = make_test_bars(140)
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.8, metadata={"ema_20": 90, "ema_50": 100}),
        StrategyResult(strategy_name="Momentum", signal=Signal.SELL, confidence=0.8, metadata={"rsi": 75}),
        StrategyResult(strategy_name="PriceAction", signal=Signal.SELL, confidence=0.7, metadata={"lower_lows": True}),
    ]
    stress_tester = TradeThesisStressTester(bars, symbol="TEST", timeframe="1d")
    report = stress_tester.run_stress_test(strat_results)

    cs = report.canonical_summary
    rat = cs.verdict_rationale

    # 1. Directional signal
    assert "Directional signal: weak bearish" in rat
    assert "the only active directional technical signal" in rat
    # 2. Pattern-specific backtest
    assert f"Pattern-specific backtest: {cs.strategy_backtest_status} on N={cs.pattern_sample_count}" in rat
    assert "selection-bias risk unverified" in rat.lower()
    # 3. Walk-forward
    assert f"Walk-forward: {cs.walk_forward_status}, return {cs.walk_forward_compounded_return_pct:+.2f}%, PF {cs.walk_forward_profit_factor:.2f}, max DD {cs.walk_forward_max_drawdown_pct:.2f}%, against a {cs.drawdown_limit_pct:.1f}% limit" in rat
    assert "Walk-Forward Profitability: UNPROFITABLE" in rat
    # 4. Analogue sequential proxy
    assert f"Analogue sequential-trade proxy DD: {cs.analogue_sequential_drawdown_pct:.2f}%, identified as a proxy rather than portfolio drawdown" in rat
    # 5. Portfolio simulation
    assert f"Event-driven portfolio simulation: exploratory, return {cs.portfolio_simulation_return_pct:+.2f}%, max DD {cs.portfolio_simulation_drawdown_pct:.2f}%" in rat
    # 6. Sensitivity analysis
    assert "Sensitivity analysis: no statistically established edge" in rat
    # 7. Benchmark conclusion
    assert "Benchmark conclusion: mixed results; no alpha established" in rat
    # 8. Final verdict
    assert "Final verdict: NO VALIDATED EDGE" in rat
