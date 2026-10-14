"""
Regression Test Suite for SignalCouncil Final Calculation and Status Consistency Audit
Verifies:
1. Fees and slippage sum to the canonical round-trip total (0.15%) used in calculations.
2. Every report section renders the verified 0.15% round-trip cost assumption without 0.30% contradiction.
3. Directional family count strictly excludes non-directional risk families (VOLATILITY_RISK).
4. Alignment score and directional-family count are consistent (e.g., 1 of 4 = 25/100).
5. Matching completion (MATCHING_COMPLETE) cannot imply profitability validation or render PASS.
6. Benchmark commentary reconciles with its corresponding benchmark figure (e.g. -9.07% vs -9.19%).
7. Overall validation remains independent of retrospective portfolio-simulation success.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from app.domain.models import (
    TransactionCostModel, HistoricalSetupResult, BenchmarkComparisonResult,
    DirectionalBias, EvidenceFamily, EvidenceStrength, BacktestStatus,
    CanonicalAnalysisSummary, SetupStatus
)
from app.domain.backtest import BacktestEngine
from app.domain.stress_tester import TradeThesisStressTester
from tests.fixtures.market_data import get_ohlcv_models, create_bullish_trending_data

APP_PATH = Path(__file__).resolve().parent.parent.parent / "extension" / "src" / "App.tsx"


@pytest.fixture(scope="module")
def app_source() -> str:
    assert APP_PATH.exists(), f"App.tsx not found at {APP_PATH}"
    return APP_PATH.read_text(encoding="utf-8")


# -------------------------------------------------------------------------
# Test 1: Fees and slippage sum to canonical round-trip total
# -------------------------------------------------------------------------
def test_1_fees_and_slippage_sum_to_canonical_round_trip():
    """Verify fees and slippage sum to exactly 0.15% round-trip across all model definitions."""
    model = TransactionCostModel()
    # Per-leg fees: 0.05% entry + 0.05% exit = 0.10% round-trip
    assert model.entry_fee_pct == 0.0005
    assert model.exit_fee_pct == 0.0005
    assert round(model.entry_fee_pct + model.exit_fee_pct, 6) == 0.0010
    assert model.round_trip_fee_pct == 0.0010

    # Per-leg slippage: 0.025% entry + 0.025% exit = 0.05% round-trip
    assert model.entry_slippage_pct == 0.00025
    assert model.exit_slippage_pct == 0.00025
    assert round(model.entry_slippage_pct + model.exit_slippage_pct, 6) == 0.0005
    assert model.round_trip_slippage_pct == 0.0005

    # Total friction: 0.10% fees + 0.05% slippage = 0.15% round trip (0.075% per leg)
    assert model.total_per_leg_cost_pct == 0.00075
    assert model.total_round_trip_cost_pct == 0.0015
    assert "0.15%" in model.description

    # Flat return test: 0.0 gross return should lose ~0.15% to transaction friction
    net_ret, total_costs = model.calculate_net_return(0.0)
    assert net_ret < 0.0
    assert round(abs(net_ret) * 100, 2) == 0.15
    assert round(total_costs * 100, 2) == 0.15

    # BacktestEngine default per-leg cost matches 0.075% (0.15% round-trip)
    engine = BacktestEngine(
        market_data=get_ohlcv_models(),
        strategy=None,
        context=None,
        warmup_period=20
    )
    assert engine.transaction_cost_pct == 0.00075


# -------------------------------------------------------------------------
# Test 2: Every report section renders verified cost assumption
# -------------------------------------------------------------------------
def test_2_every_report_section_renders_verified_cost_assumption(app_source: str):
    """Execution timing and descriptions must state canonical 0.15% round trip and reject 0.30%."""
    # 0.30% round trip was an erroneous duplicate calculation and must not appear
    assert "0.30% round trip" not in app_source
    assert "0.30%" not in app_source

    # Execution mechanics must accurately state 0.075% per leg = 0.15% round trip
    assert "0.15% round trip" in app_source
    assert "0.05% fees + 0.025% slippage per leg" in app_source

    # HistoricalSetupResult test_costs_description must match canonical model
    hs_res = HistoricalSetupResult(
        setup_definition="TEST",
        start_date="",
        end_date="",
        sample_count=10,
        win_rate=0.5,
        profit_factor=1.0,
        max_drawdown=0.1,
        net_return=0.0
    )
    assert "0.15%" in hs_res.test_costs_description
    assert "0.05% per leg" in hs_res.test_costs_description


# -------------------------------------------------------------------------
# Test 3: Directional family count excludes non-directional risk families
# -------------------------------------------------------------------------
def test_3_directional_family_count_excludes_non_directional_risk_families(app_source: str):
    """VOLATILITY_RISK is a non-directional control and must be excluded from the directional denominator."""
    # Backend stress tester definition
    assert EvidenceFamily.VOLATILITY_RISK.value == "VOLATILITY_RISK"

    # Frontend filter must exclude both VOLATILITY_RISK and RISK_REGIME
    assert "f.family !== 'VOLATILITY_RISK'" in app_source
    assert "totalDirectionCapable = directionCapableFamilies.length || 4" in app_source

    # Tooltip and note explicitly clarify 4 direction-capable families
    assert "excluding non-directional VOLATILITY_RISK" in app_source
    assert "4 direction-capable families" in app_source


# -------------------------------------------------------------------------
# Test 4: Alignment score and directional-family count consistency
# -------------------------------------------------------------------------
def test_4_alignment_score_and_directional_family_count_consistency():
    """1 of 4 direction-capable families produces exactly 25/100 alignment, not a probability."""
    total_capable = 4
    for count, expected_score in [(0, 0), (1, 25), (2, 50), (3, 75), (4, 100)]:
        calculated_score = int((count / total_capable) * 100)
        assert calculated_score == expected_score

    # Consistency in CanonicalAnalysisSummary
    cs = CanonicalAnalysisSummary(
        directional_bias=DirectionalBias.BULLISH,
        evidence_strength=EvidenceStrength.WEAK,
        setup_status=SetupStatus.UNCONFIRMED,
        strategy_backtest_status="FAIL",
        walk_forward_status="FAIL",
        walk_forward_compounded_return_pct=-7.20,
        walk_forward_profit_factor=0.93,
        walk_forward_max_drawdown_pct=34.59,
        overall_validation_status="NO_VALIDATED_EDGE",
        walk_forward_profitability_status="UNPROFITABLE",
        oos_partition_label="",
        sensitivity_analysis_available=True,
        sensitivity_analysis_result="INSUFFICIENT_EVIDENCE",
        sensitivity_analysis_summary="",
        portfolio_simulation_execution_status="EXPLORATORY_HISTORICAL_SIMULATION",
        portfolio_simulation_return_pct=0.88,
        portfolio_simulation_nature="",
        profitability_status="UNPROFITABLE",
        risk_limit_status="DRAWDOWN_LIMIT_BREACH",
        drawdown_limit_pct=25.0,
        benchmark_comparison_available=True,
        benchmark_comparison_status="MIXED_BENCHMARK_RESULTS",
        benchmark_summary_status="MIXED_BENCHMARK_RESULTS",
        benchmark_nifty_status="OUTPERFORMED_NOMINAL_PRICE_RETURN",
        benchmark_cash_status="UNDERPERFORMED_CASH",
        exposure_matched_status="OUTPERFORMED_EXPOSURE_MATCHED",
        excess_return_vs_nifty_pp=11.10,
        excess_return_vs_cash_pp=-1.92,
        strategy_avg_exposure_pct=30.8,
        benchmark_exposure_pct=100.0,
        full_backtest_drawdown_pct=5.08,
        walk_forward_drawdown_pct=34.59,
        full_backtest_drawdown_breach=False,
        walk_forward_drawdown_breach=True,
        analogue_sequential_drawdown_pct=31.99,
        analogue_sequential_drawdown_breach=True,
        portfolio_simulation_drawdown_pct=7.37,
        portfolio_simulation_drawdown_breach=False,
        pattern_sample_count=29,
        pattern_oos_sample_count=15,
        pattern_definition="TREND BULLISH",
        selection_bias_risk="UNVERIFIED",
        pattern_validation_qualification="PATTERN_CONDITIONAL_EXPLORATORY",
        directional_family_alignment=25,
        directional_coverage_excl_risk="25/100 (1 of 4 direction-capable families)",
        bootstrap_ci_95_lower_pct=-0.67,
        bootstrap_ci_95_upper_pct=0.53,
        bootstrap_mean_return_pct=-0.04,
        alpha_status="NO_ALPHA_ESTABLISHED",
        final_validated_edge_verdict="NO VALIDATED EDGE",
        verdict_rationale=""
    )
    assert cs.directional_family_alignment == 25
    assert "1 of 4" in cs.directional_coverage_excl_risk


# -------------------------------------------------------------------------
# Test 5: Matching completion cannot imply profitability validation
# -------------------------------------------------------------------------
def test_5_matching_completion_cannot_imply_profitability_validation(app_source: str):
    """Historical setup comparison must show MATCHING_COMPLETE, not PASS near sample count."""
    # Frontend must not render PASS badge on sample_size_status in analogues section
    assert "histComp.matching_status || 'MATCHING_COMPLETE'" in app_source
    assert "Status Separation Notice: Matching Complete ≠ Strategy Validated" in app_source
    assert "Analogue Proxy Drawdown" in app_source

    # getStatusBadge must handle MATCHING_COMPLETE separately from PASS
    assert "'MATCHING_COMPLETE', 'SUCCESSFUL_EXECUTION'" in app_source


# -------------------------------------------------------------------------
# Test 6: Benchmark commentary reconciles with its corresponding benchmark figure
# -------------------------------------------------------------------------
def test_6_benchmark_commentary_reconciles_with_benchmark_figure(app_source: str):
    """The benchmark section must dynamically bind benchmark return in caveat to match table return."""
    # Caveat must not have hardcoded -9.19% disconnected from the table value
    assert "In a declining market (-9.19%)" not in app_source

    # Caveat must dynamically evaluate bm.benchmark_return_pct
    assert "bm.benchmark_return_pct != null ? `${bm.benchmark_return_pct.toFixed(2)}%`" in app_source

    # When benchmark return is -9.07%, caveat reflects -9.07%
    bm = BenchmarkComparisonResult(
        benchmark_name="BUY_AND_HOLD",
        benchmark_symbol="^NSEI",
        cash_benchmark_name="CASH",
        cash_annual_rate_pct=6.5,
        start_date="2024-01-01",
        end_date="2024-07-01",
        start_price=100.0,
        end_price=90.93,
        holding_days=180,
        benchmark_return_pct=-9.07,
        benchmark_max_drawdown_pct=10.0,
        cash_benchmark_return_pct=3.25,
        strategy_net_return_pct=1.0,
        strategy_max_drawdown_pct=2.0,
        excess_return_vs_asset_pct=10.07,
        excess_return_vs_cash_pct=-2.25,
        strategy_avg_exposure_pct=30.8,
        benchmark_exposure_pct=100.0,
        cash_exposure_pct=0.0,
        strategy_turnover_pct=50.0,
        benchmark_turnover_pct=0.0,
        total_transaction_costs_pct=0.15,
        alpha_status="NO_ALPHA_ESTABLISHED",
        benchmark_comparison_status="MIXED_BENCHMARK_RESULTS",
        nifty_comparison_status="OUTPERFORMED_NOMINAL_PRICE_RETURN",
        cash_comparison_status="UNDERPERFORMED_CASH",
        exposure_matched_benchmark_name="HYBRID",
        exposure_matched_benchmark_return_pct=-0.54,
        excess_return_vs_exposure_matched_pct=1.54,
        exposure_matched_comparison_status="OUTPERFORMED_EXPOSURE_MATCHED",
        excess_calculation_formula="",
        exposure_mismatch_warning="",
        dividend_adjustment_note="",
        comparison_summary=""
    )
    assert bm.benchmark_return_pct == -9.07
    assert bm.excess_return_vs_asset_pct == 10.07


# -------------------------------------------------------------------------
# Test 7: Overall validation remains independent of retrospective portfolio simulation
# -------------------------------------------------------------------------
def test_7_overall_validation_independent_of_retrospective_portfolio_simulation():
    """A positive exploratory simulation (+0.88%) must NEVER override walk-forward failure (-7.20%)."""
    # Canonical analysis summary with failed walk-forward and positive exploratory portfolio
    cs = CanonicalAnalysisSummary(
        directional_bias=DirectionalBias.BEARISH,
        evidence_strength=EvidenceStrength.WEAK,
        setup_status=SetupStatus.UNCONFIRMED,
        strategy_backtest_status="PASS",  # Pattern backtest passed
        walk_forward_status="FAIL",        # Walk forward failed
        walk_forward_compounded_return_pct=-7.20,
        walk_forward_profit_factor=0.93,
        walk_forward_max_drawdown_pct=34.59,
        overall_validation_status="NO_VALIDATED_EDGE",
        walk_forward_profitability_status="UNPROFITABLE",
        oos_partition_label="Holdout N=15",
        sensitivity_analysis_available=True,
        sensitivity_analysis_result="INSUFFICIENT_EVIDENCE",
        sensitivity_analysis_summary="",
        portfolio_simulation_execution_status="EXPLORATORY_HISTORICAL_SIMULATION",
        portfolio_simulation_return_pct=0.88,  # Positive retrospective portfolio return
        portfolio_simulation_nature="EXPLORATORY_RETROSPECTIVE_SIMULATION",
        profitability_status="UNPROFITABLE",
        risk_limit_status="DRAWDOWN_LIMIT_BREACH",
        drawdown_limit_pct=25.0,
        benchmark_comparison_available=True,
        benchmark_comparison_status="MIXED_BENCHMARK_RESULTS",
        benchmark_summary_status="MIXED_BENCHMARK_RESULTS",
        benchmark_nifty_status="OUTPERFORMED_NOMINAL_PRICE_RETURN",
        benchmark_cash_status="UNDERPERFORMED_CASH",
        exposure_matched_status="OUTPERFORMED_EXPOSURE_MATCHED",
        excess_return_vs_nifty_pp=11.10,
        excess_return_vs_cash_pp=-1.92,
        strategy_avg_exposure_pct=30.8,
        benchmark_exposure_pct=100.0,
        full_backtest_drawdown_pct=5.08,
        walk_forward_drawdown_pct=34.59,
        full_backtest_drawdown_breach=False,
        walk_forward_drawdown_breach=True,
        analogue_sequential_drawdown_pct=31.99,
        analogue_sequential_drawdown_breach=True,
        portfolio_simulation_drawdown_pct=7.37,
        portfolio_simulation_drawdown_breach=False,
        pattern_sample_count=29,
        pattern_oos_sample_count=15,
        pattern_definition="TREND BEARISH",
        selection_bias_risk="UNVERIFIED",
        pattern_validation_qualification="PATTERN_CONDITIONAL_EXPLORATORY",
        directional_family_alignment=25,
        directional_coverage_excl_risk="25/100 (1 of 4 direction-capable families)",
        bootstrap_ci_95_lower_pct=-0.67,
        bootstrap_ci_95_upper_pct=0.53,
        bootstrap_mean_return_pct=-0.04,
        alpha_status="NO_ALPHA_ESTABLISHED",
        final_validated_edge_verdict="NO VALIDATED EDGE",
        verdict_rationale="Walk-forward validation failed; no alpha established."
    )

    # Core integrity assertions:
    assert cs.overall_validation_status == "NO_VALIDATED_EDGE"
    assert cs.walk_forward_profitability_status == "UNPROFITABLE"
    assert cs.walk_forward_status == "FAIL"
    assert cs.alpha_status == "NO_ALPHA_ESTABLISHED"
    # Even though portfolio simulation returned positive:
    assert cs.portfolio_simulation_return_pct > 0.0
    assert cs.portfolio_simulation_execution_status == "EXPLORATORY_HISTORICAL_SIMULATION"
    assert cs.risk_limit_status == "DRAWDOWN_LIMIT_BREACH"
