"""
Regression tests for SignalCouncil Trader View Direction and Analysis-Context Consistency.
Covers Cases A through F:
- Case A: Bullish moving-average alignment (EMA 20 > EMA 50, bias BULLISH).
- Case B: Bearish moving-average alignment (EMA 20 < EMA 50, bias BEARISH).
- Case C: Weak directional evidence (1 of 4 direction-capable families).
- Case D: Instrument switch state isolation (prevents stale cross-instrument state).
- Case E: Price-level semantics (proper role labels, signed deltas, distinct risk/structural stops).
- Case F: Contradictory narrative detection and deterministic regeneration.
"""

from pathlib import Path
import pytest
import pandas as pd
import numpy as np

from app.domain.models import (
    OHLCV, Signal, DirectionalBias, EvidenceFamily, EvidenceStrength,
    SetupStatus, StrategyResult, TradeGeometry, CanonicalAnalysisSummary,
    InstrumentMetadata
)
from app.domain.evidence import EvidenceLedgerBuilder
from app.domain.stress_tester import TradeThesisStressTester
from app.agents.council import generate_deterministic_explanation, sanitize_and_validate_review_narrative
from tests.fixtures.market_data import get_ohlcv_models, create_bullish_trending_data

APP_PATH = Path(__file__).resolve().parent.parent.parent / "extension" / "src" / "App.tsx"


@pytest.fixture(scope="module")
def app_source() -> str:
    assert APP_PATH.exists(), f"App.tsx not found at {APP_PATH}"
    return APP_PATH.read_text(encoding="utf-8")


def _make_canonical_summary(
    directional_bias: DirectionalBias = DirectionalBias.BULLISH,
    evidence_strength: EvidenceStrength = EvidenceStrength.WEAK,
    setup_status: SetupStatus = SetupStatus.UNCONFIRMED,
    ema_20: float = 340.50,
    ema_50: float = 330.20,
    ema_relationship: str = "BULLISH_ABOVE",
    directional_family_alignment: int = 25,
    directional_coverage_excl_risk: str = "25/100 (1 of 4 direction-capable families)",
    target_price: float = 345.34,
    risk_stop_price: float = 323.15,
    structural_stop_price: float = 325.81,
    reward_to_risk_ratio: float = 1.05,
) -> CanonicalAnalysisSummary:
    return CanonicalAnalysisSummary(
        directional_bias=directional_bias,
        evidence_strength=evidence_strength,
        setup_status=setup_status,
        strategy_backtest_status="FAIL",
        walk_forward_status="FAIL",
        walk_forward_compounded_return_pct=-7.20,
        walk_forward_profit_factor=0.93,
        walk_forward_max_drawdown_pct=34.59,
        overall_validation_status="NO_VALIDATED_EDGE",
        walk_forward_profitability_status="UNPROFITABLE",
        oos_partition_label="Chronological Holdout (N=15; Selection Independence Unverified)",
        sensitivity_analysis_available=True,
        sensitivity_analysis_result="INSUFFICIENT_EVIDENCE",
        sensitivity_analysis_summary="Non-overlapping blocks 95% CI spans zero across all 5 offsets",
        portfolio_simulation_execution_status="EXPLORATORY_HISTORICAL_SIMULATION",
        portfolio_simulation_return_pct=0.88,
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
        pattern_definition="TREND BULLISH RSI PULLBACK VOL EXPANSION",
        selection_bias_risk="UNVERIFIED",
        pattern_validation_qualification="PATTERN_CONDITIONAL_EXPLORATORY",
        directional_family_alignment=directional_family_alignment,
        directional_coverage_excl_risk=directional_coverage_excl_risk,
        bootstrap_ci_95_lower_pct=-0.67,
        bootstrap_ci_95_upper_pct=0.53,
        bootstrap_mean_return_pct=-0.04,
        alpha_status="NO_ALPHA_ESTABLISHED",
        final_validated_edge_verdict="NO VALIDATED EDGE",
        verdict_rationale="Evaluated without demonstrated alpha edge.",
        ema_20=ema_20,
        ema_50=ema_50,
        ema_relationship=ema_relationship,
        target_price=target_price,
        risk_stop_price=risk_stop_price,
        structural_stop_price=structural_stop_price,
        reward_to_risk_ratio=reward_to_risk_ratio,
    )


# -------------------------------------------------------------------------
# Case A: Bullish moving-average alignment
# -------------------------------------------------------------------------
def test_case_a_bullish_moving_average_alignment():
    """EMA 20 > EMA 50, Canonical bias is BULLISH.
    Narrative must never describe EMA 20 as below EMA 50 or setup as bearish."""
    cs = _make_canonical_summary(
        directional_bias=DirectionalBias.BULLISH,
        evidence_strength=EvidenceStrength.WEAK,
        ema_20=340.50,
        ema_50=330.20,
        ema_relationship="BULLISH_ABOVE",
        directional_family_alignment=25,
    )
    explanation = generate_deterministic_explanation(cs)

    assert "above the 50-day EMA" in explanation
    assert "bullish trend conditions" in explanation
    assert "below the 50-day EMA" not in explanation
    assert "bearish trend conditions" not in explanation
    assert "supports the bearish view" not in explanation
    assert "downward trend regime" not in explanation


def test_case_a_stress_tester_produces_bullish_canonical_ema():
    """Ensure stress tester populates canonical ema_20, ema_50, and ema_relationship on bullish data."""
    data = create_bullish_trending_data(num_bars=100)
    meta = InstrumentMetadata(
        symbol="AAPL",
        canonical_symbol="AAPL",
        instrument_type="EQUITY",
        exchange="NASDAQ",
        currency="USD"
    )
    tester = TradeThesisStressTester(data, symbol="AAPL", timeframe="1d", instrument=meta)
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.BUY, confidence=0.8, metadata={"ema_20": 340.0, "ema_50": 330.0}),
        StrategyResult(strategy_name="Breakout", signal=Signal.NEUTRAL, confidence=0.5, metadata={}),
    ]
    report = tester.run_stress_test(strat_results)

    cs = report.canonical_summary
    assert cs is not None
    assert cs.ema_20 is not None
    assert cs.ema_50 is not None
    assert cs.ema_20 > cs.ema_50
    assert cs.ema_relationship == "BULLISH_ABOVE"
    assert cs.directional_bias == DirectionalBias.BULLISH
    assert "below the 50-day EMA" not in report.explanation


# -------------------------------------------------------------------------
# Case B: Bearish moving-average alignment
# -------------------------------------------------------------------------
def test_case_b_bearish_moving_average_alignment():
    """EMA 20 < EMA 50, Canonical bias is BEARISH.
    Narrative accurately describes the EMA relationship as bearish."""
    cs = _make_canonical_summary(
        directional_bias=DirectionalBias.BEARISH,
        evidence_strength=EvidenceStrength.MODERATE,
        ema_20=315.00,
        ema_50=330.00,
        ema_relationship="BEARISH_BELOW",
        directional_family_alignment=50,
        directional_coverage_excl_risk="50/100 (2 of 4 direction-capable families)",
    )
    explanation = generate_deterministic_explanation(cs)

    assert "below the 50-day EMA" in explanation
    assert "bearish trend conditions" in explanation
    assert "above the 50-day EMA" not in explanation
    assert "bullish trend conditions" not in explanation
    assert "supports the bullish view" not in explanation


# -------------------------------------------------------------------------
# Case C: Weak directional evidence
# -------------------------------------------------------------------------
def test_case_c_weak_directional_evidence():
    """One of four direction-capable families votes in the prevailing direction.
    Alignment remains consistent with existing scoring (e.g. 25/100).
    Narrative correctly distinguishes directional bias from evidence strength."""
    cs = _make_canonical_summary(
        directional_bias=DirectionalBias.BULLISH,
        evidence_strength=EvidenceStrength.WEAK,
        ema_20=340.00,
        ema_50=330.00,
        ema_relationship="BULLISH_ABOVE",
        directional_family_alignment=25,
        directional_coverage_excl_risk="25/100 (1 of 4 direction-capable families)",
    )
    explanation = generate_deterministic_explanation(cs)

    # Narrative explicitly mentions that only 1 direction-capable family supports the bullish view
    assert "only 1 direction-capable evidence family" in explanation
    assert "supports the bullish view" in explanation
    assert "confirmation remains weak" in explanation
    # And does NOT claim that the view is bearish
    assert "bearish view" not in explanation


# -------------------------------------------------------------------------
# Case D: Instrument switch state isolation
# -------------------------------------------------------------------------
def test_case_d_instrument_switch_frontend_isolation(app_source: str):
    """Verify frontend discards in-flight requests, clears previous report, and isolates cache keys."""
    # Verifies activeRequestIdRef exists to prevent race conditions across symbol switches
    assert "activeRequestIdRef" in app_source
    # Verifies setReport(null) is called when switching/initiating new analysis
    assert "setReport(null)" in app_source
    # Verifies response is discarded if it doesn't match active request id
    assert "reqId !== activeRequestIdRef.current" in app_source
    # Verifies response symbol is validated against requested symbol
    assert "data.symbol.toUpperCase() !== targetSymbol" in app_source


def test_case_d_backend_context_snapshot_isolation():
    """Verify backend stress tests produce isolated canonical snapshots without cross-contamination."""
    data_aapl = create_bullish_trending_data(num_bars=80)
    meta_aapl = InstrumentMetadata(symbol="AAPL", canonical_symbol="AAPL", instrument_type="EQUITY", exchange="NASDAQ", currency="USD")
    tester_aapl = TradeThesisStressTester(data_aapl, symbol="AAPL", timeframe="1d", instrument=meta_aapl)
    res_aapl = tester_aapl.run_stress_test([
        StrategyResult(strategy_name="Trend", signal=Signal.BUY, confidence=0.8, metadata={"ema_20": 340.0, "ema_50": 330.0})
    ])

    data_bear = get_ohlcv_models()
    meta_bear = InstrumentMetadata(symbol="MSFT", canonical_symbol="MSFT", instrument_type="EQUITY", exchange="NASDAQ", currency="USD")
    tester_bear = TradeThesisStressTester(data_bear, symbol="MSFT", timeframe="1d", instrument=meta_bear)
    res_bear = tester_bear.run_stress_test([
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.8, metadata={"ema_20": 300.0, "ema_50": 320.0})
    ])

    # Contexts must be completely distinct
    assert res_aapl.symbol == "AAPL"
    assert res_bear.symbol == "MSFT"
    assert res_aapl.canonical_summary.directional_bias == DirectionalBias.BULLISH
    assert res_bear.canonical_summary.directional_bias == DirectionalBias.BEARISH
    assert res_aapl.canonical_summary.ema_relationship == "BULLISH_ABOVE"
    assert res_bear.canonical_summary.ema_relationship == "BEARISH_BELOW"


# -------------------------------------------------------------------------
# Case E: Price-level semantics
# -------------------------------------------------------------------------
def test_case_e_price_level_semantics(app_source: str):
    """Verify target, stop, structural invalidation, and reward/risk semantics in Trader View."""
    # Upside target must be labelled 'Potential Upside Target', not ambiguous 'Support / Target'
    assert "isUpsideTarget ? 'Potential Upside Target' : 'Potential Downside Target'" in app_source
    # Pts delta must use signed direction (+ for upside target, - for downside)
    assert "targetDistPts != null ? `${isUpsideTarget ? '+' : '-'}${targetDistPts.toFixed(2)} pts` : '--'" in app_source
    # Clear distinction between Risk Stop and Structural Invalidation
    assert "Important Level Distinction:" in app_source
    assert "Risk Stop (" in app_source
    assert "Structural Invalidation (" in app_source
    # Reward/risk ratio derived canonically
    assert "rrRatio" in app_source
    assert "geom?.reward_to_risk_ratio" in app_source


# -------------------------------------------------------------------------
# Case F: Contradictory narrative rejection and regeneration
# -------------------------------------------------------------------------
def test_case_f_contradictory_narrative_sanitization():
    """Inject a contradictory explanation and confirm validator rejects and regenerates it."""
    # Stale contradictory explanation from a previous bearish run injected into a bullish setup
    stale_bearish_narrative = (
        "The 20-day EMA is below the 50-day EMA, indicating bearish trend conditions. "
        "However, only 1 direction-capable evidence family currently supports the bearish view..."
    )

    cs = _make_canonical_summary(
        directional_bias=DirectionalBias.BULLISH,
        evidence_strength=EvidenceStrength.WEAK,
        ema_20=340.0,
        ema_50=330.0,
        ema_relationship="BULLISH_ABOVE",
        directional_family_alignment=25,
    )

    clean_narrative = sanitize_and_validate_review_narrative(
        text=stale_bearish_narrative,
        canonical_summary=cs,
    )

    # Contradiction must be rejected and regenerated
    assert "below the 50-day EMA" not in clean_narrative
    assert "bearish trend conditions" not in clean_narrative
    assert "supports the bearish view" not in clean_narrative
    assert "above the 50-day EMA" in clean_narrative
    assert "supports the bullish view" in clean_narrative


def test_case_f_frontend_contradiction_validator(app_source: str):
    """Verify frontend has consistency validator that rejects contradictory report.explanation."""
    assert "const isContradictory =" in app_source
    assert "textLower.includes('below the 50-day ema')" in app_source
    assert "textLower.includes('supports the bearish view')" in app_source
    assert "plainEnglishExplanation = deterministicExplanation" in app_source
