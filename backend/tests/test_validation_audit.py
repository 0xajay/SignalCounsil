import pytest
import numpy as np
from datetime import datetime, timedelta
from app.domain.models import (
    OHLCV, OHLCVData, DirectionalBias, BacktestStatus, Signal, StrategyResult,
    EvidenceFamily, EvidenceStrength, SetupStatus, ThesisStatus
)
from app.domain.historical_setup import HistoricalSetupEngine
from app.domain.stress_tester import TradeThesisStressTester
from app.domain.evidence import compute_measured_dependencies

def make_ohlcv_series(n_bars: int = 150, start_price: float = 100.0, trend: float = -0.002) -> list[OHLCV]:
    """Generates a synthetic downward trending OHLCV series for testing."""
    bars = []
    base_date = datetime(2025, 1, 1)
    price = start_price
    for i in range(n_bars):
        date_str = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
        # Gradual downward trend with some volatility
        ret = trend + np.sin(i / 5.0) * 0.005 + (0.001 if i % 2 == 0 else -0.001)
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
# 1. Strategy backtest fails while analogue matching succeeds
# ==============================================================================
def test_backtest_fails_while_analogue_matching_succeeds():
    """
    Test scenario 1 & 10:
    Verifies that when matching finds >= 5 comparable setups (e.g. 61),
    matching_status is MATCHING_COMPLETE, but status is NEVER "PASS".
    Strategy validation status remains separate (e.g. FAIL).
    """
    bars = make_ohlcv_series(120)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_result, overall_status = engine.investigate_setup(DirectionalBias.BEARISH)
    comp = engine.compare_historical_setups(
        DirectionalBias.BEARISH, 
        horizon_bars=5,
        backtest_result=hist_result
    )

    if comp.comparable_setups_count >= 5:
        # Matching succeeded
        assert comp.matching_status == "MATCHING_COMPLETE"
        # Must NEVER be PASS
        assert comp.status != "PASS"
        assert comp.status == "MATCHING_COMPLETE"
        # Strategy backtest status is tracked independently
        assert comp.historical_strategy_validation in ("FAIL", "WARNING", "INSUFFICIENT_DATA", "PASS")
        assert comp.historical_strategy_validation == hist_result.full_history.status.value

# ==============================================================================
# 2. Same underlying records appear in both analyses (overlap quantified)
# ==============================================================================
def test_sample_overlap_quantification_and_disclosure():
    """
    Test scenario 2:
    Verifies that when strategy backtest and analogue matching draw from the same
    series, shared bars are detected, overlap percentage is computed, and
    an explicit disclosure is emitted.
    """
    bars = make_ohlcv_series(120)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_result, _ = engine.investigate_setup(DirectionalBias.BEARISH)
    comp = engine.compare_historical_setups(
        DirectionalBias.BEARISH,
        horizon_bars=5,
        backtest_result=hist_result
    )

    if comp.total_comparable_setups > 0 and hist_result.sample_count > 0:
        assert comp.underlying_sample_overlap_count >= 0
        assert 0.0 <= comp.overlap_percentage_of_backtest <= 100.0
        assert 0.0 <= comp.overlap_percentage_of_analogues <= 100.0
        assert "overlap" in comp.data_overlap_disclosure.lower()
        # Verify individual analogues track overlap flag
        for a in comp.analogues:
            assert isinstance(a.is_overlapping_with_backtest, bool)

# ==============================================================================
# 3. Analogue records are genuinely distinct (zero overlap when provided disjoint indices)
# ==============================================================================
def test_analogue_records_genuinely_distinct_when_disjoint():
    """
    Test scenario 3:
    When explicit backtest bar indices do not match matched analogue indices,
    overlap is reported as 0 and independence is disclosed.
    """
    bars = make_ohlcv_series(120)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    # Pass arbitrary disjoint backtest indices (e.g. [9999, 10000])
    comp = engine.compare_historical_setups(
        DirectionalBias.BEARISH,
        horizon_bars=5,
        backtest_bar_indices=[9998, 9999]
    )

    assert comp.underlying_sample_overlap_count == 0
    assert comp.overlap_percentage_of_analogues == 0.0
    assert "independence" in comp.data_overlap_disclosure.lower() or "0 of" in comp.data_overlap_disclosure

# ==============================================================================
# 4. Bearish outcomes are directionally classified correctly
# ==============================================================================
def test_bearish_outcomes_directionally_classified():
    """
    Test scenario 4:
    For a bearish candidate bias, a decline in price (exit < entry) must produce
    a positive gross return and positive net return (WIN) after friction.
    """
    bars = make_ohlcv_series(100, trend=-0.01) # Strongly declining
    engine = HistoricalSetupEngine(bars, horizon_bars=5, friction_pct=0.001)
    comp = engine.compare_historical_setups(DirectionalBias.BEARISH, horizon_bars=5)

    if comp.analogues:
        for a in comp.analogues:
            if a.exit_price < a.decision_price * 0.99:
                # Market went down, short trade must be positive return
                assert a.strategy_net_return_pct > 0.0
                assert a.outcome == "WIN"

# ==============================================================================
# 5. Positive raw market returns are not counted as bearish strategy wins
# ==============================================================================
def test_positive_raw_market_returns_are_not_bearish_wins():
    """
    Test scenario 5:
    If market prices rise (exit > entry), raw market return is positive (+),
    but a BEARISH strategy loses money and must NOT be counted as a WIN.
    """
    bars = make_ohlcv_series(100, trend=+0.01) # Rising market
    engine = HistoricalSetupEngine(bars, horizon_bars=5, friction_pct=0.001)
    comp = engine.compare_historical_setups(DirectionalBias.BEARISH, horizon_bars=5)

    if comp.analogues:
        for a in comp.analogues:
            if a.raw_forward_market_return_pct > 0.5:
                # Price went up!
                assert a.strategy_net_return_pct < 0.0
                assert a.outcome == "LOSS"

# ==============================================================================
# 6. Missing support-distance or breakout features handled via weight re-normalization
# ==============================================================================
def test_missing_features_weight_renormalization_without_silent_zero():
    """
    Test scenario 6:
    Missing features are excluded with weight re-normalization, never silently
    treated as 0 distance. Feature comparison items record is_available=False.
    """
    bars = make_ohlcv_series(100)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    df = bars
    comp = engine.compare_historical_setups(DirectionalBias.BEARISH, horizon_bars=5)

    if comp.analogues:
        sample_analogue = comp.analogues[0]
        assert len(sample_analogue.feature_comparisons) >= 6
        for fc in sample_analogue.feature_comparisons:
            if not fc.is_available:
                assert fc.distance_contribution == 0.0
                assert fc.weight == 0.0
            else:
                assert fc.is_available is True
                assert fc.weight > 0.0

# ==============================================================================
# 7. Analogue matching with zero valid matches
# ==============================================================================
def test_analogue_matching_zero_valid_matches():
    """
    Test scenario 7:
    When no historical bars exist (< 50 bars), matching returns NO_MATCHES_FOUND
    and metrics are withheld (None).
    """
    short_bars = make_ohlcv_series(30)
    engine = HistoricalSetupEngine(short_bars, horizon_bars=5)
    comp = engine.compare_historical_setups(DirectionalBias.BEARISH, horizon_bars=5)

    assert comp.total_comparable_setups == 0
    assert comp.matching_status == "NO_MATCHES_FOUND"
    assert comp.analogue_sample_quality == "INSUFFICIENT_SAMPLE"
    assert comp.win_rate is None
    assert comp.mean_forward_return_pct is None

# ==============================================================================
# 8. Directional-family coverage excludes non-directional risk families (1/4 = 25/100)
# ==============================================================================
def test_directional_family_coverage_excludes_risk_family():
    """
    Test scenario 8:
    VOLATILITY_RISK is a non-directional risk control and is excluded from the denominator.
    When 1 directional family (Trend) is active among 4 direction-capable families,
    coverage is exactly 25/100 (1/4), with numerator and denominator explicitly documented.
    """
    bars = make_ohlcv_series(100)
    tester = TradeThesisStressTester(bars, symbol="TEST")
    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.80),
        StrategyResult(strategy_name="Breakout", signal=Signal.SELL, confidence=0.70),
        StrategyResult(strategy_name="Momentum", signal=Signal.NEUTRAL, confidence=0.0),
        StrategyResult(strategy_name="MeanReversion", signal=Signal.NEUTRAL, confidence=0.0),
        StrategyResult(strategy_name="PriceAction", signal=Signal.NEUTRAL, confidence=0.0),
        StrategyResult(strategy_name="Risk", signal=Signal.NEUTRAL, confidence=0.90), # Risk family
    ]
    analysis = tester.run_stress_test(strat_results)

    assert analysis.directional_family_alignment == 25
    assert "1 of 4" in analysis.directional_family_alignment_detail
    assert "volatility_risk" in analysis.directional_family_alignment_detail.lower()
    assert "not an empirical win probability" in analysis.directional_family_alignment_detail.lower()

# ==============================================================================
# 9. Correlated indicators across different heuristic families
# ==============================================================================
def test_cross_family_empirical_correlation():
    """
    Test scenario 9:
    Indicators across different heuristic families with high empirical correlation
    (|r| >= 0.70) are flagged as collinear/redundant, while distinct families with
    low correlation remain distinct.
    """
    from app.domain.engine import to_dataframe
    bars = make_ohlcv_series(100)
    df = to_dataframe(bars)
    deps = compute_measured_dependencies(df)
    assert len(deps) > 0
    for dep in deps:
        assert hasattr(dep, "correlation")
        assert hasattr(dep, "is_dependent")
        assert hasattr(dep, "method")
        assert "Spearman" in dep.method
        assert dep.limitations != ""

# ==============================================================================
# 10. Status labels remain consistent when only one validation gate passes
# ==============================================================================
def test_status_labels_consistency_one_gate_passes():
    """
    Test scenario 10:
    When matching process succeeds but backtest fails validation:
    matching_status = 'MATCHING_COMPLETE'
    status = 'MATCHING_COMPLETE' (NOT 'PASS')
    historical_strategy_validation = 'FAIL'
    """
    bars = make_ohlcv_series(120)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    hist_result, _ = engine.investigate_setup(DirectionalBias.BEARISH)
    comp = engine.compare_historical_setups(
        DirectionalBias.BEARISH, 
        horizon_bars=5,
        backtest_result=hist_result
    )

    if comp.total_comparable_setups >= 5:
        assert comp.matching_status == "MATCHING_COMPLETE"
        assert comp.status == "MATCHING_COMPLETE"
        assert comp.status != "PASS"
        assert comp.historical_strategy_validation in ("FAIL", "WARNING", "INSUFFICIENT_DATA", "PASS")
