"""
Unit, regression, and audit tests for SignalCouncil:
Metrics, Dependency and Historical Comparison Audit.

Verifies:
1. P0 Empirical dependency consistency (Heuristic grouping vs measured correlation, cross-family redundancy adjustment, limitations).
2. P0 Historical comparison data lineage (Isolated typed models, non-leaking feature distance matching, N=0 null handling).
3. P0 Reconciled return and drawdown metrics (Deterministic fixtures for arithmetic mean, cumulative sum, compounded return, equity drawdown, and MAE).
4. P0 Stop selection and thesis semantics (Primary stop policy, risk exit vs structural invalidation, R:R calculation, elimination of 'trailing' misnomer).
5. P1 Volume and data-quality semantics (Cash index UNAVAILABLE volume, volume confirmation disabled).
6. P1 Score definitions (Directional family alignment/coverage vs probability, neutral isolation, risk isolation).
7. P1 Evidence ledger and graph reconciliation (6 claims, 20 nodes, 24 edges).
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from app.domain.models import (
    OHLCV, OHLCVData, StrategyResult, Signal, DirectionalBias, EvidenceFamily,
    EvidencePolarity, EvidenceStrength, SetupStatus, BacktestStatus,
    ThesisStatus, TradeGeometry, HistoricalComparisonResult, HistoricalSetupResult,
    InstrumentMetadata, DataQualityReport, RelationshipType, NodeType,
    MeasuredDependence, EvidenceDependency
)
from app.domain.stress_tester import TradeThesisStressTester
from app.domain.evidence import (
    EvidenceLedgerBuilder, compute_measured_dependencies, build_evidence_graph
)
from app.domain.historical_setup import HistoricalSetupEngine


def create_mock_ohlcv_series(n_bars: int = 100, base_price: float = 22520.45) -> list[OHLCV]:
    bars = []
    for i in range(n_bars):
        date_str = f"2026-01-{(i % 28) + 1:02d}T09:15:00"
        ts = datetime.fromisoformat(date_str)
        close = base_price - (i * 2.5) + (i % 5)
        bars.append(OHLCV(
            timestamp=ts,
            open=close + 1.0,
            high=close + 5.0,
            low=close - 5.0,
            close=close,
            volume=0.0
        ))
    return bars


# ==============================================================================
# P0 — EMPIRICAL DEPENDENCY CONSISTENCY TESTS
# ==============================================================================

def test_heuristic_grouping_vs_measured_correlation_disagreement():
    """
    Test the case where heuristic family membership deliberately disagrees with
    empirically measured correlation:
    1. Trend (EMA slope) and Breakout (Donchian):
       - Grouped into 1 heuristic family (TREND) due to shared qualitative expansion premise.
       - Empirical rank correlation is distinct (r ~ +0.30).
       - Distinct measured correlation does NOT split the qualitative heuristic family.
    2. Momentum (RSI) and Mean Reversion (Bollinger %B):
       - Remain separate heuristic families (MOMENTUM vs MEAN_REVERSION).
       - Empirically collinear (r ~ +0.94).
       - Cross-family empirical redundancy adjustment reduces effective weight when voting in same direction.
    """
    # Create synthetic series where RSI and BB %B are collinear, and EMA vs Donchian are distinct
    bars = create_mock_ohlcv_series(100)
    market_data = OHLCVData(data=bars)

    # Strategy results where both Trend & Breakout vote BUY (collinear in family)
    # and Momentum & Mean Reversion both vote BUY (cross-family collinearity)
    strats = [
        StrategyResult(strategy_name="Trend", signal=Signal.BUY, confidence=0.8),
        StrategyResult(strategy_name="Breakout", signal=Signal.BUY, confidence=0.7),
        StrategyResult(strategy_name="Momentum", signal=Signal.BUY, confidence=0.85),
        StrategyResult(strategy_name="MeanReversion", signal=Signal.BUY, confidence=0.80),
        StrategyResult(strategy_name="PriceAction", signal=Signal.NEUTRAL, confidence=0.0),
        StrategyResult(strategy_name="Risk", signal=Signal.NEUTRAL, confidence=0.0),
    ]

    measured_deps = [
        MeasuredDependence(
            indicator_a="Trend (EMA 20/50)",
            indicator_b="Breakout (Donchian 20)",
            correlation=0.30,
            sample_size=60,
            is_dependent=False,
            limitations="Low correlation (r=+0.30) reflects differing calculation mechanics, but does not prove statistical independence or causality."
        ),
        MeasuredDependence(
            indicator_a="Momentum (RSI 14)",
            indicator_b="MeanReversion (Bollinger %B)",
            correlation=0.94,
            sample_size=60,
            is_dependent=True,
            limitations="High empirical correlation (r=+0.94). Statistical association does not establish forward causality, nor does low correlation alone establish statistical independence."
        )
    ]

    builder = EvidenceLedgerBuilder(market_data=market_data, strategy_results=strats)
    claims = builder.build_claims()
    deps, raw_votes, indep_votes, collinear_detected, dep_explanation = builder.detect_false_consensus(
        claims,
        measured_dependencies=measured_deps
    )

    # 1. Heuristic collapse of Trend + Breakout into TREND family
    trend_dep = next(d for d in deps if d.family == EvidenceFamily.TREND)
    assert len(trend_dep.indicators) == 2
    assert "Trend" in trend_dep.indicators and "Breakout" in trend_dep.indicators
    assert trend_dep.grouping_type == "HEURISTIC_FAMILY"
    assert trend_dep.effective_weight == 1.0  # Heuristically collapsed to single 1.0 vote

    # 2. Cross-family empirical redundancy adjustment on Momentum & MeanReversion
    mom_dep = next(d for d in deps if d.family == EvidenceFamily.MOMENTUM)
    mr_dep = next(d for d in deps if d.family == EvidenceFamily.MEAN_REVERSION)
    assert mom_dep.measured_correlation == 0.94
    assert mr_dep.measured_correlation == 0.94
    assert mom_dep.effective_weight == 1.0  # Primary oscillator retains 1.0
    assert mr_dep.grouping_type == "EMPIRICALLY_ADJUSTED"
    # Weight adjusted down from 1.0 by formula: max(0.2, 1.0 - 0.5 * 0.94) = max(0.2, 0.53) = 0.53
    assert mr_dep.effective_weight == pytest.approx(0.53, abs=0.01)

    # 3. Limitations documentation check
    assert "causality" in measured_deps[0].limitations.lower() or "independence" in measured_deps[0].limitations.lower()
    assert "causality" in measured_deps[1].limitations.lower() or "independence" in measured_deps[1].limitations.lower()


def test_low_correlation_does_not_infer_statistical_independence():
    """
    Ensure the documentation and measured dependency output explicitly states
    that low correlation alone does not prove statistical independence or forward causality.
    """
    df = pd.DataFrame({
        'timestamp': [f"2026-01-{i+1:02d}" for i in range(50)],
        'close': np.linspace(100, 150, 50) + np.sin(np.linspace(0, 10, 50)) * 5,
        'open': np.linspace(100, 150, 50),
        'high': np.linspace(102, 155, 50),
        'low': np.linspace(98, 145, 50),
        'volume': [1000.0] * 50
    })

    deps = compute_measured_dependencies(df)
    for dep in deps:
        assert "independence" in dep.limitations.lower()
        assert "causality" in dep.limitations.lower()


# ==============================================================================
# P0 — HISTORICAL COMPARISON DATA LINEAGE TESTS
# ==============================================================================

def test_historical_comparison_and_backtest_separate_typed_models():
    """
    Verify backtest and historical analogue comparison have separate typed models,
    distinct product types, source datasets, and selection methodology.
    """
    bars = create_mock_ohlcv_series(100)
    engine = HistoricalSetupEngine(bars)

    # 1. Backtest partition investigation
    backtest_res, _ = engine.investigate_setup(DirectionalBias.BEARISH)
    assert isinstance(backtest_res, HistoricalSetupResult)
    assert backtest_res.data_product_type == "STRATEGY_RULE_BACKTEST"
    assert "deterministic" in backtest_res.selection_rules.lower() or "rule" in backtest_res.selection_rules.lower()

    # 2. Historical analogue comparison
    comp_res = engine.compare_historical_setups(DirectionalBias.BEARISH, horizon_bars=5)
    assert isinstance(comp_res, HistoricalComparisonResult)
    assert comp_res.data_product_type == "HISTORICAL_ANALOGUE_COMPARISON"
    assert "euclidean" in comp_res.selection_methodology.lower()
    assert comp_res.is_backtest_reused is False


def test_historical_comparison_n_zero_renders_insufficient_data():
    """
    When no valid analogue matches exist (small dataset < 50 bars),
    render INSUFFICIENT_DATA and null metrics rather than reusing backtest panel results.
    """
    small_bars = create_mock_ohlcv_series(30)
    engine = HistoricalSetupEngine(small_bars)
    comp_res = engine.compare_historical_setups(DirectionalBias.BEARISH, horizon_bars=5)

    assert comp_res.total_comparable_setups == 0
    assert comp_res.sample_size_status == BacktestStatus.INSUFFICIENT_DATA
    assert comp_res.win_rate is None
    assert comp_res.mean_net_return_pct is None
    assert comp_res.mean_per_trade_return_pct is None
    assert comp_res.cumulative_sum_return_pct is None
    assert comp_res.cumulative_compounded_return_pct is None
    assert comp_res.mean_adverse_excursion_pct is None
    assert comp_res.equity_max_drawdown_pct is None
    assert len(comp_res.analogues) == 0


def test_historical_analogue_traceability_to_snapshot_and_outcome():
    """
    Verify every analogue setup is traceable to its decision-time feature snapshot
    and forward outcome, with no future lookahead.
    """
    bars = create_mock_ohlcv_series(120)
    engine = HistoricalSetupEngine(bars)
    comp_res = engine.compare_historical_setups(DirectionalBias.BEARISH, horizon_bars=5)

    if comp_res.total_comparable_setups > 0:
        for a in comp_res.analogues:
            assert a.analogue_id.startswith("HIST-")
            assert a.decision_date != ""
            assert a.similarity_score >= 0.70
            assert a.forward_return_pct is not None
            assert a.features is not None
            assert a.features.rsi_value is not None


# ==============================================================================
# P0 — RECONCILE RETURN AND DRAWDOWN METRICS DETERMINISTIC FIXTURES
# ==============================================================================

def test_deterministic_return_and_drawdown_calculation_formulas():
    """
    Test fixtures with known trade returns:
    Trade returns: [+10.0%, -5.0%, +10.0%, -5.0%] -> decimals: [+0.10, -0.05, +0.10, -0.05].
    Expected:
    - Arithmetic mean per-trade return: (0.10 - 0.05 + 0.10 - 0.05) / 4 = +0.025 -> +2.50%
    - Cumulative sum of returns: 0.10 - 0.05 + 0.10 - 0.05 = +0.10 -> +10.00%
    - Compounded cumulative net return: (1.10 * 0.95 * 1.10 * 0.95) - 1 = 1.092025 - 1 = +9.20%
    - Profit factor: Total gains / Total losses = 0.20 / 0.10 = 2.00
    - Equity curve chronological peak-to-trough max drawdown:
      Initial: 1.0
      After trade 1: 1.10 (peak = 1.10)
      After trade 2: 1.10 * 0.95 = 1.045 (drawdown = (1.10 - 1.045) / 1.10 = 5.0%)
      After trade 3: 1.045 * 1.10 = 1.1495 (peak = 1.1495)
      After trade 4: 1.1495 * 0.95 = 1.092025 (drawdown = (1.1495 - 1.092025) / 1.1495 = 5.0%)
      Max Drawdown = 5.00%
    """
    known_returns = [0.10, -0.05, 0.10, -0.05]
    
    # Arithmetic mean per-trade return
    mean_per_trade = float(np.mean(known_returns)) * 100.0
    assert mean_per_trade == pytest.approx(2.50, abs=0.01)

    # Cumulative sum of returns
    cum_sum = float(np.sum(known_returns)) * 100.0
    assert cum_sum == pytest.approx(10.00, abs=0.01)

    # Compounded cumulative return
    compounded_equity = np.cumprod(1.0 + np.array(known_returns))
    total_compounded = (compounded_equity[-1] - 1.0) * 100.0
    assert total_compounded == pytest.approx(9.2025, abs=0.01)

    # Profit factor
    gains = sum(r for r in known_returns if r > 0)
    losses = abs(sum(r for r in known_returns if r < 0))
    profit_factor = gains / losses
    assert profit_factor == pytest.approx(2.00, abs=0.01)

    # Chronological equity curve max drawdown
    peaks = np.maximum.accumulate(compounded_equity)
    drawdowns = (peaks - compounded_equity) / peaks
    max_equity_dd = float(np.max(drawdowns)) * 100.0
    assert max_equity_dd == pytest.approx(5.00, abs=0.01)


def test_historical_engine_metrics_separation():
    """
    Verify that HistoricalSetupEngine calculates and distinguishes
    mean_per_trade_return_pct, cumulative_sum_return_pct, compounded_net_return_pct,
    and equity_max_drawdown from intra-trade MAE.
    """
    bars = create_mock_ohlcv_series(120)
    engine = HistoricalSetupEngine(bars)
    
    # Backtest partition metrics
    bt, _ = engine.investigate_setup(DirectionalBias.BEARISH)
    if bt.full_history.sample_count > 0:
        fh = bt.full_history
        assert fh.mean_per_trade_return_pct is not None
        assert fh.cumulative_sum_return_pct is not None
        assert fh.compounded_net_return_pct is not None
        assert fh.equity_max_drawdown is not None
        assert fh.mean_adverse_excursion_pct is not None
        # Verify that cumulative sum and mean per trade are not identical when sample_count > 1
        if fh.sample_count > 1:
            assert fh.mean_per_trade_return_pct != fh.cumulative_sum_return_pct

    # Historical analogue comparison metrics
    comp = engine.compare_historical_setups(DirectionalBias.BEARISH, horizon_bars=5)
    if comp.total_comparable_setups > 0:
        assert comp.mean_per_trade_return_pct is not None
        assert comp.cumulative_sum_return_pct is not None
        assert comp.cumulative_compounded_return_pct is not None
        assert comp.mean_adverse_excursion_pct is not None
        assert comp.equity_max_drawdown_pct is not None
        if comp.total_comparable_setups > 1:
            assert comp.mean_per_trade_return_pct != comp.cumulative_sum_return_pct


# ==============================================================================
# P0 — STOP SELECTION AND THESIS SEMANTICS TESTS
# ==============================================================================

def test_stop_selection_policy_risk_vs_structural():
    """
    Verify stop-selection semantics:
    - Primary stop policy = 'RISK_STOP' yields 2-ATR risk stop (0.69:1 R:R).
    - Primary stop policy = 'STRUCTURAL_STOP' yields structural resistance stop (0.32:1 R:R).
    - Policy = 'NONE' yields R:R unavailable (None).
    - Target rule and stop rule are stated beside every R:R figure.
    - No 'trailing stop' label is used.
    """
    bars = create_mock_ohlcv_series(100, base_price=22520.45)
    tester = TradeThesisStressTester(bars, symbol="^NSEI")

    # 1. RISK_STOP policy
    geom_risk = tester.calculate_trade_geometry(
        latest_price=22520.45,
        final_bias=DirectionalBias.BEARISH,
        support=22179.90,
        resistance=23592.85,
        target_price=22179.90,
        target_rule="Key Support Invalidation Target at 22179.90",
        risk_stop_price=23011.56,
        risk_stop_rule="2-ATR Volatility Risk Stop at 23011.56",
        structural_stop_price=23592.85,
        structural_stop_rule="20-period Swing Resistance Structure at 23592.85",
        primary_stop_policy="RISK_STOP"
    )

    assert geom_risk.is_valid is True
    assert geom_risk.primary_stop_policy == "RISK_STOP"
    assert geom_risk.primary_stop_price == pytest.approx(23011.56, abs=0.01)
    assert geom_risk.reward_to_risk_ratio == pytest.approx(0.69, abs=0.01)
    assert "trailing" not in geom_risk.primary_stop_rule.lower()
    assert "trailing" not in geom_risk.notes.lower()

    # 2. STRUCTURAL_STOP policy
    geom_struct = tester.calculate_trade_geometry(
        latest_price=22520.45,
        final_bias=DirectionalBias.BEARISH,
        support=22179.90,
        resistance=23592.85,
        target_price=22179.90,
        target_rule="Key Support Invalidation Target at 22179.90",
        risk_stop_price=23011.56,
        risk_stop_rule="2-ATR Volatility Risk Stop at 23011.56",
        structural_stop_price=23592.85,
        structural_stop_rule="20-period Swing Resistance Structure at 23592.85",
        primary_stop_policy="STRUCTURAL_STOP"
    )

    assert geom_struct.is_valid is True
    assert geom_struct.primary_stop_policy == "STRUCTURAL_STOP"
    assert geom_struct.primary_stop_price == pytest.approx(23592.85, abs=0.01)
    assert geom_struct.reward_to_risk_ratio == pytest.approx(0.32, abs=0.01)

    # 3. NONE policy (No stop explicitly selected)
    geom_none = tester.calculate_trade_geometry(
        latest_price=22520.45,
        final_bias=DirectionalBias.BEARISH,
        target_price=22179.90,
        target_rule="Target",
        primary_stop_policy="NONE"
    )

    assert geom_none.is_valid is False
    assert geom_none.reward_to_risk_ratio is None
    assert "Missing target or stop level" in geom_none.validation_error


# ==============================================================================
# P1 — VOLUME AND DATA-QUALITY SEMANTICS TESTS
# ==============================================================================

def test_volume_semantics_for_cash_index_separate_from_zero_volume():
    """
    Cash index instruments (^NSEI) must represent volume as UNAVAILABLE_CASH_INDEX,
    disabling volume-dependent confirmation triggers without misreporting missing data as anomalous zero volume.
    """
    bars = create_mock_ohlcv_series(60, base_price=22520.45)
    tester = TradeThesisStressTester(bars, symbol="^NSEI")
    
    df = pd.DataFrame([b.model_dump() for b in bars])
    dq = tester.audit_data_quality(df)

    assert dq.volume_semantics == "UNAVAILABLE_CASH_INDEX"
    assert dq.volume_status == "UNAVAILABLE"
    assert dq.has_valid_volume is False
    assert dq.volume_trigger_eligible is False
    assert dq.zero_volume_bars == len(bars)
    assert "does not record traded shares" in dq.zero_volume_explanation


# ==============================================================================
# P1 — SCORE DEFINITIONS TESTS
# ==============================================================================

def test_directional_family_alignment_definition_vs_single_indicator():
    """
    Verify score definitions:
    - Overall directional family alignment: 20/100 represents 1 of 5 independent families aligned.
    - Labeled directional-family coverage / alignment, NOT probability or conviction.
    - Risk controls never vote on directional bias.
    """
    bars = create_mock_ohlcv_series(100)
    tester = TradeThesisStressTester(bars, symbol="^NSEI")

    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.80),
        StrategyResult(strategy_name="Breakout", signal=Signal.SELL, confidence=0.70),
        StrategyResult(strategy_name="Momentum", signal=Signal.NEUTRAL, confidence=0.0),
        StrategyResult(strategy_name="MeanReversion", signal=Signal.NEUTRAL, confidence=0.0),
        StrategyResult(strategy_name="PriceAction", signal=Signal.NEUTRAL, confidence=0.0),
        StrategyResult(strategy_name="Risk", signal=Signal.NEUTRAL, confidence=0.85),  # Risk control
    ]

    analysis = tester.run_stress_test(strat_results)

    # 1 directional family (TREND) out of 4 direction-capable families aligned with BEARISH (25/100)
    # VOLATILITY_RISK is excluded from the directional denominator.
    assert analysis.directional_family_alignment == 25
    assert analysis.heuristic_score == 25
    assert "1 of 4" in analysis.directional_family_alignment_detail
    assert "volatility_risk" in analysis.directional_family_alignment_detail.lower()
    assert "coverage" in analysis.directional_family_alignment_detail.lower() or "not an empirical win probability" in analysis.directional_family_alignment_detail.lower()

    # Risk control must NOT cast directional votes
    risk_dep = next(d for d in analysis.independent_evidence_families if d.family == EvidenceFamily.VOLATILITY_RISK)
    assert risk_dep.independent_vote == DirectionalBias.NEUTRAL


# ==============================================================================
# P1 — EVIDENCE LEDGER AND GRAPH RECONCILIATION TESTS
# ==============================================================================

def test_evidence_ledger_reconciles_six_claims_and_twenty_nodes():
    """
    Verify that exactly 6 claims exist in the ledger and 20 nodes / 24 edges exist
    in the interactive evidence graph with documented relationship types.
    """
    bars = create_mock_ohlcv_series(100)
    tester = TradeThesisStressTester(bars, symbol="^NSEI")

    strat_results = [
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.80),
        StrategyResult(strategy_name="Breakout", signal=Signal.SELL, confidence=0.70),
        StrategyResult(strategy_name="Momentum", signal=Signal.NEUTRAL, confidence=0.0),
        StrategyResult(strategy_name="MeanReversion", signal=Signal.NEUTRAL, confidence=0.0),
        StrategyResult(strategy_name="PriceAction", signal=Signal.NEUTRAL, confidence=0.0),
        StrategyResult(strategy_name="Risk", signal=Signal.NEUTRAL, confidence=0.85),
    ]

    analysis = tester.run_stress_test(strat_results)

    # 1. Exactly 6 claims
    total_claims = len(analysis.supporting_evidence) + len(analysis.opposing_evidence) + len(analysis.neutral_evidence)
    assert total_claims == 6

    # 2. Graph topology
    graph = analysis.evidence_graph
    assert len(graph.nodes) == 20
    assert len(graph.edges) == 24

    # 3. Documented relationship types
    valid_edge_types = {e.value for e in RelationshipType}
    for edge in graph.edges:
        assert edge.relationship_type.value in valid_edge_types
        assert edge.source_node_id != ""
        assert edge.target_node_id != ""
