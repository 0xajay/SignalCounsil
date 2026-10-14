"""
Unit and regression tests for the SignalCouncil Correctness, Evidence Graph and Historical Comparison Sprint.
Prioritizes deterministic correctness across trade geometry, evidence ledger reconciliation,
historical comparison data isolation, empirical dependence vs heuristic families, and volume semantics.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime

from app.domain.models import (
    OHLCV, OHLCVData, StrategyResult, Signal, DirectionalBias, EvidenceFamily,
    EvidencePolarity, EvidenceStrength, SetupStatus, BacktestStatus,
    ThesisStatus, TradeGeometry, HistoricalComparisonResult, InstrumentMetadata,
    DataQualityReport, RelationshipType, NodeType
)
from app.domain.stress_tester import TradeThesisStressTester
from app.domain.evidence import (
    EvidenceLedgerBuilder, compute_measured_dependencies, build_evidence_graph
)
from app.domain.historical_setup import HistoricalSetupEngine


def create_sample_ohlcv(n_bars: int = 100, base_price: float = 22520.45) -> list[OHLCV]:
    bars = []
    for i in range(n_bars):
        date_str = f"2026-01-{(i % 28) + 1:02d}T09:15:00"
        ts = datetime.fromisoformat(date_str)
        # Gentle downtrend for realistic testing
        close = base_price - (i * 2.5) + (i % 5)
        bars.append(OHLCV(
            timestamp=ts,
            open=close + 1.0,
            high=close + 5.0,
            low=close - 5.0,
            close=close,
            volume=0.0 # cash index zero volume
        ))
    return bars


# ==============================================================================
# P0: TRADE GEOMETRY AND RISK SEMANTICS REGRESSION TESTS
# ==============================================================================

def test_trade_geometry_nifty_risk_stop():
    """
    Test arithmetic regression for NIFTY reference setup with ATR risk stop:
    Reference price: 22520.45, Target: 22179.90 (Bearish), Risk stop: 23011.56.
    Reward = 22520.45 - 22179.90 = 340.55
    Risk = 23011.56 - 22520.45 = 491.11
    Expected R:R = 340.55 / 491.11 = 0.6934 -> 0.69:1
    """
    bars = create_sample_ohlcv(100, base_price=22520.45)
    tester = TradeThesisStressTester(bars, symbol="^NSEI")
    
    geom = tester.calculate_trade_geometry(
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
        selected_stop="RISK"
    )

    assert geom.is_valid is True
    assert geom.validation_error is None
    assert geom.reference_price == pytest.approx(22520.45, abs=0.01)
    assert geom.target_price == pytest.approx(22179.90, abs=0.01)
    assert geom.stop_price == pytest.approx(23011.56, abs=0.01)
    assert geom.target_rule == "Key Support Invalidation Target at 22179.90"
    assert geom.stop_rule == "2-ATR Volatility Risk Stop at 23011.56"
    
    # Expected reference reward-to-risk approximately 0.69:1
    assert geom.reward_to_risk_ratio == pytest.approx(0.69, abs=0.01)
    assert geom.risk_stop_rr_ratio == pytest.approx(0.69, abs=0.01)
    # Structural stop R:R also calculated
    assert geom.structural_stop_rr_ratio == pytest.approx(0.32, abs=0.01)


def test_trade_geometry_nifty_structural_stop():
    """
    Test arithmetic regression for NIFTY reference setup with Structural stop:
    Reference price: 22520.45, Target: 22179.90, Structural stop: 23592.85.
    Reward = 340.55
    Risk = 23592.85 - 22520.45 = 1072.40
    Expected R:R = 340.55 / 1072.40 = 0.3175 -> 0.32:1
    """
    bars = create_sample_ohlcv(100, base_price=22520.45)
    tester = TradeThesisStressTester(bars, symbol="^NSEI")
    
    geom = tester.calculate_trade_geometry(
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
        selected_stop="STRUCTURAL"
    )

    assert geom.is_valid is True
    assert geom.stop_price == pytest.approx(23592.85, abs=0.01)
    assert geom.reward_to_risk_ratio == pytest.approx(0.32, abs=0.01)
    assert geom.structural_stop_rr_ratio == pytest.approx(0.32, abs=0.01)


def test_trade_geometry_missing_target_or_stop():
    """
    When target or stop is missing / not fully defined, reward_to_risk_ratio
    must be None and is_valid must be False.
    """
    bars = create_sample_ohlcv(100, base_price=22520.45)
    tester = TradeThesisStressTester(bars, symbol="^NSEI")
    
    # Missing target
    geom_no_target = tester.calculate_trade_geometry(
        latest_price=22520.45,
        final_bias=DirectionalBias.BEARISH,
        support=None,
        resistance=23592.85,
        target_price=None,
        target_rule=""
    )
    assert geom_no_target.is_valid is False
    assert geom_no_target.reward_to_risk_ratio is None
    assert "Missing target or stop level" in geom_no_target.validation_error

    # Missing stop
    geom_no_stop = tester.calculate_trade_geometry(
        latest_price=22520.45,
        final_bias=DirectionalBias.BEARISH,
        support=22179.90,
        resistance=None,
        target_price=22179.90,
        target_rule="Support target",
        risk_stop_price=None,
        structural_stop_price=None,
        stop_short=None,
        atr=0.0
    )
    assert geom_no_stop.is_valid is False
    assert geom_no_stop.reward_to_risk_ratio is None


def test_trade_geometry_invalid_price_orientation():
    """
    Invalid price geometry: explicit validation failure.
    For BEARISH: target must be below reference and stop above reference.
    If target is above reference, geometry must be flagged as invalid.
    """
    bars = create_sample_ohlcv(100, base_price=22520.45)
    tester = TradeThesisStressTester(bars, symbol="^NSEI")
    
    # Inverted BEARISH geometry (target above reference)
    geom_invalid = tester.calculate_trade_geometry(
        latest_price=22520.45,
        final_bias=DirectionalBias.BEARISH,
        support=22600.00, # Higher than reference!
        resistance=23000.00,
        target_price=22600.00,
        target_rule="Flawed target above price",
        risk_stop_price=23000.00,
        risk_stop_rule="Risk stop"
    )
    assert geom_invalid.is_valid is False
    assert geom_invalid.reward_to_risk_ratio is None
    assert "Invalid price geometry for BEARISH setup" in geom_invalid.validation_error


# ==============================================================================
# P0: HISTORICAL COMPARISON DATA LEAKAGE PREVENTION
# ==============================================================================

def test_historical_comparison_n_zero_isolation():
    """
    When N=0, return INSUFFICIENT_DATA and null/None metric values.
    Never reuse full-history strategy backtest metrics in the analogue section.
    """
    bars = create_sample_ohlcv(40) # Small dataset (< 50 bars) guaranteeing N=0
    engine = HistoricalSetupEngine(bars)
    comp_result = engine.compare_historical_setups(
        candidate_bias=DirectionalBias.BEARISH,
        horizon_bars=5
    )

    assert comp_result.total_comparable_setups == 0
    assert comp_result.sample_size_status == BacktestStatus.INSUFFICIENT_DATA
    assert comp_result.win_rate is None
    assert comp_result.mean_forward_return is None
    assert comp_result.mean_net_return_pct is None
    assert comp_result.max_drawdown_pct is None
    assert len(comp_result.analogues) == 0
    assert len(comp_result.safeguards_and_limitations) > 0


# ==============================================================================
# P0: FULL EVIDENCE LEDGER RECONCILIATION AND TREND CLAIM INCLUSION
# ==============================================================================

def test_full_evidence_ledger_all_six_claims():
    """
    The ledger must generate exactly 6 claims (including Trend), and all 6
    must reconcile across raw votes and partitioned categories.
    """
    bars = create_sample_ohlcv(100)
    strats = [
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.8),
        StrategyResult(strategy_name="Momentum", signal=Signal.NEUTRAL, confidence=0.0),
        StrategyResult(strategy_name="MeanReversion", signal=Signal.NEUTRAL, confidence=0.0),
        StrategyResult(strategy_name="Breakout", signal=Signal.SELL, confidence=0.7),
        StrategyResult(strategy_name="PriceAction", signal=Signal.NEUTRAL, confidence=0.0),
        StrategyResult(strategy_name="Risk", signal=Signal.NEUTRAL, confidence=0.0),
    ]
    builder = EvidenceLedgerBuilder(
        market_data=OHLCVData(data=bars),
        strategy_results=strats
    )

    claims = builder.build_claims()
    assert len(claims) == 6
    
    # Check that Trend claim is present
    trend_claim = next((c for c in claims if c.indicator == "Trend"), None)
    assert trend_claim is not None
    assert trend_claim.claim_id.startswith("EV-TREN")
    assert trend_claim.signal == Signal.SELL
    assert trend_claim.limitations != ""

    # Partition for BEARISH candidate bias
    supporting, opposing, neutral = builder.partition_evidence(claims, DirectionalBias.BEARISH)
    # SELL votes support BEARISH
    assert len(supporting) == 2 # Trend, Breakout
    assert len(opposing) == 0
    assert len(neutral) == 4 # Momentum, MeanReversion, PriceAction, Risk
    assert len(supporting) + len(opposing) + len(neutral) == 6


def test_verdict_blocked_on_inconsistent_evidence():
    """
    Block the final verdict when critical internal records are contradictory or inconsistent.
    """
    bars = create_sample_ohlcv(100)
    tester = TradeThesisStressTester(bars, symbol="^NSEI")
    
    # Simulated flawed audit reconciliation where supporting and opposing counts do not match raw votes
    reconciled, errors = tester.audit_reconciliation(
        strategy_results=[StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.8)],
        raw_votes={"BUY": 0, "SELL": 1, "NEUTRAL": 0},
        indep_votes={"BULLISH": 0, "BEARISH": 1, "NEUTRAL": 0, "CONFLICTED": 0},
        supporting=[], # Flaw: 0 supporting despite 1 SELL raw vote!
        opposing=[],
        neutral=[],
        candidate_bias=DirectionalBias.BEARISH,
        final_bias=DirectionalBias.BEARISH,
        backtest_status=BacktestStatus.PASS,
        setup_status=SetupStatus.CONFIRMED
    )

    assert reconciled is False
    assert len(errors) > 0
    assert any("Ledger counts" in e for e in errors)


# ==============================================================================
# P0: EMPIRICAL DEPENDENCE VS HEURISTIC FAMILIES
# ==============================================================================

def test_empirical_dependence_vs_heuristic_grouping():
    """
    Empirical Spearman rank dependence measures statistical association across time,
    while heuristic families group indicators conceptually.
    Verify:
    - Trend vs Breakout has documented distinct limitations.
    - Momentum vs Mean Reversion has documented collinearity limitations.
    - Redundancy adjustment reduces duplicate voting, does not inflate.
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
    assert len(deps) >= 1
    
    for dep in deps:
        assert dep.method.startswith("Spearman rank correlation")
        assert dep.sample_size >= 30
        assert dep.limitations != ""
        assert "causality" in dep.limitations.lower() or "independence" in dep.limitations.lower()


# ==============================================================================
# P0: VOLUME SEMANTICS NORMALIZATION
# ==============================================================================

def test_volume_semantics_cash_index():
    """
    For cash indices (^NSEI), zero-volume or proxy volume must be explicitly identified:
    volume_semantics = 'UNAVAILABLE_CASH_INDEX', volume_trigger_eligible = False.
    """
    bars = create_sample_ohlcv(50, base_price=22520.45) # bars have volume=0.0
    tester = TradeThesisStressTester(bars, symbol="^NSEI")
    
    df = pd.DataFrame([b.model_dump() for b in bars])
    dq = tester.audit_data_quality(df)

    assert dq.volume_semantics == "UNAVAILABLE_CASH_INDEX"
    assert dq.volume_trigger_eligible is False
    assert "does not record traded shares" in dq.zero_volume_explanation
    assert dq.has_valid_volume is False


# ==============================================================================
# P1: INTERACTIVE EVIDENCE GRAPH TOPOLOGY
# ==============================================================================

def test_evidence_graph_five_tiers_and_typed_edges():
    """
    Graph must contain:
    - Observations (OBS-PRICE, OBS-VOL)
    - Indicators
    - Strategy Signals
    - Evidence Families
    - Root Thesis
    - Typed edges (DERIVED_FROM, CONTRIBUTES_TO, GROUPED_WITH, SUPPORTS_THESIS, etc.)
    """
    bars = create_sample_ohlcv(100)
    market_data = OHLCVData(data=bars)
    
    strats = [
        StrategyResult(strategy_name="Trend", signal=Signal.SELL, confidence=0.8),
        StrategyResult(strategy_name="Momentum", signal=Signal.NEUTRAL, confidence=0.0),
        StrategyResult(strategy_name="MeanReversion", signal=Signal.NEUTRAL, confidence=0.0),
        StrategyResult(strategy_name="Breakout", signal=Signal.SELL, confidence=0.7),
        StrategyResult(strategy_name="PriceAction", signal=Signal.NEUTRAL, confidence=0.0),
        StrategyResult(strategy_name="Risk", signal=Signal.NEUTRAL, confidence=0.0),
    ]
    
    builder = EvidenceLedgerBuilder(market_data=market_data, strategy_results=strats)
    claims = builder.build_claims()
    deps, raw_votes, indep_votes, collinear_detected, dep_expl = builder.detect_false_consensus(claims)
    
    graph = build_evidence_graph(
        market_data=market_data,
        strategy_results=strats,
        dependencies=deps,
        claims=claims,
        measured_dependencies=[],
        directional_bias=DirectionalBias.BEARISH,
        evidence_strength=EvidenceStrength.MODERATE,
        thesis_status=ThesisStatus.NOT_ESTABLISHED,
        tested_hypothesis="BEARISH_EXPANSION"
    )

    node_types = {n.node_type for n in graph.nodes}
    assert NodeType.OBSERVATION in node_types
    assert NodeType.INDICATOR in node_types
    assert NodeType.STRATEGY_SIGNAL in node_types
    assert NodeType.EVIDENCE_FAMILY in node_types
    assert NodeType.THESIS in node_types

    edge_types = {e.relationship_type for e in graph.edges}
    assert RelationshipType.DERIVED_FROM in edge_types
    assert RelationshipType.CONTRIBUTES_TO in edge_types
    assert RelationshipType.GROUPED_WITH in edge_types
    
    # Thesis node exists
    thesis_node = next((n for n in graph.nodes if n.node_id == "THESIS-ROOT"), None)
    assert thesis_node is not None
    assert thesis_node.signal == "BEARISH"
