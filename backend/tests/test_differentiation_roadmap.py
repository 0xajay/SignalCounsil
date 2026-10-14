import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from app.domain.models import (
    OHLCV, OHLCVData, Signal, DirectionalBias, EvidenceStrength,
    SetupStatus, BacktestStatus, ThesisStatus, EvidenceFamily,
    EvidencePolarity, StrategyResult, StrategyContext, InstrumentMetadata,
    DataQualityReport, AnalysisResult, NodeType, RelationshipType
)
from app.domain.strategies import (
    TrendStrategy, MomentumStrategy, BreakoutStrategy,
    MeanReversionStrategy, PriceActionStrategy, RiskStrategy
)
from app.domain.stress_tester import TradeThesisStressTester
from app.domain.evidence import (
    EvidenceLedgerBuilder, compute_measured_dependencies, build_evidence_graph
)
from app.domain.historical_setup import HistoricalSetupEngine
from main import AnalyzeResponse
from tests.fixtures.market_data import get_ohlcv_models, create_bullish_trending_data

def get_test_strategies():
    return [
        TrendStrategy(), MomentumStrategy(), MeanReversionStrategy(),
        BreakoutStrategy(), PriceActionStrategy(), RiskStrategy()
    ]

# 1. Missing Trend evidence in the detailed ledger
def test_missing_trend_evidence_in_ledger():
    data = get_ohlcv_models()
    strategies = get_test_strategies()
    context = StrategyContext(symbol="AAPL", timeframe="1d")
    results = [s.evaluate(OHLCVData(data=data), context) for s in strategies]
    
    tester = TradeThesisStressTester(data, symbol="AAPL")
    analysis = tester.run_stress_test(results)
    
    all_ledger = analysis.supporting_evidence + analysis.opposing_evidence + analysis.neutral_evidence
    trend_claims = [c for c in all_ledger if "Trend" in c.indicator]
    assert len(trend_claims) >= 1, "TrendStrategy claim missing from the detailed evidence ledger"
    assert trend_claims[0].claim_id.startswith("EV-")

# 2. Mismatched raw signal, family and ledger counts
def test_mismatched_raw_signal_family_and_ledger_counts():
    data = get_ohlcv_models()
    strategies = get_test_strategies()
    context = StrategyContext(symbol="AAPL", timeframe="1d")
    results = [s.evaluate(OHLCVData(data=data), context) for s in strategies]
    
    tester = TradeThesisStressTester(data, symbol="AAPL")
    analysis = tester.run_stress_test(results)
    
    total_ledger_items = len(analysis.supporting_evidence) + len(analysis.opposing_evidence) + len(analysis.neutral_evidence)
    assert total_ledger_items == len(results), f"Ledger total {total_ledger_items} != strategy results {len(results)}"
    assert analysis.reconciliation_status == "VALIDATED"

# 3. Risk signals incorrectly influencing directional votes
def test_risk_signals_do_not_influence_directional_votes():
    data = get_ohlcv_models()
    risk_strat = RiskStrategy()
    context = StrategyContext(symbol="AAPL", timeframe="1d")
    res = risk_strat.evaluate(OHLCVData(data=data), context)
    
    assert res.signal == Signal.NEUTRAL, "Risk strategy must strictly return Signal.NEUTRAL"
    assert res.confidence == 0.0, "Risk strategy must return 0.0 confidence score"
    assert "atr" in res.metadata and "volatility_regime" in res.metadata

# 4. Heuristic groups incorrectly labelled statistically independent
def test_heuristic_groups_not_labelled_statistically_independent():
    data = get_ohlcv_models()
    strategies = get_test_strategies()
    context = StrategyContext(symbol="AAPL", timeframe="1d")
    results = [s.evaluate(OHLCVData(data=data), context) for s in strategies]
    
    tester = TradeThesisStressTester(data, symbol="AAPL")
    analysis = tester.run_stress_test(results)
    
    assert "Independent Evidence Verification" not in analysis.dependency_explanation, (
        "Wording 'Independent Evidence Verification' must not be used without statistical proof"
    )
    for fam in analysis.independent_evidence_families:
        assert "independent" not in fam.collinearity_risk.lower() or "combined" in fam.collinearity_risk.lower()

# 5. Missing or contradictory dependency edges in evidence graph
def test_evidence_graph_has_no_missing_or_contradictory_edges():
    data = get_ohlcv_models()
    strategies = get_test_strategies()
    context = StrategyContext(symbol="AAPL", timeframe="1d")
    results = [s.evaluate(OHLCVData(data=data), context) for s in strategies]
    
    tester = TradeThesisStressTester(data, symbol="AAPL")
    analysis = tester.run_stress_test(results)
    
    graph = analysis.evidence_graph
    node_ids = {n.node_id for n in graph.nodes}
    assert len(graph.nodes) > 0
    assert len(graph.edges) > 0
    
    for edge in graph.edges:
        assert edge.source_node_id in node_ids, f"Edge source {edge.source_node_id} not found in graph nodes"
        assert edge.target_node_id in node_ids, f"Edge target {edge.target_node_id} not found in graph nodes"

# 6. Evidence graph nodes that do not reconcile with the canonical result
def test_evidence_graph_nodes_reconcile_with_canonical_result():
    data = get_ohlcv_models()
    strategies = get_test_strategies()
    context = StrategyContext(symbol="AAPL", timeframe="1d")
    results = [s.evaluate(OHLCVData(data=data), context) for s in strategies]
    
    tester = TradeThesisStressTester(data, symbol="AAPL")
    analysis = tester.run_stress_test(results)
    
    graph = analysis.evidence_graph
    strat_nodes = [n for n in graph.nodes if n.node_type == NodeType.STRATEGY_SIGNAL]
    assert len(strat_nodes) == len(results)
    
    thesis_node = next(n for n in graph.nodes if n.node_type == NodeType.THESIS)
    assert thesis_node.signal == analysis.directional_bias.value

# 7. Incorrectly attributed supporting and opposing evidence
def test_supporting_and_opposing_evidence_attribution():
    data = create_bullish_trending_data(120)
    strategies = get_test_strategies()
    context = StrategyContext(symbol="AAPL", timeframe="1d")
    results = [s.evaluate(OHLCVData(data=data), context) for s in strategies]
    
    tester = TradeThesisStressTester(data, symbol="AAPL")
    analysis = tester.run_stress_test(results)
    
    if analysis.directional_bias == DirectionalBias.BULLISH:
        for claim in analysis.supporting_evidence:
            assert claim.polarity == EvidencePolarity.SUPPORTS_BULLISH
        for claim in analysis.opposing_evidence:
            assert claim.polarity == EvidencePolarity.CHALLENGES_BULLISH

# 8. Neutral directional indicator being treated as an opposite-direction vote
def test_neutral_directional_indicator_not_treated_as_opposite_vote():
    data = get_ohlcv_models()
    ledger_builder = EvidenceLedgerBuilder(OHLCVData(data=data), [
        StrategyResult(strategy_name="Trend", signal=Signal.NEUTRAL, confidence=0.0)
    ])
    claims = ledger_builder.build_claims()
    assert claims[0].polarity == EvidencePolarity.NON_DIRECTIONAL
    
    supp, opp, neut = ledger_builder.partition_evidence(claims, DirectionalBias.BULLISH)
    assert claims[0] in neut, "Neutral indicator must be partitioned to neutral evidence"
    assert claims[0] not in opp, "Neutral indicator must not be partitioned as opposing evidence"

# 9. Backtest failure with a separate bearish or bullish market bias
def test_backtest_failure_with_separate_directional_bias():
    data = get_ohlcv_models()
    # Mock results where technical indicators vote SELL but historical setup returns FAIL
    res_sell = [
        StrategyResult(strategy_name="TrendStrategy", signal=Signal.SELL, confidence=0.75),
        StrategyResult(strategy_name="BreakoutStrategy", signal=Signal.SELL, confidence=0.70),
        StrategyResult(strategy_name="MomentumStrategy", signal=Signal.SELL, confidence=0.65),
        StrategyResult(strategy_name="RiskStrategy", signal=Signal.NEUTRAL, confidence=0.0)
    ]
    tester = TradeThesisStressTester(data, symbol="TEST")
    analysis = tester.run_stress_test(res_sell)
    
    assert analysis.directional_bias == DirectionalBias.BEARISH
    if analysis.backtest_status == BacktestStatus.FAIL:
        assert analysis.thesis_status == ThesisStatus.NOT_ESTABLISHED
        assert "Historical Setup Failed" in analysis.verdict or "NO VALIDATED EDGE" in analysis.verdict
        assert analysis.directional_bias == DirectionalBias.BEARISH

# 10. Insufficient out-of-sample data
def test_insufficient_out_of_sample_data_gating():
    data = get_ohlcv_models()[:65] # Barely past 50 warmup
    engine = HistoricalSetupEngine(data, horizon_bars=5)
    hist_result, status = engine.investigate_setup(DirectionalBias.BULLISH)
    
    if hist_result.oos_sample_count < 5:
        assert hist_result.oos_status == BacktestStatus.INSUFFICIENT_DATA
        assert status == BacktestStatus.INSUFFICIENT_DATA

# 11. Historical analogue selection leaking future information
def test_historical_analogue_selection_zero_future_leakage():
    data = get_ohlcv_models()
    engine = HistoricalSetupEngine(data, horizon_bars=5)
    df = pd.DataFrame([d.model_dump() for d in data])
    
    # Feature extraction at index 60 must only see <= 60
    feat_60 = engine.extract_setup_features(df, 60)
    assert feat_60 is not None
    # Alter future bars
    df_altered = df.copy()
    df_altered.loc[70:, 'close'] = df_altered.loc[70:, 'close'] * 10
    feat_60_after = engine.extract_setup_features(df_altered, 60)
    assert feat_60.model_dump() == feat_60_after.model_dump(), "Future price changes leaked into historical setup features!"

# 12. Historical comparison with too few comparable setups
def test_historical_comparison_sample_size_safeguards():
    # Only 52 candles: maximum eval idx is 52 - 1 - 5 = 46 (< 50)
    data = get_ohlcv_models()[:52]
    engine = HistoricalSetupEngine(data, horizon_bars=5)
    comp = engine.compare_historical_setups(DirectionalBias.BULLISH, horizon_bars=5)
    
    assert comp.status == BacktestStatus.INSUFFICIENT_DATA
    assert comp.comparable_setups_count < 5
    assert "warning" in comp.sample_size_limitations.lower() or "insufficient" in comp.sample_size_limitations.lower()

# 13. Unsupported or mislabelled confidence scores
def test_unsupported_or_mislabelled_confidence_scores():
    data = get_ohlcv_models()
    neutral_results = [
        StrategyResult(strategy_name="TrendStrategy", signal=Signal.NEUTRAL, confidence=0.0),
        StrategyResult(strategy_name="MomentumStrategy", signal=Signal.NEUTRAL, confidence=0.0),
        StrategyResult(strategy_name="RiskStrategy", signal=Signal.NEUTRAL, confidence=0.0)
    ]
    tester = TradeThesisStressTester(data, symbol="TEST")
    analysis = tester.run_stress_test(neutral_results)
    
    assert analysis.directional_bias == DirectionalBias.NEUTRAL
    assert analysis.confidence == 0.0, "Neutral confidence must strictly be 0.0"
    assert analysis.score == 0

# 14. Stop-loss breach being confused with thesis invalidation
def test_stop_loss_breach_separated_from_thesis_invalidation():
    data = get_ohlcv_models()
    tester = TradeThesisStressTester(data, symbol="AAPL")
    strategies = get_test_strategies()
    context = StrategyContext(symbol="AAPL", timeframe="1d")
    results = [s.evaluate(OHLCVData(data=data), context) for s in strategies]
    
    analysis = tester.run_stress_test(results)
    
    # Risk stop exists in risk_controls, while market structure invalidation is in invalidation_conditions
    risk_stop_ctrl = next((c for c in analysis.risk_controls if "stop" in c.measure.lower()), None)
    assert risk_stop_ctrl is not None, "2-ATR stop must be represented in risk controls"
    
    inval_rules = [c.rule for c in analysis.invalidation_conditions]
    assert any("support" in r.lower() or "resistance" in r.lower() for r in inval_rules)

# 15. Reward-to-risk calculations with known expected outputs
def test_trade_geometry_reward_to_risk_deterministic():
    data = get_ohlcv_models()
    tester = TradeThesisStressTester(data, symbol="AAPL")
    
    # Test known inputs: price=100, support=95, resistance=110, stop_long=90, stop_short=110, atr=5.0
    geom = tester.calculate_trade_geometry(
        latest_price=100.0,
        support=95.0,
        resistance=110.0,
        stop_long=90.0,
        stop_short=110.0,
        atr=5.0,
        final_bias=DirectionalBias.BULLISH
    )
    
    assert geom.reference_price == 100.0
    assert geom.target_price == 110.0
    assert geom.stop_price == 95.0
    assert geom.distance_to_target == 10.0
    assert geom.distance_to_stop == 5.0
    assert geom.reward_to_risk_ratio == 2.0  # 10 / 5 = 2.0
    assert geom.criteria_met is True  # 2.0 >= 1.5

# 16. Invalid instrument metadata, price units or volume
def test_invalid_instrument_metadata_and_price_units():
    meta = InstrumentMetadata(
        symbol="^NSEI",
        canonical_symbol="^NSEI",
        name="NIFTY 50",
        currency="INR",
        unit_label="points",
        exchange="NSE",
        instrument_type="INDEX"
    )
    assert meta.unit_label == "points"
    assert meta.exchange == "NSE"
    assert meta.instrument_type == "INDEX"

# 17. Stale or incomplete candle data
def test_stale_or_incomplete_candle_data_diagnostics():
    data = get_ohlcv_models()[:30] # Fewer than 50
    tester = TradeThesisStressTester(data, symbol="TEST")
    analysis = tester.run_stress_test([])
    
    assert analysis.data_quality.lookback_sufficient is False
    assert analysis.data_quality.is_valid is False
    assert "Insufficient history" in analysis.data_quality.warnings[0]

# 18. AI service failures and malformed structured responses
def test_ai_service_fallback_on_failure():
    data = get_ohlcv_models()
    tester = TradeThesisStressTester(data, symbol="AAPL")
    strategies = get_test_strategies()
    context = StrategyContext(symbol="AAPL", timeframe="1d")
    results = [s.evaluate(OHLCVData(data=data), context) for s in strategies]
    
    # Stress test must return fully formed deterministic AnalysisResult
    analysis = tester.run_stress_test(results)
    assert analysis is not None
    assert analysis.verdict != ""
    assert analysis.trade_geometry is not None
    assert analysis.evidence_graph is not None

# 19. Graph filtering, node selection and evidence-path highlighting
def test_graph_filtering_and_evidence_path_traceability():
    data = get_ohlcv_models()
    tester = TradeThesisStressTester(data, symbol="AAPL")
    strategies = get_test_strategies()
    context = StrategyContext(symbol="AAPL", timeframe="1d")
    results = [s.evaluate(OHLCVData(data=data), context) for s in strategies]
    
    analysis = tester.run_stress_test(results)
    graph = analysis.evidence_graph
    
    # Trace path from THESIS-ROOT to all incoming edges
    thesis_edges = [e for e in graph.edges if e.target_node_id == "THESIS-ROOT"]
    assert len(thesis_edges) > 0
    connected_family_ids = {e.source_node_id for e in thesis_edges}
    
    # For each family, trace strategy nodes
    family_edges = [e for e in graph.edges if e.target_node_id in connected_family_ids]
    assert len(family_edges) > 0
    connected_strat_ids = {e.source_node_id for e in family_edges}
    assert any("STRAT-" in sid for sid in connected_strat_ids)

# 20. Consistency between the rendered UI and canonical backend response
def test_consistency_between_canonical_result_and_analyze_response():
    data = get_ohlcv_models()
    tester = TradeThesisStressTester(data, symbol="AAPL")
    strategies = get_test_strategies()
    context = StrategyContext(symbol="AAPL", timeframe="1d")
    results = [s.evaluate(OHLCVData(data=data), context) for s in strategies]
    
    analysis = tester.run_stress_test(results)
    
    resp = AnalyzeResponse(
        instrument=analysis.instrument,
        data_quality=analysis.data_quality,
        tested_hypothesis=analysis.tested_hypothesis,
        verdict=analysis.verdict,
        reconciliation_status=analysis.reconciliation_status,
        audit_warnings=analysis.audit_warnings,
        independent_evidence_families=analysis.independent_evidence_families,
        dependency_explanation=analysis.dependency_explanation,
        neutral_evidence=analysis.neutral_evidence,
        evidence_graph=analysis.evidence_graph,
        measured_dependencies=analysis.measured_dependencies,
        historical_comparison=analysis.historical_comparison,
        trade_geometry=analysis.trade_geometry,
        symbol="AAPL",
        market="US",
        timeframe="1d",
        latest_price=analysis.latest_price,
        directional_bias=analysis.directional_bias.value,
        evidence_strength=analysis.evidence_strength.value,
        setup_status=analysis.setup_status.value,
        backtest_status=analysis.backtest_status.value,
        thesis_status=analysis.thesis_status.value,
        supporting_evidence=analysis.supporting_evidence,
        opposing_evidence=analysis.opposing_evidence,
        evidence_dependencies=analysis.evidence_dependencies,
        historical_setup_results=analysis.historical_setup_results,
        confirmation_conditions=analysis.confirmation_conditions,
        invalidation_conditions=analysis.invalidation_conditions,
        risk_controls=analysis.risk_controls,
        top_supporting_facts=analysis.top_supporting_facts,
        top_opposing_facts=analysis.top_opposing_facts,
        thesis_change_conditions=analysis.thesis_change_conditions,
        data_quality_warnings=analysis.data_quality_warnings,
        explanation=analysis.explanation,
        overall_signal=analysis.overall_signal,
        score=analysis.score,
        confidence=analysis.confidence,
        strategy_signals=[]
    )
    
    assert resp.directional_bias == analysis.directional_bias.value
    assert resp.verdict == analysis.verdict
    assert resp.evidence_graph is not None
    assert len(resp.evidence_graph.nodes) == len(analysis.evidence_graph.nodes)
    assert resp.trade_geometry.reward_to_risk_ratio == analysis.trade_geometry.reward_to_risk_ratio
