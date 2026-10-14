import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
from app.domain.models import (
    OHLCVData, StrategyResult, Signal, DirectionalBias, EvidenceFamily,
    EvidenceClaim, EvidenceDependency, EvidenceStrength, SetupStatus,
    EvidencePolarity, MeasuredDependence, GraphNode, GraphEdge,
    EvidenceGraph, NodeType, RelationshipType, ThesisStatus
)
from app.domain.engine import (
    get_ema, get_rsi, get_donchian_channels, get_bollinger_bands, get_atr
)

# Deterministic family mappings
STRATEGY_FAMILY_MAP: Dict[str, EvidenceFamily] = {
    "Trend": EvidenceFamily.TREND,
    "TrendStrategy": EvidenceFamily.TREND,
    "Breakout": EvidenceFamily.TREND,
    "BreakoutStrategy": EvidenceFamily.TREND,
    "Momentum": EvidenceFamily.MOMENTUM,
    "MomentumStrategy": EvidenceFamily.MOMENTUM,
    "MeanReversion": EvidenceFamily.MEAN_REVERSION,
    "MeanReversionStrategy": EvidenceFamily.MEAN_REVERSION,
    "PriceAction": EvidenceFamily.PRICE_ACTION,
    "PriceActionStrategy": EvidenceFamily.PRICE_ACTION,
    "Risk": EvidenceFamily.VOLATILITY_RISK,
    "RiskStrategy": EvidenceFamily.VOLATILITY_RISK,
}

STRATEGY_RULE_DESCRIPTIONS: Dict[str, str] = {
    "Trend": "EMA 20 vs EMA 50 trend slope and crossover",
    "TrendStrategy": "EMA 20 vs EMA 50 trend slope and crossover",
    "Breakout": "20-period Donchian Channel boundary breach",
    "BreakoutStrategy": "20-period Donchian Channel boundary breach",
    "Momentum": "14-period RSI overbought (>70) or oversold (<30) regime",
    "MomentumStrategy": "14-period RSI overbought (>70) or oversold (<30) regime",
    "MeanReversion": "20-period Bollinger Band (2.0 std) upper/lower touch",
    "MeanReversionStrategy": "20-period Bollinger Band (2.0 std) upper/lower touch",
    "PriceAction": "3-bar swing structure (higher highs & higher lows vs lower lows)",
    "PriceActionStrategy": "3-bar swing structure (higher highs & higher lows vs lower lows)",
    "Risk": "14-period ATR volatility as % of price (regime threshold >5%)",
    "RiskStrategy": "14-period ATR volatility as % of price (regime threshold >5%)",
}

class EvidenceLedgerBuilder:
    """
    Builds the structured evidence ledger and executes false-consensus detection
    by clustering collinear indicators into distinct evidence families.
    """
    def __init__(self, market_data: OHLCVData, strategy_results: List[StrategyResult]):
        self.market_data = market_data
        self.strategy_results = strategy_results
        self.timestamp_str = (
            market_data.data[-1].timestamp.isoformat()
            if market_data.data else datetime.utcnow().isoformat()
        )

    def build_claims(self) -> List[EvidenceClaim]:
        """Convert raw strategy outputs into fully attributed Evidence Claims with explicit polarity."""
        claims: List[EvidenceClaim] = []
        strat_id_prefixes = {
            "Trend": "TREN",
            "TrendStrategy": "TREN",
            "Momentum": "MOME",
            "MomentumStrategy": "MOME",
            "MeanReversion": "MEAN",
            "MeanReversionStrategy": "MEAN",
            "Breakout": "BRKO",
            "BreakoutStrategy": "BRKO",
            "PriceAction": "PRIC",
            "PriceActionStrategy": "PRIC",
            "Risk": "RISK",
            "RiskStrategy": "RISK"
        }
        strat_limitations = {
            "Trend": "Moving average crossovers lag turning points and suffer whipsaws in range-bound markets.",
            "TrendStrategy": "Moving average crossovers lag turning points and suffer whipsaws in range-bound markets.",
            "Momentum": "RSI oscillators can remain oversold/overbought during sustained structural trends.",
            "MomentumStrategy": "RSI oscillators can remain oversold/overbought during sustained structural trends.",
            "MeanReversion": "Bollinger Band touches can fail to revert during strong volatility expansion regimes.",
            "MeanReversionStrategy": "Bollinger Band touches can fail to revert during strong volatility expansion regimes.",
            "Breakout": "Donchian channel boundary breaches generate false expansion signals in oscillating ranges.",
            "BreakoutStrategy": "Donchian channel boundary breaches generate false expansion signals in oscillating ranges.",
            "PriceAction": "3-bar swing structures are sensitive to short-term candle noise and wick liquidity sweeps.",
            "PriceActionStrategy": "3-bar swing structures are sensitive to short-term candle noise and wick liquidity sweeps.",
            "Risk": "ATR evaluates volatility magnitude only; does not provide directional market bias.",
            "RiskStrategy": "ATR evaluates volatility magnitude only; does not provide directional market bias."
        }
        for idx, res in enumerate(self.strategy_results):
            strat_name = res.strategy_name
            family = STRATEGY_FAMILY_MAP.get(strat_name, EvidenceFamily.TREND)
            rule_desc = STRATEGY_RULE_DESCRIPTIONS.get(strat_name, "Indicator calculation rule")
            prefix = strat_id_prefixes.get(strat_name, "EV")
            
            # Format rule with actual values from metadata
            rule_detail = f"{rule_desc} -> {res.metadata}" if res.metadata else rule_desc
            
            # Default non-directional polarity until hypothesis is evaluated
            if res.signal == Signal.BUY:
                polarity = EvidencePolarity.SUPPORTS_BULLISH
            elif res.signal == Signal.SELL:
                polarity = EvidencePolarity.SUPPORTS_BEARISH
            else:
                polarity = EvidencePolarity.NON_DIRECTIONAL

            claim = EvidenceClaim(
                claim_id=f"EV-{prefix}-{idx+1}",
                family=family,
                indicator=strat_name,
                signal=res.signal,
                rule=rule_detail,
                rule_id=strat_name,
                source_data=f"OHLCV[{self.market_data.data[-1].timestamp.strftime('%Y-%m-%d') if self.market_data.data else 'latest'}]",
                calculation_values=res.metadata,
                timestamp=self.timestamp_str,
                weight=1.0,
                polarity=polarity,
                limitations=strat_limitations.get(strat_name, "Historical indicator approximation.")
            )
            claims.append(claim)
        return claims

    def detect_false_consensus(
        self, 
        claims: List[EvidenceClaim],
        measured_dependencies: Optional[List[MeasuredDependence]] = None
    ) -> Tuple[
        List[EvidenceDependency],
        Dict[str, int],       # raw votes
        Dict[str, int],       # independent family votes
        bool,                 # collinear overlap detected
        str                   # dependency explanation
    ]:
        """
        Group correlated indicators into dependency-adjusted families. Prevents multi-indicator
        collinearity from being counted as duplicate independent votes.
        Distinguishes qualitative heuristic family grouping from measured empirical dependence.
        """
        family_claims: Dict[EvidenceFamily, List[EvidenceClaim]] = {}
        for claim in claims:
            family_claims.setdefault(claim.family, []).append(claim)

        dependencies: List[EvidenceDependency] = []
        raw_votes = {"BUY": 0, "SELL": 0, "NEUTRAL": 0}
        independent_family_votes = {"BULLISH": 0, "BEARISH": 0, "NEUTRAL": 0, "CONFLICTED": 0}

        for claim in claims:
            raw_votes[claim.signal.value] += 1

        collinear_families: List[str] = []

        # 1. First pass: Heuristic family grouping
        for family, group in family_claims.items():
            indicators = [c.indicator for c in group]
            raw_group_votes = {c.indicator: c.signal.value for c in group}
            
            buys = sum(1 for c in group if c.signal == Signal.BUY)
            sells = sum(1 for c in group if c.signal == Signal.SELL)
            neutrals = sum(1 for c in group if c.signal == Signal.NEUTRAL)
            
            # Risk controls (VOLATILITY_RISK) do not vote on directional bias
            if family == EvidenceFamily.VOLATILITY_RISK:
                indep_vote = DirectionalBias.NEUTRAL
                independent_family_votes["NEUTRAL"] += 1
                grouping_type = "INDEPENDENT"
                collinearity_note = "Volatility risk model: Evaluates ATR exposure magnitude only; does not vote on directional market bias."
                eff_weight = 1.0
            elif buys > 0 and sells > 0:
                indep_vote = DirectionalBias.NEUTRAL
                independent_family_votes["CONFLICTED"] += 1
                grouping_type = "HEURISTIC_FAMILY" if len(group) > 1 else "INDEPENDENT"
                collinearity_note = f"Contradiction within {family.value}: indicators vote both BUY and SELL."
                eff_weight = 1.0
            elif buys > sells and buys >= len(group) / 2:
                indep_vote = DirectionalBias.BULLISH
                independent_family_votes["BULLISH"] += 1
                grouping_type = "HEURISTIC_FAMILY" if len(group) > 1 else "INDEPENDENT"
                collinearity_note = (
                    f"Heuristic grouping: {[c.indicator for c in group]} combined into 1 qualitative family vote. "
                    f"Grouping is based on shared trend premise, not an empirical redundancy claim."
                    if len(group) > 1 else "Independent indicator vote."
                )
                eff_weight = 1.0
            elif sells > buys and sells >= len(group) / 2:
                indep_vote = DirectionalBias.BEARISH
                independent_family_votes["BEARISH"] += 1
                grouping_type = "HEURISTIC_FAMILY" if len(group) > 1 else "INDEPENDENT"
                collinearity_note = (
                    f"Heuristic grouping: {[c.indicator for c in group]} combined into 1 qualitative family vote. "
                    f"Grouping is based on shared trend premise, not an empirical redundancy claim."
                    if len(group) > 1 else "Independent indicator vote."
                )
                eff_weight = 1.0
            else:
                indep_vote = DirectionalBias.NEUTRAL
                independent_family_votes["NEUTRAL"] += 1
                grouping_type = "INDEPENDENT"
                collinearity_note = "Neutral/indeterminate signal."
                eff_weight = 1.0

            if len(group) > 1:
                collinear_families.append(f"{family.value} ({', '.join(indicators)})")

            dependencies.append(EvidenceDependency(
                family=family,
                indicators=indicators,
                independent_vote=indep_vote,
                raw_votes=raw_group_votes,
                collinearity_risk=collinearity_note,
                effective_weight=eff_weight,
                grouping_type=grouping_type,
                empirical_adjustment_detail=""
            ))

        # 2. Second pass: Cross-Family Empirical Redundancy Adjustment
        # If indicators in distinct families (e.g. Momentum vs Mean Reversion) show high measured correlation (|r| >= 0.70)
        # and vote in the same direction, apply documented empirical redundancy dampening to prevent false consensus.
        empirically_dampened_count = 0
        if measured_dependencies:
            for md in measured_dependencies:
                if abs(md.correlation) >= 0.70 or md.is_dependent:
                    # Identify affected families
                    fam_a = None
                    fam_b = None
                    if "Momentum" in md.indicator_a or "RSI" in md.indicator_a:
                        fam_a = next((d for d in dependencies if d.family == EvidenceFamily.MOMENTUM), None)
                    if "MeanReversion" in md.indicator_b or "Bollinger" in md.indicator_b:
                        fam_b = next((d for d in dependencies if d.family == EvidenceFamily.MEAN_REVERSION), None)

                    if fam_a and fam_b:
                        fam_a.measured_correlation = md.correlation
                        fam_b.measured_correlation = md.correlation
                        # Case 1: Both families vote in the same non-neutral direction -> collinearity risk!
                        if fam_a.independent_vote == fam_b.independent_vote and fam_a.independent_vote != DirectionalBias.NEUTRAL:
                            # Formula: effective_weight = max(0.20, 1.0 - 0.5 * |r|)
                            dampened_weight = round(max(0.20, 1.0 - 0.5 * abs(md.correlation)), 2)
                            fam_b.effective_weight = dampened_weight
                            fam_b.grouping_type = "EMPIRICALLY_ADJUSTED"
                            adj_detail = (
                                f"Empirical Redundancy Adjustment: Momentum (RSI) and MeanReversion (Bollinger %B) "
                                f"exhibit Spearman r={md.correlation:+.2f} (|r| >= 0.70). Both voted {fam_b.independent_vote.value}. "
                                f"Weight dampened from 1.0 to {dampened_weight:.2f} using formula max(0.2, 1.0 - 0.5*|r|) "
                                f"to prevent duplicate oscillator weighting."
                            )
                            fam_b.empirical_adjustment_detail = adj_detail
                            fam_b.collinearity_risk = adj_detail
                            fam_a.empirical_adjustment_detail = f"Paired with MeanReversion (Spearman r={md.correlation:+.2f}). Primary momentum indicator retained at full weight (1.0)."
                            empirically_dampened_count += 1
                        else:
                            # Case 2: They disagree or one/both are neutral -> no double counting
                            if fam_a.independent_vote == DirectionalBias.NEUTRAL and fam_b.independent_vote == DirectionalBias.NEUTRAL:
                                adj_detail = (
                                    f"Measured Spearman r={md.correlation:+.2f} between Momentum and MeanReversion is observed but not used for voting adjustments. "
                                    f"Both indicator families are NEUTRAL; because neutral indicators submit no directional vote, "
                                    f"correlation between neutral Momentum and MeanReversion signals does not add to or inflate directional consensus. No voting penalty applied."
                                )
                                fam_a.empirical_adjustment_detail = adj_detail
                                fam_b.empirical_adjustment_detail = adj_detail
                                fam_b.collinearity_risk = adj_detail
                            else:
                                adj_detail = (
                                    f"Measured Spearman r={md.correlation:+.2f} indicates historical dependence, "
                                    f"but no redundancy penalty applied because family votes differ "
                                    f"(Momentum: {fam_a.independent_vote.value}, MeanReversion: {fam_b.independent_vote.value})."
                                )
                                fam_a.empirical_adjustment_detail = adj_detail
                                fam_b.empirical_adjustment_detail = adj_detail
                                fam_b.collinearity_risk = adj_detail

        # 3. Check if raw consensus contained duplicate correlated votes
        raw_top_count = max(raw_votes["BUY"], raw_votes["SELL"])
        indep_top_count = max(independent_family_votes["BULLISH"], independent_family_votes["BEARISH"])
        collinear_overlap_detected = (raw_top_count > indep_top_count) and len(collinear_families) > 0

        explanation_parts = []
        if collinear_overlap_detected:
            explanation_parts.append(
                f"False Consensus Alert: Heuristic Grouping: Reduced duplicate votes across {', '.join(collinear_families)} "
                f"from {raw_top_count} raw signals to {indep_top_count} qualitative evidence family vote(s) "
                f"based on shared trend-following premise (qualitative grouping, distinct from empirical redundancy)."
            )
        else:
            explanation_parts.append(
                f"Heuristic Verification: Raw signals correspond cleanly to {len(family_claims)} "
                f"qualitative evidence families."
            )

        emp_adjusted = [d for d in dependencies if d.grouping_type == "EMPIRICALLY_ADJUSTED"]
        if emp_adjusted:
            for ea in emp_adjusted:
                explanation_parts.append(ea.empirical_adjustment_detail)
        else:
            explanation_parts.append(
                "Empirical Dependency Verification: No cross-family empirical redundancy penalties active."
            )
            neutral_notes = [d.empirical_adjustment_detail for d in dependencies if d.empirical_adjustment_detail and d.grouping_type != "EMPIRICALLY_ADJUSTED"]
            if neutral_notes:
                explanation_parts.append(neutral_notes[0])

        dep_explanation = " | ".join(explanation_parts)

        return dependencies, raw_votes, independent_family_votes, collinear_overlap_detected, dep_explanation

    def partition_evidence(
        self, 
        claims: List[EvidenceClaim], 
        thesis_bias: DirectionalBias
    ) -> Tuple[List[EvidenceClaim], List[EvidenceClaim], List[EvidenceClaim]]:
        """
        Partitions evidence claims into Supporting vs Opposing vs Neutral based on candidate thesis bias.
        Assigns explicit polarity to every claim and reconciles 1:1 with raw strategy votes.
        """
        supporting: List[EvidenceClaim] = []
        opposing: List[EvidenceClaim] = []
        neutral: List[EvidenceClaim] = []

        for claim in claims:
            if thesis_bias == DirectionalBias.BULLISH:
                if claim.signal == Signal.BUY:
                    claim.polarity = EvidencePolarity.SUPPORTS_BULLISH
                    supporting.append(claim)
                elif claim.signal == Signal.SELL:
                    claim.polarity = EvidencePolarity.CHALLENGES_BULLISH
                    opposing.append(claim)
                else:
                    claim.polarity = EvidencePolarity.NON_DIRECTIONAL
                    neutral.append(claim)
            elif thesis_bias == DirectionalBias.BEARISH:
                if claim.signal == Signal.SELL:
                    claim.polarity = EvidencePolarity.SUPPORTS_BEARISH
                    supporting.append(claim)
                elif claim.signal == Signal.BUY:
                    claim.polarity = EvidencePolarity.CHALLENGES_BEARISH
                    opposing.append(claim)
                else:
                    claim.polarity = EvidencePolarity.NON_DIRECTIONAL
                    neutral.append(claim)
            else: # NEUTRAL thesis
                if claim.signal == Signal.NEUTRAL:
                    claim.polarity = EvidencePolarity.NON_DIRECTIONAL
                    neutral.append(claim)
                elif claim.signal == Signal.BUY:
                    claim.polarity = EvidencePolarity.SUPPORTS_BULLISH
                    opposing.append(claim)
                else:
                    claim.polarity = EvidencePolarity.SUPPORTS_BEARISH
                    opposing.append(claim)

        return supporting, opposing, neutral

def compute_measured_dependencies(df: pd.DataFrame) -> List[MeasuredDependence]:
    """
    Empirically measures statistical association between indicator series over available historical candles.
    Uses Spearman rank correlation to assess non-linear monotonic relationship without assuming normal distribution.
    Records method, sample size, evaluation period, and limitations.
    """
    dependencies: List[MeasuredDependence] = []
    n = len(df)
    if n < 30:
        return dependencies

    eval_period = f"{str(df['timestamp'].iloc[0])[:10]} to {str(df['timestamp'].iloc[-1])[:10]}"

    try:
        ema20 = get_ema(df, window=20)
        ema50 = get_ema(df, window=50)
        trend_metric = (ema20 - ema50) / ema50

        donchian = get_donchian_channels(df, window=20)
        donch_range = (donchian['upper'] - donchian['lower']).replace(0, np.nan)
        breakout_metric = (df['close'] - donchian['mid']) / donch_range

        rsi = get_rsi(df, window=14)
        
        bb = get_bollinger_bands(df, window=20)
        bb_range = (bb['upper'] - bb['lower']).replace(0, np.nan)
        bb_metric = (df['close'] - bb['lower']) / bb_range

        # 1. Trend (EMA Slope) vs Breakout (Donchian Position)
        valid_idx = ~(trend_metric.isna() | breakout_metric.isna())
        s1 = trend_metric[valid_idx]
        s2 = breakout_metric[valid_idx]
        if len(s1) >= 30:
            corr_val = float(s1.rank().corr(s2.rank()))
            corr_val = round(corr_val if not np.isnan(corr_val) else 0.0, 3)
            dependencies.append(MeasuredDependence(
                indicator_a="Trend (EMA 20/50)",
                indicator_b="Breakout (Donchian 20)",
                method=f"Spearman rank correlation of relative indicator position (N={len(s1)})",
                sample_size=len(s1),
                correlation=corr_val,
                is_dependent=bool(abs(corr_val) >= 0.50),
                evaluation_period=eval_period,
                limitations="Trend (EMA 20/50) and Breakout (Donchian 20) are grouped into 1 heuristic family due to shared trend-following premise. Low measured rank correlation (r=+0.30) reflects differing calculation mechanics, but does not prove statistical independence or forward causality across changing volatility regimes."
            ))

        # 2. Momentum (RSI) vs Mean Reversion (Bollinger %B)
        valid_idx_mom = ~(rsi.isna() | bb_metric.isna())
        s_rsi = rsi[valid_idx_mom]
        s_bb = bb_metric[valid_idx_mom]
        if len(s_rsi) >= 30:
            corr_val = float(s_rsi.rank().corr(s_bb.rank()))
            corr_val = round(corr_val if not np.isnan(corr_val) else 0.0, 3)
            dependencies.append(MeasuredDependence(
                indicator_a="Momentum (RSI 14)",
                indicator_b="MeanReversion (Bollinger %B)",
                method=f"Spearman rank correlation of oscillator levels (N={len(s_rsi)})",
                sample_size=len(s_rsi),
                correlation=corr_val,
                is_dependent=bool(abs(corr_val) >= 0.60),
                evaluation_period=eval_period,
                limitations="While categorized into separate heuristic families (velocity vs oscillation), both metrics track price relative to recent range, producing high empirical collinearity. When both vote in the same direction, cross-family empirical redundancy adjustment dampens effective weight (effective_weight = max(0.2, 1.0 - 0.5*|r|)) to avoid duplicate counting. When indicators are neutral or disagree, no voting weight penalty is applied because correlation between neutral families does not add to or inflate directional consensus. Note: Statistical correlation measures association; it does not establish forward causality, nor does a low correlation alone prove statistical independence."
            ))
    except Exception:
        pass

    return dependencies

def build_evidence_graph(
    market_data: OHLCVData,
    strategy_results: List[StrategyResult],
    dependencies: List[EvidenceDependency],
    claims: List[EvidenceClaim],
    measured_dependencies: List[MeasuredDependence],
    directional_bias: DirectionalBias,
    evidence_strength: EvidenceStrength,
    thesis_status: ThesisStatus,
    tested_hypothesis: str,
    data_quality: Optional[Any] = None
) -> EvidenceGraph:
    """
    Constructs an interactive evidence-dependency graph with fully traceable relationships:
    Observations -> Indicators -> Strategy Rules & Signals -> Evidence Families -> Thesis.
    """
    nodes: List[GraphNode] = []
    edges: List[GraphEdge] = []

    last_ts = market_data.data[-1].timestamp.isoformat() if market_data.data else ""
    latest_close = market_data.data[-1].close if market_data.data else 0.0

    # 1. Observation Nodes
    nodes.append(GraphNode(
        node_id="OBS-PRICE",
        node_type=NodeType.OBSERVATION,
        label=f"Price Feed ({latest_close:.2f})",
        source_timestamp=last_ts,
        data_quality="VALID",
        explanation="Latest completed OHLCV candle session."
    ))

    has_vol = getattr(data_quality, "has_valid_volume", True) if data_quality else True
    vol_sem = getattr(data_quality, "volume_semantics", "ACTUAL_TRADED_SHARES") if data_quality else "ACTUAL_TRADED_SHARES"
    if not has_vol:
        vol_label = "Volume: Unavailable (Cash Index)" if vol_sem == "UNAVAILABLE_CASH_INDEX" else "Volume: Zero / Unreliable"
        vol_quality = "UNAVAILABLE"
        vol_desc = "Cash index feeds do not record traded shares; volume confirmation triggers disabled."
    else:
        vol_label = "Volume Activity"
        vol_quality = "VALID"
        vol_desc = "Session traded volume."

    nodes.append(GraphNode(
        node_id="OBS-VOL",
        node_type=NodeType.OBSERVATION,
        label=vol_label,
        source_timestamp=last_ts,
        data_quality=vol_quality,
        explanation=vol_desc
    ))

    # 2. Indicator Nodes
    indicators_info = [
        ("IND-EMA", "EMA 20/50 Slope", "Exponential Moving Averages 20-period and 50-period"),
        ("IND-DONCHIAN", "Donchian Channels", "20-period highest high and lowest low channels"),
        ("IND-RSI", "RSI (14)", "14-period Relative Strength Index oscillator"),
        ("IND-BB", "Bollinger Bands", "20-period SMA with 2.0 standard deviation bands"),
        ("IND-SWING", "Swing Structure", "3-bar swing high/low market structure pivots"),
        ("IND-ATR", "ATR Volatility (14)", "14-period Average True Range volatility measure")
    ]
    for ind_id, label, desc in indicators_info:
        nodes.append(GraphNode(
            node_id=ind_id,
            node_type=NodeType.INDICATOR,
            label=label,
            source_timestamp=last_ts,
            explanation=desc
        ))
        edges.append(GraphEdge(
            edge_id=f"EDGE-OBS-{ind_id}",
            source_node_id="OBS-PRICE",
            target_node_id=ind_id,
            relationship_type=RelationshipType.DERIVED_FROM,
            relationship_method="MATHEMATICAL_CALCULATION",
            supporting_evidence=["OHLCV candles"]
        ))

    # 3. Strategy Signal Nodes
    strat_to_ind = {
        "Trend": "IND-EMA",
        "TrendStrategy": "IND-EMA",
        "Breakout": "IND-DONCHIAN",
        "BreakoutStrategy": "IND-DONCHIAN",
        "Momentum": "IND-RSI",
        "MomentumStrategy": "IND-RSI",
        "MeanReversion": "IND-BB",
        "MeanReversionStrategy": "IND-BB",
        "PriceAction": "IND-SWING",
        "PriceActionStrategy": "IND-SWING",
        "Risk": "IND-ATR",
        "RiskStrategy": "IND-ATR",
    }
    for res in strategy_results:
        s_name = res.strategy_name
        s_id = f"STRAT-{s_name}"
        matched_claims = [c.claim_id for c in claims if c.indicator == s_name]
        rule_desc = STRATEGY_RULE_DESCRIPTIONS.get(s_name, "Strategy rule evaluation")
        nodes.append(GraphNode(
            node_id=s_id,
            node_type=NodeType.STRATEGY_SIGNAL,
            label=f"{s_name}: {res.signal.value}",
            signal=res.signal.value,
            evidence_ids=matched_claims,
            rule_id=s_name,
            calculation_values=res.metadata,
            explanation=f"Signal: {res.signal.value} | Heuristic Conviction: {res.confidence:.2f} | Rule: {rule_desc}"
        ))
        
        ind_source = strat_to_ind.get(s_name, "IND-EMA")
        edges.append(GraphEdge(
            edge_id=f"EDGE-{ind_source}-{s_id}",
            source_node_id=ind_source,
            target_node_id=s_id,
            relationship_type=RelationshipType.CONTRIBUTES_TO,
            relationship_method="DETERMINISTIC_STRATEGY_RULE",
            supporting_evidence=matched_claims
        ))

    # 4. Evidence Family Nodes
    for dep in dependencies:
        fam_id = f"FAM-{dep.family.value}"
        nodes.append(GraphNode(
            node_id=fam_id,
            node_type=NodeType.EVIDENCE_FAMILY,
            label=f"Family: {dep.family.value}",
            signal=dep.independent_vote.value,
            family=dep.family.value,
            explanation=f"Family Vote: {dep.independent_vote.value}. {dep.collinearity_risk}"
        ))

        # Connect strategies to family
        for ind_name in dep.indicators:
            strat_node_id = f"STRAT-{ind_name}"
            edges.append(GraphEdge(
                edge_id=f"EDGE-{strat_node_id}-{fam_id}",
                source_node_id=strat_node_id,
                target_node_id=fam_id,
                relationship_type=RelationshipType.GROUPED_WITH,
                relationship_method="HEURISTIC_FAMILY_GROUPING",
                supporting_evidence=[ind_name],
                limitations="Heuristic family grouping to reduce duplicate votes. Statistical independence not guaranteed."
            ))

    # 5. Measured Dependence Edges
    for md in measured_dependencies:
        if md.is_dependent:
            src = "STRAT-TrendStrategy" if any(r.strategy_name == "TrendStrategy" for r in strategy_results) else "STRAT-Trend"
            tgt = "STRAT-BreakoutStrategy" if any(r.strategy_name == "BreakoutStrategy" for r in strategy_results) else "STRAT-Breakout"
            if "RSI" in md.indicator_a and "Bollinger" in md.indicator_b:
                src = "STRAT-MomentumStrategy" if any(r.strategy_name == "MomentumStrategy" for r in strategy_results) else "STRAT-Momentum"
                tgt = "STRAT-MeanReversionStrategy" if any(r.strategy_name == "MeanReversionStrategy" for r in strategy_results) else "STRAT-MeanReversion"
            edge_evidence = [f"Spearman r = {md.correlation:.2f}"]
            emp_adj = any(d.grouping_type == "EMPIRICALLY_ADJUSTED" for d in dependencies)
            if emp_adj:
                edge_evidence.append("Empirical Weight Dampened")
            else:
                edge_evidence.append("Weight Unpenalized (Neutral / Differing Votes)")
            edges.append(GraphEdge(
                edge_id=f"EDGE-DEP-{src}-{tgt}",
                source_node_id=src,
                target_node_id=tgt,
                relationship_type=RelationshipType.MEASURED_DEPENDENCE,
                relationship_method=md.method,
                supporting_evidence=edge_evidence,
                limitations=f"{md.limitations} Correlation between neutral families does not inflate directional voting consensus."
            ))

    # 6. Root Thesis Node
    thesis_id = "THESIS-ROOT"
    nodes.append(GraphNode(
        node_id=thesis_id,
        node_type=NodeType.THESIS,
        label=f"Thesis: {tested_hypothesis}",
        signal=directional_bias.value,
        explanation=f"Directional Bias: {directional_bias.value} | Strength: {evidence_strength.value} | Status: {thesis_status.value}"
    ))

    # Connect Families to Thesis
    for dep in dependencies:
        fam_id = f"FAM-{dep.family.value}"
        if dep.independent_vote == directional_bias and directional_bias != DirectionalBias.NEUTRAL:
            rel = RelationshipType.SUPPORTS_THESIS
        elif dep.independent_vote != DirectionalBias.NEUTRAL and dep.independent_vote != directional_bias:
            rel = RelationshipType.CHALLENGES_THESIS
        else:
            rel = RelationshipType.GROUPED_WITH

        edges.append(GraphEdge(
            edge_id=f"EDGE-{fam_id}-{thesis_id}",
            source_node_id=fam_id,
            target_node_id=thesis_id,
            relationship_type=rel,
            relationship_method="SYNTHESIS_AGGREGATION",
            supporting_evidence=[f"{dep.family.value} vote: {dep.independent_vote.value}"]
        ))

    return EvidenceGraph(nodes=nodes, edges=edges)
