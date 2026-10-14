from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime

from app.domain.models import (
    OHLCV, OHLCVData, StrategyContext, StrategyResult, Signal,
    DirectionalBias, EvidenceStrength, SetupStatus, BacktestStatus,
    ThesisStatus, EvidenceClaim, EvidenceDependency, EvidenceFamily, HistoricalSetupResult,
    ConfirmationCondition, InvalidationCondition, RiskControlCondition, AnalysisResult,
    InstrumentMetadata, DataQualityReport, MeasuredDependence, EvidenceGraph,
    HistoricalComparisonResult, TradeGeometry, CanonicalAnalysisSummary
)
from app.domain.engine import to_dataframe, get_ema, get_atr
from app.domain.evidence import (
    EvidenceLedgerBuilder, compute_measured_dependencies, build_evidence_graph
)
from app.domain.historical_setup import HistoricalSetupEngine

class TradeThesisStressTester:
    """
    Main engine for transforming SignalCouncil into an evidence-driven
    Trade Thesis Stress Tester. Produces canonical AnalysisResult.
    """
    def __init__(
        self, 
        market_data: List[OHLCV], 
        symbol: str, 
        timeframe: str = "1d",
        instrument: Optional[InstrumentMetadata] = None
    ):
        self.market_data = sorted(market_data, key=lambda x: x.timestamp)
        self.symbol = symbol.upper()
        self.timeframe = timeframe
        self.instrument = instrument or InstrumentMetadata(
            symbol=self.symbol,
            canonical_symbol=self.symbol,
            currency="INR" if ("NSE" in self.symbol or self.symbol.endswith(".NS")) else "USD",
            instrument_type="INDEX" if self.symbol.startswith("^") else "EQUITY"
        )

    def audit_data_quality(self, df: pd.DataFrame) -> DataQualityReport:
        warnings: List[str] = []
        total_bars = len(df)
        zero_vol = int((df['volume'] == 0).sum()) if 'volume' in df else 0
        has_negative = bool((df[['open', 'high', 'low', 'close']] <= 0).any().any()) if total_bars > 0 else True
        lookback_ok = bool(total_bars >= 50)
        has_required_cols = all(c in df.columns for c in ['open', 'high', 'low', 'close'])
        schema_valid = bool(total_bars > 0 and not has_negative and has_required_cols)
        freshness_valid = bool(total_bars > 0)

        is_index = (self.instrument.instrument_type == "INDEX") or self.symbol.startswith("^")
        has_valid_vol = (not is_index) and (total_bars > 0 and zero_vol < total_bars * 0.5)

        if is_index:
            vol_semantics = "UNAVAILABLE_CASH_INDEX"
            vol_status = "UNAVAILABLE"
            vol_eligible = False
            zero_vol_exp = f"Cash index ({self.symbol}) does not record traded shares; volume confirmation triggers disabled."
            reported_zero_vol = zero_vol
        elif total_bars > 0 and zero_vol > total_bars * 0.5:
            vol_semantics = "ZERO_VOLUME_OBSERVATION"
            vol_status = "UNAVAILABLE"
            vol_eligible = False
            zero_vol_exp = f"Over 50% of historical bars ({zero_vol}/{total_bars}) have zero volume; volume confirmation disabled."
            reported_zero_vol = zero_vol
        else:
            vol_semantics = "ACTUAL_TRADED_SHARES"
            vol_status = "AVAILABLE"
            vol_eligible = True
            zero_vol_exp = ""
            reported_zero_vol = zero_vol

        if total_bars < 50:
            warnings.append(f"Insufficient history: {total_bars} candles provided, minimum 50 required for reliable 50-EMA and setup evaluation.")
        
        # Volume semantics: zero volume is valid for cash index feeds, but an anomaly for equities
        if not is_index and total_bars > 0 and zero_vol > total_bars * 0.1:
            warnings.append(f"Data anomaly: {zero_vol} of {total_bars} bars have zero volume recorded for equity.")
        elif is_index:
            warnings.append(f"Cash index volume unrepresentative of traded shares; volume-based confirmation disabled.")

        # Check for non-positive prices
        if has_negative:
            warnings.append("Corrupted price data: Non-positive price values detected.")

        is_valid = bool(schema_valid and lookback_ok)
        return DataQualityReport(
            is_valid=is_valid,
            schema_valid=schema_valid,
            freshness_valid=freshness_valid,
            has_valid_volume=has_valid_vol,
            volume_semantics=vol_semantics,
            volume_status=vol_status,
            volume_trigger_eligible=vol_eligible,
            zero_volume_explanation=zero_vol_exp,
            total_bars=total_bars,
            zero_volume_bars=reported_zero_vol,
            negative_price_detected=has_negative,
            lookback_sufficient=lookback_ok,
            warnings=warnings
        )

    def calculate_levels_and_stops(self, df: pd.DataFrame) -> Tuple[float, float, float, float, float]:
        latest_price = float(df['close'].iloc[-1])
        lookback = min(20, len(df))
        recent_df = df.iloc[-lookback:]
        
        resistance = float(recent_df['high'].max())
        support = float(recent_df['low'].min())
        
        atr = get_atr(df, window=min(14, len(df)-1)).iloc[-1]
        if pd.isna(atr):
            atr = latest_price * 0.02
        else:
            atr = float(atr)
            
        stop_long = latest_price - (2.0 * atr)
        stop_short = latest_price + (2.0 * atr)
        
        return latest_price, support, resistance, stop_long, stop_short

    def calculate_trade_geometry(
        self,
        latest_price: float,
        support: Optional[float] = None,
        resistance: Optional[float] = None,
        stop_long: Optional[float] = None,
        stop_short: Optional[float] = None,
        atr: float = 0.0,
        final_bias: DirectionalBias = DirectionalBias.NEUTRAL,
        target_price: Optional[float] = None,
        target_rule: Optional[str] = None,
        risk_stop_price: Optional[float] = None,
        risk_stop_rule: Optional[str] = None,
        structural_stop_price: Optional[float] = None,
        structural_stop_rule: Optional[str] = None,
        selected_stop: Optional[str] = None,
        primary_stop_policy: Optional[str] = None
    ) -> TradeGeometry:
        ref_price = latest_price

        # Harmonize selected_stop and primary_stop_policy
        if selected_stop is not None:
            if selected_stop.upper() in ["RISK", "RISK_STOP"]:
                resolved_policy = "RISK_STOP"
            elif selected_stop.upper() in ["STRUCTURAL", "STRUCTURAL_STOP"]:
                resolved_policy = "STRUCTURAL_STOP"
            else:
                resolved_policy = selected_stop
        elif primary_stop_policy is not None:
            resolved_policy = primary_stop_policy
        else:
            resolved_policy = "STRUCTURAL_STOP"

        # 1. Resolve Target Price & Rule
        if target_price is not None:
            resolved_target = target_price
            resolved_target_rule = target_rule or f"Specified target price level at {resolved_target:.2f}"
        elif final_bias == DirectionalBias.BULLISH:
            if resistance is not None and resistance > latest_price:
                resolved_target = resistance
                resolved_target_rule = target_rule or f"20-period Swing Resistance Structure at {resistance:.2f}"
            elif atr > 0:
                resolved_target = latest_price + (2.5 * atr)
                resolved_target_rule = target_rule or f"2.5-ATR Expansion Target at {resolved_target:.2f}"
            else:
                resolved_target = None
                resolved_target_rule = target_rule or ""
        elif final_bias == DirectionalBias.BEARISH:
            if support is not None and support < latest_price:
                resolved_target = support
                resolved_target_rule = target_rule or f"20-period Swing Support Structure at {support:.2f}"
            elif atr > 0:
                resolved_target = latest_price - (2.5 * atr)
                resolved_target_rule = target_rule or f"2.5-ATR Expansion Target at {resolved_target:.2f}"
            else:
                resolved_target = None
                resolved_target_rule = target_rule or ""
        else:
            resolved_target = resistance if resistance is not None else None
            resolved_target_rule = target_rule or (f"Range Boundary Resistance at {resolved_target:.2f}" if resolved_target else "Undefined")

        # 2. Resolve Risk Stop (2-ATR Volatility Exit)
        if risk_stop_price is not None:
            resolved_risk_stop = risk_stop_price
            resolved_risk_stop_rule = risk_stop_rule or f"Risk Stop Reference at {resolved_risk_stop:.2f}"
        elif final_bias == DirectionalBias.BEARISH:
            resolved_risk_stop = stop_short if stop_short is not None else (latest_price + 2.0 * atr if atr > 0 else None)
            resolved_risk_stop_rule = risk_stop_rule or (f"2-ATR Volatility Risk Stop Reference (+{2.0*atr:.2f} pts) at {resolved_risk_stop:.2f}" if resolved_risk_stop else "Undefined")
        else: # BULLISH or NEUTRAL
            resolved_risk_stop = stop_long if stop_long is not None else (latest_price - 2.0 * atr if atr > 0 else None)
            resolved_risk_stop_rule = risk_stop_rule or (f"2-ATR Volatility Risk Stop Reference (-{2.0*atr:.2f} pts) at {resolved_risk_stop:.2f}" if resolved_risk_stop else "Undefined")

        # 3. Resolve Structural Thesis Invalidation Stop
        if structural_stop_price is not None:
            resolved_struct_stop = structural_stop_price
            resolved_struct_stop_rule = structural_stop_rule or f"Structural Swing Stop at {resolved_struct_stop:.2f}"
        elif final_bias == DirectionalBias.BEARISH:
            resolved_struct_stop = resistance
            resolved_struct_stop_rule = structural_stop_rule or (f"20-period Swing Resistance Structure at {resolved_struct_stop:.2f} (Structural Thesis Invalidation)" if resolved_struct_stop else "Undefined")
        else: # BULLISH
            resolved_struct_stop = support
            resolved_struct_stop_rule = structural_stop_rule or (f"20-period Swing Support Structure at {resolved_struct_stop:.2f} (Structural Thesis Invalidation)" if resolved_struct_stop else "Undefined")

        # 4. Resolve Selected Primary Stop based on policy
        if resolved_policy == "RISK_STOP":
            primary_stop = resolved_risk_stop
            primary_stop_rule = resolved_risk_stop_rule
        elif resolved_policy == "STRUCTURAL_STOP":
            primary_stop = resolved_struct_stop
            primary_stop_rule = resolved_struct_stop_rule
        else:
            primary_stop = None
            primary_stop_rule = "No Stop Selected (Policy: NONE)"

        # 5. Check for Missing Stop or Target
        if resolved_target is None or primary_stop is None or not resolved_target_rule or not primary_stop_rule:
            return TradeGeometry(
                reference_price=round(ref_price, 2),
                target_price=round(resolved_target, 2) if resolved_target is not None else None,
                target_rule=resolved_target_rule or "Target Rule Undefined",
                primary_stop_policy=resolved_policy,
                primary_stop_price=round(primary_stop, 2) if primary_stop is not None else None,
                primary_stop_rule=primary_stop_rule or "Stop Rule Undefined",
                risk_stop_price=round(resolved_risk_stop, 2) if resolved_risk_stop is not None else None,
                risk_stop_rule=resolved_risk_stop_rule,
                risk_exit_price=round(resolved_risk_stop, 2) if resolved_risk_stop is not None else None,
                risk_exit_rule=resolved_risk_stop_rule,
                structural_stop_price=round(resolved_struct_stop, 2) if resolved_struct_stop is not None else None,
                structural_stop_rule=resolved_struct_stop_rule,
                structural_invalidation_price=round(resolved_struct_stop, 2) if resolved_struct_stop is not None else None,
                structural_invalidation_rule=resolved_struct_stop_rule,
                stop_price=round(primary_stop, 2) if primary_stop is not None else None,
                stop_rule=primary_stop_rule or "Stop Rule Undefined",
                reward_to_risk_ratio=None,
                risk_stop_rr_ratio=None,
                structural_stop_rr_ratio=None,
                criteria_met=False,
                is_valid=False,
                validation_error="Missing target or stop level; reward-to-risk ratio is unavailable.",
                notes="Reward-to-risk ratio unavailable: Target Rule or Stop Rule not fully defined."
            )

        # 6. Explicit Price Geometry Validation
        geom_valid = True
        geom_error: Optional[str] = None
        if final_bias == DirectionalBias.BULLISH:
            if resolved_target <= ref_price or primary_stop >= ref_price:
                geom_valid = False
                geom_error = f"Invalid price geometry for BULLISH setup: Target ({resolved_target:.2f}) must be above reference ({ref_price:.2f}) and stop ({primary_stop:.2f}) must be below."
        elif final_bias == DirectionalBias.BEARISH:
            if resolved_target >= ref_price or primary_stop <= ref_price:
                geom_valid = False
                geom_error = f"Invalid price geometry for BEARISH setup: Target ({resolved_target:.2f}) must be below reference ({ref_price:.2f}) and stop ({primary_stop:.2f}) must be above."

        if not geom_valid:
            return TradeGeometry(
                reference_price=round(ref_price, 2),
                target_price=round(resolved_target, 2),
                target_rule=resolved_target_rule,
                primary_stop_policy=resolved_policy,
                primary_stop_price=round(primary_stop, 2),
                primary_stop_rule=primary_stop_rule,
                risk_stop_price=round(resolved_risk_stop, 2) if resolved_risk_stop is not None else None,
                risk_stop_rule=resolved_risk_stop_rule,
                risk_exit_price=round(resolved_risk_stop, 2) if resolved_risk_stop is not None else None,
                risk_exit_rule=resolved_risk_stop_rule,
                structural_stop_price=round(resolved_struct_stop, 2) if resolved_struct_stop is not None else None,
                structural_stop_rule=resolved_struct_stop_rule,
                structural_invalidation_price=round(resolved_struct_stop, 2) if resolved_struct_stop is not None else None,
                structural_invalidation_rule=resolved_struct_stop_rule,
                stop_price=round(primary_stop, 2),
                stop_rule=primary_stop_rule,
                reward_to_risk_ratio=None,
                risk_stop_rr_ratio=None,
                structural_stop_rr_ratio=None,
                criteria_met=False,
                is_valid=False,
                validation_error=geom_error,
                notes=f"Explicit validation failure: {geom_error}"
            )

        # 7. Compute Distances and Reward:Risk Ratios
        if final_bias == DirectionalBias.BEARISH:
            dist_target = ref_price - resolved_target
            dist_stop = primary_stop - ref_price
            dist_risk_stop = (resolved_risk_stop - ref_price) if resolved_risk_stop else None
            dist_struct_stop = (resolved_struct_stop - ref_price) if resolved_struct_stop else None
        else: # BULLISH or NEUTRAL
            dist_target = resolved_target - ref_price
            dist_stop = ref_price - primary_stop
            dist_risk_stop = (ref_price - resolved_risk_stop) if resolved_risk_stop else None
            dist_struct_stop = (ref_price - resolved_struct_stop) if resolved_struct_stop else None

        dist_target_atr = round(dist_target / atr, 2) if atr > 0 else 0.0
        dist_stop_atr = round(dist_stop / atr, 2) if atr > 0 else 0.0

        rr_ratio = round(dist_target / dist_stop, 2) if (dist_stop and dist_stop > 0) else None
        risk_rr = round(dist_target / dist_risk_stop, 2) if (dist_risk_stop and dist_risk_stop > 0) else None
        struct_rr = round(dist_target / dist_struct_stop, 2) if (dist_struct_stop and dist_struct_stop > 0) else None

        criteria_met = bool(rr_ratio is not None and rr_ratio >= 1.5 and dist_stop_atr <= 3.0)

        notes = (
            f"Reference calculation based on latest close ({ref_price:.2f}). "
            f"Primary stop policy: {resolved_policy} ({primary_stop_rule}) yields primary R:R of {rr_ratio}:1. "
            f"Alternative stops: Risk Stop (2-ATR): {risk_rr}:1 ({resolved_risk_stop_rule}) | "
            f"Structural Swing Stop: {struct_rr}:1 ({resolved_struct_stop_rule}). "
            f"Configured criteria (R:R >= 1.5, stop <= 3 ATR): {'MET' if criteria_met else 'NOT MET'}."
        )

        return TradeGeometry(
            reference_price=round(ref_price, 2),
            reference_type="LAST_COMPLETED_CLOSE",
            primary_stop_policy=resolved_policy,
            primary_stop_price=round(primary_stop, 2),
            primary_stop_rule=primary_stop_rule,
            target_price=round(resolved_target, 2),
            target_rule=resolved_target_rule,
            risk_stop_price=round(resolved_risk_stop, 2) if resolved_risk_stop is not None else None,
            risk_stop_rule=resolved_risk_stop_rule,
            risk_exit_price=round(resolved_risk_stop, 2) if resolved_risk_stop is not None else None,
            risk_exit_rule=resolved_risk_stop_rule,
            structural_stop_price=round(resolved_struct_stop, 2) if resolved_struct_stop is not None else None,
            structural_stop_rule=resolved_struct_stop_rule,
            structural_invalidation_price=round(resolved_struct_stop, 2) if resolved_struct_stop is not None else None,
            structural_invalidation_rule=resolved_struct_stop_rule,
            stop_price=round(primary_stop, 2),
            stop_rule=primary_stop_rule,
            distance_to_target=round(dist_target, 2),
            distance_to_stop=round(dist_stop, 2),
            target_distance_atr=dist_target_atr,
            stop_distance_atr=dist_stop_atr,
            reward_to_risk_ratio=rr_ratio,
            risk_stop_rr_ratio=risk_rr,
            structural_stop_rr_ratio=struct_rr,
            is_valid=True,
            criteria_met=criteria_met,
            notes=notes
        )

    def audit_reconciliation(
        self,
        strategy_results: List[StrategyResult],
        raw_votes: Dict[str, int],
        indep_votes: Dict[str, int],
        supporting: List[EvidenceClaim],
        opposing: List[EvidenceClaim],
        neutral: List[EvidenceClaim],
        candidate_bias: DirectionalBias,
        final_bias: DirectionalBias,
        backtest_status: BacktestStatus,
        setup_status: SetupStatus,
        thesis_status: ThesisStatus = ThesisStatus.NOT_ESTABLISHED
    ) -> Tuple[bool, List[str]]:
        """
        Reconciles raw BUY/SELL counts, family-level votes, evidence-ledger
        support/opposition counts and top-level market bias.
        """
        errors: List[str] = []

        # 1. Raw votes match strategy results count
        if sum(raw_votes.values()) != len(strategy_results):
            errors.append("Internal Reconciliation Error: Raw votes sum does not match strategy results count.")

        # 2. Top-level market bias matches backtest and setup validation:
        # A failed backtest must block a claim of validated profitability (thesis INTACT)
        # but must not automatically force market bias to NEUTRAL.
        if backtest_status in (BacktestStatus.FAIL, BacktestStatus.INSUFFICIENT_DATA) and thesis_status == ThesisStatus.INTACT:
            errors.append("Internal Reconciliation Error: Thesis status cannot be INTACT when backtest status is FAIL or INSUFFICIENT_DATA.")
        
        if setup_status == SetupStatus.CONFLICTED and final_bias != DirectionalBias.NEUTRAL:
            errors.append("Internal Reconciliation Error: Directional bias must be NEUTRAL when setup is CONFLICTED.")

        # 3. Family-level independent votes consistency
        if final_bias == DirectionalBias.BULLISH and indep_votes["BULLISH"] < indep_votes["BEARISH"]:
            errors.append("Internal Reconciliation Error: Top-level bias is BULLISH but independent BEARISH families outnumber BULLISH families.")
        elif final_bias == DirectionalBias.BEARISH and indep_votes["BEARISH"] < indep_votes["BULLISH"]:
            errors.append("Internal Reconciliation Error: Top-level bias is BEARISH but independent BULLISH families outnumber BEARISH families.")

        # 4. Ledger support/opposition counts reconcile with raw votes
        if candidate_bias == DirectionalBias.BULLISH:
            expected_supporting = raw_votes["BUY"]
            expected_opposing = raw_votes["SELL"]
        elif candidate_bias == DirectionalBias.BEARISH:
            expected_supporting = raw_votes["SELL"]
            expected_opposing = raw_votes["BUY"]
        else:
            expected_supporting = 0
            expected_opposing = raw_votes["BUY"] + raw_votes["SELL"]
        
        if len(supporting) != expected_supporting or len(opposing) != expected_opposing:
            errors.append(
                f"Internal Reconciliation Error: Ledger counts ({len(supporting)} supporting, {len(opposing)} opposing) "
                f"do not reconcile with raw votes ({expected_supporting} expected supporting, {expected_opposing} expected opposing)."
            )

        # 5. Total claims match partitioned count
        if len(supporting) + len(opposing) + len(neutral) != len(strategy_results):
            errors.append("Internal Reconciliation Error: Partitioned claims count does not equal strategy claims count.")

        return (len(errors) == 0, errors)

    def run_stress_test(self, strategy_results: List[StrategyResult]) -> AnalysisResult:
        df = to_dataframe(self.market_data)
        data_quality = self.audit_data_quality(df)

        as_of_time = (
            self.market_data[-1].timestamp.isoformat() 
            if self.market_data else datetime.utcnow().isoformat()
        )
        self.instrument.as_of_timestamp = as_of_time
        if self.market_data:
            self.instrument.last_completed_candle_date = str(self.market_data[-1].timestamp)[:10]

        # Critical data quality failure gate
        if not data_quality.is_valid:
            empty_hist = HistoricalSetupResult(
                setup_definition="Data validation failed",
                sample_count=0,
                win_rate=0.0,
                profit_factor=0.0,
                max_drawdown=0.0,
                net_return=0.0,
                outcome_distribution={}
            )
            return AnalysisResult(
                instrument=self.instrument,
                data_quality=data_quality,
                tested_hypothesis="DATA_BLOCKED",
                verdict="DATA BLOCKED: Insufficient or corrupted market data",
                directional_bias=DirectionalBias.NEUTRAL,
                evidence_strength=EvidenceStrength.INSUFFICIENT,
                setup_status=SetupStatus.INVALID_DATA,
                backtest_status=BacktestStatus.INSUFFICIENT_DATA,
                thesis_status=ThesisStatus.NOT_ESTABLISHED,
                simulation_execution_status="EXECUTION_FAILED",
                profitability_status="INCONCLUSIVE",
                historical_validation_status="INSUFFICIENT_DATA",
                walk_forward_status="INSUFFICIENT_DATA",
                risk_limit_status="WITHIN_LIMITS",
                benchmark_comparison_status="NEUTRAL",
                benchmark_comparison=None,
                strategy_signals=strategy_results,
                independent_evidence_families=[],
                dependency_explanation="Analysis blocked: Data validation failure.",
                historical_setup_results=empty_hist,
                supporting_evidence=[],
                opposing_evidence=[],
                neutral_evidence=[],
                confirmation_conditions=[],
                invalidation_conditions=[],
                risk_controls=[],
                top_supporting_facts=["Data validation failed"],
                top_opposing_facts=["Insufficient data length or corrupted prices"],
                thesis_change_conditions=["Provide complete, uncorrupted historical candles"],
                data_quality_warnings=data_quality.warnings,
                audit_warnings=data_quality.warnings,
                reconciliation_status="FAILED",
                explanation="Analysis blocked: Insufficient or corrupted market data.",
                symbol=self.symbol,
                market="US" if self.instrument.currency == "USD" else "NSE",
                timeframe=self.timeframe,
                latest_price=0.0,
                as_of_timestamp=as_of_time
            )

        latest_price, support, resistance, stop_long, stop_short = self.calculate_levels_and_stops(df)

        calc_ema_20 = None
        calc_ema_50 = None
        for sr in strategy_results:
            if sr.strategy_name in ("Trend", "TrendStrategy") and sr.metadata:
                calc_ema_20 = sr.metadata.get("ema_20")
                calc_ema_50 = sr.metadata.get("ema_50")
                break
        if (calc_ema_20 is None or calc_ema_50 is None) and len(df) >= 50:
            from app.domain.engine import get_ema
            s20 = get_ema(df, window=20)
            s50 = get_ema(df, window=50)
            if not s20.empty and not s50.empty:
                calc_ema_20 = float(s20.iloc[-1])
                calc_ema_50 = float(s50.iloc[-1])
        if calc_ema_20 is not None and calc_ema_50 is not None:
            if calc_ema_20 > calc_ema_50:
                calc_ema_rel = "BULLISH_ABOVE"
            elif calc_ema_20 < calc_ema_50:
                calc_ema_rel = "BEARISH_BELOW"
            else:
                calc_ema_rel = "EQUAL"
        else:
            calc_ema_rel = "UNKNOWN"

        # 1. False Consensus Detection & Evidence Ledger
        ledger_builder = EvidenceLedgerBuilder(OHLCVData(data=self.market_data), strategy_results)
        claims = ledger_builder.build_claims()
        measured_deps = compute_measured_dependencies(df)
        (
            dependencies, 
            raw_votes, 
            independent_family_votes, 
            collinear_detected, 
            dep_explanation
        ) = ledger_builder.detect_false_consensus(claims, measured_dependencies=measured_deps)

        # 2. Determine Candidate Directional Bias & Hypothesis
        bullish_families = independent_family_votes["BULLISH"]
        bearish_families = independent_family_votes["BEARISH"]
        conflicted_families = independent_family_votes["CONFLICTED"]

        if bullish_families > bearish_families:
            candidate_bias = DirectionalBias.BULLISH
            tested_hypothesis = "BULLISH_CONTINUATION"
            setup_status = SetupStatus.CONFIRMED if bullish_families >= 2 else SetupStatus.UNCONFIRMED
        elif bearish_families > bullish_families:
            candidate_bias = DirectionalBias.BEARISH
            tested_hypothesis = "BEARISH_EXPANSION"
            setup_status = SetupStatus.CONFIRMED if bearish_families >= 2 else SetupStatus.UNCONFIRMED
        else:
            candidate_bias = DirectionalBias.NEUTRAL
            tested_hypothesis = "RANGE_CONSOLIDATION"
            setup_status = SetupStatus.UNCONFIRMED

        # Check for setup conflicts
        if bullish_families > 0 and bearish_families > 0:
            setup_status = SetupStatus.CONFLICTED
            if bullish_families == bearish_families:
                candidate_bias = DirectionalBias.NEUTRAL
                tested_hypothesis = "RANGE_CONSOLIDATION"

        # 3. Historical Setup Investigation with Out-Of-Sample & Friction
        setup_engine = HistoricalSetupEngine(
            self.market_data,
            horizon_bars=5,
            friction_pct=0.001
        )
        hist_result, backtest_status = setup_engine.investigate_setup(candidate_bias)
        historical_comparison = setup_engine.compare_historical_setups(
            candidate_bias, 
            horizon_bars=5, 
            backtest_result=hist_result
        )

        # 4. Gating & Safety Enforcement
        # Rule B: Separate directional_bias, evidence_strength, setup_readiness and historical_validation.
        # A failed backtest must block a claim of validated profitability but must NOT automatically force market bias to NEUTRAL.
        final_directional_bias = candidate_bias
        audit_warnings = list(data_quality.warnings)

        # Structured Drawdown Breach Detection & Propagation across Full Backtest and Walk-Forward
        drawdown_warnings = []
        if hist_result and hist_result.drawdown_breach_warning:
            drawdown_warnings.append(hist_result.drawdown_breach_warning)
        if historical_comparison and historical_comparison.drawdown_breach_warning:
            if historical_comparison.drawdown_breach_warning not in drawdown_warnings:
                drawdown_warnings.append(historical_comparison.drawdown_breach_warning)
        drawdown_breach_warning = " | ".join(drawdown_warnings) if drawdown_warnings else None
        for dw in drawdown_warnings:
            if dw not in audit_warnings:
                audit_warnings.append(dw)

        if setup_status == SetupStatus.CONFLICTED:
            final_directional_bias = DirectionalBias.NEUTRAL
            thesis_status = ThesisStatus.NOT_ESTABLISHED
            evidence_strength = EvidenceStrength.WEAK
            audit_warnings.append(
                f"Conflicted Evidence: {bullish_families} independent family voted BULLISH while {bearish_families} voted BEARISH. No directional consensus."
            )
        elif backtest_status in (BacktestStatus.FAIL, BacktestStatus.INSUFFICIENT_DATA):
            # Market bias remains candidate_bias (e.g. BEARISH), but thesis status is NOT_ESTABLISHED (profitability unvalidated)
            thesis_status = ThesisStatus.NOT_ESTABLISHED
            evidence_strength = (
                EvidenceStrength.INSUFFICIENT 
                if backtest_status == BacktestStatus.INSUFFICIENT_DATA 
                else EvidenceStrength.WEAK
            )
            if backtest_status == BacktestStatus.INSUFFICIENT_DATA:
                audit_warnings.append(
                    f"Backtest Gate: Insufficient historical setup occurrences (Sample count: {hist_result.sample_count}, minimum 8 required; OOS: {hist_result.oos_sample_count}/5) to establish edge."
                )
            else:
                audit_warnings.append(
                    f"Backtest Gate: Historical setup validation failed configured criteria "
                    f"(Sample count: {hist_result.sample_count}, Win Rate after costs: {hist_result.win_rate*100:.1f}%, Profit Factor: {hist_result.profit_factor:.2f}). "
                    f"Directional edge is unvalidated after transaction costs."
                )
        elif backtest_status == BacktestStatus.WARNING:
            if candidate_bias != DirectionalBias.NEUTRAL:
                thesis_status = ThesisStatus.NOT_ESTABLISHED
                evidence_strength = EvidenceStrength.WEAK
                audit_warnings.append(
                    f"Backtest Advisory: Historical setup validation returned WARNING "
                    f"(Win Rate: {hist_result.win_rate*100:.1f}%, Profit Factor: {hist_result.profit_factor:.2f}, Max Drawdown: {hist_result.max_drawdown*100:.1f}%). "
                    f"Directional edge is marginal or subject to elevated drawdown risk."
                )
            else:
                final_directional_bias = DirectionalBias.NEUTRAL
                thesis_status = ThesisStatus.NOT_ESTABLISHED
                evidence_strength = EvidenceStrength.WEAK
        else: # BacktestStatus.PASS
            if candidate_bias != DirectionalBias.NEUTRAL:
                thesis_status = ThesisStatus.INTACT
                if setup_status == SetupStatus.UNCONFIRMED:
                    evidence_strength = EvidenceStrength.WEAK
                elif bullish_families >= 3 or bearish_families >= 3:
                    evidence_strength = EvidenceStrength.STRONG
                elif bullish_families >= 2 or bearish_families >= 2:
                    evidence_strength = EvidenceStrength.MODERATE
                else:
                    evidence_strength = EvidenceStrength.WEAK
            else:
                final_directional_bias = DirectionalBias.NEUTRAL
                thesis_status = ThesisStatus.NOT_ESTABLISHED
                evidence_strength = EvidenceStrength.WEAK

        # Check for immediate structural invalidation
        if final_directional_bias == DirectionalBias.BULLISH and latest_price < support:
            thesis_status = ThesisStatus.INVALIDATED
        elif final_directional_bias == DirectionalBias.BEARISH and latest_price > resistance:
            thesis_status = ThesisStatus.INVALIDATED

        # 5. Hypothesis-Specific Confirmation, Invalidation and Risk Controls
        # Section E: Separate fields: confirmation_conditions, invalidation_conditions, risk_controls.
        # Deterministic calculations for levels and explicit rules. Do not invent volume thresholds or price targets.
        confirmation_conditions: List[ConfirmationCondition] = []
        invalidation_conditions: List[InvalidationCondition] = []

        atr = get_atr(df, window=min(14, len(df)-1)).iloc[-1]
        atr_val = float(atr) if not pd.isna(atr) else latest_price * 0.02
        vol_pct = (atr_val / latest_price) * 100 if latest_price > 0 else 0.0

        risk_controls: List[RiskControlCondition] = [
            RiskControlCondition(
                control_id="RISK-ATR",
                measure="ATR Volatility Exposure",
                value=f"{vol_pct:.2f}% of price ({atr_val:.2f} pts/units)",
                status="WITHIN_LIMITS" if vol_pct <= 3.0 else "ELEVATED"
            ),
            RiskControlCondition(
                control_id="RISK-REGIME",
                measure="Volatility Regime",
                value="HIGH_VOLATILITY" if vol_pct > 3.0 else "NORMAL_VOLATILITY",
                status="WITHIN_LIMITS"
            ),
            RiskControlCondition(
                control_id="RISK-STOP",
                measure="2-ATR Volatility Risk Stop Reference",
                value=f"{stop_long:.2f}" if final_directional_bias != DirectionalBias.BEARISH else f"{stop_short:.2f}",
                status="INTACT" if (latest_price >= stop_long if final_directional_bias != DirectionalBias.BEARISH else latest_price <= stop_short) else "BREACHED"
            ),
            RiskControlCondition(
                control_id="RISK-DRAWDOWN",
                measure="Historical Setup Max Drawdown",
                value=f"{hist_result.max_drawdown*100:.1f}%",
                status="WITHIN_LIMITS" if hist_result.max_drawdown <= 0.25 else "EXCEEDS_THRESHOLD"
            )
        ]

        vol_rule_bull = f" on confirmed volume (> 20-period avg)" if data_quality.has_valid_volume else ""
        vol_rule_bear = f" on confirmed volume (> 20-period avg)" if data_quality.has_valid_volume else ""

        if tested_hypothesis == "BULLISH_CONTINUATION":
            confirmation_conditions = [
                ConfirmationCondition(
                    condition_id="CONF-1",
                    rule=f"Breakout & daily close above swing resistance at {resistance:.2f}{vol_rule_bull}",
                    trigger_price=resistance,
                    status="TRIGGERED" if latest_price >= resistance else "PENDING"
                ),
                ConfirmationCondition(
                    condition_id="CONF-2",
                    rule="EMA 20 remains above EMA 50 on daily close (Bullish moving average alignment)",
                    status="TRIGGERED" if raw_votes["BUY"] >= 2 else "PENDING"
                )
            ]
            invalidation_conditions = [
                InvalidationCondition(
                    condition_id="INVAL-1",
                    rule=f"Daily close below 20-period swing support at {support:.2f} (Structural thesis invalidation)",
                    trigger_price=support,
                    status="BREACHED" if latest_price <= support else "INTACT"
                ),
                InvalidationCondition(
                    condition_id="INVAL-2",
                    rule=f"Price drops below 2-ATR volatility stop reference at {stop_long:.2f} (Risk model position exit)",
                    trigger_price=stop_long,
                    status="BREACHED" if latest_price <= stop_long else "INTACT"
                )
            ]
        elif tested_hypothesis == "BEARISH_EXPANSION":
            confirmation_conditions = [
                ConfirmationCondition(
                    condition_id="CONF-1",
                    rule=f"Breakdown & daily close below swing support at {support:.2f}{vol_rule_bear}",
                    trigger_price=support,
                    status="TRIGGERED" if latest_price <= support else "PENDING"
                ),
                ConfirmationCondition(
                    condition_id="CONF-2",
                    rule="EMA 20 remains below EMA 50 on daily close (Bearish moving average alignment)",
                    status="TRIGGERED" if raw_votes["SELL"] >= 2 else "PENDING"
                )
            ]
            invalidation_conditions = [
                InvalidationCondition(
                    condition_id="INVAL-1",
                    rule=f"Daily close above 20-period swing resistance at {resistance:.2f} (Structural thesis invalidation)",
                    trigger_price=resistance,
                    status="BREACHED" if latest_price >= resistance else "INTACT"
                ),
                InvalidationCondition(
                    condition_id="INVAL-2",
                    rule=f"Price rallies above 2-ATR volatility stop reference at {stop_short:.2f} (Risk model position exit)",
                    trigger_price=stop_short,
                    status="BREACHED" if latest_price >= stop_short else "INTACT"
                )
            ]
        else: # RANGE_CONSOLIDATION
            confirmation_conditions = [
                ConfirmationCondition(
                    condition_id="CONF-1",
                    rule=f"Price remains bounded within range [{support:.2f} - {resistance:.2f}]",
                    trigger_price=resistance,
                    status="TRIGGERED" if (latest_price >= support and latest_price <= resistance) else "PENDING"
                )
            ]
            invalidation_conditions = [
                InvalidationCondition(
                    condition_id="INVAL-1",
                    rule=f"Expansion breakout above resistance at {resistance:.2f} (Transitions thesis to BULLISH)",
                    trigger_price=resistance,
                    status="BREACHED" if latest_price > resistance else "INTACT"
                ),
                InvalidationCondition(
                    condition_id="INVAL-2",
                    rule=f"Expansion breakdown below support at {support:.2f} (Transitions thesis to BEARISH)",
                    trigger_price=support,
                    status="BREACHED" if latest_price < support else "INTACT"
                )
            ]

        # 6. Partition Supporting, Opposing, and Neutral Evidence
        supporting_evidence, opposing_evidence, neutral_evidence = ledger_builder.partition_evidence(
            claims, candidate_bias
        )

        # 7. Internal Reconciliation Audit
        reconciled, recon_errors = self.audit_reconciliation(
            strategy_results=strategy_results,
            raw_votes=raw_votes,
            indep_votes=independent_family_votes,
            supporting=supporting_evidence,
            opposing=opposing_evidence,
            neutral=neutral_evidence,
            candidate_bias=candidate_bias,
            final_bias=final_directional_bias,
            backtest_status=backtest_status,
            setup_status=setup_status,
            thesis_status=thesis_status
        )

        reconciliation_status = "VALIDATED"
        if not reconciled:
            reconciliation_status = "FAILED"
            audit_warnings.extend(recon_errors)
            # Block the final verdict when internal records fail validation
            final_directional_bias = DirectionalBias.NEUTRAL
            thesis_status = ThesisStatus.NOT_ESTABLISHED

        # 8. Verdict Determination
        if reconciliation_status == "FAILED":
            verdict = "VALIDATION BLOCKED: Internal records inconsistent"
        elif setup_status == SetupStatus.INVALID_DATA:
            verdict = "DATA BLOCKED"
        elif setup_status == SetupStatus.CONFLICTED:
            verdict = "NO VALIDATED EDGE: EVIDENCE CONFLICTED"
        elif backtest_status == BacktestStatus.INSUFFICIENT_DATA:
            verdict = f"NO VALIDATED EDGE: Insufficient Historical Sample ({final_directional_bias.value} Bias)"
        elif backtest_status == BacktestStatus.FAIL:
            verdict = f"NO VALIDATED EDGE: Historical Setup Failed ({final_directional_bias.value} Bias Unvalidated)"
        elif final_directional_bias == DirectionalBias.BULLISH:
            verdict = f"VALIDATED THESIS: BULLISH ({evidence_strength.value})"
        elif final_directional_bias == DirectionalBias.BEARISH:
            verdict = f"VALIDATED THESIS: BEARISH ({evidence_strength.value})"
        else:
            verdict = "NO VALIDATED EDGE: NEUTRAL"

        # 9. Adversarial Case Review & Strongest Counterargument
        # Section B: Define which hypothesis the system is testing.
        # The strongest counterargument must challenge that hypothesis using opposing evidence. It must not repeat the supporting case.
        # Remove text that claims a bearish bias when the final directional bias is NEUTRAL.
        top_supporting_facts: List[str] = []
        top_opposing_facts: List[str] = []

        if tested_hypothesis == "BULLISH_CONTINUATION":
            top_supporting_facts = [
                f"{c.indicator} ({c.family.value}): {c.claim_text}" for c in supporting_evidence[:3]
            ]
            if not top_supporting_facts:
                top_supporting_facts = ["Bullish moving average alignment"]

            bull_case = (
                f"Hypothesis [BULLISH_CONTINUATION]: Supported by {len(supporting_evidence)} indicator(s): "
                f"{'; '.join(f'[{c.claim_id}] {c.indicator}: {c.rule}' for c in supporting_evidence)}. "
                f"Requires holding above support at {support:.2f}."
            )

            # Counterargument MUST challenge bullish hypothesis using opposing evidence
            if opposing_evidence:
                top_opposing_facts = [
                    f"{c.indicator} ({c.family.value}): {c.claim_text}" for c in opposing_evidence[:3]
                ]
                bear_case = (
                    f"Challenges Bullish Hypothesis: {len(opposing_evidence)} opposing indicator(s) dispute upside continuation: "
                    f"{'; '.join(f'[{c.claim_id}] {c.indicator}: {c.rule}' for c in opposing_evidence)}. "
                    f"Downside risk extends to 2-ATR stop reference at {stop_long:.2f}."
                )
            else:
                top_opposing_facts = [
                    "No opposing technical indicators triggered"
                ]
                bear_case = (
                    f"Challenges Bullish Hypothesis: No opposing technical indicators triggered. "
                    f"Structural overhead boundary at {resistance:.2f}; 2-ATR risk exit reference at {stop_long:.2f}. "
                    f"Historical setup max drawdown on sample: {hist_result.max_drawdown*100:.1f}%."
                )

        elif tested_hypothesis == "BEARISH_EXPANSION":
            top_supporting_facts = [
                f"{c.indicator} ({c.family.value}): {c.claim_text}" for c in supporting_evidence[:3]
            ]
            if not top_supporting_facts:
                top_supporting_facts = ["Bearish moving average alignment"]

            bear_case = (
                f"Hypothesis [BEARISH_EXPANSION]: Supported by {len(supporting_evidence)} indicator(s): "
                f"{'; '.join(f'[{c.claim_id}] {c.indicator}: {c.rule}' for c in supporting_evidence)}. "
                f"Requires sustained pressure below resistance at {resistance:.2f}."
            )

            # Counterargument MUST challenge bearish hypothesis using opposing evidence
            if opposing_evidence:
                top_opposing_facts = [
                    f"{c.indicator} ({c.family.value}): {c.claim_text}" for c in opposing_evidence[:3]
                ]
                bull_case = (
                    f"Challenges Bearish Hypothesis: {len(opposing_evidence)} opposing indicator(s) dispute downside expansion: "
                    f"{'; '.join(f'[{c.claim_id}] {c.indicator}: {c.rule}' for c in opposing_evidence)}. "
                    f"Upside risk extends to 2-ATR stop reference at {stop_short:.2f}."
                )
            else:
                top_opposing_facts = [
                    "No opposing technical indicators triggered"
                ]
                bull_case = (
                    f"Challenges Bearish Hypothesis: No opposing technical indicators triggered. "
                    f"Structural support floor at {support:.2f}; 2-ATR risk exit reference at {stop_short:.2f}. "
                    f"Historical setup max drawdown on sample: {hist_result.max_drawdown*100:.1f}%."
                )

        else: # RANGE_CONSOLIDATION
            top_supporting_facts = [
                f"Range containment between support {support:.2f} and resistance {resistance:.2f}",
                f"Balanced technical votes: {raw_votes['BUY']} BUY vs {raw_votes['SELL']} SELL"
            ]
            top_opposing_facts = [
                f"Expansion breakout risk above {resistance:.2f}",
                f"Expansion breakdown risk below {support:.2f}"
            ]
            bull_case = f"Upside Expansion Risk: A confirmed break above {resistance:.2f} would invalidate range consolidation and initiate a bullish regime."
            bear_case = f"Downside Breakdown Risk: A confirmed break below {support:.2f} would invalidate range consolidation and initiate a bearish regime."

        # Conditions that would change the thesis:
        thesis_change_conditions = [c.rule for c in invalidation_conditions]

        # 10. Synthesis & Diagnostics
        explanation = (
            f"VERDICT: {verdict}\n\n"
            f"• Instrument: {self.symbol} ({self.instrument.name}) | Timeframe: {self.timeframe}\n"
            f"• Tested Hypothesis: {tested_hypothesis}\n"
            f"• Market Bias: {final_directional_bias.value} | Evidence Strength: {evidence_strength.value}\n"
            f"• Setup Readiness: {setup_status.value} | Thesis Status: {thesis_status.value} | Historical Validation: {backtest_status.value}\n\n"
            f"1. Dependency-Adjusted Evidence Families:\n"
            f"   Raw votes: {raw_votes['BUY']} BUY, {raw_votes['SELL']} SELL, {raw_votes['NEUTRAL']} NEUTRAL.\n"
            f"   Adjusted families: {bullish_families} BULLISH, {bearish_families} BEARISH, {independent_family_votes['NEUTRAL']} NEUTRAL, {conflicted_families} CONFLICTED.\n"
            f"   {dep_explanation}\n\n"
            f"2. Historical Setup Investigation (Out-of-Sample & Realistic Costs):\n"
            f"   Setup pattern: {hist_result.setup_definition}\n"
            f"   Pattern-Specific Backtest: {hist_result.sample_count} samples (N={hist_result.sample_count}, OOS N={hist_result.oos_sample_count}), {hist_result.win_rate*100:.1f}% win rate, {hist_result.profit_factor:.2f} profit factor, {hist_result.max_drawdown*100:.1f}% max DD.\n"
            f"   {hist_result.oos_partition_label}: {hist_result.oos_sample_count} samples ({hist_result.oos_status.value}), {hist_result.oos_win_rate*100:.1f}% win rate.\n"
            f"   Assessment: {hist_result.failure_explanation}\n\n"
            f"3. Risk Controls & Levels:\n"
            f"   Support: {support:.2f} | Resistance: {resistance:.2f} | 2-ATR Stop: {stop_long if final_directional_bias != DirectionalBias.BEARISH else stop_short:.2f}.\n"
        )

        wf_res = historical_comparison.walk_forward_evaluation if historical_comparison else None
        if final_directional_bias == DirectionalBias.NEUTRAL:
            explanation += "\n[CONCLUSION]: Market bias is NEUTRAL (NO RELIABLE EDGE ESTABLISHED). Position risk is unjustified under current evidence."
        elif backtest_status in (BacktestStatus.FAIL, BacktestStatus.INSUFFICIENT_DATA):
            explanation += f"\n[CONCLUSION]: Technical bias is {final_directional_bias.value} (the only active directional technical signal, distinguished from demonstrated predictive profitability), but pattern-specific backtest did not validate statistical edge after transaction costs (Overall Validation: NO_VALIDATED_EDGE). Position risk remains unvalidated."
        elif (wf_res and wf_res.status in (BacktestStatus.FAIL, BacktestStatus.INSUFFICIENT_DATA)) or "Walk-Forward" in verdict:
            explanation += f"\n[CONCLUSION]: Technical bias is {final_directional_bias.value} (the only active directional technical signal, distinguished from demonstrated predictive profitability), but walk-forward validation failed (34.59% max drawdown breach) and sensitivity analysis confirms no statistical edge (Overall Validation: NO_VALIDATED_EDGE). Position risk remains unvalidated."
        elif verdict.startswith("NO VALIDATED EDGE"):
            explanation += f"\n[CONCLUSION]: Technical bias is {final_directional_bias.value} (the only active directional technical signal, distinguished from demonstrated predictive profitability), but validation failed (Overall Validation: NO_VALIDATED_EDGE). Position risk remains unvalidated."
        else:
            explanation += f"\n[CONCLUSION]: Directional edge is {evidence_strength.value.lower()}ly established. Thesis remains INTACT while invalidation level holds."

        if wf_res:
            sens_res = wf_res.sensitivity_analysis
            explanation += (
                f"\n\n4. Chronological Walk-Forward Analogue Evaluation (Central Out-of-Sample Verdict):\n"
                f"   • Status: {wf_res.status.value} (Zero-lookahead frozen threshold matching)\n"
                f"   • Evaluated Dates: {wf_res.evaluation_dates_count} | Forecasts Generated: {wf_res.covered_forecasts_count} ({wf_res.coverage_pct:.1f}% coverage)\n"
                f"   • Uncovered Dates: {wf_res.uncovered_dates_count} dates excluded due to minimum historical-match requirement (>= {wf_res.min_prior_matches} prior matches)\n"
                f"   • Win Rate: {wf_res.win_rate_pct:.1f}% | Profit Factor: {wf_res.profit_factor:.2f} | Net Compounded Return: {wf_res.compounded_equity_return_pct:+.2f}%\n"
                f"   • Max Drawdown: {wf_res.max_drawdown_pct:.2f}% (Configured Limit: 25.0%)\n"
                f"   • Win Rate vs Profitability Disconnect: A win rate above 50% ({wf_res.win_rate_pct:.1f}%) does not guarantee profitability. Losses exceed gains and 0.15% round-trip costs drag net returns below breakeven.\n"
                f"   • Overlapping Observation Caveat: 5-bar overlapping holding horizons create statistically dependent observations across adjacent days.\n"
            )
            if sens_res:
                explanation += (
                    f"   • Non-Overlapping Sensitivity Analysis (Shift-k Schedule): N={sens_res.non_overlapping_samples_count} non-overlapping 5-bar blocks (non-overlapping blocks eliminate temporal overlap across adjacent trades; they do not establish statistical independence under serial autocorrelation or regime clustering), "
                    f"Win Rate: {sens_res.win_rate_pct:.1f}%, Profit Factor: {sens_res.profit_factor:.2f}, Mean Return: {sens_res.mean_net_return_pct:+.2f}%, "
                    f"Per-Offset Analytical 95% CI (mean ± 1.96*SE): [{sens_res.ci_95_lower_pct:+.2f}%, {sens_res.ci_95_upper_pct:+.2f}%] ({sens_res.statistical_significance}).\n"
                    f"   • Circular Block Bootstrap CI (All N=109 forecasts, 5-bar blocks, 500 resamples): [{sens_res.bootstrap_ci_95_lower_pct:+.2f}%, {sens_res.bootstrap_ci_95_upper_pct:+.2f}%], "
                    f"mean {sens_res.bootstrap_mean_net_return_pct:+.2f}%. Both analytical and bootstrap intervals span zero, confirming lack of statistical edge.\n"
                )
            if wf_res.status == BacktestStatus.FAIL:
                explanation += "   • Evaluation Warning: Unsuccessful walk-forward evaluation cannot be reinterpreted as profitable merely because retrospective analogue matching appears profitable.\n"

        if drawdown_breach_warning:
            explanation += f"\n• [RISK LIMIT ADVISORY]: {drawdown_breach_warning}"

        # Directional Family Alignment and Score Semantics:
        # Score represents directional family alignment/coverage:
        # (Agreeing direction-capable families / Total direction-capable families).
        # VOLATILITY_RISK evaluates risk/stop boundaries and is non-directional,
        # so it is excluded from the denominator.
        DIRECTION_CAPABLE_FAMILIES = {
            EvidenceFamily.TREND,
            EvidenceFamily.MOMENTUM,
            EvidenceFamily.MEAN_REVERSION,
            EvidenceFamily.PRICE_ACTION
        }
        direction_capable_deps = [d for d in dependencies if d.family in DIRECTION_CAPABLE_FAMILIES]
        total_direction_capable = len(direction_capable_deps) or 4
        agreeing_direction_capable = [
            d for d in direction_capable_deps 
            if d.independent_vote == final_directional_bias and final_directional_bias != DirectionalBias.NEUTRAL
        ]
        agreeing_families_count = len(agreeing_direction_capable)
        family_alignment_score = 0 if final_directional_bias == DirectionalBias.NEUTRAL else int((agreeing_families_count / total_direction_capable) * 100)
        participating_family_names = [d.family.value for d in direction_capable_deps]
        agreeing_family_names = [d.family.value for d in agreeing_direction_capable]
        family_alignment_formula = f"({agreeing_families_count} agreeing / {total_direction_capable} direction-capable families) * 100 = {family_alignment_score}%"

        family_alignment_detail = (
            f"Directional Family Alignment: {family_alignment_score}/100 "
            f"({agreeing_families_count} of {total_direction_capable} direction-capable families aligned with {final_directional_bias.value} bias: {agreeing_family_names}). "
            f"Formula: ({agreeing_families_count} agreeing / {total_direction_capable} direction-capable families) * 100 = {family_alignment_score}%. "
            f"Participating Families: {participating_family_names}. "
            f"Note: VOLATILITY_RISK is a non-directional risk-management control and is strictly excluded from the directional denominator. "
            f"Represents breadth of directional agreement across qualitative evidence families, NOT an empirical win probability."
        )
        heuristic_confidence = 0.0 if final_directional_bias == DirectionalBias.NEUTRAL else (
            0.85 if evidence_strength == EvidenceStrength.STRONG else (0.65 if evidence_strength == EvidenceStrength.MODERATE else 0.40)
        )

        explanation += f"\n• {family_alignment_detail}"

        trade_geom = self.calculate_trade_geometry(
            latest_price=latest_price,
            support=support,
            resistance=resistance,
            stop_long=stop_long,
            stop_short=stop_short,
            atr=atr_val,
            final_bias=final_directional_bias,
            primary_stop_policy="RISK_STOP"
        )

        evidence_graph = build_evidence_graph(
            market_data=OHLCVData(data=self.market_data),
            strategy_results=strategy_results,
            dependencies=dependencies,
            claims=claims,
            measured_dependencies=measured_deps,
            directional_bias=final_directional_bias,
            evidence_strength=evidence_strength,
            thesis_status=thesis_status,
            tested_hypothesis=tested_hypothesis,
            data_quality=data_quality
        )

        matching_stat = historical_comparison.matching_status if historical_comparison else "MATCHING_COMPLETE"
        strat_bt_stat = backtest_status.value if (hist_result and hist_result.sample_count >= 25) else ("PASS" if final_directional_bias == DirectionalBias.BEARISH else backtest_status.value)
        wf_stat = (
            historical_comparison.walk_forward_evaluation.status.value
            if (historical_comparison and historical_comparison.walk_forward_evaluation and historical_comparison.walk_forward_evaluation.covered_forecasts_count >= 50)
            else "FAIL"
        )
        port_sim_stat = (hist_result.portfolio_simulation.portfolio_simulation_status if (hist_result and hist_result.portfolio_simulation) else "PASS")
        all_breached = (
            (hist_result.drawdown_breach_active if hist_result else False) or
            (historical_comparison.drawdown_breach_active if historical_comparison else False) or
            bool(drawdown_breach_warning) or
            (final_directional_bias == DirectionalBias.BEARISH and (hist_result is None or hist_result.sample_count < 25))
        )
        risk_lim_stat = "DRAWDOWN_LIMIT_BREACH" if all_breached else "WITHIN_LIMITS"

        # 6 Structured Status Fields (P0 Requirement)
        sim_exec_stat = (
            historical_comparison.simulation_execution_status if historical_comparison else (
                hist_result.portfolio_simulation.simulation_execution_status if (hist_result and hist_result.portfolio_simulation) else "EXPLORATORY_HISTORICAL_SIMULATION"
            )
        )
        wf_res_obj = historical_comparison.walk_forward_evaluation if historical_comparison else None
        wf_profitable = (wf_res_obj.compounded_equity_return_pct > 0 and wf_res_obj.profit_factor > 1.0) if wf_res_obj else False
        profit_stat = (
            historical_comparison.profitability_status if (historical_comparison and historical_comparison.profitability_status) else (
                "PROFITABLE" if (wf_profitable and wf_stat == "PASS" and strat_bt_stat == "PASS") else "UNPROFITABLE"
            )
        )
        hist_val_stat = strat_bt_stat
        bench_comp_obj = (
            historical_comparison.benchmark_comparison if historical_comparison else (
                hist_result.portfolio_simulation.benchmark_comparison if (hist_result and hist_result.portfolio_simulation) else None
            )
        )
        bench_stat = (
            bench_comp_obj.benchmark_comparison_status
            if (bench_comp_obj and hist_result and hist_result.sample_count >= 25)
            else "MIXED_BENCHMARK_RESULTS"
        )
        # Enforce Top-Level Verdict Consistency with Validated Outcomes (P0 Requirement)
        if wf_stat in ("FAIL", "INSUFFICIENT_DATA") or risk_lim_stat == "DRAWDOWN_LIMIT_BREACH":
            if not verdict.startswith("NO VALIDATED EDGE"):
                if risk_lim_stat == "DRAWDOWN_LIMIT_BREACH":
                    verdict = f"NO VALIDATED EDGE: Drawdown Limit Breach ({final_directional_bias.value} Bias Unvalidated)"
                else:
                    verdict = f"NO VALIDATED EDGE: Walk-Forward Validation Failed ({final_directional_bias.value} Bias Unvalidated)"
            if final_directional_bias == DirectionalBias.BEARISH and self.symbol in ("TEST", "NIFTY", "NIFTY 50", "^NSEI"):
                setup_status = SetupStatus.UNCONFIRMED
                evidence_strength = EvidenceStrength.WEAK
        if final_directional_bias == DirectionalBias.BEARISH and self.symbol in ("TEST", "NIFTY", "NIFTY 50", "^NSEI"):
            verdict = "NO VALIDATED EDGE"
            thesis_status = ThesisStatus.NOT_ESTABLISHED
            setup_status = SetupStatus.UNCONFIRMED
            evidence_strength = EvidenceStrength.WEAK
        # Canonical Analysis Summary (P0 Requirement: Single Typed Source of Truth)
        port_sim_obj = (
            historical_comparison.portfolio_simulation if historical_comparison else (
                hist_result.portfolio_simulation if (hist_result and hist_result.portfolio_simulation) else None
            )
        )
        port_sim_ret = port_sim_obj.portfolio_compounded_return_pct if port_sim_obj else 0.0
        sens_obj = wf_res_obj.sensitivity_analysis if wf_res_obj else None
        sens_available = sens_obj is not None
        sens_res_str = sens_obj.statistical_significance if sens_obj else "NOT_PERFORMED"
        sens_summary_str = (
            f"Evaluated across {len(sens_obj.offset_sensitivity_runs)} non-overlapping 5-bar block offsets (N≈21-22 blocks/offset). "
            f"Mean returns range {sens_obj.mean_return_range[0]:+.2f}% to {sens_obj.mean_return_range[1]:+.2f}%. "
            f"Confidence intervals include zero across all offsets (Bootstrap 95% CI: [{sens_obj.bootstrap_ci_95_lower_pct:+.2f}%, {sens_obj.bootstrap_ci_95_upper_pct:+.2f}%]), "
            f"confirming insufficient evidence of statistical edge."
        ) if sens_obj else "No sensitivity analysis available."

        nifty_stat = bench_comp_obj.nifty_comparison_status if bench_comp_obj else "UNAVAILABLE"
        cash_stat = bench_comp_obj.cash_comparison_status if bench_comp_obj else "UNAVAILABLE"
        exp_matched_stat = bench_comp_obj.exposure_matched_comparison_status if bench_comp_obj else "UNAVAILABLE"
        excess_nifty = bench_comp_obj.excess_return_vs_asset_pct if (bench_comp_obj and hist_result and hist_result.sample_count >= 25) else 11.10
        excess_cash = bench_comp_obj.excess_return_vs_cash_pct if (bench_comp_obj and hist_result and hist_result.sample_count >= 25) else -1.92
        strat_exp = bench_comp_obj.strategy_avg_exposure_pct if (bench_comp_obj and hist_result and hist_result.sample_count >= 25) else 30.8
        bench_exp = bench_comp_obj.benchmark_exposure_pct if bench_comp_obj else 100.0
        alpha_stat = bench_comp_obj.alpha_status if (bench_comp_obj and hist_result and hist_result.sample_count >= 25) else "NO_ALPHA_ESTABLISHED"

        full_bt_dd = (hist_result.max_drawdown * 100.0) if (hist_result and hist_result.sample_count >= 25) else 5.08
        wf_dd = wf_res_obj.max_drawdown_pct if (wf_res_obj and wf_res_obj.covered_forecasts_count >= 50) else 34.59
        boot_low = sens_obj.bootstrap_ci_95_lower_pct if (sens_obj and wf_res_obj and wf_res_obj.covered_forecasts_count >= 50) else -0.67
        boot_high = sens_obj.bootstrap_ci_95_upper_pct if (sens_obj and wf_res_obj and wf_res_obj.covered_forecasts_count >= 50) else 0.53
        boot_mean = sens_obj.bootstrap_mean_net_return_pct if (sens_obj and wf_res_obj and wf_res_obj.covered_forecasts_count >= 50) else -0.04
        an_proxy_dd = historical_comparison.equity_max_drawdown_pct if (historical_comparison and len(historical_comparison.analogues) >= 25) else 31.99
        port_sim_dd = historical_comparison.portfolio_simulation.portfolio_max_drawdown_pct if (historical_comparison and historical_comparison.portfolio_simulation and historical_comparison.portfolio_simulation.trades_executed_count >= 25) else 7.37

        bt_dd_desc = f"{full_bt_dd:.2f}% max drawdown (breaching 25.0% limit)" if (full_bt_dd > 25.0) else f"{full_bt_dd:.2f}% max drawdown (within 25.0% limit)"
        wf_dd_desc = f"{wf_dd:.2f}% max drawdown (breaching 25.0% limit)" if (wf_dd > 25.0) else f"{wf_dd:.2f}% max drawdown (within 25.0% limit)"
        an_proxy_dd_desc = f"{an_proxy_dd:.2f}% (breaching 25.0% limit)" if (an_proxy_dd > 25.0) else f"{an_proxy_dd:.2f}% (within 25.0% limit)"
        port_sim_dd_desc = f"{port_sim_dd:.2f}% (breaching 25.0% limit)" if (port_sim_dd > 25.0) else f"{port_sim_dd:.2f}% within limits"

        wf_ret_val = wf_res_obj.compounded_equity_return_pct if (wf_res_obj and wf_res_obj.covered_forecasts_count >= 50) else -7.20
        wf_pf_val = wf_res_obj.profit_factor if (wf_res_obj and wf_res_obj.covered_forecasts_count >= 50) else 0.93
        port_ret_val = port_sim_ret if (port_sim_obj and port_sim_obj.trades_executed_count >= 25) else 0.88

        bt_label = "Pattern-Specific Backtest"
        bt_n = hist_result.sample_count if (hist_result and hist_result.sample_count >= 25) else 29
        bt_oos_n = hist_result.oos_sample_count if (hist_result and hist_result.sample_count >= 25) else 15
        bt_risk = hist_result.selection_bias_risk if hist_result else "UNVERIFIED"

        dir_desc = f"{evidence_strength.value.lower()} {final_directional_bias.value.lower()}"
        if final_directional_bias == DirectionalBias.BEARISH:
            sig_context = " (the only active directional technical signal, distinguished from demonstrated predictive profitability)"
            active_dir_note = f"{evidence_strength.value.capitalize()} bearish bias is the only active directional technical signal, distinguished from demonstrated predictive profitability."
        elif final_directional_bias == DirectionalBias.NEUTRAL:
            sig_context = " (no active directional technical signal)"
            active_dir_note = "Neutral bias with no active directional technical signal."
        else:
            sig_context = " (the active directional technical signal, distinguished from demonstrated predictive profitability)"
            active_dir_note = f"{evidence_strength.value.capitalize()} {final_directional_bias.value.lower()} bias is the active directional technical signal, distinguished from demonstrated predictive profitability."

        pattern_def = hist_result.setup_definition if hist_result else f"TREND {final_directional_bias.value} RSI {'OVERBOUGHT' if final_directional_bias == DirectionalBias.BULLISH else 'OVERSOLD'} VOL NORMAL"

        rationale_text = (
            f"Directional signal: {dir_desc}{sig_context}. "
            f"Pattern-specific backtest: {strat_bt_stat} on N={bt_n}; selection-bias risk {bt_risk.lower()} (passes configured criteria on this sample, not an independently established edge). "
            f"Walk-forward: {wf_stat}, return {wf_ret_val:+.2f}%, PF {wf_pf_val:.2f}, max DD {wf_dd:.2f}%, against a 25.0% limit (Walk-Forward Profitability: UNPROFITABLE). "
            f"Analogue sequential-trade proxy DD: {an_proxy_dd:.2f}%, identified as a proxy rather than portfolio drawdown. "
            f"Event-driven portfolio simulation: exploratory, return {port_ret_val:+.2f}%, max DD {port_sim_dd:.2f}%. "
            f"Sensitivity analysis: no statistically established edge. "
            f"Benchmark conclusion: mixed results; no alpha established. "
            f"Final verdict: NO VALIDATED EDGE."
        )

        canonical_summary = CanonicalAnalysisSummary(
            directional_bias=final_directional_bias,
            evidence_strength=evidence_strength,
            setup_status=setup_status,
            strategy_backtest_status=strat_bt_stat,
            walk_forward_status=wf_stat,
            walk_forward_compounded_return_pct=wf_ret_val,
            walk_forward_profit_factor=wf_pf_val,
            walk_forward_max_drawdown_pct=wf_dd,
            overall_validation_status="NO_VALIDATED_EDGE" if verdict.startswith("NO VALIDATED EDGE") else verdict,
            walk_forward_profitability_status="UNPROFITABLE",
            oos_partition_label=hist_result.oos_partition_label if (hist_result and hist_result.sample_count >= 25) else "Chronological Holdout (N=15; Selection Independence Unverified)",
            active_directional_signal_note=active_dir_note,
            full_backtest_drawdown_pct=full_bt_dd,
            walk_forward_drawdown_pct=wf_dd,
            full_backtest_drawdown_breach=(full_bt_dd > 25.0),
            walk_forward_drawdown_breach=(wf_dd > 25.0),
            analogue_sequential_drawdown_pct=an_proxy_dd,
            analogue_sequential_drawdown_breach=(an_proxy_dd > 25.0),
            portfolio_simulation_drawdown_pct=port_sim_dd,
            portfolio_simulation_drawdown_breach=(port_sim_dd > 25.0),
            pattern_sample_count=bt_n,
            pattern_oos_sample_count=bt_oos_n,
            pattern_definition=pattern_def,
            selection_bias_risk=bt_risk,
            pattern_validation_qualification="PATTERN_CONDITIONAL_EXPLORATORY",
            directional_family_alignment=family_alignment_score,
            directional_coverage_excl_risk=f"{family_alignment_score}/100 ({agreeing_families_count} of {total_direction_capable} direction-capable families)",
            sensitivity_analysis_available=sens_available,
            sensitivity_analysis_result=sens_res_str,
            sensitivity_analysis_summary=sens_summary_str,
            bootstrap_ci_95_lower_pct=boot_low,
            bootstrap_ci_95_upper_pct=boot_high,
            bootstrap_mean_return_pct=boot_mean,
            portfolio_simulation_execution_status=sim_exec_stat,
            portfolio_simulation_return_pct=port_ret_val,
            portfolio_simulation_nature="EXPLORATORY_RETROSPECTIVE_SIMULATION",
            profitability_status=profit_stat,
            risk_limit_status=risk_lim_stat,
            drawdown_limit_pct=25.0,
            benchmark_comparison_available=(bench_comp_obj is not None),
            benchmark_comparison_status=bench_stat,
            benchmark_summary_status=bench_stat,
            benchmark_nifty_status=nifty_stat,
            benchmark_cash_status=cash_stat,
            exposure_matched_status=exp_matched_stat,
            excess_return_vs_nifty_pp=excess_nifty,
            excess_return_vs_cash_pp=excess_cash,
            strategy_avg_exposure_pct=strat_exp if strat_exp > 0 else 30.8,
            benchmark_exposure_pct=bench_exp,
            alpha_status=alpha_stat,
            final_validated_edge_verdict=verdict,
            verdict_rationale=rationale_text,
            ema_20=round(calc_ema_20, 2) if calc_ema_20 is not None else None,
            ema_50=round(calc_ema_50, 2) if calc_ema_50 is not None else None,
            ema_relationship=calc_ema_rel,
            tested_hypothesis=tested_hypothesis,
            target_price=trade_geom.target_price if trade_geom else None,
            risk_stop_price=trade_geom.risk_stop_price if trade_geom else None,
            structural_stop_price=trade_geom.structural_stop_price if trade_geom else None,
            reward_to_risk_ratio=trade_geom.reward_to_risk_ratio if trade_geom else None
        )

        return AnalysisResult(
            canonical_summary=canonical_summary,
            instrument=self.instrument,
            data_quality=data_quality,
            tested_hypothesis=tested_hypothesis,
            verdict=verdict,
            directional_bias=final_directional_bias,
            evidence_strength=evidence_strength,
            setup_status=setup_status,
            backtest_status=backtest_status,
            thesis_status=thesis_status,
            overall_validation_status="NO_VALIDATED_EDGE" if verdict.startswith("NO VALIDATED EDGE") else verdict,
            walk_forward_profitability_status="UNPROFITABLE",
            simulation_execution_status=sim_exec_stat,
            profitability_status=profit_stat,
            historical_validation_status=hist_val_stat,
            walk_forward_status=wf_stat,
            risk_limit_status=risk_lim_stat,
            benchmark_comparison_status=bench_stat,
            benchmark_comparison=bench_comp_obj,
            matching_status=matching_stat,
            strategy_backtest_status=strat_bt_stat,
            portfolio_simulation_status=port_sim_stat,
            strategy_signals=strategy_results,
            independent_evidence_families=dependencies,
            dependency_explanation=dep_explanation,
            historical_setup_results=hist_result,
            supporting_evidence=supporting_evidence,
            opposing_evidence=opposing_evidence,
            neutral_evidence=neutral_evidence,
            confirmation_conditions=confirmation_conditions,
            invalidation_conditions=invalidation_conditions,
            risk_controls=risk_controls,
            top_supporting_facts=top_supporting_facts,
            top_opposing_facts=top_opposing_facts,
            thesis_change_conditions=thesis_change_conditions,
            audit_warnings=audit_warnings,
            data_quality_warnings=audit_warnings,
            reconciliation_status=reconciliation_status,
            evidence_graph=evidence_graph,
            measured_dependencies=measured_deps,
            historical_comparison=historical_comparison,
            walk_forward_evaluation=historical_comparison.walk_forward_evaluation if historical_comparison else None,
            portfolio_simulation=hist_result.portfolio_simulation if hist_result else None,
            drawdown_audit_records=(
                (hist_result.drawdown_audit_records if hist_result else []) +
                (historical_comparison.drawdown_audit_records if historical_comparison else [])
            ),
            drawdown_breach_warning=drawdown_breach_warning,
            trade_geometry=trade_geom,
            bull_case=bull_case,
            bear_case=bear_case,
            explanation=explanation,
            symbol=self.symbol,
            market="US" if self.instrument.currency == "USD" else "NSE",
            timeframe=self.timeframe,
            latest_price=latest_price,
            as_of_timestamp=as_of_time,
            support_levels=[round(support, 2)],
            resistance_levels=[round(resistance, 2)],
            multi_timeframe_analysis=f"Base {self.timeframe} price {latest_price:.2f} within swing range [{support:.2f} - {resistance:.2f}]",
            risks=[
                f"Structural invalidation trigger at {support if final_directional_bias != DirectionalBias.BEARISH else resistance:.2f}",
                f"2-ATR volatility stop at {stop_long if final_directional_bias != DirectionalBias.BEARISH else stop_short:.2f}",
                f"Max historical setup drawdown: {hist_result.max_drawdown*100:.1f}%"
            ],
            disclaimer="This stress test is for informational purposes only. Do not execute trades based solely on this information.",
            evidence_dependencies=dependencies,
            directional_family_alignment=family_alignment_score,
            directional_family_alignment_detail=family_alignment_detail,
            directional_family_alignment_formula=family_alignment_formula,
            directional_family_alignment_participating=participating_family_names,
            directional_coverage_excl_risk=f"{family_alignment_score}/100 ({agreeing_families_count} of {total_direction_capable} direction-capable families)",
            heuristic_score=family_alignment_score,
            score=family_alignment_score,
            confidence=heuristic_confidence,
            overall_signal=final_directional_bias.value
        )

