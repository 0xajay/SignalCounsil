import base64
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware

from app.adapters.yahoo_finance import get_ohlcv, validate_symbol, get_latest_price, get_instrument_metadata, _normalize_symbol, resolve_symbol
from app.domain.models import (
    OHLCVData, StrategyContext, Signal, StrategyResult,
    DirectionalBias, EvidenceStrength, SetupStatus, BacktestStatus,
    ThesisStatus, EvidenceClaim, EvidenceDependency, HistoricalSetupResult,
    ConfirmationCondition, InvalidationCondition, RiskControlCondition, TradeThesisStressTestResult,
    PerformanceMetrics, InstrumentMetadata, DataQualityReport, AnalysisResult,
    EvidenceGraph, MeasuredDependence, HistoricalComparisonResult, TradeGeometry,
    CanonicalAnalysisSummary
)
from app.domain.strategies import (
    TrendStrategy, MomentumStrategy, MeanReversionStrategy,
    BreakoutStrategy, PriceActionStrategy, RiskStrategy
)
from app.agents.chart_agent import ChartParserAgent
from app.agents.council import SignalCouncilOrchestrator
from app.domain.backtest import BacktestEngine
from app.domain.stress_tester import TradeThesisStressTester

from app.config import settings

app = FastAPI(title="SignalCouncil API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    symbol: str
    market: Optional[str] = None
    timeframe: str = "1d"
    screenshot: Optional[str] = None

class SignalResponse(BaseModel):
    strategy: str
    signal: str
    confidence: float

class AnalyzeResponse(BaseModel):
    # Canonical Domain Fields
    canonical_summary: Optional[CanonicalAnalysisSummary] = None
    instrument: Optional[InstrumentMetadata] = None
    data_quality: Optional[DataQualityReport] = None
    tested_hypothesis: str = "RANGE_CONSOLIDATION"
    verdict: str = "NO VALIDATED EDGE"
    reconciliation_status: str = "VALIDATED"
    audit_warnings: List[str] = []
    independent_evidence_families: List[EvidenceDependency] = []
    dependency_explanation: str = ""
    neutral_evidence: List[EvidenceClaim] = []

    # Advanced Differentiation Fields
    evidence_graph: Optional[EvidenceGraph] = None
    measured_dependencies: List[MeasuredDependence] = []
    historical_comparison: Optional[HistoricalComparisonResult] = None
    trade_geometry: Optional[TradeGeometry] = None

    symbol: str
    market: str
    timeframe: str
    latest_price: float
    directional_bias: str
    evidence_strength: str
    setup_status: str
    backtest_status: str
    thesis_status: str
    simulation_execution_status: str = "EXPLORATORY_HISTORICAL_SIMULATION"
    profitability_status: str = "UNPROFITABLE"
    historical_validation_status: str = "FAIL"
    walk_forward_status: str = "FAIL"
    risk_limit_status: str = "DRAWDOWN_LIMIT_BREACH"
    benchmark_comparison_status: str = "MIXED_BENCHMARK_RESULTS"
    directional_coverage_excl_risk: Optional[str] = None
    supporting_evidence: List[EvidenceClaim] = []
    opposing_evidence: List[EvidenceClaim] = []
    evidence_dependencies: List[EvidenceDependency] = []
    historical_setup_results: HistoricalSetupResult
    confirmation_conditions: List[ConfirmationCondition] = []
    invalidation_conditions: List[InvalidationCondition] = []
    risk_controls: List[RiskControlCondition] = []
    top_supporting_facts: List[str] = []
    top_opposing_facts: List[str] = []
    thesis_change_conditions: List[str] = []
    data_quality_warnings: List[str] = []
    explanation: str

    # Compatibility & UI metadata
    overall_signal: str
    directional_family_alignment: int = 0
    directional_family_alignment_detail: str = ""
    directional_family_alignment_formula: str = ""
    directional_family_alignment_participating: List[str] = []
    heuristic_score: int = 50
    score: int
    confidence: float
    strategy_signals: List[SignalResponse]
    support_levels: List[float] = []
    resistance_levels: List[float] = []
    multi_timeframe_analysis: str = ""
    risks: List[str] = []
    bull_case: str = ""
    bear_case: str = ""
    disclaimer: str = ""

class StrategyBacktestResult(BaseModel):
    strategy_name: str
    metrics: PerformanceMetrics

class BacktestResponse(BaseModel):
    symbol: str
    timeframe: str
    results: List[StrategyBacktestResult]

import logging

log_level = getattr(logging, settings.log_level, logging.INFO)
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@app.get("/health", tags=["Health"])
@app.get("/healthz", tags=["Health"])
@app.get("/api/v1/health", tags=["Health"])
@app.get("/", tags=["Health"])
def health_check():
    """Lightweight health check endpoint returning service status without exposing credentials."""
    return {
        "status": "healthy",
        "service": "signal-council-api",
        "version": "1.0.0",
        "environment": settings.environment,
    }

@app.post("/api/v1/analyze", response_model=AnalyzeResponse)
async def analyze_symbol(request: AnalyzeRequest):
    clean_sym = request.symbol.strip().upper()
    canonical_sym = resolve_symbol(clean_sym, market=request.market)
    logger.info(f"--- STARTING ANALYSIS FOR {clean_sym} (market={request.market}, provider_ticker={canonical_sym}) ---")
    if not validate_symbol(canonical_sym):
        raise HTTPException(status_code=400, detail=f"Invalid symbol: {request.symbol} for market {request.market}")
        
    # 1. Fetch Market Data (Multi-Timeframe)
    try:
        logger.info(f"Fetching market data for {canonical_sym} {request.timeframe}...")
        raw_data = get_ohlcv(canonical_sym, interval=request.timeframe, limit=200)
        latest_price = get_latest_price(canonical_sym)
        
        # Determine Macro and Micro timeframes based on requested timeframe
        tf_map = {
            "1m": ("5m", "1m"),
            "5m": ("15m", "1m"),
            "15m": ("1h", "5m"),
            "1h": ("1d", "15m"),
            "1d": ("1wk", "1h"),
            "1wk": ("1mo", "1d"),
            "1mo": ("3mo", "1wk")
        }
        macro_tf, micro_tf = tf_map.get(request.timeframe, ("1wk", "1h"))
        
        macro_data = get_ohlcv(canonical_sym, interval=macro_tf, limit=20)
        
        logger.info(f"Multi-Timeframe data fetched. Base: {request.timeframe}, Macro: {macro_tf}")
    except Exception as e:
        logger.error(f"Market data error: {e}")
        raise HTTPException(status_code=500, detail=f"Market data error: {str(e)}")
        
    market_data = OHLCVData(data=raw_data)
    macro_market_data = OHLCVData(data=macro_data)
    context = StrategyContext(symbol=clean_sym, timeframe=request.timeframe)
    
    # 2. Evaluate Deterministic Strategies
    logger.info("Evaluating deterministic technical strategies...")
    strategies = [
        TrendStrategy(), MomentumStrategy(), MeanReversionStrategy(),
        BreakoutStrategy(), PriceActionStrategy(), RiskStrategy()
    ]
    
    signals = []
    strategy_results: List[StrategyResult] = []
    for strat in strategies:
        res = strat.evaluate(market_data, context)
        strategy_results.append(res)
        signals.append(SignalResponse(
            strategy=res.strategy_name,
            signal=res.signal.value,
            confidence=res.confidence
        ))
    logger.info(f"Strategies evaluated. Signals: {[s.signal for s in signals]}")

    # Calculate simple macro trend
    macro_trend = "Neutral"
    if len(macro_data) > 10:
        macro_ma = sum(c.close for c in macro_data[-10:]) / 10
        macro_trend = "Bullish" if latest_price > macro_ma else "Bearish"

    # Fetch Instrument Metadata
    instrument_meta = get_instrument_metadata(canonical_sym, market=request.market)
    instrument_meta.symbol = clean_sym

    # 3. Deterministic Trade Thesis Stress Testing
    logger.info("Running Deterministic Trade Thesis Stress Tester...")
    stress_tester = TradeThesisStressTester(
        raw_data, symbol=clean_sym, timeframe=request.timeframe, instrument=instrument_meta
    )
    stress_test_result = stress_tester.run_stress_test(strategy_results)
    stress_test_result.latest_price = latest_price
    stress_test_result.multi_timeframe_analysis = (
        f"Base {request.timeframe} ({len(raw_data)} candles) vs Macro {macro_tf} (Trend: {macro_trend}). "
        f"Price: {latest_price:.2f} within swing range [{stress_test_result.support_levels[0] if stress_test_result.support_levels else 0:.2f} - "
        f"{stress_test_result.resistance_levels[0] if stress_test_result.resistance_levels else 0:.2f}]."
    )

    # 4. Chart Analysis (if image provided)
    chart_observation_text = ""
    if request.screenshot:
        logger.info("Processing screenshot with ChartParserAgent...")
        for attempt in range(3):
            try:
                img_data = request.screenshot
                if "," in img_data:
                    img_data = img_data.split(",")[1]
                image_bytes = base64.b64decode(img_data)
                chart_agent = ChartParserAgent()
                obs = await chart_agent.analyze_chart(image_bytes)
                chart_observation_text = obs.model_dump_json(indent=2)
                break
            except Exception as e:
                logger.warning(f"Chart parsing attempt {attempt+1} failed: {e}")
                if ("503" in str(e) or "429" in str(e)) and attempt < 2:
                    import asyncio
                    await asyncio.sleep(2)
                    continue
                chart_observation_text = f"Error parsing chart image: {str(e)}"
                break

    # 5. Adversarial Trade Review via Council Orchestrator
    logger.info("Running Adversarial Trade Review...")
    orchestrator = SignalCouncilOrchestrator()
    try:
        final_result = await orchestrator.stress_test_trade_thesis(
            stress_test_result, chart_observation=chart_observation_text
        )
    except Exception as e:
        logger.warning(f"Adversarial LLM review skipped: {e}")
        final_result = stress_test_result

    # Agreement score index (not a probability, explicitly derived from independent families)
    total_families = len(final_result.evidence_dependencies) or 1
    agreeing_families = len([d for d in final_result.evidence_dependencies if d.independent_vote == final_result.directional_bias])
    calibrated_score = int((agreeing_families / total_families) * 100) if final_result.directional_bias != DirectionalBias.NEUTRAL else 0

    confidence_score = 0.0 if final_result.directional_bias == DirectionalBias.NEUTRAL else (
        0.85 if final_result.evidence_strength == EvidenceStrength.STRONG else (
            0.65 if final_result.evidence_strength == EvidenceStrength.MODERATE else 0.40
        )
    )

    logger.info("--- STRESS TEST ANALYSIS COMPLETE ---")
    return AnalyzeResponse(
        canonical_summary=final_result.canonical_summary,
        instrument=final_result.instrument,
        data_quality=final_result.data_quality,
        tested_hypothesis=final_result.tested_hypothesis,
        verdict=final_result.verdict,
        reconciliation_status=final_result.reconciliation_status,
        audit_warnings=final_result.audit_warnings,
        independent_evidence_families=final_result.independent_evidence_families or final_result.evidence_dependencies,
        dependency_explanation=final_result.dependency_explanation,
        neutral_evidence=final_result.neutral_evidence,
        evidence_graph=final_result.evidence_graph,
        measured_dependencies=final_result.measured_dependencies,
        historical_comparison=final_result.historical_comparison,
        trade_geometry=final_result.trade_geometry,
        symbol=final_result.symbol or clean_sym,
        market=request.market or instrument_meta.exchange or "GLOBAL",
        timeframe=request.timeframe,
        latest_price=latest_price,
        directional_bias=final_result.directional_bias.value,
        evidence_strength=final_result.evidence_strength.value,
        setup_status=final_result.setup_status.value,
        backtest_status=final_result.backtest_status.value,
        thesis_status=final_result.thesis_status.value,
        simulation_execution_status=final_result.simulation_execution_status,
        profitability_status=final_result.profitability_status,
        historical_validation_status=final_result.historical_validation_status,
        walk_forward_status=final_result.walk_forward_status,
        risk_limit_status=final_result.risk_limit_status,
        benchmark_comparison_status=final_result.benchmark_comparison_status,
        directional_coverage_excl_risk=final_result.directional_coverage_excl_risk or (
            final_result.canonical_summary.directional_coverage_excl_risk if final_result.canonical_summary else None
        ),
        supporting_evidence=final_result.supporting_evidence,
        opposing_evidence=final_result.opposing_evidence,
        evidence_dependencies=final_result.evidence_dependencies,
        historical_setup_results=final_result.historical_setup_results,
        confirmation_conditions=final_result.confirmation_conditions,
        invalidation_conditions=final_result.invalidation_conditions,
        risk_controls=final_result.risk_controls,
        top_supporting_facts=final_result.top_supporting_facts,
        top_opposing_facts=final_result.top_opposing_facts,
        thesis_change_conditions=final_result.thesis_change_conditions,
        data_quality_warnings=final_result.data_quality_warnings or final_result.audit_warnings,
        explanation=final_result.explanation,
        overall_signal=final_result.directional_bias.value,
        directional_family_alignment=final_result.directional_family_alignment,
        directional_family_alignment_detail=final_result.directional_family_alignment_detail,
        directional_family_alignment_formula=final_result.directional_family_alignment_formula,
        directional_family_alignment_participating=final_result.directional_family_alignment_participating,
        heuristic_score=final_result.heuristic_score,
        score=calibrated_score,
        confidence=confidence_score,
        strategy_signals=signals,
        support_levels=final_result.support_levels,
        resistance_levels=final_result.resistance_levels,
        multi_timeframe_analysis=final_result.multi_timeframe_analysis,
        risks=final_result.risks,
        bull_case=final_result.bull_case,
        bear_case=final_result.bear_case,
        disclaimer=final_result.disclaimer
    )

@app.post("/api/v1/backtest", response_model=BacktestResponse)
async def backtest_symbol(request: AnalyzeRequest):
    clean_sym = request.symbol.strip().upper()
    canonical_sym = resolve_symbol(clean_sym, market=request.market)
    logger.info(f"--- STARTING BACKTEST FOR {clean_sym} (market={request.market}, provider_ticker={canonical_sym}) ---")
    if not validate_symbol(canonical_sym):
        raise HTTPException(status_code=400, detail=f"Invalid symbol: {request.symbol} for market {request.market}")
        
    try:
        raw_data = get_ohlcv(canonical_sym, interval=request.timeframe, limit=200) # Fetch more data for backtesting
    except Exception as e:
        logger.error(f"Market data error: {e}")
        raise HTTPException(status_code=500, detail=f"Market data error: {str(e)}")
        
    context = StrategyContext(symbol=clean_sym, timeframe=request.timeframe)
    
    strategies = [
        TrendStrategy(), MomentumStrategy(), MeanReversionStrategy(),
        BreakoutStrategy(), PriceActionStrategy(), RiskStrategy()
    ]
    
    results = []
    for strat in strategies:
        try:
            engine = BacktestEngine(market_data=raw_data, strategy=strat, context=context, warmup_period=20)
            metrics = engine.run()
            results.append(StrategyBacktestResult(
                strategy_name=strat.__class__.__name__,
                metrics=metrics
            ))
        except Exception as e:
            logger.error(f"Backtest failed for {strat.__class__.__name__}: {e}")
            
    logger.info(f"--- BACKTEST COMPLETE for {clean_sym} ---")
    
    return BacktestResponse(
        symbol=clean_sym,
        timeframe=request.timeframe,
        results=results
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host=settings.host, port=settings.port, reload=(settings.environment == "development"))

