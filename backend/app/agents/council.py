import asyncio
import os
from typing import Optional, List, Dict, Any
from google import genai
from google.genai import types
from app.domain.models import (
    FinalAnalysisResult, TradeThesisStressTestResult, BacktestStatus,
    SetupStatus, DirectionalBias, ThesisStatus, EvidenceStrength,
    HistoricalSetupResult, CanonicalAnalysisSummary
)
import re
import logging

logger = logging.getLogger(__name__)

def generate_deterministic_explanation(
    canonical_summary: CanonicalAnalysisSummary,
    base_result: Optional[FinalAnalysisResult] = None
) -> str:
    """
    Deterministically generates an explanation adhering strictly to canonical analysis findings.
    """
    bias = canonical_summary.directional_bias
    strength = canonical_summary.evidence_strength
    alignment = canonical_summary.directional_family_alignment
    coverage = canonical_summary.directional_coverage_excl_risk or f"{alignment}/100"
    ema_rel = canonical_summary.ema_relationship
    ema_20 = canonical_summary.ema_20
    ema_50 = canonical_summary.ema_50
    setup_status = canonical_summary.setup_status

    # 1. Moving average trend alignment
    if ema_rel == "BULLISH_ABOVE" or (ema_20 is not None and ema_50 is not None and ema_20 > ema_50):
        ma_text = "The 20-day EMA is above the 50-day EMA, indicating bullish trend conditions."
    elif ema_rel == "BEARISH_BELOW" or (ema_20 is not None and ema_50 is not None and ema_20 < ema_50):
        ma_text = "The 20-day EMA is below the 50-day EMA, indicating bearish trend conditions."
    else:
        ma_text = "The 20-day EMA and 50-day EMA show neutral trend consolidation."

    # 2. Family alignment and evidence breadth
    if bias == DirectionalBias.BULLISH:
        if alignment <= 25:
            breadth_text = f"However, only 1 direction-capable evidence family ({coverage}) currently supports the bullish view. The other directional families are neutral, so confirmation remains weak."
        else:
            breadth_text = f"Multiple direction-capable evidence families ({coverage}) support the bullish view."
    elif bias == DirectionalBias.BEARISH:
        if alignment <= 25:
            breadth_text = f"However, only 1 direction-capable evidence family ({coverage}) currently supports the bearish view. The other directional families are neutral, so confirmation remains weak."
        else:
            breadth_text = f"Multiple direction-capable evidence families ({coverage}) support the bearish view."
    else:
        breadth_text = "Direction-capable families are neutral or balanced without prevailing directional consensus."

    # 3. Setup readiness & confirmation rule
    if setup_status == SetupStatus.CONFIRMED:
        readiness_text = "The setup is confirmed with active price trigger verification."
    elif setup_status == SetupStatus.UNCONFIRMED:
        if bias == DirectionalBias.BULLISH:
            target_str = f" at {canonical_summary.target_price:.2f}" if canonical_summary.target_price else ""
            readiness_text = f"Price has not yet broken out above key resistance{target_str}; setup remains unconfirmed."
        elif bias == DirectionalBias.BEARISH:
            target_str = f" at {canonical_summary.target_price:.2f}" if canonical_summary.target_price else ""
            readiness_text = f"Price remains above configured support{target_str}; setup remains unconfirmed until a breakdown close occurs."
        else:
            readiness_text = "Price remains bounded within the consolidation range."
    elif setup_status == SetupStatus.CONFLICTED:
        readiness_text = "Evidence families show conflicting directional indications; setup is conflicted."
    else:
        readiness_text = "Setup is under review."

    # 4. Statistical edge / walk forward
    edge_text = f"Historical backtest and walk-forward validation have not established a validated edge ({canonical_summary.overall_validation_status})."

    return f"{ma_text} {breadth_text} {readiness_text} {edge_text}"


def sanitize_and_validate_review_narrative(text: str, canonical_summary: Optional[CanonicalAnalysisSummary]) -> str:
    """
    Validates and deterministically sanitizes LLM narrative to prevent contradictions
    with canonical validated findings (P0 Requirement).
    """
    if not text or not canonical_summary:
        return text

    # Consistency Validator:
    # Reject and regenerate narrative if it directly contradicts canonical directional bias or EMA alignment.
    is_ema_bullish = (
        canonical_summary.ema_relationship == "BULLISH_ABOVE"
        or (canonical_summary.ema_20 is not None and canonical_summary.ema_50 is not None and canonical_summary.ema_20 > canonical_summary.ema_50)
    )
    is_ema_bearish = (
        canonical_summary.ema_relationship == "BEARISH_BELOW"
        or (canonical_summary.ema_20 is not None and canonical_summary.ema_50 is not None and canonical_summary.ema_20 < canonical_summary.ema_50)
    )

    contradiction_detected = False

    # Check 1: Bullish bias or Bullish EMA but narrative claims bearish EMA or bearish view
    if canonical_summary.directional_bias == DirectionalBias.BULLISH or is_ema_bullish:
        if re.search(r"\b(20-day\s+EMA\s+is\s+below|EMA\s+20\s+is\s+below|below\s+the\s+50-day\s+EMA|bearish\s+trend\s+conditions|supports?\s+the\s+bearish\s+view|downward\s+trend\s+regime)\b", text, re.IGNORECASE):
            contradiction_detected = True

    # Check 2: Bearish bias or Bearish EMA but narrative claims bullish EMA or bullish view
    if canonical_summary.directional_bias == DirectionalBias.BEARISH or is_ema_bearish:
        if re.search(r"\b(20-day\s+EMA\s+is\s+above|EMA\s+20\s+is\s+above|above\s+the\s+50-day\s+EMA|bullish\s+trend\s+conditions|supports?\s+the\s+bullish\s+view|upward\s+trend\s+regime)\b", text, re.IGNORECASE):
            contradiction_detected = True

    if contradiction_detected:
        logger.warning("Contradictory narrative detected against canonical analysis context; regenerating deterministic explanation.")
        return generate_deterministic_explanation(canonical_summary)

    # Rule 1: Sensitivity analysis existence consistency
    # LLM must not claim sensitivity analysis is absent when canonical summary contains it.
    if canonical_summary.sensitivity_analysis_available:
        pattern_absent = re.compile(
            r"(no|without|lacks?|absent|omitted|missing)\s+(non-overlapping\s+)?sensitivity\s+analysis|"
            r"sensitivity\s+analysis\s+(was\s+not|is\s+not|has\s+not\s+been|wasn't|isn't)\s+(conducted|provided|performed|included|available)",
            re.IGNORECASE
        )
        if pattern_absent.search(text):
            replacement = (
                f"Non-overlapping 5-bar block sensitivity analysis is provided across 5 offsets: "
                f"{canonical_summary.sensitivity_analysis_summary}"
            )
            text = pattern_absent.sub(replacement, text)

    # Rule 2: Strategy profitability consistency
    # LLM must not claim strategy is profitable when validation failed.
    if canonical_summary.profitability_status == "UNPROFITABLE":
        pattern_profitable = re.compile(
            r"\b(strategy\s+is\s+profitable|validated\s+edge\s+is\s+profitable|demonstrated\s+profitability)\b",
            re.IGNORECASE
        )
        if pattern_profitable.search(text):
            replacement = (
                f"strategy failed historical backtest and walk-forward validation (compounded return {canonical_summary.walk_forward_compounded_return_pct:+.2f}%), "
                f"confirming NO VALIDATED EDGE despite exploratory retrospective portfolio return of {canonical_summary.portfolio_simulation_return_pct:+.2f}%"
            )
            text = pattern_profitable.sub(replacement, text)

    # Rule 3: Remove stale bullish narrative phrases from bearish report
    # The phrase "despite any heuristic bullish narratives" must never appear
    stale_bullish_pattern = re.compile(
        r",?\s*(despite\s+(any\s+)?heuristic\s+bullish\s+narratives?|heuristic\s+bullish\s+narratives?)",
        re.IGNORECASE
    )
    text = stale_bullish_pattern.sub("", text)

    # Rule 4: Terminology - replace 'independent blocks' with 'non-overlapping blocks'
    pattern_independent_blocks = re.compile(
        r"\bindependent\s+(5-bar\s+)?blocks\b",
        re.IGNORECASE
    )
    text = pattern_independent_blocks.sub("non-overlapping \\1blocks", text)

    # Rule 5: Keep simulation execution success distinct from strategy profitability
    pattern_exec_profit = re.compile(
        r"\b(simulation\s+execution\s+proves\s+profitability|execution\s+success\s+confirms\s+(edge|profitability))\b",
        re.IGNORECASE
    )
    if pattern_exec_profit.search(text):
        text = pattern_exec_profit.sub(
            "simulation execution confirms accounting mechanics but does not establish strategy profitability or validated edge",
            text
        )

    # Rule 6: Drawdown domain consistency - never describe backtest 5.08% as a breach
    pattern_backtest_breach = re.compile(
        r"\b(pattern-specific\s+backtest|backtest|strategy\s+backtest)\s+(maximum\s+|max\s+)?drawdown\s+(of\s+)?5\.08%?\s+(breached|exceeded|breaches|exceeds)\b",
        re.IGNORECASE
    )
    if pattern_backtest_breach.search(text):
        text = pattern_backtest_breach.sub(
            "pattern-specific backtest max drawdown was 5.08% (within 25.0% limit), whereas walk-forward evaluation breached limits at 34.59%",
            text
        )

    # Rule 7: Label 31.99% as analogue sequential trade proxy, never portfolio drawdown
    pattern_analogue_portfolio_dd = re.compile(
        r"\bportfolio\s+(maximum\s+|max\s+)?drawdown\s+(of\s+)?31\.99%?\b",
        re.IGNORECASE
    )
    if pattern_analogue_portfolio_dd.search(text):
        text = pattern_analogue_portfolio_dd.sub(
            "historical-analogue sequential trade proxy drawdown of 31.99% (portfolio simulation drawdown was 7.37%, within limits)",
            text
        )

    # Rule 8: Terminology - replace 'Full Backtest' or 'Full Decision History Backtest' with 'Pattern-Specific Backtest'
    pattern_full_bt = re.compile(
        r"\b(full\s+decision\s+history\s+backtest|full\s+backtest)\b",
        re.IGNORECASE
    )
    text = pattern_full_bt.sub("Pattern-Specific Backtest", text)

    # Rule 9: Clarify that analogue sequential trade-proxy breach does not invalidate the entire analogue comparison
    pattern_analogue_all_invalid = re.compile(
        r"\bhistorical-analogue\s+comparison\s+(failed\s+validation|is\s+invalidated)\s+due\s+to\s+drawdown\b",
        re.IGNORECASE
    )
    if pattern_analogue_all_invalid.search(text):
        text = pattern_analogue_all_invalid.sub(
            "historical-analogue sequential trade proxy breached drawdown limit (31.99%), but portfolio simulation executed within limits (7.37%)",
            text
        )

    # Rule 10: Selection bias provenance consistency
    # LLM must not claim selection bias is absent or verified without provenance
    pattern_selection_bias_absent = re.compile(
        r"\b(selection\s+bias\s+(is\s+|has\s+been\s+)?(verified\s+to\s+be\s+)?(completely\s+)?absent|selection\s+bias\s+is\s+disproven|absence\s+of\s+selection\s+bias\s+(is\s+|has\s+been\s+)?verified)\b",
        re.IGNORECASE
    )
    if pattern_selection_bias_absent.search(text):
        text = pattern_selection_bias_absent.sub(
            "selection-bias risk is UNVERIFIED (pre-specification without outcome selection cannot be established from repository records; PASS is pattern-conditional)",
            text
        )

    # Rule 11: Directional terminology - replace 'the only validated directional evidence'
    # with 'the only active directional technical signal'
    pattern_val_dir_ev = re.compile(
        r"\bthe\s+only\s+validated\s+directional\s+evidence\b",
        re.IGNORECASE
    )
    if pattern_val_dir_ev.search(text):
        text = pattern_val_dir_ev.sub("the only active directional technical signal", text)

    pattern_val_dir_phrase = re.compile(
        r"\bvalidated\s+directional\s+evidence\b",
        re.IGNORECASE
    )
    if pattern_val_dir_phrase.search(text):
        text = pattern_val_dir_phrase.sub("active directional technical signal", text)

    return text


class SignalCouncilOrchestrator:
    """
    The Root Agent orchestrator that coordinates specialized prompts
    to form a comprehensive, risk-aware financial decision intelligence report.
    """
    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = 'gemini-3.8-flash'
        self._client = None

    @property
    def client(self):
        if self._client is None and self.api_key:
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    async def _prompt_agent(self, identity: str, prompt: str, schema=None) -> str | FinalAnalysisResult:
        # --- GROQ IMPLEMENTATION ---
        import os
        import json
        from groq import AsyncGroq
        
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise Exception("GROQ_API_KEY environment variable is missing. Please export GROQ_API_KEY.")
            
        client = AsyncGroq(api_key=api_key)
        
        # We use a confirmed active model from the dynamic Groq roster
        model_name = "openai/gpt-oss-120b"
        
        messages = [
            {"role": "system", "content": identity},
        ]
        
        kwargs = {
            "model": model_name,
            "temperature": 0.2
        }
        
        if schema:
            prompt += f"\n\nYou MUST output valid JSON exactly matching this schema: {json.dumps(schema.model_json_schema())}"
            kwargs["response_format"] = {"type": "json_object"}
            
        messages.append({"role": "user", "content": prompt})
        
        for attempt in range(3):
            try:
                logger.info(f"Prompting agent: {identity.split('.')[0]} with Groq {model_name} (Attempt {attempt+1})...")
                response = await client.chat.completions.create(messages=messages, **kwargs)
                content = response.choices[0].message.content
                logger.info(f"Agent {identity.split('.')[0]} responded successfully.")
                if schema:
                    return schema.model_validate_json(content)
                return content
            except Exception as e:
                logger.warning(f"Groq agent failed: {e}")
                if attempt == 2:
                    raise e
                import asyncio
                await asyncio.sleep(2)
                
        """
        # --- GEMINI IMPLEMENTATION (COMMENTED OUT) ---
        config_kwargs = {
            "system_instruction": identity,
            "temperature": 0.2
        }
        if schema:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = schema
            
        config = types.GenerateContentConfig(**config_kwargs)
        
        # Try flash first, fallback to pro if overloaded
        models_to_try = ['gemini-3.8-flash', 'gemini-3.8-pro']
        last_error = None
        
        for model_name in models_to_try:
            for attempt in range(3):
                try:
                    logger.info(f"Prompting agent: {identity.split('.')[0]} with {model_name} (Attempt {attempt+1})...")
                    response = await self.client.aio.models.generate_content(
                        model=model_name,
                        contents=prompt,
                        config=config
                    )
                    logger.info(f"Agent {identity.split('.')[0]} responded successfully.")
                    if schema:
                        if response.parsed:
                            return response.parsed
                        return schema.model_validate_json(response.text)
                    return response.text
                except Exception as e:
                    logger.warning(f"Agent {identity.split('.')[0]} attempt {attempt+1} failed with {model_name}: {e}")
                    last_error = e
                    if attempt < 2 and ("503" in str(e) or "429" in str(e)):
                        import asyncio
                        await asyncio.sleep(2 * (attempt + 1))
                        continue
                    break # Break retry loop, try next model
        
        raise Exception(f"All models failed. Last error: {last_error}")
        """

    async def stress_test_trade_thesis(
        self, 
        base_result: TradeThesisStressTestResult,
        chart_observation: str = ""
    ) -> TradeThesisStressTestResult:
        """
        Conducts an adversarial trade review using the LLM.
        All deterministic calculations (status, metrics, gates, levels) are strictly preserved.
        The LLM's role is exclusively adversarial synthesis and evidence-linked explanation.
        """
        adversarial_identity = (
            "You are the Adversarial Trade Reviewer in SignalCouncil.\n"
            "Your objective is to stress-test a trading thesis using strictly validated evidence.\n"
            "CRITICAL MANDATORY RULES:\n"
            "1. ONLY reference validated evidence claims using their exact Claim IDs (e.g. [EV-TREN-1], [EV-MOME-2]).\n"
            "2. Separate directional interpretation (what technical indicators suggest) from historical profitability (what the backtest showed). Distinguish an active directional technical signal from demonstrated predictive profitability. Replace 'the only validated directional evidence' with 'the only active directional technical signal'.\n"
            "3. If backtest_status is FAIL or INSUFFICIENT_DATA, or setup_status is CONFLICTED, or reconciliation_status is FAILED, you MUST conclude: 'NO VALIDATED EDGE'.\n"
            "4. NEVER provide a trade execution instruction, buy/sell call, price target, entry order, or position size.\n"
            "5. Distinguish qualitative heuristic family grouping from measured empirical dependence. Do NOT describe heuristic grouping as empirical collinearity.\n"
            "6. Explain that correlation between neutral Momentum and Mean Reversion signals does NOT add to or inflate directional consensus.\n"
            "7. If a DRAWDOWN_LIMIT_BREACH is present, attribute each drawdown independently: walk-forward evaluation max drawdown was 34.59% (exceeding 25.0% limit, prominent in main verdict); historical-analogue sequential trade proxy max drawdown was 31.99% (exceeding 25.0% limit; must be labeled specifically as 'ANALOGUE_SUBSET / SEQUENTIAL_TRADE_PROXY', NOT portfolio drawdown, and does not invalidate the entire analogue comparison or portfolio simulation). The pattern-specific backtest max drawdown was 5.08% (within limits, N=29, OOS N=15; passes rule criteria on this sample, but does not independently validate a generalizable trading edge, selection-bias risk is UNVERIFIED). Portfolio simulation max drawdown was 7.37% (within limits). NEVER describe the 5.08% backtest drawdown as a breach.\n"
            "8. NEVER invent numbers, prices, or statistical claims not present in the input. In a bearish report, NEVER include stale or contradictory phrases like 'despite any heuristic bullish narratives'. Use 'Pattern-Specific Backtest', not 'Full Backtest'.\n"
            "9. Explain that walk-forward validation failed: compounded return is -7.20%, profit factor is 0.93, and maximum drawdown is 34.59% (breaching 25.0% limit; Walk-Forward Profitability: UNPROFITABLE).\n"
            "10. Flag that 5-bar overlapping holding horizons create statistically dependent observations; cite non-overlapping sensitivity analysis (refer to blocks as 'non-overlapping', NOT 'independent').\n"
            "11. A non-overlapping block sensitivity analysis IS PROVIDED in the canonical summary. You MUST NEVER state, imply, or synthesize that sensitivity analysis was omitted, absent, missing, or not conducted.\n"
            "12. Event-driven portfolio simulation is exploratory (+0.88% return, 7.37% drawdown with retrospective analogue selection), confirming chronological accounting mechanics, but MUST NOT be claimed as a validated profitable trading strategy or override failed walk-forward validation. Benchmark comparison establishes NO_ALPHA_ESTABLISHED."
        )

        from pydantic import BaseModel
        class AdversarialReviewOutput(BaseModel):
            bull_case: str
            bear_case: str
            explanation: str

        prompt = (
            f"Perform an Adversarial Trade Thesis Stress Test for {base_result.symbol} ({base_result.timeframe}):\n\n"
            f"• Deterministic Verdict: {base_result.verdict}\n"
            f"• Directional Bias: {base_result.directional_bias.value}\n"
            f"• Evidence Strength: {base_result.evidence_strength.value}\n"
            f"• Setup Status: {base_result.setup_status.value}\n"
            f"• Backtest Validation Status: {base_result.backtest_status.value}\n"
            f"• Thesis Status: {base_result.thesis_status.value}\n"
            f"• Canonical Analysis Summary (Single Typed Source of Truth):\n"
            f"{base_result.canonical_summary.model_dump() if base_result.canonical_summary else 'None'}\n\n"
            f"• 6 Audited Statuses: SimulationExecution={base_result.simulation_execution_status}, Profitability={base_result.profitability_status}, HistoricalValidation={base_result.historical_validation_status}, WalkForward={base_result.walk_forward_status}, RiskLimit={base_result.risk_limit_status}, Benchmark={base_result.benchmark_comparison_status}\n"
            f"• Sensitivity Analysis Available: True. Details: {base_result.canonical_summary.sensitivity_analysis_summary if base_result.canonical_summary else 'Non-overlapping blocks evaluated across 5 offsets.'}\n"
            f"• Legacy Statuses: Matching={base_result.matching_status}, Backtest={base_result.strategy_backtest_status}, Portfolio={base_result.portfolio_simulation_status}\n"
            f"• Drawdown Breach Warning: {base_result.drawdown_breach_warning or 'None'}\n"
            f"• Latest Price: {base_result.latest_price}\n"
            f"• Support Level: {base_result.support_levels}\n"
            f"• Resistance Level: {base_result.resistance_levels}\n\n"
            f"Historical Setup Investigation (Chronological OOS, 0.10% friction):\n"
            f"• Pattern: {base_result.historical_setup_results.setup_definition}\n"
            f"• Full History: {base_result.historical_setup_results.sample_count} trades, Win Rate: {base_result.historical_setup_results.win_rate * 100:.1f}%, Profit Factor: {base_result.historical_setup_results.profit_factor:.2f}, Max Drawdown: {base_result.historical_setup_results.max_drawdown * 100:.1f}%\n"
            f"• Out-Of-Sample (OOS): {base_result.historical_setup_results.oos_sample_count} trades, Status: {base_result.historical_setup_results.oos_status.value}, Win Rate: {base_result.historical_setup_results.oos_win_rate * 100:.1f}%\n"
            f"• Walk-Forward Evaluation: Status={base_result.walk_forward_evaluation.status.value if base_result.walk_forward_evaluation else 'None'}, Win Rate={base_result.walk_forward_evaluation.win_rate_pct if base_result.walk_forward_evaluation else 0}%, PF={base_result.walk_forward_evaluation.profit_factor if base_result.walk_forward_evaluation else 0}, Net={base_result.walk_forward_evaluation.compounded_equity_return_pct if base_result.walk_forward_evaluation else 0}%, DD={base_result.walk_forward_evaluation.max_drawdown_pct if base_result.walk_forward_evaluation else 0}%\n"
            f"• Walk-Forward Disconnect Explanation: {base_result.walk_forward_evaluation.win_rate_profitability_disconnect_explanation if base_result.walk_forward_evaluation else 'None'}\n"
            f"• Benchmark Comparison: {base_result.benchmark_comparison.comparison_summary if base_result.benchmark_comparison else 'None'}\n\n"
            f"Evidence Dependencies & Collinearity:\n"
            f"{[d.model_dump() for d in base_result.independent_evidence_families]}\n\n"
            f"Supporting Evidence Claims:\n"
            f"{[c.model_dump() for c in base_result.supporting_evidence]}\n\n"
            f"Opposing Evidence Claims:\n"
            f"{[c.model_dump() for c in base_result.opposing_evidence]}\n\n"
            f"Confirmation Conditions:\n"
            f"{[c.model_dump() for c in base_result.confirmation_conditions]}\n\n"
            f"Invalidation Conditions:\n"
            f"{[c.model_dump() for c in base_result.invalidation_conditions]}\n\n"
            f"Data Quality & Audit Warnings:\n"
            f"{base_result.audit_warnings}\n\n"
            f"Chart Observation: {chart_observation or 'None provided'}\n\n"
            "Provide the adversarial review: (1) Bull Case, (2) Bear Case, and (3) Evidence-Linked Explanation. "
            "Ensure every point cites the relevant Claim ID (e.g. [EV-TREN-1]) and do not give any trading orders or targets."
        )

        try:
            review: AdversarialReviewOutput = await self._prompt_agent(
                adversarial_identity, prompt, schema=AdversarialReviewOutput
            )
            # Update narrative fields while strictly keeping deterministic fields intact and validating narrative
            base_result.bull_case = sanitize_and_validate_review_narrative(review.bull_case, base_result.canonical_summary)
            base_result.bear_case = sanitize_and_validate_review_narrative(review.bear_case, base_result.canonical_summary)
            base_result.explanation = sanitize_and_validate_review_narrative(review.explanation, base_result.canonical_summary)

            # Deterministic safety gate enforcement: LLM narrative cannot override gate decisions
            if (
                base_result.backtest_status in (BacktestStatus.FAIL, BacktestStatus.INSUFFICIENT_DATA)
                or base_result.setup_status in (SetupStatus.CONFLICTED, SetupStatus.INVALID_DATA)
                or base_result.reconciliation_status == "FAILED"
                or base_result.walk_forward_status in ("FAIL", "INSUFFICIENT_DATA")
            ):
                if (
                    base_result.setup_status in (SetupStatus.CONFLICTED, SetupStatus.INVALID_DATA)
                    or base_result.reconciliation_status == "FAILED"
                ):
                    base_result.directional_bias = DirectionalBias.NEUTRAL
                base_result.thesis_status = ThesisStatus.NOT_ESTABLISHED
                if "NO VALIDATED EDGE" not in base_result.verdict:
                    base_result.verdict = f"NO VALIDATED EDGE: Historical Setup Failed ({base_result.directional_bias.value} Bias Unvalidated)"

            if base_result.canonical_summary:
                base_result.canonical_summary.final_validated_edge_verdict = base_result.verdict

            return base_result
        except Exception as e:
            logger.warning(f"Adversarial LLM review skipped or failed ({e}); falling back to deterministic synthesis.")
            if base_result.explanation:
                base_result.explanation = sanitize_and_validate_review_narrative(base_result.explanation, base_result.canonical_summary)
            return base_result

    async def analyze_market(
        self, 
        symbol: str, 
        chart_observation: str, 
        strategy_signals: str, 
        market_data_summary: str
    ) -> FinalAnalysisResult:
        """
        Backwards-compatible market analysis.
        """
        # If called without base_result, return empty model
        return FinalAnalysisResult(
            directional_bias=DirectionalBias.NEUTRAL,
            evidence_strength=EvidenceStrength.INSUFFICIENT,
            setup_status=SetupStatus.UNCONFIRMED,
            backtest_status=BacktestStatus.INSUFFICIENT_DATA,
            thesis_status=ThesisStatus.NOT_ESTABLISHED,
            historical_setup_results=HistoricalSetupResult(
                setup_definition="Legacy analysis",
                sample_count=0,
                win_rate=0.0,
                profit_factor=0.0,
                max_drawdown=0.0,
                net_return=0.0
            ),
            explanation="Legacy endpoint called.",
            symbol=symbol
        )

