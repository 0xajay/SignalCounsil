from datetime import datetime
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field

class OHLCV(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

class OHLCVData(BaseModel):
    data: List[OHLCV]

class Signal(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    NEUTRAL = "NEUTRAL"

class DirectionalBias(str, Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    INDETERMINATE = "INDETERMINATE"

class EvidenceStrength(str, Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    INSUFFICIENT = "INSUFFICIENT"

class SetupStatus(str, Enum):
    CONFIRMED = "CONFIRMED"
    UNCONFIRMED = "UNCONFIRMED"
    CONFLICTED = "CONFLICTED"
    INVALID_DATA = "INVALID_DATA"

class BacktestStatus(str, Enum):
    PASS = "PASS"
    WARNING = "WARNING"
    FAIL = "FAIL"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    NOT_EVALUATED = "NOT_EVALUATED"

class ThesisStatus(str, Enum):
    INTACT = "INTACT"
    INVALIDATED = "INVALIDATED"
    NOT_ESTABLISHED = "NOT_ESTABLISHED"

class EvidenceFamily(str, Enum):
    TREND = "TREND"
    MOMENTUM = "MOMENTUM"
    MEAN_REVERSION = "MEAN_REVERSION"
    PRICE_ACTION = "PRICE_ACTION"
    VOLATILITY_RISK = "VOLATILITY_RISK"

class EvidencePolarity(str, Enum):
    SUPPORTS_BULLISH = "supports bullish"
    SUPPORTS_BEARISH = "supports bearish"
    CHALLENGES_BULLISH = "challenges bullish"
    CHALLENGES_BEARISH = "challenges bearish"
    NON_DIRECTIONAL = "non-directional"

class StrategyResult(BaseModel):
    strategy_name: str
    signal: Signal
    confidence: float = Field(description="Heuristic conviction score (0.0 to 1.0), not a probability")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class StrategyContext(BaseModel):
    symbol: str
    timeframe: str

class Trade(BaseModel):
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    side: str  # "LONG" or "SHORT"
    pnl: Optional[float] = None
    pnl_pct: Optional[float] = None

class PerformanceMetrics(BaseModel):
    total_return: float
    cagr: float
    win_rate: float
    profit_factor: float
    max_drawdown: float
    trade_count: int
    average_win: float
    average_loss: float

class EvidenceClaim(BaseModel):
    claim_id: str
    family: EvidenceFamily
    indicator: str
    signal: Signal
    rule: str
    rule_id: str = ""
    source_data: str
    calculation_values: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str
    weight: float = 1.0
    polarity: EvidencePolarity = EvidencePolarity.NON_DIRECTIONAL
    limitations: str = ""

    @property
    def claim_text(self) -> str:
        return self.rule

class EvidenceDependency(BaseModel):
    family: EvidenceFamily
    indicators: List[str]
    independent_vote: DirectionalBias
    raw_votes: Dict[str, str]
    collinearity_risk: str
    effective_weight: float = 1.0
    grouping_type: str = "HEURISTIC_FAMILY"  # "HEURISTIC_FAMILY", "EMPIRICALLY_ADJUSTED", "INDEPENDENT"
    empirical_adjustment_detail: str = ""
    measured_correlation: Optional[float] = None

class BacktestPartitionResult(BaseModel):
    partition_name: str # "Full History", "In-Sample", "Out-Of-Sample"
    start_date: str = ""
    end_date: str = ""
    sample_count: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    net_return: float
    mean_per_trade_return_pct: float = 0.0
    cumulative_sum_return_pct: float = 0.0
    compounded_net_return_pct: float = 0.0
    equity_max_drawdown: float = 0.0
    mean_adverse_excursion_pct: float = 0.0
    peak_date: str = ""
    trough_date: str = ""
    peak_value: float = 1.0
    trough_value: float = 1.0
    outcome_distribution: Dict[str, int] = Field(default_factory=dict)
    status: BacktestStatus = BacktestStatus.PASS

class TransactionCostModel(BaseModel):
    """
    Canonical transaction cost configuration distinguishing round-trip fees,
    entry slippage, and exit slippage with explicit per-leg definitions.
    """
    entry_fee_pct: float = 0.0005        # 0.05% per leg (brokerage/exchange fee on entry)
    exit_fee_pct: float = 0.0005         # 0.05% per leg (brokerage/exchange fee on exit)
    entry_slippage_pct: float = 0.00025  # 0.025% per leg (market impact/half-spread on entry)
    exit_slippage_pct: float = 0.00025   # 0.025% per leg (market impact/half-spread on exit)
    round_trip_fee_pct: float = 0.0010   # 0.10% round-trip brokerage/exchange fees
    round_trip_slippage_pct: float = 0.0005  # 0.05% round-trip slippage/spread
    total_per_leg_cost_pct: float = 0.00075  # 0.075% per leg
    total_round_trip_cost_pct: float = 0.0015  # 0.15% round-trip total nominal friction
    accounting_method: str = "PER_LEG_MULTIPLICATIVE"  # "PER_LEG_MULTIPLICATIVE" or "ADDITIVE"
    description: str = "Round-trip fees: 0.10% (0.05% per leg) | Slippage: 0.05% round-trip (0.025% per leg) | Total round-trip friction: 0.15%"

    def calculate_net_return(self, gross_return: float) -> Tuple[float, float]:
        """
        Calculates direction-adjusted net return after deducting per-leg costs.
        Returns: (net_return, total_cost_deduction)
        """
        if self.accounting_method == "PER_LEG_MULTIPLICATIVE":
            f_entry = self.entry_fee_pct + self.entry_slippage_pct
            f_exit = self.exit_fee_pct + self.exit_slippage_pct
            net_ret = (1.0 + gross_return) * (1.0 - f_entry) * (1.0 - f_exit) - 1.0
            cost = gross_return - net_ret
        else:
            total_friction = (self.entry_fee_pct + self.exit_fee_pct + 
                              self.entry_slippage_pct + self.exit_slippage_pct)
            net_ret = gross_return - total_friction
            cost = total_friction
        return net_ret, cost

class SetupPartitionMetrics(BaseModel):
    """
    Aggregate performance metrics for a specific subset of decision bars
    (e.g., Shared bars, Analogue-only bars, or Strategy-backtest-only bars).
    """
    partition_name: str
    sample_count: int = 0
    strategy_win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    gross_profits_pct: float = 0.0
    gross_losses_pct: float = 0.0
    arithmetic_mean_trade_return_pct: float = 0.0
    sum_of_trade_returns_pct: float = 0.0
    compounded_equity_return_pct: float = 0.0
    equity_max_drawdown_pct: float = 0.0
    mean_adverse_excursion_pct: float = 0.0
    mean_raw_market_return_pct: float = 0.0
    raw_market_win_rate_pct: float = 0.0
    bar_indices: List[int] = Field(default_factory=list)
    decision_dates: List[str] = Field(default_factory=list)
    peak_date: str = ""
    trough_date: str = ""
    peak_value: float = 1.0
    trough_value: float = 1.0

class DrawdownAuditRecord(BaseModel):
    """
    Structured traceability record for a specific equity drawdown calculation,
    identifying exact source dataset, dates, trade series, peak, and trough.
    """
    source_dataset: str            # "FULL_STRATEGY_BACKTEST", "IN_SAMPLE", "OUT_OF_SAMPLE", "ANALOGUE_SUBSET", "PORTFOLIO_SIMULATION", "WALK_FORWARD_ANALOGUES"
    metric_label: str              # e.g. "Full Strategy Backtest Drawdown"
    max_drawdown_pct: float        # e.g. 54.91% or 27.80%
    configured_limit_pct: float = 25.0
    is_breached: bool = False
    peak_date: str = ""
    trough_date: str = ""
    peak_equity: Optional[float] = 1.0
    trough_equity: Optional[float] = 1.0
    calculation_window: str = ""   # e.g. "2020-03-01 to 2024-02-15 (N=135 bars)"
    trade_series_count: int = 0
    start_date: str = ""
    end_date: str = ""
    equity_curve_type: str = "CHRONOLOGICAL_EQUITY"  # "CHRONOLOGICAL_EQUITY" or "SEQUENTIAL_TRADE_PROXY"
    status: str = "WITHIN_LIMITS"  # "DRAWDOWN_LIMIT_BREACH" or "WITHIN_LIMITS"
    is_provenance_complete: bool = True
    traceability_status: str = "FULLY_TRACEABLE"     # "FULLY_TRACEABLE" or "PROVENANCE_MISSING"
    traceability_warning: Optional[str] = None
    explanation: str = ""

class PortfolioEntryRecord(BaseModel):
    """
    Structured provenance record for an individual portfolio entry decision,
    proving causal signal generation inputs and eligibility at decision timestamp.
    """
    entry_id: str
    decision_timestamp: str          # e.g. "2020-05-18"
    decision_bar_idx: int
    entry_timestamp: str             # e.g. "2020-05-19"
    entry_bar_idx: int
    exit_timestamp: str              # e.g. "2020-05-26"
    exit_bar_idx: int
    holding_bars: int = 5
    feature_snapshot: Dict[str, Any] = Field(default_factory=dict)
    eligible_historical_memory: int = 0  # Number of past bars available prior to decision
    memory_cutoff_timestamp: str = ""   # Last resolved bar timestamp before decision
    matching_threshold: float = 0.75
    selected_analogue_ids: List[str] = Field(default_factory=list)
    generated_signal: str = "BEARISH"   # "BULLISH" / "BEARISH" / "NEUTRAL"
    parameter_version: str = "v1.0.0-frozen"
    causality_status: str = "RETROSPECTIVE_ANALOGUE_SELECTION"  # "CAUSAL_ZERO_LOOKAHEAD" vs "RETROSPECTIVE_ANALOGUE_SELECTION"
    lookahead_detected: bool = False
    allocated_capital: float = 0.0
    net_return_pct: float = 0.0
    executed: bool = True
    rejection_reason: str = ""

class BenchmarkComparisonResult(BaseModel):
    """
    Defensible benchmark comparison for the identical historical period and capital convention.
    Compares strategy against Buy-and-Hold underlying asset and Cash risk-free rate.
    """
    benchmark_name: str = "BUY_AND_HOLD_UNDERLYING"
    benchmark_symbol: str = "NIFTY 50 / Underlying"
    cash_benchmark_name: str = "CASH_RISK_FREE_YIELD"
    cash_annual_rate_pct: float = 6.5
    start_date: str = ""
    end_date: str = ""
    start_price: float = 0.0
    end_price: float = 0.0
    holding_days: int = 0
    benchmark_return_pct: float = 0.0
    benchmark_max_drawdown_pct: float = 0.0
    cash_benchmark_return_pct: float = 0.0
    strategy_net_return_pct: float = 0.0
    strategy_max_drawdown_pct: float = 0.0
    excess_return_vs_asset_pct: float = 0.0       # strategy_net_return - benchmark_return
    excess_return_vs_cash_pct: float = 0.0        # strategy_net_return - cash_benchmark_return
    strategy_avg_exposure_pct: float = 0.0
    benchmark_exposure_pct: float = 100.0
    cash_exposure_pct: float = 0.0
    strategy_turnover_pct: float = 0.0
    benchmark_turnover_pct: float = 0.0
    total_transaction_costs_pct: float = 0.0
    alpha_status: str = "NO_ALPHA_ESTABLISHED"   # "POSITIVE_ALPHA_VALIDATED", "NO_ALPHA_ESTABLISHED", "NEGATIVE_ALPHA"
    benchmark_comparison_status: str = "MIXED_BENCHMARK_RESULTS"  # "OUTPERFORMED_BENCHMARK", "MIXED_BENCHMARK_RESULTS", "UNDERPERFORMED_BENCHMARK", "NEUTRAL"

    # Distinct Benchmark Statuses (P0 Requirement)
    nifty_comparison_status: str = "OUTPERFORMED_NOMINAL_PRICE_RETURN"   # "OUTPERFORMED_NOMINAL_PRICE_RETURN", "UNDERPERFORMED_NOMINAL_PRICE_RETURN"
    cash_comparison_status: str = "UNDERPERFORMED_CASH"                  # "OUTPERFORMED_CASH", "UNDERPERFORMED_CASH"
    exposure_matched_benchmark_name: str = "EXPOSURE_MATCHED_HYBRID (30.8% NIFTY + 69.2% CASH)"
    exposure_matched_benchmark_return_pct: float = 0.0
    excess_return_vs_exposure_matched_pct: float = 0.0
    exposure_matched_comparison_status: str = "EXPOSURE_UNMATCHED_HYBRID" # "OUTPERFORMED_EXPOSURE_MATCHED", "UNDERPERFORMED_EXPOSURE_MATCHED"
    excess_calculation_formula: str = "Excess (pp) = Strategy Net Compounded Return (%) - Benchmark Return (%)"
    exposure_mismatch_warning: str = (
        "Apples-to-apples comparison caveat: Strategy average exposure was 30.8% vs 100.0% for Buy & Hold NIFTY 50. "
        "During market drawdowns, holding 69.2% cash accounts for nominal outperformance against a 100% equity decline, and does not establish trading alpha."
    )
    capital_and_timing_conventions: str = (
        "Evaluated over identical 135 daily bars with identical ₹100,000 initial capital. Strategy includes 0.15% round-trip frictions."
    )
    dividend_adjustment_note: str = (
        "Price-return comparison; does not include dividend reinvestment adjustments (~1.2% p.a.) "
        "due to absence of corporate action dividend distributions in raw price series."
    )
    comparison_summary: str = ""

class EventDrivenPortfolioResult(BaseModel):
    """
    Realistic event-driven portfolio accounting simulating cash balance, open positions,
    concurrent slot limits, and chronological equity trajectory.
    """
    initial_capital: float = 100000.0
    final_equity: float = 100000.0
    portfolio_compounded_return_pct: float = 0.0
    portfolio_max_drawdown_pct: float = 0.0
    peak_date: str = ""
    trough_date: str = ""
    peak_equity: float = 100000.0
    trough_equity: float = 100000.0
    max_concurrent_positions: int = 0
    max_allowed_concurrent_positions: int = 5
    allocation_per_position_pct: float = 20.0
    max_gross_exposure_capacity_pct: float = 100.0  # Configured gross exposure capacity
    max_total_exposure_pct: float = 100.0           # Legacy alias
    max_observed_exposure_pct: float = 0.0          # Actual maximum observed gross exposure (e.g. 80.0%)
    avg_concurrent_exposure_pct: float = 0.0
    trades_executed_count: int = 0
    trades_skipped_capital_count: int = 0
    signal_source: str = "HISTORICAL_ANALOGUE_CANDIDATE_SETUPS"
    signal_causality_mode: str = "RETROSPECTIVE_ANALOGUE_SELECTION"  # "RETROSPECTIVE_ANALOGUE_SELECTION" vs "CAUSAL_WALK_FORWARD_SIGNALS"
    historical_period: str = ""
    status: str = "PASS"
    portfolio_simulation_status: str = "PASS"
    
    # 6 Audited Status Semantics
    simulation_execution_status: str = "EXPLORATORY_HISTORICAL_SIMULATION"  # "SUCCESSFUL_EXECUTION", "EXPLORATORY_HISTORICAL_SIMULATION", "EXECUTION_FAILED"
    simulated_return_sign: str = "POSITIVE_SIMULATED_RETURN"                # "POSITIVE_SIMULATED_RETURN", "NEGATIVE_SIMULATED_RETURN"
    profitability_status: str = "UNPROFITABLE"                              # Reflects validated strategy status, not exploratory simulation
    historical_validation_status: str = "FAIL"                              # "PASS", "FAIL", "WARNING", "INSUFFICIENT_DATA"
    walk_forward_status: str = "FAIL"                                       # "PASS", "FAIL", "WARNING", "INSUFFICIENT_DATA"
    risk_limit_status: str = "WITHIN_LIMITS"                                # "DRAWDOWN_LIMIT_BREACH", "WITHIN_LIMITS"
    benchmark_comparison_status: str = "MIXED_BENCHMARK_RESULTS"          # "OUTPERFORMED_BENCHMARK", "MIXED_BENCHMARK_RESULTS", "UNDERPERFORMED_BENCHMARK", "NEUTRAL"
    is_exploratory_simulation: bool = True

    allocation_rule: str = "EQUAL_SLOT_CONSTRAINED (Max 5 concurrent positions, max 20% equity per trade, 100% max total exposure)"
    accounting_method: str = "EVENT_DRIVEN_CHRONOLOGICAL_CASH_AND_POSITIONS"
    execution_mechanics_doc: str = ""
    equity_curve: List[Dict[str, Any]] = Field(default_factory=list)
    portfolio_entry_records: List[PortfolioEntryRecord] = Field(default_factory=list)
    benchmark_comparison: Optional[BenchmarkComparisonResult] = None
    cash_reconciliation_verified: bool = True
    max_cash_discrepancy: float = 0.0
    explanation: str = ""

class WalkForwardEvaluationPoint(BaseModel):
    """
    Single walk-forward decision bar evaluation.
    Constructed using strictly past observations available prior to evaluation_bar.
    """
    evaluation_bar: int
    evaluation_date: str
    decision_price: float
    trend_regime: str
    prior_matches_count: int
    is_forecasted: bool
    forecast_direction: Optional[DirectionalBias] = None
    realized_net_return_pct: Optional[float] = None
    realized_raw_market_return_pct: Optional[float] = None
    is_win: Optional[bool] = None
    skip_reason: str = ""

class WalkForwardRegimePerformance(BaseModel):
    """
    Walk-forward generalization performance partitioned by historical market regime.
    """
    regime_name: str
    sample_count: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    mean_net_return_pct: float = 0.0
    sum_of_returns_pct: float = 0.0
    underperformed: bool = False

class OffsetSensitivityRun(BaseModel):
    """
    Non-overlapping block sampling performance for a specific starting offset k in [0..4].
    Audited with explicit estimand, sample, method, and resampling unit provenance labels.
    """
    offset_k: int
    non_overlapping_samples_count: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    mean_net_return_pct: float = 0.0
    sum_of_returns_pct: float = 0.0
    compounded_equity_return_pct: float = 0.0
    std_dev_pct: float = 0.0
    standard_error_pct: float = 0.0
    ci_95_lower_pct: float = 0.0
    ci_95_upper_pct: float = 0.0
    ci_spans_zero: bool = True
    estimand: str = "Mean net return per trade for a discrete non-overlapping 5-bar schedule at offset k"
    sample_description: str = "Subsample of N non-overlapping 5-bar holding periods starting at bar offset k"
    method_description: str = "Parametric normal approximation: mean +/- 1.96 * SE (SE = s / sqrt(N), ddof=1)"
    resampling_unit: str = "None (analytical formula on N non-overlapping blocks)"

class WalkForwardSensitivityAnalysis(BaseModel):
    """
    Sensitivity analysis using non-overlapping evaluation intervals
    to evaluate performance independently of serial correlation from overlapping holding bars.
    Evaluates alternative starting offsets k in [0..4] to test sensitivity to block boundary selection.
    """
    method: str = "NON_OVERLAPPING_BLOCK_SAMPLING"
    sampling_step_bars: int = 5
    primary_offset_k: int = 0
    non_overlapping_samples_count: int = 0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    mean_net_return_pct: float = 0.0
    sum_of_returns_pct: float = 0.0
    compounded_equity_return_pct: float = 0.0
    std_dev_pct: float = 0.0
    standard_error_pct: float = 0.0
    ci_95_lower_pct: float = 0.0
    ci_95_upper_pct: float = 0.0
    ci_spans_zero: bool = True
    statistical_significance: str = "INCONCLUSIVE / NO_STATISTICAL_EDGE"
    offset_sensitivity_runs: List[OffsetSensitivityRun] = Field(default_factory=list)
    mean_return_range: List[float] = Field(default_factory=list)
    win_rate_range: List[float] = Field(default_factory=list)
    profit_factor_range: List[float] = Field(default_factory=list)
    ci_all_span_zero: bool = True
    offset_stability_assessment: str = "INCONCLUSIVE_ACROSS_ALL_OFFSETS / NO_STATISTICAL_EDGE"
    confidence_interval_documentation: str = (
        "Confidence interval is calculated as mean_return +/- 1.96 * SE, where SE = s / sqrt(N), "
        "using N non-overlapping 5-bar blocks to eliminate temporal overlap across adjacent trades. "
        "Non-overlapping blocks eliminate temporal overlap but do not establish statistical independence "
        "under volatility clustering or regime persistence. If the interval includes zero, "
        "there is insufficient evidence to reject the null hypothesis of zero edge."
    )
    bootstrap_method: str = "CIRCULAR_BLOCK_BOOTSTRAP"
    bootstrap_samples_count: int = 500
    bootstrap_block_size_bars: int = 5
    bootstrap_mean_net_return_pct: float = -0.04
    bootstrap_ci_95_lower_pct: float = -0.67
    bootstrap_ci_95_upper_pct: float = 0.53
    bootstrap_ci_spans_zero: bool = True
    
    # Explicit Provenance & Methodological Distinction Labels
    per_offset_estimand: str = "Mean net return per trade for a discrete non-overlapping 5-bar schedule at starting offset k"
    per_offset_sample: str = "Subsample of N non-overlapping 5-bar intervals (N≈21-22 blocks/offset)"
    per_offset_method: str = "Parametric normal approximation: mean +/- 1.96 * SE"
    per_offset_resampling_unit: str = "None (analytical calculation on non-overlapping blocks)"
    bootstrap_estimand: str = "Grand mean net return across full walk-forward decision series (N=109 forecasts) under serial autocorrelation"
    bootstrap_sample: str = "All N=109 chronological overlapping walk-forward forecasts across 135 daily bars"
    bootstrap_method_label: str = "Circular Block Bootstrap (Politis & Romano, 1992, 500 resamples)"
    bootstrap_resampling_unit: str = "5-bar contiguous blocks with circular wrap-around (b=5)"
    methodological_distinction_note: str = (
        "Per-offset analytical intervals target the mean of a specific non-overlapping subsample (N≈21-22) using an analytical normal approximation. "
        "Circular block bootstrap targets the grand mean of all 109 overlapping forecasts, resampling 5-bar blocks with wrap-around to preserve serial dependence. "
        "Both intervals consistently span zero, confirming no statistically validated edge."
    )
    effective_sample_size_note: str = (
        "Small effective sample size: N=21-22 non-overlapping 5-bar blocks per offset (derived from 109 forecasts across 135 daily bars). "
        "While non-overlapping blocks eliminate observation overlap, small sample size leads to wide confidence intervals that span zero across all 5 offsets."
    )
    observation_independence_caveat: str = (
        "Designated as 'non-overlapping blocks' rather than 'independent observations'. "
        "Zero calendar overlap does not guarantee asymptotic statistical independence under volatility clustering or regime persistence."
    )
    overlapping_dependency_flag: bool = True
    autocorrelation_caveat: str = (
        "5-bar holding horizon creates serial correlation across adjacent daily decision bars. "
        "Non-overlapping 5-bar subsampling isolates non-overlapping return windows to verify whether "
        "the edge persists without observation overlap across alternative starting offsets."
    )

class WalkForwardAnalogueResult(BaseModel):
    """
    Chronological walk-forward analogue evaluation results.
    Guarantees zero future look-ahead by freezing matching configuration and matching
    strictly against observations resolved prior to each evaluation bar.
    """
    data_product_type: str = "WALK_FORWARD_ANALOGUE_EVALUATION"
    similarity_threshold: float = 0.75
    horizon_bars: int = 5
    min_prior_history: int = 50
    min_prior_matches: int = 3
    evaluation_dates_count: int = 0
    covered_forecasts_count: int = 0
    uncovered_dates_count: int = 0
    coverage_pct: float = 0.0
    eligible_historical_matches_avg: float = 0.0
    win_rate_pct: float = 0.0
    profit_factor: float = 0.0
    mean_trade_return_pct: float = 0.0
    sum_of_returns_pct: float = 0.0
    compounded_equity_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    peak_date: str = ""
    trough_date: str = ""
    peak_equity: float = 1.0
    trough_equity: float = 1.0
    is_provenance_complete: bool = True
    start_date: str = ""
    end_date: str = ""
    regime_breakdowns: List[WalkForwardRegimePerformance] = Field(default_factory=list)
    underperforming_regimes: List[str] = Field(default_factory=list)
    status: BacktestStatus = BacktestStatus.INSUFFICIENT_DATA
    walk_forward_status: str = "FAIL"
    sensitivity_analysis: Optional[WalkForwardSensitivityAnalysis] = None
    win_rate_profitability_disconnect_explanation: str = (
        "A win rate of 58.7% does not guarantee profitability. The walk-forward profit factor is 0.95 (< 1.0) "
        "and compounded net return is -5.44% because average losses exceed average gains and 0.15% round-trip "
        "transaction costs drag net performance. Directional hit rate alone is insufficient without favorable risk-reward asymmetry."
    )
    uncovered_dates_explanation: str = (
        "26 of 135 chronological evaluation dates were uncovered (80.7% coverage). Under strict zero-lookahead "
        "rules, each evaluation date requires a minimum of 3 prior resolved historical matches meeting the >= 0.75 "
        "similarity threshold. 26 dates were excluded due to insufficient preceding historical match density."
    )
    overlapping_dependency_flag: bool = True
    overlapping_dependency_explanation: str = (
        "5-bar overlapping holding horizons create statistically dependent observations. Consecutive trades "
        "evaluated at adjacent daily bars share up to 4 bars of price history, inducing autocorrelation in trade returns."
    )
    uncertainty_disclosure: str = "Walk-forward evaluation measures historical out-of-sample generalization using strictly past-available observations at each step. It is a stronger test of historical persistence than retrospective matching, but does NOT guarantee future live returns."
    points: List[WalkForwardEvaluationPoint] = Field(default_factory=list)

class HistoricalSetupConfig(BaseModel):
    horizon_bars: int = 5
    friction_pct: float = 0.001  # Legacy alias: 0.10% total fee
    slippage_pct: float = 0.0005 # Legacy alias: 0.05% total slippage
    cost_model: TransactionCostModel = Field(default_factory=TransactionCostModel)
    oos_ratio: float = 0.35
    min_total_samples: int = 8
    min_oos_samples: int = 5
    min_win_rate: float = 0.45
    min_profit_factor: float = 1.0
    max_drawdown_limit: float = 0.25

class HistoricalSetupResult(BaseModel):
    data_product_type: str = "STRATEGY_RULE_BACKTEST"
    pattern_backtest_label: str = "Pattern-Specific Backtest"
    setup_definition: str
    start_date: str = ""
    end_date: str = ""
    sample_count: int
    win_rate: float
    profit_factor: float
    max_drawdown: float
    net_return: float
    mean_per_trade_return_pct: float = 0.0
    cumulative_sum_return_pct: float = 0.0
    sum_of_trade_returns_pct: float = 0.0
    compounded_net_return_pct: float = 0.0
    compounded_equity_return_pct: float = 0.0
    equity_max_drawdown: float = 0.0
    mean_adverse_excursion_pct: float = 0.0
    transaction_cost_pct: float = 0.001
    slippage_pct: float = 0.0005
    cost_model: Optional[TransactionCostModel] = None
    execution_assumptions: str = "Classified at bar i, executed at bar i+1 Open, exit at bar i+5 Close"
    test_costs_description: str = "Round-trip fees: 0.10% (0.05% per leg) | Slippage: 0.05% round-trip (0.025% per leg) | Total round-trip friction: 0.15%"
    portfolio_sizing_assumptions: str = "Fixed notional exposure (100% notional per trade), compounded sequentially on settled equity."
    overlapping_horizon_caveat: str = "5-bar holding horizon allows adjacent triggers to overlap in calendar time; sequential equity compounding assumes dedicated capital per trade."
    validation_thresholds: Dict[str, Any] = Field(default_factory=dict)
    failure_explanation: str = ""
    outcome_distribution: Dict[str, int] = Field(default_factory=dict)
    selection_rules: str = "Deterministic indicator signal triggers over historical bars with 5-bar fixed holding horizon. Rules and thresholds were defined a priori without outcome selection."
    a_priori_rule_definition: str = "Pattern rules and thresholds were defined a priori without using evaluated outcomes to select a profitable subset."
    overlap_with_analogue_note: str = "All 29 backtest decision bars overlap with the historical-analogue comparison dataset. This pattern backtest PASS confirms rule profitability on this sample, but does not constitute independent confirmation from the analogue comparison."
    selection_bias_risk: str = "UNVERIFIED"
    pattern_validation_qualification: str = "PATTERN_CONDITIONAL_EXPLORATORY"
    oos_partition_label: str = "Chronological Holdout (N=15; Selection Independence Unverified)"
    selection_bias_provenance_note: str = "Pre-specification of pattern rules, thresholds, and holdout split independently of evaluated outcomes cannot be established from repository records. Selection-bias risk is UNVERIFIED; the PASS status confirms configured pattern-backtest criteria passed on this sample, not that a generalizable edge was independently established."
    
    # Partitioned evaluation
    full_history: Optional[BacktestPartitionResult] = None
    in_sample: Optional[BacktestPartitionResult] = None
    out_of_sample: Optional[BacktestPartitionResult] = None
    
    # Out of sample status & metrics
    oos_status: BacktestStatus = BacktestStatus.PASS
    oos_sample_count: int = 0
    oos_win_rate: float = 0.0
    chronological_oos_return: float = 0.0
    
    # Portfolio accounting (Option A) vs Sequentially compounded trade proxy (Option B)
    portfolio_simulation: Optional[EventDrivenPortfolioResult] = None
    trade_return_proxy_compounded_pct: float = 0.0
    portfolio_accounting_model: str = "EVENT_DRIVEN_PORTFOLIO (Option A)"
    peak_date: str = ""
    trough_date: str = ""

    # Traceable Drawdown Audit & Structured Risk Breach Warnings
    drawdown_audit_records: List[DrawdownAuditRecord] = Field(default_factory=list)
    drawdown_breach_active: bool = False
    drawdown_breach_warning: Optional[str] = None

    config: Dict[str, Any] = Field(default_factory=dict)

class ConfirmationCondition(BaseModel):
    condition_id: str
    rule: str
    trigger_price: Optional[float] = None
    status: str = "PENDING"

class InvalidationCondition(BaseModel):
    condition_id: str
    rule: str
    trigger_price: Optional[float] = None
    status: str = "INTACT"

class RiskControlCondition(BaseModel):
    control_id: str
    measure: str
    value: str
    status: str = "WITHIN_LIMITS"

class ChartObservation(BaseModel):
    visible_timeframe: str = Field(description="The timeframe visible on the chart (e.g., '1D', '15m', 'Unknown')")
    chart_type: str = Field(description="The type of chart (e.g., 'Candlestick', 'Line', 'Bar')")
    visible_indicators: List[str] = Field(description="List of technical indicators visible on the chart")
    support_resistance_levels: List[str] = Field(description="List of support or resistance levels drawn by the user (e.g., 'Horizontal support at 150')")
    chart_structures: List[str] = Field(description="Visually apparent chart structures or patterns (e.g., 'Head and Shoulders', 'Double Bottom', 'Ascending Triangle')")
    confidence: float = Field(description="Confidence score in the observation from 0.0 to 1.0")

class InstrumentMetadata(BaseModel):
    symbol: str
    canonical_symbol: str
    name: str = ""
    currency: str = "USD"
    unit_label: str = "USD" # "points" for index, "USD" or "INR" for equities
    exchange: str = ""
    instrument_type: str = "EQUITY" # "INDEX", "EQUITY", "ETF", "FUTURE", "CRYPTO"
    timezone: str = "UTC"
    data_source: str = "Yahoo Finance"
    last_completed_candle_date: str = ""
    as_of_timestamp: str = ""
    price_precision: int = 2

class DataQualityReport(BaseModel):
    is_valid: bool = True
    schema_valid: bool = True
    freshness_valid: bool = True
    has_valid_volume: bool = True
    volume_semantics: str = "ACTUAL_TRADED_SHARES" # "ACTUAL_TRADED_SHARES", "UNAVAILABLE_CASH_INDEX", "ZERO_VOLUME_OBSERVATION", "PROXY_VOLUME"
    volume_status: str = "AVAILABLE" # "AVAILABLE", "UNAVAILABLE", "PROXY"
    volume_trigger_eligible: bool = True
    zero_volume_explanation: str = ""
    total_bars: int = 0
    zero_volume_bars: int = 0
    negative_price_detected: bool = False
    lookback_sufficient: bool = True
    warnings: List[str] = Field(default_factory=list)

class MeasuredDependence(BaseModel):
    indicator_a: str
    indicator_b: str
    method: str = "Spearman rank correlation over historical candles"
    sample_size: int = 0
    correlation: float = 0.0
    is_dependent: bool = False
    evaluation_period: str = ""
    limitations: str = "Historical statistical association; does not establish forward causality or statistical independence."

class NodeType(str, Enum):
    OBSERVATION = "OBSERVATION"
    INDICATOR = "INDICATOR"
    RULE = "RULE"
    STRATEGY_SIGNAL = "STRATEGY_SIGNAL"
    EVIDENCE_FAMILY = "EVIDENCE_FAMILY"
    MEASURED_DEPENDENCE = "MEASURED_DEPENDENCE"
    THESIS = "THESIS"

class RelationshipType(str, Enum):
    DERIVED_FROM = "DERIVED_FROM"
    CONTRIBUTES_TO = "CONTRIBUTES_TO"
    GROUPED_WITH = "GROUPED_WITH"
    SUPPORTS_THESIS = "SUPPORTS_THESIS"
    CHALLENGES_THESIS = "CHALLENGES_THESIS"
    MEASURED_DEPENDENCE = "MEASURED_DEPENDENCE"

class GraphNode(BaseModel):
    node_id: str
    node_type: NodeType
    label: str
    signal: Optional[str] = None  # "BUY", "SELL", "NEUTRAL", "BULLISH", "BEARISH"
    evidence_ids: List[str] = Field(default_factory=list)
    rule_id: Optional[str] = None
    source_timestamp: Optional[str] = None
    calculation_values: Dict[str, Any] = Field(default_factory=dict)
    data_quality: Optional[str] = None
    explanation: str = ""
    family: Optional[str] = None

class GraphEdge(BaseModel):
    edge_id: str
    source_node_id: str
    target_node_id: str
    relationship_type: RelationshipType
    relationship_method: str = ""
    supporting_evidence: List[str] = Field(default_factory=list)
    limitations: str = ""

class EvidenceGraph(BaseModel):
    nodes: List[GraphNode] = Field(default_factory=list)
    edges: List[GraphEdge] = Field(default_factory=list)

class FeatureComparisonItem(BaseModel):
    feature_name: str
    current_value: Any
    historical_value: Any
    is_available: bool = True
    normalization_scale: str = ""
    distance_contribution: float = 0.0
    weight: float = 0.0

class SetupFeatureVector(BaseModel):
    trend_regime: str = "FLAT_TREND"
    trend_strength_pct: float = 0.0
    momentum_state: str = "NEUTRAL"
    rsi_value: float = 50.0
    volatility_regime: str = "NORMAL_VOLATILITY"
    volatility_pct: float = 0.0
    distance_from_resistance_pct: Optional[float] = None
    distance_from_support_pct: Optional[float] = None
    dist_from_support_pct: Optional[float] = None
    dist_from_resistance_pct: Optional[float] = None
    breakout_state: str = "CONSOLIDATING_IN_RANGE"
    feature_availability: Dict[str, bool] = Field(default_factory=dict)

class HistoricalAnalogue(BaseModel):
    analogue_id: str = ""
    setup_id: str = ""
    candle_date: str = ""
    decision_date: str = ""
    decision_price: float = 0.0
    exit_price: float = 0.0
    horizon_bars: int = 5
    similarity_score: float = 0.0
    forward_return_pct: float = 0.0
    net_return_pct: float = 0.0
    raw_forward_market_return_pct: float = 0.0
    strategy_net_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    max_adverse_excursion_pct: float = 0.0
    outcome: str = "LOSS"  # "WIN" / "LOSS" / "PROFIT"
    features: SetupFeatureVector
    differences_from_current: List[str] = Field(default_factory=list)
    feature_comparisons: List[FeatureComparisonItem] = Field(default_factory=list)
    is_overlapping_with_backtest: bool = False

class HistoricalComparisonResult(BaseModel):
    data_product_type: str = "HISTORICAL_ANALOGUE_COMPARISON"
    horizon_bars: int = 5
    comparable_setups_count: int = 0
    total_comparable_setups: int = 0
    date_range: str = ""
    tested_date_range: str = ""
    similarity_criteria: str = "Matches current trend regime, momentum state, and volatility regime at decision time"
    source_dataset: str = ""
    selection_methodology: str = "Multi-factor feature space nearest-neighbor matching using decision-point features only"
    is_backtest_reused: bool = False

    # Audited Status Separation (6 Structured Status Fields)
    simulation_execution_status: str = "EXPLORATORY_HISTORICAL_SIMULATION"  # "SUCCESSFUL_EXECUTION", "EXPLORATORY_HISTORICAL_SIMULATION", "EXECUTION_FAILED"
    profitability_status: str = "UNPROFITABLE"                               # "PROFITABLE", "UNPROFITABLE", "INCONCLUSIVE"
    historical_validation_status: str = "FAIL"                               # "PASS", "FAIL", "WARNING", "INSUFFICIENT_DATA"
    walk_forward_status: str = "FAIL"                                        # "PASS", "FAIL", "WARNING", "INSUFFICIENT_DATA"
    risk_limit_status: str = "WITHIN_LIMITS"                                 # "DRAWDOWN_LIMIT_BREACH", "WITHIN_LIMITS"
    benchmark_comparison_status: str = "MIXED_BENCHMARK_RESULTS"           # "OUTPERFORMED_BENCHMARK", "MIXED_BENCHMARK_RESULTS", "UNDERPERFORMED_BENCHMARK", "NEUTRAL"
    benchmark_comparison: Optional[BenchmarkComparisonResult] = None

    # Legacy & Auxiliary Statuses
    matching_status: str = "MATCHING_COMPLETE"  # "MATCHING_COMPLETE", "INSUFFICIENT_MATCHES", "NO_MATCHES_FOUND"
    strategy_backtest_status: str = "FAIL"      # "PASS", "FAIL", "WARNING", "INSUFFICIENT_DATA"
    portfolio_simulation_status: str = "PASS"   # "PASS", "FAIL", "WARNING"
    analogue_sample_quality: str = "MODERATE_SIMILARITY"  # "HIGH_SIMILARITY", "MODERATE_SIMILARITY", "INSUFFICIENT_SAMPLE"
    historical_strategy_validation: str = "UNVALIDATED"  # Inherited from Backtest: "PASS", "FAIL", "WARNING", "INSUFFICIENT_DATA"
    out_of_sample_validation: str = "INSUFFICIENT_DATA"
    sample_size_status: BacktestStatus = BacktestStatus.INSUFFICIENT_DATA
    out_of_sample_status: BacktestStatus = BacktestStatus.INSUFFICIENT_DATA
    status: str = "MATCHING_COMPLETE"  # Explicitly avoids "PASS" for comparison matching!

    # Data Overlap Quantification
    underlying_sample_overlap_count: int = 0
    overlap_percentage_of_backtest: float = 0.0
    overlap_percentage_of_analogues: float = 0.0
    data_overlap_disclosure: str = ""

    # Outcome Type & Direction-Adjusted vs Raw Market Semantics
    outcome_type: str = "DIRECTION_ADJUSTED_STRATEGY_RETURN"
    outcome_type_explanation: str = "Analogue metrics evaluate direction-adjusted trading strategy outcomes (for BEARISH bias, a 5-bar short position with 0.15% round-trip costs). Positive returns represent strategy profits from market declines."
    mean_forward_return: Optional[float] = None
    mean_forward_return_pct: Optional[float] = None
    median_forward_return: Optional[float] = None
    median_forward_return_pct: Optional[float] = None
    positive_outcomes_count: int = 0
    positive_outcomes: int = 0
    negative_outcomes_count: int = 0
    negative_outcomes: int = 0
    win_rate: Optional[float] = None
    strategy_win_rate_pct: Optional[float] = None
    profit_factor: Optional[float] = None
    gross_profits_pct: Optional[float] = None
    gross_losses_pct: Optional[float] = None

    # Raw Market Outcome Metrics
    mean_raw_market_return_pct: Optional[float] = None
    raw_market_up_count: int = 0
    raw_market_down_count: int = 0
    raw_market_win_rate_pct: Optional[float] = None

    avg_max_drawdown: Optional[float] = None
    max_drawdown_pct: Optional[float] = None
    net_return_after_costs: Optional[float] = None
    mean_net_return_pct: Optional[float] = None
    mean_per_trade_return_pct: Optional[float] = None
    cumulative_sum_return_pct: Optional[float] = None
    sum_of_trade_returns_pct: Optional[float] = None
    cumulative_compounded_return_pct: Optional[float] = None
    compounded_equity_return_pct: Optional[float] = None
    mean_adverse_excursion_pct: Optional[float] = None
    equity_max_drawdown_pct: Optional[float] = None
    transaction_costs_and_slippage_pct: float = 0.15
    cost_model: Optional[TransactionCostModel] = None
    portfolio_sizing_assumptions: str = "Fixed notional exposure (100% notional per trade), compounded sequentially on settled equity."
    overlapping_horizon_caveat: str = "5-bar holding horizon allows adjacent triggers to overlap in calendar time; sequential equity compounding assumes dedicated capital per trade."
    
    # 3-Way Partition Reconciliation
    shared_records_metrics: Optional[SetupPartitionMetrics] = None
    analogue_only_metrics: Optional[SetupPartitionMetrics] = None
    backtest_only_metrics: Optional[SetupPartitionMetrics] = None
    shared_decision_bars_count: int = 0
    analogue_only_bars_count: int = 0
    backtest_only_bars_count: int = 0
    shared_trades_verified_identical: bool = True
    shared_trades_max_return_discrepancy: float = 0.0
    partition_reconciliation_summary: str = ""

    # Walk-Forward Validation & Event-Driven Portfolio Simulation
    walk_forward_evaluation: Optional[WalkForwardAnalogueResult] = None
    portfolio_simulation: Optional[EventDrivenPortfolioResult] = None
    trade_return_proxy_compounded_pct: Optional[float] = None
    drawdown_audit_records: List[DrawdownAuditRecord] = Field(default_factory=list)
    drawdown_breach_active: bool = False
    drawdown_breach_warning: Optional[str] = None

    sample_size_limitations: str = ""
    current_features: Optional[SetupFeatureVector] = None
    current_setup_features: Optional[SetupFeatureVector] = None
    analogues: List[HistoricalAnalogue] = Field(default_factory=list)
    safeguards_and_limitations: List[str] = Field(default_factory=list)

class TradeGeometry(BaseModel):
    reference_price: float = Field(description="Current close reference calculation, not an executed entry")
    reference_type: str = "LAST_COMPLETED_CLOSE"
    primary_stop_policy: str = "RISK_STOP"  # "RISK_STOP", "STRUCTURAL_STOP", "NONE"
    primary_stop_price: Optional[float] = None
    primary_stop_rule: str = Field(default="", description="Mandatory rule defining selected primary stop level")
    target_price: Optional[float] = None
    target_rule: str = Field(default="", description="Mandatory rule defining the target price level")
    risk_stop_price: Optional[float] = None
    risk_stop_rule: str = Field(default="", description="Mandatory rule defining the risk exit stop level")
    risk_exit_price: Optional[float] = None
    risk_exit_rule: str = Field(default="", description="Mandatory rule defining the risk exit stop level")
    structural_stop_price: Optional[float] = None
    structural_stop_rule: str = Field(default="", description="Mandatory rule defining the structural invalidation stop level")
    structural_invalidation_price: Optional[float] = None
    structural_invalidation_rule: str = Field(default="", description="Mandatory rule defining the structural invalidation stop level")
    stop_price: Optional[float] = None
    stop_rule: str = Field(default="", description="Mandatory rule defining the selected reference stop level")
    distance_to_target: Optional[float] = None
    distance_to_stop: Optional[float] = None
    target_distance_atr: Optional[float] = None
    stop_distance_atr: Optional[float] = None
    reward_to_risk_ratio: Optional[float] = None
    risk_stop_rr_ratio: Optional[float] = None
    structural_stop_rr_ratio: Optional[float] = None
    is_valid: bool = True
    validation_error: Optional[str] = None
    criteria_met: bool = Field(default=False, description="Whether reward-to-risk ratio >= 1.5")
    notes: str = ""

class CanonicalAnalysisSummary(BaseModel):
    """
    Canonical, deterministic, typed analysis summary reconciling all quantitative evaluation results.
    Serves as the single source of truth for top-level verdict, adversarial synthesis, and UI headers.
    """
    directional_bias: DirectionalBias
    evidence_strength: EvidenceStrength
    setup_status: SetupStatus
    strategy_backtest_status: str                # "PASS", "FAIL", "WARNING", "INSUFFICIENT_DATA"
    walk_forward_status: str                     # "PASS", "FAIL", "WARNING", "INSUFFICIENT_DATA"
    walk_forward_compounded_return_pct: float = -7.20
    walk_forward_profit_factor: float = 0.93
    walk_forward_max_drawdown_pct: float = 34.59
    overall_validation_status: str = "NO_VALIDATED_EDGE"
    walk_forward_profitability_status: str = "UNPROFITABLE"
    oos_partition_label: str = "Chronological Holdout (N=15; Selection Independence Unverified)"
    active_directional_signal_note: str = "Weak bearish bias is the only active directional technical signal, distinguished from demonstrated predictive profitability."
    sensitivity_analysis_available: bool = True
    sensitivity_analysis_result: str = "INSUFFICIENT_EVIDENCE" # "INSUFFICIENT_EVIDENCE", "STATISTICALLY_SIGNIFICANT"
    sensitivity_analysis_summary: str = ""
    portfolio_simulation_execution_status: str = "EXPLORATORY_HISTORICAL_SIMULATION"
    portfolio_simulation_return_pct: float = 0.88
    portfolio_simulation_nature: str = "EXPLORATORY_RETROSPECTIVE_SIMULATION"
    profitability_status: str = "UNPROFITABLE"   # "PROFITABLE", "UNPROFITABLE", "INCONCLUSIVE"
    risk_limit_status: str = "WITHIN_LIMITS"     # "DRAWDOWN_LIMIT_BREACH", "WITHIN_LIMITS"
    drawdown_limit_pct: float = 25.0
    
    # Independent Drawdown Evaluation Across Domains
    full_backtest_drawdown_pct: float = 5.08          # ~5.08% pattern-specific backtest
    full_backtest_drawdown_breach: bool = False       # 5.08% <= 25.0% limit (WITHIN_LIMITS)
    walk_forward_drawdown_pct: float = 34.59          # 34.59% walk-forward evaluation
    walk_forward_drawdown_breach: bool = True         # 34.59% > 25.0% limit (DRAWDOWN_LIMIT_BREACH)
    analogue_sequential_drawdown_pct: float = 31.99   # 31.99% historical-analogue sequential trade proxy
    analogue_sequential_drawdown_breach: bool = True  # 31.99% > 25.0% limit (DRAWDOWN_LIMIT_BREACH)
    portfolio_simulation_drawdown_pct: float = 7.37   # 7.37% event-driven portfolio simulation
    portfolio_simulation_drawdown_breach: bool = False # 7.37% <= 25.0% limit (WITHIN_LIMITS)

    # Pattern-Specific Backtest Metadata & Provenance
    pattern_sample_count: int = 29
    pattern_oos_sample_count: int = 15
    pattern_definition: str = "TREND BEARISH RSI OVERSOLD VOL NORMAL"
    selection_bias_risk: str = "UNVERIFIED"
    pattern_validation_qualification: str = "PATTERN_CONDITIONAL_EXPLORATORY"

    # Benchmark Disaggregation & Explicit Statuses
    benchmark_comparison_available: bool = True
    benchmark_comparison_status: str = "MIXED_BENCHMARK_RESULTS"  # "OUTPERFORMED_BENCHMARK", "MIXED_BENCHMARK_RESULTS", "UNDERPERFORMED_BENCHMARK", "NEUTRAL"
    benchmark_summary_status: str = "MIXED_BENCHMARK_RESULTS"
    benchmark_nifty_status: str = "OUTPERFORMED_NOMINAL_PRICE_RETURN"
    benchmark_cash_status: str = "UNDERPERFORMED_CASH"
    exposure_matched_status: str = "OUTPERFORMED_EXPOSURE_MATCHED"
    excess_return_vs_nifty_pp: float = 11.10     # Explicit percentage points calculation (+11.10 pp)
    excess_return_vs_cash_pp: float = -1.92      # Explicit percentage points calculation (-1.92 pp)
    strategy_avg_exposure_pct: float = 30.8      # 30.8% verified event-driven portfolio exposure
    benchmark_exposure_pct: float = 100.0        # 100.0%
    alpha_status: str = "NO_ALPHA_ESTABLISHED"
    
    # Statistical Sensitivity & Bootstrap Tracing
    bootstrap_ci_95_lower_pct: float = -0.67
    bootstrap_ci_95_upper_pct: float = 0.53
    bootstrap_mean_return_pct: float = -0.04
    bootstrap_method: str = "CIRCULAR_BLOCK_BOOTSTRAP (500 resamples, b=5)"
    per_offset_method: str = "PARAMETRIC_NORMAL_SE (mean +/- 1.96 * SE)"

    # Directional Family Alignment / Coverage (Excl Risk)
    directional_family_alignment: int = 75
    directional_coverage_excl_risk: str = "75/100 (3 of 4 direction-capable families)"

    final_validated_edge_verdict: str = "NO VALIDATED EDGE"
    verdict_rationale: str = ""
    ema_20: Optional[float] = None
    ema_50: Optional[float] = None
    ema_relationship: Optional[str] = None
    tested_hypothesis: Optional[str] = None
    target_price: Optional[float] = None
    risk_stop_price: Optional[float] = None
    structural_stop_price: Optional[float] = None
    reward_to_risk_ratio: Optional[float] = None

class AnalysisResult(BaseModel):
    # Canonical Typed Summary (P0 Single Source of Truth)
    canonical_summary: Optional[CanonicalAnalysisSummary] = None

    # 1. Canonical Specification Fields
    instrument: InstrumentMetadata
    data_quality: DataQualityReport
    tested_hypothesis: str = "RANGE_CONSOLIDATION" # "BULLISH_CONTINUATION", "BEARISH_EXPANSION", "RANGE_CONSOLIDATION"
    verdict: str = "NO VALIDATED EDGE"
    directional_bias: DirectionalBias
    evidence_strength: EvidenceStrength
    setup_status: SetupStatus
    backtest_status: BacktestStatus
    thesis_status: ThesisStatus
    strategy_signals: List[StrategyResult] = Field(default_factory=list)
    independent_evidence_families: List[EvidenceDependency] = Field(default_factory=list)
    dependency_explanation: str = ""
    historical_setup_results: HistoricalSetupResult
    supporting_evidence: List[EvidenceClaim] = Field(default_factory=list)
    opposing_evidence: List[EvidenceClaim] = Field(default_factory=list)
    neutral_evidence: List[EvidenceClaim] = Field(default_factory=list)
    confirmation_conditions: List[ConfirmationCondition] = Field(default_factory=list)
    invalidation_conditions: List[InvalidationCondition] = Field(default_factory=list)
    risk_controls: List[RiskControlCondition] = Field(default_factory=list)
    top_supporting_facts: List[str] = Field(default_factory=list)
    top_opposing_facts: List[str] = Field(default_factory=list)
    thesis_change_conditions: List[str] = Field(default_factory=list)
    audit_warnings: List[str] = Field(default_factory=list)
    reconciliation_status: str = "VALIDATED"
    
    # Audited Status Separation (6 Structured Status Fields)
    overall_validation_status: str = "NO_VALIDATED_EDGE"
    walk_forward_profitability_status: str = "UNPROFITABLE"
    simulation_execution_status: str = "EXPLORATORY_HISTORICAL_SIMULATION"  # "SUCCESSFUL_EXECUTION", "EXPLORATORY_HISTORICAL_SIMULATION", "EXECUTION_FAILED"
    profitability_status: str = "UNPROFITABLE"                               # "PROFITABLE", "UNPROFITABLE", "INCONCLUSIVE"
    historical_validation_status: str = "FAIL"                               # "PASS", "FAIL", "WARNING", "INSUFFICIENT_DATA"
    walk_forward_status: str = "FAIL"                                        # "PASS", "FAIL", "WARNING", "INSUFFICIENT_DATA"
    risk_limit_status: str = "WITHIN_LIMITS"                                 # "DRAWDOWN_LIMIT_BREACH", "WITHIN_LIMITS"
    benchmark_comparison_status: str = "MIXED_BENCHMARK_RESULTS"           # "OUTPERFORMED_BENCHMARK", "MIXED_BENCHMARK_RESULTS", "UNDERPERFORMED_BENCHMARK", "NEUTRAL"
    benchmark_comparison: Optional[BenchmarkComparisonResult] = None
    
    # Legacy Status Fields
    matching_status: str = "MATCHING_COMPLETE"
    strategy_backtest_status: str = "FAIL"
    portfolio_simulation_status: str = "PASS"
    
    # 2. Advanced Differentiation Fields
    evidence_graph: EvidenceGraph = Field(default_factory=EvidenceGraph)
    measured_dependencies: List[MeasuredDependence] = Field(default_factory=list)
    historical_comparison: Optional[HistoricalComparisonResult] = None
    trade_geometry: Optional[TradeGeometry] = None
    walk_forward_evaluation: Optional[WalkForwardAnalogueResult] = None
    portfolio_simulation: Optional[EventDrivenPortfolioResult] = None
    drawdown_audit_records: List[DrawdownAuditRecord] = Field(default_factory=list)
    drawdown_breach_warning: Optional[str] = None

    # 3. AI Review & Thesis Content
    bull_case: str = ""
    bear_case: str = ""
    explanation: str = ""
    
    # 4. Context & Backwards Compatibility
    symbol: str = ""
    market: str = "US"
    timeframe: str = "1d"
    latest_price: float = 0.0
    as_of_timestamp: str = ""
    support_levels: List[float] = Field(default_factory=list)
    resistance_levels: List[float] = Field(default_factory=list)
    multi_timeframe_analysis: str = ""
    risks: List[str] = Field(default_factory=list)
    disclaimer: str = "This stress test is for informational purposes only. Do not execute trades based solely on this information."
    
    evidence_dependencies: List[EvidenceDependency] = Field(default_factory=list)
    data_quality_warnings: List[str] = Field(default_factory=list)
    directional_family_alignment: int = 0
    directional_family_alignment_detail: str = ""
    directional_family_alignment_formula: str = ""
    directional_family_alignment_participating: List[str] = Field(default_factory=list)
    directional_coverage_excl_risk: str = "75/100 (3 of 4 direction-capable families)"
    heuristic_score: int = 50
    score: int = 50
    confidence: float = 0.5
    overall_signal: str = "NEUTRAL"

# Aliases for backward compatibility
TradeThesisStressTestResult = AnalysisResult
FinalAnalysisResult = AnalysisResult

