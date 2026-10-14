import pytest
import numpy as np
from datetime import datetime, timedelta
from app.domain.models import (
    OHLCV, OHLCVData, DirectionalBias, BacktestStatus, Signal, StrategyResult,
    EvidenceFamily, EvidenceStrength, SetupStatus, ThesisStatus,
    TransactionCostModel, SetupPartitionMetrics, MeasuredDependence,
    FinalAnalysisResult, HistoricalSetupConfig
)
from app.domain.historical_setup import (
    HistoricalSetupEngine, simulate_trade_at_bar, calculate_partition_metrics
)
from app.domain.evidence import EvidenceLedgerBuilder
from app.domain.stress_tester import TradeThesisStressTester

def make_ohlcv_series(n_bars: int = 150, start_price: float = 100.0, trend: float = -0.002) -> list[OHLCV]:
    """Generates synthetic OHLCV bars for deterministic testing."""
    bars = []
    base_date = datetime(2025, 1, 1)
    price = start_price
    for i in range(n_bars):
        date_str = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
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
# 1. Per-leg versus round-trip cost arithmetic
# ==============================================================================
def test_per_leg_versus_round_trip_cost_arithmetic():
    """
    Verifies canonical transaction-cost configuration:
    - Entry fee: 0.05% (0.0005)
    - Exit fee: 0.05% (0.0005)
    - Round-trip fees: 0.10% (0.0010)
    - Entry slippage: 0.025% (0.00025)
    - Exit slippage: 0.025% (0.00025)
    - Round-trip slippage: 0.05% (0.0005)
    - Total per-leg friction: 0.075% (0.00075)
    - Total round-trip cost: 0.150% (0.00150)
    - Compounded net return formula: (1 + gross_ret) * (1 - 0.00075)^2 - 1
    """
    model = TransactionCostModel()
    assert model.entry_fee_pct == 0.0005
    assert model.exit_fee_pct == 0.0005
    assert model.round_trip_fee_pct == 0.0010
    assert model.entry_slippage_pct == 0.00025
    assert model.exit_slippage_pct == 0.00025
    assert model.round_trip_slippage_pct == 0.00050
    assert model.total_per_leg_cost_pct == 0.00075
    assert model.total_round_trip_cost_pct == 0.00150

    # Test net return on gross return +2.0%
    gross_ret = 0.02
    net_ret, total_costs = model.calculate_net_return(gross_ret)
    expected_net = (1.0 + gross_ret) * ((1.0 - 0.00075) ** 2) - 1.0
    expected_costs = gross_ret - expected_net

    assert abs(net_ret - expected_net) < 1e-12
    assert abs(total_costs - expected_costs) < 1e-12
    assert total_costs > 0.00149  # ~0.15% deduction
    assert net_ret < gross_ret

    # Test net return on flat trade (gross = 0.0)
    net_flat, flat_costs = model.calculate_net_return(0.0)
    expected_flat_net = ((1.0 - 0.00075) ** 2) - 1.0
    assert abs(net_flat - expected_flat_net) < 1e-12
    assert abs(flat_costs - (0.0 - expected_flat_net)) < 1e-12
    assert net_flat < 0.0  # Slippage & fees produce a loss on flat price action

# ==============================================================================
# 2. Identical returns for identical shared trades
# ==============================================================================
def test_identical_returns_for_identical_shared_trades():
    """
    Compares the historical analogue records that share decision timestamps with
    the strategy backtest. Asserts that gross return, total costs, and net return
    are identical within a 1e-6 numerical tolerance.
    """
    bars = make_ohlcv_series(120)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    
    bt_res, _ = engine.investigate_setup(DirectionalBias.BEARISH)
    comp_res = engine.compare_historical_setups(DirectionalBias.BEARISH, horizon_bars=5, backtest_result=bt_res)

    assert comp_res.shared_decision_bars_count > 0
    assert comp_res.shared_trades_verified_identical is True
    assert comp_res.shared_trades_max_return_discrepancy < 1e-6

    # Verify each shared trade bar explicitly
    shared_bars = set(comp_res.shared_records_metrics.bar_indices)
    assert len(shared_bars) == comp_res.shared_decision_bars_count

    for b in shared_bars:
        bt_trade = engine.backtest_trades_by_bar[b]
        # Find matching analogue
        matching_analogues = [a for a in comp_res.analogues if a.decision_date == bt_trade["decision_date"]]
        if matching_analogues:
            an = matching_analogues[0]
            assert abs(an.decision_price - bt_trade["entry_price"]) < 1e-2
            assert abs(an.strategy_net_return_pct - round(bt_trade["net_return_pct"], 2)) < 1e-4
            assert abs(an.forward_return_pct - round(bt_trade["net_return_pct"], 2)) < 1e-4

# ==============================================================================
# 3. Correct aggregation of shared, analogue-only and backtest-only records
# ==============================================================================
def test_correct_aggregation_of_partition_records():
    """
    Verifies that records are partitioned and aggregated cleanly into:
    1. Shared decision bars.
    2. Analogue-only decision bars.
    3. Strategy-backtest-only decision bars.
    """
    bars = make_ohlcv_series(130)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    
    bt_res, _ = engine.investigate_setup(DirectionalBias.BEARISH)
    comp_res = engine.compare_historical_setups(DirectionalBias.BEARISH, horizon_bars=5, backtest_result=bt_res)

    shared_m = comp_res.shared_records_metrics
    an_only_m = comp_res.analogue_only_metrics
    bt_only_m = comp_res.backtest_only_metrics

    assert isinstance(shared_m, SetupPartitionMetrics)
    assert isinstance(an_only_m, SetupPartitionMetrics)
    assert isinstance(bt_only_m, SetupPartitionMetrics)

    # Check partition counts add up to total sets
    total_analogues = comp_res.comparable_setups_count
    total_backtest = len(engine.backtest_trades_by_bar)

    assert comp_res.shared_decision_bars_count + comp_res.analogue_only_bars_count == total_analogues
    assert comp_res.shared_decision_bars_count + comp_res.backtest_only_bars_count == total_backtest

    # Verify reconciliation summary string
    assert "3-Way Partition Reconciliation" in comp_res.partition_reconciliation_summary
    assert "shared bars" in comp_res.partition_reconciliation_summary
    assert "analogue-only bars" in comp_res.partition_reconciliation_summary
    assert "backtest-only bars" in comp_res.partition_reconciliation_summary

# ==============================================================================
# 4. No empirical dependency penalty when not applicable (neutral indicators)
# ==============================================================================
def test_no_empirical_dependency_penalty_for_neutral_indicators():
    """
    Verifies that when measured correlation exists between neutral indicators
    (e.g. Momentum vs Mean Reversion with Spearman r=+0.94), the engine does NOT
    claim an active empirical redundancy penalty or reduce voting weight.
    Neutral indicators submit no directional vote, so they cannot inflate consensus.
    """
    # 4 strategy results where Momentum and Mean Reversion are NEUTRAL
    strat_results = [
        StrategyResult(strategy_name="TrendStrategy", signal=Signal.SELL, rule="Bearish EMA", weight=1.0, confidence=1.0),
        StrategyResult(strategy_name="BreakoutStrategy", signal=Signal.SELL, rule="Lower breakdown", weight=1.0, confidence=1.0),
        StrategyResult(strategy_name="MomentumStrategy", signal=Signal.NEUTRAL, rule="RSI neutral", weight=1.0, confidence=1.0),
        StrategyResult(strategy_name="MeanReversionStrategy", signal=Signal.NEUTRAL, rule="Bollinger neutral", weight=1.0, confidence=1.0)
    ]
    
    market_data = OHLCVData(data=make_ohlcv_series(60))
    builder = EvidenceLedgerBuilder(market_data, strat_results)
    claims = builder.build_claims()
    
    # Provide measured dependence with r=+0.94 between Momentum and Mean Reversion
    measured_deps = [
        MeasuredDependence(
            indicator_a="Momentum (RSI 14)",
            indicator_b="MeanReversion (Bollinger %B)",
            correlation=0.94,
            is_dependent=True,
            limitations="Collinear oscillator metrics."
        )
    ]
    
    deps, raw_votes, indep_votes, false_consensus, dep_explanation = builder.detect_false_consensus(
        claims, measured_dependencies=measured_deps
    )
    
    mom_dep = next(d for d in deps if d.family == EvidenceFamily.MOMENTUM)
    mr_dep = next(d for d in deps if d.family == EvidenceFamily.MEAN_REVERSION)
    
    # Weight must NOT be dampened because both are NEUTRAL
    assert mr_dep.effective_weight == 1.0
    assert mom_dep.effective_weight == 1.0
    assert mr_dep.grouping_type != "EMPIRICALLY_ADJUSTED"
    
    # Explanation must verify no cross-family empirical redundancy penalties active
    assert "No cross-family empirical redundancy penalties active" in dep_explanation
    assert "Both indicator families are NEUTRAL" in mr_dep.empirical_adjustment_detail

# ==============================================================================
# 5. Sum of trade returns versus compounded portfolio return
# ==============================================================================
def test_sum_of_trade_returns_versus_compounded_portfolio_return():
    """
    Test fixtures with known trade returns:
    Case A: Trades [+10.0%, -10.0%]
    - Sum of Trade Returns: 0.0%
    - Compounded Equity Return: (1.10 * 0.90) - 1.0 = -1.0%
    - Demonstrates arithmetic sum vs sequential compounding difference.
    
    Case B: Multi-trade series with exact expected metrics.
    """
    # Test fixture with 2 opposing trades of equal magnitude
    trades_fixture_a = [
        {"bar_idx": 1, "decision_date": "2025-01-01", "net_return": 0.10, "max_adverse_excursion": 0.02, "raw_market_return": 0.10},
        {"bar_idx": 2, "decision_date": "2025-01-02", "net_return": -0.10, "max_adverse_excursion": 0.11, "raw_market_return": -0.10}
    ]
    
    metrics_a = calculate_partition_metrics(trades_fixture_a, "Fixture A", position_size=1.0)
    assert metrics_a.sum_of_trade_returns_pct == 0.0
    assert metrics_a.arithmetic_mean_trade_return_pct == 0.0
    assert metrics_a.compounded_equity_return_pct == -1.0  # 1.10 * 0.90 - 1 = -0.01 (-1.0%)
    assert metrics_a.strategy_win_rate_pct == 50.0
    assert metrics_a.profit_factor == 1.0
    assert metrics_a.equity_max_drawdown_pct == 10.0

    # Test fixture with 4 trades: [+5%, +3%, -2%, +4%]
    trades_fixture_b = [
        {"bar_idx": 10, "decision_date": "2025-01-10", "net_return": 0.05, "max_adverse_excursion": 0.01, "raw_market_return": 0.05},
        {"bar_idx": 15, "decision_date": "2025-01-15", "net_return": 0.03, "max_adverse_excursion": 0.01, "raw_market_return": 0.03},
        {"bar_idx": 20, "decision_date": "2025-01-20", "net_return": -0.02, "max_adverse_excursion": 0.03, "raw_market_return": -0.02},
        {"bar_idx": 25, "decision_date": "2025-01-25", "net_return": 0.04, "max_adverse_excursion": 0.01, "raw_market_return": 0.04}
    ]
    metrics_b = calculate_partition_metrics(trades_fixture_b, "Fixture B", position_size=1.0)
    expected_sum = (0.05 + 0.03 - 0.02 + 0.04) * 100.0  # 10.0%
    expected_compounded = ((1.05 * 1.03 * 0.98 * 1.04) - 1.0) * 100.0  # ~10.23%
    assert abs(metrics_b.sum_of_trade_returns_pct - expected_sum) < 1e-4
    assert abs(metrics_b.compounded_equity_return_pct - round(expected_compounded, 2)) < 1e-2

# ==============================================================================
# 6. Stable backtest failure and out-of-sample insufficiency statuses
# ==============================================================================
def test_stable_backtest_failure_and_oos_insufficiency_statuses():
    """
    Verifies that the strategy backtest validation gate is strictly preserved:
    - Backtest status remains FAIL unless documented criteria pass.
    - Out-of-sample status remains INSUFFICIENT_DATA until N >= 5.
    - Matching success (e.g. 50 analogues) NEVER overrides strategy validation.
    """
    # OOS ratio configured such that OOS window has fewer than 5 trades (< min_oos_samples=5)
    config_short_oos = HistoricalSetupConfig(oos_ratio=0.05, min_oos_samples=5)
    bars_short = make_ohlcv_series(120, trend=-0.002)
    engine_short = HistoricalSetupEngine(bars_short, horizon_bars=5, config=config_short_oos)
    bt_res_short, overall_status = engine_short.investigate_setup(DirectionalBias.BEARISH)
    
    assert bt_res_short.oos_sample_count < 5
    assert bt_res_short.oos_status == BacktestStatus.INSUFFICIENT_DATA
    
    comp_res_short = engine_short.compare_historical_setups(DirectionalBias.BEARISH, horizon_bars=5, backtest_result=bt_res_short)
    assert comp_res_short.status != "PASS"
    assert comp_res_short.status != BacktestStatus.PASS
    assert comp_res_short.out_of_sample_validation == "INSUFFICIENT_DATA"

    # Upward trending series: short strategy loses money -> FAIL
    bars_fail = make_ohlcv_series(120, trend=0.005)
    engine_fail = HistoricalSetupEngine(bars_fail, horizon_bars=5)
    bt_res_fail, _ = engine_fail.investigate_setup(DirectionalBias.BEARISH)
    assert bt_res_fail.full_history.status in (BacktestStatus.FAIL, BacktestStatus.INSUFFICIENT_DATA)
    assert bt_res_fail.full_history.status != BacktestStatus.PASS

# ==============================================================================
# 7. UI consistency with canonical backend results
# ==============================================================================
def test_ui_consistency_and_canonical_backend_contracts():
    """
    Verifies that the backend analysis result contains all contract fields
    needed by the Chrome extension UI:
    - Canonical cost model breakdown
    - Distinct return metrics (Sum of Trade Returns & Compounded Equity Return)
    - 3-Way Partition Reconciliation metrics & summary
    - Caveats regarding overlapping holding horizons and notional sizing.
    """
    bars = make_ohlcv_series(120)
    engine = HistoricalSetupEngine(bars, horizon_bars=5)
    bt_res, _ = engine.investigate_setup(DirectionalBias.BEARISH)
    comp_res = engine.compare_historical_setups(DirectionalBias.BEARISH, horizon_bars=5, backtest_result=bt_res)
    
    # Serialize to dictionary (as sent over JSON API)
    data = comp_res.model_dump()
    
    assert "cost_model" in data
    assert data["cost_model"]["round_trip_fee_pct"] == 0.001
    assert data["cost_model"]["round_trip_slippage_pct"] == 0.0005
    assert data["cost_model"]["total_round_trip_cost_pct"] == 0.0015
    
    assert "sum_of_trade_returns_pct" in data
    assert "compounded_equity_return_pct" in data
    assert "portfolio_sizing_assumptions" in data
    assert "overlapping_horizon_caveat" in data
    
    assert "shared_records_metrics" in data
    assert "analogue_only_metrics" in data
    assert "backtest_only_metrics" in data
    assert "partition_reconciliation_summary" in data
    assert "shared_trades_verified_identical" in data
    assert data["shared_trades_verified_identical"] is True
