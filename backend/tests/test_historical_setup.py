import pytest
from app.domain.models import DirectionalBias, BacktestStatus
from app.domain.historical_setup import HistoricalSetupEngine
from tests.fixtures.market_data import get_ohlcv_models

def test_historical_setup_engine():
    data = get_ohlcv_models() # 100 deterministic bars
    engine = HistoricalSetupEngine(data, horizon_bars=5, friction_pct=0.001)

    result, status = engine.investigate_setup(candidate_bias=DirectionalBias.BULLISH)

    assert result.sample_count > 0
    assert 0.0 <= result.win_rate <= 1.0
    assert result.profit_factor >= 0.0
    assert result.max_drawdown >= 0.0
    assert result.transaction_cost_pct == 0.001
    assert "win" in result.outcome_distribution
    assert "loss" in result.outcome_distribution
    assert result.outcome_distribution["win"] + result.outcome_distribution["loss"] == result.sample_count
    assert status in [BacktestStatus.PASS, BacktestStatus.WARNING, BacktestStatus.FAIL, BacktestStatus.INSUFFICIENT_DATA]

    # Verify Out-Of-Sample separation
    assert result.oos_sample_count >= 0
    assert 0.0 <= result.oos_win_rate <= 1.0

def test_historical_setup_insufficient_data():
    data = get_ohlcv_models()[:30] # Only 30 bars (< 50 minimum)
    engine = HistoricalSetupEngine(data, horizon_bars=5)

    result, status = engine.investigate_setup(candidate_bias=DirectionalBias.BULLISH)
    assert status == BacktestStatus.INSUFFICIENT_DATA
    assert result.sample_count == 0
