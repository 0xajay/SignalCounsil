import pytest
from app.domain.models import StrategyContext, PerformanceMetrics
from app.domain.strategies import TrendStrategy
from app.domain.backtest import BacktestEngine
from tests.fixtures.market_data import get_ohlcv_models

def test_backtest_engine_run():
    data = get_ohlcv_models()
    strategy = TrendStrategy()
    context = StrategyContext(symbol="TEST", timeframe="1d")
    
    engine = BacktestEngine(
        market_data=data,
        strategy=strategy,
        context=context,
        initial_capital=10000.0,
        warmup_period=50
    )
    
    metrics = engine.run()
    
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.trade_count >= 0
    assert isinstance(metrics.total_return, float)
    assert isinstance(metrics.cagr, float)
    assert isinstance(metrics.win_rate, float)
    assert isinstance(metrics.profit_factor, float)
    assert isinstance(metrics.max_drawdown, float)
    assert isinstance(metrics.average_win, float)
    assert isinstance(metrics.average_loss, float)
