import pytest
from app.domain.models import OHLCVData, StrategyContext, Signal, StrategyResult
from app.domain.strategies import (
    TrendStrategy, MomentumStrategy, MeanReversionStrategy,
    BreakoutStrategy, PriceActionStrategy, RiskStrategy
)
from tests.fixtures.market_data import get_ohlcv_models

@pytest.fixture
def market_data():
    return OHLCVData(data=get_ohlcv_models())

@pytest.fixture
def context():
    return StrategyContext(symbol="TEST", timeframe="1d")

@pytest.mark.parametrize("strategy_class, expected_name", [
    (TrendStrategy, "Trend"),
    (MomentumStrategy, "Momentum"),
    (MeanReversionStrategy, "MeanReversion"),
    (BreakoutStrategy, "Breakout"),
    (PriceActionStrategy, "PriceAction"),
    (RiskStrategy, "Risk"),
])
def test_strategy_evaluation(market_data, context, strategy_class, expected_name):
    strategy = strategy_class()
    result = strategy.evaluate(market_data, context)
    
    assert isinstance(result, StrategyResult)
    assert result.strategy_name == expected_name
    assert isinstance(result.signal, Signal)
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.metadata, dict)
