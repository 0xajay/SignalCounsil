import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.domain.models import OHLCV

def get_deterministic_ohlcv() -> pd.DataFrame:
    """Generate 100 rows of deterministic synthetic market data."""
    np.random.seed(42)
    base_price = 100
    
    # Generate random walk
    returns = np.random.normal(0, 1, 100)
    close = base_price + np.cumsum(returns)
    
    # Generate realistic-ish wicks
    high = close + np.random.uniform(0.5, 2.0, 100)
    low = close - np.random.uniform(0.5, 2.0, 100)
    open_price = close - np.random.uniform(-1.0, 1.0, 100)
    volume = np.random.randint(1000, 5000, 100)
    
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': open_price,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    })
    return df

def get_ohlcv_models() -> list[OHLCV]:
    """Return the deterministic dataframe as a list of OHLCV domain models."""
    df = get_deterministic_ohlcv()
    return [
        OHLCV(
            timestamp=row['timestamp'],
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=int(row['volume'])
        ) for _, row in df.iterrows()
    ]

def create_bullish_trending_data(num_bars: int = 120) -> list[OHLCV]:
    """Generate steadily upward trending market data where bullish setups pass historical tests."""
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(num_bars)]
    close = 100.0 + np.arange(num_bars) * 0.8 + np.sin(np.arange(num_bars)) * 0.5
    high = close + 1.2
    low = close - 0.8
    open_p = close - 0.3
    volume = [2500] * num_bars
    return [
        OHLCV(
            timestamp=dates[i],
            open=float(open_p[i]),
            high=float(high[i]),
            low=float(low[i]),
            close=float(close[i]),
            volume=int(volume[i])
        )
        for i in range(num_bars)
    ]
