import pandas as pd
import numpy as np
from typing import List
import ta
from .models import OHLCV

def to_dataframe(data: List[OHLCV]) -> pd.DataFrame:
    """Helper to convert OHLCV models into a sorted pandas DataFrame."""
    df = pd.DataFrame([d.model_dump() for d in data])
    return df.sort_values('timestamp').reset_index(drop=True)

def get_ema(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate EMA purely functionally."""
    return ta.trend.ema_indicator(df['close'], window=window)

def get_sma(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate SMA purely functionally."""
    return ta.trend.sma_indicator(df['close'], window=window)

def get_rsi(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate RSI purely functionally."""
    return ta.momentum.rsi(df['close'], window=window)

def get_macd(df: pd.DataFrame, window_slow: int = 26, window_fast: int = 12, window_sign: int = 9) -> pd.DataFrame:
    """Calculate MACD purely functionally."""
    macd = ta.trend.MACD(df['close'], window_slow, window_fast, window_sign)
    return pd.DataFrame({
        'macd': macd.macd(),
        'signal': macd.macd_signal(),
        'histogram': macd.macd_diff()
    })

def get_stochastic(df: pd.DataFrame, window: int = 14, smooth_window: int = 3) -> pd.DataFrame:
    """Calculate Stochastic purely functionally."""
    stoch = ta.momentum.StochasticOscillator(
        df['high'], df['low'], df['close'], window=window, smooth_window=smooth_window
    )
    return pd.DataFrame({
        'k': stoch.stoch(),
        'd': stoch.stoch_signal()
    })

def get_bollinger_bands(df: pd.DataFrame, window: int = 20, window_dev: float = 2.0) -> pd.DataFrame:
    """Calculate Bollinger Bands purely functionally."""
    indicator = ta.volatility.BollingerBands(df['close'], window=window, window_dev=window_dev)
    return pd.DataFrame({
        'upper': indicator.bollinger_hband(),
        'mid': indicator.bollinger_mavg(),
        'lower': indicator.bollinger_lband()
    })

def get_keltner_channels(df: pd.DataFrame, window: int = 20, window_atr: int = 10) -> pd.DataFrame:
    """Calculate Keltner Channels purely functionally."""
    indicator = ta.volatility.KeltnerChannel(
        df['high'], df['low'], df['close'], window=window, window_atr=window_atr
    )
    return pd.DataFrame({
        'upper': indicator.keltner_channel_hband(),
        'mid': indicator.keltner_channel_mband(),
        'lower': indicator.keltner_channel_lband()
    })

def get_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Calculate ATR purely functionally."""
    return ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=window)

def get_adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Calculate ADX purely functionally."""
    indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=window)
    return pd.DataFrame({
        'adx': indicator.adx(),
        'pos_di': indicator.adx_pos(),
        'neg_di': indicator.adx_neg()
    })

def get_donchian_channels(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Calculate Donchian Channels purely functionally."""
    indicator = ta.volatility.DonchianChannel(df['high'], df['low'], df['close'], window=window)
    return pd.DataFrame({
        'upper': indicator.donchian_channel_hband(),
        'mid': indicator.donchian_channel_mband(),
        'lower': indicator.donchian_channel_lband()
    })

def get_supertrend(df: pd.DataFrame, period: int = 7, multiplier: float = 3.0) -> pd.DataFrame:
    """Calculate Supertrend purely functionally."""
    high = df['high']
    low = df['low']
    close = df['close']
    
    hl2 = (high + low) / 2
    atr = get_atr(df, window=period)
    
    basic_ub = hl2 + (multiplier * atr)
    basic_lb = hl2 - (multiplier * atr)
    
    final_ub = np.zeros(len(df))
    final_lb = np.zeros(len(df))
    supertrend = np.zeros(len(df))
    direction = np.zeros(len(df)) # 1 for up, -1 for down
    
    for i in range(len(df)):
        if i == 0 or np.isnan(basic_ub.iloc[i]):
            final_ub[i] = basic_ub.iloc[i]
            final_lb[i] = basic_lb.iloc[i]
            supertrend[i] = final_ub[i]
            direction[i] = 1
            continue
            
        # Final Upper Band
        if basic_ub.iloc[i] < final_ub[i-1] or close.iloc[i-1] > final_ub[i-1]:
            final_ub[i] = basic_ub.iloc[i]
        else:
            final_ub[i] = final_ub[i-1]
            
        # Final Lower Band
        if basic_lb.iloc[i] > final_lb[i-1] or close.iloc[i-1] < final_lb[i-1]:
            final_lb[i] = basic_lb.iloc[i]
        else:
            final_lb[i] = final_lb[i-1]
            
        # Supertrend and Direction
        if supertrend[i-1] == final_ub[i-1] and close.iloc[i] <= final_ub[i]:
            supertrend[i] = final_ub[i]
            direction[i] = -1
        elif supertrend[i-1] == final_ub[i-1] and close.iloc[i] > final_ub[i]:
            supertrend[i] = final_lb[i]
            direction[i] = 1
        elif supertrend[i-1] == final_lb[i-1] and close.iloc[i] >= final_lb[i]:
            supertrend[i] = final_lb[i]
            direction[i] = 1
        elif supertrend[i-1] == final_lb[i-1] and close.iloc[i] < final_lb[i]:
            supertrend[i] = final_ub[i]
            direction[i] = -1
            
    return pd.DataFrame({
        'supertrend': supertrend,
        'direction': direction
    })
