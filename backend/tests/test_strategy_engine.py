import pytest
import pandas as pd
from app.domain.engine import (
    to_dataframe,
    get_ema, get_sma, get_rsi, get_macd, get_stochastic,
    get_bollinger_bands, get_keltner_channels, get_atr,
    get_adx, get_supertrend, get_donchian_channels
)
from tests.fixtures.market_data import get_ohlcv_models

@pytest.fixture
def df():
    data = get_ohlcv_models()
    return to_dataframe(data)

def test_ema(df):
    ema = get_ema(df, window=14)
    assert len(ema) == 100
    assert not pd.isna(ema.iloc[-1])

def test_sma(df):
    sma = get_sma(df, window=14)
    assert len(sma) == 100
    assert not pd.isna(sma.iloc[-1])

def test_rsi(df):
    rsi = get_rsi(df, window=14)
    assert len(rsi) == 100
    assert 0 <= rsi.iloc[-1] <= 100

def test_macd(df):
    macd = get_macd(df)
    assert 'macd' in macd.columns
    assert 'signal' in macd.columns
    assert 'histogram' in macd.columns
    assert not pd.isna(macd['macd'].iloc[-1])

def test_stochastic(df):
    stoch = get_stochastic(df)
    assert 'k' in stoch.columns
    assert 'd' in stoch.columns
    assert not pd.isna(stoch['k'].iloc[-1])

def test_bollinger(df):
    bb = get_bollinger_bands(df)
    assert 'upper' in bb.columns
    assert 'mid' in bb.columns
    assert 'lower' in bb.columns
    assert not pd.isna(bb['upper'].iloc[-1])
    # Upper band should be > mid band
    assert bb['upper'].iloc[-1] > bb['mid'].iloc[-1]

def test_keltner(df):
    kc = get_keltner_channels(df)
    assert 'upper' in kc.columns
    assert 'mid' in kc.columns
    assert 'lower' in kc.columns
    assert kc['upper'].iloc[-1] > kc['mid'].iloc[-1]

def test_atr(df):
    atr = get_atr(df, window=14)
    assert len(atr) == 100
    assert atr.iloc[-1] > 0

def test_adx(df):
    adx = get_adx(df, window=14)
    assert 'adx' in adx.columns
    assert 'pos_di' in adx.columns
    assert 'neg_di' in adx.columns
    assert not pd.isna(adx['adx'].iloc[-1])

def test_supertrend(df):
    st = get_supertrend(df)
    assert 'supertrend' in st.columns
    assert 'direction' in st.columns
    # Direction should be 1 or -1
    assert st['direction'].iloc[-1] in [1, -1]

def test_donchian(df):
    dc = get_donchian_channels(df)
    assert 'upper' in dc.columns
    assert 'lower' in dc.columns
    assert 'mid' in dc.columns
    assert dc['upper'].iloc[-1] >= dc['lower'].iloc[-1]
