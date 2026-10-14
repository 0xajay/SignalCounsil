import pytest
from app.adapters.yahoo_finance import get_ohlcv, get_latest_price, validate_symbol, _normalize_symbol, resolve_symbol
from app.domain.models import OHLCV

@pytest.mark.parametrize("symbol", [
    "AAPL",
    "NIFTY",
    "RELIANCE",
    "ITC"
])
def test_validate_symbol_valid(symbol):
    """Test that valid symbols are correctly recognized."""
    assert validate_symbol(symbol) is True

def test_validate_symbol_invalid():
    """Test that an invalid symbol returns False."""
    assert validate_symbol("INVALID_TICKER_12345") is False

@pytest.mark.parametrize("symbol", [
    "AAPL",
    "NIFTY",
    "RELIANCE",
    "ITC"
])
def test_get_latest_price(symbol):
    """Test that we can retrieve a positive latest price."""
    price = get_latest_price(symbol)
    assert isinstance(price, float)
    assert price > 0

@pytest.mark.parametrize("symbol", [
    "AAPL",
    "NIFTY",
    "RELIANCE"
])
def test_get_ohlcv(symbol):
    """Test that OHLCV data is returned correctly shaped."""
    data = get_ohlcv(symbol, interval="1d", limit=5)
    
    # Assert we get data back
    assert len(data) > 0
    assert len(data) <= 5
    
    # Check structure of the last candle
    last_candle = data[-1]
    assert isinstance(last_candle, OHLCV)
    assert last_candle.open > 0
    assert last_candle.high >= last_candle.low
    assert last_candle.close > 0
    assert last_candle.volume >= 0

@pytest.mark.parametrize("raw_input,expected", [
    ("NSE:RELIANCE", "RELIANCE.NS"),
    ("NSE:NIFTY", "^NSEI"),
    ("NSE:BANKNIFTY", "^NSEBANK"),
    ("BSE:TCS", "TCS.BO"),
    ("BSE:SENSEX", "^BSESN"),
    ("NASDAQ:AAPL", "AAPL"),
    ("NYSE:IBM", "IBM"),
    ("LSE:VOD", "VOD.L"),
    ("TSX:SHOP", "SHOP.TO"),
    ("ASX:BHP", "BHP.AX"),
    ("XETRA:SAP", "SAP.DE"),
    ("BINANCE:BTCUSDT", "BTC-USD"),
    ("FX:EURUSD", "EURUSD=X"),
    ("RELIANCE.NS", "RELIANCE.NS"),
    ("TCS.BO", "TCS.BO"),
    ("AAPL", "AAPL"),
    ("VOD.L", "VOD.L"),
    ("BTCUSDT", "BTC-USD"),
    ("EURUSD", "EURUSD=X"),
    ("GOLD", "GC=F"),
    ("SPX", "^GSPC"),
    ("FTSE", "^FTSE"),
    ("DAX", "^GDAXI"),
])
def test_normalize_symbol_with_market_prefixes(raw_input, expected):
    """Test that market prefixes like NSE, BSE, NASDAQ, LSE, TSX, ASX, BINANCE, FX are normalized."""
    assert _normalize_symbol(raw_input) == expected

@pytest.mark.parametrize("raw_input,market,expected", [
    ("RELIANCE", "NSE", "RELIANCE.NS"),
    ("NIFTY", "NSE", "^NSEI"),
    ("NIFTY50", "NSE", "^NSEI"),
    ("BANKNIFTY", "NSE", "^NSEBANK"),
    ("TCS", "BSE", "TCS.BO"),
    ("SENSEX", "BSE", "^BSESN"),
    ("AAPL", "NASDAQ", "AAPL"),
    ("IBM", "US", "IBM"),
    ("VOD", "LSE", "VOD.L"),
    ("SHOP", "TSX", "SHOP.TO"),
    ("BHP", "ASX", "BHP.AX"),
    ("SAP", "XETRA", "SAP.DE"),
    ("MC", "EURONEXT", "MC.PA"),
    ("7203", "TSE", "7203.T"),
    ("0700", "HKEX", "0700.HK"),
    ("BTCUSDT", "BINANCE", "BTC-USD"),
    ("EURUSD", "FX", "EURUSD=X"),
    ("RELIANCE.NS", "NSE", "RELIANCE.NS"),
    ("TCS.BO", "BSE", "TCS.BO"),
    ("AAPL.NS", "NASDAQ", "AAPL"),
])
def test_normalize_symbol_with_market_parameter(raw_input, market, expected):
    """Test that clean symbols are normalized across all global markets."""
    assert _normalize_symbol(raw_input, market=market) == expected


def test_resolve_symbol_naked_and_market():
    """Test that resolve_symbol accurately resolves naked tickers like ITC to valid exchange tickers."""
    assert resolve_symbol("ITC") == "ITC.NS"
    assert resolve_symbol("RELIANCE") == "RELIANCE.NS"
    assert resolve_symbol("AAPL") == "AAPL"
    assert resolve_symbol("ITC", market="NSE") == "ITC.NS"
    assert resolve_symbol("ITC", market="BSE") == "ITC.BO"
    assert resolve_symbol("VOD", market="LSE") == "VOD.L"
