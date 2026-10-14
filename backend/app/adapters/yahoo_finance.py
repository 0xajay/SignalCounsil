from typing import List, Optional, Dict, Tuple
import pandas as pd
import yfinance as yf
from app.domain.models import OHLCV, InstrumentMetadata

# Comprehensive Global Exchange to Yahoo Finance Suffix Map
EXCHANGE_SUFFIX_MAP: Dict[str, str] = {
    # India
    "NSE": ".NS", "NSI": ".NS",
    "BSE": ".BO", "BOM": ".BO",
    
    # United States (no suffix)
    "NASDAQ": "", "NYSE": "", "AMEX": "", "BATS": "", "ARCA": "", "US": "", "OTC": "",
    
    # United Kingdom
    "LSE": ".L", "LON": ".L",
    
    # Canada
    "TSX": ".TO", "TOR": ".TO", "TSXV": ".V", "VAN": ".V",
    
    # Europe
    "XETRA": ".DE", "XETR": ".DE", "FRA": ".F", "GER": ".DE", "FWB": ".F",
    "EURONEXT": ".PA", "PARIS": ".PA", "EPA": ".PA",
    "AMS": ".AS", "AMSTERDAM": ".AS",
    "BRU": ".BR", "BRUSSELS": ".BR",
    "LIS": ".LS", "LISBON": ".LS",
    "MIL": ".MI", "MILAN": ".MI", "BIT": ".MI",
    "BME": ".MC", "MADRID": ".MC",
    "SIX": ".SW", "SWX": ".SW", "VTX": ".SW",
    
    # Asia / Pacific
    "ASX": ".AX", "NZX": ".NZ",
    "TSE": ".T", "TYO": ".T", "JPX": ".T",
    "HKEX": ".HK", "HKG": ".HK",
    "SGX": ".SI",
    "KRX": ".KS", "KOSPI": ".KS", "KOSDAQ": ".KQ",
    "TWSE": ".TW", "TPEX": ".TWO",
    
    # Latin America & Others
    "BVMF": ".SA", "BOVESPA": ".SA", "SAO": ".SA",
    "BMV": ".MX", "MEX": ".MX",
    "JSE": ".JO"
}

# Major Global Indices Map
GLOBAL_INDEX_MAP: Dict[str, str] = {
    # India
    "NIFTY": "^NSEI", "NIFTY50": "^NSEI", "NIFTY 50": "^NSEI",
    "BANKNIFTY": "^NSEBANK", "NIFTYBANK": "^NSEBANK", "NIFTY BANK": "^NSEBANK",
    "SENSEX": "^BSESN", "BSESN": "^BSESN",
    
    # US
    "SPX": "^GSPC", "SP500": "^GSPC", "S&P500": "^GSPC", "S&P 500": "^GSPC",
    "NDX": "^IXIC", "NASDAQ100": "^IXIC", "COMP": "^IXIC",
    "DJI": "^DJI", "DOW": "^DJI", "DJIA": "^DJI",
    "RUT": "^RUT", "RUSSELL2000": "^RUT",
    "VIX": "^VIX",
    
    # Europe & UK
    "FTSE": "^FTSE", "FTSE100": "^FTSE",
    "DAX": "^GDAXI", "GDAXI": "^GDAXI",
    "CAC": "^FCHI", "CAC40": "^FCHI",
    "STOXX50": "^STOXX50E",
    
    # Asia
    "NIKKEI": "^N225", "N225": "^N225",
    "HSI": "^HSI", "HANGSENG": "^HSI",
    "ASX200": "^AXJO", "AXJO": "^AXJO",
    "KOSPI": "^KS11", "TWII": "^TWII"
}

# Known equities fast lookup (instant resolution for major market equities)
SYMBOL_MAP: Dict[str, str] = {
    # Indian Large Caps / Nifty 50
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "SBIN": "SBIN.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "LT": "LT.NS",
    "ITC": "ITC.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "AXISBANK": "AXISBANK.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "BAJAJFINSV": "BAJAJFINSV.NS",
    "MARUTI": "MARUTI.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "TITAN": "TITAN.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
    "TATAMOTORS": "TATAMOTORS.NS",
    "TATASTEEL": "TATASTEEL.NS",
    "NTPC": "NTPC.NS",
    "ONGC": "ONGC.NS",
    "POWERGRID": "POWERGRID.NS",
    "COALINDIA": "COALINDIA.NS",
    "JSWSTEEL": "JSWSTEEL.NS",
    "ADANIENT": "ADANIENT.NS",
    "ADANIPORTS": "ADANIPORTS.NS",
    "M&M": "M&M.NS",
    "MM": "M&M.NS",
    "WIPRO": "WIPRO.NS",
    "HCLTECH": "HCLTECH.NS",
    "TECHM": "TECHM.NS",
    "ULTRACEMCO": "ULTRACEMCO.NS",
    "GRASIM": "GRASIM.NS",
    "CIPLA": "CIPLA.NS",
    "HEROMOTOCO": "HEROMOTOCO.NS",
    "DRREDDY": "DRREDDY.NS",
    "EICHERMOT": "EICHERMOT.NS",
    "DIVISLAB": "DIVISLAB.NS",
    "BRITANNIA": "BRITANNIA.NS",
    "APOLLOHOSP": "APOLLOHOSP.NS",
    "TATACONSUM": "TATACONSUM.NS",
    "BPCL": "BPCL.NS",
    "HINDALCO": "HINDALCO.NS",
    "BEL": "BEL.NS",
    "TRENT": "TRENT.NS",
    "SHRIRAMFIN": "SHRIRAMFIN.NS",
    "INDUSINDBK": "INDUSINDBK.NS",
    "NESTLEIND": "NESTLEIND.NS",
    "ZOMATO": "ZOMATO.NS",
    "PAYTM": "PAYTM.NS",
    "JIOFIN": "JIOFIN.NS",
    "HAL": "HAL.NS",
    "VEDL": "VEDL.NS",
    "IOC": "IOC.NS",
    "DLF": "DLF.NS",
    "VBL": "VBL.NS",
    "CHOLAFIN": "CHOLAFIN.NS",
    "SIEMENS": "SIEMENS.NS",
    "PFC": "PFC.NS",
    "RECLTD": "RECLTD.NS",
    "GAIL": "GAIL.NS",
    "BANKBARODA": "BANKBARODA.NS",
    "PNB": "PNB.NS",
    "CANBK": "CANBK.NS",
    "IRCTC": "IRCTC.NS",
    "BHEL": "BHEL.NS",
    "IRFC": "IRFC.NS",
    "SUZLON": "SUZLON.NS",
    "IDEA": "IDEA.NS",
    "YESBANK": "YESBANK.NS",
    "POLYCAB": "POLYCAB.NS",
    "HAVELLS": "HAVELLS.NS",
    "PIDILITIND": "PIDILITIND.NS",
    "DABUR": "DABUR.NS",
    "GODREJCP": "GODREJCP.NS",
    "MARICO": "MARICO.NS",
    "BERGEPAINT": "BERGEPAINT.NS",
    "INDIGO": "INDIGO.NS",
    "MOTHERSON": "MOTHERSON.NS",
    "BOSCHLTD": "BOSCHLTD.NS",
    "TVSMOTOR": "TVSMOTOR.NS",
    "BAJAJ-AUTO": "BAJAJ-AUTO.NS",
    "LTIM": "LTIM.NS",
    "PERSISTENT": "PERSISTENT.NS",
    "COFORGE": "COFORGE.NS",
    "MPHASIS": "MPHASIS.NS",
    "MUTHOOTFIN": "MUTHOOTFIN.NS",
    "AUBANK": "AUBANK.NS",
    "FEDERALBNK": "FEDERALBNK.NS",
    "IDFCFIRSTB": "IDFCFIRSTB.NS",
    "BANDHANBNK": "BANDHANBNK.NS"
}

# Runtime resolution cache for dynamic multi-market lookups
_RESOLVED_SYMBOL_CACHE: Dict[Tuple[str, Optional[str]], str] = {}

def _normalize_symbol(symbol: str, market: Optional[str] = None) -> str:
    """Normalize any global instrument ticker to Yahoo Finance provider format."""
    sym = symbol.strip().upper()
    if not sym:
        return ""

    mkt = market.strip().upper() if market else None

    # 1. Check embedded prefixes (e.g. "LSE:VOD", "TSX:SHOP", "BINANCE:BTCUSDT", "NSE:RELIANCE")
    if ":" in sym:
        prefix, clean = sym.split(":", 1)
        prefix = prefix.strip()
        clean = clean.strip()
        if not mkt:
            mkt = prefix
        sym = clean

    # 2. Global Index Lookup
    if sym in GLOBAL_INDEX_MAP:
        return GLOBAL_INDEX_MAP[sym]

    # 3. Crypto handling (e.g. BINANCE, COINBASE, or symbols ending with USDT/USD)
    is_crypto_mkt = mkt in ["BINANCE", "COINBASE", "BYBIT", "OKX", "KRAKEN", "CRYPTO", "KUCOIN"]
    if is_crypto_mkt or sym.endswith("USDT") or (sym.endswith("USD") and not any(sym.endswith(s) for s in [".NS", ".BO", ".TO", ".AX"])):
        base = sym.replace("USDT", "").replace("USD", "").replace("-", "")
        if base in ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "DOT", "LINK", "MATIC"] or is_crypto_mkt or sym.endswith("USDT"):
            return f"{base}-USD"

    # 4. Forex handling (e.g. FX:EURUSD, EUR/USD, EURUSD)
    is_forex_mkt = mkt in ["FX", "FOREX", "OANDA", "FX_IDC"]
    if is_forex_mkt or "/" in sym:
        pair = sym.replace("/", "")
        if len(pair) == 6:
            return f"{pair}=X"
    if len(sym) == 6 and sym.isalpha() and not mkt:
        # Check standard fiat currencies
        major_fiats = {"USD", "EUR", "GBP", "JPY", "INR", "AUD", "CAD", "CHF", "NZD", "CNY", "HKD", "SGD"}
        if sym[:3] in major_fiats and sym[3:] in major_fiats:
            return f"{sym}=X"

    # 5. Commodities / Futures handling
    if sym in ["GOLD"]:
        return "GC=F"
    if sym in ["SILVER"]:
        return "SI=F"
    if sym in ["CRUDEOIL", "OIL", "WTI"]:
        return "CL=F"

    # 6. Explicit market normalization using global exchange mapping
    if mkt:
        if mkt in EXCHANGE_SUFFIX_MAP:
            suffix = EXCHANGE_SUFFIX_MAP[mkt]
            if suffix:
                clean_sym = sym
                for s in [".BO", ".NS", ".L", ".TO", ".AX", ".DE", ".PA", ".HK", ".T", ".MI", ".MC", ".SW"]:
                    if clean_sym.endswith(s):
                        clean_sym = clean_sym[:-len(s)]
                return clean_sym if clean_sym.endswith(suffix) else f"{clean_sym}{suffix}"
            else:
                # US markets: strip foreign suffixes
                return sym.split('.')[0]

    # 7. Check internal fast lookup
    if sym in SYMBOL_MAP:
        return SYMBOL_MAP[sym]

    # 8. If ticker already has a dot (e.g. VOD.L, SHOP.TO, RELIANCE.NS, BMW.DE) or starts with ^ or has =F / =X
    if "." in sym or sym.startswith("^") or sym.endswith("=F") or sym.endswith("=X") or sym.endswith("-USD"):
        return sym

    # Default to un-suffixed symbol (standard for US equities like AAPL, MSFT)
    return sym

def resolve_symbol(symbol: str, market: Optional[str] = None) -> str:
    """
    Resolve any symbol and optional market to a verified Yahoo Finance ticker.
    Supports instant cached lookup, rich asset mappings, and dynamic candidate discovery
    for naked tickers submitted without an explicit market.
    """
    clean_sym = symbol.strip().upper()
    if not clean_sym:
        return ""

    mkt = market.strip().upper() if market else None
    cache_key = (clean_sym, mkt)
    if cache_key in _RESOLVED_SYMBOL_CACHE:
        return _RESOLVED_SYMBOL_CACHE[cache_key]

    norm = _normalize_symbol(clean_sym, market=mkt)

    # If explicit market or prefix was provided, or already formatted (e.g. has suffix, index, forex, future)
    if mkt or "." in norm or norm.startswith("^") or norm.endswith("=F") or norm.endswith("=X") or norm.endswith("-USD"):
        _RESOLVED_SYMBOL_CACHE[cache_key] = norm
        return norm

    # Naked ticker with no market specified: test if naked US ticker has price data
    try:
        t = yf.Ticker(norm)
        if not t.history(period="5d").empty:
            _RESOLVED_SYMBOL_CACHE[cache_key] = norm
            return norm
    except Exception:
        pass

    # If naked symbol produced no data, try standard global market suffixes
    for candidate_suffix in [".NS", ".BO", ".L", ".TO", ".AX", ".DE", "-USD", "=X"]:
        try:
            cand = f"{clean_sym}{candidate_suffix}"
            t = yf.Ticker(cand)
            if not t.history(period="5d").empty:
                _RESOLVED_SYMBOL_CACHE[cache_key] = cand
                return cand
        except Exception:
            continue

    _RESOLVED_SYMBOL_CACHE[cache_key] = norm
    return norm

def _get_period_for_interval(interval: str) -> str:
    """Choose Yahoo Finance compliant period based on timeframe granularity."""
    if interval in ["1m", "2m", "5m"]:
        return "7d"
    elif interval in ["15m", "30m", "60m", "1h"]:
        return "60d"
    elif interval in ["1d"]:
        return "2y"
    elif interval in ["1wk", "1mo"]:
        return "5y"
    return "1y"

def get_instrument_metadata(symbol: str, market: Optional[str] = None) -> InstrumentMetadata:
    """Fetch rich instrument metadata with fallback to deterministic heuristics."""
    norm_symbol = resolve_symbol(symbol, market=market)
    mkt = market.strip().upper() if market else None
    
    # 1. Determine canonical instrument type
    if norm_symbol.startswith("^"):
        instrument_type = "INDEX"
    elif norm_symbol.endswith("-USD") or norm_symbol.endswith("-INR") or norm_symbol in ["BTC-USD", "ETH-USD"]:
        instrument_type = "CRYPTO"
    elif norm_symbol.endswith("=F"):
        instrument_type = "FUTURE"
    elif norm_symbol.endswith("=X"):
        instrument_type = "FOREX"
    else:
        instrument_type = "EQUITY"

    is_index = (instrument_type == "INDEX")
    
    # Currency and exchange deterministic defaults
    if norm_symbol.endswith(".NS") or norm_symbol.endswith(".BO") or norm_symbol in ["^NSEI", "^NSEBANK", "^BSESN"] or mkt in ["NSE", "BSE"]:
        currency = "INR"
        exchange = "BSE" if (mkt == "BSE" or norm_symbol.endswith(".BO")) else "NSE"
        timezone = "Asia/Kolkata"
    elif norm_symbol.endswith(".L") or mkt in ["LSE", "LON"]:
        currency = "GBP"
        exchange = "LSE"
        timezone = "Europe/London"
    elif norm_symbol.endswith((".TO", ".V")) or mkt in ["TSX", "TSXV"]:
        currency = "CAD"
        exchange = "TSX"
        timezone = "America/Toronto"
    elif norm_symbol.endswith(".AX") or mkt in ["ASX"]:
        currency = "AUD"
        exchange = "ASX"
        timezone = "Australia/Sydney"
    elif norm_symbol.endswith((".DE", ".F", ".PA", ".AS", ".BR", ".MI", ".MC", ".SW")):
        currency = "EUR"
        exchange = "XETRA" if norm_symbol.endswith((".DE", ".F")) else "EURONEXT"
        timezone = "Europe/Berlin"
    elif norm_symbol.endswith(".T") or mkt in ["TSE", "TYO"]:
        currency = "JPY"
        exchange = "TSE"
        timezone = "Asia/Tokyo"
    elif norm_symbol.endswith(".HK") or mkt in ["HKEX", "HKG"]:
        currency = "HKD"
        exchange = "HKEX"
        timezone = "Asia/Hong_Kong"
    elif instrument_type == "CRYPTO":
        currency = "USD"
        exchange = mkt or "CRYPTO"
        timezone = "UTC"
    elif instrument_type == "FOREX":
        currency = norm_symbol[-5:-2] if len(norm_symbol) >= 5 else "USD"
        exchange = "FOREX"
        timezone = "UTC"
    else:
        currency = "USD"
        exchange = mkt or "NASDAQ"
        timezone = "America/New_York"

    unit_label = "points" if is_index else currency
    name = symbol.upper()
    
    try:
        ticker = yf.Ticker(norm_symbol)
        fast_info = getattr(ticker, "fast_info", None)
        if fast_info:
            c = getattr(fast_info, "currency", currency) or currency
            if not is_index:
                currency = c
                unit_label = c
            ex = getattr(fast_info, "exchange", exchange) or exchange
            if ex == "NSI" or "NSE" in str(ex).upper():
                exchange = "NSE"
            elif ex == "BOM" or "BSE" in str(ex).upper():
                exchange = "BSE"
            else:
                exchange = ex
            timezone = getattr(fast_info, "timezone", timezone) or timezone
            qt = getattr(fast_info, "quote_type", None)
            if qt:
                qt_str = str(qt).upper()
                if qt_str in ["ETF", "MUTUALFUND"]:
                    instrument_type = "ETF"
                elif qt_str in ["FUTURE"]:
                    instrument_type = "FUTURE"
                elif qt_str in ["CRYPTOCURRENCY"]:
                    instrument_type = "CRYPTO"
                elif qt_str in ["INDEX"]:
                    instrument_type = "INDEX"
                    unit_label = "points"
        info = getattr(ticker, "info", {}) or {}
        if isinstance(info, dict) and info.get("shortName"):
            name = info["shortName"]
    except Exception:
        pass

    if norm_symbol == "^NSEI":
        name = "NIFTY 50"
        exchange = "NSE"
        instrument_type = "INDEX"
        unit_label = "points"
    elif norm_symbol == "^NSEBANK":
        name = "NIFTY BANK"
        exchange = "NSE"
        instrument_type = "INDEX"
        unit_label = "points"

    return InstrumentMetadata(
        symbol=symbol.upper(),
        canonical_symbol=norm_symbol,
        name=name,
        currency=currency,
        unit_label=unit_label,
        exchange=exchange,
        instrument_type=instrument_type,
        timezone=timezone,
        data_source="Yahoo Finance"
    )

def get_ohlcv(symbol: str, interval: str = "1d", limit: int = 100, market: Optional[str] = None) -> List[OHLCV]:
    """Fetch OHLCV data functionally with timeframe-compliant history query."""
    norm_symbol = resolve_symbol(symbol, market=market)
    ticker = yf.Ticker(norm_symbol)
    
    period = _get_period_for_interval(interval)
    df = ticker.history(period=period, interval=interval)
    
    if df.empty:
        # Fallback candidate check if original symbol was naked
        resolved = resolve_symbol(symbol, market=market)
        if resolved != norm_symbol:
            ticker = yf.Ticker(resolved)
            df = ticker.history(period=period, interval=interval)

    if df.empty:
        return []
        
    # Get the last 'limit' rows
    df = df.tail(limit)
    
    results = []
    for index, row in df.iterrows():
        dt = index.to_pydatetime()
        results.append(OHLCV(
            timestamp=dt,
            open=float(row['Open']),
            high=float(row['High']),
            low=float(row['Low']),
            close=float(row['Close']),
            volume=int(row['Volume']) if 'Volume' in row and not pd.isna(row['Volume']) else 0
        ))
    return results

def get_latest_price(symbol: str, market: Optional[str] = None) -> float:
    """Fetch the latest price functionally."""
    norm_symbol = resolve_symbol(symbol, market=market)
    ticker = yf.Ticker(norm_symbol)
    df = ticker.history(period="5d")
    if df.empty:
        # If naked ticker failed, try resolving with fallback candidates
        resolved = resolve_symbol(symbol, market=market)
        if resolved != norm_symbol:
            ticker = yf.Ticker(resolved)
            df = ticker.history(period="5d")
    if df.empty:
        raise ValueError(f"No price data found for {symbol}")
    return float(df['Close'].iloc[-1])

def validate_symbol(symbol: str, market: Optional[str] = None) -> bool:
    """Validate symbol functionally across global markets with smart fallbacks."""
    try:
        norm_symbol = resolve_symbol(symbol, market=market)
        ticker = yf.Ticker(norm_symbol)
        df = ticker.history(period="5d")
        return not df.empty
    except Exception:
        return False
