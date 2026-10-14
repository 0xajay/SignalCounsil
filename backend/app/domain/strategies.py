from typing import Protocol
import pandas as pd
from app.domain.models import OHLCVData, StrategyContext, StrategyResult, Signal
from app.domain.engine import (
    to_dataframe, get_ema, get_rsi, get_bollinger_bands, 
    get_donchian_channels, get_atr
)

class Strategy(Protocol):
    def evaluate(self, market_data: OHLCVData, context: StrategyContext) -> StrategyResult:
        ...

class TrendStrategy:
    def evaluate(self, market_data: OHLCVData, context: StrategyContext) -> StrategyResult:
        df = to_dataframe(market_data.data)
        if len(df) < 50:
            return StrategyResult(strategy_name="Trend", signal=Signal.NEUTRAL, confidence=0.0)
            
        ema_short = get_ema(df, window=20)
        ema_long = get_ema(df, window=50)
        
        last_short = ema_short.iloc[-1]
        last_long = ema_long.iloc[-1]
        
        if pd.isna(last_short) or pd.isna(last_long):
            return StrategyResult(strategy_name="Trend", signal=Signal.NEUTRAL, confidence=0.0)
            
        signal = Signal.BUY if last_short > last_long else Signal.SELL
        confidence = 0.8 if abs(last_short - last_long) / last_long > 0.01 else 0.5
        
        return StrategyResult(
            strategy_name="Trend", 
            signal=signal, 
            confidence=confidence,
            metadata={"ema_20": float(last_short), "ema_50": float(last_long)}
        )

class MomentumStrategy:
    def evaluate(self, market_data: OHLCVData, context: StrategyContext) -> StrategyResult:
        df = to_dataframe(market_data.data)
        if len(df) < 14:
            return StrategyResult(strategy_name="Momentum", signal=Signal.NEUTRAL, confidence=0.0)
            
        rsi = get_rsi(df, window=14)
        last_rsi = rsi.iloc[-1]
        
        if pd.isna(last_rsi):
            return StrategyResult(strategy_name="Momentum", signal=Signal.NEUTRAL, confidence=0.0)
            
        if last_rsi < 30:
            signal = Signal.BUY
            confidence = (30 - last_rsi) / 30.0
        elif last_rsi > 70:
            signal = Signal.SELL
            confidence = (last_rsi - 70) / 30.0
        else:
            signal = Signal.NEUTRAL
            confidence = 0.0
            
        return StrategyResult(
            strategy_name="Momentum", 
            signal=signal, 
            confidence=min(1.0, float(confidence)),
            metadata={"rsi": float(last_rsi)}
        )

class MeanReversionStrategy:
    def evaluate(self, market_data: OHLCVData, context: StrategyContext) -> StrategyResult:
        df = to_dataframe(market_data.data)
        if len(df) < 20:
            return StrategyResult(strategy_name="MeanReversion", signal=Signal.NEUTRAL, confidence=0.0)
            
        bb = get_bollinger_bands(df, window=20)
        close = df['close'].iloc[-1]
        upper = bb['upper'].iloc[-1]
        lower = bb['lower'].iloc[-1]
        
        if pd.isna(upper) or pd.isna(lower):
            return StrategyResult(strategy_name="MeanReversion", signal=Signal.NEUTRAL, confidence=0.0)
            
        if close <= lower:
            signal = Signal.BUY
            confidence = 0.8
        elif close >= upper:
            signal = Signal.SELL
            confidence = 0.8
        else:
            signal = Signal.NEUTRAL
            confidence = 0.0
            
        return StrategyResult(
            strategy_name="MeanReversion", 
            signal=signal, 
            confidence=confidence,
            metadata={"close": float(close), "upper_band": float(upper), "lower_band": float(lower)}
        )

class BreakoutStrategy:
    def evaluate(self, market_data: OHLCVData, context: StrategyContext) -> StrategyResult:
        df = to_dataframe(market_data.data)
        if len(df) < 20:
            return StrategyResult(strategy_name="Breakout", signal=Signal.NEUTRAL, confidence=0.0)
            
        dc = get_donchian_channels(df, window=20)
        close = df['close'].iloc[-1]
        upper = dc['upper'].iloc[-1]
        lower = dc['lower'].iloc[-1]
        
        if pd.isna(upper) or pd.isna(lower):
            return StrategyResult(strategy_name="Breakout", signal=Signal.NEUTRAL, confidence=0.0)
            
        # Breakout assumes trend continuation
        if close >= upper:
            signal = Signal.BUY
            confidence = 0.7
        elif close <= lower:
            signal = Signal.SELL
            confidence = 0.7
        else:
            signal = Signal.NEUTRAL
            confidence = 0.0
            
        return StrategyResult(
            strategy_name="Breakout", 
            signal=signal, 
            confidence=confidence,
            metadata={"close": float(close), "donchian_upper": float(upper), "donchian_lower": float(lower)}
        )

class PriceActionStrategy:
    def evaluate(self, market_data: OHLCVData, context: StrategyContext) -> StrategyResult:
        df = to_dataframe(market_data.data)
        if len(df) < 3:
            return StrategyResult(strategy_name="PriceAction", signal=Signal.NEUTRAL, confidence=0.0)
            
        # Simple higher-highs, higher-lows check for the last 3 candles
        last_3 = df.tail(3)
        highs = last_3['high'].values
        lows = last_3['low'].values
        
        hh = highs[2] > highs[1] > highs[0]
        hl = lows[2] > lows[1] > lows[0]
        
        lh = highs[2] < highs[1] < highs[0]
        ll = lows[2] < lows[1] < lows[0]
        
        if hh and hl:
            signal = Signal.BUY
            confidence = 0.6
        elif lh and ll:
            signal = Signal.SELL
            confidence = 0.6
        else:
            signal = Signal.NEUTRAL
            confidence = 0.0
            
        return StrategyResult(
            strategy_name="PriceAction", 
            signal=signal, 
            confidence=confidence,
            metadata={"higher_highs": bool(hh), "lower_lows": bool(ll)}
        )

class RiskStrategy:
    def evaluate(self, market_data: OHLCVData, context: StrategyContext) -> StrategyResult:
        df = to_dataframe(market_data.data)
        if len(df) < 14:
            return StrategyResult(strategy_name="Risk", signal=Signal.NEUTRAL, confidence=0.0)
            
        atr = get_atr(df, window=14)
        last_atr = atr.iloc[-1]
        close = df['close'].iloc[-1]
        
        if pd.isna(last_atr):
            return StrategyResult(strategy_name="Risk", signal=Signal.NEUTRAL, confidence=0.0)
            
        volatility_pct = (last_atr / close) * 100
        
        # Risk engine strictly evaluates risk regime; it never casts a directional BUY or SELL vote
        if volatility_pct > 5.0:
            regime = "EXTREME"
            risk_status = "ELEVATED_VOLATILITY_RISK"
        elif volatility_pct > 3.0:
            regime = "ELEVATED"
            risk_status = "CAUTION_ELEVATED_VOLATILITY"
        else:
            regime = "NORMAL"
            risk_status = "WITHIN_LIMITS"
            
        return StrategyResult(
            strategy_name="Risk", 
            signal=Signal.NEUTRAL, 
            confidence=0.0,
            metadata={
                "atr": float(last_atr),
                "volatility_pct": float(volatility_pct),
                "volatility_regime": regime,
                "risk_limit_status": risk_status,
                "drawdown_limit_ref": 0.25,
                "stop_loss_multiplier": 2.0
            }
        )
