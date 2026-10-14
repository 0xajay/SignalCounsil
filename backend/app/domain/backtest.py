from typing import List, Optional
from app.domain.models import OHLCV, OHLCVData, StrategyContext, Signal, Trade, PerformanceMetrics
from app.domain.strategies import Strategy

class BacktestEngine:
    """
    Backtesting engine that explicitly protects against:
    1. Look-ahead bias & Future leakage (Strategy is strictly passed data sliced up to index i)
    2. Incorrect execution timing (Signals on close of candle 'i' execute at open of candle 'i+1')
    """
    def __init__(
        self, 
        market_data: List[OHLCV], 
        strategy: Strategy, 
        context: StrategyContext,
        initial_capital: float = 10000.0,
        warmup_period: int = 50,
        transaction_cost_pct: float = 0.00075 # 0.075% friction per trade leg (0.15% round trip: 0.10% fees + 0.05% slippage)
    ):
        self.market_data = sorted(market_data, key=lambda x: x.timestamp)
        self.strategy = strategy
        self.context = context
        self.initial_capital = initial_capital
        self.warmup_period = warmup_period
        self.transaction_cost_pct = transaction_cost_pct
        
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []

    def run(self) -> PerformanceMetrics:
        if len(self.market_data) <= self.warmup_period:
            raise ValueError("Not enough data to run backtest beyond warmup period")
            
        capital = self.initial_capital
        current_position: Optional[Trade] = None
        
        for i in range(self.warmup_period, len(self.market_data) - 1):
            # 1. Prevent Look-ahead bias & Future leakage
            # Strategy ONLY sees data up to current index `i` (representing the close of candle i)
            visible_data = OHLCVData(data=self.market_data[:i + 1])
            
            # 2. Evaluate Strategy
            result = self.strategy.evaluate(visible_data, self.context)
            
            # 3. Prevent Incorrect Execution Timing
            # A signal generated at the close of candle `i` can only be realistically executed
            # at the open of the next candle `i+1`.
            next_candle = self.market_data[i + 1]
            execution_price = next_candle.open
            execution_time = next_candle.timestamp
            
            if current_position is None:
                if result.signal == Signal.BUY:
                    current_position = Trade(
                        entry_time=execution_time,
                        entry_price=execution_price,
                        side="LONG"
                    )
            else:
                if current_position.side == "LONG" and result.signal == Signal.SELL:
                    # Close the position with realistic friction
                    current_position.exit_time = execution_time
                    current_position.exit_price = execution_price
                    gross_pnl_pct = (execution_price - current_position.entry_price) / current_position.entry_price
                    # Deduct round-trip friction (entry and exit)
                    net_pnl_pct = (1.0 + gross_pnl_pct) * ((1.0 - self.transaction_cost_pct) ** 2) - 1.0
                    current_position.pnl_pct = net_pnl_pct
                    current_position.pnl = current_position.entry_price * net_pnl_pct
                    
                    self.trades.append(current_position)
                    
                    # Update capital
                    capital = capital * (1 + current_position.pnl_pct)
                    current_position = None
            
            self.equity_curve.append(capital)
            
        # Optional: Close out any remaining position on the last candle
        if current_position is not None:
            final_candle = self.market_data[-1]
            current_position.exit_time = final_candle.timestamp
            current_position.exit_price = final_candle.close
            gross_pnl_pct = (final_candle.close - current_position.entry_price) / current_position.entry_price
            net_pnl_pct = (1.0 + gross_pnl_pct) * ((1.0 - self.transaction_cost_pct) ** 2) - 1.0
            current_position.pnl_pct = net_pnl_pct
            current_position.pnl = current_position.entry_price * net_pnl_pct
            self.trades.append(current_position)
            capital = capital * (1 + current_position.pnl_pct)
            self.equity_curve.append(capital)
            
        return self._calculate_metrics()

    def _calculate_metrics(self) -> PerformanceMetrics:
        if not self.trades:
            return PerformanceMetrics(
                total_return=0.0, cagr=0.0, win_rate=0.0, profit_factor=0.0,
                max_drawdown=0.0, trade_count=0, average_win=0.0, average_loss=0.0
            )
            
        winning_trades = [t for t in self.trades if t.pnl_pct is not None and t.pnl_pct > 0]
        losing_trades = [t for t in self.trades if t.pnl_pct is not None and t.pnl_pct <= 0]
        
        gross_profit = sum(t.pnl_pct for t in winning_trades if t.pnl_pct)
        gross_loss = abs(sum(t.pnl_pct for t in losing_trades if t.pnl_pct))
        
        win_rate = len(winning_trades) / len(self.trades)
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_win = gross_profit / len(winning_trades) if winning_trades else 0.0
        avg_loss = gross_loss / len(losing_trades) if losing_trades else 0.0
        
        # Max Drawdown
        peak = self.initial_capital
        max_dd = 0.0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak
            if dd > max_dd:
                max_dd = dd
                
        # Total Return
        final_equity = self.equity_curve[-1] if self.equity_curve else self.initial_capital
        total_return = (final_equity - self.initial_capital) / self.initial_capital
        
        # CAGR (Compound Annual Growth Rate)
        days = (self.market_data[-1].timestamp - self.market_data[0].timestamp).days
        years = days / 365.25 if days > 0 else 0
        
        if years > 0 and final_equity > 0:
            cagr = ((final_equity / self.initial_capital) ** (1 / years)) - 1
        else:
            cagr = 0.0
            
        return PerformanceMetrics(
            total_return=total_return,
            cagr=cagr,
            win_rate=win_rate,
            profit_factor=profit_factor,
            max_drawdown=max_dd,
            trade_count=len(self.trades),
            average_win=avg_win,
            average_loss=avg_loss
        )
