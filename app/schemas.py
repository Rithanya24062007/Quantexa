from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class AssetInfo(BaseModel):
    symbol: str
    name: str
    category: str  # Equity, Crypto, Commodity
    ticker: str
    currency: str = "USD"

class OHLCVBar(BaseModel):
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    adjusted_close: Optional[float] = None

class HistoricalDataResponse(BaseModel):
    symbol: str
    name: str
    category: str
    bars: List[OHLCVBar]
    start_date: str
    end_date: str
    total_bars: int

class IndicatorSeries(BaseModel):
    date: str
    close: float
    sma_fast: Optional[float] = None
    sma_slow: Optional[float] = None
    ema_fast: Optional[float] = None
    ema_slow: Optional[float] = None
    daily_return: Optional[float] = None
    cumulative_return: Optional[float] = None
    rolling_volatility: Optional[float] = None
    drawdown: Optional[float] = None

class QuantMetricsSummary(BaseModel):
    symbol: str
    period_start: str
    period_end: str
    total_return_pct: float
    cagr_pct: float
    annualized_volatility_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    calmar_ratio: float
    win_rate_pct: Optional[float] = None

class BacktestRequest(BaseModel):
    symbol: str = "NVDA"
    strategy_type: str = "sma_crossover"  # sma_crossover, ema_trend, momentum, mean_reversion
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    initial_capital: float = 100000.0
    commission_bps: float = 10.0  # 10 bps = 0.1%
    position_size_pct: float = 100.0  # % of available equity allocated per trade
    
    # Strategy specific parameters
    fast_period: int = 20
    slow_period: int = 50
    momentum_period: int = 14
    bb_period: int = 20
    bb_std_dev: float = 2.0

class TradeLogItem(BaseModel):
    trade_id: int
    entry_date: str
    exit_date: str
    type: str  # BUY or SELL
    entry_price: float
    exit_price: float
    position_size: float
    gross_pnl: float
    commission: float
    net_pnl: float
    net_return_pct: float
    holding_days: int
    exit_reason: str

class EquityCurvePoint(BaseModel):
    date: str
    strategy_value: float
    benchmark_value: float
    drawdown: float
    benchmark_drawdown: float
    position: int  # 1 for long, 0 for flat

class BacktestResponse(BaseModel):
    symbol: str
    strategy_name: str
    initial_capital: float
    final_strategy_value: float
    final_benchmark_value: float
    
    strategy_metrics: QuantMetricsSummary
    benchmark_metrics: QuantMetricsSummary
    
    trade_summary: Dict[str, Any]  # total_trades, win_rate, profit_factor, avg_trade_return, etc.
    equity_curve: List[EquityCurvePoint]
    trades: List[TradeLogItem]

class CorrelationMatrixResponse(BaseModel):
    symbols: List[str]
    matrix: Dict[str, Dict[str, float]]
    rolling_correlation: List[Dict[str, Any]]
