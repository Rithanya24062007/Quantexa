import pytest
import pandas as pd
import numpy as np
from app.schemas import BacktestRequest
from app.services.backtester import run_backtest_simulation

def test_sma_crossover_backtest():
    dates = pd.date_range("2020-01-01", periods=200)
    # Sine wave prices to produce clean crossover signals
    prices = 100 + 20 * np.sin(np.linspace(0, 4 * np.pi, 200))
    df = pd.DataFrame({
        'open': prices,
        'high': prices + 2,
        'low': prices - 2,
        'close': prices,
        'volume': 1000000
    }, index=dates)

    req = BacktestRequest(
        symbol="NVDA",
        strategy_type="sma_crossover",
        initial_capital=100000.0,
        commission_bps=10.0,
        fast_period=10,
        slow_period=30
    )

    result = run_backtest_simulation(df, req)
    assert result.symbol == "NVDA"
    assert result.initial_capital == 100000.0
    assert len(result.equity_curve) == 200
    assert result.trade_summary["total_trades"] > 0
    assert result.strategy_metrics.sharpe_ratio is not None
