import pytest
import pandas as pd
import numpy as np
from app.services.quant_engine import (
    calculate_sma, calculate_ema, calculate_drawdown_series, 
    compute_quant_metrics, calculate_rsi, calculate_bollinger_bands
)

def test_sma_calculation():
    prices = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0])
    sma = calculate_sma(prices, period=3)
    assert np.isnan(sma[0])
    assert np.isnan(sma[1])
    assert pytest.approx(sma[2], 0.01) == 11.0
    assert pytest.approx(sma[4], 0.01) == 13.0

def test_ema_calculation():
    prices = pd.Series([10.0, 12.0, 14.0, 16.0, 18.0])
    ema = calculate_ema(prices, period=3)
    assert len(ema) == 5
    assert ema.iloc[-1] > 15.0

def test_drawdown_calculation():
    returns = pd.Series([0.1, 0.05, -0.2, 0.1, -0.1])
    dd = calculate_drawdown_series(returns)
    assert dd.iloc[2] < 0.0
    assert dd.min() <= -0.15

def test_quant_metrics_summary():
    dates = pd.date_range("2021-01-01", periods=100)
    prices = np.linspace(100, 200, 100)
    df = pd.DataFrame({'close': prices}, index=dates)
    df.symbol = "TEST"

    metrics = compute_quant_metrics(df)
    assert metrics.total_return_pct == 100.0
    assert metrics.cagr_pct > 0.0
    assert metrics.sharpe_ratio != 0.0
    assert metrics.max_drawdown_pct == 0.0  # monotonically increasing
