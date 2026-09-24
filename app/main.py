from fastapi import FastAPI, Query, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import pandas as pd

from app.schemas import (
    AssetInfo, HistoricalDataResponse, OHLCVBar,
    QuantMetricsSummary, IndicatorSeries, BacktestRequest,
    BacktestResponse, CorrelationMatrixResponse
)
from app.services.data_fetcher import (
    SUPPORTED_ASSETS, fetch_historical_ohlcv, get_multi_asset_aligned_returns
)
from app.services.quant_engine import (
    compute_quant_metrics, generate_indicator_series, compute_cross_asset_correlation
)
from app.services.backtester import run_backtest_simulation

app = FastAPI(
    title="Quantitative Financial Intelligence & Backtesting API",
    description="API for multi-asset quantitative analysis, indicator calculations, cross-asset correlation, and realistic backtesting.",
    version="1.0.0"
)

# Enable CORS for frontend application
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Quantitative Multi-Asset Platform API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/api/assets", response_model=List[AssetInfo])
def get_assets():
    """Returns supported assets list (NVDA, BTC, GOLD)."""
    return [AssetInfo(**info) for info in SUPPORTED_ASSETS.values()]

@app.get("/api/historical", response_model=HistoricalDataResponse)
def get_historical_data(
    symbol: str = Query("NVDA", description="Asset symbol (NVDA, BTC, GOLD)"),
    start_date: str = Query("2020-01-01", description="Start date YYYY-MM-DD"),
    end_date: Optional[str] = Query(None, description="End date YYYY-MM-DD"),
    force_refresh: bool = Query(False, description="Force refresh from yfinance")
):
    """Fetches historical OHLCV data for specified asset."""
    symbol_upper = symbol.upper()
    try:
        df = fetch_historical_ohlcv(symbol_upper, start_date=start_date, end_date=end_date, force_refresh=force_refresh)
        if df.empty:
            raise HTTPException(status_code=404, detail=f"Data not found for symbol '{symbol_upper}'")

        asset_info = SUPPORTED_ASSETS.get(symbol_upper, {
            "symbol": symbol_upper, "name": symbol_upper, "category": "Custom", "ticker": symbol_upper
        })

        bars = []
        for idx, row in df.iterrows():
            date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
            bars.append(OHLCVBar(
                date=date_str,
                open=round(float(row['open']), 2),
                high=round(float(row['high']), 2),
                low=round(float(row['low']), 2),
                close=round(float(row['close']), 2),
                volume=round(float(row['volume']), 2),
                adjusted_close=round(float(row.get('adjusted_close', row['close'])), 2)
            ))

        return HistoricalDataResponse(
            symbol=asset_info["symbol"],
            name=asset_info["name"],
            category=asset_info["category"],
            bars=bars,
            start_date=bars[0].date if bars else start_date,
            end_date=bars[-1].date if bars else end_date,
            total_bars=len(bars)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/indicators")
def get_indicators(
    symbol: str = Query("NVDA"),
    start_date: str = Query("2020-01-01"),
    end_date: Optional[str] = Query(None),
    fast_sma: int = Query(20),
    slow_sma: int = Query(50),
    fast_ema: int = Query(12),
    slow_ema: int = Query(26)
):
    """Returns technical indicator time-series and overall quantitative metrics summary."""
    df = fetch_historical_ohlcv(symbol, start_date=start_date, end_date=end_date)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No historical data for {symbol}")

    df.symbol = symbol
    metrics = compute_quant_metrics(df, price_col='close')
    series = generate_indicator_series(df, fast_sma=fast_sma, slow_sma=slow_sma, fast_ema=fast_ema, slow_ema=slow_ema)

    return {
        "symbol": symbol,
        "metrics": metrics,
        "series": series
    }

@app.post("/api/backtest", response_model=BacktestResponse)
def run_backtest(req: BacktestRequest = Body(...)):
    """Executes quantitative strategy backtest simulation."""
    df = fetch_historical_ohlcv(req.symbol, start_date=req.start_date or "2020-01-01", end_date=req.end_date)
    if df.empty:
        raise HTTPException(status_code=404, detail=f"No data available for symbol '{req.symbol}'")

    try:
        response = run_backtest_simulation(df, req)
        return response
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/correlation", response_model=CorrelationMatrixResponse)
def get_correlation(
    start_date: str = Query("2020-01-01"),
    end_date: Optional[str] = Query(None),
    rolling_window: int = Query(60)
):
    """Calculates cross-asset correlation matrix and rolling correlation for NVDA, BTC, GOLD."""
    aligned_df = get_multi_asset_aligned_returns(symbols=["NVDA", "BTC", "GOLD"], start_date=start_date, end_date=end_date)
    if aligned_df.empty:
        raise HTTPException(status_code=404, detail="Unable to align data across assets.")

    response = compute_cross_asset_correlation(aligned_df, rolling_window=rolling_window)
    return response
