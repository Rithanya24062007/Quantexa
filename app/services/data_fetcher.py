import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import os
import json

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")
os.makedirs(CACHE_DIR, exist_ok=True)
DB_PATH = os.path.join(CACHE_DIR, "market_data_cache.db")

SUPPORTED_ASSETS = {
    "NVDA": {
        "symbol": "NVDA",
        "name": "NVIDIA Corporation",
        "category": "Equity",
        "ticker": "NVDA",
        "currency": "USD"
    },
    "BTC": {
        "symbol": "BTC",
        "name": "Bitcoin USD",
        "category": "Cryptocurrency",
        "ticker": "BTC-USD",
        "currency": "USD"
    },
    "GOLD": {
        "symbol": "GOLD",
        "name": "SPDR Gold Shares (Gold)",
        "category": "Commodity",
        "ticker": "GLD",
        "currency": "USD"
    }
}

def init_db():
    """Initialize SQLite table for storing historical OHLCV data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ohlcv_cache (
            symbol TEXT,
            date TEXT,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume REAL,
            adjusted_close REAL,
            PRIMARY KEY (symbol, date)
        )
    """)
    conn.commit()
    conn.close()

init_db()

def fetch_historical_ohlcv(symbol: str, start_date: str = "2020-01-01", end_date: str = None, force_refresh: bool = False) -> pd.DataFrame:
    """
    Fetches historical OHLCV data for symbol (NVDA, BTC, GOLD).
    Uses local SQLite cache to avoid unnecessary network calls and rate limits.
    Returns cleaned pandas DataFrame with DatetimeIndex.
    """
    symbol_upper = symbol.upper()
    if symbol_upper not in SUPPORTED_ASSETS:
        # Fallback to direct ticker if given as custom asset
        ticker_symbol = symbol_upper
        asset_info = {"symbol": symbol_upper, "name": symbol_upper, "category": "Custom", "ticker": symbol_upper}
    else:
        asset_info = SUPPORTED_ASSETS[symbol_upper]
        ticker_symbol = asset_info["ticker"]

    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")

    if not force_refresh:
        df_cached = _load_from_cache(symbol_upper, start_date, end_date)
        if not df_cached.empty and len(df_cached) > 50:
            return df_cached

    print(f"[DataFetcher] Fetching fresh data from yfinance for ticker '{ticker_symbol}'...")
    try:
        ticker = yf.Ticker(ticker_symbol)
        # Fetch max or full history back to start_date
        df = ticker.history(start=start_date, end=end_date, auto_adjust=False)
        if df.empty:
            # Retry with period='max'
            df = ticker.history(period="max", auto_adjust=False)
        
        if df.empty:
            raise ValueError(f"No data returned for ticker {ticker_symbol}")

        # Data Cleaning & Normalization
        df.reset_index(inplace=True)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None).dt.strftime('%Y-%m-%d')
        elif 'Datetime' in df.columns:
            df['Date'] = pd.to_datetime(df['Datetime']).dt.tz_localize(None).dt.strftime('%Y-%m-%d')

        # Drop duplicates and sort
        df = df.drop_duplicates(subset=['Date']).sort_values('Date')
        
        # Standardize column names
        rename_dict = {}
        for col in df.columns:
            c_low = col.lower()
            if c_low == 'open': rename_dict[col] = 'open'
            elif c_low == 'high': rename_dict[col] = 'high'
            elif c_low == 'low': rename_dict[col] = 'low'
            elif c_low in ['close', 'adj close']: 
                if c_low == 'close': rename_dict[col] = 'close'
                if c_low == 'adj close': rename_dict[col] = 'adjusted_close'
            elif c_low == 'volume': rename_dict[col] = 'volume'

        df.rename(columns=rename_dict, inplace=True)
        
        if 'adjusted_close' not in df.columns:
            df['adjusted_close'] = df['close']

        required_cols = ['Date', 'open', 'high', 'low', 'close', 'volume', 'adjusted_close']
        df = df[[c for c in required_cols if c in df.columns]]
        
        # Fill missing numeric values
        numeric_cols = ['open', 'high', 'low', 'close', 'adjusted_close', 'volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df[numeric_cols] = df[numeric_cols].ffill().bfill()

        # Save to SQLite Cache
        _save_to_cache(symbol_upper, df)

        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Filter date range
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        df = df[(df.index >= start_dt) & (df.index <= end_dt)]
        
        return df

    except Exception as e:
        print(f"[DataFetcher Error] Failed to fetch live yfinance data for {symbol_upper}: {e}")
        # Try returning whatever is cached
        return _load_from_cache(symbol_upper, start_date, end_date)

def _save_to_cache(symbol: str, df: pd.DataFrame):
    """Save cleaned OHLCV data into SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        records = []
        for _, row in df.iterrows():
            records.append((
                symbol,
                str(row['Date']),
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                float(row['volume']),
                float(row.get('adjusted_close', row['close']))
            ))
        
        conn.executemany("""
            INSERT OR REPLACE INTO ohlcv_cache 
            (symbol, date, open, high, low, close, volume, adjusted_close)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, records)
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[Cache Save Error]: {e}")

def _load_from_cache(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Load cached OHLCV data from SQLite database."""
    try:
        conn = sqlite3.connect(DB_PATH)
        query = """
            SELECT date as Date, open, high, low, close, volume, adjusted_close
            FROM ohlcv_cache
            WHERE symbol = ? AND date >= ? AND date <= ?
            ORDER BY date ASC
        """
        df = pd.read_sql_query(query, conn, params=(symbol, start_date, end_date))
        conn.close()
        
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        return df
    except Exception as e:
        print(f"[Cache Read Error]: {e}")
        return pd.DataFrame()

def get_multi_asset_aligned_returns(symbols: list = None, start_date: str = "2020-01-01", end_date: str = None) -> pd.DataFrame:
    """
    Returns aligned close prices & daily returns dataframe for multiple assets (NVDA, BTC, GOLD).
    Used for cross-asset correlation analysis.
    """
    if not symbols:
        symbols = ["NVDA", "BTC", "GOLD"]

    close_prices = {}
    for s in symbols:
        df = fetch_historical_ohlcv(s, start_date=start_date, end_date=end_date)
        if not df.empty:
            close_prices[s] = df['close']

    df_combined = pd.DataFrame(close_prices)
    df_combined = df_combined.ffill().dropna()
    return df_combined
