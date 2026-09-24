import pandas as pd
import numpy as np
from typing import Dict, List, Any
from app.schemas import QuantMetricsSummary, IndicatorSeries, CorrelationMatrixResponse

TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.02  # 2.0% benchmark risk-free rate

def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculates Simple Moving Average (SMA)."""
    return series.rolling(window=period).mean()

def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculates Exponential Moving Average (EMA)."""
    return series.ewm(span=period, adjust=False).mean()

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Calculates Bollinger Bands (Middle, Upper, Lower)."""
    sma = calculate_sma(series, period)
    rstd = series.rolling(window=period).std()
    upper = sma + (rstd * std_dev)
    lower = sma - (rstd * std_dev)
    return sma, upper, lower

def calculate_drawdown_series(returns_series: pd.Series) -> pd.Series:
    """Calculates continuous drawdown series given daily returns."""
    cum_returns = (1 + returns_series).cumprod()
    peak = cum_returns.cummax()
    drawdown = (cum_returns - peak) / peak
    return drawdown

def compute_quant_metrics(df: pd.DataFrame, price_col: str = 'close', risk_free_rate: float = RISK_FREE_RATE) -> QuantMetricsSummary:
    """
    Computes rigorous quantitative metrics:
    - Total Return & CAGR
    - Annualized Volatility
    - Sharpe Ratio & Sortino Ratio
    - Maximum Drawdown & Calmar Ratio
    """
    if df.empty or len(df) < 2:
        return QuantMetricsSummary(
            symbol="UNKNOWN",
            period_start="",
            period_end="",
            total_return_pct=0.0,
            cagr_pct=0.0,
            annualized_volatility_pct=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown_pct=0.0,
            calmar_ratio=0.0
        )

    prices = df[price_col].values
    daily_returns = pd.Series(prices).pct_change().dropna()
    
    total_bars = len(prices)
    years = total_bars / TRADING_DAYS_PER_YEAR if total_bars >= TRADING_DAYS_PER_YEAR else total_bars / 365.0
    if years <= 0:
        years = 0.001

    start_val = prices[0]
    end_val = prices[-1]
    
    total_return = (end_val - start_val) / start_val
    cagr = (end_val / start_val) ** (1.0 / max(years, 0.01)) - 1.0 if end_val > 0 and start_val > 0 else 0.0

    daily_vol = daily_returns.std()
    ann_vol = daily_vol * np.sqrt(TRADING_DAYS_PER_YEAR) if not np.isnan(daily_vol) else 0.0

    # Sharpe Ratio
    avg_daily_return = daily_returns.mean()
    ann_return = cagr  # using CAGR for compound annualized return
    excess_return = ann_return - risk_free_rate
    sharpe_ratio = excess_return / ann_vol if ann_vol > 0 else 0.0

    # Sortino Ratio (Downside deviation)
    downside_returns = daily_returns[daily_returns < 0]
    downside_std = downside_returns.std() * np.sqrt(TRADING_DAYS_PER_YEAR) if len(downside_returns) > 0 else 0.0
    sortino_ratio = excess_return / downside_std if downside_std > 0 else 0.0

    # Maximum Drawdown
    drawdown_series = calculate_drawdown_series(daily_returns)
    max_dd = abs(drawdown_series.min()) if not drawdown_series.empty else 0.0

    # Calmar Ratio
    calmar_ratio = ann_return / max_dd if max_dd > 0 else 0.0

    symbol_name = getattr(df, 'symbol', 'ASSET')
    start_str = str(df.index[0].strftime('%Y-%m-%d')) if hasattr(df.index[0], 'strftime') else str(df.index[0])
    end_str = str(df.index[-1].strftime('%Y-%m-%d')) if hasattr(df.index[-1], 'strftime') else str(df.index[-1])

    return QuantMetricsSummary(
        symbol=symbol_name,
        period_start=start_str,
        period_end=end_str,
        total_return_pct=round(total_return * 100, 2),
        cagr_pct=round(cagr * 100, 2),
        annualized_volatility_pct=round(ann_vol * 100, 2),
        sharpe_ratio=round(sharpe_ratio, 2),
        sortino_ratio=round(sortino_ratio, 2),
        max_drawdown_pct=round(max_dd * 100, 2),
        calmar_ratio=round(calmar_ratio, 2)
    )

def generate_indicator_series(df: pd.DataFrame, fast_sma: int = 20, slow_sma: int = 50, fast_ema: int = 12, slow_ema: int = 26) -> List[IndicatorSeries]:
    """Generates bar-by-bar indicator series for charting."""
    df_calc = df.copy()
    df_calc['daily_return'] = df_calc['close'].pct_change()
    df_calc['cum_return'] = (1 + df_calc['daily_return'].fillna(0)).cumprod() - 1
    
    df_calc['sma_fast'] = calculate_sma(df_calc['close'], fast_sma)
    df_calc['sma_slow'] = calculate_sma(df_calc['close'], slow_sma)
    df_calc['ema_fast'] = calculate_ema(df_calc['close'], fast_ema)
    df_calc['ema_slow'] = calculate_ema(df_calc['close'], slow_ema)
    
    df_calc['rolling_vol'] = df_calc['daily_return'].rolling(20).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    df_calc['drawdown'] = calculate_drawdown_series(df_calc['daily_return'].fillna(0))

    result = []
    for idx, row in df_calc.iterrows():
        date_str = idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)
        result.append(IndicatorSeries(
            date=date_str,
            close=round(float(row['close']), 2),
            sma_fast=round(float(row['sma_fast']), 2) if not pd.isna(row['sma_fast']) else None,
            sma_slow=round(float(row['sma_slow']), 2) if not pd.isna(row['sma_slow']) else None,
            ema_fast=round(float(row['ema_fast']), 2) if not pd.isna(row['ema_fast']) else None,
            ema_slow=round(float(row['ema_slow']), 2) if not pd.isna(row['ema_slow']) else None,
            daily_return=round(float(row['daily_return']), 4) if not pd.isna(row['daily_return']) else 0.0,
            cumulative_return=round(float(row['cum_return']), 4) if not pd.isna(row['cum_return']) else 0.0,
            rolling_volatility=round(float(row['rolling_vol']) * 100, 2) if not pd.isna(row['rolling_vol']) else None,
            drawdown=round(float(row['drawdown']) * 100, 2) if not pd.isna(row['drawdown']) else 0.0
        ))
    return result

def compute_cross_asset_correlation(aligned_df: pd.DataFrame, rolling_window: int = 60) -> CorrelationMatrixResponse:
    """
    Computes static Pearson correlation matrix and rolling correlation series across assets.
    """
    returns_df = aligned_df.pct_change().dropna()
    corr_matrix = returns_df.corr().to_dict()
    
    # Format correlation matrix values cleanly
    formatted_matrix = {}
    symbols = list(returns_df.columns)
    for s1 in symbols:
        formatted_matrix[s1] = {}
        for s2 in symbols:
            formatted_matrix[s1][s2] = round(float(corr_matrix[s1][s2]), 3)

    # Rolling Correlation between key pairs (NVDA vs BTC, NVDA vs GOLD, BTC vs GOLD)
    rolling_corr_list = []
    if len(returns_df) > rolling_window:
        for i in range(rolling_window, len(returns_df)):
            sub_df = returns_df.iloc[i-rolling_window:i]
            date_str = returns_df.index[i].strftime('%Y-%m-%d')
            sub_corr = sub_df.corr()
            
            entry = {"date": date_str}
            if "NVDA" in symbols and "BTC" in symbols:
                entry["NVDA_BTC"] = round(float(sub_corr.loc["NVDA", "BTC"]), 3)
            if "NVDA" in symbols and "GOLD" in symbols:
                entry["NVDA_GOLD"] = round(float(sub_corr.loc["NVDA", "GOLD"]), 3)
            if "BTC" in symbols and "GOLD" in symbols:
                entry["BTC_GOLD"] = round(float(sub_corr.loc["BTC", "GOLD"]), 3)
            
            rolling_corr_list.append(entry)

    return CorrelationMatrixResponse(
        symbols=symbols,
        matrix=formatted_matrix,
        rolling_correlation=rolling_corr_list
    )
