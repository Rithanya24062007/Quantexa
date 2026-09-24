import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
from app.schemas import BacktestRequest, BacktestResponse, QuantMetricsSummary, EquityCurvePoint, TradeLogItem
from app.services.quant_engine import (
    calculate_sma, calculate_ema, calculate_rsi, 
    calculate_bollinger_bands, compute_quant_metrics, calculate_drawdown_series, TRADING_DAYS_PER_YEAR
)

def run_backtest_simulation(df: pd.DataFrame, req: BacktestRequest) -> BacktestResponse:
    """
    Simulates quantitative trading strategy bar-by-bar with realistic portfolio equity curve tracking,
    transaction costs, position sizing, entry/exit prices, and benchmark comparison.
    """
    if df.empty or len(df) < 50:
        raise ValueError("Insufficient historical data for backtesting (minimum 50 bars required).")

    df_strat = df.copy()
    prices = df_strat['close'].values
    dates = [idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx) for idx in df_strat.index]
    n_bars = len(prices)

    # 1. Generate Strategy Signals (1 = LONG, 0 = CASH)
    signals = np.zeros(n_bars, dtype=int)
    strat_type = req.strategy_type.lower()

    if strat_type == "sma_crossover":
        sma_fast = calculate_sma(df_strat['close'], req.fast_period).values
        sma_slow = calculate_sma(df_strat['close'], req.slow_period).values
        for i in range(1, n_bars):
            if not np.isnan(sma_fast[i]) and not np.isnan(sma_slow[i]):
                if sma_fast[i] > sma_slow[i]:
                    signals[i] = 1
                else:
                    signals[i] = 0

    elif strat_type == "ema_trend":
        ema_fast = calculate_ema(df_strat['close'], req.fast_period).values
        ema_slow = calculate_ema(df_strat['close'], req.slow_period).values
        for i in range(1, n_bars):
            if not np.isnan(ema_fast[i]) and not np.isnan(ema_slow[i]):
                if prices[i] > ema_fast[i] and ema_fast[i] > ema_slow[i]:
                    signals[i] = 1
                else:
                    signals[i] = 0

    elif strat_type == "momentum":
        rsi = calculate_rsi(df_strat['close'], req.momentum_period).values
        current_pos = 0
        for i in range(1, n_bars):
            if rsi[i] < 35:  # Oversold momentum entry
                current_pos = 1
            elif rsi[i] > 65:  # Overbought exit
                current_pos = 0
            signals[i] = current_pos

    elif strat_type == "mean_reversion":
        sma, upper, lower = calculate_bollinger_bands(df_strat['close'], req.bb_period, req.bb_std_dev)
        upper_v, lower_v = upper.values, lower.values
        current_pos = 0
        for i in range(1, n_bars):
            if not np.isnan(lower_v[i]):
                if prices[i] <= lower_v[i]:
                    current_pos = 1  # Buy at lower band
                elif prices[i] >= upper_v[i]:
                    current_pos = 0  # Sell at upper band
            signals[i] = current_pos
    else:
        # Default fallback SMA crossover
        sma_fast = calculate_sma(df_strat['close'], 20).values
        sma_slow = calculate_sma(df_strat['close'], 50).values
        for i in range(1, n_bars):
            if not np.isnan(sma_fast[i]) and not np.isnan(sma_slow[i]):
                signals[i] = 1 if sma_fast[i] > sma_slow[i] else 0

    # 2. Portfolio Simulation Engine
    initial_cash = req.initial_capital
    cash = initial_cash
    position = 0.0  # Units of asset held
    commission_rate = req.commission_bps / 10000.0  # 10 bps = 0.0010
    pos_pct = min(max(req.position_size_pct / 100.0, 0.1), 1.0)

    strategy_equity = np.zeros(n_bars, dtype=float)
    benchmark_equity = np.zeros(n_bars, dtype=float)
    
    trades: List[TradeLogItem] = []
    trade_id_counter = 1
    
    in_trade = False
    entry_price = 0.0
    entry_date = ""
    entry_bar_idx = 0
    trade_units = 0.0
    trade_entry_cash_spent = 0.0

    # Benchmark: Buy and Hold on Day 0
    benchmark_units = (initial_cash * (1 - commission_rate)) / prices[0]

    for i in range(n_bars):
        current_price = prices[i]
        current_date = dates[i]

        target_signal = signals[i]

        # Trade Execution Logic
        if not in_trade and target_signal == 1:
            # Enter BUY Position
            allocatable_cash = cash * pos_pct
            entry_price = current_price
            comm = allocatable_cash * commission_rate
            trade_units = (allocatable_cash - comm) / entry_price
            cash -= allocatable_cash
            position += trade_units
            
            in_trade = True
            entry_date = current_date
            entry_bar_idx = i
            trade_entry_cash_spent = allocatable_cash

        elif in_trade and target_signal == 0:
            # Exit SELL Position
            exit_price = current_price
            gross_proceeds = trade_units * exit_price
            exit_comm = gross_proceeds * commission_rate
            net_proceeds = gross_proceeds - exit_comm
            
            gross_pnl = gross_proceeds - trade_entry_cash_spent
            total_comm = (trade_entry_cash_spent * commission_rate) + exit_comm
            net_pnl = net_proceeds - trade_entry_cash_spent
            net_return_pct = (net_pnl / trade_entry_cash_spent) * 100.0 if trade_entry_cash_spent > 0 else 0.0

            cash += net_proceeds
            position -= trade_units

            trades.append(TradeLogItem(
                trade_id=trade_id_counter,
                entry_date=entry_date,
                exit_date=current_date,
                type="BUY",
                entry_price=round(float(entry_price), 2),
                exit_price=round(float(exit_price), 2),
                position_size=round(float(trade_units), 4),
                gross_pnl=round(float(gross_pnl), 2),
                commission=round(float(total_comm), 2),
                net_pnl=round(float(net_pnl), 2),
                net_return_pct=round(float(net_return_pct), 2),
                holding_days=i - entry_bar_idx,
                exit_reason="Signal Change (Bearish/Exit)"
            ))
            trade_id_counter += 1
            in_trade = False

        # Portfolio Valuation
        current_strat_val = cash + (position * current_price)
        current_bench_val = benchmark_units * current_price
        
        strategy_equity[i] = current_strat_val
        benchmark_equity[i] = current_bench_val

    # If still in trade at the last bar, close out for reporting
    if in_trade:
        exit_price = prices[-1]
        gross_proceeds = trade_units * exit_price
        exit_comm = gross_proceeds * commission_rate
        net_proceeds = gross_proceeds - exit_comm
        gross_pnl = gross_proceeds - trade_entry_cash_spent
        total_comm = (trade_entry_cash_spent * commission_rate) + exit_comm
        net_pnl = net_proceeds - trade_entry_cash_spent
        net_return_pct = (net_pnl / trade_entry_cash_spent) * 100.0

        trades.append(TradeLogItem(
            trade_id=trade_id_counter,
            entry_date=entry_date,
            exit_date=dates[-1],
            type="BUY",
            entry_price=round(float(entry_price), 2),
            exit_price=round(float(exit_price), 2),
            position_size=round(float(trade_units), 4),
            gross_pnl=round(float(gross_pnl), 2),
            commission=round(float(total_comm), 2),
            net_pnl=round(float(net_pnl), 2),
            net_return_pct=round(float(net_return_pct), 2),
            holding_days=(n_bars - 1) - entry_bar_idx,
            exit_reason="End of Backtest Period"
        ))

    # 3. Equity Curves & Drawdown Calculations
    strat_returns = pd.Series(strategy_equity).pct_change().fillna(0)
    bench_returns = pd.Series(benchmark_equity).pct_change().fillna(0)
    
    strat_dd = calculate_drawdown_series(strat_returns)
    bench_dd = calculate_drawdown_series(bench_returns)

    equity_curve_points: List[EquityCurvePoint] = []
    for i in range(n_bars):
        equity_curve_points.append(EquityCurvePoint(
            date=dates[i],
            strategy_value=round(float(strategy_equity[i]), 2),
            benchmark_value=round(float(benchmark_equity[i]), 2),
            drawdown=round(float(strat_dd.iloc[i]) * 100, 2),
            benchmark_drawdown=round(float(bench_dd.iloc[i]) * 100, 2),
            position=int(signals[i])
        ))

    # 4. Quantitative Metrics Summary
    df_strat_equity = pd.DataFrame({'close': strategy_equity}, index=df_strat.index)
    df_strat_equity.symbol = f"{req.symbol} ({req.strategy_type.upper()})"
    strat_metrics = compute_quant_metrics(df_strat_equity, price_col='close')

    df_bench_equity = pd.DataFrame({'close': benchmark_equity}, index=df_strat.index)
    df_bench_equity.symbol = f"{req.symbol} (BUY & HOLD)"
    bench_metrics = compute_quant_metrics(df_bench_equity, price_col='close')

    # Trade Summary Statistics
    total_trades = len(trades)
    winning_trades = [t for t in trades if t.net_pnl > 0]
    losing_trades = [t for t in trades if t.net_pnl <= 0]
    
    win_rate = (len(winning_trades) / total_trades * 100.0) if total_trades > 0 else 0.0
    total_gross_gain = sum(t.net_pnl for t in winning_trades)
    total_gross_loss = abs(sum(t.net_pnl for t in losing_trades))
    profit_factor = (total_gross_gain / total_gross_loss) if total_gross_loss > 0 else (99.9 if total_gross_gain > 0 else 0.0)
    
    avg_trade_return = np.mean([t.net_return_pct for t in trades]) if total_trades > 0 else 0.0

    strat_metrics.win_rate_pct = round(win_rate, 2)

    trade_summary = {
        "total_trades": total_trades,
        "winning_trades": len(winning_trades),
        "losing_trades": len(losing_trades),
        "win_rate_pct": round(win_rate, 2),
        "profit_factor": round(profit_factor, 2),
        "avg_trade_return_pct": round(float(avg_trade_return), 2),
        "total_commission_paid": round(float(sum(t.commission for t in trades)), 2)
    }

    return BacktestResponse(
        symbol=req.symbol,
        strategy_name=req.strategy_type.upper().replace("_", " "),
        initial_capital=req.initial_capital,
        final_strategy_value=round(float(strategy_equity[-1]), 2),
        final_benchmark_value=round(float(benchmark_equity[-1]), 2),
        strategy_metrics=strat_metrics,
        benchmark_metrics=bench_metrics,
        trade_summary=trade_summary,
        equity_curve=equity_curve_points,
        trades=trades
    )
