from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from config import FEE_RATE, SLIPPAGE, START_BALANCE
from strategies.hybrid_strategy import StrategyParams, add_indicators, position_size


@dataclass
class Trade:
    side: str
    result: str
    entry: float
    exit: float
    pnl: float
    balance: float
    timestamp: pd.Timestamp
    risk: float
    r_multiple: float


@dataclass
class BacktestResult:
    final_balance: float
    trades: List[Trade]
    winrate: float
    profit_factor: float
    max_drawdown: float
    expectancy_r: float
    equity_curve: List[float]


def run_backtest(df: pd.DataFrame, p: StrategyParams) -> BacktestResult:
    bal = START_BALANCE
    in_pos = False
    side = ""
    entry = sl = tp = 0.0
    trail_active = False
    qty = 0.0
    risk = 0.0
    trades: List[Trade] = []
    equity: List[float] = []

    for i, row in df.iterrows():
        if i < max(p.donchian, p.vol_sma, 220):
            continue

        if in_pos:
            high = row["High"]
            low = row["Low"]
            hit = None
            exit_price = None

            if side == "long":
                if p.trail_mult > 0:
                    if not trail_active and high >= entry + (entry - sl):
                        trail_active = True
                    if trail_active:
                        trail_sl = max(sl, high - row["ATR"] * p.trail_mult)
                        sl = trail_sl
                if low <= sl:
                    exit_price = sl * (1 - SLIPPAGE)
                    hit = "SL"
                elif high >= tp:
                    exit_price = tp * (1 - SLIPPAGE)
                    hit = "TP"
            else:
                if p.trail_mult > 0:
                    if not trail_active and low <= entry - (sl - entry):
                        trail_active = True
                    if trail_active:
                        trail_sl = min(sl, low + row["ATR"] * p.trail_mult)
                        sl = trail_sl
                if high >= sl:
                    exit_price = sl * (1 + SLIPPAGE)
                    hit = "SL"
                elif low <= tp:
                    exit_price = tp * (1 + SLIPPAGE)
                    hit = "TP"

            if hit and exit_price is not None:
                if side == "long":
                    pnl = (exit_price - entry) * qty
                else:
                    pnl = (entry - exit_price) * qty
                fee = qty * exit_price * FEE_RATE
                bal += pnl - fee
                trades.append(
                    Trade(side.upper(), hit, entry, exit_price, pnl - fee, bal, row["timestamp"], risk, (pnl - fee) / risk if risk else 0)
                )
                in_pos = False
                equity.append(bal)
                continue

        if in_pos:
            equity.append(bal)
            continue

        if (not row["vol_ok"]) or (not row.get("atr_floor_ok", True)):
            equity.append(bal)
            continue

        rsi_ok_long = (not p.rsi_filter) or (row["RSI"] > 50)
        macd_ok_long = (not p.macd_filter) or (row["MACD_hist"] > 0)
        rsi_ok_short = (not p.rsi_filter) or (row["RSI"] < 50)
        macd_ok_short = (not p.macd_filter) or (row["MACD_hist"] < 0)

        pb_ok = True
        if p.pullback_to_ema:
            recent = df.iloc[max(0, i - p.pullback_lookback) : i]
            touched = ((recent["Low"] <= recent["Close"].ewm(span=20).mean()) | (recent["Low"] <= recent["Close"].ewm(span=50).mean())).any()
            pb_ok = touched

        wr_ok_long = (not p.use_wr_filter) or (row["WR14"] < p.wr_long_thresh)
        wr_ok_short = (not p.use_wr_filter) or (row["WR14"] > p.wr_short_thresh)

        if row["regime_long"] and row["Close"] > row["Donchian_H"] and rsi_ok_long and macd_ok_long and wr_ok_long and pb_ok:
            qty = position_size(bal, row["ATR"], p.sl_mult, p.risk_pct)
            if qty <= 0:
                equity.append(bal)
                continue
            entry = row["Close"] * (1 + SLIPPAGE)
            sl = entry - row["ATR"] * p.sl_mult
            tp = entry + row["ATR"] * p.tp_mult
            risk = (entry - sl) * qty
            trail_active = False
            bal -= qty * entry * FEE_RATE
            in_pos = True
            side = "long"
            trades.append(Trade("LONG", "ENTRY", entry, 0.0, 0.0, bal, row["timestamp"], risk, 0.0))
            equity.append(bal)
            continue

        if row["regime_short"] and row["Close"] < row["Donchian_L"] and rsi_ok_short and macd_ok_short and wr_ok_short and pb_ok:
            qty = position_size(bal, row["ATR"], p.sl_mult, p.risk_pct)
            if qty <= 0:
                equity.append(bal)
                continue
            entry = row["Close"] * (1 - SLIPPAGE)
            sl = entry + row["ATR"] * p.sl_mult
            tp = entry - row["ATR"] * p.tp_mult
            risk = (sl - entry) * qty
            trail_active = False
            bal -= qty * entry * FEE_RATE
            in_pos = True
            side = "short"
            trades.append(Trade("SHORT", "ENTRY", entry, 0.0, 0.0, bal, row["timestamp"], risk, 0.0))
            equity.append(bal)
            continue

        equity.append(bal)

    closed = [t for t in trades if t.result in ("TP", "SL")]
    if not closed:
        return BacktestResult(bal, trades, 0.0, 0.0, 0.0, 0.0, equity)

    wins = [t for t in closed if t.result == "TP"]
    winrate = len(wins) / len(closed)
    gross_p = sum(t.pnl for t in wins)
    gross_l = abs(sum(t.pnl for t in closed if t.result == "SL"))
    pf = gross_p / gross_l if gross_l > 0 else float("inf")
    eq = np.array([t.balance for t in closed])
    peaks = np.maximum.accumulate(eq)
    dd = (eq - peaks) / peaks
    max_dd = dd.min() if len(dd) else 0.0
    r_list = [t.r_multiple for t in closed if t.risk]
    expectancy = sum(r_list) / len(r_list) if r_list else 0.0

    return BacktestResult(bal, trades, winrate, pf, max_dd, expectancy, equity)


def print_summary(res: BacktestResult) -> None:
    closed = [t for t in res.trades if t.result in ("TP", "SL")]
    print("\n================ BACKTEST SUMMARY ================")
    print(f"Starting Balance: ${START_BALANCE:.2f}")
    print(f"Final Balance:    ${res.final_balance:.2f}")
    print(f"Closed Trades:    {len(closed)}")
    print(f"Win Rate:         {res.winrate*100:.2f}%")
    print(f"Profit Factor:    {res.profit_factor:.3f}")
    print(f"Max Drawdown:     {res.max_drawdown*100:.2f}%")
    print(f"Expectancy (R):   {res.expectancy_r:.2f}")
    print("==================================================\n")


__all__ = ["Trade", "BacktestResult", "run_backtest", "print_summary"]
