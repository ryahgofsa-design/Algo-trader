import json
import os
import time
from datetime import date, datetime
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from config import DATA_DIR, DEFAULT_TIMEFRAME, FEE_RATE, SLIPPAGE, START_BALANCE
from data.data_loader import choose_exchange, ensure_dir, safe_fetch_ohlcv
from strategies.hybrid_strategy import StrategyParams, add_indicators, position_size
from utils.logger import log


def _load_state(state_path: str) -> Tuple[float, Optional[Dict[str, Any]], Optional[pd.Timestamp], Optional[date], float, float, int, int, int, Optional[Tuple[int, int]], float, bool, bool, bool]:
    balance = START_BALANCE
    pos: Optional[Dict[str, Any]] = None
    last_ts = None
    last_day: Optional[date] = None
    day_start_balance = balance
    week_start_balance = balance
    trades_today = 0
    day_losses = 0
    week_losses = 0
    week_index: Optional[Tuple[int, int]] = None
    peak_balance = balance
    paused_daily = False
    paused_weekly = False
    paused_dd = False

    if os.path.exists(state_path):
        try:
            with open(state_path, "r") as f:
                st = json.load(f)
            balance = st.get("balance", balance)
            pos = st.get("pos", None)
            last_ts_str = st.get("last_ts")
            if last_ts_str:
                last_ts = pd.to_datetime(last_ts_str, utc=True)
            last_day_str = st.get("last_day")
            if last_day_str:
                last_day = datetime.fromisoformat(last_day_str).date()
            day_start_balance = st.get("day_start_balance", balance)
            week_start_balance = st.get("week_start_balance", balance)
            trades_today = st.get("trades_today", 0)
            day_losses = st.get("day_losses", 0)
            week_losses = st.get("week_losses", 0)
            wk = st.get("week_index")
            if wk:
                week_index = tuple(wk)
            peak_balance = st.get("peak_balance", balance)
            paused_daily = st.get("paused_daily", False)
            paused_weekly = st.get("paused_weekly", False)
            paused_dd = st.get("paused_dd", False)
        except Exception:
            log("Failed to load previous state; starting fresh.")
    return (
        balance,
        pos,
        last_ts,
        last_day,
        day_start_balance,
        week_start_balance,
        trades_today,
        day_losses,
        week_losses,
        week_index,
        peak_balance,
        paused_daily,
        paused_weekly,
        paused_dd,
    )


def run_paper_trader(args, params: StrategyParams) -> None:
    state_path = os.path.join(DATA_DIR, "trader_state.json")
    daily_log_path = os.path.join(DATA_DIR, "paper_daily_log.csv")
    (
        balance,
        pos,
        last_ts,
        last_day,
        day_start_balance,
        week_start_balance,
        trades_today,
        day_losses,
        week_losses,
        week_index,
        peak_balance,
        paused_daily,
        paused_weekly,
        paused_dd,
    ) = _load_state(state_path)

    ensure_dir(DATA_DIR)
    if not os.path.exists(daily_log_path):
        with open(daily_log_path, "w") as f:
            f.write("date,day_start,day_end,pnl,trades\n")

    log("Starting paper trader. Ctrl+C to stop.")
    exch = choose_exchange()
    params.donchian = 28
    params.sl_mult = 1.8
    params.tp_mult = 4.0

    MAX_ADDS = 1
    ADD_AT_ATR = 1.0
    ADD_SIZE_FRAC = 0.5

    while True:
        try:
            candles = safe_fetch_ohlcv(exch, args.pair, DEFAULT_TIMEFRAME, limit=400, retries=5, backoff=1.5)
            if candles is None:
                log("Failed to fetch klines after retries; sleeping before next attempt")
                time.sleep(args.poll)
                continue
            live_df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
            live_df["timestamp"] = pd.to_datetime(live_df["timestamp"], unit="ms", utc=True)
            live_df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
            live_df.sort_values("timestamp", inplace=True)
            live_df.reset_index(drop=True, inplace=True)
            live_df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
            live_df = add_indicators(live_df, params)
            if len(live_df) < 2:
                time.sleep(args.poll)
                continue
            row = live_df.iloc[-2]
            if last_ts is not None and pd.to_datetime(row["timestamp"], utc=True) == pd.to_datetime(last_ts, utc=True):
                time.sleep(args.poll)
                continue
            last_ts = pd.to_datetime(row["timestamp"], utc=True)

            current_day = row["timestamp"].date()
            current_week = row["timestamp"].isocalendar()[:2]
            if last_day is None:
                last_day = current_day
                day_start_balance = balance
                trades_today = 0
                day_losses = 0
                week_index = current_week
                week_start_balance = balance
                week_losses = 0
            elif current_day != last_day:
                day_pnl = balance - day_start_balance
                with open(daily_log_path, "a") as f:
                    f.write(f"{last_day},{day_start_balance:.2f},{balance:.2f},{day_pnl:.2f},{trades_today}\n")
                last_day = current_day
                day_start_balance = balance
                trades_today = 0
                day_losses = 0
            if week_index is None:
                week_index = current_week
                week_start_balance = balance
                week_losses = 0
            elif current_week != week_index:
                week_index = current_week
                week_start_balance = balance
                week_losses = 0

            peak_balance = max(peak_balance, balance)
            dd_pct = (balance - peak_balance) / peak_balance if peak_balance > 0 else 0
            if not paused_dd and dd_pct <= -0.20:
                paused_dd = True
            if paused_dd and dd_pct >= -0.10:
                paused_dd = False

            daily_pnl_pct = (balance - day_start_balance) / day_start_balance if day_start_balance > 0 else 0
            weekly_pnl_pct = (balance - week_start_balance) / week_start_balance if week_start_balance > 0 else 0
            if day_losses >= 2 or daily_pnl_pct <= -0.04:
                paused_daily = True
            if current_day != last_day:
                paused_daily = False
            if week_losses >= 3 or weekly_pnl_pct <= -0.08:
                paused_weekly = True
            if current_week != week_index:
                paused_weekly = False

            action = "NONE"

            if pos:
                hit = None
                if pos["side"] == "long":
                    if row["Low"] <= pos["sl"]:
                        exit_price = pos["sl"] * (1 - SLIPPAGE)
                        hit = ("SL", exit_price)
                    elif row["High"] >= pos["tp"]:
                        exit_price = pos["tp"] * (1 - SLIPPAGE)
                        hit = ("TP", exit_price)
                    else:
                        if pos.get("adds_done", 0) < MAX_ADDS and row["High"] >= pos["entry"] + row["ATR"] * ADD_AT_ATR:
                            add_price = pos["entry"] + row["ATR"] * ADD_AT_ATR
                            add_qty = pos.get("initial_qty", pos["qty"]) * ADD_SIZE_FRAC
                            pos["qty"] += add_qty
                            pos["adds_done"] = pos.get("adds_done", 0) + 1
                            balance -= add_qty * add_price * FEE_RATE
                            log(f"ADD LONG @ {row['timestamp']} price {add_price:.2f} qty {add_qty:.6f}")
                else:
                    if row["High"] >= pos["sl"]:
                        exit_price = pos["sl"] * (1 + SLIPPAGE)
                        hit = ("SL", exit_price)
                    elif row["Low"] <= pos["tp"]:
                        exit_price = pos["tp"] * (1 + SLIPPAGE)
                        hit = ("TP", exit_price)
                    else:
                        if pos.get("adds_done", 0) < MAX_ADDS and row["Low"] <= pos["entry"] - row["ATR"] * ADD_AT_ATR:
                            add_price = pos["entry"] - row["ATR"] * ADD_AT_ATR
                            add_qty = pos.get("initial_qty", pos["qty"]) * ADD_SIZE_FRAC
                            pos["qty"] += add_qty
                            pos["adds_done"] = pos.get("adds_done", 0) + 1
                            balance -= add_qty * add_price * FEE_RATE
                            log(f"ADD SHORT @ {row['timestamp']} price {add_price:.2f} qty {add_qty:.6f}")
                if hit:
                    label, px = hit
                    if pos["side"] == "long":
                        pnl = (px - pos["entry"]) * pos["qty"]
                    else:
                        pnl = (pos["entry"] - px) * pos["qty"]
                    fee = pos["qty"] * px * FEE_RATE
                    balance += pnl - fee
                    trades_today += 1
                    if pnl - fee < 0:
                        day_losses += 1
                        week_losses += 1
                    action = f"EXIT {pos['side']} {label} @ {px:.2f}"
                    log(f"EXIT @ {row['timestamp']} price {row['Close']:.2f} equity {balance:.2f} action {action}")
                    pos = None
                    time.sleep(args.poll)
                    continue

            if not pos and not (paused_daily or paused_weekly or paused_dd):
                if row["vol_ok"] and (row["ATR"] / row["Close"] > params.atr_floor_mult):
                    rsi_ok_long = (not params.rsi_filter) or (row["RSI"] > 50)
                    macd_ok_long = (not params.macd_filter) or (row["MACD_hist"] > 0)
                    rsi_ok_short = (not params.rsi_filter) or (row["RSI"] < 50)
                    macd_ok_short = (not params.macd_filter) or (row["MACD_hist"] < 0)
                    wr_ok_long = (not params.use_wr_filter) or (row["WR14"] < params.wr_long_thresh)
                    wr_ok_short = (not params.use_wr_filter) or (row["WR14"] > params.wr_short_thresh)

                    if row["regime_long"] and row["Close"] > row["Donchian_H"] and rsi_ok_long and macd_ok_long and wr_ok_long:
                        qty = position_size(balance, row["ATR"], params.sl_mult, params.risk_pct)
                        if qty > 0:
                            entry = row["Close"] * (1 + SLIPPAGE)
                            balance -= qty * entry * FEE_RATE
                            sl = entry - row["ATR"] * params.sl_mult
                            tp = entry + row["ATR"] * params.tp_mult
                            pos = {"side": "long", "entry": entry, "qty": qty, "initial_qty": qty, "adds_done": 0, "sl": sl, "tp": tp}
                            trades_today += 1
                            action = f"ENTRY LONG @ {entry:.2f}"
                            log(f"ENTER LONG entry {entry:.2f} sl {sl:.2f} tp {tp:.2f}")
                    elif row["regime_short"] and row["Close"] < row["Donchian_L"] and rsi_ok_short and macd_ok_short and wr_ok_short:
                        qty = position_size(balance, row["ATR"], params.sl_mult, params.risk_pct)
                        if qty > 0:
                            entry = row["Close"] * (1 - SLIPPAGE)
                            balance -= qty * entry * FEE_RATE
                            sl = entry + row["ATR"] * params.sl_mult
                            tp = entry - row["ATR"] * params.tp_mult
                            pos = {"side": "short", "entry": entry, "qty": qty, "initial_qty": qty, "adds_done": 0, "sl": sl, "tp": tp}
                            trades_today += 1
                            action = f"ENTRY SHORT @ {entry:.2f}"
                            log(f"ENTER SHORT entry {entry:.2f} sl {sl:.2f} tp {tp:.2f}")

            status = "FLAT" if not pos else f"IN {pos['side'].upper()} entry {pos['entry']:.2f} sl {pos['sl']:.2f} tp {pos['tp']:.2f}"
            if paused_dd or paused_daily or paused_weekly:
                pause_reason = []
                if paused_dd:
                    pause_reason.append("DD>20%")
                if paused_daily:
                    pause_reason.append("Daily guard")
                if paused_weekly:
                    pause_reason.append("Weekly guard")
                status += f" | PAUSED ({'/'.join(pause_reason)})"
            log(
                f"NEW {DEFAULT_TIMEFRAME.upper()} CANDLE @ {row['timestamp']} price {row['Close']:.2f} equity {balance:.2f} action {action} status {status}"
            )

            state = {
                "balance": balance,
                "pos": pos,
                "last_ts": last_ts.isoformat() if last_ts is not None else None,
                "last_day": last_day.isoformat() if last_day else None,
                "day_start_balance": day_start_balance,
                "week_start_balance": week_start_balance,
                "trades_today": trades_today,
                "day_losses": day_losses,
                "week_losses": week_losses,
                "week_index": list(week_index) if week_index else None,
                "peak_balance": peak_balance,
                "paused_daily": paused_daily,
                "paused_weekly": paused_weekly,
                "paused_dd": paused_dd,
            }
            with open(state_path, "w") as f:
                json.dump(state, f)

            time.sleep(args.poll)
        except KeyboardInterrupt:
            log("Paper trader stopped by user.")
            break
        except Exception as e:
            log(f"Paper trader error: {e}")
            time.sleep(args.poll)


__all__ = ["run_paper_trader"]
