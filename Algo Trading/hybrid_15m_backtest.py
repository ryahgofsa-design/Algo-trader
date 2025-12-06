"""
1H Turtle Trend System with 4H Regime Filter
Run modes:
  python hybrid_15m_backtest.py backtest --months 12
  python hybrid_15m_backtest.py optimize --months 12
"""

import argparse
import os
import time
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone, timedelta, date
from typing import List, Tuple, Dict, Any, Optional

import warnings
import ccxt
import numpy as np
import pandas as pd

DEFAULT_PAIR = "ETH/USDT"
DEFAULT_TIMEFRAME = "1h"
HTF_TIMEFRAME = "4h"
LOOKBACK_MONTHS = 12
START_BALANCE = 200.0

FEE_RATE = 0.0006
SLIPPAGE = 0.0003
DATA_DIR = "data_hybrid"


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")


def data_path(pair: str, timeframe: str) -> str:
    safe_pair = pair.replace("/", "_").lower()
    return os.path.join(DATA_DIR, f"{safe_pair}_{timeframe}.csv")




def choose_exchange() -> ccxt.Exchange:
    # Prefer Binance Global explicitly to avoid regional binanceus endpoints.
    try:
        inst = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
            "urls": {"api": "https://api.binance.com"},
        })
        inst.load_markets()
        log(f"Using exchange: {inst.id}")
        return inst
    except Exception as e:
        log(f"Binance global init failed: {e}. Falling back to available ccxt exchanges.")
        for ex in ("mexc", "binanceus", "binance"):
            try:
                inst = getattr(ccxt, ex)({"enableRateLimit": True})
                inst.load_markets()
                log(f"Using exchange: {inst.id}")
                return inst
            except Exception:
                continue
    raise RuntimeError("No exchange available via ccxt.")

# Suppress pandas future warning about 'H' alias
warnings.filterwarnings("ignore", message=".*'H' is deprecated.*")


def fetch_ohlcv(pair: str, timeframe: str, months: int) -> pd.DataFrame:
    exchange = choose_exchange()
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=months * 30)
    since = int(start.timestamp() * 1000)

    all_candles: List[List[Any]] = []
    limit = 1000

    while True:
        batch = exchange.fetch_ohlcv(pair, timeframe, since=since, limit=limit)
        if not batch:
            break
        all_candles.extend(batch)
        since = batch[-1][0] + 1
        if len(batch) < limit:
            break
        time.sleep(exchange.rateLimit / 1000)

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    ensure_dir(DATA_DIR)
    path = data_path(pair, timeframe)
    df.to_csv(path, index=False)
    log(f"Saved {len(df)} candles to {path}")
    return df


def safe_fetch_ohlcv(exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 500, retries: int = 5, backoff: float = 1.5):
    """Fetch klines with retries and exponential backoff. Returns list of candles or None on repeated failure."""
    for attempt in range(1, retries + 1):
        try:
            return exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        except Exception as e:
            log(f"fetch_ohlcv error (attempt {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(backoff ** attempt)
            else:
                return None


def load_data(pair: str, timeframe: str, months: int) -> pd.DataFrame:
    path = data_path(pair, timeframe)
    ensure_dir(DATA_DIR)
    if os.path.exists(path):
        log(f"Loading cached data: {path}")
        df = pd.read_csv(path)
    else:
        log("Downloading historical data...")
        df = fetch_ohlcv(pair, timeframe, months)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}, inplace=True)
    return df


def ema(series: pd.Series, length: int) -> pd.Series:
    return series.ewm(span=length, adjust=False).mean()


def atr(df: pd.DataFrame, length: int) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / length, adjust=False).mean()


def adx(df: pd.DataFrame, length: int) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0.0), index=df.index)
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr_val = tr.ewm(alpha=1 / length, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr_val
    minus_di = 100 * minus_dm.ewm(alpha=1 / length, adjust=False).mean() / atr_val
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)) * 100
    return dx.ewm(alpha=1 / length, adjust=False).mean()


def rsi(df: pd.DataFrame, length: int) -> pd.Series:
    close = df["Close"]
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / length, min_periods=length).mean()
    avg_loss = loss.ewm(alpha=1 / length, min_periods=length).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(df: pd.DataFrame, fast: int, slow: int, signal: int):
    close = df["Close"]
    macd_line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def williams_r(df: pd.DataFrame, length: int) -> pd.Series:
    high = df["High"].rolling(length).max()
    low = df["Low"].rolling(length).min()
    wr = -100 * (high - df["Close"]) / (high - low)
    return wr


@dataclass
class StrategyParams:
    donchian: int = 24
    adx_len: int = 14
    adx_min: float = 18.0
    atr_len: int = 14
    sl_mult: float = 1.8
    tp_mult: float = 4.0
    trail_mult: float = 0.0
    risk_pct: float = 0.02
    vol_sma: int = 20
    vol_mult: float = 1.1
    rsi_filter: bool = False
    macd_filter: bool = True
    pullback_to_ema: bool = False
    pullback_lookback: int = 3
    atr_floor_mult: float = 0.01  # ATR/Close minimum
    use_wr_filter: bool = False
    wr_long_thresh: float = -30.0
    wr_short_thresh: float = -70.0


def pair_params(pair: str) -> StrategyParams:
    """Return tuned params (ETH-only baseline)."""
    return StrategyParams(
        donchian=24,
        adx_min=18,
        sl_mult=1.8,
        tp_mult=4.0,
        trail_mult=0.0,
        risk_pct=0.05,
        vol_mult=1.1,
        atr_floor_mult=0.01,
        macd_filter=True,
    )


def add_indicators(df_1h: pd.DataFrame, p: StrategyParams) -> pd.DataFrame:
    df = df_1h.copy()
    df["ATR"] = atr(df, p.atr_len)
    df["ADX"] = adx(df, p.adx_len)
    df["RSI"] = rsi(df, 14)
    df["MACD_line"], df["MACD_signal"], df["MACD_hist"] = macd(df, 12, 26, 9)
    df["WR14"] = williams_r(df, 14)
    df["Donchian_H"] = df["High"].rolling(p.donchian).max().shift(1)
    df["Donchian_L"] = df["Low"].rolling(p.donchian).min().shift(1)
    df["Vol_SMA"] = df["Volume"].rolling(p.vol_sma).mean()
    df["vol_ok"] = df["Volume"] > (df["Vol_SMA"] * p.vol_mult)
    df["atr_floor_ok"] = (df["ATR"] / df["Close"]) > p.atr_floor_mult

    # 4H regime from 1H resample
    htf = (
        df[["timestamp", "Close", "High", "Low"]]
        .set_index("timestamp")
        .resample("4H")
        .agg({"Close": "last", "High": "max", "Low": "min"})
    )
    htf["EMA50"] = ema(htf["Close"], 50)
    htf["EMA200"] = ema(htf["Close"], 200)
    htf["ADX_4h"] = adx(
        htf.rename(columns={"High": "High", "Low": "Low", "Close": "Close"}), p.adx_len
    )
    htf = htf.reindex(df["timestamp"]).ffill()
    df["EMA50_4h"] = htf["EMA50"].values
    df["EMA200_4h"] = htf["EMA200"].values
    df["ADX_4h"] = htf["ADX_4h"].values
    df["regime_long"] = (df["Close"] > df["EMA200_4h"]) & (df["EMA50_4h"] > df["EMA200_4h"]) & (df["ADX_4h"] >= p.adx_min)
    df["regime_short"] = (df["Close"] < df["EMA200_4h"]) & (df["EMA50_4h"] < df["EMA200_4h"]) & (df["ADX_4h"] >= p.adx_min)
    return df


@dataclass
class Trade:
    side: str
    result: str
    entry: float
    exit: float
    pnl: float
    balance: float
    timestamp: datetime
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


def position_size(balance: float, atr_val: float, sl_mult: float, risk_pct: float) -> float:
    if atr_val <= 0:
        return 0.0
    risk = balance * risk_pct
    return risk / (atr_val * sl_mult)


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

        # manage open
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
                trades.append(Trade(side.upper(), hit, entry, exit_price, pnl - fee, bal, row["timestamp"], risk, (pnl - fee) / risk if risk else 0))
                in_pos = False
                equity.append(bal)
                continue

        if in_pos:
            equity.append(bal)
            continue

        # entries
        # entries
        if (not row["vol_ok"]) or (not row.get("atr_floor_ok", True)):
            equity.append(bal)
            continue

        # long breakout
        rsi_ok_long = (not p.rsi_filter) or (row["RSI"] > 50)
        macd_ok_long = (not p.macd_filter) or (row["MACD_hist"] > 0)
        rsi_ok_short = (not p.rsi_filter) or (row["RSI"] < 50)
        macd_ok_short = (not p.macd_filter) or (row["MACD_hist"] < 0)

        pb_ok = True
        if p.pullback_to_ema:
            recent = df.iloc[max(0, i - p.pullback_lookback): i]
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

        # short breakout
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="1H Turtle Trend with 4H filter")
    parser.add_argument("mode", nargs="?", default="paper", choices=["backtest", "optimize", "paper"], help="Run mode")
    parser.add_argument("--pair", default=DEFAULT_PAIR)
    parser.add_argument("--months", type=int, default=LOOKBACK_MONTHS)
    parser.add_argument("--poll", type=int, default=60, help="Polling seconds for paper mode")
    return parser.parse_args()


def grid_search(df: pd.DataFrame, base: StrategyParams) -> List[Tuple[BacktestResult, StrategyParams]]:
    grid = {
        "donchian": [20, 22],
        "sl_mult": [1.5, 1.8],
        "tp_mult": [3.8, 4.0],
        "trail_mult": [0.0],
        "adx_min": [16, 18],
        "vol_mult": [1.0],
        "rsi_filter": [False],
        "macd_filter": [False],
        "pullback_to_ema": [False],
        "atr_floor_mult": [0.008, 0.01],
    }
    keys = list(grid.keys())
    results: List[Tuple[BacktestResult, StrategyParams]] = []

    def recurse(idx: int, cur: StrategyParams):
        if idx == len(keys):
            res = run_backtest(add_indicators(df, cur), cur)
            results.append((res, cur))
            return
        k = keys[idx]
        for val in grid[k]:
            new_params = StrategyParams(**{**asdict(cur), k: val})
            recurse(idx + 1, new_params)

    recurse(0, base)
    results.sort(key=lambda x: (x[0].profit_factor, x[0].winrate), reverse=True)
    return results


def main() -> None:
    args = parse_args()
    params = pair_params(args.pair)
    df = load_data(args.pair, DEFAULT_TIMEFRAME, args.months)
    df = add_indicators(df, params)

    if args.mode == "backtest":
        res = run_backtest(df, params)
        print_summary(res)
    elif args.mode == "optimize":
        results = grid_search(df, params)
        top = results[:5]
        for i, (res, p) in enumerate(top, 1):
            print(f"\nRank {i}")
            print_summary(res)
            print(p)

    elif args.mode == "paper":
        # Simple polling paper trader: one position max, prints each candle decision
        state_path = os.path.join(DATA_DIR, "trader_state.json")
        daily_log_path = os.path.join(DATA_DIR, "paper_daily_log.csv")

        # load state if exists
        balance = START_BALANCE
        pos: Optional[Dict[str, Any]] = None
        last_ts = None
        last_day: Optional[date] = None
        day_start_balance = balance
        week_start_balance = balance
        trades_today = 0
        day_losses = 0
        week_losses = 0
        week_index: Optional[Tuple[int, int]] = None  # (year, week)
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
            except KeyboardInterrupt:
                print("Stopped during state load.")
                return
            except Exception:
                pass

        ensure_dir(DATA_DIR)
        if not os.path.exists(daily_log_path):
            with open(daily_log_path, "w") as f:
                f.write("date,day_start,day_end,pnl,trades\n")

        print("Starting paper trader. Ctrl+C to stop.")
        exch = choose_exchange()
        # Enforce Rank1 strategy baseline for live/paper trading
        params.donchian = 28
        params.sl_mult = 1.8
        params.tp_mult = 4.0
        # Pyramiding settings (1 add at +1 ATR, add size 50% of initial)
        MAX_ADDS = 1
        ADD_AT_ATR = 1.0
        ADD_SIZE_FRAC = 0.5
        while True:
            try:
                candles = safe_fetch_ohlcv(exch, args.pair, DEFAULT_TIMEFRAME, limit=400, retries=5, backoff=1.5)
                if candles is None:
                    # transient failure; wait and retry next loop without freezing indicators
                    log("Failed to fetch klines after retries; sleeping before next attempt")
                    time.sleep(args.poll)
                    continue
                live_df = pd.DataFrame(candles, columns=["timestamp","open","high","low","close","volume"])
                live_df["timestamp"] = pd.to_datetime(live_df["timestamp"], unit="ms", utc=True)
                # Deduplicate and ensure chronological order
                live_df.drop_duplicates(subset=["timestamp"], keep="last", inplace=True)
                live_df.sort_values("timestamp", inplace=True)
                live_df.reset_index(drop=True, inplace=True)
                live_df.rename(columns={"open":"Open","high":"High","low":"Low","close":"Close","volume":"Volume"}, inplace=True)
                live_df = add_indicators(live_df, params)
                # Use the last *closed* candle to match backtest behavior (avoid partial current candle)
                if len(live_df) < 2:
                    time.sleep(args.poll)
                    continue
                row = live_df.iloc[-2]
                # compare closed-candle timestamp for last processed
                if last_ts is not None and pd.to_datetime(row["timestamp"], utc=True) == pd.to_datetime(last_ts, utc=True):
                    time.sleep(args.poll)
                    continue
                last_ts = pd.to_datetime(row["timestamp"], utc=True)

                # roll daily log if new day
                current_day = row["timestamp"].date()
                current_week = row["timestamp"].isocalendar()[:2]  # (year, week)
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
                # new week reset
                if week_index is None:
                    week_index = current_week
                    week_start_balance = balance
                    week_losses = 0
                elif current_week != week_index:
                    week_index = current_week
                    week_start_balance = balance
                    week_losses = 0

                # update peak and DD-based pause
                peak_balance = max(peak_balance, balance)
                dd_pct = (balance - peak_balance) / peak_balance if peak_balance > 0 else 0
                if not paused_dd and dd_pct <= -0.20:
                    paused_dd = True
                if paused_dd and dd_pct >= -0.10:
                    paused_dd = False

                # apply daily/weekly guards
                daily_pnl_pct = (balance - day_start_balance) / day_start_balance if day_start_balance > 0 else 0
                weekly_pnl_pct = (balance - week_start_balance) / week_start_balance if week_start_balance > 0 else 0
                if day_losses >= 2 or daily_pnl_pct <= -0.04:
                    paused_daily = True
                if current_day != last_day:
                    paused_daily = False  # already reset above
                if week_losses >= 3 or weekly_pnl_pct <= -0.08:
                    paused_weekly = True
                if current_week != week_index:
                    paused_weekly = False

                action = "NONE"

                # manage open position
                if pos:
                    hit = None
                    # check exits
                    if pos["side"] == "long":
                        if row["Low"] <= pos["sl"]:
                            exit_price = pos["sl"] * (1 - SLIPPAGE)
                            hit = ("SL", exit_price)
                        elif row["High"] >= pos["tp"]:
                            exit_price = pos["tp"] * (1 - SLIPPAGE)
                            hit = ("TP", exit_price)
                        else:
                            # pyramiding add for long
                            if pos.get("adds_done", 0) < MAX_ADDS and row["High"] >= pos["entry"] + row["ATR"] * ADD_AT_ATR:
                                add_price = pos["entry"] + row["ATR"] * ADD_AT_ATR
                                add_qty = pos.get("initial_qty", pos["qty"]) * ADD_SIZE_FRAC
                                pos["qty"] += add_qty
                                pos["adds_done"] = pos.get("adds_done", 0) + 1
                                balance -= add_qty * add_price * FEE_RATE
                                print(f"=== ADD LONG @ {row['timestamp']} ===")
                                print(f"Add Price: {add_price:.2f} | Add Qty: {add_qty:.6f} | New Qty: {pos['qty']:.6f}")
                    else:
                        if row["High"] >= pos["sl"]:
                            exit_price = pos["sl"] * (1 + SLIPPAGE)
                            hit = ("SL", exit_price)
                        elif row["Low"] <= pos["tp"]:
                            exit_price = pos["tp"] * (1 + SLIPPAGE)
                            hit = ("TP", exit_price)
                        else:
                            # pyramiding add for short
                            if pos.get("adds_done", 0) < MAX_ADDS and row["Low"] <= pos["entry"] - row["ATR"] * ADD_AT_ATR:
                                add_price = pos["entry"] - row["ATR"] * ADD_AT_ATR
                                add_qty = pos.get("initial_qty", pos["qty"]) * ADD_SIZE_FRAC
                                pos["qty"] += add_qty
                                pos["adds_done"] = pos.get("adds_done", 0) + 1
                                balance -= add_qty * add_price * FEE_RATE
                                print(f"=== ADD SHORT @ {row['timestamp']} ===")
                                print(f"Add Price: {add_price:.2f} | Add Qty: {add_qty:.6f} | New Qty: {pos['qty']:.6f}")
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
                        print(f"=== EXIT @ {row['timestamp']} ===")
                        print(f"Price: {row['Close']:.2f}")
                        print(f"Equity: {balance:.2f}")
                        print(f"Action: {action}")
                        pos = None
                        time.sleep(args.poll)
                        continue


                # entry logic (one position max)
                if not pos and not (paused_daily or paused_weekly or paused_dd):
                    if row["vol_ok"] and (row["ATR"]/row["Close"] > params.atr_floor_mult):
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
                                pos = {"side":"long","entry":entry,"qty":qty,"initial_qty":qty,"adds_done":0,"sl":sl,"tp":tp}
                                trades_today += 1
                                action = f"ENTRY LONG @ {entry:.2f}"
                                print(f"Action: ENTER LONG")
                                print(f"Entry: {entry:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
                        elif row["regime_short"] and row["Close"] < row["Donchian_L"] and rsi_ok_short and macd_ok_short and wr_ok_short:
                            qty = position_size(balance, row["ATR"], params.sl_mult, params.risk_pct)
                            if qty > 0:
                                entry = row["Close"] * (1 - SLIPPAGE)
                                balance -= qty * entry * FEE_RATE
                                sl = entry + row["ATR"] * params.sl_mult
                                tp = entry - row["ATR"] * params.tp_mult
                                pos = {"side":"short","entry":entry,"qty":qty,"initial_qty":qty,"adds_done":0,"sl":sl,"tp":tp}
                                trades_today += 1
                                action = f"ENTRY SHORT @ {entry:.2f}"
                                print(f"Action: ENTER SHORT")
                                print(f"Entry: {entry:.2f} | SL: {sl:.2f} | TP: {tp:.2f}")
                # status print each new candle
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
                print(f"=== NEW {DEFAULT_TIMEFRAME.upper()} CANDLE @ {row['timestamp']} ===")
                print(f"Price: {row['Close']:.2f}")
                print(f"Equity: {balance:.2f}")
                print(f"Action: {action if action!='NONE' else 'NONE'}")
                print(f"Status: {status}")

                # persist state
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
                print("Paper trader stopped.")
                break
            except KeyboardInterrupt:
                print("Paper trader stopped.")
                break
            except Exception as e:
                print("Paper trader error:", e)
                time.sleep(args.poll)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped by user.")
