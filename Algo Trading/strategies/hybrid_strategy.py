from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd

from config import HTF_TIMEFRAME


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

    # Higher timeframe regime
    htf = (
        df[["timestamp", "Close", "High", "Low"]]
        .set_index("timestamp")
        .resample(HTF_TIMEFRAME.upper())
        .agg({"Close": "last", "High": "max", "Low": "min"})
    )
    htf["EMA50"] = ema(htf["Close"], 50)
    htf["EMA200"] = ema(htf["Close"], 200)
    htf["ADX_4h"] = adx(htf.rename(columns={"High": "High", "Low": "Low", "Close": "Close"}), p.adx_len)
    htf = htf.reindex(df["timestamp"]).ffill()
    df["EMA50_4h"] = htf["EMA50"].values
    df["EMA200_4h"] = htf["EMA200"].values
    df["ADX_4h"] = htf["ADX_4h"].values
    df["regime_long"] = (df["Close"] > df["EMA200_4h"]) & (df["EMA50_4h"] > df["EMA200_4h"]) & (
        df["ADX_4h"] >= p.adx_min
    )
    df["regime_short"] = (df["Close"] < df["EMA200_4h"]) & (df["EMA50_4h"] < df["EMA200_4h"]) & (
        df["ADX_4h"] >= p.adx_min
    )
    return df


def position_size(balance: float, atr_val: float, sl_mult: float, risk_pct: float) -> float:
    if atr_val <= 0:
        return 0.0
    risk = balance * risk_pct
    return risk / (atr_val * sl_mult)


__all__ = [
    "StrategyParams",
    "pair_params",
    "add_indicators",
    "position_size",
    "ema",
    "atr",
    "adx",
    "rsi",
    "macd",
    "williams_r",
]
