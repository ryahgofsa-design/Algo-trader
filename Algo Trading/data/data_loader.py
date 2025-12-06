import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any, List, Optional

import ccxt
import pandas as pd
import warnings

from config import DATA_DIR
from utils.logger import log

warnings.filterwarnings("ignore", message=".*'H' is deprecated.*")


def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def data_path(pair: str, timeframe: str) -> str:
    safe_pair = pair.replace("/", "_").lower()
    return os.path.join(DATA_DIR, f"{safe_pair}_{timeframe}.csv")


def choose_exchange() -> ccxt.Exchange:
    """Prefer Binance global; fall back to other ccxt exchanges if needed."""
    try:
        inst = ccxt.binance(
            {
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
                "urls": {"api": "https://api.binance.com"},
            }
        )
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


def safe_fetch_ohlcv(
    exchange: ccxt.Exchange, symbol: str, timeframe: str, limit: int = 500, retries: int = 5, backoff: float = 1.5
):
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
