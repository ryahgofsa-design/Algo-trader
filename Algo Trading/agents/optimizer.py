from dataclasses import asdict
from typing import List, Tuple

import pandas as pd

from agents.backtester import BacktestResult, run_backtest
from strategies.hybrid_strategy import StrategyParams, add_indicators


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


__all__ = ["grid_search"]
