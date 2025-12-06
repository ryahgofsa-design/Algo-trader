"""Algo trading bot CLI entrypoint."""

import argparse

from config import DEFAULT_PAIR, DEFAULT_TIMEFRAME, LOOKBACK_MONTHS
from agents.backtester import print_summary, run_backtest
from agents.optimizer import grid_search
from agents.paper_trader import run_paper_trader
from data.data_loader import load_data
from strategies.hybrid_strategy import StrategyParams, add_indicators, pair_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Algo trading bot")
    parser.add_argument("mode", nargs="?", default="paper", choices=["backtest", "optimize", "paper"], help="Run mode")
    parser.add_argument("--pair", default=DEFAULT_PAIR)
    parser.add_argument("--months", type=int, default=LOOKBACK_MONTHS)
    parser.add_argument("--poll", type=int, default=60, help="Polling seconds for paper mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = pair_params(args.pair)

    if args.mode in ("backtest", "optimize"):
        df = load_data(args.pair, DEFAULT_TIMEFRAME, args.months)
        df = add_indicators(df, params)

        if args.mode == "backtest":
            res = run_backtest(df, params)
            print_summary(res)
        else:
            results = grid_search(df, params)
            top = results[:5]
            for i, (res, p) in enumerate(top, 1):
                print(f"\nRank {i}")
                print_summary(res)
                print(p)
    else:
        run_paper_trader(args, params)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Stopped by user.")
