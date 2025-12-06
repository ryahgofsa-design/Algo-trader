# Algo Trading Bot

A modular crypto trading bot that supports backtesting, grid-search optimization, and paper trading for a hybrid Donchian/indicator strategy. The project organizes agents, strategies, data utilities, and shared configuration so each workflow can be run from a single CLI entrypoint.

## Project structure
- `algo_trading_bot.py`: CLI entrypoint to run backtests, optimization, or paper trading.
- `agents/`: Execution engines
  - `backtester.py`: Simulates trades with fee/slippage assumptions and reports metrics.
  - `optimizer.py`: Performs grid-search over strategy parameters and ranks results.
  - `paper_trader.py`: Stateful paper trader with persistent balance/position tracking.
- `strategies/`: Strategy logic and indicator helpers (e.g., `hybrid_strategy.py`).
- `data/`: Cached OHLCV CSVs, paper logs, and loader utilities.
- `config.py`: Shared defaults such as pair, timeframes, and cost assumptions.
- `utils/`: Common helpers like logging.

## Requirements
- Python 3.10+
- Dependencies: `pandas`, `numpy`, `ta` (for indicators). Install via:
  ```bash
  pip install -r requirements.txt
  ```
  (If no `requirements.txt` is present, install the packages above manually.)

## Usage
Run commands from the repo root (`Algo Trading`).

### Backtest
Download or place OHLCV data under `data/` named `{pair}_{timeframe}.csv` (e.g., `eth_usdt_1h.csv`). Then run:
```bash
python algo_trading_bot.py backtest --pair ETH/USDT --months 12
```

### Optimize parameters
```bash
python algo_trading_bot.py optimize --pair ETH/USDT --months 12
```
Shows the top parameter sets with summary stats.

### Paper trade
```bash
python algo_trading_bot.py paper --pair ETH/USDT --months 12 --poll 60
```
Uses cached data for indicator context and persists paper-trading state to `data/trader_state.json` after each cycle. Daily guardrails and pyramiding controls are enforced as implemented in `agents/paper_trader.py`.

## Notes
- Default configuration is defined in `config.py`. Adjust values like `DEFAULT_PAIR`, timeframes, risk settings, fees, and slippage to fit your environment.
- Strategy parameters per pair live in `strategies/hybrid_strategy.py` via `pair_params`.
- Logs for paper trading are appended to `data/paper_daily_log.csv`.
- The codebase was refactored to separate agents, strategies, and data utilities to simplify future extensions (e.g., live trading agents).
