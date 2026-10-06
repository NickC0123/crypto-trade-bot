# Crypto Trade Bot

A disciplined experiment: find out whether a simple strategy beats holding BTC on Coinbase
**after fees**, before risking any real money. See the staged plan in the project's `plan/game-plan.md`.

Current stage: **backtesting**. Nothing here places orders or needs API keys.

## Setup
```
pip install -r requirements.txt
python -m pytest
```

## Usage
```
python -m bot download --symbol BTC/USD --timeframe 1d --since 2019-01-01   # public candles, no key
python -m bot backtest --data data/coinbase_BTC-USD_1d.csv
python -m bot backtest --synthetic                                         # offline demo on fake data
```

`backtest` tunes each strategy's parameters on the first 70% of history and reports how the
winning settings did on the remaining 30% it never saw. Only the `test_` columns count.

## Assumptions
- Spot only, long only, no leverage. Starting cash 250.
- Coinbase Advanced lowest tier taker fee 0.60% per side, plus 0.1% slippage.
- Signals use data up to a bar's close and fill at the next bar's open.
- 15% stop-loss per position (`--stop 0` to disable); after a stop, the strategy waits for a fresh entry signal.

## Layout
- `bot/data.py` downloads, caches and loads candles; also generates synthetic data for tests.
- `bot/strategies.py` holds strategies as pure functions: candles to target exposure (0 to 1).
- `bot/backtest.py` is the simulator, parameter sweep and train/test split.
