"""Candle data: download from an exchange, cache to CSV, load, or synthesize for tests."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


def csv_path(exchange: str, symbol: str, timeframe: str) -> Path:
    return DATA_DIR / f"{exchange}_{symbol.replace('/', '-')}_{timeframe}.csv"


def download(
    symbol: str = "BTC/USD",
    timeframe: str = "1d",
    since: str = "2019-01-01",
    exchange: str = "coinbase",
) -> Path:
    """Fetch all candles since `since` via ccxt and write them to data/. Public data, no API key needed."""
    import ccxt  # imported lazily so backtests on cached data don't need it

    # requests_trust_env makes ccxt honour HTTPS_PROXY and REQUESTS_CA_BUNDLE like other tools do
    ex = getattr(ccxt, exchange)({"enableRateLimit": True, "requests_trust_env": True})
    since_ms = ex.parse8601(f"{since}T00:00:00Z")
    step_ms = ex.parse_timeframe(timeframe) * 1000
    rows: list[list] = []
    while since_ms <= ex.milliseconds():
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since_ms, limit=300)
        # an empty batch means no trading in that window (e.g. before the coin was listed): skip past it
        since_ms = batch[-1][0] + step_ms if batch else since_ms + 300 * step_ms
        rows.extend(batch)
        time.sleep(ex.rateLimit / 1000)

    df = drop_unfinished(pd.DataFrame(rows, columns=COLUMNS).drop_duplicates("timestamp"), step_ms, ex.milliseconds())
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    path = csv_path(exchange, symbol, timeframe)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def drop_unfinished(df: pd.DataFrame, step_ms: int, now_ms: int) -> pd.DataFrame:
    """Drop candles that haven't closed yet; the exchange returns today's bar while it is still moving."""
    return df[df["timestamp"] + step_ms <= now_ms]


def load(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df.set_index("timestamp").sort_index()[COLUMNS[1:]]


def synthetic(days: int = 1500, seed: int = 0, start_price: float = 20_000.0) -> pd.DataFrame:
    """Random-walk daily candles with alternating bull/bear regimes. For tests and offline demos only."""
    rng = np.random.default_rng(seed)
    regime_len = 120
    drift = np.repeat(rng.choice([0.004, -0.003, 0.0], size=days // regime_len + 1), regime_len)[:days]
    rets = drift + rng.normal(0, 0.035, days)
    close = start_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[start_price], close[:-1]])
    wick = np.abs(rng.normal(0, 0.01, days))
    idx = pd.date_range("2020-01-01", periods=days, freq="D", tz="UTC", name="timestamp")
    return pd.DataFrame(
        {
            "open": open_,
            "high": np.maximum(open_, close) * (1 + wick),
            "low": np.minimum(open_, close) * (1 - wick),
            "close": close,
            "volume": rng.uniform(100, 1000, days),
        },
        index=idx,
    )
