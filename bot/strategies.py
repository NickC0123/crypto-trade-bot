"""Strategies are pure functions: candles -> target exposure in [0, 1] per bar (0 = cash, 1 = fully in).

The signal on bar t is decided using data up to and including bar t's close; the
backtester fills it at bar t+1's open, so there is no look-ahead.
"""

from __future__ import annotations

import pandas as pd


def buy_hold(df: pd.DataFrame) -> pd.Series:
    return pd.Series(1.0, index=df.index)


def sma_cross(df: pd.DataFrame, fast: int = 20, slow: int = 100) -> pd.Series:
    """Trend following: in the market while the fast moving average is above the slow one."""
    f = df["close"].rolling(fast).mean()
    s = df["close"].rolling(slow).mean()
    return (f > s).astype(float)


def donchian(df: pd.DataFrame, entry: int = 55, exit: int = 20) -> pd.Series:
    """Breakout: buy a new `entry`-bar high, sell a new `exit`-bar low (turtle-style)."""
    hi = df["high"].rolling(entry).max().shift(1)
    lo = df["low"].rolling(exit).min().shift(1)
    pos, out = 0.0, []
    for close, h, l in zip(df["close"], hi, lo):
        if pos == 0 and close > h:
            pos = 1.0
        elif pos == 1 and close < l:
            pos = 0.0
        out.append(pos)
    return pd.Series(out, index=df.index)


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def rsi_reversion(df: pd.DataFrame, period: int = 14, buy_below: float = 30, sell_above: float = 55) -> pd.Series:
    """Mean reversion: buy when oversold, sell once it recovers."""
    r = rsi(df["close"], period)
    pos, out = 0.0, []
    for v in r:
        if pos == 0 and v < buy_below:
            pos = 1.0
        elif pos == 1 and v > sell_above:
            pos = 0.0
        out.append(pos)
    return pd.Series(out, index=df.index)


STRATEGIES = {
    "buy_hold": (buy_hold, {}),
    "sma_cross": (sma_cross, {"fast": [10, 20, 50], "slow": [50, 100, 200]}),
    "donchian": (donchian, {"entry": [20, 55, 100], "exit": [10, 20, 50]}),
    "rsi_reversion": (rsi_reversion, {"buy_below": [20, 25, 30], "sell_above": [50, 55, 65]}),
}
