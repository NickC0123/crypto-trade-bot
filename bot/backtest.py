"""Long-only spot backtester with fees, slippage, and a stop-loss.

Defaults model a small Coinbase Advanced Trade account: lowest volume tier is
0.60% taker / 0.40% maker. We assume market (taker) orders to stay conservative.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TAKER_FEE = 0.006
SLIPPAGE = 0.001


@dataclass
class Result:
    equity: pd.Series
    trades: int
    fees_paid: float
    params: dict = field(default_factory=dict)

    @property
    def total_return(self) -> float:
        return self.equity.iloc[-1] / self.equity.iloc[0] - 1

    @property
    def max_drawdown(self) -> float:
        return float((self.equity / self.equity.cummax() - 1).min())

    @property
    def cagr(self) -> float:
        years = (self.equity.index[-1] - self.equity.index[0]).days / 365.25
        return (1 + self.total_return) ** (1 / years) - 1 if years > 0 else 0.0

    @property
    def sharpe(self) -> float:
        r = self.equity.pct_change().dropna()
        return float(r.mean() / r.std() * np.sqrt(365)) if r.std() > 0 else 0.0

    def summary(self) -> dict:
        return {
            "return": f"{self.total_return:+.1%}",
            "cagr": f"{self.cagr:+.1%}",
            "max_dd": f"{self.max_drawdown:.1%}",
            "sharpe": round(self.sharpe, 2),
            "trades": self.trades,
            "fees": round(self.fees_paid, 2),
        }


def run(
    df: pd.DataFrame,
    signal: pd.Series,
    cash: float = 250.0,
    fee: float = TAKER_FEE,
    slippage: float = SLIPPAGE,
    stop_loss: float | None = 0.15,
    min_order: float = 1.0,
) -> Result:
    """Simulate trading `signal` (target exposure 0..1) on `df`.

    Signals act at the next bar's open. The stop-loss is checked against each bar's
    low and, if hit, exits at the stop price; the position stays flat until the
    signal goes to 0 and back to 1.
    """
    target = signal.reindex(df.index).fillna(0).clip(0, 1).shift(1).fillna(0)
    units, entry, stopped = 0.0, 0.0, False
    trades, fees = 0, 0.0
    equity = np.empty(len(df))

    for i, (o, l, c, t) in enumerate(zip(df["open"], df["low"], df["close"], target)):
        if t == 0:
            stopped = False
        want = 0.0 if stopped else t
        value = cash + units * o
        delta = want * value - units * o
        if abs(delta) >= min_order:
            px = o * (1 + slippage) if delta > 0 else o * (1 - slippage)
            if delta > 0:
                spend = min(delta, cash / (1 + fee))
                bought = spend / px
                entry = (entry * units + px * bought) / (units + bought) if units + bought else px
                units += bought
                cash -= spend + spend * fee
                fees += spend * fee
            else:
                sold = min(-delta / o, units)
                proceeds = sold * px
                units -= sold
                cash += proceeds - proceeds * fee
                fees += proceeds * fee
            trades += 1

        if stop_loss and units > 0 and l <= entry * (1 - stop_loss):
            px = min(o, entry * (1 - stop_loss)) * (1 - slippage)
            proceeds = units * px
            cash += proceeds - proceeds * fee
            fees += proceeds * fee
            units, stopped = 0.0, True
            trades += 1

        equity[i] = cash + units * c

    return Result(pd.Series(equity, index=df.index), trades, fees)


def sweep(df: pd.DataFrame, strategy, grid: dict, **kwargs) -> list[Result]:
    """Run every parameter combination in `grid`; results sorted best Sharpe first."""
    results = []
    keys = list(grid)
    for combo in itertools.product(*grid.values()) if grid else [()]:
        params = dict(zip(keys, combo))
        res = run(df, strategy(df, **params), **kwargs)
        res.params = params
        results.append(res)
    return sorted(results, key=lambda r: r.sharpe, reverse=True)


def train_test(df: pd.DataFrame, strategy, grid: dict, split: float = 0.7, **kwargs) -> tuple[Result, Result]:
    """Tune on the first `split` of the data, then score the best params on the unseen rest.

    The test signal is computed on the full history so indicators are warmed up,
    but only test-period bars are traded.
    """
    cut = int(len(df) * split)
    best = sweep(df.iloc[:cut], strategy, grid, **kwargs)[0]
    full_signal = strategy(df, **best.params)
    test = run(df.iloc[cut:], full_signal.iloc[cut:], **kwargs)
    test.params = best.params
    return best, test


def walk_forward(df: pd.DataFrame, strategy, grid: dict, train: int = 730, test: int = 180, **kwargs) -> pd.DataFrame:
    """Re-tune on each rolling `train`-bar window and score on the `test` bars right after it.

    Gives many out-of-sample periods instead of one, so a result can't hinge on a single
    lucky stretch. Each test window starts flat; the position isn't carried between windows.
    """
    rows = []
    for start in range(0, len(df) - train - test + 1, test):
        cut = start + train
        best = sweep(df.iloc[start:cut], strategy, grid, **kwargs)[0]
        signal = strategy(df.iloc[: cut + test], **best.params).iloc[cut:]
        res = run(df.iloc[cut : cut + test], signal, **kwargs)
        rows.append({"start": df.index[cut], "end": df.index[cut + test - 1], "params": best.params,
                     "return": res.total_return, "max_dd": res.max_drawdown, "trades": res.trades,
                     "fees": res.fees_paid})
    return pd.DataFrame(rows)
