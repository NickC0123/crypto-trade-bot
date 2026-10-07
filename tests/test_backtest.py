import pandas as pd
import pytest

from bot import backtest, data, strategies


@pytest.fixture
def df():
    return data.synthetic(days=600, seed=1)


def flat(n, price=100.0):
    idx = pd.date_range("2024-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open": price, "high": price, "low": price, "close": price, "volume": 1.0}, index=idx)


def test_all_cash_keeps_money(df):
    res = backtest.run(df, pd.Series(0.0, index=df.index), cash=250)
    assert res.trades == 0
    assert res.equity.iloc[-1] == pytest.approx(250)


def test_round_trip_costs_fees_and_slippage():
    df = flat(5)
    sig = pd.Series([1, 1, 0, 0, 0], index=df.index, dtype=float)
    res = backtest.run(df, sig, cash=100, fee=0.006, slippage=0.001, stop_loss=None)
    assert res.trades == 2
    # buy at t1 open, sell at t3 open: lose ~2x fee + 2x slippage
    assert res.equity.iloc[-1] == pytest.approx(100 / (1 + 0.006) * 0.999 / 1.001 * (1 - 0.006), rel=1e-3)
    assert res.fees_paid > 1.1


def test_signal_fills_next_bar_no_lookahead():
    df = flat(4)
    df.loc[df.index[2:], ["open", "high", "low", "close"]] = 200.0  # price jumps after the signal
    sig = pd.Series([0, 1, 0, 0], index=df.index, dtype=float)  # signal on bar 1, filled at bar 2 open
    res = backtest.run(df, sig, cash=100, fee=0, slippage=0, stop_loss=None)
    assert res.equity.iloc[1] == pytest.approx(100)  # still cash on the signal bar
    assert res.equity.iloc[-1] == pytest.approx(100)  # bought and sold at 200, gained nothing


def test_stop_loss_caps_loss():
    df = flat(4)
    df.loc[df.index[2], ["low", "close"]] = [50.0, 60.0]
    df.loc[df.index[3], ["open", "high", "low", "close"]] = 60.0
    sig = pd.Series(1.0, index=df.index)
    res = backtest.run(df, sig, cash=100, fee=0, slippage=0, stop_loss=0.1)
    assert res.equity.iloc[-1] == pytest.approx(90)


@pytest.mark.parametrize("name", list(strategies.STRATEGIES))
def test_strategies_return_valid_exposure(df, name):
    fn, _ = strategies.STRATEGIES[name]
    sig = fn(df)
    assert sig.index.equals(df.index)
    assert sig.between(0, 1).all()


def test_train_test_runs(df):
    fn, grid = strategies.STRATEGIES["sma_cross"]
    train, test = backtest.train_test(df, fn, grid)
    assert train.params == test.params
    assert test.equity.index[0] > train.equity.index[-1]


def test_drop_unfinished_keeps_only_closed_candles():
    day = 86_400_000
    raw = pd.DataFrame({"timestamp": [0, day, 2 * day], "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    assert list(data.drop_unfinished(raw, day, now_ms=2 * day + 5)["timestamp"]) == [0, day]
    assert len(data.drop_unfinished(raw, day, now_ms=3 * day)) == 3
