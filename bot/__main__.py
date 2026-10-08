"""CLI.

    python -m bot download --symbol BTC/USD --timeframe 1d --since 2019-01-01
    python -m bot backtest --data data/coinbase_BTC-USD_1d.csv
    python -m bot backtest --synthetic          # offline demo on fake data
    python -m bot walkforward --data data/coinbase_BTC-USD_1d.csv
"""

from __future__ import annotations

import argparse

import pandas as pd

from bot import backtest, data
from bot.strategies import STRATEGIES


def cmd_download(args):
    path = data.download(args.symbol, args.timeframe, args.since, args.exchange)
    print(f"saved {len(data.load(path))} candles to {path}")


def cmd_backtest(args):
    df = data.synthetic() if args.synthetic else data.load(args.data)
    names = [args.strategy] if args.strategy else list(STRATEGIES)
    kw = dict(cash=args.cash, fee=args.fee, stop_loss=args.stop or None)
    cut = df.index[int(len(df) * args.split)]
    print(f"{len(df)} bars {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}; "
          f"tune before {cut:%Y-%m-%d}, test after. fee={args.fee:.2%} cash={args.cash}\n")

    rows = []
    for name in names:
        fn, grid = STRATEGIES[name]
        # the baseline holds through everything; a stop would knock it out for good
        skw = {**kw, "stop_loss": None} if name == "buy_hold" else kw
        train, test = backtest.train_test(df, fn, grid, split=args.split, **skw)
        rows.append({"strategy": name, "params": train.params,
                     **{f"train_{k}": v for k, v in train.summary().items() if k in ("return", "sharpe")},
                     **{f"test_{k}": v for k, v in test.summary().items()}})
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(pd.DataFrame(rows).set_index("strategy").to_string())
    print("\nJudge strategies on the test_ columns only; train_ numbers are tuned and flattering.")


def cmd_walkforward(args):
    df = data.synthetic() if args.synthetic else data.load(args.data)
    names = [args.strategy] if args.strategy else list(STRATEGIES)
    kw = dict(cash=args.cash, fee=args.fee, stop_loss=args.stop or None)
    print(f"{len(df)} bars {df.index[0]:%Y-%m-%d} to {df.index[-1]:%Y-%m-%d}; re-tune on {args.train} bars, "
          f"score the next {args.test}, repeat. fee={args.fee:.2%} cash={args.cash}\n")
    hold = backtest.walk_forward(df, *STRATEGIES["buy_hold"], args.train, args.test, **{**kw, "stop_loss": None})
    rows = []
    for name in names:
        fn, grid = STRATEGIES[name]
        skw = {**kw, "stop_loss": None} if name == "buy_hold" else kw
        wf = backtest.walk_forward(df, fn, grid, args.train, args.test, **skw)
        rows.append({"strategy": name, "windows": len(wf),
                     "compounded": f"{(1 + wf['return']).prod() - 1:+.1%}",
                     "median_window": f"{wf['return'].median():+.1%}",
                     "worst_window": f"{wf['return'].min():+.1%}",
                     "beat_hold": f"{(wf['return'] > hold['return']).sum()}/{len(wf)}",
                     "worst_dd": f"{wf['max_dd'].min():.1%}",
                     "trades": int(wf["trades"].sum()), "fees": round(wf["fees"].sum(), 2)})
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(pd.DataFrame(rows).set_index("strategy").to_string())


def main():
    p = argparse.ArgumentParser(prog="bot")
    sub = p.add_subparsers(required=True)

    d = sub.add_parser("download", help="fetch candles to data/")
    d.add_argument("--symbol", default="BTC/USD")
    d.add_argument("--timeframe", default="1d")
    d.add_argument("--since", default="2019-01-01")
    d.add_argument("--exchange", default="coinbase")
    d.set_defaults(func=cmd_download)

    b = sub.add_parser("backtest", help="tune on early data, score on later data")
    src = b.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="CSV from `download`")
    src.add_argument("--synthetic", action="store_true", help="use fake random-walk data")
    b.add_argument("--strategy", choices=list(STRATEGIES))
    b.add_argument("--cash", type=float, default=250.0)
    b.add_argument("--fee", type=float, default=backtest.TAKER_FEE)
    b.add_argument("--stop", type=float, default=0.15, help="stop-loss fraction, 0 to disable")
    b.add_argument("--split", type=float, default=0.7)
    b.set_defaults(func=cmd_backtest)

    w = sub.add_parser("walkforward", help="re-tune and score over many rolling out-of-sample windows")
    src = w.add_mutually_exclusive_group(required=True)
    src.add_argument("--data", help="CSV from `download`")
    src.add_argument("--synthetic", action="store_true", help="use fake random-walk data")
    w.add_argument("--strategy", choices=list(STRATEGIES))
    w.add_argument("--cash", type=float, default=250.0)
    w.add_argument("--fee", type=float, default=backtest.TAKER_FEE)
    w.add_argument("--stop", type=float, default=0.15, help="stop-loss fraction, 0 to disable")
    w.add_argument("--train", type=int, default=730, help="bars to tune on per window")
    w.add_argument("--test", type=int, default=180, help="bars to score per window")
    w.set_defaults(func=cmd_walkforward)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
