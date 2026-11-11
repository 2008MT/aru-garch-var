"""Shared data preparation for the ARU.AX study."""
import numpy as np, pandas as pd, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
STALE = [("2022-03-17", "2022-04-22"), ("2022-10-07", "2022-10-20")]   # runs of identical closes


def load(clean=True):
    """Return a DataFrame indexed by date with prices, log-returns (percent) and excess returns.

    clean=True  : drop the duplicated row and set the returns contaminated by stale quotes to NaN
    clean=False : the price file as supplied (duplicate row dropped only)
    """
    df = pd.read_csv(os.path.join(ROOT, "data", "aru_xao_daily.csv"), parse_dates=["date"])
    # second 2022-12-01 row repeats the index level of 2022-11-28 -> spurious
    df = df[~df.date.duplicated(keep="first")].set_index("date").sort_index()
    df["r"] = 100 * np.log(df.aru).diff()          # ARU log-return, percent
    df["m"] = 100 * np.log(df.xao).diff()          # market log-return, percent
    df["rf"] = df.rf_pct                            # daily risk-free rate, already in percent
    df["stale"] = False
    if clean:
        for a, b in STALE:
            # returns inside the flat block and the first 'catch-up' return after it
            idx = df.index[(df.index > a) & (df.index <= b)]
            nxt = df.index[df.index > b][0]
            df.loc[idx.append(pd.DatetimeIndex([nxt])), "stale"] = True
        df.loc[df.stale, ["r", "m"]] = np.nan
    df["tick"] = np.where(df.aru < 0.10 - 1e-12, 0.001, 0.005)
    df["rel_tick"] = 100 * df.tick / df.aru
    return df.iloc[1:]
