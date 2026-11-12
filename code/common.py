"""Shared data preparation for the ARU study (Bloomberg daily data)."""
import numpy as np, pandas as pd, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_calendar():
    """All exchange days in the sample, including the days on which ARU was in a trading halt (no close)."""
    df = pd.read_csv(os.path.join(ROOT, "data", "aru_bloomberg_daily.csv"), parse_dates=["date"]).set_index("date").sort_index()
    return df


def load():
    """Trading days only, with log-returns in per cent.

    r     ARU close-to-close log-return (measured from the last close before a trading halt)
    m     All Ordinaries log-return over the same interval;  rf  risk-free return over the same interval
    mid   mid-point of the closing bid and ask (when bid < ask);  rmid  log-return of the mid-point
    hs    quoted half-spread as a percentage of the mid-point;    tick  minimum price step at the close
    """
    cal = load_calendar()
    cal["rfcum"] = cal.rf_pct.cumsum()
    d = cal[cal.aru.notna()].copy()                    # drop trading-halt days
    d["r"] = 100 * np.log(d.aru).diff()
    d["m"] = 100 * np.log(d.xao).diff()
    d["rf"] = d.rfcum.diff()
    ok = d.bid.notna() & d.ask.notna() & (d.ask > d.bid + 1e-12)
    d["mid"] = np.where(ok, (d.bid + d.ask) / 2, np.nan)
    d["rmid"] = 100 * np.log(d.mid).diff()            # NaN unless both days have valid quotes
    d["hs"] = 100 * (d.ask - d.bid) / (2 * d.mid)
    d["tick"] = np.where(d.aru < 0.10 - 1e-9, 0.001, 0.005)
    d["rel_tick"] = 100 * d.tick / d.aru
    d["qtick"] = np.where(d.bid < 0.10 - 1e-9, 0.001, 0.005)
    d["spread_ticks"] = np.where(ok, np.rint((d.ask - d.bid) / d.qtick), np.nan)
    return d.iloc[1:]
