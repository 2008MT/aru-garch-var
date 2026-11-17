"""Shared data preparation for the ARU study (Bloomberg daily data)."""
import numpy as np, pandas as pd, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


# Reserve Bank of Australia cash rate target (% a year): changes effective on these dates, 2018-2025.
# Source: RBA, "Cash Rate Target", https://www.rba.gov.au/statistics/cash-rate/
RBA_CASH_RATE = [("2016-08-03", 1.50), ("2019-06-05", 1.25), ("2019-07-03", 1.00), ("2019-10-02", 0.75), ("2020-03-04", 0.50),
                 ("2020-03-20", 0.25), ("2020-11-04", 0.10), ("2022-05-04", 0.35), ("2022-06-08", 0.85), ("2022-07-06", 1.35),
                 ("2022-08-03", 1.85), ("2022-09-07", 2.35), ("2022-10-05", 2.60), ("2022-11-02", 2.85), ("2022-12-07", 3.10),
                 ("2023-02-08", 3.35), ("2023-03-08", 3.60), ("2023-05-03", 3.85), ("2023-06-07", 4.10), ("2023-11-08", 4.35),
                 ("2025-02-19", 4.10), ("2025-05-21", 3.85)]


def rba_rf(dates):
    """Daily risk-free return in per cent on each exchange day: the cash rate target accrued over the calendar
    days since the previous exchange day (Actual/365, the Australian money-market convention)."""
    steps = pd.Series({pd.Timestamp(d): r for d, r in RBA_CASH_RATE}).sort_index()
    rate = steps.reindex(steps.index.union(dates)).ffill().reindex(dates)
    days = pd.Series(dates, index=dates).diff().dt.days.fillna(1)
    return rate * days / 365


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
    cal["rf_pct"] = rba_rf(cal.index)                 # Australian risk-free rate (replaces any rf column in the file)
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
