# Data

The price data come from the Bloomberg Terminal and are not redistributed here.

To run the code, place a file `aru_bloomberg_daily.csv` in this folder with one row per exchange
day from 15 August 2018 to 16 June 2025 (1,729 rows) and these columns:

| column   | content                                                              | Bloomberg field            |
|----------|----------------------------------------------------------------------|----------------------------|
| `date`   | exchange day, `YYYY-MM-DD`                                            |                            |
| `aru`    | closing price of Arafura Rare Earths, A$ (blank on trading-halt days) | `ARU AU Equity`, `PX_LAST` |
| `bid`    | closing bid, A$                                                       | `PX_BID`                   |
| `ask`    | closing ask, A$                                                       | `PX_ASK`                   |
| `volume` | shares traded (blank on trading-halt days)                            | `PX_VOLUME`                |
| `xao`    | closing level of the All Ordinaries index                             | `AS30 Index`, `PX_LAST`    |
| `rf_pct` | daily risk-free rate in per cent (Kenneth French data library)        |                            |

Prices are unadjusted and days without a trade are left blank (no fill).
