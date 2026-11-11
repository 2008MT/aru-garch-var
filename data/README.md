# Data

The price data are not included: they were extracted from a licensed market-data terminal
(trading room of the Faculty of Economics, University of Montpellier) and may not be redistributed.

To run the code, place a file `aru_xao_daily.csv` in this folder with four columns:

| column   | content                                                         |
|----------|-----------------------------------------------------------------|
| `date`   | trading day, `YYYY-MM-DD`                                        |
| `aru`    | closing price of Arafura Rare Earths (ASX: ARU), A$              |
| `xao`    | closing level of the All Ordinaries index                        |
| `rf_pct` | daily risk-free rate in per cent (Kenneth French data library)   |

Sample: 15 August 2018 to 16 June 2025 (1,727 trading days).
