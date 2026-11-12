# An Empirical Study in Junior Mining Stock Risk Management

**Weak-Form Efficiency Tests and Value-at-Risk Estimation with GARCH Models**
Mael Tok (McGill University). First version December 2025; this version November 2026.

📄 **Paper:** [`TOK_GARCH_RISK_JUNIOR_MINING.pdf`](TOK_GARCH_RISK_JUNIOR_MINING.pdf)

## Summary

Daily risk of Arafura Rare Earths (ASX: ARU), a pre-production rare-earth developer, 2018–2025.

- **Efficiency.** Returns have a significant first-order autocorrelation of −0.110, but it is
  bid–ask bounce on a coarse price grid (one tick ≈ 2.2% of the price). The close sits at the bid
  or the ask on 85% of days, and the autocorrelation is insignificant (−0.040) in returns measured
  at quote mid-points. A contrarian rule earns 0.34%/day at closing prices, nothing at mid-quotes,
  and loses 1.52%/day when it buys at the ask and sells at the bid: no exploitable inefficiency.
- **Volatility models.** AR(1)–GARCH and EGARCH with Normal, Student-t and skewed-t innovations,
  plus a GARCH model with Poisson jumps. Tails are heavy (four to five degrees of freedom) and
  skewed to the upside.
- **Value-at-Risk.** 959 out-of-sample one-day forecasts for long and short positions. Gaussian
  models understate short-side risk, symmetric-t models overstate long-side risk; skewed-t and
  jump models pass every coverage test.
- **Interpretation.** The equity of an unfinanced project behaves like an option on it, which is
  why the fat tail is on the upside and why the short seller bears the risk a Gaussian model misses.

## Repository layout

```
TOK_GARCH_RISK_JUNIOR_MINING.pdf   the paper
main.tex, refs.bib, numbers.tex    LaTeX source (compiles as is from tables/ and figures/)
code/                              Python code that produces every table, figure and number
results/                           summary output of the code (JSON)
data/                              data description (Bloomberg prices are not redistributed)
```

## Reproducing the results

Requires Python 3 (`pip install -r requirements.txt`) and, for the figures, a LaTeX installation.
Add the data file described in `data/README.md`, then run from `code/`:

| Step | Command | Output |
|---|---|---|
| 1 | `python part1_efficiency.py` | descriptive statistics, market model, autocorrelation and variance-ratio tests, Roll spread, mid-quote tests, contrarian rule |
| 2 | `python part2_var.py` | six AR(1)–(E)GARCH models, diagnostics, out-of-sample VaR backtests (~2 min) |
| 3 | `python part3_jump.py` | GARCH with Poisson jumps and its backtest (~2 min) |
| 4 | `python figures.py` | all figures |
| 5 | `python tables.py` | LaTeX tables and every number quoted in the text (`numbers.tex`) |

Then compile: `pdflatex main && bibtex main && pdflatex main && pdflatex main`.

## Acknowledgements

The empirical work on which the paper is based was carried out jointly with Dr. Roman Mestre during
a research internship at the Faculty of Economics, University of Montpellier, in June 2025. He taught
me the methods used here. The paper itself is my own; see the statement of contributions at the end of the paper.
