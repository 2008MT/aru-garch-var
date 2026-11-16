"""Build the landing page of the project website (index.html) from results/*.json.

Every number on the page is read from the result files, so the site stays in step with the paper.
Figures are the paper's own (figures/*.pdf), converted to PNG with pdftoppm (poppler)."""
import json, os, subprocess
from common import ROOT

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
REPO = "https://github.com/2008MT/aru-garch-var"
PAPER = "TOK_GARCH_RISK_JUNIOR_MINING.pdf"
# Add the links here once the paper is posted; a link that is None is left off the page.
SSRN = None
ARXIV = None

CSS = """
:root{--bg:#fcfcfb;--card:#ffffff;--ink:#0b0b0b;--ink2:#52514e;--line:#e6e5e0;--link:#1f4e8c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif}
main{max-width:1000px;margin:0 auto;padding:48px 20px 64px}
h1{font-size:2rem;line-height:1.2;margin:0 0 6px;letter-spacing:-0.01em}
.sub{font-size:1.15rem;color:var(--ink2);margin:0 0 14px;max-width:none}
h2{font-size:1.3rem;margin:56px 0 8px}
p{margin:0 0 14px;max-width:72ch}
a{color:var(--link)}
.meta{color:var(--ink2);margin:0 0 20px}
.links a{display:inline-block;margin:0 14px 6px 0;font-weight:600}
.lead{font-size:1.1rem}
.tiles{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:28px 0 8px}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:14px 16px;min-width:0}
.tile b{display:block;font-size:1.45rem;line-height:1.25;white-space:nowrap}
.tile span{color:var(--ink2);font-size:0.86rem;line-height:1.35;display:block;margin-top:4px}
figure{margin:16px 0 22px}
figure a{display:block;background:#fff;border:1px solid var(--line);border-radius:8px;padding:14px}
figure img{display:block;width:100%;height:auto}
figure.narrow{max-width:760px}
figcaption{color:var(--ink2);font-size:0.9rem;margin-top:8px;max-width:80ch}
.scroll{overflow-x:auto}
table{border-collapse:collapse;margin:8px 0 4px;font-size:0.93rem;font-variant-numeric:tabular-nums}
th,td{padding:5px 14px 5px 0;text-align:right;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}
th{color:var(--ink2);font-weight:600}
td.best{font-weight:600}
pre{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;font-size:0.85rem;white-space:pre-wrap;overflow-wrap:anywhere;max-width:80ch}
footer{margin-top:64px;padding-top:20px;border-top:1px solid var(--line);color:var(--ink2);font-size:0.9rem}
@media (max-width:760px){.tiles{grid-template-columns:repeat(2,1fr)}h1{font-size:1.6rem}main{padding:32px 16px 48px}figure a{padding:8px}}
@media (max-width:520px){.tiles{grid-template-columns:1fr}}
"""

MINUS = "−"


def n(x, d=2, sign=False):
    """Number with a real minus sign; optional explicit plus."""
    s = f"{x:,.{d}f}"
    if float(s.replace(",", "")) == 0: s = s.lstrip("-")
    if s.startswith("-"): return MINUS + s[1:]
    return ("+" + s) if sign else s


def png(name):
    """Convert figures/<name>.pdf to PNG for the web (skipped if poppler is missing and the PNG exists)."""
    src = os.path.join(FIG, name + ".pdf"); dst = os.path.join(FIG, name)
    try:
        subprocess.run(["pdftoppm", "-r", "200", "-png", "-singlefile", src, dst], check=True)
    except (OSError, subprocess.CalledProcessError):
        if not os.path.exists(dst + ".png"): raise
    return f"figures/{name}.png"


def figure(name, alt, caption, narrow=False):
    src = png(name)
    cls = ' class="narrow"' if narrow else ""
    return (f'<figure{cls}><a href="{src}"><img src="{src}" alt="{alt}" loading="lazy"></a>'
            f"<figcaption>{caption}</figcaption></figure>")


def main():
    P1 = json.load(open(os.path.join(RES, "part1.json")))
    P2 = json.load(open(os.path.join(RES, "part2.json")))
    P3 = json.load(open(os.path.join(RES, "part3.json")))
    D = P1["desc"]["aru"]; Q = P1["quotes"]; K = P1["contrarian"]; S = P1["serial"]["aru"]
    SC, SM = Q["serial_close"], Q["serial_mid"]
    M = P2["full"]; O = dict(P2["oos"]); O["GARCH-J"] = P3["oos"]
    at_quote = 100 * (Q["at_bid"] + Q["at_ask"])
    exp1 = O["HS"]["left_0.01"]["expected"]
    nus = [M[m]["params"].get("nu", M[m]["params"].get("eta")) for m in M if "nu" in M[m]["params"] or "eta" in M[m]["params"]]
    gauss_short = sorted(O[m]["right_0.01"]["viol"] for m in ("Normal", "GARCH-N", "EGARCH-N"))
    es = O["EGARCH-st"]

    links = [f'<a href="{PAPER}">Read the paper (PDF)</a>']
    if SSRN: links.append(f'<a href="{SSRN}">SSRN</a>')
    if ARXIV: links.append(f'<a href="{ARXIV}">arXiv</a>')
    links.append(f'<a href="{REPO}">Code on GitHub</a>')

    tiles = [
        (f"{at_quote:.0f}%", "of days the closing price is exactly the bid or the ask"),
        (f"{n(S['rho1']['rho'], 3)} &rarr; {n(SM['rho1']['rho'], 3)}",
         "first-day autocorrelation of returns at closing prices, and at quote mid-points"),
        (f"{n(K['close'], 2, True)}% &rarr; {n(K['net'], 2)}%",
         "daily return of a contrarian rule on paper, and after paying the quoted spread"),
        (f"{O['Normal']['right_0.01']['viol']} &rarr; {es['right_0.01']['viol']}",
         f"breaches of the 1% short-side risk limit, Gaussian benchmark against the skewed-<i>t</i> model ({exp1:.1f} expected)"),
    ]

    # --- tables
    def row(cells, best=None):
        return "<tr>" + "".join(f'<td class="best">{c}</td>' if best == i else f"<td>{c}</td>" for i, c in enumerate(cells)) + "</tr>"

    t_quotes = ('<div class="scroll"><table><tr><th>Daily returns measured at</th><th>Autocorrelation, lag 1</th><th>Robust <i>t</i></th>'
                "<th>Variance ratio, 2 days</th><th>Standard deviation</th></tr>"
                + row(["Closing prices", n(SC["rho1"]["rho"], 3), n(SC["rho1"]["t_rob"], 2), n(SC["vr"]["2"]["vr"], 2), f"{Q['sd_close']:.2f}%"])
                + row(["Quote mid-points", n(SM["rho1"]["rho"], 3), n(SM["rho1"]["t_rob"], 2), n(SM["vr"]["2"]["vr"], 2), f"{Q['sd_mid']:.2f}%"])
                + "</table></div>")
    t_rule = ('<div class="scroll"><table><tr><th>Contrarian rule valued</th><th>Return a day</th><th><i>t</i>-statistic</th></tr>'
              + row(["At closing prices", f"{n(K['close'], 2, True)}%", n(K["close_t"], 1)])
              + row(["At quote mid-points", f"{n(K['mid'], 2, True)}%", n(K["mid_t"], 1)])
              + row(["Buying at the ask, selling at the bid", f"{n(K['net'], 2, True)}%", n(K["net_t"], 1)])
              + "</table></div>")
    lab = {"GARCH-N": "GARCH, Normal", "EGARCH-N": "EGARCH, Normal", "GARCH-t": "GARCH, Student-<i>t</i>", "EGARCH-t": "EGARCH, Student-<i>t</i>",
           "GARCH-st": "GARCH, skewed <i>t</i>", "EGARCH-st": "EGARCH, skewed <i>t</i>"}
    aic = {m: M[m]["aic"] * M[m]["n"] for m in M}; best = min(aic, key=aic.get)
    rows = ""
    for m in M:
        par = M[m]["params"]; nu = par.get("nu", par.get("eta")); lam = par.get("lambda")
        cells = [lab[m], n(M[m]["ll"], 1), n(aic[m], 1), f"{nu:.2f}" if nu else "", n(lam, 3) if lam is not None else ""]
        rows += "<tr>" + "".join(f'<td class="best">{c}</td>' if (m == best and i == 2) else f"<td>{c}</td>" for i, c in enumerate(cells)) + "</tr>"
    rows += row(["GARCH, Poisson jumps", n(P3["ll"], 1), n(P3["aic"], 1), "", ""])
    t_models = ('<div class="scroll"><table><tr><th>Model</th><th>Log-likelihood</th><th>AIC</th><th>Degrees of freedom</th><th>Skewness &lambda;</th></tr>'
                + rows + "</table></div>")

    def bt(m, key):
        v = O[m][key]; return f"{v['viol']}*" if v["p_uc"] < 0.05 else f"{v['viol']}"
    names = [("Normal", "Static Normal"), ("HS", "Historical simulation"), ("GARCH-N", "GARCH, Normal"), ("EGARCH-N", "EGARCH, Normal"),
             ("GARCH-t", "GARCH, Student-<i>t</i>"), ("EGARCH-t", "EGARCH, Student-<i>t</i>"), ("GARCH-st", "GARCH, skewed <i>t</i>"),
             ("EGARCH-st", "EGARCH, skewed <i>t</i>"), ("GARCH-J", "GARCH, Poisson jumps")]
    exp5 = O["HS"]["left_0.05"]["expected"]
    t_bt = ('<div class="scroll"><table><tr><th>Model</th><th>Long, 1%</th><th>Short, 1%</th><th>Long, 5%</th><th>Short, 5%</th></tr>'
            + "".join(row([l, bt(m, "left_0.01"), bt(m, "right_0.01"), bt(m, "left_0.05"), bt(m, "right_0.05")]) for m, l in names)
            + row(["<i>Expected</i>", f"<i>{exp1:.1f}</i>", f"<i>{exp1:.1f}</i>", f"<i>{exp5:.1f}</i>", f"<i>{exp5:.1f}</i>"])
            + "</table></div>")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Junior mining stock risk: efficiency tests and Value-at-Risk with GARCH models | Mael Tok</title>
<meta name="description" content="Weak-form efficiency tests and out-of-sample Value-at-Risk for Arafura Rare Earths, 2018 to 2025: bid-ask bounce, skewed fat tails and GARCH models.">
<style>{CSS}</style>
</head>
<body>
<main>
<h1>An Empirical Study in Junior Mining Stock Risk Management</h1>
<p class="sub">Weak-form efficiency tests and Value-at-Risk estimation with GARCH models</p>
<p class="meta">Mael Tok · McGill University · first version December 2025, this version November 2026</p>
<p class="links">{''.join(links)}</p>

<p class="lead">Arafura Rare Earths is an Australian company developing a rare-earth mine that is not yet financed or built. Its shares are about
{D['sd'] / P1['desc']['xao']['sd']:.0f} times as volatile as the market. This paper asks two questions about that risk: can the daily return be predicted from past prices,
and how well do standard models forecast the loss a position can suffer in a day?</p>
<p>The return looks predictable, but the pattern is an artefact of how closing prices are recorded and cannot be traded.
The risk is predictable, and what matters most is its shape: the largest moves are upward, so a short position carries more tail risk than a long one,
and the usual models get one side or the other wrong.</p>

<div class="tiles">
{''.join(f'<div class="tile"><b>{b}</b><span>{s}</span></div>' for b, s in tiles)}
</div>

<h2>1. The stock</h2>
<p>The sample is {P1['n_days']:,} exchange days from August 2018 to June 2025. Annualised volatility is {D['ann_vol']:.0f}%, and the distribution of returns is
skewed to the right: the largest move is a rise of {P1['jump']['simple']:.0f}% on 14 March 2024, when the Australian Government announced
A$840 million of funding for the project. Two of the three largest falls follow discounted share placements.</p>
{figure("fig1_price_returns", "Closing price and daily log-returns of Arafura Rare Earths, 2018 to 2025",
        "Closing price (log scale) and daily log-returns.")}

<h2>2. A coarse price grid</h2>
<p>Shares below A$0.10 trade in steps of A$0.001 and shares above it in steps of A$0.005. For this stock one step, or tick, is on average
{P1['price']['rel_tick_mean']:.1f}% of the price, the same order of magnitude as a typical daily move. The bid&ndash;ask spread is one tick on {100 * Q['one_tick']:.0f}% of days,
and the closing price does not change at all on {100 * D['zero']:.0f}% of days.</p>
{figure("fig_ticks", "One tick and the quoted spread as a percentage of the price, and the distribution of daily price changes in ticks",
        "Left: one tick and the quoted bid&ndash;ask spread as a percentage of the price. Right: daily change in the closing price, counted in ticks.")}

<h2>3. Is the return predictable?</h2>
<p>Daily returns are negatively autocorrelated: a rise tends to be followed by a small fall. The effect survives tests that allow for changing volatility.
It is not mispricing. The closing price is the price of the last trade, which is at the bid after a sale and at the ask after a purchase, and it sits at one of
the two on {at_quote:.0f}% of days. Prices that bounce between bid and ask produce this autocorrelation even when the value of the stock follows a random walk.</p>
<p>Measuring returns at the mid-point of the closing bid and ask removes the bounce, and most of the autocorrelation goes with it.</p>
{t_quotes}
{figure("fig2_acf", "Autocorrelations of returns at closing prices, at quote mid-points, and of squared returns",
        "Autocorrelations at lags of 1 to 20 days. Bars outside the orange band are significant at the 5% level. Squared returns (right) show the volatility clustering that the risk models use.")}
<p>A rule that bets on the reversal, buying after a fall and selling short after a rise, looks profitable at closing prices.
Valued at mid-quotes it earns nothing, and a trader who buys at the ask and sells at the bid loses money every day.</p>
{t_rule}

<h2>4. Modelling the risk</h2>
<p>Six models combine two volatility equations, GARCH and EGARCH, with three distributions for the daily shock: Normal, Student-<i>t</i>, and a skewed
Student-<i>t</i> that lets one tail be heavier than the other. A seventh adds Poisson jumps. The choice of distribution matters far more than the choice of
volatility equation. Tails are heavy, with {min(nus):.1f} to {max(nus):.1f} degrees of freedom, and the right tail is heavier than the left. The jump model agrees:
{P3['post_up']} of the {P3['n_post50']} days it identifies as jumps are rises.</p>
{t_models}
<p style="color:var(--ink2);font-size:0.9rem">A lower AIC is better. The preferred model is in bold.</p>
{figure("fig3_qq", "Quantile-quantile plots of standardised residuals under Normal, Student-t and skewed-t distributions",
        "Standardised residuals against what each distribution predicts. Points on the line mean the model fits. The outlier in every panel is 14 March 2024.", narrow=True)}

<h2>5. Value-at-Risk out of sample</h2>
<p>Each model is estimated on data up to one day and used to forecast the next day's 1% and 5% loss limits, for a long and for a short position, on
{P2['oos_n']} days from August 2021. A correct model is breached on 1% or 5% of days. Gaussian models are breached about twice too often on the short side
({gauss_short[0]} to {gauss_short[-1]} times against {exp1:.1f} expected). Symmetric fat tails fix that but become too cautious on the long side.
Models with an asymmetric distribution pass every test, and so does a plain historical quantile: for this stock the shape of the distribution carries
most of the information.</p>
{t_bt}
<p style="color:var(--ink2);font-size:0.9rem">Number of days on which the loss limit was breached. An asterisk marks a count that differs significantly from the expected number (Kupiec test, 5% level).</p>
{figure("fig_violations", "Number of Value-at-Risk violations by model, tail and level",
        "The same counts as a chart. Hatched bars are rejected by the coverage test.")}
{figure("fig4_oos_var", "Out-of-sample 1% and 99% quantile forecasts against realised returns",
        "Forecast 1% and 99% limits of the preferred model (blue) and of a static Normal benchmark (orange), with realised returns in grey. Circles are breaches of the blue band.")}

<h2>6. Why the fat tail is on the upside</h2>
<p>The equity of a company whose only asset is an unbuilt, unfinanced project behaves like an option on that project. Option values are convex,
so news that the project will go ahead raises the share price by more than equally sized bad news lowers it, and financing news arrives in single
announcements. Stocks with this lottery-like profile are known to attract investors who pay for the chance of a jump. The party who bears the jump risk
is the short seller, which is the risk a Gaussian model understates.</p>

<h2>Cite</h2>
<pre>Tok, M. (2026). An Empirical Study in Junior Mining Stock Risk Management: Weak-Form Efficiency Tests and Value-at-Risk Estimation with GARCH Models. Working paper, November 2026. https://2008mt.github.io/aru-garch-var/</pre>

<footer>
<p>Data: Bloomberg Terminal (daily closes, closing quotes and volume), not redistributed. Method, limitations and references are in the
<a href="{PAPER}">paper</a>; every table, figure and number is reproducible from the <a href="{REPO}">code</a>. This is a study of method, not investment advice.</p>
<p>The empirical work on which the paper is based was carried out jointly with Dr. Roman Mestre during a research internship at the Faculty of Economics,
University of Montpellier. A statement of contributions and a declaration of generative-AI use are at the end of the paper.</p>
</footer>
</main>
</body>
</html>
"""
    open(os.path.join(ROOT, "index.html"), "w", encoding="utf-8").write(html)
    print("index.html written,", len(html), "bytes")


if __name__ == "__main__":
    main()
