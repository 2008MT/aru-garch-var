"""Figures for the paper (vector PDF, Latin Modern to match the LaTeX text)."""
import json, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt, matplotlib.dates as mdates
from scipy import stats
from arch.univariate import SkewStudent
from common import load, ROOT, STALE
from part1_efficiency import acf_robust

plt.rcParams.update({"pgf.texsystem": "pdflatex", "pgf.rcfonts": False, "text.usetex": True,
                     "pgf.preamble": r"\usepackage[T1]{fontenc}\usepackage{lmodern}\usepackage{amsmath}",
                     "font.family": "serif", "font.size": 9.5, "axes.labelsize": 9.5,
                     "axes.titlesize": 10, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
                     "axes.axisbelow": True, "legend.frameon": False, "axes.unicode_minus": False})
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1a1a19", "#8a8983"
OUT = os.path.join(ROOT, "figures"); W = 6.3
P1 = json.load(open(os.path.join(ROOT, "results", "part1.json")))["clean"]
P2 = json.load(open(os.path.join(ROOT, "results", "part2.json")))


def shade(ax):
    for a, b in STALE:
        ax.axvspan(pd.Timestamp(a), pd.Timestamp(b), color="#d9d8d3", lw=0, zorder=0)


def fig_price_returns():
    raw = load(False); df = load(True)
    fig, (a, b) = plt.subplots(2, 1, figsize=(W, 4.3), sharex=True, gridspec_kw={"height_ratios": [1, 1.15], "hspace": 0.12})
    a.plot(raw.index, raw.aru, color=INK, lw=0.8); a.set_yscale("log"); shade(a)
    a.set_yticks([0.05, 0.1, 0.2, 0.4, 0.6]); a.set_yticklabels(["0.05", "0.10", "0.20", "0.40", "0.60"]); a.minorticks_off()
    a.set_ylabel(r"Close (A\$, log scale)")
    a.text(0.008, 0.93, "(a) Closing price", transform=a.transAxes, va="top", fontsize=9.5)
    b.vlines(df.index, 0, df.r, color=BLUE, lw=0.55); shade(b); b.axhline(0, color=INK, lw=0.5)
    b.set_ylabel(r"Daily log-return (\%)"); b.set_ylim(-32, 64)
    b.text(0.008, 0.95, "(b) Daily log-returns", transform=b.transAxes, va="top", fontsize=9.5)
    def note(date, y, text, tx, ty, ha):
        b.annotate(text, xy=(pd.Timestamp(date), y), xytext=(pd.Timestamp(tx), ty), fontsize=8, color=INK, ha=ha, va="center",
                   arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5, shrinkA=2, shrinkB=2))
    note("2019-06-13", df.r["2019-06-13"], "Rare-earth rally\n(US--China trade tensions)", "2019-10-01", 37, "left")
    t = df.r["2020-03-01":"2020-03-31"].idxmin()
    note(t, df.r[t], "COVID-19 sell-off", "2020-06-20", -27, "left")
    note("2024-03-14", df.r["2024-03-14"], "A\\$840m Commonwealth\nfunding package", "2023-12-20", 50, "right")
    b.text(pd.Timestamp("2022-06-01"), 30, "excluded\n(filled gaps)", fontsize=8, color="#52514e", ha="left", va="center")
    b.xaxis.set_major_locator(mdates.YearLocator()); b.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    b.set_xlim(raw.index[0] - pd.Timedelta(days=20), raw.index[-1] + pd.Timedelta(days=20))
    fig.savefig(os.path.join(OUT, "fig1_price_returns.pdf"), bbox_inches="tight"); plt.close(fig)


def fig_acf():
    r = load(True).r.dropna().values; L = 20
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.5), gridspec_kw={"wspace": 0.28})
    for ax, x, title in ((axes[0], r, "(a) Returns $r_t$"), (axes[1], (r - r.mean()) ** 2, "(b) Squared returns $r_t^2$")):
        est = [acf_robust(x, k) for k in range(1, L + 1)]
        rho = np.array([e[0] for e in est]); rob = 1.96 * np.array([e[2] for e in est]); iid = 1.96 / np.sqrt(len(x))
        lags = np.arange(1, L + 1)
        ax.bar(lags, rho, width=0.55, color=BLUE, zorder=3)
        ax.axhline(0, color=INK, lw=0.5)
        if ax is axes[0]:
            ax.plot(lags, rob, color=ORANGE, lw=1.0, zorder=4, label=r"robust 95\% band"); ax.plot(lags, -rob, color=ORANGE, lw=1.0, zorder=4)
        ax.axhline(iid, color=MUTED, lw=0.8, ls=(0, (4, 2)), label=r"i.i.d.\ 95\% band"); ax.axhline(-iid, color=MUTED, lw=0.8, ls=(0, (4, 2)))
        ax.set_xlabel("Lag (trading days)"); ax.set_xticks([1, 5, 10, 15, 20]); ax.set_title(title, loc="left", pad=4)
        ax.set_ylim(-0.17, 0.17)
    axes[0].set_ylabel("Autocorrelation"); axes[0].legend(loc="upper right", handlelength=1.6)
    fig.savefig(os.path.join(OUT, "fig2_acf.pdf"), bbox_inches="tight"); plt.close(fig)


def fig_qq():
    z_all = pd.read_csv(os.path.join(ROOT, "results", "insample_series.csv"), index_col=0)
    fig, axes = plt.subplots(1, 3, figsize=(W, 2.45), sharey=True, gridspec_kw={"wspace": 0.08})
    specs = [("EGARCH-N", "(a) Normal"), ("EGARCH-t", "(b) Student-$t$"), ("EGARCH-st", "(c) Skewed $t$")]
    for ax, (name, title) in zip(axes, specs):
        z = np.sort(z_all[name + "_z"].dropna().values); n = len(z); pp = (np.arange(1, n + 1) - 0.5) / n
        par = P2["full"][name]["params"]
        if name.endswith("-N"): q = stats.norm.ppf(pp)
        elif name.endswith("-st"): q = SkewStudent().ppf(pp, np.array([par["eta"], par["lambda"]]))
        else: q = stats.t.ppf(pp, par["nu"]) * np.sqrt((par["nu"] - 2) / par["nu"])
        lim = (-9, 13)
        ax.plot(lim, lim, color=INK, lw=0.6, zorder=2)
        ax.scatter(q, z, s=7, color=BLUE, lw=0, zorder=3, clip_on=True)
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal"); ax.set_title(title, loc="left", pad=4)
        ax.set_xlabel("Model quantile"); ax.set_xticks([-8, -4, 0, 4, 8, 12]); ax.set_yticks([-8, -4, 0, 4, 8, 12])
    axes[0].set_ylabel("Standardised residual $\\hat z_t$")
    fig.savefig(os.path.join(OUT, "fig3_qq.pdf"), bbox_inches="tight"); plt.close(fig)


def fig_oos():
    F = pd.read_csv(os.path.join(ROOT, "results", "oos_var.csv"), index_col=0, parse_dates=True)
    x = np.arange(len(F))                                  # trading-day axis (the two excluded blocks are skipped)
    fig, ax = plt.subplots(figsize=(W, 3.3))
    ax.vlines(x, 0, F.r, color="#b9b8b2", lw=0.5, zorder=2)
    m = "EGARCH-st"
    ax.plot(x, F[m + "_L01"], color=BLUE, lw=0.9, zorder=3, label=r"EGARCH skewed-$t$, 1\% and 99\% quantiles")
    ax.plot(x, F[m + "_R01"], color=BLUE, lw=0.9, zorder=3)
    ax.plot(x, F["Normal_L01"], color=ORANGE, lw=1.0, ls=(0, (4, 2)), zorder=3, label=r"Static Normal, 1\% and 99\% quantiles")
    ax.plot(x, F["Normal_R01"], color=ORANGE, lw=1.0, ls=(0, (4, 2)), zorder=3)
    hit = (F.r < F[m + "_L01"]) | (F.r > F[m + "_R01"])
    ax.scatter(x[hit.values], F.r[hit], s=16, facecolor="white", edgecolor=INK, lw=0.8, zorder=5, label="Violation (EGARCH skewed-$t$)")
    ax.axhline(0, color=INK, lw=0.4)
    yrs = [d for d in pd.date_range("2022-01-01", "2025-01-01", freq="YS")]
    ax.set_xticks([int(np.searchsorted(F.index, d)) for d in yrs]); ax.set_xticklabels([d.year for d in yrs])
    ax.set_xlim(-8, len(F) + 8); ax.set_ylim(-52, 64); ax.set_ylabel(r"Daily log-return (\%)")
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, handlelength=2.2, borderaxespad=0.2, columnspacing=1.6)
    fig.savefig(os.path.join(OUT, "fig4_oos_var.pdf"), bbox_inches="tight"); plt.close(fig)



def fig_ticks():
    """(a) one tick as a share of the price; (b) distribution of daily price changes measured in ticks."""
    raw = load(False); df = load(True)
    fig, (a, b) = plt.subplots(1, 2, figsize=(W, 2.45), gridspec_kw={"wspace": 0.3})
    a.plot(raw.index, raw.rel_tick, color=INK, lw=0.7)
    a.set_ylabel(r"One tick as \% of price"); a.set_ylim(0, 5.4); a.set_title("(a) Size of the minimum price step", loc="left", pad=4)
    a.xaxis.set_major_locator(mdates.YearLocator(2)); a.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    d = df.dropna(subset=["r"]); prev = d.aru / np.exp(d.r / 100)            # previous close implied by the return
    tick_prev = np.where(prev < 0.10 - 1e-9, 0.001, 0.005)
    k = np.rint((d.aru - prev) / tick_prev).astype(int).clip(-6, 6)
    cnt = k.value_counts().reindex(range(-6, 7), fill_value=0) / len(k) * 100
    b.bar(cnt.index, cnt.values, width=0.62, color=BLUE, zorder=3)
    for x in (-1, 0, 1): b.text(x, cnt[x] + 0.6, f"{cnt[x]:.0f}\\%", ha="center", va="bottom", fontsize=8, color=INK)
    b.set_xticks(range(-6, 7, 2)); b.set_xticklabels([r"$\leq\!-6$", "$-4$", "$-2$", "$0$", "$2$", "$4$", r"$\geq\!6$"])
    b.set_xlabel("Daily change in closing price (ticks)"); b.set_ylabel(r"Share of days (\%)"); b.set_ylim(0, 30)
    b.set_title("(b) Daily price changes in ticks", loc="left", pad=4); b.grid(axis="x", visible=False)
    fig.savefig(os.path.join(OUT, "fig_ticks.pdf"), bbox_inches="tight"); plt.close(fig)
    return dict(share0=float(cnt[0]), share1=float(cnt[-1] + cnt[1]), share_le2=float(cnt[[-2, -1, 0, 1, 2]].sum()))


def fig_violations():
    """Observed versus expected violations for every model, tail and level (summary of the backtest table)."""
    O = dict(P2["oos"]); P3 = json.load(open(os.path.join(ROOT, "results", "part3.json"))); O["GARCH-J"] = P3["oos"]
    order = ["Normal", "HS", "GARCH-N", "EGARCH-N", "GARCH-t", "EGARCH-t", "GARCH-st", "EGARCH-st", "GARCH-J"]
    lab = {"Normal": "Static Normal", "HS": "Historical simulation", "GARCH-N": "GARCH, Normal", "EGARCH-N": "EGARCH, Normal",
           "GARCH-t": "GARCH, Student-$t$", "EGARCH-t": "EGARCH, Student-$t$", "GARCH-st": "GARCH, skewed $t$",
           "EGARCH-st": "EGARCH, skewed $t$", "GARCH-J": "GARCH, Poisson jumps"}
    fig, axes = plt.subplots(2, 2, figsize=(W, 4.5), sharey=True, gridspec_kw={"wspace": 0.08, "hspace": 0.42})
    panels = [("left_0.01", r"(a) Long position, 1\% VaR"), ("right_0.01", r"(b) Short position, 1\% VaR"),
              ("left_0.05", r"(c) Long position, 5\% VaR"), ("right_0.05", r"(d) Short position, 5\% VaR")]
    y = np.arange(len(order))[::-1]
    for ax, (key, title) in zip(axes.ravel(), panels):
        v = np.array([O[m][key]["viol"] for m in order]); rej = np.array([O[m][key]["p_uc"] < 0.05 for m in order])
        e = O["HS"][key]["expected"]
        ax.barh(y[~rej], v[~rej], height=0.6, color=BLUE, zorder=3, label="Coverage not rejected")
        ax.barh(y[rej], v[rej], height=0.6, facecolor="white", edgecolor=ORANGE, hatch="////", lw=0.9, zorder=3, label=r"Rejected at 5\% (Kupiec test)")
        ax.axvline(e, color=INK, lw=0.9, ls=(0, (4, 2)), zorder=4, label="Expected number")
        for yi, vi in zip(y, v): ax.text(vi + 0.012 * ax.get_xlim()[1] + (0.5 if key.endswith("0.01") else 1.2), yi, str(vi), va="center", ha="left", fontsize=8, color=INK, zorder=6,
                                      bbox=dict(facecolor="white", edgecolor="none", pad=0.6))
        ax.set_xlim(0, 26 if key.endswith("0.01") else 72); ax.set_title(title, loc="left", pad=4); ax.grid(axis="y", visible=False)
        ax.set_ylim(-0.6, len(order) - 0.1); ax.set_yticks(y); ax.set_yticklabels([lab[m] for m in order])
    for ax in axes[1]: ax.set_xlabel("Number of violations in 941 forecasts")
    h, l = axes[0, 1].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.045), handlelength=1.8, columnspacing=2.0)
    fig.savefig(os.path.join(OUT, "fig_violations.pdf"), bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    fig_price_returns(); fig_acf(); fig_qq(); fig_oos(); t = fig_ticks()
    json.dump(t, open(os.path.join(ROOT, "results", "ticks.json"), "w"))
    if os.path.exists(os.path.join(ROOT, "results", "part3.json")) and "oos" in json.load(open(os.path.join(ROOT, "results", "part3.json"))): fig_violations()
    print("figures written", t)
