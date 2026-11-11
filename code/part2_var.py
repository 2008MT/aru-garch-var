"""Part II: AR(1)-GARCH / EGARCH models under Normal, Student-t and skewed-t innovations,
in-sample diagnostics, and out-of-sample Value-at-Risk backtests.
Writes results/part2.json and results/oos_var.csv."""
import json, os, sys, time, warnings, numpy as np, pandas as pd
from scipy import stats
from arch import arch_model
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from common import load, ROOT

warnings.filterwarnings("ignore")
SPECS = {
    "GARCH-N":    dict(vol="GARCH", p=1, q=1, dist="normal"),
    "EGARCH-N":   dict(vol="EGARCH", p=1, o=1, q=1, dist="normal"),
    "GARCH-t":    dict(vol="GARCH", p=1, q=1, dist="t"),
    "EGARCH-t":   dict(vol="EGARCH", p=1, o=1, q=1, dist="t"),
    "GARCH-st":   dict(vol="GARCH", p=1, q=1, dist="skewt"),
    "EGARCH-st":  dict(vol="EGARCH", p=1, o=1, q=1, dist="skewt"),
}
LEVELS = (0.01, 0.05)
T0 = 750          # first estimation window (about three years)
REFIT = int(os.environ.get("REFIT", 1))   # re-estimate every REFIT days


def build(r, spec):
    return arch_model(r, mean="AR", lags=1, rescale=False, **spec)


def dist_params(res):
    names = [n for n in res.params.index if n in ("nu", "eta", "lambda")]
    return res.params[names].values


def quantiles(res, probs):
    return res.model.distribution.ppf(np.asarray(probs), dist_params(res))


# ---------- backtests ----------
def kupiec(hits, p):
    n = len(hits); x = int(hits.sum()); pi = x / n
    ll0 = (n - x) * np.log(1 - p) + x * np.log(p)
    ll1 = (n - x) * np.log(1 - pi) + (x * np.log(pi) if x > 0 else 0.0)
    lr = -2 * (ll0 - ll1)
    return x, pi, lr, stats.chi2.sf(lr, 1)


def christoffersen(hits):
    h = np.asarray(hits).astype(int)
    n00 = np.sum((h[:-1] == 0) & (h[1:] == 0)); n01 = np.sum((h[:-1] == 0) & (h[1:] == 1))
    n10 = np.sum((h[:-1] == 1) & (h[1:] == 0)); n11 = np.sum((h[:-1] == 1) & (h[1:] == 1))
    pi01 = n01 / max(n00 + n01, 1); pi11 = n11 / max(n10 + n11, 1); pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    def l(a, b, q):   # a successes-with-prob-(1-q), b with prob q
        return (a * np.log(1 - q) if a > 0 else 0.0) + (b * np.log(q) if b > 0 else 0.0)
    ll0 = l(n00 + n10, n01 + n11, pi)
    ll1 = l(n00, n01, pi01) + l(n10, n11, pi11)
    lr = -2 * (ll0 - ll1)
    return lr, stats.chi2.sf(lr, 1), int(n11)


def backtest(r, q, p, tail):
    """r: realised returns; q: VaR quantile forecasts (left tail: r<q is a violation; right: r>q)."""
    hits = (r < q) if tail == "left" else (r > q)
    x, pi, lr_uc, p_uc = kupiec(hits.values, p)
    lr_ind, p_ind, n11 = christoffersen(hits.values)
    lr_cc = lr_uc + lr_ind
    u = (r - q) if tail == "left" else (q - r)           # negative when violated
    ql = np.mean((p - (u < 0)) * u)                       # quantile ('tick') loss
    return dict(n=int(len(r)), viol=x, rate=pi, expected=p * len(r), lr_uc=lr_uc, p_uc=p_uc, lr_ind=lr_ind, p_ind=p_ind,
                n11=n11, lr_cc=lr_cc, p_cc=stats.chi2.sf(lr_cc, 2), qloss=float(ql), avg_var=float(np.mean(np.abs(q))))


# ---------- full-sample estimation ----------
def full_sample(r):
    out = {}; series = {}
    for name, spec in SPECS.items():
        res = build(r, spec).fit(disp="off", options={"maxiter": 2000})
        par = res.params; se = res.std_err
        z = (res.resid / res.conditional_volatility).dropna()
        n = int(res.nobs); k = len(par)
        o = dict(conv=int(res.convergence_flag), ll=res.loglikelihood, aic=res.aic / n, bic=res.bic / n, k=k, n=n,
                 params={a: float(b) for a, b in par.items()}, se={a: float(b) for a, b in se.items()},
                 pval={a: float(b) for a, b in res.pvalues.items()})
        if spec["vol"] == "GARCH":
            pers = par["alpha[1]"] + par["beta[1]"]; o["uncond_vol"] = float(np.sqrt(par["omega"] / (1 - pers)))
        else:
            pers = par["beta[1]"]; o["uncond_vol"] = float(np.exp(0.5 * par["omega"] / (1 - pers)))
        o["persistence"] = float(pers); o["half_life"] = float(np.log(0.5) / np.log(pers))
        lb = acorr_ljungbox(z, lags=[10], model_df=1); lb2 = acorr_ljungbox(z ** 2, lags=[10])
        lm = het_arch(z, nlags=5)
        o.update(q10=float(lb.lb_stat.iloc[0]), q10_p=float(lb.lb_pvalue.iloc[0]), q2_10=float(lb2.lb_stat.iloc[0]),
                 q2_10_p=float(lb2.lb_pvalue.iloc[0]), arch5=float(lm[0]), arch5_p=float(lm[1]),
                 z_skew=float(stats.skew(z)), z_kurt=float(stats.kurtosis(z, fisher=False)),
                 z_min=float(z.min()), z_max=float(z.max()))
        qs = quantiles(res, [0.01, 0.05, 0.95, 0.99]); o["quant"] = [float(x) for x in qs]
        mu = (r - res.resid).dropna(); sig = res.conditional_volatility.dropna(); rr = r.loc[sig.index]
        o["insample"] = {}
        for p, ql, qr in ((0.01, qs[0], qs[3]), (0.05, qs[1], qs[2])):
            o["insample"][f"left_{p}"] = backtest(rr, mu + sig * ql, p, "left")
            o["insample"][f"right_{p}"] = backtest(rr, mu + sig * qr, p, "right")
        out[name] = o
        series[name + "_sig"] = sig; series[name + "_mu"] = mu; series[name + "_z"] = z
    return out, pd.DataFrame(series)


# ---------- out-of-sample ----------
def oos(r):
    n = len(r); cols = {}
    for name, spec in SPECS.items():
        t_start = time.time()
        mu = np.full(n, np.nan); sig = np.full(n, np.nan); Q = np.full((n, 4), np.nan); start = None
        for t0 in range(T0, n, REFIT):
            res = build(r.iloc[:t0], spec).fit(disp="off", starting_values=start, options={"maxiter": 2000})
            start = res.params.values
            t1 = min(t0 + REFIT, n)
            fx = build(r.iloc[:t1], spec).fix(res.params)        # same parameters, filter run forward
            s = fx.conditional_volatility.iloc[t0:t1].values
            m = (r.iloc[:t1] - fx.resid).iloc[t0:t1].values
            mu[t0:t1] = m; sig[t0:t1] = s; Q[t0:t1] = quantiles(res, [0.01, 0.05, 0.95, 0.99])
        for j, lab in enumerate(("L01", "L05", "R05", "R01")):
            cols[f"{name}_{lab}"] = mu + sig * Q[:, j]
        cols[f"{name}_sig"] = sig
        print(name, "done in %.0fs" % (time.time() - t_start), flush=True)
    # benchmarks: historical simulation (rolling 500 days) and static Normal (expanding window)
    hs = {lab: np.full(n, np.nan) for lab in ("L01", "L05", "R05", "R01")}; sn = {k: v.copy() for k, v in hs.items()}
    zq = stats.norm.ppf([0.01, 0.05, 0.95, 0.99]); rv = r.values
    for t in range(T0, n):
        w = rv[t - 500:t]; e = rv[:t]
        for j, lab in enumerate(("L01", "L05", "R05", "R01")):
            hs[lab][t] = np.quantile(w, [0.01, 0.05, 0.95, 0.99][j])
            sn[lab][t] = e.mean() + e.std(ddof=1) * zq[j]
    for lab in hs:
        cols[f"HS_{lab}"] = hs[lab]; cols[f"Normal_{lab}"] = sn[lab]
    F = pd.DataFrame(cols, index=r.index); F["r"] = r
    F = F.iloc[T0:]
    out = {}
    for name in list(SPECS) + ["HS", "Normal"]:
        out[name] = {}
        for p, L, Rr in ((0.01, "L01", "R01"), (0.05, "L05", "R05")):
            out[name][f"left_{p}"] = backtest(F.r, F[f"{name}_{L}"], p, "left")
            out[name][f"right_{p}"] = backtest(F.r, F[f"{name}_{Rr}"], p, "right")
    return out, F


if __name__ == "__main__":
    clean = "--raw" not in sys.argv
    tag = "" if clean else "_raw"
    r = load(clean).r.dropna()
    full, ser = full_sample(r)
    R = {"n": int(len(r)), "T0": T0, "refit": REFIT, "full": full}
    ser.to_csv(os.path.join(ROOT, "results", f"insample_series{tag}.csv"))
    if "--no-oos" not in sys.argv:
        o, F = oos(r); R["oos"] = o; R["oos_start"] = str(F.index[0].date()); R["oos_n"] = int(len(F))
        F.to_csv(os.path.join(ROOT, "results", f"oos_var{tag}.csv"))
    json.dump(R, open(os.path.join(ROOT, "results", f"part2{tag}.json"), "w"), indent=1, default=float)
    for k, v in full.items():
        print(f"{k:10s} conv={v['conv']} LL={v['ll']:.2f} AIC={v['aic']:.4f} BIC={v['bic']:.4f} pers={v['persistence']:.3f} "
              f"Q10 p={v['q10_p']:.3f} Q2 p={v['q2_10_p']:.3f} ARCH p={v['arch5_p']:.3f} zskew={v['z_skew']:.2f} zkurt={v['z_kurt']:.1f}")
        print("    ", {a: round(b, 4) for a, b in v["params"].items()})
        print("     IS:", {a: (b["viol"], round(b["p_uc"], 3)) for a, b in v["insample"].items()})
    if "oos" in R:
        print("OOS from", R["oos_start"], "n", R["oos_n"])
        for k, v in R["oos"].items():
            print(f"{k:10s}", {a: (b["viol"], round(b["p_uc"], 3), round(b["p_ind"], 3), round(b["p_cc"], 3), round(b["qloss"], 4)) for a, b in v.items()})
