"""Part I: descriptive statistics, market model, serial-dependence and variance-ratio tests,
bid-ask bounce diagnostics. Writes results/part1.json."""
import json, os, numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
import warnings; warnings.filterwarnings('ignore')
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from common import load, ROOT


def describe(x):
    x = np.asarray(x); n = len(x)
    jb = stats.jarque_bera(x)
    return dict(n=n, mean=x.mean(), median=np.median(x), sd=x.std(ddof=1), min=x.min(), max=x.max(),
                skew=stats.skew(x), kurt=stats.kurtosis(x, fisher=False), jb=jb.statistic, jb_p=jb.pvalue,
                zero=float((x == 0).mean()), ann_vol=x.std(ddof=1) * np.sqrt(252))


def acf_robust(x, k):
    """Sample autocorrelation at lag k with the usual 1/sqrt(T) s.e. and the
    heteroskedasticity-robust s.e. of Lo and MacKinlay (1988)."""
    x = np.asarray(x) - np.mean(x); T = len(x)
    g0 = np.sum(x ** 2)
    rho = np.sum(x[k:] * x[:-k]) / g0
    se_iid = 1 / np.sqrt(T)
    se_rob = np.sqrt(np.sum(x[k:] ** 2 * x[:-k] ** 2)) / g0
    return rho, se_iid, se_rob


def robust_Q(x, m):
    """Box-Pierce statistic with heteroskedasticity-robust autocorrelation variances
    (Lo-MacKinlay 1988; Lobato-Nankervis-Savin 2001). Asymptotically chi2(m) under an m.d.s. null."""
    q = 0.0
    for k in range(1, m + 1):
        rho, _, se = acf_robust(x, k)
        q += (rho / se) ** 2
    return q, stats.chi2.sf(q, m)


def variance_ratio(x, q):
    """Lo-MacKinlay (1988) variance ratio with overlapping q-period returns.
    Returns VR(q), z (iid) and z* (heteroskedasticity-robust)."""
    x = np.asarray(x); T = len(x); mu = x.mean()
    s1 = np.sum((x - mu) ** 2) / (T - 1)
    xq = np.convolve(x, np.ones(q), mode="valid")           # overlapping q-day sums
    m = q * (T - q + 1) * (1 - q / T)
    sq = np.sum((xq - q * mu) ** 2) / m
    vr = sq / s1
    z = (vr - 1) / np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * T))
    d = x - mu; den = np.sum(d ** 2) ** 2
    theta = 0.0
    for j in range(1, q):
        delta = T * np.sum(d[j:] ** 2 * d[:-j] ** 2) / den
        theta += (2 * (q - j) / q) ** 2 * delta
    zs = (vr - 1) / np.sqrt(theta / T)
    return vr, z, zs


def serial_block(x):
    out = {}
    for k in (1, 2, 3, 4, 5):
        rho, se, ser = acf_robust(x, k)
        out[f"rho{k}"] = dict(rho=rho, t_iid=rho / se, t_rob=rho / ser)
    for m in (5, 10, 20):
        lb = acorr_ljungbox(x, lags=[m])
        q, p = robust_Q(x, m)
        out[f"Q{m}"] = dict(lb=float(lb.lb_stat.iloc[0]), lb_p=float(lb.lb_pvalue.iloc[0]), rob=q, rob_p=p)
        lb2 = acorr_ljungbox((x - np.mean(x)) ** 2, lags=[m])
        out[f"Q2_{m}"] = dict(lb=float(lb2.lb_stat.iloc[0]), lb_p=float(lb2.lb_pvalue.iloc[0]))
    lm = het_arch(x - np.mean(x), nlags=5)
    out["arch5"] = dict(lm=lm[0], p=lm[1])
    out["vr"] = {q: dict(zip(("vr", "z", "zs"), variance_ratio(x, q))) for q in (2, 5, 10, 20)}
    return out


def run(clean=True):
    df = load(clean)
    d = df.dropna(subset=["r", "m"]).copy()
    R = {"n_total": int(len(df)), "n_used": int(len(d)), "start": str(d.index[0].date()), "end": str(d.index[-1].date()),
         "n_stale": int(df.stale.sum())}
    R["desc"] = {"aru": describe(d.r), "xao": describe(d.m)}
    R["price"] = dict(min=float(df.aru.min()), max=float(df.aru.max()), first=float(df.aru.iloc[0]), last=float(df.aru.iloc[-1]),
                      share_below_10c=float((df.aru < 0.1 - 1e-12).mean()),
                      rel_tick_mean=float(df.rel_tick.mean()), rel_tick_median=float(df.rel_tick.median()),
                      rel_tick_min=float(df.rel_tick.min()), rel_tick_max=float(df.rel_tick.max()))
    # --- market model, excess returns, Newey-West (HAC) standard errors
    y = d.r - d.rf; X = sm.add_constant(d.m - d.rf)
    ols = sm.OLS(y, X).fit()
    hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    R["capm"] = dict(alpha=ols.params.iloc[0], beta=ols.params.iloc[1], r2=ols.rsquared,
                     alpha_se=hac.bse.iloc[0], beta_se=hac.bse.iloc[1], alpha_t=hac.tvalues.iloc[0], beta_t=hac.tvalues.iloc[1],
                     alpha_p=hac.pvalues.iloc[0], beta_p=hac.pvalues.iloc[1], beta1_t=(ols.params.iloc[1] - 1) / hac.bse.iloc[1],
                     alpha_se_ols=ols.bse.iloc[0], beta_se_ols=ols.bse.iloc[1], resid_sd=float(np.sqrt(ols.scale)),
                     corr=float(np.corrcoef(d.r, d.m)[0, 1]))
    e = ols.resid.values
    R["desc"]["resid"] = describe(e)
    R["serial"] = {"aru": serial_block(d.r.values), "resid": serial_block(e), "xao": serial_block(d.m.values)}
    # --- bid-ask bounce: Roll (1984) implied spread and tick size
    x = d.r.values - d.r.mean(); cov1 = np.mean(x[1:] * x[:-1])
    R["roll"] = dict(cov1=cov1, spread=2 * np.sqrt(-cov1) if cov1 < 0 else float("nan"))
    # sub-samples by tick regime (price < 10c: tick 0.1c ; price >= 10c: tick 0.5c)
    for name, mask in (("low", d.tick.shift(1) == 0.001), ("high", d.tick.shift(1) == 0.005)):
        pair = pd.DataFrame({"a": d.r, "b": d.r.shift(1)})[mask].dropna()
        a = pair.a - d.r.mean(); b = pair.b - d.r.mean()
        c = float(np.mean(a * b)); rho = c / d.r.var()
        R["roll"][name] = dict(n=int(len(pair)), cov1=c, rho1=float(np.corrcoef(pair.a, pair.b)[0, 1]),
                               spread=2 * np.sqrt(-c) if c < 0 else float("nan"),
                               rel_tick=float(d.rel_tick[mask].mean()))
    # --- how much is the AR(1) worth?  contrarian rule: position = -sign(r_{t-1}), traded at the close
    ar = sm.OLS(d.r.values[1:], sm.add_constant(d.r.values[:-1])).fit(cov_type="HC0")
    R["ar1"] = dict(phi=ar.params[1], t_rob=ar.tvalues[1], r2=ar.rsquared,
                    mean_abs_pred=float(np.mean(np.abs(ar.params[1] * x[:-1]))))
    c = pd.DataFrame({"r": d.r, "pos": -np.sign(d.r.shift(1)), "half": d.rel_tick.shift(1) / 2}).dropna()
    c["gross"] = c.pos * c.r
    c["turn"] = (c.pos - c.pos.shift(1).fillna(0)).abs()          # units traded at the previous close
    c["net"] = c.gross - c.turn * c.half                           # pay half of a one-tick spread per unit
    tstat = lambda v: float(v.mean() / v.std(ddof=1) * np.sqrt(len(v)))
    R["contrarian"] = dict(n=int(len(c)), gross=float(c.gross.mean()), gross_t=tstat(c.gross), net=float(c.net.mean()),
                           net_t=tstat(c.net), turnover=float(c.turn.mean()), cost=float((c.turn * c.half).mean()),
                           half_spread=float(c.half.mean()), mean_abs_r=float(d.r.abs().mean()),
                           breakeven_half=float(c.gross.mean() / c.turn.mean()))
    return R, d, e


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    R, d, e = run(True); R0, _, _ = run(False)
    json.dump({"clean": R, "raw": R0}, open(os.path.join(ROOT, "results", "part1.json"), "w"), indent=1, default=float)
    pd.Series(e, index=d.index, name="resid").to_csv(os.path.join(ROOT, "results", "capm_resid.csv"))
    def show(R):
        print({k: R[k] for k in ("n_total", "n_used", "start", "end", "n_stale")})
        for k, v in R["desc"].items(): print(k, {a: round(b, 4) for a, b in v.items()})
        print("price", {a: round(b, 4) for a, b in R["price"].items()})
        print("capm", {a: round(b, 4) for a, b in R["capm"].items()})
        for s in ("aru", "resid", "xao"):
            B = R["serial"][s]; print("==", s)
            print("  rho", {k: tuple(round(x, 3) for x in B[k].values()) for k in B if k.startswith("rho")})
            print("  Q  ", {k: tuple(round(x, 4) for x in B[k].values()) for k in B if k.startswith("Q")})
            print("  arch5", B["arch5"]); print("  VR", {q: tuple(round(x, 3) for x in v.values()) for q, v in B["vr"].items()})
        print("roll", R["roll"]); print("ar1", R["ar1"]); print("contrarian", R["contrarian"])
    print("######## CLEAN"); show(R); print("######## RAW"); show(R0)
