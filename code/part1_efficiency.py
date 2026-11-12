"""Part I: descriptive statistics, market model, serial-dependence and variance-ratio tests,
bid-ask bounce diagnostics with closing quotes, contrarian trading rule. Writes results/part1.json."""
import json, os, warnings, numpy as np, pandas as pd
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import acorr_ljungbox, het_arch
from common import load, load_calendar, ROOT

warnings.filterwarnings("ignore")


def describe(x):
    x = np.asarray(x); jb = stats.jarque_bera(x)
    return dict(n=len(x), mean=x.mean(), median=np.median(x), sd=x.std(ddof=1), min=x.min(), max=x.max(),
                skew=stats.skew(x), kurt=stats.kurtosis(x, fisher=False), jb=jb.statistic, jb_p=jb.pvalue,
                zero=float((x == 0).mean()), ann_vol=x.std(ddof=1) * np.sqrt(252))


def acf_robust(x, k):
    """Sample autocorrelation at lag k with the 1/sqrt(T) s.e. and the heteroskedasticity-robust
    s.e. of Lo and MacKinlay (1988)."""
    x = np.asarray(x) - np.mean(x); T = len(x)
    g0 = np.sum(x ** 2)
    rho = np.sum(x[k:] * x[:-k]) / g0
    return rho, 1 / np.sqrt(T), np.sqrt(np.sum(x[k:] ** 2 * x[:-k] ** 2)) / g0


def robust_Q(x, m):
    """Box-Pierce statistic with robust autocorrelation variances (Lobato, Nankervis and Savin 2001)."""
    q = sum((acf_robust(x, k)[0] / acf_robust(x, k)[2]) ** 2 for k in range(1, m + 1))
    return q, stats.chi2.sf(q, m)


def variance_ratio(x, q):
    """Lo-MacKinlay (1988) variance ratio with overlapping q-period returns: VR(q), z (iid), z* (robust)."""
    x = np.asarray(x); T = len(x); mu = x.mean()
    s1 = np.sum((x - mu) ** 2) / (T - 1)
    xq = np.convolve(x, np.ones(q), mode="valid")
    sq = np.sum((xq - q * mu) ** 2) / (q * (T - q + 1) * (1 - q / T))
    vr = sq / s1
    z = (vr - 1) / np.sqrt(2 * (2 * q - 1) * (q - 1) / (3 * q * T))
    d = x - mu; den = np.sum(d ** 2) ** 2
    theta = sum((2 * (q - j) / q) ** 2 * T * np.sum(d[j:] ** 2 * d[:-j] ** 2) / den for j in range(1, q))
    return vr, z, (vr - 1) / np.sqrt(theta / T)


def serial_block(x):
    out = {}
    for k in (1, 2, 3, 4, 5, 7):
        rho, se, ser = acf_robust(x, k)
        out[f"rho{k}"] = dict(rho=rho, t_iid=rho / se, t_rob=rho / ser)
    for m in (5, 10, 20):
        lb = acorr_ljungbox(x, lags=[m]); q, p = robust_Q(x, m)
        out[f"Q{m}"] = dict(lb=float(lb.lb_stat.iloc[0]), lb_p=float(lb.lb_pvalue.iloc[0]), rob=q, rob_p=p)
        lb2 = acorr_ljungbox((x - np.mean(x)) ** 2, lags=[m])
        out[f"Q2_{m}"] = dict(lb=float(lb2.lb_stat.iloc[0]), lb_p=float(lb2.lb_pvalue.iloc[0]))
    lm = het_arch(x - np.mean(x), nlags=5)
    out["arch5"] = dict(lm=lm[0], p=lm[1])
    out["vr"] = {q: dict(zip(("vr", "z", "zs"), variance_ratio(x, q))) for q in (2, 5, 10, 20)}
    return out


def tstat(v):
    return float(v.mean() / v.std(ddof=1) * np.sqrt(len(v)))


def run():
    cal = load_calendar(); df = load(); d = df.dropna(subset=["r", "m"]).copy()
    R = {"n_days": int(len(cal)), "n_halt": int(cal.aru.isna().sum()), "n_ret": int(len(d)),
         "start": str(cal.index[0].date()), "end": str(cal.index[-1].date()), "first_ret": str(d.index[0].date())}
    mfull = 100 * np.log(cal.xao).diff().dropna()            # index returns on every exchange day
    R["desc"] = {"aru": describe(d.r), "xao": describe(mfull)}
    R["n_index"] = int(len(mfull))
    j = d.index.get_loc(pd.Timestamp("2024-03-14")); R["jump"] = dict(simple=float(100 * (d.aru.iloc[j] / d.aru.iloc[j - 1] - 1)), log=float(d.r.iloc[j]))
    val = (df.aru * df.volume)
    R["liq"] = dict(min_volume=float(df.volume.min()), median_value_m=float(val.median() / 1e6), mean_value_m=float(val.mean() / 1e6),
                    p10_value_m=float(val.quantile(0.10) / 1e6))
    R["price"] = dict(min=float(df.aru.min()), max=float(df.aru.max()), share_below_10c=float((df.aru < 0.1 - 1e-9).mean()),
                      rel_tick_mean=float(df.rel_tick.mean()), rel_tick_median=float(df.rel_tick.median()),
                      rel_tick_min=float(df.rel_tick.min()), rel_tick_max=float(df.rel_tick.max()))
    # monthly maximum daily return (lottery-type payoffs)
    mx = d.r.groupby([d.index.year, d.index.month]).max()
    R["lottery"] = dict(n_months=int(len(mx)), mean_max=float(mx.mean()), share_gt10=float((mx > 10).mean()),
                        n_up10=int((d.r > 10).sum()), n_down10=int((d.r < -10).sum()),
                        n_up15=int((d.r > 15).sum()), n_down15=int((d.r < -15).sum()))
    # --- market model in excess returns, Newey-West (HAC) standard errors
    y = d.r - d.rf; X = sm.add_constant(d.m - d.rf)
    ols = sm.OLS(y, X).fit(); hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
    R["capm"] = dict(alpha=ols.params.iloc[0], beta=ols.params.iloc[1], r2=ols.rsquared, alpha_se=hac.bse.iloc[0],
                     beta_se=hac.bse.iloc[1], alpha_t=hac.tvalues.iloc[0], beta_t=hac.tvalues.iloc[1],
                     beta1_t=(ols.params.iloc[1] - 1) / hac.bse.iloc[1], resid_sd=float(np.sqrt(ols.scale)),
                     corr=float(np.corrcoef(d.r, d.m)[0, 1]))
    e = ols.resid.values
    R["desc"]["resid"] = describe(e)
    R["serial"] = {"aru": serial_block(d.r.values), "resid": serial_block(e), "xao": serial_block(mfull.values)}
    # --- Roll (1984) implied spread, overall and by tick regime
    x = d.r.values - d.r.mean(); cov1 = np.mean(x[1:] * x[:-1])
    R["roll"] = dict(cov1=cov1, spread=2 * np.sqrt(-cov1) if cov1 < 0 else float("nan"))
    prev_tick = d.tick.shift(1)
    for name, mask in (("low", prev_tick == 0.001), ("high", prev_tick == 0.005)):
        pair = pd.DataFrame({"a": d.r, "b": d.r.shift(1), "am": d.rmid, "bm": d.rmid.shift(1)})[mask]
        p = pair.dropna(subset=["a", "b"]); c = float(np.mean((p.a - d.r.mean()) * (p.b - d.r.mean())))
        pm = pair.dropna(subset=["am", "bm"])
        R["roll"][name] = dict(n=int(len(p)), rho1=float(np.corrcoef(p.a, p.b)[0, 1]), spread=2 * np.sqrt(-c) if c < 0 else float("nan"),
                               rel_tick=float(d.rel_tick[mask].mean()), quoted=float(2 * d.hs[mask].mean()),
                               rho1_mid=float(np.corrcoef(pm.am, pm.bm)[0, 1]), n_mid=int(len(pm)))
    # --- closing quotes: spread, location of the close, mid-quote returns
    q = df.dropna(subset=["mid"])
    loc = np.where(np.isclose(q.aru, q.bid), "bid", np.where(np.isclose(q.aru, q.ask), "ask", "other"))
    mid = df.rmid.dropna(); rc = df.r.reindex(mid.index)
    seq = df[["r", "rmid"]].copy(); seq["ok"] = seq.rmid.notna()
    # longest usable sample for serial tests on mid-quote returns: all days with a mid-quote return (gaps are isolated days)
    R["quotes"] = dict(n_days=int(len(q)), n_invalid=int(df.mid.isna().sum()), spread_mean=float(2 * q.hs.mean()), spread_median=float(2 * q.hs.median()),
                       half_mean=float(q.hs.mean()), one_tick=float((q.spread_ticks == 1).mean()), two_ticks=float((q.spread_ticks == 2).mean()),
                       more_ticks=float((q.spread_ticks > 2).mean()), at_bid=float((loc == "bid").mean()), at_ask=float((loc == "ask").mean()),
                       inside=float((loc == "other").mean()), n_mid=int(len(mid)), sd_mid=float(mid.std(ddof=1)), sd_close=float(rc.std(ddof=1)),
                       noise_share=float(1 - mid.var(ddof=1) / rc.var(ddof=1)), corr=float(np.corrcoef(mid, rc)[0, 1]),
                       zero_mid=float((mid == 0).mean()),
                       serial_mid=serial_block(mid.values), serial_close=serial_block(rc.values))
    # --- AR(1) fit and the contrarian rule  w_t = -sign(r_{t-1})
    ar = sm.OLS(d.r.values[1:], sm.add_constant(d.r.values[:-1])).fit(cov_type="HC0")
    R["ar1"] = dict(phi=ar.params[1], t_rob=ar.tvalues[1], r2=ar.rsquared)
    c = pd.DataFrame({"r": df.r, "rmid": df.rmid, "pos": -np.sign(df.r.shift(1)), "hs": df.hs.shift(1)}).dropna()
    c["turn"] = (c.pos - c.pos.shift(1).fillna(0)).abs()            # units traded at the previous close
    c["at_close"] = c.pos * c.r                                       # valued at closing prices
    c["at_mid"] = c.pos * c.rmid                                      # valued at quote mid-points
    c["net"] = c.at_mid - c.turn * c.hs                               # buy at the ask, sell at the bid
    R["contrarian"] = dict(n=int(len(c)), close=float(c.at_close.mean()), close_t=tstat(c.at_close), mid=float(c.at_mid.mean()),
                           mid_t=tstat(c.at_mid), net=float(c.net.mean()), net_t=tstat(c.net), turnover=float(c.turn.mean()),
                           cost=float((c.turn * c.hs).mean()), half_spread=float(c.hs.mean()), mean_abs_r=float(d.r.abs().mean()),
                           ann_close=float(252 * c.at_close.mean()), ann_net=float(252 * c.net.mean()))
    return R, d, e


if __name__ == "__main__":
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    R, d, e = run()
    json.dump(R, open(os.path.join(ROOT, "results", "part1.json"), "w"), indent=1, default=float)
    rnd = lambda v: {a: (round(float(b), 4) if not isinstance(b, (dict, str)) else b) for a, b in v.items()}
    print({k: R[k] for k in ("n_days", "n_halt", "n_ret", "start", "end", "first_ret")})
    for k, v in R["desc"].items(): print(k, rnd(v))
    for k in ("liq", "price", "lottery", "capm", "ar1", "contrarian", "jump"): print(k, rnd(R[k]))
    def show(B, name):
        print("==", name)
        print("  rho", {k: tuple(round(x, 3) for x in B[k].values()) for k in B if k.startswith("rho")})
        print("  Q  ", {k: tuple(round(x, 4) for x in B[k].values()) for k in B if k.startswith("Q")})
        print("  arch5", rnd(B["arch5"]), " VR", {q: tuple(round(x, 3) for x in v.values()) for q, v in B["vr"].items()})
    for s in ("aru", "resid", "xao"): show(R["serial"][s], s)
    print("roll", {k: (rnd(v) if isinstance(v, dict) else round(float(v), 4)) for k, v in R["roll"].items()})
    Q = R["quotes"]; print("quotes", rnd({k: v for k, v in Q.items() if not isinstance(v, dict)}))
    show(Q["serial_mid"], "mid-quote returns"); show(Q["serial_close"], "close returns, same days")
