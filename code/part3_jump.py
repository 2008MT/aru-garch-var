"""Part III: AR(1)-GARCH(1,1) with compound-Poisson jumps (Jorion 1988).

    r_t   = c + phi r_{t-1} + eps_t
    eps_t = sqrt(h_t) z_t + sum_{k=1}^{N_t} J_k - lam*muJ ,   z_t ~ N(0,1), N_t ~ Poisson(lam), J_k ~ N(muJ, sigJ^2)
    h_t   = omega + alpha eps_{t-1}^2 + beta h_{t-1}

Conditional on N_t = n the innovation is Normal with mean (n - lam) muJ and variance h_t + n sigJ^2, so the
density is a Poisson-weighted mixture of Normals.  Writes results/part3.json and results/oos_jump.csv."""
import json, os, sys, time, warnings, numpy as np, pandas as pd
from scipy import optimize, stats
from scipy.signal import lfilter
from common import load, ROOT
from part2_var import backtest, T0

warnings.filterwarnings("ignore")
NMAX = 20
NS = np.arange(NMAX + 1)
NAMES = ["c", "phi", "omega", "alpha", "beta", "lam", "muJ", "sigJ"]


def unpack(x):
    """optimiser parameters -> model parameters (alpha = p*s, beta = p*(1-s) keeps alpha+beta = p < 1)."""
    c, phi, omega, p, s, lam, muJ, sigJ = x
    return np.array([c, phi, omega, p * s, p * (1 - s), lam, muJ, sigJ])


def filt(th, r):
    """innovations eps_t and diffusive variances h_t for t = 1..n-1, plus the one-step-ahead h_n."""
    c, phi, omega, alpha, beta, lam, muJ, sigJ = th
    eps = r[1:] - c - phi * r[:-1]
    h0 = max(eps.var() - lam * (muJ ** 2 + sigJ ** 2), 0.05 * eps.var())
    x = omega + alpha * np.concatenate(([h0], eps ** 2))          # x[0] built so that h[0] ~ h0
    x[0] = h0 * (1 - beta)
    h = lfilter([1.0], [1.0, -beta], x, zi=[beta * h0])[0]        # h[0] = h0, h[t] = omega + alpha eps[t-1]^2 + beta h[t-1]
    return eps, h[:-1], h[-1]


def mix_pdf(e, h, th):
    lam, muJ, sigJ = th[5], th[6], th[7]
    w = stats.poisson.pmf(NS, lam)
    m = (NS - lam) * muJ; v = h[:, None] + NS[None, :] * sigJ ** 2
    return (w * np.exp(-0.5 * (e[:, None] - m) ** 2 / v) / np.sqrt(2 * np.pi * v))


def ll_obs(th, r):
    eps, h, _ = filt(th, r)
    return np.log(mix_pdf(eps, h, th).sum(1) + 1e-300)


def negll(x, r):
    v = -ll_obs(unpack(x), r).sum()
    return v if np.isfinite(v) else 1e10


BOUNDS = [(-2, 2), (-0.6, 0.6), (1e-3, 60), (1e-3, 0.999), (1e-3, 0.999), (1e-3, 2.0), (-30, 30), (0.5, 60)]


def fit(r, start=None):
    starts = [start] if start is not None else [
        np.array([0.0, -0.1, 1.0, 0.9, 0.1, lam, mu, sg]) for lam in (0.02, 0.1, 0.4) for mu in (0.0, 4.0) for sg in (6.0, 14.0)]
    best = None
    for s0 in starts:
        o = optimize.minimize(negll, s0, args=(r,), method="L-BFGS-B", bounds=BOUNDS, options={"maxiter": 2000, "maxfun": 20000})
        if best is None or o.fun < best.fun: best = o
    return best


def mix_quantile(p, mu, h, th):
    """p-quantile of mu + eps for each day (vectorised bisection on the mixture cdf)."""
    lam, muJ, sigJ = th[5], th[6], th[7]
    w = stats.poisson.pmf(NS, lam); m = (NS - lam) * muJ
    h = np.atleast_1d(h); s = np.sqrt(h[:, None] + NS[None, :] * sigJ ** 2)
    lo = np.full(len(h), -400.0); hi = np.full(len(h), 400.0)
    for _ in range(70):
        mid = 0.5 * (lo + hi)
        cdf = (w * stats.norm.cdf((mid[:, None] - m) / s)).sum(1)
        lo = np.where(cdf < p, mid, lo); hi = np.where(cdf < p, hi, mid)
    return mu + 0.5 * (lo + hi)


def sandwich(th, r, rel=3e-3):
    k = len(th); hstep = rel * np.maximum(np.abs(th), 0.1)
    f = lambda t: ll_obs(t, r)
    n = len(f(th)); S = np.zeros((n, k))
    for i in range(k):
        e = np.zeros(k); e[i] = hstep[i]; S[:, i] = (f(th + e) - f(th - e)) / (2 * hstep[i])
    H = np.zeros((k, k)); F = lambda t: f(t).sum()
    for i in range(k):
        for j in range(i, k):
            ei = np.zeros(k); ej = np.zeros(k); ei[i] = hstep[i]; ej[j] = hstep[j]
            H[i, j] = H[j, i] = (F(th + ei + ej) - F(th + ei - ej) - F(th - ei + ej) + F(th - ei - ej)) / (4 * hstep[i] * hstep[j])
    Hi = np.linalg.inv(-H)
    return np.sqrt(np.diag(Hi @ (S.T @ S) @ Hi))


if __name__ == "__main__":
    rs = load().r.dropna(); r = rs.values; n = len(r)
    o = fit(r); th = unpack(o.x); se = sandwich(th, r)
    eps, h, _ = filt(th, r); k = len(th); T = len(eps); ll = -o.fun
    lam, muJ, sigJ = th[5], th[6], th[7]
    dens = mix_pdf(eps, h, th); post = 1 - dens[:, 0] / dens.sum(1)            # P(N_t >= 1 | r_t)
    jv = lam * (muJ ** 2 + sigJ ** 2)
    R = dict(params=dict(zip(NAMES, th)), se=dict(zip(NAMES, se)), ll=ll, k=k, n=T, aic=-2 * ll + 2 * k, bic=-2 * ll + k * np.log(T),
             persistence=th[3] + th[4], jumps_per_year=252 * lam, days_between=1 / lam, jump_var=jv, mean_h=float(h.mean()),
             jump_var_share=float(jv / (h.mean() + jv)), n_post50=int((post > 0.5).sum()), n_post90=int((post > 0.9).sum()),
             post_up=int(((post > 0.5) & (eps > 0)).sum()), post_down=int(((post > 0.5) & (eps < 0)).sum()),
             post_20240314=float(post[rs.index[1:] == pd.Timestamp("2024-03-14")][0]),
             uncond_sd=float(np.sqrt(th[2] / (1 - th[3] - th[4]) + jv)))
    print("LL %.2f AIC %.1f BIC %.1f" % (ll, R["aic"], R["bic"]))
    print(pd.DataFrame({"est": th, "se": se}, index=NAMES).round(4))
    print({a: (round(b, 3) if isinstance(b, float) else b) for a, b in R.items() if a not in ("params", "se")})
    pd.Series(post, index=rs.index[1:], name="post").to_csv(os.path.join(ROOT, "results", "jump_posterior.csv"))

    if "--no-oos" not in sys.argv:
        t_start = time.time(); Q = np.full((n, 4), np.nan); x0 = o.x; probs = [0.01, 0.05, 0.95, 0.99]
        for t0 in range(T0, n):
            oo = fit(r[:t0], start=x0)
            if (t0 - T0) % 100 == 0:                                   # periodic global restart guards against local optima
                og = fit(r[:t0]); oo = og if og.fun < oo.fun - 1e-6 else oo
            x0 = oo.x; tt = unpack(oo.x)
            _, _, hnext = filt(tt, r[:t0]); mu = tt[0] + tt[1] * r[t0 - 1]
            Q[t0] = [mix_quantile(p, mu, hnext, tt)[0] for p in probs]
        F = pd.DataFrame(Q, index=rs.index, columns=["GARCH-J_L01", "GARCH-J_L05", "GARCH-J_R05", "GARCH-J_R01"]).iloc[T0:]
        F["r"] = rs.iloc[T0:]; F.to_csv(os.path.join(ROOT, "results", "oos_jump.csv"))
        R["oos"] = {"left_0.01": backtest(F.r, F["GARCH-J_L01"], 0.01, "left"), "right_0.01": backtest(F.r, F["GARCH-J_R01"], 0.01, "right"),
                    "left_0.05": backtest(F.r, F["GARCH-J_L05"], 0.05, "left"), "right_0.05": backtest(F.r, F["GARCH-J_R05"], 0.05, "right")}
        print("OOS done in %.0fs" % (time.time() - t_start))
        print({a: (b["viol"], round(b["p_uc"], 3), round(b["p_cc"], 3), round(b["qloss"], 4)) for a, b in R["oos"].items()})
    json.dump(R, open(os.path.join(ROOT, "results", "part3.json"), "w"), indent=1, default=float)
