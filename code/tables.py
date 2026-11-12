"""Writes the LaTeX tables and a file of number macros used in the text."""
import json, os, numpy as np, pandas as pd
import statsmodels.api as sm
from common import ROOT

P1 = json.load(open(os.path.join(ROOT, "results", "part1.json")))
P2 = json.load(open(os.path.join(ROOT, "results", "part2.json")))
F = pd.read_csv(os.path.join(ROOT, "results", "oos_var.csv"), index_col=0, parse_dates=True)
P3 = json.load(open(os.path.join(ROOT, "results", "part3.json"))); TK = json.load(open(os.path.join(ROOT, "results", "ticks.json")))
TAB = os.path.join(ROOT, "tables")


def f(x, d=2):
    s = f"{x:,.{d}f}".replace(",", "{,}")
    if float(f"{x:.{d}f}") == 0: s = s.replace("-", "")
    return "$-$" + s[1:] if s.startswith("-") else s
def p(x): return "$<$0.001" if x < 0.001 else f"{x:.3f}"
def paren(x, d=3): return "(" + f(x, d).replace("$-$", "$-$") + ")"
def write(name, s): open(os.path.join(TAB, name), "w").write(s)

# ---------------- Table 1: descriptive statistics ----------------
D = P1["desc"]; rows = [("Observations", "n", 0), ("Mean (\\%)", "mean", 3), ("Median (\\%)", "median", 3), ("Standard deviation (\\%)", "sd", 3),
        ("Annualised volatility (\\%)", "ann_vol", 1), ("Minimum (\\%)", "min", 2), ("Maximum (\\%)", "max", 2), ("Skewness", "skew", 2),
        ("Kurtosis", "kurt", 2), ("Jarque--Bera", "jb", 0), ("Zero-return days (\\%)", "zero", 1)]
s = "\\begin{tabular}{@{}lrrr@{}}\n\\toprule\n & ARU & XAO & Market-model residual \\\\\n\\midrule\n"
for lab, k, d in rows:
    vals = []
    for c in ("aru", "xao", "resid"):
        v = D[c][k] * (100 if k == "zero" else 1)
        vals.append("---" if (k == "zero" and c == "resid") else f(v, d))
    s += f"{lab} & " + " & ".join(vals) + " \\\\\n"
write("tab_desc.tex", s + "\\bottomrule\n\\end{tabular}\n")

# ---------------- Table 2: serial dependence ----------------
S = P1["serial"]; A, E = S["aru"], S["resid"]
s = "\\begin{tabular}{@{}l rrr c rrr@{}}\n\\toprule\n & \\multicolumn{3}{c}{Returns $r_t$} & & \\multicolumn{3}{c}{Market-model residuals $\\hat\\varepsilon_t$} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(l){6-8}\n"
s += "\\multicolumn{8}{@{}l}{\\textit{Panel A: Autocorrelations}} \\\\\n Lag $k$ & $\\hat\\rho_k$ & $t$ & $t^{*}$ & & $\\hat\\rho_k$ & $t$ & $t^{*}$ \\\\\n\\midrule\n"
for k in range(1, 6):
    a, e = A[f"rho{k}"], E[f"rho{k}"]
    s += f"{k} & {f(a['rho'],3)} & {f(a['t_iid'],2)} & {f(a['t_rob'],2)} & & {f(e['rho'],3)} & {f(e['t_iid'],2)} & {f(e['t_rob'],2)} \\\\\n"
s += "\\addlinespace\n\\multicolumn{8}{@{}l}{\\textit{Panel B: Portmanteau tests}} \\\\\n Lags $m$ & $Q_{\\mathrm{LB}}$ & $p$ & $p^{*}$ & & $Q_{\\mathrm{LB}}$ & $p$ & $p^{*}$ \\\\\n\\midrule\n"
for m in (5, 10, 20):
    a, e = A[f"Q{m}"], E[f"Q{m}"]
    s += f"{m} & {f(a['lb'],1)} & {p(a['lb_p'])} & {p(a['rob_p'])} & & {f(e['lb'],1)} & {p(e['lb_p'])} & {p(e['rob_p'])} \\\\\n"
s += "\\addlinespace\n\\multicolumn{8}{@{}l}{\\textit{Panel C: Variance ratios}} \\\\\n Horizon $q$ & $\\mathrm{VR}(q)$ & $z$ & $z^{*}$ & & $\\mathrm{VR}(q)$ & $z$ & $z^{*}$ \\\\\n\\midrule\n"
for q in ("2", "5", "10", "20"):
    a, e = A["vr"][q], E["vr"][q]
    s += f"{q} & {f(a['vr'],3)} & {f(a['z'],2)} & {f(a['zs'],2)} & & {f(e['vr'],3)} & {f(e['z'],2)} & {f(e['zs'],2)} \\\\\n"
s += "\\addlinespace\n\\multicolumn{8}{@{}l}{\\textit{Panel D: Dependence in squared series}} \\\\\n & Statistic & $p$ & & & Statistic & $p$ & \\\\\n\\midrule\n"
s += f"Ljung--Box $Q^2(10)$ & {f(A['Q2_10']['lb'],1)} & {p(A['Q2_10']['lb_p'])} & & & {f(E['Q2_10']['lb'],1)} & {p(E['Q2_10']['lb_p'])} & \\\\\n"
s += f"ARCH-LM(5) & {f(A['arch5']['lm'],1)} & {p(A['arch5']['p'])} & & & {f(E['arch5']['lm'],1)} & {p(E['arch5']['p'])} & \\\\\n"
write("tab_serial.tex", s + "\\bottomrule\n\\end{tabular}\n")

# ---------------- Table 3: tick regimes ----------------
R = P1["roll"]; QT = P1["quotes"]; SM = QT["serial_mid"]; SC = QT["serial_close"]; K = P1["contrarian"]
s = "\\begin{tabular}{@{}lrrrrrr@{}}\n\\toprule\n & & Tick/price & Quoted & \\multicolumn{2}{c}{Closing prices} & Mid-quotes \\\\\n\\cmidrule(lr){5-6}\\cmidrule(l){7-7}\n"
s += "Previous close & Days & (\\%) & spread (\\%) & $\\hat\\rho_1$ & Roll spread (\\%) & $\\hat\\rho_1$ \\\\\n\\midrule\n"
for key, lab in (("low", "Below A\\$0.10"), ("high", "A\\$0.10 and above")):
    v = R[key]
    s += f"{lab} & " + f"{v['n']:,}".replace(",", "{,}") + f" & {f(v['rel_tick'],2)} & {f(v['quoted'],2)} & {f(v['rho1'],3)} & {f(v['spread'],2)} & {f(v['rho1_mid'],3)} \\\\\n"
s += "\\midrule\nFull sample & " + f"{P1['n_ret']-1:,}".replace(",", "{,}") + f" & {f(P1['price']['rel_tick_mean'],2)} & {f(QT['spread_mean'],2)} & {f(A['rho1']['rho'],3)} & {f(R['spread'],2)} & {f(SM['rho1']['rho'],3)} \\\\\n"
write("tab_tick.tex", s + "\\bottomrule\n\\end{tabular}\n")

# ---------------- Table: closing prices versus quote mid-points ----------------
s = "\\begin{tabular}{@{}l rr@{}}\n\\toprule\n"
s += "\\multicolumn{3}{@{}l}{\\textit{Panel A: Closing quotes}} \\\\\n\\midrule\n"
s += f"Quoted spread, mean (\\% of mid-quote) & \\multicolumn{{2}}{{r}}{{{f(QT['spread_mean'],2)}}} \\\\\n"
s += f"Quoted spread, median (\\%) & \\multicolumn{{2}}{{r}}{{{f(QT['spread_median'],2)}}} \\\\\n"
s += f"Days with a spread of one tick / two ticks / more (\\%) & \\multicolumn{{2}}{{r}}{{{f(100*QT['one_tick'],0)} / {f(100*QT['two_ticks'],0)} / {f(100*QT['more_ticks'],0)}}} \\\\\n"
s += f"Close at the bid / at the ask / elsewhere (\\%) & \\multicolumn{{2}}{{r}}{{{f(100*QT['at_bid'],0)} / {f(100*QT['at_ask'],0)} / {f(100*QT['inside'],0)}}} \\\\\n"
s += "\\addlinespace\n\\multicolumn{3}{@{}l}{\\textit{Panel B: Serial dependence of daily returns}} \\\\\n & Closing price & Mid-quote \\\\\n\\midrule\n"
s += f"Standard deviation (\\%) & {f(QT['sd_close'],2)} & {f(QT['sd_mid'],2)} \\\\\n"
s += f"$\\hat\\rho_1$ & {f(SC['rho1']['rho'],3)} & {f(SM['rho1']['rho'],3)} \\\\\n"
s += f"\\quad robust $t^{{*}}$ & {f(SC['rho1']['t_rob'],2)} & {f(SM['rho1']['t_rob'],2)} \\\\\n"
s += f"$\\mathrm{{VR}}(2)$ & {f(SC['vr']['2']['vr'],3)} & {f(SM['vr']['2']['vr'],3)} \\\\\n"
s += f"\\quad robust $z^{{*}}$ & {f(SC['vr']['2']['zs'],2)} & {f(SM['vr']['2']['zs'],2)} \\\\\n"
s += f"$\\mathrm{{VR}}(5)$ & {f(SC['vr']['5']['vr'],3)} & {f(SM['vr']['5']['vr'],3)} \\\\\n"
s += f"\\quad robust $z^{{*}}$ & {f(SC['vr']['5']['zs'],2)} & {f(SM['vr']['5']['zs'],2)} \\\\\n"
s += f"Robust portmanteau $Q^{{*}}(5)$, $p$-value & {p(SC['Q5']['rob_p'])} & {p(SM['Q5']['rob_p'])} \\\\\n"
s += "\\addlinespace\n\\multicolumn{3}{@{}l}{\\textit{Panel C: Contrarian rule, average return (\\% per day)}} \\\\\n & Return & $t$-statistic \\\\\n\\midrule\n"
s += f"Valued at closing prices & {f(K['close'],2)} & {f(K['close_t'],2)} \\\\\n"
s += f"Valued at mid-quotes & {f(K['mid'],2)} & {f(K['mid_t'],2)} \\\\\n"
s += f"Buying at the ask and selling at the bid & {f(K['net'],2)} & {f(K['net_t'],2)} \\\\\n"
write("tab_quotes.tex", s + "\\bottomrule\n\\end{tabular}\n")

# ---------------- Table 4: model estimates ----------------
M = P2["full"]; names = list(M)
head = {"GARCH-N": "GARCH", "EGARCH-N": "EGARCH", "GARCH-t": "GARCH", "EGARCH-t": "EGARCH", "GARCH-st": "GARCH", "EGARCH-st": "EGARCH"}


s = "\\begin{tabular}{@{}l rr rr rr@{}}\n\\toprule\n & \\multicolumn{2}{c}{Normal} & \\multicolumn{2}{c}{Student-$t$} & \\multicolumn{2}{c}{Skewed $t$} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(l){6-7}\n & " + " & ".join(head[n] for n in names) + " \\\\\n\\midrule\n"
s += "\\multicolumn{7}{@{}l}{\\textit{Panel A: Parameter estimates}} \\\\\n"
prm = [("$c$", ["Const"]), ("$\\phi$", ["r[1]"]), ("$\\omega$", ["omega"]), ("$\\alpha$", ["alpha[1]"]), ("$\\gamma$", ["gamma[1]"]),
       ("$\\beta$", ["beta[1]"]), ("$\\nu$", ["nu", "eta"]), ("$\\lambda$", ["lambda"])]
for lab, keys in prm:
    est, se = [], []
    for n in names:
        k = next((k for k in keys if k in M[n]["params"]), None)
        est.append(f(M[n]["params"][k], 3) if k else "")
        se.append(paren(M[n]["se"][k], 3) if k else "")
    s += f"{lab} & " + " & ".join(est) + " \\\\\n & " + " & ".join(se) + " \\\\\n"
s += "\\addlinespace\n\\multicolumn{7}{@{}l}{\\textit{Panel B: Volatility dynamics and fit}} \\\\\n"
s += "Persistence & " + " & ".join(f(M[n]["persistence"], 3) for n in names) + " \\\\\n"
s += "Half-life (days) & " + " & ".join(f(M[n]["half_life"], 1) for n in names) + " \\\\\n"
s += "Log-likelihood & " + " & ".join(f(M[n]["ll"], 1) for n in names) + " \\\\\n"
s += "Parameters & " + " & ".join(str(M[n]["k"]) for n in names) + " \\\\\n"
s += "AIC & " + " & ".join(f(M[n]["aic"] * M[n]["n"], 1) for n in names) + " \\\\\n"
s += "BIC & " + " & ".join(f(M[n]["bic"] * M[n]["n"], 1) for n in names) + " \\\\\n"
s += "\\addlinespace\n\\multicolumn{7}{@{}l}{\\textit{Panel C: Diagnostics on standardised residuals ($p$-values)}} \\\\\n"
s += "Ljung--Box $Q(10)$ & " + " & ".join(p(M[n]["q10_p"]) for n in names) + " \\\\\n"
s += "Ljung--Box $Q^2(10)$ & " + " & ".join(p(M[n]["q2_10_p"]) for n in names) + " \\\\\n"
s += "ARCH-LM(5) & " + " & ".join(p(M[n]["arch5_p"]) for n in names) + " \\\\\n"
write("tab_models.tex", s + "\\bottomrule\n\\end{tabular}\n")

# ---------------- Table 5: out-of-sample backtests ----------------
O = dict(P2["oos"]); O["GARCH-J"] = P3["oos"]; lab = {"GARCH-J": "GARCH, Poisson jumps", "Normal": "Static Normal", "HS": "Historical simulation", "GARCH-N": "GARCH, Normal", "EGARCH-N": "EGARCH, Normal",
       "GARCH-t": "GARCH, Student-$t$", "EGARCH-t": "EGARCH, Student-$t$", "GARCH-st": "GARCH, skewed $t$", "EGARCH-st": "EGARCH, skewed $t$"}
order = ["Normal", "HS", "GARCH-N", "EGARCH-N", "GARCH-t", "EGARCH-t", "GARCH-st", "EGARCH-st", "GARCH-J"]
s = "\\begin{tabular}{@{}l rrrr c rrrr@{}}\n\\toprule\n & \\multicolumn{4}{c}{Left tail (long position)} & & \\multicolumn{4}{c}{Right tail (short position)} \\\\\n\\cmidrule(lr){2-5}\\cmidrule(l){7-10}\n"
s += " & Viol. & $p_{\\mathrm{UC}}$ & $p_{\\mathrm{CC}}$ & Loss & & Viol. & $p_{\\mathrm{UC}}$ & $p_{\\mathrm{CC}}$ & Loss \\\\\n\\midrule\n"
for lev, ttl in (("0.01", "Panel A: 1\\% tail probability"), ("0.05", "Panel B: 5\\% tail probability")):
    e = O["HS"][f"left_{lev}"]["expected"]
    s += f"\\multicolumn{{10}}{{@{{}}l}}{{\\textit{{{ttl}}} (expected violations: {e:.1f})}} \\\\\n"
    for n in order:
        L, Rr = O[n][f"left_{lev}"], O[n][f"right_{lev}"]
        s += f"{lab[n]} & {L['viol']} & {p(L['p_uc'])} & {p(L['p_cc'])} & {f(L['qloss'],3)} & & {Rr['viol']} & {p(Rr['p_uc'])} & {p(Rr['p_cc'])} & {f(Rr['qloss'],3)} \\\\\n"
    if lev == "0.01": s += "\\addlinespace\n"
write("tab_oos.tex", s + "\\bottomrule\n\\end{tabular}\n")


# ---------------- Table: GARCH with Poisson jumps ----------------
J = P3["params"]; JS = P3["se"]
s = "\\begin{tabular}{@{}l rrrrr c rrr@{}}\n\\toprule\n & \\multicolumn{5}{c}{Mean and diffusive variance} & & \\multicolumn{3}{c}{Jumps} \\\\\n\\cmidrule(lr){2-6}\\cmidrule(l){8-10}\n"
s += " & $c$ & $\\phi$ & $\\omega$ & $\\alpha$ & $\\beta$ & & $\\kappa$ & $\\theta$ & $\\delta$ \\\\\n\\midrule\n"
s += "Estimate & " + " & ".join(f(J[k], 3) for k in ("c", "phi", "omega", "alpha", "beta")) + " & & " + " & ".join(f(J[k], 3) for k in ("lam", "muJ", "sigJ")) + " \\\\\n"
s += "Standard error & " + " & ".join(paren(JS[k], 3) for k in ("c", "phi", "omega", "alpha", "beta")) + " & & " + " & ".join(paren(JS[k], 3) for k in ("lam", "muJ", "sigJ")) + " \\\\\n"
write("tab_jump.tex", s + "\\bottomrule\n\\end{tabular}\n")

# ---------------- Diebold-Mariano comparisons of quantile loss ----------------
def qloss(r, q, pr, tail):
    u = (r - q) if tail == "left" else (q - r)
    return (pr - (u < 0)) * u
DM = {}
for other in ("HS", "Normal", "GARCH-N", "EGARCH-t"):
    for pr, L, Rr in ((0.01, "L01", "R01"), (0.05, "L05", "R05")):
        for tail, c in (("left", L), ("right", Rr)):
            d = qloss(F.r, F[f"{other}_{c}"], pr, tail) - qloss(F.r, F[f"EGARCH-st_{c}"], pr, tail)
            m = sm.OLS(d.values, np.ones(len(d))).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
            DM[f"{other}_{tail}_{pr}"] = float(m.tvalues[0])
print("DM t-stats (positive = EGARCH-st has lower loss):"); [print("  ", k, round(v, 2)) for k, v in DM.items()]
json.dump(DM, open(os.path.join(ROOT, "results", "dm.json"), "w"), indent=1)

# ---------------- number macros for the text ----------------
mac = {}
def num(name, val, d=2, pct=False):
    s_ = f"{val:,.{d}f}".replace(",", "{,}")
    mac[name] = ("\\ensuremath{-}" + s_[1:]) if s_.startswith("-") else s_      # works in text and in math mode
C = P1["capm"]; PR = P1["price"]; LQ = P1["liq"]; LT = P1["lottery"]
num("nDays", P1["n_days"], 0); num("nHalt", P1["n_halt"], 0); num("nTraded", P1["n_days"] - P1["n_halt"], 0); num("nUsed", P1["n_ret"], 0)
num("nIndex", P1["n_index"], 0); num("nGarch", M["GARCH-N"]["n"], 0)
num("jumpSimple", P1["jump"]["simple"], 0); num("jumpLog", P1["jump"]["log"], 0)
num("volRatio", D["aru"]["sd"] / D["xao"]["sd"], 0)
num("liqMedian", LQ["median_value_m"], 1); num("liqMinVol", LQ["min_volume"], 0)
num("lotMonths", LT["n_months"], 0); num("lotShare", 100 * LT["share_gt10"], 0); num("lotMeanMax", LT["mean_max"], 0)
num("lotUpTen", LT["n_up10"], 0); num("lotDownTen", LT["n_down10"], 0)
_r1 = A["rho1"]["rho"]
num("vrImpFive", 1 + 2 * _r1 * 4 / 5, 2); num("vrImpTen", 1 + 2 * _r1 * 9 / 10, 2); num("vrImpTwenty", 1 + 2 * _r1 * 19 / 20, 2)
num("rhoSeven", A["rho7"]["rho"], 3); num("rhoSevenTrob", A["rho7"]["t_rob"], 2)
num("qSpread", QT["spread_mean"], 1); num("qSpreadMed", QT["spread_median"], 1); num("qHalf", QT["half_mean"], 1); num("qOneTick", 100 * QT["one_tick"], 0)
num("qTwoTick", 100 * QT["two_ticks"], 0); num("qAtBid", 100 * QT["at_bid"], 0); num("qAtAsk", 100 * QT["at_ask"], 0); num("qAtQuote", 100 * (QT["at_bid"] + QT["at_ask"]), 0)
num("sameVrTwo", SC["vr"]["2"]["vr"], 2); num("qNdays", QT["n_days"], 0); num("qInside", 100 * QT["inside"], 0); num("qNmid", QT["n_mid"], 0); num("qNoise", 100 * QT["noise_share"], 0); num("qInvalid", QT["n_invalid"], 0)
num("midRho", SM["rho1"]["rho"], 3); num("midRhoT", SM["rho1"]["t_iid"], 2); num("midRhoTrob", SM["rho1"]["t_rob"], 2)
num("midVrTwo", SM["vr"]["2"]["vr"], 2); num("midVrTwoZ", SM["vr"]["2"]["zs"], 2); num("midQp", SM["Q5"]["rob_p"], 2)
num("sameRho", SC["rho1"]["rho"], 3); num("sameRhoTrob", SC["rho1"]["t_rob"], 2); num("midDrop", 100 * (1 - SM["rho1"]["rho"] / SC["rho1"]["rho"]), 0)
num("qLowSpread", R["low"]["quoted"], 2); num("qHighSpread", R["high"]["quoted"], 2); num("midRhoLow", R["low"]["rho1_mid"], 2); num("midRhoHigh", R["high"]["rho1_mid"], 2)
num("aruSD", D["aru"]["sd"], 2); num("aruAnnVol", D["aru"]["ann_vol"], 0); num("xaoSD", D["xao"]["sd"], 2); num("xaoAnnVol", D["xao"]["ann_vol"], 0)
num("aruSkew", D["aru"]["skew"], 2); num("aruKurt", D["aru"]["kurt"], 1); num("aruMax", D["aru"]["max"], 1); num("aruMin", D["aru"]["min"], 1)
num("aruMean", D["aru"]["mean"], 3); num("zeroShare", 100 * D["aru"]["zero"], 0); num("xaoSkew", D["xao"]["skew"], 2)
num("capmAlpha", C["alpha"], 3); num("capmAlphaSE", C["alpha_se"], 3); num("capmAlphaT", C["alpha_t"], 2); num("capmBeta", C["beta"], 2)
num("capmBetaSE", C["beta_se"], 2); num("capmBetaT", C["beta_t"], 1); num("capmBetaOneT", C["beta1_t"], 2); num("capmRsq", 100 * C["r2"], 1)
num("capmIdio", 100 * (1 - C["r2"]), 0); num("capmCorr", C["corr"], 2); num("capmResidSD", C["resid_sd"], 2)
num("rhoOne", A["rho1"]["rho"], 3); num("rhoOneT", A["rho1"]["t_iid"], 2); num("rhoOneTrob", A["rho1"]["t_rob"], 2)
num("rhoOneRes", E["rho1"]["rho"], 3); num("rhoOneResTrob", E["rho1"]["t_rob"], 2)
num("vrTwo", A["vr"]["2"]["vr"], 2); num("vrTwoZ", A["vr"]["2"]["zs"], 2); num("vrFive", A["vr"]["5"]["vr"], 2); num("vrFiveZ", A["vr"]["5"]["zs"], 2)
num("vrTen", A["vr"]["10"]["vr"], 2); num("vrTenZ", A["vr"]["10"]["zs"], 2); num("vrTwenty", A["vr"]["20"]["vr"], 2); num("vrTwentyZ", A["vr"]["20"]["zs"], 2)
num("arRsq", 100 * P1["ar1"]["r2"], 1); num("meanAbsRet", K["mean_abs_r"], 1)
num("qTwoTen", A["Q2_10"]["lb"], 0); num("qTwoTenXao", S["xao"]["Q2_10"]["lb"], 0); num("archLM", A["arch5"]["lm"], 1)
num("xaoRhoOne", S["xao"]["rho1"]["rho"], 2); num("xaoRhoOneT", S["xao"]["rho1"]["t_iid"], 1); num("xaoRhoOneTrob", S["xao"]["rho1"]["t_rob"], 1)
num("tickMean", PR["rel_tick_mean"], 1); num("tickMedian", PR["rel_tick_median"], 1); num("tickMin", PR["rel_tick_min"], 1); num("tickMax", PR["rel_tick_max"], 1)
num("priceMin", PR["min"], 3); num("priceMax", PR["max"], 2); num("shareBelow", 100 * PR["share_below_10c"], 0)
num("rollSpread", R["spread"], 1); num("rollLow", R["low"]["spread"], 1); num("rollHigh", R["high"]["spread"], 1)
num("rhoLow", R["low"]["rho1"], 2); num("rhoHigh", R["high"]["rho1"], 2); num("tickLow", R["low"]["rel_tick"], 1); num("tickHigh", R["high"]["rel_tick"], 1)
num("conGross", K["close"], 2); num("conGrossT", K["close_t"], 1); num("conMid", K["mid"], 2); num("conMidT", K["mid_t"], 1)
num("conNet", K["net"], 2); num("conNetT", K["net_t"], 1); num("conLoss", -K["net"], 2); num("conCost", K["cost"], 2)
num("conTurn", K["turnover"], 2); num("conHalf", K["half_spread"], 1); num("conAnn", K["ann_close"], 0)
for n in names:
    t = n.replace("-", "").replace("GARCH", "G").replace("EG", "EG")
    key = {"GARCH-N": "gn", "EGARCH-N": "en", "GARCH-t": "gt", "EGARCH-t": "et", "GARCH-st": "gs", "EGARCH-st": "es"}[n]
    num(key + "LL", M[n]["ll"], 1); num(key + "Pers", M[n]["persistence"], 3); num(key + "Half", M[n]["half_life"], 0)
    num(key + "AIC", M[n]["aic"] * M[n]["n"], 1); num(key + "BIC", M[n]["bic"] * M[n]["n"], 1); num(key + "Phi", M[n]["params"]["r[1]"], 3)
    num(key + "C", M[n]["params"]["Const"], 2); num(key + "Kurt", M[n]["z_kurt"], 1); num(key + "Zmax", M[n]["z_max"], 1)
    if "nu" in M[n]["params"]: num(key + "Nu", M[n]["params"]["nu"], 2)
    if "eta" in M[n]["params"]: num(key + "Nu", M[n]["params"]["eta"], 2); num(key + "Lam", M[n]["params"]["lambda"], 3); num(key + "LamT", M[n]["params"]["lambda"] / M[n]["se"]["lambda"], 1)
    if "gamma[1]" in M[n]["params"]: num(key + "Gam", M[n]["params"]["gamma[1]"], 3); num(key + "GamT", M[n]["params"]["gamma[1]"] / M[n]["se"]["gamma[1]"], 1)
num("gnUncond", M["GARCH-N"]["uncond_vol"], 1); num("gnAlpha", M["GARCH-N"]["params"]["alpha[1]"], 3); num("gnBeta", M["GARCH-N"]["params"]["beta[1]"], 3); num("gnOmega", M["GARCH-N"]["params"]["omega"], 3)
num("lrNormT", 2 * (M["EGARCH-t"]["ll"] - M["EGARCH-N"]["ll"]), 0); num("lrTSkew", 2 * (M["EGARCH-st"]["ll"] - M["EGARCH-t"]["ll"]), 1)
num("lrTSkewG", 2 * (M["GARCH-st"]["ll"] - M["GARCH-t"]["ll"]), 1)
num("oosN", P2["oos_n"], 0); num("oosWindow", P2["T0"], 0); num("oosExpOne", 0.01 * P2["oos_n"], 1); num("oosExpFive", 0.05 * P2["oos_n"], 1)
qs = M["EGARCH-st"]["quant"]; num("esQone", qs[0], 2); num("esQninenine", qs[3], 2); qt = M["EGARCH-t"]["quant"]; num("etQone", qt[0], 2); num("etQninenine", qt[3], 2)
num("nQone", -2.326, 2); num("nQninenine", 2.326, 2)
num("avgVarESL", O["EGARCH-st"]["left_0.01"]["avg_var"], 1); num("avgVarESR", O["EGARCH-st"]["right_0.01"]["avg_var"], 1)
num("avgVarNL", O["Normal"]["left_0.01"]["avg_var"], 1); num("avgVarNR", O["Normal"]["right_0.01"]["avg_var"], 1)
_m = "EGARCH-st"; _hit = (F.r < F[_m + "_L01"]) | (F.r > F[_m + "_R01"]); _s = F[_m + "_sig"]
num("violN", int(_hit.sum()), 0); num("violSigRank", 100 * _s.rank(pct=True)[_hit].mean(), 0)
num("corrSigR", float(np.corrcoef(_s ** 2, F.r ** 2)[0, 1]), 2)
num("esJumpLimit", F.loc["2024-03-14", _m + "_R01"], 1); num("nJumpLimit", F.loc["2024-03-14", "Normal_R01"], 1)
num("jumpRatioN", F.loc["2024-03-14", "r"] / F.loc["2024-03-14", "Normal_R01"], 0)
num("jumpKappa", P3["params"]["lam"], 2); num("jumpTheta", P3["params"]["muJ"], 1); num("jumpDelta", P3["params"]["sigJ"], 1)
num("jumpThetaT", P3["params"]["muJ"] / P3["se"]["muJ"], 1); num("jumpLL", P3["ll"], 1); num("jumpAIC", P3["aic"], 1); num("jumpBIC", P3["bic"], 1)
num("jumpShare", 100 * P3["jump_var_share"], 0); num("jumpPerYear", P3["jumps_per_year"], 0); num("jumpN", P3["n_post50"], 0)
num("jumpUp", P3["post_up"], 0); num("jumpDown", P3["post_down"], 0); num("jumpPers", P3["persistence"], 2)
num("tickZero", TK["share0"], 0); num("tickOne", TK["share1"], 0); num("tickTwo", TK["share_le2"], 0)
for k, v in DM.items():
    a, b, c = k.split("_"); nm = "dm" + a.replace("-", "") + ("L" if b == "left" else "R") + ("One" if c == "0.01" else "Five")
    num(nm, v, 2)
open(os.path.join(ROOT, "numbers.tex"), "w").write("% generated by code/tables.py -- do not edit\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in mac.items()))
print(len(mac), "macros"); print({k: v for k, v in mac.items()})
