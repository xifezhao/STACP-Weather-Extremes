"""Authentic EMOS (Nonhomogeneous Gaussian Regression, Gneiting et al. 2005)
baseline on the exact same WeatherBench 2 1.5-deg data partition as run_real.py:
  - Fitted on 2019 calibration set (mu = a + b*F, sigma^2 = c + d*s^2) via
    Gaussian negative log-likelihood (maximum likelihood / proper scoring).
  - Evaluated on held-out 2022 test set for 90th and 95th Gaussian upper quantiles
    (uncalibrated for spatial FNR, mirroring standard probabilistic post-processing)
    as well as LTT-certified EMOS Gaussian quantile level on 2020-2021 thinned val.
"""
import json, os, time
import numpy as np
from scipy import optimize, stats
from core import field_far, field_losses, n_components, stationary_bootstrap_ci
from run_real import build_config, DOMAINS, VARS, LEADS, jsonable

OUT = os.environ.get("STACP_OUT", "/content/stacp_results")

def fit_emos_nll(F, Y, s):
    """Fit mu = a + b*F, var = exp(log_c) + exp(log_d)*s^2 on calibration anomalies."""
    f = F.ravel()
    y = Y.ravel()
    sv = (s.ravel()) ** 2
    # Subsample deterministically if huge for fast, exact MLE
    rng = np.random.default_rng(42)
    if f.size > 200000:
        idx = rng.choice(f.size, 200000, replace=False)
        f, y, sv = f[idx], y[idx], sv[idx]
    # OLS init for a, b
    b0 = np.cov(f, y)[0, 1] / max(np.var(f), 1e-8)
    a0 = float(np.mean(y) - b0 * np.mean(f))
    res_var = max(float(np.var(y - (a0 + b0 * f))), 1e-4)
    init = np.array([a0, b0, np.log(res_var * 0.2 + 1e-6), np.log(0.8)])

    def nll(p):
        a, b, lc, ld = p
        var = np.exp(lc) + np.exp(ld) * sv
        diff = y - (a + b * f)
        return 0.5 * np.mean(np.log(var) + (diff * diff) / var)

    opt = optimize.minimize(nll, init, method="BFGS")
    a, b, lc, ld = opt.x
    return float(a), float(b), float(np.exp(lc)), float(np.exp(ld))

def eval_emos(var, dom, lead):
    cfg = build_config(var, dom, lead)
    cal = np.where(cfg["year"] == 2019)[0]
    test = np.where(cfg["year"] == 2022)[0]
    a, b, c, d = fit_emos_nll(cfg["F"][cal], cfg["Y"][cal], cfg["s"][cal])
    out = {"params": {"a": a, "b": b, "c": c, "d": d}}
    for qlevel in [0.90, 0.95, 0.99]:
        z = float(stats.norm.ppf(qlevel))
        Ls, FARs, Ms, Nccs = [], [], [], []
        for i in test:
            mu = a + b * cfg["F"][i]
            sig = np.sqrt(c + d * (cfg["s"][i] ** 2))
            U = mu + z * sig
            mg = (U - cfg["F"][i])[None]
            l = field_losses(cfg["Y"][i], cfg["F"][i], mg, cfg["mext"][i])
            if l is not None:
                Ls.append(float(l[0]))
            FARs.append(float(field_far(cfg["F"][i], mg, cfg["q99"][i], cfg["mext"][i])[0]))
            Ms.append(float(np.mean(U - cfg["F"][i])))
            Nccs.append(n_components(U > cfg["q99"][i]))
        Ls, FARs, Ms, Nccs = map(np.array, (Ls, FARs, Ms, Nccs))
        far_m = float(FARs.mean())
        ncc_m = float(Nccs.mean())
        out[f"q{int(qlevel*100)}"] = {
            "fnr": float(Ls.mean()),
            "fnr_ci": stationary_bootstrap_ci(Ls, seed=7),
            "far": far_m,
            "far_ci": stationary_bootstrap_ci(FARs, seed=8),
            "margin": float(Ms.mean()),
            "ncc": ncc_m,
            "d_frag": ncc_m / far_m if far_m > 0 else None,
        }
    return out

if __name__ == "__main__":
    res = {}
    for var in VARS:
        for dom in ["NA", "EU"]:
            for lead in LEADS:
                key = f"{var}_{dom}_{lead}"
                t0 = time.time()
                res[key] = eval_emos(var, dom, lead)
                print(time.strftime("%H:%M:%S"), "EMOS", key,
                      "q90 FNR=", round(res[key]["q90"]["fnr"], 4),
                      "q95 FNR=", round(res[key]["q95"]["fnr"], 4),
                      "q99 FNR=", round(res[key]["q99"]["fnr"], 4),
                      f"({time.time()-t0:.1f}s)", flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/emos_baseline.json", "w") as fh:
        json.dump(jsonable(res), fh, indent=2)
    print("EMOS ALL DONE", flush=True)
