"""STACP v2 real-data experiments on WeatherBench2 (ERA5 truth, Pangu-Weather base model).

Data partition by VALID date (no overlap):
  climatology (q99, mean, sd)  : ERA5 1990-2017
  scale model s(X)             : 2018
  calibration (scores, GPD)    : 2019
  validation (LTT)             : 2020, 2021 (thinned, dt = 5 d); pooled 2020-2021
  test                         : 2022
Every number written to the output JSON is computed here.  No constants are
injected.  Run:  python run_real.py  (expects /content/stacp_data/*.npz)
"""
import json, math, os, sys, time
import numpy as np
import pandas as pd
from core import (BETA_GRID, BinnedFamily, SpatialOp, field_far, field_losses,
                  fixed_sequence_test, hb_pvalue, margins_binned, margins_phys_offset,
                  margins_std_offset, n_components, n_min, rhat_max_lattice,
                  spatial_cluster_maxima, stationary_bootstrap_ci,
                  stationary_bootstrap_indices, gpd_pwm)

D = os.environ.get("STACP_DATA", "/content/stacp_data")
OUT = os.environ.get("STACP_OUT", "/content/stacp_results")
DOMAINS = {"NA": dict(lat=(25, 60), lon=(230, 300)),   # North America
           "EU": dict(lat=(35, 70), lon=(-10, 40))}    # Europe
VARS = ["t2m", "z500"]
LEADS = [72, 120, 168]
ALPHAS = [0.05, 0.10]
DELTA = 0.05
DT = 5
EDGES = np.array([-1.0, -0.33, 0.33, 1.0])   # bins of standardised forecast anomaly
OP = SpatialOp(r=1, sigma=1.0)
LAM = np.round(np.arange(15.0, 0.0, -0.05), 3)                    # standardised offset grid
CPHYS = {"t2m": np.round(np.arange(30.0, 0.0, -0.1), 2),          # K
         "z500": np.round(np.arange(500.0, 0.0, -1.0), 1)}        # m
SEASON = {12: "DJF", 1: "DJF", 2: "DJF", 3: "MAM", 4: "MAM", 5: "MAM",
          6: "JJA", 7: "JJA", 8: "JJA", 9: "SON", 10: "SON", 11: "SON"}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if (o != o or o in (float("inf"), float("-inf"))) else round(float(o), 6)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


# ---------------------------------------------------------------------------
def domain_index(lat, lon, dom):
    la = np.where((lat >= dom["lat"][0]) & (lat <= dom["lat"][1]))[0]
    lo0, lo1 = dom["lon"]
    if lo0 < 0:
        lo = np.r_[np.where(lon >= 360 + lo0)[0], np.where(lon <= lo1)[0]]
    else:
        lo = np.where((lon >= lo0) & (lon <= lo1))[0]
    return la, lo


def climatology(era, times, la, lo):
    t = pd.DatetimeIndex(times)
    sel = (t.year >= 1990) & (t.year <= 2017)
    X = era[sel][:, la][:, :, lo].astype(np.float64)
    doy = t[sel].dayofyear.values
    q99 = np.zeros((367,) + X.shape[1:]); mu = np.zeros_like(q99); sd = np.zeros_like(q99)
    for d in range(1, 367):
        dist = np.abs(doy - d); dist = np.minimum(dist, 366 - dist)
        w = X[dist <= 15]
        q99[d] = np.quantile(w, 0.99, axis=0); mu[d] = w.mean(0); sd[d] = w.std(0)
    return q99, mu, sd


def build_config(var, dom_name, lead):
    e = np.load(f"{D}/era5_{var}.npz"); p = np.load(f"{D}/pangu_{var}.npz")
    la, lo = domain_index(e["lat"], e["lon"], DOMAINS[dom_name])
    era = e["data"]; et = e["time"].astype("datetime64[D]")
    eidx = {d: i for i, d in enumerate(et)}
    q99c, muc, sdc = climatology(era, e["time"], la, lo)
    li = list(p["leads"]).index(lead)
    init = p["time"].astype("datetime64[D]")
    valid = init + np.timedelta64(lead // 24, "D")
    ok = np.array([v in eidx for v in valid])
    init, valid = init[ok], valid[ok]
    F = p["data"][ok, li][:, la][:, :, lo].astype(np.float64)
    Y = np.stack([era[eidx[v]] for v in valid])[:, la][:, :, lo].astype(np.float64)
    vt = pd.DatetimeIndex(valid)
    doy = vt.dayofyear.values
    cfg = dict(var=var, dom=dom_name, lead=lead, lat=e["lat"][la], lon=e["lon"][lo],
               init=init, valid=valid, year=vt.year.values, month=vt.month.values,
               F=F, Y=Y, q99=q99c[doy], mu=muc[doy], sd=sdc[doy])
    # scale model s(X): RMS residual per grid point and calendar month (3-month window), 2018
    r = Y - F
    s_m = np.zeros((13,) + F.shape[1:])
    y18 = cfg["year"] == 2018
    for m in range(1, 13):
        mm = {(m - 2) % 12 + 1, m, m % 12 + 1}
        sel = y18 & np.isin(cfg["month"], list(mm))
        s_m[m] = np.sqrt(np.mean(r[sel] ** 2, axis=0))
    cfg["s"] = np.maximum(s_m[cfg["month"]], 1e-3)
    cfg["bins"] = np.digitize((F - cfg["mu"]) / np.maximum(cfg["sd"], 1e-6), EDGES)
    cfg["mext"] = Y > cfg["q99"]
    return cfg


def calibration_sets(cfg, u_prob=0.95):
    sel = np.where(cfg["year"] == 2019)[0]
    S = np.maximum(cfg["Y"][sel] - cfg["F"][sel], 0.0) / cfg["s"][sel]
    B = cfg["bins"][sel]
    M = len(EDGES) + 1
    scores = [S[B == m] for m in range(M)]
    us = [np.quantile(sc[sc > 0], u_prob) if np.sum(sc > 0) > 50 else np.quantile(S[S > 0], u_prob)
          for sc in scores]
    cms = [np.concatenate([spatial_cluster_maxima(S[i], B[i] == m, us[m]) for i in range(len(sel))])
           for m in range(M)]
    return S, B, scores, cms


def threshold_diagnostic(S, B):
    out = []
    M = len(EDGES) + 1
    for m in range(M):
        sc = S[B == m]
        row = {"bin": m, "N": int(sc.size)}
        for up in [0.90, 0.925, 0.95, 0.975, 0.99]:
            u = np.quantile(sc[sc > 0], up)
            cm = np.concatenate([spatial_cluster_maxima(S[i], B[i] == m, u) for i in range(S.shape[0])])
            exc = cm[cm > u] - u
            if exc.size >= 10:
                xi, sig, xi_raw = gpd_pwm(exc)
                row[f"u{up}"] = dict(u=u, n_exc=int(exc.size), mean_excess=float(exc.mean()),
                                     xi_raw=xi_raw, xi=xi, sigma=sig, sigma_star=sig - xi * u)
            else:
                row[f"u{up}"] = dict(u=u, n_exc=int(exc.size))
        out.append(row)
    return out


# ---------------------------------------------------------------------------
def make_families(cfg, scores, cms, S_cal):
    fam = {"emp": BinnedFamily(scores, kind="emp"),
           "gpd": BinnedFamily(scores, cms, kind="gpd"),
           "gpd_narrow": BinnedFamily(scores, cms, kind="gpd", xi_clip=(-0.10, 0.30)),
           "gpd_wide": BinnedFamily(scores, cms, kind="gpd", xi_clip=(-0.30, 0.60))}
    q_pt = float(np.quantile(S_cal, 0.90))
    cphys = CPHYS[cfg["var"]]
    gens = {
        "emp_noop": lambda i: margins_binned(fam["emp"], cfg["bins"][i], cfg["s"][i], None),
        "emp_op": lambda i: margins_binned(fam["emp"], cfg["bins"][i], cfg["s"][i], OP),
        "gpd_noop": lambda i: margins_binned(fam["gpd"], cfg["bins"][i], cfg["s"][i], None),
        "gpd_op": lambda i: margins_binned(fam["gpd"], cfg["bins"][i], cfg["s"][i], OP),
        "gpdN_op": lambda i: margins_binned(fam["gpd_narrow"], cfg["bins"][i], cfg["s"][i], OP),
        "gpdW_op": lambda i: margins_binned(fam["gpd_wide"], cfg["bins"][i], cfg["s"][i], OP),
        "off_std": lambda i: margins_std_offset(LAM, cfg["s"][i]),
        "off_phys": lambda i: margins_phys_offset(cphys, cfg["F"].shape[1:]),
        "pointwise": lambda i: (q_pt * cfg["s"][i])[None],
    }
    params = {k: BETA_GRID for k in ["emp_noop", "emp_op", "gpd_noop", "gpd_op", "gpdN_op", "gpdW_op"]}
    params.update(off_std=LAM, off_phys=cphys, pointwise=np.array([0.90]))
    return fam, gens, params, q_pt


def loss_tables(cfg, gen, idx):
    n = len(idx)
    K = gen(idx[0]).shape[0]
    L = np.full((n, K), np.nan); FAR = np.zeros((n, K)); MW = np.zeros((n, K)); BA = np.zeros((n, K))
    for j, i in enumerate(idx):
        mg = gen(i)
        l = field_losses(cfg["Y"][i], cfg["F"][i], mg, cfg["mext"][i])
        if l is not None:
            L[j] = l
        FAR[j] = field_far(cfg["F"][i], mg, cfg["q99"][i], cfg["mext"][i])
        MW[j] = mg.reshape(K, -1).mean(1)
        BA[j] = (cfg["F"][i][None] + mg - cfg["mu"][i][None]).reshape(K, -1).mean(1)
    return dict(L=L, FAR=FAR, MW=MW, BA=BA)


def thinned(idx, dates):
    d = pd.DatetimeIndex(dates[idx])
    day = (d - pd.Timestamp(year=d[0].year, month=1, day=1)).days
    return idx[(day % DT) == 0]


def certify(T, rows, alpha):
    L = T["L"][rows]
    L = L[~np.isnan(L[:, 0])]
    n = L.shape[0]
    curve = L.mean(0) if n > 0 else np.full(T["L"].shape[1], np.nan)
    mono = bool(np.all(np.diff(curve) >= -1e-12)) if n > 0 else None
    if n == 0:
        return dict(n=0, khat=-1, mono=mono, curve=curve, pvals=np.array([]))
    khat, pv = fixed_sequence_test(curve, n, alpha, DELTA)
    return dict(n=n, khat=khat, mono=mono, curve=curve, pvals=pv)


def test_metrics(cfg, gen, Ttest, test_idx, k, seed=0, with_ncc=True):
    if k < 0:
        return {"certified": False}
    L = Ttest["L"][:, k]; ext = ~np.isnan(L)
    out = {"certified": True,
           "fnr": float(np.mean(L[ext])), "fnr_ci": stationary_bootstrap_ci(L[ext], seed=seed),
           "far": float(Ttest["FAR"][:, k].mean()), "far_ci": stationary_bootstrap_ci(Ttest["FAR"][:, k], seed=seed + 1),
           "margin": float(Ttest["MW"][:, k].mean()), "margin_ci": stationary_bootstrap_ci(Ttest["MW"][:, k], seed=seed + 2),
           "bound_anom": float(Ttest["BA"][:, k].mean()),
           "n_test_ext": int(ext.sum()), "n_test": int(len(L))}
    if with_ncc:
        ncc = np.array([n_components((cfg["F"][i] + gen(i)[k]) > cfg["q99"][i]) for i in test_idx])
        out.update(ncc=float(ncc.mean()), ncc_ci=stationary_bootstrap_ci(ncc, seed=seed + 3))
    return out


def autocorr(x, maxlag=10):
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    x = x - x.mean(); v = np.dot(x, x)
    return [float(np.dot(x[:-h], x[h:]) / v) if v > 0 else float("nan") for h in range(1, maxlag + 1)]


# ---------------------------------------------------------------------------
def run_config(var, dom, lead, main):
    t0 = time.time()
    cfg = build_config(var, dom, lead)
    S_cal, B_cal, scores, cms = calibration_sets(cfg)
    fam, gens, params, q_pt = make_families(cfg, scores, cms, S_cal.ravel())
    res = {"var": var, "domain": dom, "lead_h": lead,
           "grid_shape": list(cfg["F"].shape[1:]), "lat": cfg["lat"], "lon": cfg["lon"],
           "gpd_bins": fam["gpd"].info, "emp_bins": fam["emp"].info, "q_pointwise": q_pt}
    yr = cfg["year"]
    idx = {y: np.where(yr == y)[0] for y in [2018, 2019, 2020, 2021, 2022]}
    th = {y: thinned(idx[y], cfg["valid"]) for y in [2020, 2021]}
    th["pooled"] = np.r_[th[2020], th[2021]]
    daily_val = np.r_[idx[2020], idx[2021]]
    test_idx = idx[2022]
    # ---- N_ext diagnostics on thinned validation
    next_ = cfg["mext"].reshape(len(yr), -1).sum(1)
    res["n_ext_diag"] = {str(k): dict(n_fields=int(len(v)), n_with_ext=int((next_[v] > 0).sum()),
                                      next_min=int(next_[v].min()), next_median=float(np.median(next_[v])),
                                      next_max=int(next_[v].max()), n_gridpoints=int(next_.size and cfg["mext"][0].size))
                         for k, v in th.items()}
    res["n_ext_test"] = dict(n_fields=int(len(test_idx)), n_with_ext=int((next_[test_idx] > 0).sum()),
                             next_median=float(np.median(next_[test_idx])))
    # ---- residual diagnostics
    res["rmse_test"] = float(np.sqrt(np.mean((cfg["Y"][test_idx] - cfg["F"][test_idx]) ** 2)))
    fams = ["emp_noop", "emp_op", "gpd_noop", "gpd_op", "off_std", "off_phys", "pointwise"]
    if main:
        fams += ["gpdN_op", "gpdW_op"]
    # ---- loss tables
    Tval, Ttest = {}, {}
    pos = {i: j for j, i in enumerate(daily_val)}
    for f in fams:
        Tval[f] = loss_tables(cfg, gens[f], daily_val)
        Ttest[f] = loss_tables(cfg, gens[f], test_idx)
    rows = {k: np.array([pos[i] for i in v]) for k, v in th.items()}
    # ---- certification on pooled + single years
    res["cert"] = {}
    for a in ALPHAS:
        A = {"n_min": n_min(a, DELTA)}
        for vs in ["pooled", 2020, 2021]:
            n_ext_vs = int((next_[th[vs]] > 0).sum())
            A[str(vs)] = {"n": n_ext_vs, "rhat_max": rhat_max_lattice(n_ext_vs, a, DELTA)}
            for f in fams:
                if f == "pointwise":
                    continue
                if vs != "pooled" and f not in ["emp_noop", "emp_op", "gpd_noop", "gpd_op"]:
                    continue
                c = certify(Tval[f], rows[vs], a)
                k = c["khat"]
                entry = {"khat": k, "param": float(params[f][k]) if k >= 0 else None,
                         "rhat": float(c["curve"][k]) if k >= 0 else None,
                         "p_at_khat": float(c["pvals"][k]) if k >= 0 else None,
                         "rhat_first": float(c["curve"][0]),
                         "monotone": c["mono"],
                         "val_margin": float(Tval[f]["MW"][rows[vs], k].mean()) if k >= 0 else None}
                entry["test"] = test_metrics(cfg, gens[f], Ttest[f], test_idx, k, seed=7,
                                             with_ncc=(vs == "pooled"))
                if vs == "pooled" and main and a == 0.10:
                    entry["curve"] = c["curve"]; entry["pvals"] = c["pvals"]
                A[str(vs)][f] = entry
        # pointwise (not certified, fixed level)
        A["pooled"]["pointwise"] = {"param": 0.90, "test": test_metrics(cfg, gens["pointwise"], Ttest["pointwise"], test_idx, 0, seed=7)}
        res["cert"][str(a)] = A
    # ---- bootstrap variability of beta-hat (validation resampling), alpha = 0.10
    res["boot"] = {}
    r_pool = rows["pooled"]
    BI = stationary_bootstrap_indices(len(r_pool), B=500, mean_block=2, seed=11)
    for f in ["emp_op", "gpd_op", "off_std"]:
        Lv = Tval[f]["L"][r_pool]
        Lt = Ttest[f]["L"]; ext = ~np.isnan(Lt[:, 0])
        fnr_k = np.nanmean(Lt[ext], axis=0); mw_k = Ttest[f]["MW"].mean(0); far_k = Ttest[f]["FAR"].mean(0)
        ks = []
        for b in range(BI.shape[0]):
            Lb = Lv[BI[b]]; Lb = Lb[~np.isnan(Lb[:, 0])]
            ks.append(fixed_sequence_test(Lb.mean(0), Lb.shape[0], 0.10, DELTA)[0])
        ks = np.array(ks); cert = ks >= 0
        q = lambda v: [float(np.quantile(v, x)) for x in (0.05, 0.5, 0.95)] if v.size else None
        res["boot"][f] = {"cert_frac": float(cert.mean()),
                          "param_q": q(params[f][ks[cert]]), "test_margin_q": q(mw_k[ks[cert]]),
                          "test_fnr_q": q(fnr_k[ks[cert]]), "test_far_q": q(far_k[ks[cert]])}
    # ---- autocorrelation of daily validation loss at the pooled certified level
    k_ref = res["cert"]["0.1"]["pooled"]["gpd_op"]["khat"]
    if k_ref >= 0:
        ac = {}
        for y in [2020, 2021]:
            rr = np.array([pos[i] for i in idx[y]])
            ac[str(y)] = autocorr(Tval["gpd_op"]["L"][rr, k_ref])
        res["loss_autocorr_daily"] = ac
    # ---- seasonal stratification of test metrics + JJA-only certification
    res["season"] = {}
    seas = np.array([SEASON[m] for m in cfg["month"]])
    for f in ["emp_op", "gpd_op"]:
        k = res["cert"]["0.1"]["pooled"][f]["khat"]
        if k < 0:
            continue
        L = Ttest[f]["L"][:, k]; FARt = Ttest[f]["FAR"][:, k]; MWt = Ttest[f]["MW"][:, k]
        ss = seas[test_idx]
        res["season"][f] = {s: dict(n=int((ss == s).sum()), fnr=float(np.nanmean(L[ss == s])),
                                    far=float(FARt[ss == s].mean()), margin=float(MWt[ss == s].mean()))
                            for s in ["DJF", "MAM", "JJA", "SON"]}
    jja_rows = np.array([pos[i] for i in th["pooled"] if seas[i] == "JJA"], dtype=int)
    res["jja_only"] = {}
    for f in ["emp_op", "gpd_op"]:
        c = certify(Tval[f], jja_rows, 0.10)
        res["jja_only"][f] = {"n": c["n"], "n_min": n_min(0.10, DELTA), "khat": c["khat"],
                              "rhat_first": float(c["curve"][0]) if c["n"] else None}
    # ---- threshold-stability diagnostic and timing (main only)
    if main:
        res["threshold_diag"] = threshold_diagnostic(S_cal, B_cal)
        k = res["cert"]["0.1"]["pooled"]["gpd_op"]["khat"]
        if k >= 0:
            qk = fam["gpd"].q[:, k]
            t1 = time.perf_counter()
            for i in test_idx:
                U = cfg["F"][i] + OP(qk[cfg["bins"][i]]) * cfg["s"][i]
            res["ms_per_field_bound"] = 1000 * (time.perf_counter() - t1) / len(test_idx)
    # ---- IFS-ENS comparison on the ENS subset of test inits
    try:
        en = np.load(f"{D}/ens_{var}.npz"); e5 = np.load(f"{D}/era5_{var}.npz")
        la, lo = domain_index(e5["lat"], e5["lon"], DOMAINS[dom])
        li = list(en["leads"]).index(lead)
        eidx = {d: j for j, d in enumerate(en["time"].astype("datetime64[D]"))}
        sub = [i for i in test_idx if cfg["init"][i] in eidx]
        jj = [eidx[cfg["init"][i]] for i in sub]
        ensr = {}
        for nm in ["q90", "max"]:
            U = en[nm][jj, li][:, la][:, :, lo].astype(np.float64)
            Ls, FA, BA = [], [], []
            for t, i in enumerate(sub):
                m = cfg["mext"][i]
                if m.sum() > 0:
                    Ls.append(float(((cfg["Y"][i] > U[t]) & m).sum() / m.sum()))
                FA.append(float(((U[t] > cfg["q99"][i]) & ~m).sum() / (~m).sum()))
                BA.append(float((U[t] - cfg["mu"][i]).mean()))
            ensr["ens_" + nm] = dict(fnr=float(np.mean(Ls)), fnr_ci=stationary_bootstrap_ci(np.array(Ls)),
                                     far=float(np.mean(FA)), bound_anom=float(np.mean(BA)), n=len(sub))
        posT = {i: j for j, i in enumerate(test_idx)}
        rs = np.array([posT[i] for i in sub])
        for f in ["gpd_op", "emp_op", "pointwise"]:
            k = 0 if f == "pointwise" else res["cert"]["0.1"]["pooled"][f]["khat"]
            if k < 0:
                continue
            L = Ttest[f]["L"][rs, k]
            ensr[f] = dict(fnr=float(np.nanmean(L)), fnr_ci=stationary_bootstrap_ci(L[~np.isnan(L)]),
                           far=float(Ttest[f]["FAR"][rs, k].mean()), bound_anom=float(Ttest[f]["BA"][rs, k].mean()),
                           n=len(sub))
        res["ens_compare"] = ensr
    except Exception as ex:  # noqa
        res["ens_compare"] = {"error": repr(ex)}
    # ---- case study fields (main: largest-N_ext test day, plus fixed date for EU T2m)
    if main:
        cases = {}
        k_g = res["cert"]["0.1"]["pooled"]["gpd_op"]["khat"]; k_e = res["cert"]["0.1"]["pooled"]["emp_op"]["khat"]
        cand = [("max_next", test_idx[np.argmax(next_[test_idx])])]
        if var == "t2m" and dom == "EU":
            v = np.datetime64("2022-07-19")
            w = [i for i in test_idx if cfg["valid"][i] == v]
            if w:
                cand.append(("2022-07-19", w[0]))
        for name, i in cand:
            c = dict(valid=str(cfg["valid"][i]), init=str(cfg["init"][i]), Y=cfg["Y"][i], F=cfg["F"][i],
                     q99=cfg["q99"][i], n_ext=int(cfg["mext"][i].sum()),
                     U_pointwise=cfg["F"][i] + gens["pointwise"](i)[0])
            for nm, f, k in [("U_gpd_op", "gpd_op", k_g), ("U_emp_op", "emp_op", k_e), ("U_gpd_noop", "gpd_noop",
                              res["cert"]["0.1"]["pooled"]["gpd_noop"]["khat"])]:
                if k >= 0:
                    U = cfg["F"][i] + gens[f](i)[k]
                    m = cfg["mext"][i]
                    c[nm] = U
                    c[nm + "_stats"] = dict(Lcond=float(((cfg["Y"][i] > U) & m).sum() / max(1, m.sum())),
                                            ncc=n_components(U > cfg["q99"][i]))
            U = c["U_pointwise"]; m = cfg["mext"][i]
            c["U_pointwise_stats"] = dict(Lcond=float(((cfg["Y"][i] > U) & m).sum() / max(1, m.sum())),
                                          ncc=n_components(U > cfg["q99"][i]))
            cases[name] = c
        res["cases"] = cases
    log("config done", var, dom, lead, f"{time.time() - t0:.0f}s")
    return res


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    only = sys.argv[1:]
    for var in VARS:
        for dom in DOMAINS:
            for lead in LEADS:
                tag = f"{var}_{dom}_{lead}"
                if only and tag not in only:
                    continue
                fn = f"{OUT}/real_{tag}.json"
                if os.path.exists(fn):
                    continue
                main = (lead == 168)
                r = run_config(var, dom, lead, main)
                with open(fn, "w") as fh:
                    json.dump(jsonable(r), fh)
    log("ALL DONE")
