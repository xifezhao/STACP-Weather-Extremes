"""STACP v2 synthetic experiments (seeded, no injected constants).

Model on a 16x16 grid:
  latent signal A_t and error E~_t: spatially correlated Gaussian fields (exp. corr.,
  length 3 grid units), each AR(1) in time with coefficient phi;
  error marginal: symmetric GPD (xi = 1/3, sigma = 1) via Gaussian copula;
  Y_t = A_t + s * E_t,  forecast f_t = A_t,  s = 0.5 (single bin, no spatial op);
  extreme mask: Y > q99 of the stationary marginal of Y.
Since U = f + q s, a miss at an extreme point occurs iff E > q, so the true risk
R(q) depends on q only and is computed once from 50 000 independent fields.

E1  thinning / validity : phi in {0.2,...,0.9}, T_val in {365, 1095} daily fields,
    dt = ceil(2 tau), tau = -1/ln(phi); thinned vs unthinned; GPD candidates.
E2  tail family        : iid validation (n = 100 fields), n_cal in {5, 20, 100}
    calibration fields; empirical vs GPD (standard / narrow / wide xi clipping).
Outputs: /content/stacp_results/synthetic.json
"""
import json, math, os, sys, time
import numpy as np
from scipy.special import ndtr
from core import BETA_GRID, BinnedFamily, fixed_sequence_test, spatial_cluster_maxima

H = W = 16; ELL = 3.0; XI = 1.0 / 3.0; S = 0.5; ALPHA = 0.10; DELTA = 0.05
OUT = os.environ.get("STACP_OUT", "/content/stacp_results")
xy = np.stack(np.meshgrid(np.arange(H), np.arange(W), indexing="ij"), -1).reshape(-1, 2)
dist = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(-1))
LC = np.linalg.cholesky(np.exp(-dist / ELL) + 1e-9 * np.eye(H * W))


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def sym_gpd(u):
    p = np.where(u > 0.5, 2 * (1 - u), 2 * u)
    p = np.clip(p, 1e-300, 1.0)
    x = (p ** (-XI) - 1.0) / XI
    return np.where(u > 0.5, x, -x)


def ar_fields(T, phi, rng):
    eps = rng.standard_normal((T, H * W)) @ LC.T
    if phi == 0:
        return eps
    Z = np.empty_like(eps); Z[0] = eps[0]; c = math.sqrt(1 - phi * phi)
    for t in range(1, T):
        Z[t] = phi * Z[t - 1] + c * eps[t]
    return Z


def simulate(T, phi, rng):
    A = ar_fields(T, phi, rng)
    E = sym_gpd(ndtr(ar_fields(T, phi, rng)))
    return A + S * E, E


rng0 = np.random.default_rng(2026)
Ybig, Ebig = simulate(50000, 0.0, rng0)
QEXT = float(np.quantile(Ybig, 0.99))
# true risk R(q) = E[ frac of extreme points with E > q | N_ext > 0 ]
_vals, _w, _F = [], [], 0
for y, e in zip(Ybig, Ebig):
    m = y > QEXT
    if m.any():
        _F += 1; _vals.append(e[m]); _w.append(np.full(m.sum(), 1.0 / m.sum()))
_vals = np.concatenate(_vals); _w = np.concatenate(_w)
_o = np.argsort(_vals); _vals = _vals[_o]; _cw = np.r_[0, np.cumsum(_w[_o])]
P_EXT = _F / Ybig.shape[0]
del Ybig, Ebig


def true_risk(q):
    j = np.searchsorted(_vals, q, side="right")
    return float((_cw[-1] - _cw[j]) / _F)


def val_loss_matrix(Y, E, qs):
    rows = []
    for y, e in zip(Y, E):
        m = y > QEXT
        if not m.any():
            rows.append(None); continue
        v = np.sort(e[m])
        rows.append((v.size - np.searchsorted(v, qs, side="right")) / v.size)
    return rows


def families(n_cal, rng, which):
    Yc, Ec = simulate(n_cal, 0.0, rng)
    sc = np.maximum(Ec, 0.0)                     # = max(Y - f, 0)/s
    u = np.quantile(sc[sc > 0], 0.95)
    cm = np.concatenate([spatial_cluster_maxima(s.reshape(H, W), np.ones((H, W), bool), u) for s in sc])
    out = {}
    for w in which:
        if w == "emp":
            out[w] = BinnedFamily([sc.ravel()], kind="emp").q[0]
        elif w == "gpd":
            out[w] = BinnedFamily([sc.ravel()], [cm], kind="gpd").q[0]
        elif w == "gpd_narrow":
            out[w] = BinnedFamily([sc.ravel()], [cm], kind="gpd", xi_clip=(-0.10, 0.30)).q[0]
        elif w == "gpd_wide":
            out[w] = BinnedFamily([sc.ravel()], [cm], kind="gpd", xi_clip=(-0.30, 0.60)).q[0]
    out["_maxscore"] = float(sc.max())
    return out


def cert_from_rows(rows):
    L = np.array([r for r in rows if r is not None])
    if L.shape[0] == 0:
        return -1, 0, True
    curve = L.mean(0)
    mono = bool(np.all(np.diff(curve) >= -1e-12))
    k, _ = fixed_sequence_test(curve, L.shape[0], ALPHA, DELTA)
    return k, L.shape[0], mono


def wilson(k, n):
    if n == 0:
        return [None, None]
    z = 1.96; p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [c - h, c + h]


def summarize(recs):
    N = len(recs)
    cert = [r for r in recs if r["k"] >= 0]
    viol = [r for r in cert if r["R"] > ALPHA]
    return dict(n_rep=N, mean_n=float(np.mean([r["n"] for r in recs])),
                cert_rate=len(cert) / N,
                viol_uncond=len(viol) / N, viol_uncond_ci=wilson(len(viol), N),
                viol_given_cert=(len(viol) / len(cert)) if cert else None,
                viol_given_cert_ci=wilson(len(viol), len(cert)),
                mean_R_given_cert=float(np.mean([r["R"] for r in cert])) if cert else None,
                mean_margin_given_cert=float(np.mean([r["margin"] for r in cert])) if cert else None,
                monotone_all=bool(all(r["mono"] for r in recs)))


def E1(n_rep_by_T):
    out = {}
    for T_val, n_rep in n_rep_by_T.items():
        for phi in [0.2, 0.4, 0.6, 0.8, 0.9]:
            tau = -1.0 / math.log(phi); dt = int(math.ceil(2 * tau))
            rng = np.random.default_rng(1000 + int(phi * 100) + T_val)
            rec = {"thinned": [], "unthinned": []}
            for rep in range(n_rep):
                q = families(100, rng, ["gpd"])["gpd"]
                Y, E = simulate(T_val, phi, rng)
                rows = val_loss_matrix(Y, E, q)
                for mode, rr in [("thinned", rows[::dt]), ("unthinned", rows)]:
                    k, n, mono = cert_from_rows(rr)
                    rec[mode].append(dict(k=k, n=n, mono=mono, R=true_risk(q[k]) if k >= 0 else 0.0,
                                          margin=S * q[k] if k >= 0 else float("inf")))
            out[f"T{T_val}_phi{phi}"] = dict(T_val=T_val, phi=phi, tau=tau, dt=dt,
                                            thinned=summarize(rec["thinned"]),
                                            unthinned=summarize(rec["unthinned"]))
            log("E1", T_val, phi, out[f"T{T_val}_phi{phi}"]["thinned"]["cert_rate"],
                out[f"T{T_val}_phi{phi}"]["unthinned"]["viol_uncond"])
    return out


def E2(n_rep, n_val=100):
    out = {}
    fams = ["emp", "gpd", "gpd_narrow", "gpd_wide"]
    for n_cal in [5, 20, 100]:
        rng = np.random.default_rng(5000 + n_cal)
        rec = {f: [] for f in fams}; cap = {"emp_khat_at_max": 0, "emp_uncert_first": 0}
        for rep in range(n_rep):
            fq = families(n_cal, rng, fams)
            Y, E = simulate(n_val, 0.0, rng)
            for f in fams:
                rows = val_loss_matrix(Y, E, fq[f])
                k, n, mono = cert_from_rows(rows)
                rec[f].append(dict(k=k, n=n, mono=mono, R=true_risk(fq[f][k]) if k >= 0 else 0.0,
                                   margin=S * fq[f][k] if k >= 0 else float("inf")))
                if f == "emp":
                    if k < 0:
                        cap["emp_uncert_first"] += 1
                    elif abs(fq[f][k] - fq["_maxscore"]) < 1e-12:
                        cap["emp_khat_at_max"] += 1
        out[f"ncal{n_cal}"] = dict(n_cal=n_cal, n_val=n_val, **{f: summarize(rec[f]) for f in fams},
                                   emp_cap=dict(frac_uncertified=cap["emp_uncert_first"] / n_rep,
                                                frac_certified_at_sample_max=cap["emp_khat_at_max"] / n_rep))
        log("E2", n_cal, {f: out[f"ncal{n_cal}"][f]["cert_rate"] for f in fams})
    return out


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    quick = "--quick" in sys.argv
    res = dict(setup=dict(grid=[H, W], corr_length=ELL, xi_true=XI, s=S, alpha=ALPHA, delta=DELTA,
                          q_ext=QEXT, p_field_has_ext=P_EXT, beta_grid_size=int(BETA_GRID.size),
                          beta_grid_range=[float(BETA_GRID[0]), float(BETA_GRID[-1])],
                          true_risk_at_q0=true_risk(0.0)))
    res["E1"] = E1({365: 20, 1095: 10} if quick else {365: 1000, 1095: 500})
    res["E2"] = E2(20 if quick else 1000)
    with open(f"{OUT}/synthetic{'_quick' if quick else ''}.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    log("ALL DONE")
