"""STACP v2 core: candidate families, spatial operator, field-level loss,
Hoeffding-Bentkus p-values and fixed-sequence testing.

Design rules (audit-driven):
  * No result constants anywhere.  Every reported number is computed.
  * Candidate families are nested by construction (margins non-increasing along
    the grid, which is ordered conservative -> sharp); this is *checked* at run
    time (`assert_nested`).  Consequently every field-level loss and the empirical
    risk are monotone along the grid.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import ndimage, stats

# ----------------------------------------------------------------------------
# Candidate grid (fixed before validation).  Ordered conservative -> sharp.
#   beta_k = 1 - 10^{-t_k},  t from 6.0 down to 0.3 in steps of 0.025.
# ----------------------------------------------------------------------------
T_GRID = np.round(np.arange(6.0, 0.2999, -0.025), 4)
BETA_GRID = 1.0 - 10.0 ** (-T_GRID)


# ----------------------------------------------------------------------------
# Hoeffding-Bentkus p-value (Bates et al. 2021; Angelopoulos et al. 2025)
# ----------------------------------------------------------------------------
def _h1(a: float, b: float) -> float:
    if a <= 0.0:
        return -math.log(1.0 - b)
    return a * math.log(a / b) + (1.0 - a) * math.log((1.0 - a) / (1.0 - b))


def hb_pvalue(rhat: float, n: int, alpha: float) -> float:
    if n <= 0 or rhat >= alpha:
        return 1.0
    p_h = math.exp(-n * _h1(rhat, alpha))
    k = int(math.ceil(n * rhat - 1e-12))
    p_b = math.e * stats.binom.cdf(k, n, alpha)
    return float(min(1.0, p_h, p_b))


def rhat_max_lattice(n: int, alpha: float, delta: float) -> float:
    """Largest certifiable empirical risk on the lattice {k/n}; nan if none."""
    best = float("nan")
    for k in range(0, n + 1):
        if hb_pvalue(k / n, n, alpha) <= delta:
            best = k / n
        else:
            break
    return best


def n_min(alpha: float, delta: float) -> int:
    return int(math.ceil(math.log(delta) / math.log(1.0 - alpha)))


def fixed_sequence_test(risk_curve: np.ndarray, n: int, alpha: float, delta: float):
    """risk_curve[k] = empirical risk of candidate k (k=0 most conservative).
    Returns (khat, pvals).  khat = -1 means the trivial candidate beta_0 = +inf."""
    khat, pvals = -1, []
    for k, r in enumerate(risk_curve):
        p = hb_pvalue(float(r), n, alpha)
        pvals.append(p)
        if p <= delta:
            khat = k
        else:
            break
    return khat, np.array(pvals)


# ----------------------------------------------------------------------------
# GPD via probability-weighted moments with mean-coupled scale
# ----------------------------------------------------------------------------
def gpd_pwm(excess: np.ndarray, xi_clip=(-0.20, 0.45)):
    x = np.sort(np.asarray(excess, dtype=float))
    n = x.size
    a0 = x.mean()
    a1 = np.sum((n - np.arange(1, n + 1)) / (n - 1) * x) / n
    xi_raw = (a0 - 4 * a1) / (a0 - 2 * a1)
    xi = float(np.clip(xi_raw, *xi_clip))
    sigma = float(a0 * (1.0 - xi))
    return xi, sigma, float(xi_raw)


def _emp_quantile(sorted_scores: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Empirical 'higher' quantile; equals the sample maximum for beta > 1-1/N."""
    n = sorted_scores.size
    idx = np.clip(np.ceil(beta * n).astype(int) - 1, 0, n - 1)
    return sorted_scores[idx]


class BinnedFamily:
    """q[m, k]: score quantile of bin m for candidate k.  kind in {'emp','gpd'}."""

    def __init__(self, scores_by_bin, cluster_max_by_bin=None, kind="gpd",
                 u_prob=0.95, xi_clip=(-0.20, 0.45), min_exc=30):
        self.kind = kind
        self.info = []
        M = len(scores_by_bin)
        allsc = np.sort(np.concatenate(scores_by_bin))
        q = np.zeros((M, BETA_GRID.size))
        for m in range(M):
            s = np.sort(scores_by_bin[m]) if len(scores_by_bin[m]) > 50 else allsc
            q_emp = _emp_quantile(s, BETA_GRID)
            if kind == "emp":
                q[m] = q_emp
                self.info.append({"bin": m, "N": int(s.size), "max": float(s[-1])})
                continue
            pos = s[s > 0]
            u = float(np.quantile(pos, u_prob))
            zeta = float(np.mean(s > u))
            cm = cluster_max_by_bin[m] if cluster_max_by_bin is not None else s
            exc = cm[cm > u] - u
            pooled = False
            if exc.size < min_exc:  # fall back to all bins pooled
                cma = np.concatenate(cluster_max_by_bin) if cluster_max_by_bin is not None else allsc
                exc = cma[cma > u] - u
                pooled = True
            xi, sig, xi_raw = gpd_pwm(exc, xi_clip)
            tail = BETA_GRID >= 1.0 - zeta
            qq = np.minimum(q_emp, u)
            r = (1.0 - BETA_GRID[tail]) / zeta
            if abs(xi) < 1e-3:
                qq[tail] = u + sig * np.log(1.0 / r)
            else:
                qq[tail] = u + sig / xi * (r ** (-xi) - 1.0)
            q[m] = qq
            self.info.append({"bin": m, "N": int(s.size), "u": u, "zeta": zeta,
                              "n_exc": int(exc.size), "xi": xi, "xi_raw": xi_raw,
                              "sigma": sig, "pooled": pooled, "max": float(s[-1])})
        # enforce/verify nestedness along the grid (q non-increasing in k)
        assert np.all(np.diff(q, axis=1) <= 1e-9), "candidate family not nested"
        self.q = q


# ----------------------------------------------------------------------------
# Spatial operator: flat-disk dilation (radius r) + truncated Gaussian smoothing
# with boundary renormalisation.  Kernel support within B_r => no lowering.
# ----------------------------------------------------------------------------
def disk(r: int) -> np.ndarray:
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    return (x * x + y * y) <= r * r


def gauss_kernel(r: int, sigma: float) -> np.ndarray:
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    k = np.exp(-(x * x + y * y) / (2 * sigma * sigma)) * disk(r)
    return k / k.sum()


class SpatialOp:
    def __init__(self, r=1, sigma=1.0):
        self.r, self.sigma = r, sigma
        self.fp = disk(r)
        self.ker = gauss_kernel(r, sigma)

    def __call__(self, Q: np.ndarray) -> np.ndarray:
        """Q: (..., H, W).  Applied over the last two axes."""
        lead = Q.ndim - 2
        fp = self.fp.reshape((1,) * lead + self.fp.shape)
        ker = self.ker.reshape((1,) * lead + self.ker.shape)
        Qd = ndimage.maximum_filter(Q, footprint=fp, mode="nearest")
        num = ndimage.convolve(Qd, ker, mode="constant", cval=0.0)
        den = ndimage.convolve(np.ones(Q.shape[-2:]), self.ker, mode="constant", cval=0.0)
        return num / den


# ----------------------------------------------------------------------------
# Margin generators  margin_k(x) = U_k(x) - f(x)   (physical units)
# ----------------------------------------------------------------------------
def margins_binned(fam: BinnedFamily, bin_map, scale, op: SpatialOp | None):
    Q = fam.q[:, :][bin_map]            # (H, W, K)
    Q = np.moveaxis(Q, -1, 0)           # (K, H, W)
    if op is not None:
        Q = op(Q)
    return Q * scale[None]


def margins_std_offset(lams, scale):
    return lams[:, None, None] * scale[None]


def margins_phys_offset(cs, shape):
    return np.broadcast_to(cs[:, None, None], (cs.size,) + shape)


# ----------------------------------------------------------------------------
# Losses and metrics
# ----------------------------------------------------------------------------
def field_losses(Y, f, margins, mext):
    """Conditional miss fraction for every candidate. returns (K,) or None."""
    n_ext = mext.sum()
    if n_ext == 0:
        return None
    miss = (Y[None] > f[None] + margins) & mext[None]
    return miss.reshape(margins.shape[0], -1).sum(1) / n_ext


def field_far(f, margins, q99, mext):
    non = ~mext
    if non.sum() == 0:
        return np.zeros(margins.shape[0])
    alarm = (f[None] + margins > q99[None]) & non[None]
    return alarm.reshape(margins.shape[0], -1).sum(1) / non.sum()


def n_components(alarm2d: np.ndarray) -> int:
    return int(ndimage.label(alarm2d)[1])


def stationary_bootstrap_ci(x, B=1000, mean_block=5, seed=0, level=0.95):
    x = np.asarray(x, float)
    n = x.size
    if n == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    means = np.empty(B)
    for b in range(B):
        idx = np.empty(n, int)
        i = rng.integers(n)
        for t in range(n):
            if t > 0 and rng.random() < p:
                i = rng.integers(n)
            elif t > 0:
                i = (i + 1) % n
            idx[t] = i
        means[b] = x[idx].mean()
    a = (1 - level) / 2
    return [float(np.quantile(means, a)), float(np.quantile(means, 1 - a))]


def stationary_bootstrap_indices(n, B, mean_block, seed):
    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    out = np.empty((B, n), int)
    for b in range(B):
        i = rng.integers(n)
        for t in range(n):
            if t > 0 and rng.random() < p:
                i = rng.integers(n)
            elif t > 0:
                i = (i + 1) % n
            out[b, t] = i
    return out


def spatial_cluster_maxima(score2d, mask_bin2d, u_low):
    """Declustering in space: maxima of 4-connected exceedance clusters (> u_low)."""
    exc = (score2d > u_low) & mask_bin2d
    lab, nlab = ndimage.label(exc)
    if nlab == 0:
        return np.empty(0)
    return np.asarray(ndimage.maximum(score2d, lab, index=np.arange(1, nlab + 1)))
