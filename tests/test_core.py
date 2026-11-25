"""Lightweight unit tests for stacp_v2/core.py (zero network / external data required)."""
import os
import sys
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "stacp_v2"))

from core import (
    BETA_GRID,
    BinnedFamily,
    SpatialOp,
    fixed_sequence_test,
    gpd_pwm,
    hb_pvalue,
    n_components,
    n_min,
    rhat_max_lattice,
)


def test_hoeffding_bentkus_nmin_and_lattice():
    assert n_min(0.10, 0.05) == 29
    assert n_min(0.05, 0.05) == 59
    assert n_min(0.01, 0.05) == 299
    assert hb_pvalue(0.0, 28, 0.10) > 0.05
    assert hb_pvalue(0.0, 29, 0.10) <= 0.05
    assert np.isnan(rhat_max_lattice(28, 0.10, 0.05))
    assert rhat_max_lattice(42, 0.10, 0.05) == 0.0
    assert np.isclose(rhat_max_lattice(85, 0.10, 0.05), 2 / 85)
    assert np.isclose(rhat_max_lattice(122, 0.10, 0.05), 5 / 122)


def test_gpd_pwm_and_binned_family_nestedness():
    rng = np.random.default_rng(2026)
    exc = rng.uniform(0.0, 1.0, size=2000) ** (-1.0 / 3.0) - 1.0
    xi, sig, _ = gpd_pwm(exc, xi_clip=(-0.20, 0.45))
    assert -0.20 <= xi <= 0.45
    assert sig > 0.0
    scores_by_bin = [np.abs(rng.normal(0.0, 1.0, size=400)) for _ in range(5)]
    fam = BinnedFamily(scores_by_bin, kind="gpd")
    assert fam.q.shape == (5, BETA_GRID.size)
    assert np.all(np.diff(fam.q, axis=1) <= 1e-9)


def test_spatial_operator_order_preserving_and_non_lowering():
    rng = np.random.default_rng(42)
    op = SpatialOp(r=1, sigma=1.0)
    Q1 = rng.uniform(1.0, 3.0, size=(16, 16))
    Q2 = Q1 + rng.uniform(0.0, 1.0, size=(16, 16))
    S1 = op(Q1)
    S2 = op(Q2)
    assert np.all(S2 >= S1 - 1e-12)
    assert np.all(S1 >= Q1 - 1e-12)


def test_n_components_4_connectivity():
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = True
    mask[0, 1] = True
    mask[2, 2] = True
    mask[3, 3] = True
    assert n_components(mask) == 3


def test_fixed_sequence_test_halting_and_infeasibility():
    risk_zero = np.zeros(10)
    khat_small, _ = fixed_sequence_test(risk_zero, n=20, alpha=0.10, delta=0.05)
    assert khat_small == -1
    risk_curve = np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.5, 0.5, 0.5, 0.5, 0.5])
    khat, _ = fixed_sequence_test(risk_curve, n=40, alpha=0.10, delta=0.05)
    assert khat == 3
