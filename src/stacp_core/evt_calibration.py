# -*- coding: utf-8 -*-
r"""
Declustered EVT Calibration with Monotonic Splicing (Algorithm 1)
==============================================================================

This module implements the traditional statistical anchor of the STACP framework.
It parameterizes the extreme upper tail of non-conformity scores using Extreme 
Value Theory (EVT), mapping noisy empirical residuals to a Generalized Pareto 
Distribution (GPD) to generate a smoothed, monotonically increasing extrapolation 
function for downstream Learn-then-Test (LTT) certification.

Academic Alignment & Disclaimers:
------------------------------------------------------------------------------
- Ref Section 3.2: EVT is NOT utilized to assert an absolute physical truth 
  regarding the atmospheric tail. It serves strictly as a mathematically structured 
  extrapolation mechanism to prevent sample variance collapse at the distribution edge.
- Spatio-temporal Declustering: Extremes naturally cluster. We employ an N-dimensional 
  maximum filter to extract independent cluster peaks, satisfying GPD assumptions.
- Mean-Constraint Consistency: If the PWM-estimated shape parameter (\hat{\xi}) 
  is clipped to prevent infinite variance, the scale parameter (\hat{\sigma}) is 
  recalculated to preserve the expected value of the empirical exceedances.

Author: Anonymous (Prepared for Double-Blind Peer Review)
"""

import logging
from typing import Dict, Any, Callable, Union
import numpy as np
import scipy.ndimage as ndimage

logger = logging.getLogger(__name__)


class SplicedTailFunction:
    """
    A callable object representing the piecewise continuous quantile extrapolation 
    function q_m(beta). Formulated to guarantee monotonic non-decreasing behavior.
    """

    def __init__(
        self,
        raw_scores: np.ndarray,
        u_m: float,
        zeta_u: float,
        xi: float,
        sigma: float,
        xi_gumbel_tol: float = 1e-3,
        fallback_only: bool = False
    ):
        r"""
        Initializes the piecewise function.
        
        Args:
            raw_scores: All empirical scores (including zeros) for the empirical branch.
            u_m: The empirical threshold marking the start of the EVT tail.
            zeta_u: Cluster peak occurrence rate (N_u / n_ext).
            xi: Shape parameter (\hat{\xi}_m).
            sigma: Scale parameter (\hat{\sigma}_m).
            xi_gumbel_tol: Tolerance to switch to the Gumbel (exponential) limit.
            fallback_only: If True, EVT calibration failed; strictly use empirical quantiles.
        """
        # Exclude NaNs from empirical body
        self.raw_scores = raw_scores[~np.isnan(raw_scores)]
        self.u_m = u_m
        self.zeta_u = zeta_u
        self.xi = xi
        self.sigma = sigma
        self.xi_gumbel_tol = xi_gumbel_tol
        self.fallback_only = fallback_only
        self.n_ext = len(self.raw_scores)

    def _empirical_quantile(self, beta: np.ndarray) -> np.ndarray:
        """Computes the empirical quantile q_m^{emp}(beta) for the dense data body."""
        # Using numpy's robust quantile estimator
        q_emp = np.quantile(self.raw_scores, beta, method='linear')
        return np.maximum(q_emp, 0.0)

    def __call__(self, beta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Evaluates the piecewise extrapolation function at target level(s) beta.
        Ref: Equation (6) of the manuscript.
        """
        beta = np.asarray(beta, dtype=float)
        scalar_input = beta.ndim == 0
        if scalar_input:
            beta = beta[np.newaxis]
            
        result = np.zeros_like(beta)
        
        if self.fallback_only:
            # Complete failure of EVT (N_u < N_min or Numerical safeguard tripped)
            result = self._empirical_quantile(beta)
            return result[0] if scalar_input else result

        # Boolean masks for the piecewise branches
        mask_tail = beta >= (1.0 - self.zeta_u)
        mask_body = ~mask_tail

        # 1. Dense Data Body (Empirical Branch)
        if np.any(mask_body):
            # q_m(beta) = min(max(q_emp(beta), 0), u_m)
            # This guarantees a mathematically continuous merge at beta = 1 - zeta_u
            q_emp = self._empirical_quantile(beta[mask_body])
            result[mask_body] = np.minimum(q_emp, self.u_m)

        # 2. Sparse Data Tail (EVT / GPD Branch)
        if np.any(mask_tail):
            b_tail = beta[mask_tail]
            
            # Prevent division by zero at absolute 1.0 (though LTT FST rarely reaches 1.0)
            b_tail = np.clip(b_tail, 0.0, 1.0 - 1e-12)
            
            if abs(self.xi) < self.xi_gumbel_tol:
                # Gumbel limit (Exponential tail)
                extrapolation = self.u_m + self.sigma * np.log(self.zeta_u / (1.0 - b_tail))
            else:
                # Fréchet / Weibull domain
                ratio = (1.0 - b_tail) / self.zeta_u
                extrapolation = self.u_m + (self.sigma / self.xi) * (np.power(ratio, -self.xi) - 1.0)
                
            result[mask_tail] = extrapolation

        return result[0] if scalar_input else result


class ExponentialTailFunction:
    """
    Non-EVT Smooth Baseline: Exponential tail extrapolation spliced onto the empirical body.
    Formula: q_exp(beta) = u_m + sigma_exp * ln(zeta_u / (1 - beta)) for beta >= 1 - zeta_u.
    Serves as an empirical baseline to verify whether heavy-tailed GPD parameterization 
    provides distinct efficiency gains over a simple exponential decay assumption.
    """
    def __init__(
        self, 
        raw_scores: np.ndarray, 
        u_m: float, 
        zeta_u: float, 
        sigma_exp: float
    ):
        self.raw_scores = raw_scores[~np.isnan(raw_scores)]
        self.u_m = float(u_m)
        self.zeta_u = max(float(zeta_u), 1e-6)
        self.sigma_exp = max(float(sigma_exp), 1e-4)

    def _empirical_quantile(self, beta: np.ndarray) -> np.ndarray:
        q_emp = np.quantile(self.raw_scores, beta, method='linear')
        return np.maximum(q_emp, 0.0)

    def __call__(self, beta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        beta = np.asarray(beta, dtype=float)
        scalar_input = beta.ndim == 0
        if scalar_input:
            beta = beta[np.newaxis]
            
        result = np.zeros_like(beta)
        mask_tail = beta >= (1.0 - self.zeta_u)
        mask_body = ~mask_tail
        
        if np.any(mask_body):
            q_emp = self._empirical_quantile(beta[mask_body])
            result[mask_body] = np.minimum(q_emp, self.u_m)
            
        if np.any(mask_tail):
            b_tail = np.clip(beta[mask_tail], 0.0, 1.0 - 1e-12)
            result[mask_tail] = self.u_m + self.sigma_exp * np.log(self.zeta_u / (1.0 - b_tail))
            
        return result[0] if scalar_input else result


class ConstantOffsetFunction:
    """
    Standard 1D Conformal Risk Control (CRC) Baseline:
    A pure scalar offset lambda applied uniformly across the domain without 
    quantile inversion or EVT modeling: U(lambda) = q_hi + lambda * s(X).
    """
    def __init__(self, lambda_scale: float = 1.0):
        self.lambda_scale = lambda_scale

    def __call__(self, beta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        # Maps the parameter value directly as the scalar multiplier
        return beta


class EVTCalibrator:
    """
    Executes Algorithm 1: Declustered EVT Calibration with Monotonic Splicing.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the EVT Calibrator with strict a priori parameters.
        """
        evt_cfg = config.get('evt_calibration', {})
        self.p_u = evt_cfg.get('p_u', 0.95)
        self.N_min = evt_cfg.get('N_min', 50)
        
        bounds = evt_cfg.get('xi_bounds', {})
        self.xi_min = bounds.get('min', -0.3)
        self.xi_max = bounds.get('max', 0.5)
        
        self.xi_gumbel_tol = evt_cfg.get('xi_gumbel_tol', 1e-3)
        self.decluster_window = evt_cfg.get('decluster_window', 3)  # default 3x3x3 spatio-temporal window

        logger.info(
            f"EVTCalibrator initialized. Threshold p_u={self.p_u}, N_min={self.N_min}, "
            f"Shape constraints xi in [{self.xi_min}, {self.xi_max}]."
        )

    def _spatiotemporal_declustering(
        self, 
        scores: np.ndarray, 
        u_m: float
    ) -> np.ndarray:
        """
        Extracts independent cluster peaks from N-dimensional score arrays.
        Uses a maximum filter to locate local maxima, satisfying EVT independence assumptions.
        """
        # Create a boolean mask of exceedances
        is_exceedance = scores > u_m
        
        # Apply N-dimensional maximum filter
        # Size footprint determines the physical decorrelation radius
        footprint = np.ones((self.decluster_window,) * scores.ndim)
        local_max = ndimage.maximum_filter(scores, footprint=footprint, mode='constant', cval=0.0)
        
        # A point is an independent peak if it is greater than the threshold 
        # AND it is the strict local maximum within its spatio-temporal window.
        is_peak = is_exceedance & (scores == local_max)
        
        # Extract the peak values
        cluster_peaks = scores[is_peak]
        return cluster_peaks

    def calibrate(self, raw_scores: np.ndarray) -> SplicedTailFunction:
        """
        Executes the calibration pipeline on the provided raw non-conformity scores.
        
        Args:
            raw_scores: N-dimensional array of scores (e.g., Time x Lat x Lon).
                        Expected to be max(Y - q_hi, 0) / s(X).
                        
        Returns:
            SplicedTailFunction: A callable mapping beta -> extrapolated score limit.
        """
        # Flatten for purely statistical operations
        flat_scores = raw_scores.flatten()
        flat_scores = flat_scores[~np.isnan(flat_scores)]
        n_ext = len(flat_scores)
        
        # Filter strictly positive scores
        positive_scores = flat_scores[flat_scores > 0.0]
        
        if len(positive_scores) < self.N_min:
            logger.warning(f"Fallback triggered: Positive scores ({len(positive_scores)}) < N_min ({self.N_min}).")
            return SplicedTailFunction(flat_scores, 0, 0, 0, 0, fallback_only=True)
            
        # Set threshold u_m
        u_m = np.quantile(positive_scores, self.p_u)
        
        # Spatio-temporal declustering to extract independent peaks
        cluster_peaks = self._spatiotemporal_declustering(raw_scores, u_m)
        N_u = len(cluster_peaks)
        
        if N_u < self.N_min:
            logger.warning(f"Fallback triggered: Independent peaks N_u ({N_u}) < N_min ({self.N_min}).")
            return SplicedTailFunction(flat_scores, 0, 0, 0, 0, fallback_only=True)
            
        # Marginal exceedance rate across all scores (ensures continuous merge at u_m without flat plateau)
        n_exceed_all = np.sum(flat_scores > u_m)
        zeta_u = float(n_exceed_all) / n_ext
        
        # Exceedances above threshold from independent cluster peaks (used for PWM fitting)
        R = np.sort(cluster_peaks - u_m)
        
        # ----------------------------------------------------------------------
        # Probability Weighted Moments (PWM) Estimation
        # Ref Section 3.2: Robust estimation for small extreme samples
        # ----------------------------------------------------------------------
        # Create weights w_k = (N_u - 1 - k) / (N_u - 1) for 0-based indexing
        k = np.arange(N_u)
        w_k = (N_u - 1 - k) / (N_u - 1)
        
        a_0 = np.mean(R)
        a_1 = np.mean(R * w_k)
        
        # Numerical safeguard against degenerate distributions
        if (a_0 - 2 * a_1) <= 0:
            logger.warning("Numerical safeguard tripped: (a_0 - 2a_1) <= 0. Falling back to empirical.")
            return SplicedTailFunction(flat_scores, 0, 0, 0, 0, fallback_only=True)
            
        xi_raw = 2.0 - (a_0 / (a_0 - 2 * a_1))
        
        # Clip shape parameter to physical constraints
        xi_m = np.clip(xi_raw, self.xi_min, self.xi_max)
        
        # Diagnostic tracking
        if xi_raw != xi_m:
            logger.debug(f"Shape parameter clipped: raw {xi_raw:.4f} -> constrained {xi_m:.4f}")
            
        # Mean-constraint consistency: recalculate sigma using constrained xi
        sigma_m = a_0 * (1.0 - xi_m)
        
        if sigma_m <= 0:
            logger.warning(f"Numerical safeguard tripped: sigma_m ({sigma_m:.4f}) <= 0. Falling back.")
            return SplicedTailFunction(flat_scores, 0, 0, 0, 0, fallback_only=True)
            
        logger.info(
            f"[EVT Calibration Successful] N_u={N_u} peaks extracted. "
            f"Fitted GPD: xi={xi_m:.4f}, sigma={sigma_m:.4f}, u_m={u_m:.4f}."
        )
        
        # Return the parameterized monotonic extrapolation function
        return SplicedTailFunction(
            raw_scores=flat_scores,
            u_m=u_m,
            zeta_u=zeta_u,
            xi=xi_m,
            sigma=sigma_m,
            xi_gumbel_tol=self.xi_gumbel_tol,
            fallback_only=False
        )
