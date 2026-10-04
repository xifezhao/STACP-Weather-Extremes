# -*- coding: utf-8 -*-
r"""
Conditional Spatial LTT Certification (Algorithm 2)
==============================================================================

This module implements the Machine Learning anchor (Module 2) of the STACP 
framework. It systematically certifies the parameterized candidate bounding fields 
against a predefined risk tolerance using the Learn-then-Test (LTT) procedure.

Academic Alignment & Disclaimers:
------------------------------------------------------------------------------
- Ref Section 3.1: To prevent statistical risk dilution from quiescent weather 
  states, the False Negative Rate (FNR) is evaluated STRICTLY on fields where 
  extreme events are physically present (N_ext > 0).
- Ref Section 3.3: Spatial smoothing is embedded directly within the FST loop 
  to preserve certification validity. Post-hoc smoothing invalidates the bounds.
- Ref Section 4.1: If the available independent sample size (n_val) is too small, 
  certification mathematically resolves to a negative max allowable risk, and 
  the procedure will correctly halt and return a "Cannot Certify" flag.

Author: Anonymous (Prepared for Double-Blind Peer Review)
"""

import logging
from typing import Callable, Dict, Any, Tuple, Optional, List
import numpy as np
import scipy.stats as stats

logger = logging.getLogger(__name__)


def hoeffding_bentkus_pvalue(losses: np.ndarray, alpha_risk: float) -> float:
    """
    Computes the valid upper bound p-value for the null hypothesis 
    H_0: E[L] > alpha_risk, where L is a continuous random variable in [0, 1].
    
    Combines the Hoeffding (Bennett-type) bound and the Bentkus bound to achieve 
    the tightest possible finite-sample certification.
    
    Args:
        losses: 1D numpy array of conditional field-level losses (L_cond).
        alpha_risk: The target risk tolerance (\alpha_{risk}).
        
    Returns:
        float: The calculated HB p-value.
    """
    n = len(losses)
    if n == 0:
        return 1.0
        
    r_hat = np.mean(losses)
    
    # If the empirical risk exceeds the tolerance, we cannot reject H_0.
    if r_hat >= alpha_risk:
        return 1.0

    # 1. Hoeffding (Bennett-type form for bounded [0,1] variables)
    if r_hat <= 0.0:
        h1_bound = (1.0 - alpha_risk) ** n
    else:
        try:
            kl_div = (r_hat * np.log(r_hat / alpha_risk) + 
                      (1.0 - r_hat) * np.log((1.0 - r_hat) / (1.0 - alpha_risk)))
            h1_bound = np.exp(-n * kl_div)
        except Exception:
            h1_bound = 1.0

    # 2. Bentkus Bound (Theorem 1.1 in Bentkus 2004; Bates et al. 2021)
    # Uses ceil(n * r_hat) for general continuous bounded [0, 1] losses
    k = int(np.ceil(n * r_hat))
    bentkus_bound = np.e * stats.binom.cdf(k, n, alpha_risk)
    
    # The valid p-value is the minimum of the two bounds (capped at 1.0)
    p_val = min(h1_bound, bentkus_bound, 1.0)
    
    return float(p_val)


class SpatialLTTCertifier:
    """
    Executes Algorithm 2: Conditional Spatial LTT Certification via Fixed 
    Sequence Testing (FST).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the Certifier with strictly frozen a priori parameters.
        """
        risk_cfg = config.get('risk_control', {})
        self.alpha_risk = risk_cfg.get('alpha_risk', 0.10)
        self.delta = risk_cfg.get('delta', 0.05)
        
        # Candidate sequence definition
        seq_cfg = config.get('candidate_sequence', {})
        beta_start = seq_cfg.get('beta_start', 0.9990)
        beta_end = seq_cfg.get('beta_end', 0.9000)
        beta_steps = seq_cfg.get('beta_steps', 100)
        
        # Pre-generate descending beta grid
        self.beta_grid = np.linspace(beta_start, beta_end, beta_steps)
        
        # Enforce descending order (strict requirement for FST monotonicity)
        if self.beta_grid[0] < self.beta_grid[-1]:
            self.beta_grid = self.beta_grid[::-1]

        logger.info(
            f"SpatialLTTCertifier initialized.\n"
            f"  - Target Risk Tolerance (alpha) : {self.alpha_risk}\n"
            f"  - Nominal Significance (delta)  : {self.delta}\n"
            f"  - FST Sequence                  : {len(self.beta_grid)} steps ({self.beta_grid[0]:.4f} -> {self.beta_grid[-1]:.4f})"
        )

    def _feasibility_precheck(self, n_val: int):
        """
        Ref Section 5.1: Sample Size Bottleneck.
        Calculates the theoretical max allowable empirical risk (r_hat_max) 
        before entering the expensive spatial computation loop.
        """
        max_k = -1
        for k in range(n_val + 1):
            r_hat = k / n_val
            pval = hoeffding_bentkus_pvalue(np.full(n_val, r_hat), self.alpha_risk)
            if pval <= self.delta:
                max_k = k
            else:
                break
                
        r_hat_max = (max_k / n_val) if max_k != -1 else -1.0
        
        if r_hat_max < 0:
            logger.error(
                f"\n[CERTIFICATION IMPOSSIBLE]\n"
                f"Effective sample size n_val = {n_val} is too small to mathematically "
                f"guarantee alpha={self.alpha_risk} at delta={self.delta}.\n"
                f"Even an empirical risk of 0.0 will yield a p-value > {self.delta}. "
                f"Please reduce alpha_risk, increase thinning duration, or use a larger dataset."
            )
        else:
            logger.info(
                f"[FEASIBILITY CHECK] n_val = {n_val}. To achieve certification, "
                f"the empirical expected FNR must be <= {r_hat_max:.4f}."
            )
        
        return r_hat_max

    def certify(
        self,
        y_true: np.ndarray,
        q_hi_base: np.ndarray,
        error_scale: np.ndarray,
        m_ext: np.ndarray,
        q_base_generator: Callable[[float], np.ndarray],
        spatial_operator: Optional[Callable[[np.ndarray], np.ndarray]] = None,
        candidate_grid: Optional[np.ndarray] = None
    ) -> Tuple[Optional[float], List[Dict[str, float]]]:
        r"""
        Executes the Fixed Sequence Testing loop over the validation fields.
        
        Args:
            y_true: Ground truth observations (Time, Lat, Lon).
            q_hi_base: AI deterministic upper quantile prediction (Time, Lat, Lon).
            error_scale: Local error scaling field s(X) (Time, Lat, Lon).
            m_ext: Boolean mask of physical extremes M_ext(X) (Time, Lat, Lon).
            q_base_generator: Function f(beta) -> Q_base (un-smoothed EVT score field).
            spatial_operator: Optional order-preserving spatial transformation instance.
            candidate_grid: Optional custom parameter sequence (e.g. for scalar shift lambda).
            
        Returns:
            Tuple containing the certified \hat{\beta} (or None if failed), 
            and the recorded trajectory of (beta, empirical_risk, p_value).
        """
        # 1. Isolate the Conditional Set (N_ext > 0) to prevent Risk Dilution
        n_ext_k = np.sum(m_ext, axis=(1, 2))
        valid_mask = n_ext_k > 0
        n_val = int(np.sum(valid_mask))
        
        logger.info(f"Total fields: {len(y_true)}. Valid fields with active extremes (n_val): {n_val}.")
        
        n_min_hb = int(np.ceil(np.log(self.delta) / np.log(1.0 - self.alpha_risk)))
        r_hat_max = self._feasibility_precheck(n_val)
        
        self.last_info = {
            "n_val": n_val,
            "n_min": n_min_hb,
            "r_hat_max": r_hat_max,
            "min_empirical_risk": None
        }
        
        if r_hat_max < 0:
            return None, []

        # Filter the arrays strictly to the valid conditional set
        y_true_v = y_true[valid_mask]
        q_hi_base_v = q_hi_base[valid_mask]
        error_scale_v = error_scale[valid_mask]
        m_ext_v = m_ext[valid_mask]
        n_ext_k_v = n_ext_k[valid_mask]

        certified_beta = None
        trajectory = []
        grid = candidate_grid if candidate_grid is not None else self.beta_grid

        logger.info(f"Initiating Fixed Sequence Testing (FST) over {len(grid)} steps...")
        min_emp_risk = float("inf")
        
        # 2. Fixed Sequence Testing (FST)
        for beta in grid:
            
            # Step A: Query parameterization (Generates Q_base)
            q_val = q_base_generator(beta)
            if np.isscalar(q_val) or (isinstance(q_val, np.ndarray) and q_val.ndim == 0):
                q_base_v = np.full_like(q_hi_base_v, float(q_val))
            else:
                q_base_v = q_val[valid_mask]
            
            # Step B: Embed spatial smoothing INSIDE the certification loop (if requested)
            if spatial_operator is not None:
                q_smooth_v = spatial_operator(q_base_v)
            else:
                q_smooth_v = q_base_v
            
            # Step C: Construct the final spatial bounding polygons
            u_smooth_v = q_hi_base_v + q_smooth_v * error_scale_v
            
            # Step D: Evaluate Conditional Field-Level False Negative Rate (FNR)
            # missed = 1 where (True > UpperBound) AND (Extreme Mask == 1)
            missed = (y_true_v > u_smooth_v) & (m_ext_v == 1)
            
            # L_k(beta) array for all valid fields
            l_cond_k = np.sum(missed, axis=(1, 2)) / n_ext_k_v
            
            # Step E: Compute Hoeffding-Bentkus p-value
            empirical_risk = float(np.mean(l_cond_k))
            if empirical_risk < min_emp_risk:
                min_emp_risk = empirical_risk
                
            p_val = hoeffding_bentkus_pvalue(l_cond_k, self.alpha_risk)
            
            trajectory.append({
                "beta": float(beta), 
                "empirical_risk": empirical_risk, 
                "p_value": p_val
            })
            
            logger.debug(f"Testing beta={beta:.4f} | Emp. Risk={empirical_risk:.4f} | p-value={p_val:.4e}")

            # Step F: Halt on first uncertified level
            if p_val <= self.delta:
                certified_beta = float(beta)
                # FST rule: Successfully rejected H_0, continue to more aggressive beta
                continue
            else:
                logger.info(
                    f"FST Halted at beta={beta:.4f}. "
                    f"p-value ({p_val:.4f}) > delta ({self.delta})."
                )
                break

        self.last_info["min_empirical_risk"] = min_emp_risk
        if certified_beta is None:
            logger.error("\n[CERTIFICATION FAILED] The algorithm could not certify even the most conservative beta.")
        else:
            logger.info(f"\n[CERTIFICATION SUCCESS] Optimal certified threshold: hat_beta = {certified_beta:.4f}")
            
        return certified_beta, trajectory
