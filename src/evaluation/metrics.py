# -*- coding: utf-8 -*-
"""
Rigorous Evaluation Metrics for Spatial Conformal Risk Control
==============================================================================

This module implements the evaluation metrics utilized in Section 5.3 and 6.1 
of the STACP manuscript. It is designed to expose the "Recall-Precision Trade-off",
ensuring that spatial False Negative Rate (FNR) bounds are achieved efficiently 
without defaulting to uninformative, infinitely wide intervals.

Academic Alignment & Disclaimers:
------------------------------------------------------------------------------
- Ref Section 5.3 (Sharpness & Precision): FNR bounds can be trivially satisfied 
  by predicting infinite upper bounds. We mandate the joint reporting of Mean 
  Prediction Interval Width (MPIW) and Winkler Score to mathematically penalize 
  under-confidence (excessive width).
- Ref Section 6.1 (Recall-Precision Trade-off): Spatial FAR assesses the 
  operational cost of over-warning. It is defined meteorologically as the False 
  Alarm Ratio: False Positives / (False Positives + True Positives).
- All metrics are computed as vectorized spatial operations over valid grids.

Author: Anonymous (Prepared for Double-Blind Peer Review)
"""

import logging
from typing import Union
import numpy as np

logger = logging.getLogger(__name__)


def calculate_conditional_fnr(
    y_true: np.ndarray, 
    u_pred: np.ndarray, 
    m_ext_true: np.ndarray
) -> float:
    """
    Computes the Conditional False Negative Rate (FNR) strictly over regions 
    where extreme events physically occurred (N_ext > 0).
    
    Args:
        y_true: Ground truth observations.
        u_pred: The certified spatial upper bound U_smooth(X, beta).
        m_ext_true: Boolean mask of true climatological extremes (Y > q_99).
        
    Returns:
        float: The conditional FNR [0, 1]. Returns np.nan if no true extremes exist.
    """
    if y_true.shape != u_pred.shape or y_true.shape != m_ext_true.shape:
        raise ValueError("Shapes of y_true, u_pred, and m_ext_true must exactly match.")

    if y_true.ndim == 3:
        n_ext_k = np.sum(m_ext_true, axis=(1, 2))
        valid = n_ext_k > 0
        if not np.any(valid):
            logger.debug("No true extreme events present in any field. Returning NaN for FNR.")
            return np.nan
        missed = (y_true > u_pred) & (m_ext_true == 1)
        l_cond_k = np.sum(missed, axis=(1, 2))[valid] / n_ext_k[valid]
        return float(np.mean(l_cond_k))

    n_ext = np.sum(m_ext_true)
    if n_ext == 0:
        logger.debug("No true extreme events present in the provided mask. Returning NaN for FNR.")
        return np.nan

    missed_alarms = (y_true > u_pred) & (m_ext_true == 1)
    fnr = np.sum(missed_alarms) / n_ext
    return float(fnr)


def calculate_alarm_fnr(
    y_true: np.ndarray, 
    u_pred: np.ndarray, 
    m_ext_true: np.ndarray,
    q_99_clim: np.ndarray
) -> float:
    """
    Computes the Operational Alarm False Negative Rate (Alarm FNR):
    Proportion of true extreme locations where the warning alarm failed to trigger
    (i.e. U(s) <= q_99(s) despite Y(s) > q_99(s)).
    
    By construction, an alarm miss implies an upper envelope miss, so:
    L_alarm(U, Y) <= L_cov(U, Y).
    """
    alarm_mask = (u_pred > q_99_clim).astype(int)
    if y_true.ndim == 3:
        n_ext_k = np.sum(m_ext_true, axis=(1, 2))
        valid = n_ext_k > 0
        if not np.any(valid):
            return np.nan
        alarm_miss = (alarm_mask == 0) & (m_ext_true == 1)
        l_alarm_k = np.sum(alarm_miss, axis=(1, 2))[valid] / n_ext_k[valid]
        return float(np.mean(l_alarm_k))

    n_ext = np.sum(m_ext_true)
    if n_ext == 0:
        return np.nan
    alarm_miss = (alarm_mask == 0) & (m_ext_true == 1)
    return float(np.sum(alarm_miss) / n_ext)


def calculate_spatial_far(
    u_pred: np.ndarray, 
    m_ext_true: np.ndarray, 
    q_99_clim: np.ndarray
) -> float:
    """
    Computes the Spatial False Alarm Ratio (FAR).
    Assesses operational precision: when the upper bound triggers an extreme 
    warning, what proportion of the warned area fails to realize a true extreme?
    
    FAR = False Positives / Total Predicted Positives
        = sum(I(U > q_99) * (1 - M_ext)) / sum(I(U > q_99))
    
    Args:
        u_pred: The certified spatial upper bound U_smooth(X, beta).
        m_ext_true: Boolean mask of true climatological extremes (Y > q_99).
        q_99_clim: The local climatological 99th percentile threshold.
        
    Returns:
        float: The Spatial FAR [0, 1]. Returns 0.0 if no extremes are predicted.
    """
    # An alarm is triggered if the predicted upper bound exceeds the extreme threshold
    m_ext_pred = (u_pred > q_99_clim).astype(int)
    
    total_predicted_positives = np.sum(m_ext_pred)
    if total_predicted_positives == 0:
        logger.debug("No extreme events were predicted by the upper bound. Returning 0.0 for FAR.")
        return 0.0

    # A false positive occurs when an alarm is triggered, but the true event is NOT extreme
    false_positives = (m_ext_pred == 1) & (m_ext_true == 0)
    
    far = np.sum(false_positives) / total_predicted_positives
    return float(far)


def calculate_spatial_fpr(
    u_pred: np.ndarray, 
    m_ext_true: np.ndarray, 
    q_99_clim: np.ndarray
) -> float:
    """
    Computes the Spatial False Positive Rate (FPR).
    FPR = False Positives / Total True Negatives
        = sum(I(U > q_99) * (1 - M_ext)) / sum(1 - M_ext)
    """
    m_ext_pred = (u_pred > q_99_clim).astype(int)
    non_extreme = (m_ext_true == 0)
    total_non_ext = np.sum(non_extreme)
    if total_non_ext == 0:
        return 0.0
    false_positives = (m_ext_pred == 1) & non_extreme
    return float(np.sum(false_positives) / total_non_ext)


def calculate_spatial_coherence(
    u_pred: np.ndarray, 
    q_99_clim: np.ndarray
) -> dict:
    """
    Computes topological spatial coherence metrics on the warning mask.
    Evaluates whether the warning boundary forms contiguous synoptic envelopes 
    or fragmented 'salt-and-pepper' specks.
    
    Metrics:
        - num_connected_components: Number of disjoint warning polygons.
        - total_warning_pixels: Total number of warned grid cells.
        - fragmentation_index: N_cc / total_warning_pixels (lower = more coherent).
        - mean_component_size: Average pixels per connected component.
    """
    from scipy import ndimage
    
    # 2D or 3D warning mask
    m_warn = (u_pred > q_99_clim).astype(int)
    total_pixels = int(np.sum(m_warn))
    
    if total_pixels == 0:
        return {
            "num_connected_components": 0,
            "total_warning_pixels": 0,
            "fragmentation_index": 0.0,
            "mean_component_size": 0.0
        }
        
    # If 3D (time x lat x lon), evaluate field by field and average
    if m_warn.ndim == 3:
        n_cc_list = []
        for t in range(m_warn.shape[0]):
            lbl, n_cc = ndimage.label(m_warn[t])
            n_cc_list.append(n_cc)
        avg_ncc = float(np.mean(n_cc_list))
        frag_idx = avg_ncc / max(total_pixels / m_warn.shape[0], 1.0)
        mean_size = (total_pixels / m_warn.shape[0]) / max(avg_ncc, 1.0)
        return {
            "num_connected_components": avg_ncc,
            "total_warning_pixels": total_pixels,
            "fragmentation_index": float(frag_idx),
            "mean_component_size": float(mean_size)
        }
    else:
        lbl, n_cc = ndimage.label(m_warn)
        frag_idx = float(n_cc) / max(total_pixels, 1)
        mean_size = float(total_pixels) / max(n_cc, 1)
        return {
            "num_connected_components": int(n_cc),
            "total_warning_pixels": total_pixels,
            "fragmentation_index": float(frag_idx),
            "mean_component_size": float(mean_size)
        }


def calculate_mpiw(
    l_pred: np.ndarray, 
    u_pred: np.ndarray
) -> float:
    """
    Computes the Mean Prediction Interval Width (MPIW).
    Measures the aggregate sharpness (efficiency) of the certified bounding fields.
    
    Args:
        l_pred: The spatial lower bound (can be base model q_10 or symmetrically calibrated).
        u_pred: The certified spatial upper bound U_smooth.
        
    Returns:
        float: The average width of the prediction intervals across the domain.
    """
    if l_pred.shape != u_pred.shape:
        raise ValueError("Lower and upper bound arrays must have identical shapes.")
        
    width = u_pred - l_pred
    
    # Mathematical Safeguard: Intervals should be strictly non-negative.
    if np.any(width < 0):
        logger.warning(
            "[PHYSICAL INCONSISTENCY] Lower bound exceeds upper bound at certain grid points. "
            "This indicates a failure in the quantile regression head or spatial operators."
        )
        # Clip to zero to prevent negative widths from skewing the mean
        width = np.clip(width, a_min=0.0, a_max=None)
        
    mpiw = np.mean(width)
    return float(mpiw)


def calculate_winkler_score(
    y_true: np.ndarray, 
    l_pred: np.ndarray, 
    u_pred: np.ndarray, 
    alpha: float
) -> float:
    """
    Computes the continuous Winkler Score (Winkler, 1972) for interval forecasts.
    This is a strictly proper scoring rule that simultaneously penalizes:
      1. Interval width (lack of sharpness / uninformative bounds).
      2. Under-coverage (true value falling outside the bounds).
      
    A lower Winkler score strictly implies superior probabilistic forecasting performance.
    
    Args:
        y_true: Ground truth observations.
        l_pred: Spatial lower bound.
        u_pred: Certified spatial upper bound U_smooth.
        alpha: The nominal miscoverage rate (e.g., 0.10 for a 90% interval).
               Used as a penalty multiplier for out-of-bound errors.
               
    Returns:
        float: The mean Winkler score across the evaluation field.
    """
    if not (y_true.shape == l_pred.shape == u_pred.shape):
        raise ValueError("y_true, l_pred, and u_pred must have identical shapes.")

    width = u_pred - l_pred
    
    # Penalty for values falling below the lower bound
    lower_penalty = np.where(
        y_true < l_pred, 
        (2.0 / alpha) * (l_pred - y_true), 
        0.0
    )
    
    # Penalty for values exceeding the upper bound (Missed extremes)
    upper_penalty = np.where(
        y_true > u_pred, 
        (2.0 / alpha) * (y_true - u_pred), 
        0.0
    )
    
    # Winkler Score = Width + Penalty
    winkler_grid = width + lower_penalty + upper_penalty
    
    mean_winkler = np.mean(winkler_grid)
    return float(mean_winkler)
