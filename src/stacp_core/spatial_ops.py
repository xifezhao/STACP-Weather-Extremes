# -*- coding: utf-8 -*-
"""
Order-Preserving Spatial Operators for STACP Certification
==============================================================================

This module implements the spatial topological transformations embedded DIRECTLY 
within the Learn-then-Test (LTT) certification loop, as detailed in Section 3.3 
of the STACP manuscript.

Academic Alignment & Disclaimers:
------------------------------------------------------------------------------
- Ref Section 3.3 (Monotonicity Preservation): To guarantee the validity of the 
  Fixed Sequence Testing (FST), candidate bounding fields U(X, beta) must remain 
  monotonically non-decreasing with respect to beta. Thus, ONLY order-preserving 
  operations (max-pooling and strictly positive convex smoothing) are utilized.
- Ref Section 3.3 (Embedded Execution): Applying ad-hoc spatial smoothing AFTER 
  statistical certification invalidates the formal risk bounds. These operators 
  are executed prior to computing the conditional loss L_cond.
- Spherical Topology: Implements custom padding (circular for longitude, 
  edge/replicate for latitude) to prevent unphysical artifacts at the dateline 
  and poles.

Author: Anonymous (Prepared for Double-Blind Peer Review)
"""

import logging
from typing import Dict, Any, Tuple
import numpy as np
import scipy.ndimage as ndimage

logger = logging.getLogger(__name__)


class SpatialOperator:
    """
    Applies mathematically rigorously bounded spatial transformations to 2D 
    meteorological score fields, ensuring topological coherence without 
    violating statistical certification constraints.
    """

    def __init__(self, config: Dict[str, Any], variable: str = "default"):
        """
        Initialize the spatial operator with strictly fixed a priori parameters.
        
        Args:
            config (Dict): Parsed configuration from `configs/spatial_params.yaml`.
            variable (str): The meteorological variable (e.g., 'Z500', 'T2m') to 
                            fetch specific physical overrides.
        """
        # Fetch physical overrides if variable is explicitly defined, else use default
        var_config = config.get('variable_overrides', {}).get(variable.upper())
        if var_config is None:
            logger.debug(f"No specific spatial override for {variable}. Using default_topology.")
            var_config = config.get('default_topology', {})
            
        self.dilation_cfg = var_config.get('dilation', {'apply': False})
        self.smoothing_cfg = var_config.get('smoothing', {'apply': False})
        self.boundary_mode = config.get('boundary_conditions', {}).get('mode', 'circular')

        # Academic Safeguard: Enforce strictly convex smoothing methods
        if self.smoothing_cfg.get('apply'):
            method = self.smoothing_cfg.get('method', '').lower()
            if method not in ['gaussian', 'uniform']:
                raise ValueError(
                    f"\n[CRITICAL THEORETICAL VIOLATION]\n"
                    f"Smoothing method '{method}' is not formally order-preserving. "
                    f"To maintain monotonicity for FST, only 'gaussian' or 'uniform' "
                    f"(convex combinations with non-negative weights) are permitted."
                )

        logger.info(
            f"SpatialOperator initialized for '{variable}'.\n"
            f"  - Dilation: {self.dilation_cfg.get('apply')} (k={self.dilation_cfg.get('kernel_size')})\n"
            f"  - Smoothing: {self.smoothing_cfg.get('apply')} ({self.smoothing_cfg.get('method')})\n"
            f"  - Longitude Boundary: {self.boundary_mode}"
        )

    def _apply_mixed_padding(self, field: np.ndarray, pad_size: int) -> np.ndarray:
        """
        Applies physically consistent boundary conditions to Earth grids.
        For global grids covering the full longitudinal circle (>= 359 deg), 
        circular wrapping is applied. For regional sub-domains (e.g., North America), 
        edge replication is applied to prevent unphysical wrapping across boundaries.
        Latitude (second to last dim) uses edge replication at the poles.
        
        Args:
            field: Array of shape (..., Lat, Lon).
            pad_size: The padding radius.
        """
        if pad_size == 0:
            return field
            
        ndim = field.ndim
        # Determine whether grid is global.
        # If boundary_mode is "edge" or "regional", or if lon dimension indicates a regional crop,
        # use edge replication to prevent wrapping West and East boundaries together.
        lon_size = field.shape[-1]
        is_global = (self.boundary_mode == "circular") and (lon_size >= 240)
        
        pad_lon = [(0, 0)] * ndim
        pad_lon[-1] = (pad_size, pad_size)
        if is_global:
            field = np.pad(field, pad_lon, mode='wrap')
        else:
            field = np.pad(field, pad_lon, mode='edge')
            
        # Step 2: Pad Latitude (North-South) with edge replication to prevent 
        # non-physical crossover of the poles
        pad_lat = [(0, 0)] * ndim
        pad_lat[-2] = (pad_size, pad_size)
        field = np.pad(field, pad_lat, mode='edge')
        
        return field

    def _crop_padding(self, field: np.ndarray, pad_size: int) -> np.ndarray:
        """Removes the padding to return the field to its original geographical shape."""
        if pad_size == 0:
            return field
        return field[..., pad_size:-pad_size, pad_size:-pad_size]

    def apply_dilation(self, field: np.ndarray) -> np.ndarray:
        """
        Applies morphological dilation (Max-Pooling) to expand extreme envelopes,
        handling slight spatial displacement errors in the base AI forecast.
        """
        if not self.dilation_cfg.get('apply', False):
            return field
            
        k = self.dilation_cfg.get('kernel_size', 3)
        pad = k // 2
        
        # Apply mixed padding
        padded_field = self._apply_mixed_padding(field, pad)
        
        # Define footprints strictly for the spatial dimensions (Lat, Lon)
        footprint = np.ones((k, k))
        if padded_field.ndim > 2:
            # Expand footprint to match dimensions, ignoring time/batch axes
            shape = [1] * (padded_field.ndim - 2) + [k, k]
            footprint = footprint.reshape(shape)
            
        # Max filter preserves monotonicity inherently: max(A) <= max(B) if A <= B
        dilated_padded = ndimage.maximum_filter(padded_field, footprint=footprint, mode='nearest')
        
        return self._crop_padding(dilated_padded, pad)

    def apply_convex_smoothing(self, field: np.ndarray) -> np.ndarray:
        """
        Applies order-preserving convex smoothing to eliminate "salt-and-pepper" 
        artifacts and generate topologically coherent bounding polygons.
        """
        if not self.smoothing_cfg.get('apply', False):
            return field
            
        method = self.smoothing_cfg.get('method', 'gaussian')
        k = self.smoothing_cfg.get('kernel_size', 5)
        sigma = self.smoothing_cfg.get('sigma', 1.0)
        
        # Pad at least max(k // 2, ceil(4 * sigma)) to prevent Gaussian truncation boundary zero-leak
        pad = max(k // 2, int(np.ceil(4.0 * sigma)))
        
        padded_field = self._apply_mixed_padding(field, pad)
        
        if method == 'gaussian':
            # Create a sigma sequence that only smooths spatial dims
            sigmas = [0.0] * (padded_field.ndim - 2) + [sigma, sigma]
            smoothed_padded = ndimage.gaussian_filter(padded_field, sigma=sigmas, mode='nearest')
            
        elif method == 'uniform':
            size = [1] * (padded_field.ndim - 2) + [k, k]
            smoothed_padded = ndimage.uniform_filter(padded_field, size=size, mode='nearest')
            
        return self._crop_padding(smoothed_padded, pad)

    def __call__(self, field: np.ndarray) -> np.ndarray:
        """
        Executes the full spatial transformation pipeline.
        This must be called iteratively within the LTT testing sequence.
        
        Args:
            field: The base score field Q_base(X, beta) generated from EVT interpolation.
            
        Returns:
            np.ndarray: The topologically coherent Q_smooth(X, beta) field.
        """
        if np.isscalar(field) or (isinstance(field, np.ndarray) and field.ndim == 0):
            return field
        if isinstance(field, np.ndarray) and (field.size <= 1 or np.ptp(field) == 0):
            return field
            
        field_dilated = self.apply_dilation(field)
        field_smoothed = self.apply_convex_smoothing(field_dilated)
        # Guarantee point-wise envelope dominance: Q_final >= Q_base everywhere
        return np.maximum(field, field_smoothed)


def apply_spatial_dilation(field: np.ndarray, kernel_size: int = 3, boundary_mode: str = "circular") -> np.ndarray:
    """Standalone functional interface for morphological dilation on Earth grids."""
    op = SpatialOperator(
        config={
            "default_topology": {
                "dilation": {"apply": True, "kernel_size": kernel_size},
                "smoothing": {"apply": False}
            },
            "boundary_conditions": {"mode": boundary_mode}
        }
    )
    return op.apply_dilation(field)


def apply_convex_smoothing(
    field: np.ndarray, 
    kernel_size: int = 5, 
    sigma: float = 1.0, 
    method: str = "gaussian", 
    boundary_mode: str = "circular"
) -> np.ndarray:
    """Standalone functional interface for convex smoothing on Earth grids."""
    op = SpatialOperator(
        config={
            "default_topology": {
                "dilation": {"apply": False},
                "smoothing": {"apply": True, "method": method, "kernel_size": kernel_size, "sigma": sigma}
            },
            "boundary_conditions": {"mode": boundary_mode}
        }
    )
    return op.apply_convex_smoothing(field)
