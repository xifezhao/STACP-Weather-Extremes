# -*- coding: utf-8 -*-
"""
Climatological Baseline and Extreme Mask Generator
==============================================================================

This module implements the climatological definitions established in Section 2.2 
of the STACP manuscript. It computes local, seasonally varying thresholds 
(e.g., the 99th percentile) to define extreme weather anomalies, accounting for 
the continuous cyclical nature of the Earth system.

Academic Alignment & Disclaimers:
------------------------------------------------------------------------------
- Ref Section 2.2: Absolute global thresholds are physically unsuitable. We 
  compute q_99^{clim}(x, t) locally per grid point using a rolling day-of-year 
  (DOY) window to capture synoptic seasonality.
- No Data Leakage Constraint: The historical period (baseline) MUST be strictly 
  disjoint from the calibration (D_cal) and validation (D_val) periods. This 
  module explicitly checks for chronological overlap.
- Topological Masking: Outputs the boolean field M_ext(x), which serves as the 
  fundamental conditioning filter (N_ext > 0) in the spatial LTT certification.

Author: Anonymous (Prepared for Double-Blind Peer Review)
"""

import logging
import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)


class ExtremeMaskGenerator:
    """
    Computes local climatological quantiles using a circular rolling day-of-year 
    window and generates boolean spatial masks for conditional risk bounding.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the generator based on the strictly frozen a priori configuration.
        
        Args:
            config (Dict): Parsed configuration from `configs/default_stacp.yaml` 
                           (specifically the `extreme_definition` block).
        """
        ext_cfg = config.get('extreme_definition', {})
        self.percentile = ext_cfg.get('climatological_percentile', 99.0)
        self.window_days = ext_cfg.get('rolling_window_days', 31)
        self.baseline_years = ext_cfg.get('baseline_years', [1979, 2015])
        
        # Mathematical quantile expects [0, 1] range
        self.q = self.percentile / 100.0

        logger.info(
            f"ExtremeMaskGenerator initialized.\n"
            f"  - Target Percentile : {self.percentile}th (q={self.q})\n"
            f"  - DOY Rolling Window: {self.window_days} days (circular)\n"
            f"  - Baseline Reference: {self.baseline_years[0]} to {self.baseline_years[1]}"
        )

    def _get_circular_doy_mask(self, target_doy: int, half_window: int) -> np.ndarray:
        """
        Generates a valid Day-of-Year (DOY) boolean mask incorporating circular 
        wrapping to handle end-of-year and start-of-year continuity.
        
        Args:
            target_doy: The central Day of Year (1 to 366).
            half_window: Half the size of the rolling window in days.
            
        Returns:
            np.ndarray: A boolean array of length 366 where valid days are True.
        """
        doys = np.arange(1, 367)
        lower_bound = target_doy - half_window
        upper_bound = target_doy + half_window

        if lower_bound < 1:
            # Underflow: e.g., target DOY is Jan 2, window reaches back into late December
            valid_mask = (doys >= (lower_bound + 366)) | (doys <= upper_bound)
        elif upper_bound > 366:
            # Overflow: e.g., target DOY is Dec 30, window reaches forward into early January
            valid_mask = (doys >= lower_bound) | (doys <= (upper_bound - 366))
        else:
            # Standard mid-year window
            valid_mask = (doys >= lower_bound) & (doys <= upper_bound)
            
        return valid_mask

    def compute_climatology(
        self, 
        ds_historical: xr.Dataset, 
        variable: str
    ) -> xr.DataArray:
        """
        Computes the q_99^{clim}(x, DOY) reference field using historical data.
        
        Args:
            ds_historical: The raw historical dataset.
            variable: The meteorological variable of interest (e.g., 'z500').
            
        Returns:
            xr.DataArray: The climatological quantile threshold, dimensioned by (dayofyear, lat, lon).
        """
        # Validate historical data boundaries to prevent data leakage
        hist_years = ds_historical.time.dt.year.values
        min_yr, max_yr = hist_years.min(), hist_years.max()
        
        if max_yr > self.baseline_years[1] or min_yr < self.baseline_years[0]:
            logger.warning(
                f"Historical dataset spans {min_yr}-{max_yr}, which falls outside the "
                f"strictly defined baseline {self.baseline_years}. Truncating data to "
                f"prevent subtle data leakage."
            )
            ds_historical = ds_historical.sel(
                time=slice(f"{self.baseline_years[0]}-01-01", f"{self.baseline_years[1]}-12-31")
            )

        logger.info(f"Computing circular {self.percentile}th percentile over {min_yr}-{max_yr} baseline...")
        
        # Add a dayofyear coordinate if not explicitly present
        if 'dayofyear' not in ds_historical.coords:
            ds_historical = ds_historical.assign_coords(dayofyear=ds_historical.time.dt.dayofyear)

        half_window = self.window_days // 2
        q_clim_list = []

        # Iterate through every day of the year (1 to 366)
        for doy in range(1, 367):
            doy_mask = self._get_circular_doy_mask(doy, half_window)
            
            # Select all historical samples falling into the circular window across all baseline years
            valid_doys = np.where(doy_mask)[0] + 1
            window_data = ds_historical[variable].where(
                ds_historical.dayofyear.isin(valid_doys), 
                drop=True
            )
            
            # Calculate the threshold locally at each grid point
            # Use skipna=True to handle missing values over oceans/land depending on the dataset
            q_val = window_data.quantile(self.q, dim='time', skipna=True)
            q_val = q_val.assign_coords(dayofyear=doy)
            q_clim_list.append(q_val)
            
        # Concatenate along the 'dayofyear' dimension
        q_clim_da = xr.concat(q_clim_list, dim='dayofyear')
        
        logger.info(f"Climatology computation complete. Field shape: {q_clim_da.shape}")
        return q_clim_da

    def generate_mask(
        self, 
        ds_target: xr.Dataset, 
        q_clim_da: xr.DataArray, 
        variable: str
    ) -> xr.DataArray:
        """
        Evaluates a target dataset against the pre-computed climatological baseline 
        to generate the boolean extreme mask M_ext(x).
        
        Args:
            ds_target: The validation or test dataset (D_val or D_test).
            q_clim_da: The output from `compute_climatology` (q_99^{clim}).
            variable: The meteorological variable being masked.
            
        Returns:
            xr.DataArray: Boolean spatial mask indicating the presence of active extremes.
        """
        # --- ACADEMIC SAFEGUARD ---
        # Check if the target data chronologically overlaps with the baseline
        target_years = ds_target.time.dt.year.values
        if np.min(target_years) <= self.baseline_years[1]:
            raise ValueError(
                f"\n[CRITICAL DATA LEAKAGE DETECTED]\n"
                f"Target dataset includes year {np.min(target_years)}, which overlaps with or precedes "
                f"the baseline end year {self.baseline_years[1]}. The baseline period must be "
                f"strictly disjoint from the evaluation sets to satisfy LTT exchangeability."
            )
        
        logger.info("[STATISTICAL RIGOR ENFORCED] Chronological disjoint verified. Generating M_ext mask.")
        
        # Leverage xarray's groupby functionality to broadcast the dayofyear climatology 
        # onto the high-resolution specific timestamps of the target dataset.
        grouped_target = ds_target[variable].groupby('time.dayofyear')
        
        # M_ext = I(Y(x) > q_99^{clim}(x, DOY))
        m_ext = grouped_target > q_clim_da
        
        # Ensure the output explicitly acts as an integer boolean (0 or 1) 
        # to facilitate downstream spatial loss mathematical operations.
        return m_ext.astype(int)
