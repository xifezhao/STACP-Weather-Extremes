# -*- coding: utf-8 -*-
r"""
Temporal Thinning & Independent Sample Extractor for STACP
==============================================================================

This module implements the strict temporal thinning operator (\Delta t) required 
to satisfy the "Approximate Exchangeability" validity assumption outlined in 
Section 2.3 and Section 3.4 of the manuscript.

Academic Alignment & Disclaimers:
------------------------------------------------------------------------------
- Ref Section 2.3: Atmospheric fields exhibit strong serial autocorrelation 
  (e.g., blocking highs persisting for days). This module explicitly enforces a 
  temporal stride (e.g., 120 hours) exceeding the integral decorrelation timescale.
- Ref Figure 4 (Degradation Analysis): Bypassing this module will artificially 
  inflate the effective sample size (n_val) and cause catastrophic inflation 
  of the Type-I error during the Hoeffding-Bentkus certification.
- Target-Time Independence: For forecast evaluation, the thinning is strictly 
  applied to the VALIDATION TARGET TIME, not the initialization time, preventing 
  cross-lead-time event overlap.

Author: Anonymous (Prepared for Double-Blind Peer Review)
"""

import logging
from typing import Optional, Dict, Any
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


class TemporalThinner:
    """
    Applies strict temporal subsampling to spatio-temporal weather datasets to 
    extract approximately independent validation fields (D_val^{indep}).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the TemporalThinner with parameters strictly frozen a priori.
        
        Args:
            config (Dict): Parsed configuration from `configs/thinning_rules.yaml`.
                           Modifying these post-hoc violates FWER control.
        """
        self.default_stride = config['default_policy']['stride_hours']
        self.sampling_method = config['default_policy'].get('sampling_method', 'fixed_offset')
        self.variable_overrides = config.get('variable_overrides', {})
        self.reset_at_season = config.get('stratification', {}).get('reset_stride_at_season_boundary', True)

        logger.info(
            f"TemporalThinner initialized. Default stride: {self.default_stride}h. "
            f"Active overrides: {list(self.variable_overrides.keys())}."
        )

    def _get_stride_hours(self, variable: str) -> int:
        """
        Retrieve the physically derived decorrelation timescale for a specific variable.
        """
        # Map variables to their specific overrides if configured (e.g., 'Z500' -> 168h)
        for key, rules in self.variable_overrides.items():
            if key.upper() in variable.upper():
                override = rules['stride_hours']
                logger.debug(f"Variable '{variable}' matches override '{key}': Using {override}h stride.")
                return override
        
        logger.debug(f"No specific override for '{variable}'. Using default {self.default_stride}h stride.")
        return self.default_stride

    def _thin_sequence(self, times: pd.DatetimeIndex, stride_hours: int) -> pd.DatetimeIndex:
        """
        Core logic to extract an independent sequence of timestamps strictly separated 
        by at least `stride_hours`.
        """
        if len(times) == 0:
            return times

        selected_times = []
        # Sort times to guarantee chronological processing
        times = times.sort_values()
        
        # Enforce the fixed_offset deterministic sampling
        last_selected = times[0]
        selected_times.append(last_selected)
        
        delta = pd.Timedelta(hours=stride_hours)
        
        for current_time in times[1:]:
            # If the temporal gap exceeds the physical decorrelation scale, it is deemed independent
            if current_time - last_selected >= delta:
                selected_times.append(current_time)
                last_selected = current_time
                
        return pd.DatetimeIndex(selected_times)

    def apply_thinning(
        self, 
        ds: xr.Dataset, 
        variable: str, 
        time_dim: str = "time"
    ) -> xr.Dataset:
        """
        Applies the temporal thinning operator to an xarray Dataset.
        
        Args:
            ds: The input spatio-temporal dataset (e.g., validation set D_val).
            variable: The meteorological variable being processed, used to look up 
                      the physical decorrelation timescale (stride).
            time_dim: The dimension representing the TARGET valid time.
                      Must be the valid time, NOT the initialization time.
                      
        Returns:
            xr.Dataset: The thinned dataset containing only independent fields.
        """
        if time_dim not in ds.dims and time_dim not in ds.coords:
            raise ValueError(f"Time dimension '{time_dim}' not found in the dataset.")

        original_size = ds.sizes[time_dim]
        stride_hours = self._get_stride_hours(variable)
        
        # Convert xarray time coordinate to Pandas DatetimeIndex for robust manipulation
        times = pd.DatetimeIndex(ds[time_dim].values)
        sorted_times = times.sort_values()
        
        if self.reset_at_season:
            # Enforce strict decorrelation stride globally across seasons to prevent boundary leakage
            # (e.g. May 31 18:00 and June 1 00:00 separated by only 6 hours)
            delta = pd.Timedelta(hours=stride_hours)
            selected = []
            last_time = None
            for t in sorted_times:
                if last_time is None or (t - last_time >= delta):
                    selected.append(t)
                    last_time = t
            independent_times = pd.DatetimeIndex(selected)
        else:
            independent_times = self._thin_sequence(sorted_times, stride_hours)

        # Slice the original dataset to retain only the certified independent fields
        ds_thinned = ds.sel({time_dim: independent_times})
        final_size = ds_thinned.sizes[time_dim]
        
        # ACADEMIC LOGGING: Explicitly report the cost of statistical rigor
        reduction_pct = (1 - final_size / original_size) * 100 if original_size > 0 else 0
        logger.warning(
            f"\n[STATISTICAL RIGOR ENFORCED] Temporal Thinning applied for '{variable}'.\n"
            f"Decorrelation Stride : {stride_hours} hours\n"
            f"Original Sample Size : {original_size} fields\n"
            f"Independent Samples  : {final_size} fields (n_val)\n"
            f"Sample Reduction     : {reduction_pct:.1f}% discarded to guarantee approximate exchangeability.\n"
        )
        
        return ds_thinned
