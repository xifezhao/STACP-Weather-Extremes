# -*- coding: utf-8 -*-
"""
Data Fetching Interfaces for STACP Framework
==============================================================================

This module provides cloud-native, lazy-loading interfaces for retrieving 
climatological proxies (ERA5), deterministic AI forecasts (WeatherBench 2), 
and traditional ensembles (TIGGE, GEFS Reforecasts).

Academic Alignment & Disclaimers:
------------------------------------------------------------------------------
- Ref Section 2.1: ERA5 is explicitly treated as an "observational proxy". 
  Users must acknowledge its systematic biases in under-representing localized 
  extremes (e.g., convective precipitation) (Hersbach et al., 2020).
- Ref Section 4.3: Global evaluations dilute the Conditional FNR. Thus, the 
  fetchers mandate or strongly encourage the use of regional bounding boxes 
  (e.g., North America, Europe) to isolate synoptic extreme events.
- Ref Section 6.2: GEFSReforecastFetcher is provided specifically to expand 
  the independent sample pool (n_val) to overcome natural timeline bottlenecks.

Author: Anonymous (Prepared for Double-Blind Peer Review)
"""

import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, Union
import xarray as xr
import pandas as pd

# Assume fsspec and zarr are available in the environment (standard for modern geoscience)
import fsspec

logger = logging.getLogger(__name__)


class BaseDataFetcher(ABC):
    """
    Abstract Base Class for all weather dataset fetchers.
    Enforces a standardized contract for data retrieval and spatial cropping
    to ensure reproducibility across empirical evaluations.
    """

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the fetcher, optionally with a local cache directory.
        """
        self.cache_dir = cache_dir

    @abstractmethod
    def fetch(
        self,
        variables: List[str],
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> xr.Dataset:
        """
        Fetch a spatiotemporal weather dataset.

        Args:
            variables: List of meteorological variables (e.g., ['z500', 't2m']).
            start_date: Validation/Calibration start time.
            end_date: Validation/Calibration end time.
            bbox: Bounding box for regional evaluation (min_lon, max_lon, min_lat, max_lat).
                  Strongly recommended as per Section 4.3 to prevent risk dilution.

        Returns:
            xarray.Dataset: Lazy-loaded, temporally and spatially subsetted dataset.
        """
        pass

    def _crop_spatial_domain(self, ds: xr.Dataset, bbox: Tuple[float, float, float, float]) -> xr.Dataset:
        """
        Slices the global xarray dataset to a specific regional domain.
        Handles standard latitude (90 to -90) and longitude (0 to 360 or -180 to 180) formats.
        """
        min_lon, max_lon, min_lat, max_lat = bbox
        
        # Address potential longitude wrap-around issues (e.g., [0, 360) vs [-180, 180))
        if max_lon < 0 and ds.longitude.max() > 180:
            logger.warning("Longitude mismatch detected. Rolling dataset to [-180, 180).")
            ds = ds.assign_coords(longitude=(((ds.longitude + 180) % 360) - 180)).sortby('longitude')

        # Assume latitude is monotonically decreasing (90 to -90) as per standard ECMWF convention
        if ds.latitude[0] > ds.latitude[-1]:
            ds_cropped = ds.sel(
                latitude=slice(max_lat, min_lat),
                longitude=slice(min_lon, max_lon)
            )
        else:
            ds_cropped = ds.sel(
                latitude=slice(min_lat, max_lat),
                longitude=slice(min_lon, max_lon)
            )
            
        logger.info(f"Cropped domain to bbox {bbox}. Spatial shape: {ds_cropped.sizes}")
        return ds_cropped


class ERA5ProxyFetcher(BaseDataFetcher):
    """
    Fetcher for the ERA5 Reanalysis dataset (Proxy Ground Truth).
    Connects to the Google Cloud Public Datasets Zarr store via WeatherBench 2.
    """

    # Official GCP endpoint for WB2 ERA5 climatology/truth
    GCP_ERA5_ZARR_URL = "gs://weatherbench2/datasets/era5/1959-2022-6h-240x121_equiangular_with_poles_conservative.zarr"

    def fetch(
        self,
        variables: List[str],
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> xr.Dataset:
        
        logger.info(f"Connecting to ERA5 proxy ground truth via GCP: {self.GCP_ERA5_ZARR_URL}")
        
        mapper = fsspec.get_mapper(self.GCP_ERA5_ZARR_URL)
        # Consolidate metadata enables rapid cloud-native open
        ds = xr.open_zarr(mapper, consolidated=True)
        
        # Subset Time
        ds_time = ds.sel(time=slice(start_date, end_date))
        
        # Select Variables (map user names to WB2 ERA5 standard names)
        var_mapping = {'z500': 'geopotential', 't2m': '2m_temperature', 'tp': 'total_precipitation'}
        query_vars = [var_mapping.get(v.lower(), v) for v in variables]
        
        try:
            ds_vars = ds_time[query_vars]
        except KeyError as e:
            raise KeyError(f"Requested variable not found in ERA5 Zarr store. Valid variables: {list(ds.data_vars.keys())}") from e

        # Subset Space (Crucial for Section 4.3 Validation)
        if bbox:
            ds_vars = self._crop_spatial_domain(ds_vars, bbox)
        else:
            logger.warning(
                "Global domain selected. As discussed in Section 4.3, global topological "
                "risk evaluation dilutes conditional FNR (N_ext > 0 is almost always true). "
                "Regional cropping is heavily advised for impact-based extremes."
            )

        return ds_vars


class WeatherBench2Fetcher(BaseDataFetcher):
    """
    Fetcher for Deterministic AI Forecasts (e.g., GraphCast, Pangu-Weather) 
    hosted within the WeatherBench 2 ecosystem.
    """

    # Dictionary mapping model names to their cloud Zarr endpoints
    WB2_MODELS = {
        "graphcast": "gs://weatherbench2/datasets/graphcast/2016-2021_6h_0.25deg.zarr",
        "pangu": "gs://weatherbench2/datasets/pangu/2018-2021_6h_0.25deg.zarr"
    }

    def __init__(self, model_name: str, cache_dir: Optional[str] = None):
        super().__init__(cache_dir)
        self.model_name = model_name.lower()
        if self.model_name not in self.WB2_MODELS:
            raise ValueError(f"Model {self.model_name} not supported. Options: {list(self.WB2_MODELS.keys())}")
        self.zarr_url = self.WB2_MODELS[self.model_name]

    def fetch(
        self,
        variables: List[str],
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        bbox: Optional[Tuple[float, float, float, float]] = None,
        lead_time_hours: Optional[int] = None
    ) -> xr.Dataset:
        
        logger.info(f"Connecting to WB2 {self.model_name.upper()} forecasts: {self.zarr_url}")
        
        mapper = fsspec.get_mapper(self.zarr_url)
        ds = xr.open_zarr(mapper, consolidated=True)
        
        # Subset Initialization Time
        ds_time = ds.sel(time=slice(start_date, end_date))
        
        # Isolate specific lead time if requested (to evaluate efficiency at specific horizons)
        if lead_time_hours is not None:
            lead_time_td = pd.Timedelta(hours=lead_time_hours)
            try:
                ds_time = ds_time.sel(prediction_timedelta=lead_time_td)
            except KeyError:
                raise ValueError(f"Lead time {lead_time_hours}h not found in {self.model_name} outputs.")
                
        # Select variables
        ds_vars = ds_time[variables]
        
        if bbox:
            ds_vars = self._crop_spatial_domain(ds_vars, bbox)

        return ds_vars


class GEFSReforecastFetcher(BaseDataFetcher):
    """
    Fetcher for NOAA GEFSv12 Reforecasts.
    Ref Section 6.2: Addresses the physical timeline constraints of ERA5 by 
    utilizing multi-decadal reforecast ensembles to synthetically expand the 
    independent validation sample pool (n_val) for rigorous risk certification.
    """

    def fetch(
        self,
        variables: List[str],
        start_date: Union[str, pd.Timestamp],
        end_date: Union[str, pd.Timestamp],
        bbox: Optional[Tuple[float, float, float, float]] = None
    ) -> xr.Dataset:
        
        logger.info("Initializing GEFS Reforecast extraction. This operation expands n_val as per Section 6.2.")
        # Implementation placeholder: In practice, this connects to AWS NOAA PDS (Public Dataset)
        # e.g., s3://noaa-gefs-retrospective/
        raise NotImplementedError("GEFS AWS extraction logic is deferred to implementation-specific AWS configurations.")
