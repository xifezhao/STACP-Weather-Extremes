# -*- coding: utf-8 -*-
"""
Synthetic 2D Spatial Autoregressive Heavy-Tail Generator
==============================================================================

This module implements the synthetic mathematical sandbox defined in Section 4.2 
of the STACP manuscript. It isolates the statistical mechanics of conformal risk 
control from Earth system confounding factors, providing a ground-truth environment 
to validate finite-sample theorems and conduct degradation analyses.

Academic Alignment & Disclaimers:
------------------------------------------------------------------------------
- Ref Section 4.2 (Spatial Coherence): Utilizes a Gaussian Random Field (GRF) 
  with an exponential covariance function to simulate synoptic weather patterns.
- Ref Section 4.2 (Temporal Persistence): Embeds an AR(1) process to mathematically 
  define the decorrelation timescale (enabling the testing of the thinning assumption).
- Ref Section 4.2 (Extreme Heavy Tails): Employs the Probability Integral Transform 
  (PIT) to map latent Gaussian states to Student-t distributions (e.g., nu=3, 
  Fréchet maximum domain of attraction), mimicking severe convective extremes.
- Ref Section 1.2 (Regression to the Mean): The deterministic forecast systematically 
  underestimates extremes by predicting only the conditional mean of the latent state.

Author: Anonymous (Prepared for Double-Blind Peer Review)
"""

import logging
from typing import Tuple, Optional
import numpy as np
import xarray as xr
import scipy.stats as stats
from scipy.spatial import distance_matrix

logger = logging.getLogger(__name__)


class SyntheticSpatialARGenerator:
    """
    Generates synthetic spatio-temporal datasets mimicking atmospheric fields with 
    known autocorrelation, spatial coherence, and heavy-tailed extreme distributions.
    """

    def __init__(
        self, 
        grid_size: int = 50, 
        spatial_length_scale: float = 5.0, 
        decorrelation_time: float = 5.0, 
        df_t: float = 3.0,
        random_seed: int = 42
    ):
        """
        Initialize the mathematically controlled synthetic environment.
        
        Args:
            grid_size (int): Size of the 2D spatial grid (grid_size x grid_size).
            spatial_length_scale (float): Parameter rho in exp(-d / rho) covariance.
            decorrelation_time (float): e-folding timescale (tau) for AR(1) process.
                                        Determines phi = exp(-1 / tau).
            df_t (float): Degrees of freedom for Student-t tail (nu). 
                          nu=3 yields severe heavy tails (infinite 3rd+ moments).
            random_seed (int): Ensures deterministic reproducibility of the sandbox.
        """
        self.grid_size = grid_size
        self.num_points = grid_size * grid_size
        self.rho = spatial_length_scale
        
        # AR(1) coefficient phi derived from the physical decorrelation timescale
        self.phi = np.exp(-1.0 / decorrelation_time)
        
        self.df_t = df_t
        self.rng = np.random.default_rng(random_seed)

        logger.info(
            f"Synthetic Sandbox Initialized (Seed: {random_seed}).\n"
            f"  - Grid: {grid_size}x{grid_size} ({self.num_points} points)\n"
            f"  - AR(1) Persistence (phi): {self.phi:.4f} (tau={decorrelation_time})\n"
            f"  - Student-t Tails (nu): {self.df_t} (Fréchet Domain)"
        )

        self._initialize_spatial_covariance()

    def _initialize_spatial_covariance(self):
        """
        Constructs the spatial covariance matrix using an exponential decay kernel 
        and pre-computes its Cholesky decomposition for rapid GRF sampling.
        """
        logger.debug("Constructing spatial covariance matrix and Cholesky decomposition...")
        # Create 2D grid coordinates
        x = np.linspace(0, 10, self.grid_size)
        y = np.linspace(0, 10, self.grid_size)
        xv, yv = np.meshgrid(x, y)
        coords = np.column_stack((xv.ravel(), yv.ravel()))

        # Compute pairwise distance matrix
        dist_mat = distance_matrix(coords, coords)

        # Exponential covariance function: Sigma = exp(-d / rho)
        self.covariance = np.exp(-dist_mat / self.rho)

        # Cholesky decomposition for sampling: Sigma = L @ L.T
        # Add a tiny jitter to the diagonal to ensure numerical positive-definiteness
        jitter = 1e-8 * np.eye(self.num_points)
        self.L = np.linalg.cholesky(self.covariance + jitter)
        
        # Save coords for xarray construction
        self.coords = coords

    def generate_latent_ar1(self, time_steps: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulates the latent Gaussian AR(1) process.
        
        Returns:
            X_true: The exact latent state with stochastic noise.
            X_forecast: The deterministic latent state (conditional mean).
        """
        logger.debug(f"Generating latent AR(1) fields for {time_steps} time steps...")
        
        X_true = np.zeros((time_steps, self.num_points))
        X_forecast = np.zeros((time_steps, self.num_points))
        
        # Noise variance scaling to maintain stationary marginal N(0, Sigma)
        noise_scale = np.sqrt(1.0 - self.phi**2)
        
        # Pre-generate all spatial noise in a single vectorized matrix multiplication
        Z_all = self.rng.standard_normal((time_steps, self.num_points))
        spatial_noise_all = Z_all @ self.L.T
        
        # Initial state
        X_true[0] = spatial_noise_all[0]
        X_forecast[0] = X_true[0]  # Perfect initial condition

        for t in range(1, time_steps):
            X_true[t] = self.phi * X_true[t-1] + noise_scale * spatial_noise_all[t]
            X_forecast[t] = self.phi * X_true[t-1]
            
        return X_true, X_forecast

    def apply_pit(self, latent_field: np.ndarray) -> np.ndarray:
        """
        Applies the Probability Integral Transform (PIT).
        Maps N(0,1) marginals to Student-t(nu) marginals to generate heavy tails.
        """
        # CDF of Standard Normal
        u = stats.norm.cdf(latent_field)
        
        # Numerical safeguard to prevent infinities at exact 0 or 1
        u = np.clip(u, 1e-7, 1 - 1e-7)
        
        # PPF (Inverse CDF) of Student-t
        heavy_tail_field = stats.t.ppf(u, df=self.df_t)
        return heavy_tail_field

    def generate_dataset(self, time_steps: int = 1000) -> xr.Dataset:
        """
        Executes the full synthetic pipeline and packages the results into a 
        cloud-native xarray.Dataset ready for STACP pipeline ingestion.
        
        Args:
            time_steps: Total number of temporal fields to generate.
            
        Returns:
            xr.Dataset: Contains 'y_true' (observations) and 'y_forecast' 
                        (deterministic predictions causing spatial over-smoothing).
        """
        # 1. Generate Latent Gaussian Processes
        X_true, X_forecast = self.generate_latent_ar1(time_steps)
        
        # 2. Map to Physical Heavy-Tailed Extremes via PIT
        logger.debug(f"Applying PIT to map latent space to Student-t(nu={self.df_t}) extremes...")
        Y_true = self.apply_pit(X_true)
        Y_forecast = self.apply_pit(X_forecast)
        
        # 3. Reshape back to 2D spatial grid (Time, Lat, Lon)
        Y_true_2d = Y_true.reshape(time_steps, self.grid_size, self.grid_size)
        Y_forecast_2d = Y_forecast.reshape(time_steps, self.grid_size, self.grid_size)
        
        # 4. Construct the xarray Dataset
        times = np.arange(time_steps)
        lats = np.linspace(90, 0, self.grid_size)
        lons = np.linspace(0, 90, self.grid_size)
        
        ds = xr.Dataset(
            {
                "y_true": (["time", "latitude", "longitude"], Y_true_2d),
                "y_forecast": (["time", "latitude", "longitude"], Y_forecast_2d),
            },
            coords={
                "time": times,
                "latitude": lats,
                "longitude": lons,
            },
            attrs={
                "description": "STACP Synthetic Sandbox Dataset",
                "phi": self.phi,
                "decorrelation_time": -1.0 / np.log(self.phi),
                "student_t_df": self.df_t,
                "academic_purpose": "Theorem Verification & Degradation Analysis (Section 4.2)"
            }
        )
        
        logger.info(f"Synthetic Dataset successfully generated. Shape: {ds['y_true'].shape}")
        return ds
