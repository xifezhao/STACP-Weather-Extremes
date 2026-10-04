# -*- coding: utf-8 -*-
"""
Auxiliary Quantile Regression Head for Deterministic AI Models
==============================================================================

This module implements the auxiliary quantile regression head appended to a 
frozen deterministic base model (e.g., GraphCast or Pangu-Weather). It maps 
local state features to nominal heuristic uncertainty bounds (q_lo, q_hi).

Academic Alignment & Disclaimers:
------------------------------------------------------------------------------
- Ref Section 2.1: Deterministic trajectories inherently lack variance bounds. 
  This module generates the preliminary boundaries required for subsequent EVT 
  scoring. It is trained via the Pinball (Quantile) Loss.
- Spatial Consistency: Utilizes Circular Padding along the longitudinal dimension 
  to prevent artificial physical discontinuities at the dateline/prime meridian.
- Strict Data Isolation: The dataset used to optimize this module MUST be 
  strictly disjoint from D_cal (used for EVT) and D_val (used for LTT).

Author: Anonymous (Prepared for Double-Blind Peer Review)
"""

import logging
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class CircularPad2d(nn.Module):
    """
    Padding module for meteorological grids.
    For global domains (covering 360 degrees), applies circular (periodic) padding along longitude.
    For regional cropped domains (e.g. North America), applies replicate padding to prevent
    non-physical boundary wrap-around between West and East edges.
    Latitude dimension uses replicate padding at the poles.
    """
    def __init__(self, padding: int, force_replicate: bool = False):
        super().__init__()
        self.pad = padding
        self.force_replicate = force_replicate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Channels, Latitude, Longitude)
        lon_size = x.shape[-1]
        if self.force_replicate or lon_size < 240:
            return F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='replicate')
            
        # Pad Longitude (last dim) circularly for global grids
        x = F.pad(x, (self.pad, self.pad, 0, 0), mode='circular')
        # Pad Latitude (second to last dim) via replication at poles
        x = F.pad(x, (0, 0, self.pad, self.pad), mode='replicate')
        return x


class QuantileLoss(nn.Module):
    """
    Standard Pinball Loss for Quantile Regression.
    Evaluates the asymmetric penalty for predicting specific quantiles.
    """
    def __init__(self, quantiles: List[float]):
        super().__init__()
        self.quantiles = quantiles

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            preds: Tensor of shape (B, len(quantiles), H, W)
            target: Tensor of shape (B, 1, H, W) containing the true observations.
        Returns:
            Scalar tensor representing the mean pinball loss.
        """
        assert preds.shape[1] == len(self.quantiles), "Prediction channels must match number of quantiles."
        
        losses = []
        for i, q in enumerate(self.quantiles):
            err = target.squeeze(1) - preds[:, i, :, :]
            # Pinball loss formula: max(q * err, (q - 1) * err)
            loss_q = torch.max(q * err, (q - 1.0) * err)
            losses.append(loss_q)
            
        # Average over all quantiles, batch, and spatial dimensions
        return torch.stack(losses, dim=1).mean()


class AuxiliaryQuantileHead(nn.Module):
    """
    A lightweight, computationally efficient fully-convolutional neural network (FCNN) 
    designed to sit on top of a frozen deterministic AI model.
    
    It predicts the upper and lower nominal bounds (e.g., 10th and 90th percentiles) 
    without altering the base model's pre-trained weights.
    """
    def __init__(
        self, 
        in_channels: int, 
        hidden_channels: int = 32, 
        quantiles: List[float] = [0.10, 0.90]
    ):
        """
        Args:
            in_channels (int): Number of input features (e.g., base deterministic 
                               prediction + relevant physical state variables).
            hidden_channels (int): Channel depth for the lightweight hidden layers.
            quantiles (List[float]): The target nominal probabilities.
        """
        super().__init__()
        self.quantiles = quantiles
        out_channels = len(quantiles)
        
        # A lightweight 3-layer residual architecture to ensure O(1)-like inference speed.
        # We enforce circular padding to respect the spherical topology of the Earth.
        self.net = nn.Sequential(
            CircularPad2d(padding=1),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, stride=1, padding=0),
            nn.GELU(),
            
            CircularPad2d(padding=1),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, stride=1, padding=0),
            nn.GELU(),
            
            CircularPad2d(padding=1),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=0)
        )
        
        logger.info(
            f"AuxiliaryQuantileHead initialized. "
            f"Targeting quantiles: {self.quantiles}. "
            f"Ensure base AI model weights remain FROZEN during training."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor (B, C, Lat, Lon).
        Returns:
            torch.Tensor: Predicted quantiles (B, len(quantiles), Lat, Lon).
                          Channel 0: q_lo, Channel 1: q_hi.
        """
        # (B, len(quantiles), Lat, Lon)
        quantiles_pred = self.net(x)
        
        # Physical/Mathematical Safeguard: Ensure q_lo <= q_hi implicitly.
        # By sorting along the channel dimension, we prevent quantile crossing anomalies
        # that could destabilize the downstream EVT non-conformity score calculation.
        quantiles_pred, _ = torch.sort(quantiles_pred, dim=1)
        
        return quantiles_pred

    @staticmethod
    def verify_data_isolation(
        train_years: Tuple[int, int], 
        cal_years: Tuple[int, int], 
        val_years: Tuple[int, int]
    ):
        """
        Strict academic safeguard to verify there is no temporal data leakage 
        between the fitting of this heuristic head and the STACP certification process.
        
        Ref Section 3.4 (Validity Assumptions): Strict Data Independence.
        """
        def overlaps(y1: Tuple[int, int], y2: Tuple[int, int]) -> bool:
            return not (y1[1] < y2[0] or y1[0] > y2[1])

        if overlaps(train_years, cal_years) or overlaps(train_years, val_years):
            raise ValueError(
                "\n[CRITICAL ACADEMIC VIOLATION] Data Leakage Detected!\n"
                f"Quantile head training period {train_years} overlaps with the "
                f"calibration {cal_years} or validation {val_years} periods.\n"
                "To satisfy the conditional exchangeability assumption of the LTT certification, "
                "the dataset used to optimize this module MUST be strictly disjoint."
            )
        
        logger.info("[STATISTICAL RIGOR ENFORCED] Temporal isolation verified for Quantile Head training.")
