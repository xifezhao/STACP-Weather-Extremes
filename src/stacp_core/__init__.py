# -*- coding: utf-8 -*-
"""STACP core algorithms submodule."""

from .evt_calibration import (
    EVTCalibrator, 
    SplicedTailFunction, 
    ExponentialTailFunction, 
    ConstantOffsetFunction
)
from .spatial_ops import SpatialOperator, apply_spatial_dilation, apply_convex_smoothing
from .ltt_certify import SpatialLTTCertifier, hoeffding_bentkus_pvalue

__all__ = [
    "EVTCalibrator",
    "SplicedTailFunction",
    "ExponentialTailFunction",
    "ConstantOffsetFunction",
    "SpatialOperator",
    "apply_spatial_dilation",
    "apply_convex_smoothing",
    "SpatialLTTCertifier",
    "hoeffding_bentkus_pvalue",
]
