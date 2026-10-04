# Spatial Tail-Adaptive Conformal Prediction (STACP) for Weather Extremes

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Preregistered%20Configs-brightgreen.svg)]()
[![Manuscript Status](https://img.shields.io/badge/Manuscript-Under%20Review-orange.svg)]()

> **Official Implementation of:**  
> *Bridging Extreme Value Theory and Machine Learning: A Spatial Conformal Risk Control Framework for Weather Extremes*

---

## 🔬 Scientific Overview

Deterministic data-driven weather forecasting models (e.g., GraphCast, Pangu-Weather) demonstrate competitive Root Mean Square Error (RMSE) skill, but fundamentally suffer from **spatial over-smoothing** in the tails due to Mean Squared Error (MSE) regression-to-the-mean dynamics. 

Applying standard Conformal Prediction (CP) to physical atmospheric extremes encounters two core bottlenecks:
1. **Tail Sample Variance Collapse:** In the sparse extreme tail, empirical quantiles become noisy step functions, prematurely halting risk certification.
2. **Spatial Fragmentation:** Marginal point-wise coverage guarantees produce speckled "salt-and-pepper" masks rather than actionable warning regions.

**STACP** bridges classical **Extreme Value Theory (EVT)** with modern **Conformal Risk Control (CRC)** via strict decoupling of efficiency and validity:
- **Efficiency (Module 1):** Declustered upper-tail residuals parameterized via Generalized Pareto Distributions (GPD) with Probability Weighted Moments (PWM) and continuous monotonic splicing.
- **Validity (Module 2):** Spatial Learn-then-Test (LTT) certification with order-preserving operators (dilation + convex smoothing) embedded inside the testing loop, certifying the field-level Conditional False Negative Rate ($L_{cond}$) using finite-sample Hoeffding-Bentkus p-values.

```
                    Deterministic AI Model (GraphCast / Pangu)
                                       │
                                       ▼
                     Auxiliary Quantile Regression Head
                         (Nominal q_lo, q_hi Bounds)
                                       │
       ┌───────────────────────────────┴───────────────────────────────┐
       ▼                                                               ▼
[Module 1: Efficiency]                                      [Module 2: Validity]
Declustered EVT Tail Calibration                             Spatial LTT Certification
- Spatiotemporal Declustering                               - Temporal Thinning (Δt >= τ)
- PWM GPD Estimation (ξ, σ)                                 - Order-Preserving Spatial Ops
- Monotonically Spliced Quantiles                           - Hoeffding-Bentkus Finite Bound
       │                                                               │
       └───────────────────────────────┬───────────────────────────────┘
                                       │
                                       ▼
                       Certified Spatial Hazard Polygons
                   (Guaranteed Conditional FNR <= α_risk)
```

---

## ⚠️ Academic Integrity & Preregistration Protocol

To prevent **p-value hacking** and ensure the theoretical finite-sample coverage guarantees remain mathematically valid:
- All hyperparameters in `configs/default_stacp.yaml`, `configs/spatial_params.yaml`, and `configs/thinning_rules.yaml` **must remain strictly frozen prior to observing the validation fields ($D_{val}$)**.
- Validation target dates **must be thinned** by at least the decorrelation timescale ($\Delta t \ge 5$ to $7$ days) to satisfy approximate exchangeability. Violating thinning causes catastrophic Type-I error inflation (demonstrated in `experiments/02_synthetic_sim.py`).
- Evaluation sets must remain strictly disjoint from climatological baseline periods (1979–2015) to prevent subtle distribution leakage.

---

## 📁 Repository Structure

```
STACP-Weather-Extremes/
├── configs/                    # A priori locked hyperparameters (Preregistration)
│   ├── default_stacp.yaml      # Global risk tolerance, beta sequence, EVT bounds
│   ├── spatial_params.yaml     # Order-preserving dilation & convex kernel scales
│   └── thinning_rules.yaml     # Physical decorrelation timescales per variable
│
├── data/                       # Data storage protocols
│   ├── raw/                    # ERA5 proxies, WeatherBench 2, TIGGE, GEFS
│   ├── processed/              # Thinned independent sets & base forecasts
│   └── climatology/            # Circular rolling 99th percentile masks
│
├── src/                        # Core algorithmic package
│   ├── data_utils/             # Cloud-native fetchers, thinning, climatology
│   ├── models/                 # Quantile head & 2D spatial AR heavy-tail generator
│   ├── stacp_core/             # EVT calibrator, spatial ops, LTT certifier
│   └── evaluation/             # Conditional FNR, Spatial FAR, MPIW, Winkler score
│
├── experiments/                # End-to-end reproducible experiment scripts
│   ├── 00_prepare_data.py      # Historical climatology & thinned validation splits
│   ├── 01_train_qhead.py       # Frozen base model quantile head fitting
│   ├── 02_synthetic_sim.py     # Controlled Monte Carlo theorem & degradation tests
│   ├── 03_run_stacp_era5.py    # Main ERA5 + AI model empirical evaluation
│   ├── 04_run_ablation.py      # Empirical tail vs. EVT tail ablation study
│   ├── 05_gefs_expansion.py    # Multi-decadal reforecast sample expansion
│   └── 06_baseline_compare.py  # Benchmark vs. 51-member ECMWF EPS
│
├── notebooks/                  # Publication figure reproduction notebooks
│   ├── Fig2_Tail_Splicing.ipynb            # Spliced tail monotonicity
│   ├── Fig3_SampleSize_Bottleneck.ipynb    # Inverted HB certification bottleneck
│   ├── Fig4_Degradation_Analysis.ipynb     # Type-I error inflation under autocorrelation
│   ├── Fig5_FST_pvalue_Trajectory.ipynb    # FST p-value descent & premature halting
│   ├── Fig6_Sharpness_Metrics.ipynb        # Normalized FAR, MPIW & Winkler scores
│   └── Fig7_Spatial_Coherence_Map.ipynb    # Pacific Northwest Heatwave hazard maps
│
├── requirements.txt            # Python pip dependencies
├── environment.yml             # Conda environment definition
└── README.md                   # Project documentation
```

---

## 🚀 Quickstart & Reproduction Guide

### 1. Environment Setup

Using Conda (recommended for Geospatial packages):
```bash
conda env create -f environment.yml
conda activate stacp-env
```

Or using pip:
```bash
pip install -r requirements.txt
```

### 2. Verify Mathematical Sandbox (Synthetic Monte Carlo)

Run the synthetic Monte Carlo simulations to verify finite-sample coverage guarantees and temporal thinning degradation dynamics:
```bash
python experiments/run_synthetic.py
```
Outputs are saved into `results/synthetic.json`.

### 3. Real-World Experiments on WeatherBench 2 (ERA5 & Pangu-Weather)

Step 1: Extract WeatherBench 2 data arrays:
```bash
python experiments/extract_wb2.py
```

Step 2: Run STACP calibration, LTT certification, and evaluation across domains (NA, EU) and lead times (72h, 120h, 168h):
```bash
python experiments/run_real.py
```
Full evaluation metrics, parameter selections, and bootstrap confidence intervals will be recorded under `results/`.

---

## 📊 Generating Publication Figures

Each figure in the manuscript has an identical paired `.py` script and `.ipynb` notebook inside `notebooks/`:
- **Figure 2:** `jupyter notebook notebooks/Fig2_Tail_Splicing.ipynb`
- **Figure 3:** `jupyter notebook notebooks/Fig3_SampleSize_Bottleneck.ipynb`
- **Figure 4:** `jupyter notebook notebooks/Fig4_Degradation_Analysis.ipynb`
- **Figure 5:** `jupyter notebook notebooks/Fig5_FST_pvalue_Trajectory.ipynb`
- **Figure 6:** `jupyter notebook notebooks/Fig6_Sharpness_Metrics.ipynb`
- **Figure 7:** `jupyter notebook notebooks/Fig7_Spatial_Coherence_Map.ipynb`

All notebooks export publication-ready vectors (`.pdf`) and 300-DPI raster images (`.png`).

---

## 📖 Citation

If you use this framework or experimental setup in your research, please cite:

```bibtex
@article{stacp_weather_extremes_2026,
    title={Bridging Extreme Value Theory and Machine Learning: A Spatial Conformal Risk Control Framework for Weather Extremes},
    author={Anonymous Authors},
    journal={Under Review},
    year={2026}
}
```
