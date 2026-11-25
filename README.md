# Spatial Tail-Adaptive Conformal Prediction (STACP) for Weather Extremes

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Official Implementation of:**  
> Jieqiong Ding and Xiaofei Zhao (2026), *Bridging Extreme Value Theory and Machine Learning: A Spatial Conformal Risk Control Framework for Weather Extremes* (submitted to *Monthly Weather Review*).

---

## Scientific Overview

**STACP** bridges Extreme Value Theory (EVT) with Conformal Risk Control (CRC) to attach finite-sample, field-level extreme-weather risk bounds to deterministic AI weather forecasts:

- **Module 1 (Offline EVT Candidate Generation):** Normalized upper-tail residuals $S_j = \max\{Y_j - F(\mathbf{X}_j), 0\}/s(\mathbf{X}_j)$ stratified into $M=5$ standardized forecast anomaly bins (edges $\{-1.0, -0.33, 0.33, 1.0\}$) are spatially declustered and fitted with a Generalized Pareto Distribution (GPD) via Hosking–Wallis probability-weighted moments (PWM) with mean-coupled scale ($\hat\xi \in [-0.20, 0.45]$), generating a monotone $K=229$ candidate family ($\beta_k = 1 - 10^{-t_k}$, $t_k \in [6.0, 0.3]$ in steps of $0.025$).
- **Module 2 (Spatial Learn-then-Test Certification):** Order-preserving morphological dilation ($r=1$ flat disk, 5-point stencil) and truncated Gaussian smoothing ($\sigma=1.0$) are embedded inside Fixed Sequence Testing (FST) using Hoeffding–Bentkus $p$-values on temporally thinned validation fields ($\Delta t = 5$ days) with physical extremes ($N_{ext}>0$).

---

## Experimental Protocol & Data Partition

All real-data experiments use ERA5 truth and operational Pangu-Weather forecasts (plus 50-member ECMWF IFS ensembles) from **WeatherBench 2** on the standardized **$1.5^\circ$ grid** (North America: $24\times 47 = 1128$ points; Europe: $23\times 33 = 759$ points), strictly partitioned by valid date:

- **Climatology ($q_{99}^{clim}, \mu_{clim}, \sigma_{clim}$, 31-day circular window):** 1990–2017
- **Local error scale $s(\mathbf{X})$ (3-month rolling grid-point RMS residual):** 2018
- **Calibration ($\mathcal{D}_{cal}$, scores, bins, GPD fits):** 2019
- **Validation ($\mathcal{D}_{val}$, thinned $\Delta t = 5$ d, LTT certification):** 2020, 2021, and pooled 2020–2021
- **Held-out Test ($\mathcal{D}_{test}$):** 2022

Synthetic benchmarks (`stacp_v2/run_synthetic.py`) use a $16\times 16$ spatial AR(1) Gaussian signal ($\rho=3.0$) plus symmetric heavy-tailed GPD errors ($\xi=1/3, \sigma=1, s=0.5$), $\Delta t = \lceil 2\tau\rceil$, and $N_{rep}=1000$ ($T=365$ and tail-capping benchmark) / $500$ ($T=1095$) replications with fixed seeds (`2026`, `1000+100*phi+T`, `5000+n_cal`).

---

## Repository Structure

```text
STACP-Weather-Extremes/
├── stacp_v2/                        # Canonical v2 authentic experiment pipeline
│   ├── core.py                      # GPD-PWM, SpatialOp (r=1, sigma=1.0), HB p-values, FST
│   ├── extract_wb2.py               # WeatherBench 2 ERA5 / Pangu-Weather / IFS extraction (1.5°)
│   ├── run_synthetic.py             # Synthetic AR(1) + symmetric GPD Monte Carlo benchmarks
│   ├── run_real.py                  # Full NA/EU × 72/120/168 h × Z500/T2m evaluation
│   └── run_emos_baseline.py         # Parametric Gaussian EMOS baseline (q90 & q99)
├── experiments/                     # Synchronized mirror of stacp_v2/ entry points
├── results/
│   ├── SHA256SUMS.txt               # SHA-256 checksum manifest for all authentic JSONs & canonical figures
│   ├── v2_authentic/                # Authentic JSON outputs (real_*, synthetic, emos_baseline, case_metrics)
│   └── figures/                     # Canonical manuscript vector figures (Fig7, Fig8)
├── tests/
│   └── test_core.py                 # Unit tests for GPD-PWM, HB p-values, SpatialOp, and FST (CI-verified)
├── .github/workflows/ci.yml         # Automated pytest + SHA-256 verification workflow
├── CITATION.cff                     # Machine-readable software & manuscript citation metadata
├── .gitignore                       # Standard Python, Jupyter, and large-data ignore rules
├── notebooks/
│   ├── Fig2_Tail_Splicing.py        # Tail splicing schematic
│   ├── Fig3_SampleSize_Bottleneck.py# Hoeffding–Bentkus sample-size ceiling
│   ├── Fig4_Degradation_Analysis.py # Autocorrelation & thinning bottleneck
│   ├── Fig5_FST_pvalue_Trajectory.py# FST p-value trajectory & N_cal certification rate
│   ├── Fig6_Sharpness_Metrics.py    # Normalized Z500 spatial operator ablation
│   ├── Fig7_MultiLead_CrossRegion.py# Authentic multi-lead (72–168 h) & NA/EU Fig. 7 (incl. EMOS)
│   └── Fig8_Synoptic_Case_Map.py    # Authentic Sep 2022 Western NA heat-wave synoptic map (Fig. 8)
├── configs/                         # Reference YAML configuration files
├── src/                             # Modular reference library
├── LICENSE                          # MIT License
├── requirements.txt                 # Python pip dependencies
└── environment.yml                  # Conda environment specification
```

---

## Quickstart & Reproduction

### 1. Environment Setup & Integrity Verification
```bash
pip install -r requirements.txt
pytest -v tests/test_core.py
sha256sum -c results/SHA256SUMS.txt   # or: shasum -a 256 -c results/SHA256SUMS.txt
```

### 2. Run Synthetic Monte Carlo Benchmarks
```bash
python stacp_v2/run_synthetic.py
```
Outputs `synthetic.json` (archived in `results/v2_authentic/synthetic.json`).

### 3. Run Real-Data ERA5 / Pangu-Weather & EMOS Benchmarks
Set `STACP_DATA` to any local storage directory (defaults to `./data/processed`):
```bash
export STACP_DATA=./data/processed
python stacp_v2/extract_wb2.py
python stacp_v2/run_real.py
python stacp_v2/run_emos_baseline.py
```
Outputs the 12 authentic JSON files (`results/v2_authentic/real_{var}_{dom}_{lead}.json`) and `emos_baseline.json`.

### 4. Reproduce Canonical Manuscript Figures (Fig. 7 & Fig. 8)
> **Note on Canonical Figures:** In the *Monthly Weather Review* manuscript, Figures 2–6 are rendered natively via TikZ/PGFPlots directly from `results/v2_authentic/*.json`, while Figures 7 and 8 are vector PDFs archived exclusively in `results/figures/` (`Fig7-MultiLead-CrossRegion.pdf` and `Fig8-Synoptic-Case-Map.pdf`).

```bash
python notebooks/Fig7_MultiLead_CrossRegion.py
python notebooks/Fig8_Synoptic_Case_Map.py
```

---

## Archival & Citation

This repository includes a machine-readable [`CITATION.cff`](CITATION.cff) file and is tagged as release `v1.0.0-mwr-submission` corresponding to the submitted *Monthly Weather Review* manuscript. Upon manuscript acceptance, a permanent versioned snapshot will be minted with a Zenodo DOI.

```bibtex
@article{ding_zhao_2026_stacp,
  author  = {Ding, Jieqiong and Zhao, Xiaofei},
  title   = {Bridging Extreme Value Theory and Machine Learning: A Spatial Conformal Risk Control Framework for Weather Extremes},
  journal = {Monthly Weather Review},
  year    = {2026},
  note    = {Submitted}
}
```
