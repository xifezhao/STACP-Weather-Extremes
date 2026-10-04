# %% [markdown]
# # Reproducibility Notebook: Figure 3 (Sample Size Bottleneck)
# **Project:** Spatial Tail-Adaptive Conformal Prediction (STACP) Framework  
# **Manuscript Section:** 5.1 (Sample Size Feasibility and The Certification Bottleneck)
# 
# ### Academic Alignment & Purpose:
# This notebook generates **Figure 3** of the manuscript. It visualizes the strict mathematical boundaries imposed by the Learn-then-Test (LTT) certification under finite-sample constraints. 
# 
# **Key Academic Arguments Demonstrated:**
# 1. **The Physical Bottleneck:** Meteorological records (e.g., ERA5) are constrained by temporal autocorrelation. Applying a strict thinning interval ($\Delta t \geq 5$ days) drastically reduces the effective independent sample size ($n_{val} \approx 80-150$).
# 2. **Certification Impossible Region:** By inverting the Hoeffding-Bentkus inequality, we demonstrate that demanding an overly stringent risk tolerance (e.g., $\alpha_{risk} = 0.05$ globally) pushes the maximum allowable empirical FNR ($\hat{r}_{max}$) below zero, rendering statistical certification mathematically impossible.
# 3. **Justification of Experimental Design:** This plot provides a rigorous, *a priori* justification for evaluating models on targeted variables independently (avoiding massive Bonferroni penalties) and adopting a practical risk tolerance of $\alpha_{risk} = 0.10$.

# %% [code]
# 1. IMPORT REQUIRED LIBRARIES & SET ACADEMIC PLOT STYLE
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set publication-quality plotting style (AMS / Nature standard)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 11,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.5,
    "grid.linestyle": "--"
})

# %% [markdown]
# ### 2. Exact Hoeffding-Bentkus P-Value Inversion
# We implement the exact continuous HB bound used in the LTT framework and numerically invert it to find $\hat{r}_{max}$ (the maximum empirical risk that still yields $p \le \delta$).

# %% [code]
def hb_pvalue(r_hat: float, n: int, alpha: float) -> float:
    """
    Computes the Hoeffding-Bentkus upper bound p-value for H_0: E[L] > alpha, 
    where L is a bounded continuous random variable in [0,1].
    """
    if r_hat >= alpha:
        return 1.0
        
    # 1. Hoeffding (Bennett-type) Bound
    # Using tiny epsilon to avoid log(0) numerical instability
    kl_div = (r_hat * np.log(r_hat / alpha + 1e-12) + 
              (1.0 - r_hat) * np.log((1.0 - r_hat) / (1.0 - alpha + 1e-12)))
    h1_bound = np.exp(-n * kl_div)
    
    # 2. Bentkus Bound (Theorem 1.1 in Bentkus 2004)
    k = int(np.floor(n * r_hat))
    bentkus_bound = np.e * stats.binom.cdf(k, n, alpha)
    
    return min(h1_bound, bentkus_bound, 1.0)

def find_max_empirical_risk(n: int, alpha: float, delta: float = 0.05) -> float:
    """
    Finds the maximum allowable empirical risk (r_hat_max) such that p_val <= delta.
    Evaluates over the discrete grid of possible empirical mean values (k/n).
    """
    max_k = -1
    for k in range(n + 1):
        r_hat = k / n
        pval = hb_pvalue(r_hat, n, alpha)
        if pval <= delta:
            max_k = k
        else:
            break
            
    if max_k == -1:
        return -0.01  # Mathematical proxy for "Impossible to Certify"
    return max_k / n

# %% [markdown]
# ### 3. Generate Curve Data
# We iterate over the realistic domain of independent spatial fields $n_{val} \in [20, 200]$ for two distinct risk tolerances ($\alpha_{risk} = 0.10$ and $0.05$).

# %% [code]
n_vals = np.arange(20, 201)
delta_nominal = 0.05

r_max_alpha10 = np.array([find_max_empirical_risk(n, alpha=0.10, delta=delta_nominal) for n in n_vals])
r_max_alpha05 = np.array([find_max_empirical_risk(n, alpha=0.05, delta=delta_nominal) for n in n_vals])

# Find interception points where r_max drops below 0 (Certification Impossible boundary)
n_cutoff_10 = n_vals[np.argmax(r_max_alpha10 >= 0)] if np.any(r_max_alpha10 >= 0) else 200
n_cutoff_05 = n_vals[np.argmax(r_max_alpha05 >= 0)] if np.any(r_max_alpha05 >= 0) else 200

# %% [markdown]
# ### 4. Generating the Publication-Quality Figure
# Combining shaded areas (for physical regimes and mathematical dead-zones) with the inverted boundary curves.

# %% [code]
fig, ax = plt.subplots(figsize=(11, 7))

# --- A. Background Shading ---
# 1. Certification Impossible Region (y < 0)
ax.axhspan(-0.05, 0, color='red', alpha=0.08, lw=0, zorder=0)
ax.text(25, -0.015, r'Certification Impossible Region ($\hat{r}_{max} < 0$)', 
        color='darkred', fontsize=12, fontweight='bold', zorder=5)
ax.text(25, -0.03, '(Cannot certify even if empirical FNR is exactly zero)', 
        color='darkred', fontsize=10, zorder=5)

# 2. Typical ERA5 Sample Size Band (n = 80 to 150)
ax.axvspan(80, 150, color='#1f77b4', alpha=0.08, lw=0, zorder=0)
# Arrow indicating the band
ax.annotate('', xy=(80, 0.11), xytext=(150, 0.11),
            arrowprops=dict(arrowstyle='<->', color='#1f77b4', lw=2), zorder=5)
ax.text(115, 0.112, r'Typical ERA5 Validation Limit' + '\n' + r'($\Delta t \geq 5$ days & $N_{ext}>0$)', 
        color='#1f77b4', fontsize=10, fontweight='bold', ha='center', va='bottom', zorder=5)

# --- B. Plot the Bounding Curves ---
# Alpha = 0.10
ax.plot(n_vals, r_max_alpha10, color='#1f77b4', lw=2.5, 
        label=r'Target Tolerance $\alpha_{risk} = 0.10$', zorder=3)

# Alpha = 0.05
ax.plot(n_vals, r_max_alpha05, color='#ff7f0e', lw=2.5, 
        label=r'Target Tolerance $\alpha_{risk} = 0.05$', zorder=3)

# The Zero Line (Absolute Mathematical Boundary)
ax.axhline(0, color='black', linestyle='--', lw=1.5, alpha=0.7, zorder=2)

# --- C. Annotations & Markers ---
# 1. Highlight intercept for alpha=0.05
ax.scatter([n_cutoff_05], [0], color='#ff7f0e', s=80, zorder=6, edgecolor='white', lw=1.5)
bbox_props = dict(boxstyle="round,pad=0.4", fc="white", ec="#ff7f0e", lw=1.5, alpha=0.9)
ax.annotate(
    f"Mathematical Ceiling\nDrops below 0 if $n_{{val}} < {n_cutoff_05}$", 
    xy=(n_cutoff_05, 0), xycoords='data',
    xytext=(n_cutoff_05 + 15, -0.015), textcoords='data',
    arrowprops=dict(arrowstyle="->", color="#ff7f0e", lw=2),
    bbox=bbox_props, fontsize=10, fontweight='bold', color="#ff7f0e", ha="left"
)

# 2. Highlight a specific operative point for alpha=0.10 (e.g., n=100)
operative_n = 100
operative_r = find_max_empirical_risk(operative_n, 0.10, delta_nominal)
ax.scatter([operative_n], [operative_r], color='#1f77b4', s=80, zorder=6, edgecolor='white', lw=1.5)
ax.vlines(x=operative_n, ymin=0, ymax=operative_r, color='#1f77b4', linestyle=':', lw=2, zorder=2)
ax.text(operative_n + 3, operative_r, r"$\hat{r}_{\max} \approx " + f"{operative_r:.3f}$", 
        color='#1f77b4', fontsize=11, fontweight='bold', va='center')

# 3. Global Bonferroni Warning Box
warning_text = (
    "Note on Global Certification:\n"
    "Applying a massive Bonferroni penalty\n"
    "(e.g., across 20 variables/lead times)\n"
    "shifts these curves sharply downward,\n"
    "pushing them entirely into the red zone."
)
props = dict(boxstyle='round,pad=0.5', facecolor='#fffbcc', edgecolor='gray', alpha=0.9)
ax.text(25, 0.08, warning_text, fontsize=10, bbox=props, zorder=5, color='black', style='italic')

# --- D. Axes Formatting ---
ax.set_xlim(20, 200)
ax.set_ylim(-0.05, 0.125)

ax.set_xlabel(r"Independent Validation Fields ($n_{val}$)", fontsize=13, fontweight='bold')
ax.set_ylabel(r"Max Allowable Empirical FNR ($\hat{r}_{max}$)", fontsize=13, fontweight='bold')

# Custom Y-ticks
yticks = [-0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12]
ax.set_yticks(yticks)
ax.set_yticklabels([f"{y:.2f}" for y in yticks])

# Legend
ax.legend(loc='lower right', frameon=True, edgecolor='black', fontsize=11, borderpad=0.8)

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig("Fig3_SampleSize_Bottleneck.pdf", format='pdf', bbox_inches='tight')
plt.savefig("Fig3_SampleSize_Bottleneck.png", format='png', dpi=300, bbox_inches='tight')
plt.show()

# %% [markdown]
# ### Figure Caption (For Manuscript Insertion)
# **Figure 3: Sample Size Bottleneck and Minimum Certifiable Risk.** Theoretical limits of the Learn-then-Test (LTT) procedure derived by exactly inverting the Hoeffding-Bentkus inequality ($\delta = 0.05$). The curves illustrate the maximum allowable empirical False Negative Rate ($\hat{r}_{max}$) required to statistically guarantee a given risk tolerance $\alpha_{risk}$. The blue vertical band represents the realistic number of independent validation fields ($n_{val} \approx 80-150$) extractable from a typical ERA5 multi-year validation set after strict temporal thinning ($\Delta t \geq 5$ days) and conditional filtering ($N_{ext} > 0$). The red shaded area denotes the "Certification Impossible Region": if $n_{val}$ is critically low, or if the target $\alpha_{risk}$ is too stringent (e.g., orange curve where $\alpha_{risk} = 0.05$), $\hat{r}_{max}$ falls below zero, rendering mathematical certification strictly impossible even if the model perfectly bounds all extremes. This physical constraint dictates the necessity of evaluating distinct variable-lead time pairs marginally without global Bonferroni inflation.
