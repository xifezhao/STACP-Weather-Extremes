#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Figure 2: Empirical vs. EVT Tail Splicing and Monotonicity
==============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

# Set publication-quality plotting style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 11,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.5,
    'grid.linestyle': '--',
    'pdf.fonttype': 42,
    'ps.fonttype': 42
})

beta_splice = 0.95  # Threshold 1 - zeta_u
beta_min, beta_max = 0.88, 0.998
u_m = 2.0           # Empirical threshold value

# 1. Dense Data Body (Empirical)
beta_body = np.linspace(beta_min, beta_splice, 100)
q_body = 2.0 - 25 * (beta_splice - beta_body)**1.8

# 2. Empirical Tail (Simulating sample variance collapse and finite sample maximum)
beta_emp_steps = np.array([0.95, 0.965, 0.975, 0.982, 0.988, 0.993, 0.998])
q_emp_steps = np.array([2.0, 2.15, 2.8, 2.8, 3.6, 4.3, 4.3])

# 3. EVT GPD Extrapolation Tail (Strictly monotonic)
beta_evt = np.linspace(beta_splice, beta_max, 200)
sigma_over_xi = 1.5
xi = 0.4
zeta_u = 1.0 - beta_splice

q_evt = u_m + sigma_over_xi * ((zeta_u / (1 - beta_evt))**xi - 1)

fig, ax = plt.subplots(figsize=(10, 6))

# --- A. Background Shading for the Sparse Tail Region ---
ax.axvspan(beta_splice, beta_max, color='red', alpha=0.05, lw=0, zorder=0)
ax.text(0.974, 6.2, 'Sparse Tail Region', color='darkred', 
        fontsize=12, fontweight='bold', ha='center', va='center', zorder=5)

# --- B. Plot the Functions ---
ax.plot(beta_body, q_body, color='dimgray', lw=2.5, label='Dense Body (Empirical)', zorder=2)

ax.plot(beta_emp_steps, q_emp_steps, drawstyle='steps-post', color='#1f77b4', lw=2.5, 
        marker='o', markersize=6, label=r'Empirical Tail $\max(q_m^{emp}, u_m)$', zorder=3)

ax.plot(beta_evt, q_evt, color='#d62728', lw=2.5, 
        label='STACP Spliced Tail (EVT/GPD)', zorder=4)

# --- C. Guide Lines and Markers ---
ax.axvline(x=beta_splice, color='black', linestyle='--', alpha=0.6, lw=1.5, zorder=1)
ax.hlines(y=u_m, xmin=beta_min, xmax=beta_splice, color='black', linestyle='--', alpha=0.6, lw=1.5, zorder=1)
ax.scatter([beta_splice], [u_m], color='black', s=80, zorder=6)

# --- D. Academic Annotations ---
bbox_props_emp = dict(boxstyle='round,pad=0.4', fc='white', ec='#1f77b4', lw=1.5, alpha=0.9)
ax.annotate(
    'Sample Variance Collapse\nPlateaus at max observation;\nhalts FST sequence prematurely.', 
    xy=(0.993, 4.3), xycoords='data',
    xytext=(0.935, 4.8), textcoords='data',
    arrowprops=dict(arrowstyle='->', color='#1f77b4', lw=2, connectionstyle='arc3,rad=0.1'),
    bbox=bbox_props_emp, fontsize=10, fontweight='bold', color='#1f77b4', ha='left'
)

bbox_props_evt = dict(boxstyle='round,pad=0.4', fc='white', ec='#d62728', lw=1.5, alpha=0.9)
ax.annotate(
    'Monotonic Extrapolation\nProvides a continuous, structured\ncandidate space for LTT.', 
    xy=(0.985, 4.5), xycoords='data',
    xytext=(0.935, 5.7), textcoords='data',
    arrowprops=dict(arrowstyle='->', color='#d62728', lw=2, connectionstyle='arc3,rad=-0.1'),
    bbox=bbox_props_evt, fontsize=10, fontweight='bold', color='#d62728', ha='left'
)

# --- E. Axes Formatting ---
ax.set_xlim(beta_min, beta_max)
ax.set_ylim(0, 6.5)

xticks = [0.88, 0.90, 0.92, 0.94, beta_splice, 0.96, 0.98, 0.99]
xticklabels = ['0.88', '0.90', '0.92', '0.94', r'$1 - \zeta_u$', '0.96', '0.98', '0.99']
ax.set_xticks(xticks)
ax.set_xticklabels(xticklabels)

yticks = [0, 1, 2, 3, 4, 5, 6]
yticklabels = ['0', '1', r'$u_m$', '3', '4', '5', '6']
ax.set_yticks(yticks)
ax.set_yticklabels(yticklabels)

ax.set_xlabel(r'Target Index Level $\beta \to 1$', fontsize=13, fontweight='bold')
ax.set_ylabel(r'Extrapolated Quantile Score $q(\beta)$', fontsize=13, fontweight='bold')

ax.legend(loc='upper left', frameon=True, edgecolor='black', fontsize=11, borderpad=0.8)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig('Fig2_Tail_Splicing.pdf', format='pdf', bbox_inches='tight')
plt.savefig('Fig2_Tail_Splicing.png', format='png', dpi=300, bbox_inches='tight')
print("Successfully generated Fig2_Tail_Splicing.pdf and .png")
