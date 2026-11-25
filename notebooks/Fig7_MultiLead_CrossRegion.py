"""Publication-ready Figure 7: Multi-lead (72h, 120h, 168h) and cross-region (NA, EU)
authentic ERA5 + Pangu-Weather verification from results/v2_authentic/real_*.json.
"""
import json
import os
from pathlib import Path
import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl_cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

BASE = Path(__file__).resolve().parents[1] / "results" / "v2_authentic"
LEADS = [72, 120, 168]
DOMS = ["NA", "EU"]
VARS = [("z500", "Z500 (500-hPa Geopotential Height)"), ("t2m", "T2m (2-m Temperature)")]

def load_metric(var, dom, lead, method, metric):
    d = json.loads((BASE / f"real_{var}_{dom}_{lead}.json").read_text())
    t = d["cert"]["0.1"]["pooled"][method]["test"]
    if metric == "dfrag":
        far = t["far"]
        return t["ncc"] / far if far > 0 else np.nan
    return t[metric]

# Okabe-Ito colorblind-safe palette
C_PT = "#D55E00"      # vermilion
C_NOOP = "#0072B2"    # blue
C_OP = "#009E73"      # bluish green

fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.8), constrained_layout=True)

panel_labels = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]
col_titles = [
    "Conditional FNR (Test 2022)",
    "Spatial False Alarm Rate (FAR)",
    r"Fragmentation Density $D_{\mathrm{frag}}$",
]
ylabels = ["Conditional FNR", "Spatial FAR", r"$D_{\mathrm{frag}} = N_{cc}/\mathrm{FAR}$"]
metrics = ["fnr", "far", "dfrag"]

for r, (var, vlabel) in enumerate(VARS):
    for c, (metric, ctitle, ylab) in enumerate(zip(metrics, col_titles, ylabels)):
        ax = axes[r, c]
        for dom, ls, mk in [("NA", "-", "o"), ("EU", "--", "s")]:
            for method, col, mlabel in [
                ("pointwise", C_PT, "Pointwise CP"),
                ("gpd_noop", C_NOOP, "STACP GPD (No Ops)"),
                ("gpd_op", C_OP, "STACP GPD (+ Ops)"),
            ]:
                vals = [load_metric(var, dom, ld, method, metric) for ld in LEADS]
                lbl = f"{mlabel} ({dom})" if (r == 0 and c == 0) else None
                ax.plot(LEADS, vals, color=col, linestyle=ls, marker=mk,
                        linewidth=1.6, markersize=5, label=lbl)
        if metric == "fnr":
            ax.axhline(0.10, color="black", linestyle=":", linewidth=1.2,
                       label=(r"Risk Budget $\alpha_{\mathrm{risk}}=0.10$" if r == 0 else None))
            ax.set_ylim(0, 0.46 if var == "t2m" else 0.20)
        elif metric == "far":
            ax.set_ylim(0, 1.0)
        else:
            ax.set_ylim(bottom=0)
        ax.set_xticks(LEADS)
        ax.set_xticklabels([f"{ld} h\n({ld//24} d)" for ld in LEADS])
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        ax.set_ylabel(ylab)
        if r == 0:
            ax.set_title(f"{panel_labels[r][c]} {ctitle}", fontweight="bold", pad=6)
        else:
            ax.set_title(f"{panel_labels[r][c]} {vlabel.split()[0]}: {ctitle}", fontweight="bold", pad=6)
            ax.set_xlabel("Forecast Lead Time")
        if c == 0:
            ax.text(0.03, 0.92, vlabel.split()[0], transform=ax.transAxes,
                    fontsize=8.5, fontweight="bold", va="top",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.7", alpha=0.9))

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.07),
           ncol=4, frameon=True, edgecolor="0.7")

out_repo = Path(__file__).resolve().parents[1] / "results" / "figures" / "Fig7-MultiLead-CrossRegion.pdf"
out_latex = Path(__file__).resolve().parents[2] / "Latex_sources" / "Fig7-MultiLead-CrossRegion.pdf"
out_repo.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_repo, bbox_inches="tight")
fig.savefig(out_latex, bbox_inches="tight")
print("Saved authentic Figure 7 to:", out_repo, "and", out_latex)
