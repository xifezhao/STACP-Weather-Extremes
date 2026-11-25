"""Publication-ready Figure 7: Multi-lead (72h, 120h, 168h) and cross-region (NA, EU)
authentic ERA5 + Pangu-Weather verification from results/v2_authentic/real_*.json
and results/colab_full_run/emos_baseline.json, including 95% stationary bootstrap CIs.
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
    "legend.fontsize": 7.3,
    "figure.dpi": 300,
    "savefig.dpi": 300,
})

ROOT = Path(__file__).resolve().parents[1] / "results"
BASE = ROOT / "v2_authentic"
EMOS_PATH = ROOT / "colab_full_run" / "emos_baseline.json"
if not EMOS_PATH.exists():
    EMOS_PATH = BASE / "emos_baseline.json"
EMOS = json.loads(EMOS_PATH.read_text())

LEADS = [72, 120, 168]
VARS = [("z500", "Z500"), ("t2m", "T2m")]

def load_series(var, dom, method, metric):
    vals, lo, hi = [], [], []
    for ld in LEADS:
        if method == "emos99":
            t = EMOS[f"{var}_{dom}_{ld}"]["q99"]
            if metric == "dfrag":
                vals.append(t["d_frag"])
                lo.append(np.nan); hi.append(np.nan)
            else:
                vals.append(t[metric])
                ci = t.get(f"{metric}_ci")
                lo.append(ci[0] if ci else np.nan)
                hi.append(ci[1] if ci else np.nan)
        else:
            d = json.loads((BASE / f"real_{var}_{dom}_{ld}.json").read_text())
            t = d["cert"]["0.1"]["pooled"][method]["test"]
            if metric == "dfrag":
                far = t["far"]
                vals.append(t["ncc"] / far if far > 0 else np.nan)
                lo.append(np.nan); hi.append(np.nan)
            else:
                vals.append(t[metric])
                ci = t.get(f"{metric}_ci")
                lo.append(ci[0] if ci else np.nan)
                hi.append(ci[1] if ci else np.nan)
    return np.array(vals), np.array(lo), np.array(hi)

# Okabe-Ito colorblind-safe palette
C_PT = "#D55E00"      # vermilion
C_EMOS = "#CC79A7"    # reddish purple
C_NOOP = "#0072B2"    # blue
C_OP = "#009E73"      # bluish green

fig, axes = plt.subplots(2, 3, figsize=(11.0, 6.1), constrained_layout=True)

panel_labels = [["(a)", "(b)", "(c)"], ["(d)", "(e)", "(f)"]]
col_titles = [
    "Conditional FNR (95% Bootstrap CI)",
    "Spatial False Alarm Rate (FAR)",
    r"Fragmentation Density $D_{\mathrm{frag}}$",
]
ylabels = ["Conditional FNR", "Spatial FAR", r"$D_{\mathrm{frag}} = N_{cc}/\mathrm{FAR}$"]
metrics = ["fnr", "far", "dfrag"]

methods = [
    ("pointwise", C_PT, "Pointwise CP"),
    ("emos99", C_EMOS, r"EMOS ($q_{99}$)"),
    ("gpd_noop", C_NOOP, "STACP GPD (No Ops)"),
    ("gpd_op", C_OP, "STACP GPD (+ Ops)"),
]

for r, (var, vlabel) in enumerate(VARS):
    for c, (metric, ctitle, ylab) in enumerate(zip(metrics, col_titles, ylabels)):
        ax = axes[r, c]
        if metric == "fnr":
            ax.axhspan(0, 0.10, color="#E8F5E9", alpha=0.55, zorder=0)
            ax.axhline(0.10, color="black", linestyle=":", linewidth=1.25,
                       label=(r"Risk Budget $\alpha_{\mathrm{risk}}=0.10$" if r == 0 else None), zorder=1)
        for dom, ls, mk in [("NA", "-", "o"), ("EU", "--", "s")]:
            for method, col, mlabel in methods:
                vals, lo, hi = load_series(var, dom, method, metric)
                lbl = f"{mlabel} ({dom})" if (r == 0 and c == 0) else None
                ax.plot(LEADS, vals, color=col, linestyle=ls, marker=mk,
                        linewidth=1.65, markersize=4.8, label=lbl, zorder=3)
                # Shaded 95% bootstrap CI on NA curves (and EU for STACP/EMOS on FNR) to avoid clutter
                if metric in ("fnr", "far") and not np.isnan(lo).any():
                    if (dom == "NA" and method in ("pointwise", "emos99", "gpd_op")) or (
                        metric == "fnr" and dom == "EU" and method == "gpd_op"
                    ):
                        ax.fill_between(LEADS, lo, hi, color=col, alpha=0.14, linewidth=0, zorder=2)

        if metric == "fnr":
            ax.set_ylim(0, 0.47 if var == "t2m" else 0.225)
            # Clean 168h endpoint annotations
            if var == "t2m":
                ax.annotate("EU 0.402", xy=(168, 0.402), xytext=(142, 0.425),
                            fontsize=6.8, fontweight="bold", color=C_PT)
                ax.annotate("EMOS EU 0.143\n(NA 0.112)", xy=(168, 0.1425), xytext=(116, 0.195),
                            fontsize=6.5, fontweight="bold", color=C_EMOS,
                            arrowprops=dict(arrowstyle="->", color=C_EMOS, lw=0.8))
                ax.annotate("STACP\n0.043–0.046", xy=(168, 0.046), xytext=(126, 0.008),
                            fontsize=6.5, fontweight="bold", color=C_OP,
                            arrowprops=dict(arrowstyle="->", color=C_OP, lw=0.8))
            else:
                ax.annotate("Pt. CP 0.162", xy=(168, 0.162), xytext=(132, 0.185),
                            fontsize=6.8, fontweight="bold", color=C_PT)
                ax.annotate("STACP 0.030–0.035", xy=(168, 0.035), xytext=(114, 0.060),
                            fontsize=6.5, fontweight="bold", color=C_OP,
                            arrowprops=dict(arrowstyle="->", color=C_OP, lw=0.8))
        elif metric == "far":
            ax.set_ylim(0, 1.0)
        else:
            ax.set_ylim(bottom=0)

        ax.set_xticks(LEADS)
        ax.set_xticklabels([f"{ld} h\n({ld//24} d)" for ld in LEADS])
        ax.set_xlim(64, 176)
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)
        ax.set_ylabel(ylab)
        if r == 0:
            ax.set_title(f"{panel_labels[r][c]} {ctitle}", fontweight="bold", pad=6)
        else:
            ax.set_title(f"{panel_labels[r][c]} {vlabel}: {ctitle}", fontweight="bold", pad=6)
            ax.set_xlabel("Forecast Lead Time")
        if c == 0:
            ax.text(0.03, 0.94, vlabel, transform=ax.transAxes,
                    fontsize=8.5, fontweight="bold", va="top",
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.65", alpha=0.92))

handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.08),
           ncol=5, frameon=True, edgecolor="0.7", columnspacing=1.0, handlelength=2.2)

out_repo = ROOT / "figures" / "Fig7-MultiLead-CrossRegion.pdf"
out_latex = Path(__file__).resolve().parents[2] / "Latex_sources" / "Fig7-MultiLead-CrossRegion.pdf"
out_repo.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_repo, bbox_inches="tight")
fig.savefig(out_latex, bbox_inches="tight")
print("Saved enriched Figure 7 to:", out_repo, "and", out_latex)
