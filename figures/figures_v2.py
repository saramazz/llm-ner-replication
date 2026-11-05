"""
Revised Figure 2 -- blue-scale palette.

Fixes vs. previous version:
- Panel (a): F1 value labels placed above CI whisker top (not just bar top)
- Panel (b): legend moved outside axes (right side) to avoid bar overlap
- All panels: consistent dark/mid/light blue color scheme for print-friendly layout

Run:  MPLBACKEND=Agg .venv/bin/python figures_v2.py
"""
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Locked numbers (overlap, M2 scorer) ------------------------------------
NER = {
    "English": dict(P=0.834, R=0.463, F1=0.596, ci=(0.571, 0.618)),  # 0.596 so %.2f shows 0.60
    "Dutch":   dict(P=0.732, R=0.557, F1=0.633, ci=(0.585, 0.675)),
    "Italian": dict(P=0.754, R=0.574, F1=0.652, ci=(0.608, 0.696)),
}
# (restricted, realistic) accuracy@1, results/results_bel_canonical_131.json
BEL_EN = {
    "all-MiniLM": (0.949, 0.708),
    "BioBERT":    (0.950, 0.269),
    "mSapBERT":   (0.949, 0.708),
}
BEL_CEILING_EN = 0.859
BASELINES = [
    ("GPT-4o",              0.596),
    ("QuickUMLS (0.70)", 0.524),
    ("QuickUMLS (0.80)", 0.641),
    ("MedCAT",              0.629),
    ("GLiNER (0.5)", 0.223),   # prompt labels, default threshold: results_gliner_C_gpt4o_window_english.json
    ("GLiNER (0.1, tuned)", 0.687),  # generic labels, threshold chosen on train: results_gliner_D_published_window_thr0.1.json
]
SUPERVISED_REF = 0.90  # supervised SemEval-2014 Task 7, best overlap F1

# ---- Blue-scale palette ------------------------------------------------------
DARK  = "#1b4d8e"   # deep navy  -- Precision / Oracle / GPT-4o highlight
MID   = "#4a90d4"   # sky blue   -- Recall
LIGHT = "#9ecae1"   # pale blue  -- F1 / Realistic / other baselines
NAVY  = "#0d2855"   # very dark  -- supervised reference line
GREY  = "#7f9faf"   # slate grey -- coverage ceiling line

# ---- Style -------------------------------------------------------------------
rcParams.update({
    "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 300,
})


def _labels(ax, bars, fmt="%.2f", dy=0.012, err_top=None):
    """Value labels above bars; if err_top supplied, labels clear the CI whisker."""
    for i, b in enumerate(bars):
        h = b.get_height()
        top = h + (err_top[i] if err_top is not None else 0.0)
        ax.text(b.get_x() + b.get_width() / 2, top + dy, fmt % h,
                ha="center", va="bottom", fontsize=10)


fig, (axa, axb, axc) = plt.subplots(1, 3, figsize=(17, 5.2))

# ---- Panel (a): NER P / R / F1 by language ----------------------------------
langs  = list(NER)
xi     = list(range(len(langs)))
w      = 0.26
P_v    = [NER[l]["P"]  for l in langs]
R_v    = [NER[l]["R"]  for l in langs]
F1_v   = [NER[l]["F1"] for l in langs]
err_lo = [NER[l]["F1"] - NER[l]["ci"][0] for l in langs]
err_hi = [NER[l]["ci"][1] - NER[l]["F1"] for l in langs]

b1 = axa.bar([i - w for i in xi], P_v,  w, label="Precision",   color=DARK)
b2 = axa.bar(xi,                  R_v,  w, label="Recall",      color=MID)
b3 = axa.bar([i + w for i in xi], F1_v, w, label="F1 (95% CI)",
             yerr=[err_lo, err_hi], capsize=4, color=LIGHT,
             error_kw=dict(ecolor=DARK, lw=1.5, capthick=1.5))

_labels(axa, b1, dy=0.012)
_labels(axa, b2, dy=0.012)
_labels(axa, b3, dy=0.016, err_top=err_hi)  # clears CI whisker cap

axa.set_xticks(xi); axa.set_xticklabels(langs)
axa.set_ylim(0, 1.04)
axa.set_ylabel("Score")
axa.set_title("(a) NER agreement (overlap)")
axa.legend(loc="upper right", frameon=False)

# ---- Panel (b): BEL oracle vs realistic by encoder (English) ----------------
encs = list(BEL_EN)
xb   = list(range(len(encs)))
wb   = 0.36
orac = [BEL_EN[e][0] for e in encs]
real = [BEL_EN[e][1] for e in encs]

b4 = axb.bar([i - wb / 2 for i in xb], orac, wb, label="Restricted",     color=DARK)
b5 = axb.bar([i + wb / 2 for i in xb], real, wb, label="Realistic UMLS", color=LIGHT)
_labels(axb, b4)
_labels(axb, b5)

axb.axhline(BEL_CEILING_EN, ls="--", lw=1.4, color=GREY,
            label=f"Coverage ceiling ({BEL_CEILING_EN:.2f})")
axb.set_xticks(xb); axb.set_xticklabels(encs)
axb.set_ylim(0, 1.04)
axb.set_ylabel("Accuracy@1 (English)")
axb.set_title("(b) Entity linking: restricted vs realistic")

# Legend outside axes (right side) -- avoids overlap with tall oracle bars
axb.legend(bbox_to_anchor=(1.02, 0.5), loc="center left",
           frameon=True, framealpha=0.92, fontsize=10, borderpad=0.7)

# ---- Panel (c): English NER baseline comparison -----------------------------
names = [n for n, _ in BASELINES]
vals  = [v for _, v in BASELINES]
cols  = [DARK if n == "GPT-4o" else LIGHT for n in names]

bars = axc.bar(range(len(names)), vals, color=cols, width=0.62,
               edgecolor="white", linewidth=0.5)
_labels(axc, bars)

axc.axhline(SUPERVISED_REF, ls="--", lw=1.4, color=NAVY,
            label=f"Supervised ref. (overlap {SUPERVISED_REF:.2f})")
axc.set_xticks(range(len(names)))
axc.set_xticklabels(names, fontsize=10, rotation=45, ha="right", rotation_mode="anchor")  # no overlap (R1)
axc.set_ylim(0, 1.04)
axc.set_ylabel("Overlap F1 (English)")
axc.set_title("(c) NER vs baselines (English)")
axc.legend(loc="upper right", frameon=False)

plt.tight_layout()
for ext in ("svg", "png", "pdf"):
    plt.savefig(os.path.join(OUT_DIR, f"results.{ext}"),
                bbox_inches="tight", dpi=300)
# BMC does not accept SVG for submission; TIFF at 600 dpi is the high-resolution
# raster fallback to PDF for the separately-uploaded figure file.
plt.savefig(os.path.join(OUT_DIR, "results.tiff"), bbox_inches="tight", dpi=600)
plt.close()
print(f"Saved results.{{svg,png,pdf,tiff}} to {OUT_DIR}")
