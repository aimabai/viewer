"""Render the project pipeline figure: docs/pipeline.png.

Depth-supervised 3D Gaussian Splatting — from DA3 output to a browser walk-through.
Run:  python docs/make_figure.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# palette
INK = "#1f2933"
STAGE = ["#e8f0fe", "#e6f4ea", "#fef3e0", "#f3e8fd"]      # fills
EDGE = ["#4285f4", "#34a853", "#fbbc04", "#a142f4"]       # borders
CARD = "#ffffff"

fig, ax = plt.subplots(figsize=(16, 8.6))
ax.set_xlim(0, 16); ax.set_ylim(0, 8.6); ax.axis("off")

ax.text(8, 8.25, "Depth-Supervised 3D Gaussian Splatting",
        ha="center", va="center", fontsize=20, fontweight="bold", color=INK)
ax.text(8, 7.78, "InfraScan DA3 output  →  3D reconstruction  →  browser walk-through viewer",
        ha="center", va="center", fontsize=12.5, color="#5f6b7a")

stages = [
    ("1 · INPUT", "DA3 dataset (given)", [
        "240 perspective views",
        "20 scan-points x 12 yaws",
        "per view: RGB - metric depth",
        "          conf - person_mask",
        "camera poses (OpenCV c2w) + K",
    ]),
    ("2 · PREPROCESS", "src/data.py - init_pointcloud.py", [
        "sanity check (README s4)",
        "inverse person mask",
        "  people -> 0 loss weight",
        "backproject depth -> world",
        "conf-filter + voxel ds",
        "= init cloud (1.52M pts)",
    ]),
    ("3 · RECONSTRUCT", "src/train.py  (gsplat)", [
        "Gaussians init from cloud",
        "differentiable rasterizer",
        "masked photometric loss",
        "+ Pearson depth loss",
        "  (scale-invariant)",
        "-> exports/splat.ply",
    ]),
    ("4 · VIEW", "viewer/  (three.js + WebGL)", [
        "real-time splat render",
        "20 camera frustums",
        "  'stood here / saw this'",
        "click view -> backproject",
        "  depth -> drop 3D marker",
        "coverage / 'no data' cue",
    ]),
]

x0, w, gap = 0.5, 3.45, 0.42
ytop, ybot = 7.15, 2.35
for i, (tag, sub, items) in enumerate(stages):
    x = x0 + i * (w + gap)
    box = FancyBboxPatch((x, ybot), w, ytop - ybot,
                         boxstyle="round,pad=0.04,rounding_size=0.16",
                         linewidth=2.2, edgecolor=EDGE[i], facecolor=STAGE[i], zorder=2)
    ax.add_patch(box)
    ax.text(x + w / 2, ytop - 0.33, tag, ha="center", va="center",
            fontsize=13.5, fontweight="bold", color=EDGE[i], zorder=3)
    ax.text(x + w / 2, ytop - 0.72, sub, ha="center", va="center",
            fontsize=9.3, style="italic", color="#5f6b7a", zorder=3)
    # inner card with bullet lines
    card = FancyBboxPatch((x + 0.18, ybot + 0.22), w - 0.36, (ytop - ybot) - 1.35,
                          boxstyle="round,pad=0.02,rounding_size=0.08",
                          linewidth=0, facecolor=CARD, zorder=3)
    ax.add_patch(card)
    yline = ytop - 1.28
    for it in items:
        lead = "" if it.startswith("  ") else "- "
        ax.text(x + 0.34, yline, lead + it.strip() if lead else "   " + it.strip(),
                ha="left", va="center", fontsize=9.6,
                color=INK if lead else "#7a8694", zorder=4)
        yline -= 0.46
    # arrow to next stage
    if i < len(stages) - 1:
        ax.add_patch(FancyArrowPatch(
            (x + w + 0.03, (ytop + ybot) / 2), (x + w + gap - 0.03, (ytop + ybot) / 2),
            arrowstyle="-|>", mutation_scale=22, linewidth=2.4,
            color="#9aa5b1", zorder=5))

# ---- loss banner ----
ax.add_patch(FancyBboxPatch((0.5, 1.05), 15.0, 0.95,
             boxstyle="round,pad=0.03,rounding_size=0.12",
             linewidth=1.6, edgecolor="#cbd2d9", facecolor="#f7f9fb", zorder=2))
ax.text(0.85, 1.72, "Training objective", fontsize=10.5, fontweight="bold", color=INK)
ax.text(0.85, 1.30,
        r"$L \;=\; [\,0.8\,L_1 + 0.2\,L_{D\mathrm{-}SSIM}\,]_{\;\mathrm{photometric}}"
        r"\;\;+\;\; \lambda_d \cdot [\,1-\mathrm{corr}(D_{\mathrm{render}},\,D_{\mathrm{DA3}})\,]_{\;\mathrm{Pearson\;depth}}$",
        fontsize=12.5, color=INK)
ax.text(11.7, 1.30,
        u"every term × inverse person-mask\n"
        u"depth is scale-invariant;  λ_d ≈ 0.2",
        fontsize=9.2, color="#5f6b7a", va="center")

# ---- papers footer ----
ax.text(8, 0.55,
        "Grounded in:  3DGS (SIGGRAPH'23) · DS-NeRF (CVPR'22) · DNGaussian (CVPR'24) · "
        "Depth-Reg GS (CVPRW'24) · DN-Splatter (WACV'25) · "
        "RobustNeRF/SpotLessSplats (CVPR'23/TOG'25 — shadows & reflections gap)",
        ha="center", va="center", fontsize=8.6, color="#7a8694")

out = Path(__file__).resolve().parent / "pipeline.png"
fig.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
print("wrote", out)
