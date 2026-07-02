# Research grounding & design decisions

The reconstruction is **depth-supervised 3D Gaussian Splatting**, trained from
scratch on [`gsplat`](https://github.com/nerfstudio-project/gsplat). Every design
choice below is tied to a published method appropriate for this dataset's regime:
20 scan-points clustered in a small bubble (single-vantage, sparse-view), with
people walking through 67% of the 240 views (dynamic scene).

> **Methodology caveat that applies to every PSNR/SSIM figure in this document.**
> All numbers are **training-view (in-sample)** metrics — every one of the 240
> views was used in training; none were held out. This is the single largest
> methodological gap in the project. See "Known limitations" below for the
> concrete held-out-split plan that would close it.

## 1. Final specification

**Reconstruction.** `outputs/full_v5_gradedge.pt` — 2,279,744 Gaussians after
30,000 iterations, SH degree 3, trained with:

```
python train.py --dataset ../dataset --init-ply ../prepared/init_pointcloud.ply \
  --out ../outputs/full_v5_gradedge.pt --iters 30000 --sh-degree 3 --max-init-points 1000000 \
  --depth-lambda 0.2 --aniso-lambda 0.02 --size-lambda 0.05 \
  --refine-poses --pose-warmup 2000 --conf-gamma 1.5 \
  --conf-reg-gamma 2.0 --reg-conf-boost 4.0 --grad-edge-scale 8.0
```

Loss: `L = (1-λ)·L1 + λ·D-SSIM + λ_d·L_depth + λ_aniso·L_aniso + λ_size·L_size`,
masked by the inverse person mask throughout. `L_depth` is a confidence- and
edge-weighted Pearson correlation against DA3 depth (§2.2). `L_aniso`/`L_size`
are confidence-adaptive anti-needle regularization (§2.4). A small per-camera
SE3 pose correction is learned jointly (§2.5).

**Deployment.** Opacity/scale/anisotropy-pruned to 1,200,000 real Gaussians
(`op≥0.02, max_scale≤0.5m, aniso≤12`), plus 146,385 synthetic floor-fill
Gaussians (§3.3) — **1,346,385 total** in the shipped `scene.ksplat` (~97MB).
Floater-pruning by view-provenance is implemented (`prune_floaters()` in
`export.py`) but **disabled by default** pending a fix (§4.3) — the fixed-radius
version regressed on live testing and was reverted.

**Measured quality** (masked training-view PSNR, views 0/6/30/120):
**21.34 dB**, up from an unregularized-depth-weighting baseline of 20.83 dB.
Needle-shaped Gaussians (anisotropy ratio >10): 0% (was ~40% before anti-needle
regularization).

**Viewer** (`viewer/`, three.js + `@mkkellogg/gaussian-splats-3d`):
- Real-time photographic splat rendering, first-person WASD + mouse-look.
- Three trust-aware render modes — confidence heat-map, coverage (view-count,
  with a continuous-fade threshold slider), colour-by-supervising-view.
- Floor-plan minimap: a real top-down density projection with a clickable,
  locally-adaptive walkable path.
- Bidirectional click-to-inspect: click a photo pixel to ray-cast its DA3 depth
  and drop a 3D marker; click a point in the 3D reconstruction to see which of
  the 240 source photos supervised it.
- Generatively-completed floor (§3.3), baked as real Gaussians, tagged
  unsupervised in the trust data.
- Standalone raw-splat viewer (`viewer/raw.html`) for opening any exported
  `.ply`/`.ksplat` independent of the full interactive scene.
- Confidence-gated enhancement of window/reflection regions via image diffusion
  (§4.4, `outputs/`/`src/flux_enhance.py`) — an offline comparison tool, not a
  live viewer mode.

## 2. Method grounding

### 2.1 Why 3DGS, and why depth-supervised

- **3D Gaussian Splatting.** Kerbl et al., *SIGGRAPH 2023*, https://arxiv.org/abs/2308.04079.
  Rasterizes in real time and exports to a format that renders directly in the
  browser via WebGL — the deliverable is an interactive viewer, which ray-marched
  NeRF is not well suited for.
- **Depth supervision.** Deng et al., *DS-NeRF*, CVPR 2022,
  https://arxiv.org/abs/2107.02791 — depth priors let radiance fields train from
  fewer views and converge faster. A depth map is available per view (DA3), so
  it is used.
- **Depth regularization under sparse coverage.** DNGaussian (Li et al., CVPR 2024,
  https://arxiv.org/abs/2403.06912) and Chung et al. (CVPRW 2024,
  https://arxiv.org/abs/2311.13398) show unconstrained 3DGS overfits into
  floaters in the few-shot regime this dataset sits in, and that depth
  regularization fixes it. `init_pointcloud.py`'s depth-lifted initialization
  follows Chung et al.'s approach directly.

### 2.2 Depth loss: scale-invariant, not raw L1

DA3 depth is monocular-estimated: correct up to an unknown per-view scale/shift,
not hardware-sensor ground truth. Naive L1 regression would bake that error into
geometry. Primary term is a **patch-based Pearson correlation** (DN-Splatter,
Turkulainen et al., WACV 2025, https://arxiv.org/abs/2403.17822):
`L = 1 − corr(D_render, D_mono)`, scale/shift invariant by construction, at
`λ_d ≈ 0.2`.

Two extensions weight the per-pixel depth residual beyond the paper's baseline:

- **Confidence weighting.** DA3 ships a per-pixel confidence channel most
  depth-supervision papers don't have; the residual is weighted by
  `conf^γ` (`--conf-gamma`), so low-confidence pixels contribute less.
- **Edge weighting.** DN-Splatter's gradient-aware term
  (`exp(−∇I)`, its Eq. 4) downweights depth supervision at RGB edges, where
  monocular depth is least reliable (occlusion boundaries, depth discontinuities).
  Implemented as `image_gradient_weight()` in `train.py`:
  `exp(-8·|∇I|)`, computed from the training image's luminance gradient
  (`--grad-edge-scale`, 0 disables).

  **Measured effect.** Global masked training-view PSNR: 20.75 → 21.34 dB
  (+0.59 dB) over the confidence-adaptive baseline without edge weighting.
  Split by RGB edge strength (strong-edge vs. flat pixels, 6 views): edge-pixel
  PSNR improves 14.16 → 14.77 dB (+0.61 dB), with a small give-back in flat
  regions (26.63 → 26.41 dB, −0.22 dB) — concentrated exactly where the
  mechanism targets, and a favorable net trade. See `docs/eval_final.png` for
  the corresponding render panel.

### 2.3 Photometric loss

Standard 3DGS appearance loss: `L_rgb = (1−λ)·L1 + λ·D-SSIM`, `λ=0.2` (Kerbl
2023). Both photometric and depth terms are multiplied by the inverse person
mask, so people contribute zero gradient — surfaces occluded by a person in one
view are learned from the other views where they're visible.

### 2.4 Anti-needle regularization

Unconstrained 3DGS under this dataset's sparse coverage initially produced
highly anisotropic "needle" Gaussians — thin slivers that satisfy one training
view's photometric loss edge-on while reading as visual spikes from any other
angle (measured: ~40% of Gaussians with anisotropy ratio >10, some exceeding
1000×). Fixed with a direct penalty,
`λ_aniso·relu(max_scale/min_scale − 6) + λ_size·relu(max_scale − 0.25m)`
(`λ_aniso=0.02, λ_size=0.05`), plus gsplat's `antialiased` rasterization mode.
Result: needle fraction 40% → 0%. Not from a specific paper — the same
philosophy as scale/opacity pruning in the broader 3DGS compaction literature
(Mip-Splatting, Compact-3DGS).

**Confidence-adaptive extension.** The penalty above is flat: every Gaussian
gets the same cap regardless of how much the underlying depth is trusted. Since
gsplat's packed rasterization mode already returns the global index and
projected pixel of every Gaussian visible in the current training view
(`info["gaussian_ids"]`, `info["means2d"]`, no extra render pass), DA3
confidence can be sampled at each visible Gaussian's own pixel and used to scale
its regularization pressure:

```
distrust    = (1 − conf)^γ            # γ = --conf-reg-gamma, default 2.0
reg_weight  = 1 + boost · distrust     # boost = --reg-conf-boost, default 4.0
```

Confident regions keep `reg_weight ≈ 1` (unchanged pressure); regions DA3 was
unsure about — window glass, reflections, oblique surfaces — get up to 5× the
pressure, pushed harder toward small, round geometry. `--reg-conf-boost 0`
recovers the flat penalty exactly.

**Measured effect.** Global training-view PSNR is flat (20.83 → 20.75 dB,
within run-to-run noise). Split by DA3 pixel confidence (low <0.3, high ≥0.7,
6 views), the effect is directional and consistent with the mechanism: fit
degrades slightly in low-confidence regions (16.54 → 16.08 dB) and improves
slightly in high-confidence regions (19.68 → 20.04 dB) — less overfitting to
distrusted supervision, more capacity where the supervision is trustworthy. Not
visible in the global average; only shows up conditioned on confidence.

### 2.5 Pose refinement

`--refine-poses` learns a small per-camera SE3 correction (axis-angle rotation +
translation) on top of DA3's given poses, with a 2000-iteration warm-up (letting
Gaussian geometry settle before a second set of free parameters joins the
optimization) and an L2 penalty keeping corrections small. Final magnitudes:
~1.2° rotation, ~6.2cm translation on average — plausible calibration-error
scale. Standard bundle-adjustment-style pose refinement; not from a specific
paper.

**Measured PSNR gain from this correction is confounded with training fit.** A
per-camera correction is fit to that camera's own training image and has no
learned delta for a genuinely novel pose — so its measured benefit is partly
(perhaps mostly) the model getting to reproduce a training view with its own
camera also free to move, not evidence of better underlying geometry. The
held-out split in "Known limitations" is what would isolate the two effects.

### 2.6 Multi-view consistency filtering (initialization)

`init_pointcloud.py --consistency-check` drops a depth-lifted point if it
disagrees, beyond a relative tolerance, with an adjacent-yaw view of the same
scan-point. 3.7% of points dropped. **Scope note:** all 12 yaws at one
scan-point share the same camera position (zero baseline) — this is a
monocular self-consistency check across two crops of the same 360° capture, not
stereo triangulation. It catches a real class of monocular-depth outlier
(cheaply, using views already on disk) but cannot catch errors the depth model
makes consistently across both crops.

## 3. Alternatives explored and not shipped

### 3.1 2DGS

2D Gaussian Splatting (Huang et al., SIGGRAPH 2024,
https://arxiv.org/abs/2403.17888) represents primitives as flat, surface-aligned
disks with normal-consistency and distortion regularizers reported to produce
cleaner indoor walls/floors than 3DGS. Tried (`src/train_2dgs.py`) on the
hypothesis that flat surfels would help the floor/ceiling regions seen only at
grazing angles.

- First attempt (`dist_lambda=100` from iteration 0) collapsed: ~10.6dB masked
  PSNR, blocky depth. The distortion loss is too aggressive for this sparse,
  single-vantage capture.
- Second attempt (`dist_lambda=1`, delayed to iteration 7000) fixed the
  collapse (18.4dB) but stayed ~3dB behind the regularized 3DGS baseline (21.4dB,
  same era) despite 2.6× more primitives (6.34M vs 2.43M), with localized
  orientation-noise artifacts concentrated around windows and reflective
  surfaces.

**Conclusion:** 2DGS's surface regularizers assume denser multi-view coverage
than a single-vantage, 20-scan-point capture provides. Not adopted. See
`docs/2dgs_vs_3dgs.png`. Both checkpoints retained (`outputs/full_2dgs.pt`,
`outputs/full_2dgs_v2.pt`).

### 3.2 Floater pruning: a regression, reverted

`compute_provenance()` (`export.py`) already tags every Gaussian with
`n_views` — how many of the 240 views confirmed it via depth-consistency — for
the trust-aware viewer modes. That signal was used only for visualization, never
to prune the exported model: 15-17% of Gaussians across every trained version
had `n_views==0` at a mean opacity of ~0.55-0.59 (visible, not near-transparent),
reported during testing as visible floating artifacts.

Pruning every `n_views==0` Gaussian is wrong: the median unconfirmed point sits
7.6cm from confirmed geometry — legitimate detail that missed a strict
depth-consistency tolerance, not a floater. `prune_floaters()` requires both
signals — never confirmed **and** farther than a fixed 15cm from the nearest
confirmed point — before dropping a Gaussian.

**This regressed on deployment.** A fixed absolute distance threshold doesn't
account for point density thinning naturally with distance from the scan-point
cluster: pruning was 10.3% near the cameras but 61.3% at 4-6m out, gutting
sparse-but-legitimate far geometry (reported as the far wall going black).
Reverted immediately (`--floater-dist 0`, now also the code default). A
density-relative version (isolation distance compared against each point's
local neighbor spacing, not a global constant) closes the imbalance in a
rendered comparison (`docs/floater_pruning_compare.png`) with one minor,
localized new artifact — implemented but **not deployed**, pending a validation
pass beyond a single rendered comparison. Current shipped state:
floater-pruning disabled.

### 3.3 Floor/ceiling completion

Single-pitch capture means the floor directly underfoot and the ceiling
directly overhead are never in any image — zero observations, not a training
deficiency. Three designs, in order:

1. Flat colored plane (height from point-cloud percentiles, color from nearest
   real points) — geometrically honest but read as an obviously separate,
   disconnected surface.
2. Classical-inpainting-textured plane (top-down projection, OpenCV Telea
   inpainting into the holes, no generative model) — more textured, still read
   as a separate flat plane, with streaky extrapolation at hole boundaries.
3. **Shipped:** generative fill baked directly into the Gaussian model. Only
   unobserved floor pixels (78.4% of the floor's top-down footprint) are
   completed via SDXL-inpainting (mask-conditioned), then every hole pixel
   becomes an individual small, flat, floor-aligned Gaussian merged into the
   exported set (146k synthetic alongside 1.2M real) — not a separate mesh.
   Ceiling is not filled (judged too irregular for the same top-down-projection
   approach). Synthetic Gaussians are tagged `confidence=0, n_views=0` in the
   trust data, so Coverage/Confidence mode correctly shows this region as
   generated, not observed.

### 3.4 Window/reflection enhancement

Windows, mirrors, and reflective surfaces reconstruct as soft, low-detail
regions — DA3 confidence is low there, and single-vantage capture gives no
cross-view signal to resolve them. A confidence-gated image-enhancement pass
(`src/artifixer_enhance.py`, `src/flux_enhance.py`) renders a view, runs it
through an image-diffusion model as img2img, and blends the result against the
original render using DA3 confidence as a per-pixel gate.

This is **deterministic conditioning on an existing render** (the Difix3D+
family, Wang et al., CVPR 2025 Oral, https://arxiv.org/abs/2503.01774) — not
NVIDIA ArtiFixer's (SIGGRAPH 2026, https://arxiv.org/abs/2603.00492) **opacity
mixing**, which intervenes at the noise-initialization stage of denoising
(starting from the render where geometry exists, from noise where it doesn't)
specifically so it can generate content in genuinely empty regions. ArtiFixer
requires ~80GB VRAM and ~34GB of weights, infeasible on this instance's 24GB
card; there is no smaller variant. The distinction has a concrete symptom: an
early ungated full-frame repaint hallucinated walls/ceiling detail that was
never there — exactly the empty-region failure opacity mixing exists to
prevent. The confidence/coverage channels already computed for the trust-aware
viewer are the right control signal for an opacity-mixing-style approach with a
model that supports it.

**Enhancer:** initially SD-Turbo/SDXL, later swapped for GGUF-quantized
Flux.1-schnell (transformer Q4_K_S 6.8GB + T5-XXL Q3_K_S 2.1GB, both ungated
community re-quantizations — ~9.5GB total vs. the ~34GB the full-precision
weights would need), run in an isolated Python environment (`.venv-flux`)
since modern `diffusers`' GGUF loading requires torch ≥2.4 and `gsplat`'s
prebuilt wheel requires torch 2.1.2 — the two cannot share a venv. The two
stages hand off through the filesystem: `render_for_enhance.py` (gsplat venv)
renders and caches views; `flux_enhance.py` (`.venv-flux`) runs the diffusion
step. `from_single_file`'s default architecture-config source is the gated
original repo; both the transformer and VAE configs are instead supplied
locally (`configs/flux_schnell_transformer_config/`,
`configs/flux_schnell_vae_config/`) — public, weight-free architecture metadata,
not a workaround for the gating itself.

**Result** (`docs/flux_enhancement.png`, 4-step schnell inference,
`guidance_scale=0`): visibly sharper, higher-frequency detail in window regions
than the SDXL version. Same caveat, more visible with a stronger model: the
enhancement is plausible and confident, not accurate — it fabricates storefront
signage text that doesn't match the ground truth. A stronger model makes the
hallucination more convincing, not more correct, which is exactly the failure
mode opacity mixing (trained end-to-end with 3D consistency, not a post-hoc 2D
blend) is designed to avoid. This remains an offline comparison tool
(`outputs/artifixer_flux.png`-style panels), not a live viewer mode.

## 4. Known limitations

- **All PSNR/SSIM figures are training-view, not held-out.** Every one of the
  240 views was used in training. The fix: exclude every 12th view from
  training (matching the `--eval-mode interval --eval-interval 12` convention
  used elsewhere on this dataset, giving 20 held-out views spread across all 20
  scan-points), train identically otherwise, and re-run each comparison
  PSNR-on-held-out. Expected direction of change: the anti-needle result should
  hold or strengthen (needles overfitting individual training views should
  generalize worse, not better); pose refinement's measured gain should shrink
  substantially, since a per-camera correction has no learned delta for a
  held-out pose (isolable by training pose corrections only on non-held-out
  views and checking whether held-out renders, using their original poses,
  still improve over a no-refinement baseline).
- **Windows, mirrors, and reflective surfaces** reconstruct as soft, low-detail
  regions. Person masks correctly exclude people but not their shadows or
  reflections, which are view-inconsistent but unmasked and reconstruct as
  static texture or floaters. Addressed partially via the confidence-gated
  enhancement pass (§3.4); the reconstruction itself is unaffected, since this
  is a display-time correction, not a retrain.
- **Floor/ceiling beyond the coverage bubble** (the ~9×6×11m region the 20
  scan-points actually constrain, out of the theoretical 12×20m recorded floor
  plane) are geometrically unconstrained by definition. The floor is
  generatively completed (§3.3, tagged unsupervised); the ceiling is not filled.
- **A specific undetected person.** A person is reconstructed and visible from
  multiple angles at yaw 240 across every scan-point despite masking.
  Root-caused by backprojecting the offending rendered pixel to its 3D point,
  looking up which source view supervised it (the same per-Gaussian lookup
  built for the trust-aware viewer), and inspecting that photo directly: the
  upstream YOLOv8m-seg detector supplied with the dataset missed this one
  detection outright (`views/000003_pz000_y300_normal.jpg`, 0% mask coverage)
  — not a flaw in this project's masking logic, a miss in the provided labels,
  which are used as given (re-running detection is out of scope, matching the
  brief's stance on not re-running DA3). A second-opinion detector or a
  per-scan-point temporal-consistency check across the 12 yaws would likely
  catch this class of miss.
- **Floater-pruning is implemented but disabled** (§3.2) — the fixed-threshold
  version regressed on deployment; a density-adaptive version exists but is
  unvalidated beyond a single rendered comparison.

## 5. References

| Paper | Venue | Used for |
|---|---|---|
| Kerbl et al., 3D Gaussian Splatting | SIGGRAPH 2023 | base method |
| Deng et al., Depth-supervised NeRF | CVPR 2022 | depth supervision |
| Li et al., DNGaussian | CVPR 2024 | sparse-view depth regularization |
| Chung et al., Depth-Regularized 3DGS | CVPRW 2024 | depth-lifted initialization |
| Turkulainen et al., DN-Splatter | WACV 2025 | Pearson + gradient-aware depth loss |
| Huang et al., 2D Gaussian Splatting | SIGGRAPH 2024 | explored, not adopted (§3.1) |
| Sabour et al., SpotLessSplats | ACM TOG 2025 | cited context for unmasked distractors |
| Wang et al., Difix3D+ | CVPR 2025 Oral | correct reference class for §3.4 |
| NVIDIA, ArtiFixer | SIGGRAPH 2026 | opacity mixing; infeasible at this scale |
| Kocasari (TUM), Higher-Frequency Geometry Recovery | open thesis proposal | motivated §2.2's edge-weighting revisit |
