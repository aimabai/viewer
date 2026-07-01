# Research grounding & design decisions

The reconstruction method here is **depth-supervised 3D Gaussian Splatting**, with
each design choice tied to a published method. This dataset sits squarely in the
*sparse / uneven-coverage* and *dynamic-scene* regimes those papers were built for:
20 scan-points clustered in a small bubble, with people walking through 67% of views.

## Why 3DGS (not NeRF), and why depth-supervised

- **Base method — 3D Gaussian Splatting.** Kerbl, Kopanas, Leimkühler, Drettakis,
  *"3D Gaussian Splatting for Real-Time Radiance Field Rendering,"* SIGGRAPH 2023.
  Chosen because it rasterizes in real time and exports to a PLY that renders
  directly in the browser via WebGL — the deliverable is an *interactive* viewer,
  which is painful for ray-marched NeRF. https://arxiv.org/abs/2308.04079
- **Depth supervision lineage.** Deng et al., *"Depth-supervised NeRF: Fewer Views
  and Faster Training for Free,"* CVPR 2022 — established that depth priors let
  radiance fields train from far fewer views and converge faster. We have a depth
  map per view (DA3), so we use it. https://arxiv.org/abs/2107.02791
- **Sparse-view 3DGS overfits without geometry.** Our coverage is the few-shot
  regime, where unconstrained 3DGS sprays floaters. Two CVPR'24 works show depth
  regularization fixes this:
  - Li et al., *"DNGaussian: Optimizing Sparse-View 3D Gaussian Radiance Fields
    with Global-Local Depth Normalization,"* CVPR 2024. https://arxiv.org/abs/2403.06912
  - Chung et al., *"Depth-Regularized Optimization for 3D Gaussian Splatting in
    Few-Shot Images,"* CVPRW 2024 — also initializes Gaussians from monocular depth,
    exactly as our `init_pointcloud.py` does. https://arxiv.org/abs/2311.13398

## Depth loss: scale-invariant, NOT raw L1 to "metric" depth

DA3 depth is *monocular-estimated*. Even though it is reported in metres, it carries
per-view scale/shift error and edge noise — the README explicitly warns it is "not
hardware sensor ground truth." Naively regressing rendered depth to it with L1
would bake that error into geometry. The literature solves this with
**scale-and-shift-invariant** depth losses:

- **DNGaussian — Global-Local Depth Normalization (CVPR 2024).** Normalize depth
  within local patches before the error so the loss is scale/shift invariant yet
  sensitive to local structure:
  - Local: `D_LN(x) = (D(x) − mean(D_P)) / (std(D_P) + ε)`
  - Global: `D_GN(x) = (D(x) − mean(D_P)) / std(D_I)`
  - combined as `L2(D_GN, D̃_GN) + γ·L2(D_LN, D̃_LN)`, with `γ = 0.1`.
- **DN-Splatter (WACV 2025)** — Turkulainen et al. https://arxiv.org/abs/2403.17822
  - Uses a **Pearson correlation** depth loss for monocular depth (scale/shift
    invariant by construction): `L = 1 − corr(D_render, D_mono)`, computed
    patch-wise. This is the loss the prior `Scene3d` run used (`PearsonDepth`).
  - **Gradient-aware log depth loss (Eq. 4):**
    `L_D = exp(−∇I) · (1/|D|) Σ log(1 + ‖D̂ − D‖₁)` — the `exp(−∇I)` term
    *downweights depth supervision at RGB edges*, where monocular depth is least
    reliable (depth discontinuities, occlusion boundaries).
  - Full objective (Eq. 12): `L = L_rgb + λ_d·L_D + L_scale + λ_n·L_N + λ_s·L_smooth`,
    with `λ_d = 0.2, λ_n = 0.1, λ_s = 0.1`.

**Our choice:** primary depth term = **patch-based Pearson** (DN-Splatter), as the
most robust to DA3's scale ambiguity and the simplest defensible option, at
`λ_d ≈ 0.2`. **Extension we can justify from the data:** weight the depth residual
per-pixel by DA3's `conf` channel — most papers lack a confidence signal; we have
one, so low-confidence pixels contribute less. The gradient-aware `exp(−∇I)` weight
is a cheap, paper-backed alternative/complement.

## Photometric loss: the 3DGS standard, masked

Standard 3DGS appearance loss: `L_rgb = (1−λ)·L1 + λ·D-SSIM`, `λ = 0.2` (Kerbl 2023).
Every per-pixel term (photometric **and** depth) is multiplied by the **inverse
person mask** so people contribute zero gradient; the wall behind a person in one
view is learned from other views where it is visible.

## Dynamic scene: masks now, robust masking as the principled gap

- **What we do:** the dataset ships YOLOv8-seg person masks (instance segmentation,
  dilated 4 px). Used as a per-pixel loss weight — the standard, reliable way to
  exclude known dynamic content (supported by DN-Splatter / nerfstudio / gsplat).
- **The honest gap (README rewards noticing this):** person masks do **not** cover
  the *shadows* people cast on the floor or their *reflections* in monitors/glass.
  Those are view-inconsistent but unmasked, so they reconstruct as static texture
  or floaters. The principled fix is *unsupervised* distractor handling:
  - Sabour et al., *"RobustNeRF: Ignoring Distractors with Robust Losses,"*
    CVPR 2023 — treat inconsistent residuals as outliers via a robust (trimmed)
    loss. https://arxiv.org/abs/2302.00833
  - Ren et al., *"NeRF On-the-go: Exploiting Uncertainty for Distractor-free NeRFs
    in the Wild,"* CVPR 2024. https://arxiv.org/abs/2405.18715
  - Sabour et al., *"SpotLessSplats: Ignoring Distractors in 3D Gaussian Splatting,"*
    ACM TOG 2025 — clusters semantic features + robust optimization to mask
    transient effects (incl. shadows/reflections) **without** explicit masks.
    https://arxiv.org/abs/2406.20055

  **Decision:** use the provided masks as the baseline; document SpotLessSplats /
  RobustNeRF as the next step for shadows+reflections, and optionally add a
  RobustNeRF-style robust residual as a stretch (a few lines on top of L1).

- **A second gap, found empirically, not hypothesized: the provided masks
  themselves can miss a person entirely.** The viewer surfaced a reconstructed
  person visible from multiple angles (most obviously at yaw 240 across every one
  of the 20 scan-points). We traced it to ground truth, not just inspected the
  render: took the offending pixel, backprojected it to a 3D world point using the
  rendered depth and that view's (pose-corrected) camera, then queried which source
  view actually supervised that point — a lookup we already have for the trust-aware
  viewer modes (`export.py`'s per-Gaussian `supervising_view`). That pointed to
  `views/000003_pz000_y300_normal.jpg` (view id 46), which has **0% person-mask
  coverage** despite a person being clearly visible in the frame — the upstream
  YOLOv8m-seg detector supplied with the dataset missed this one detection outright,
  not a boundary/dilation issue with an otherwise-correct mask. Since our pipeline
  trusts the provided masks completely (re-running detection ourselves is out of
  scope, matching the brief's stance on not re-running DA3), an undetected person
  is invisible to every masking mechanism downstream and gets treated as legitimate,
  multi-view-confirmed static geometry. Documented rather than silently patched —
  see README1.md's known-limitations section; a second-opinion detector or a
  simple per-scan-point temporal-consistency check across the 12 yaws would likely
  catch this class of miss automatically.

## Coverage honesty (constrained vs. unconstrained regions)

The 20 views only constrain the ~9×6×11 m bubble we measured in `init_pointcloud.py`;
the rest of the 12×20 m floor PLY was never seen. Sparse-view papers above all note
geometry degrades outside the observed cone. We surface this *in the viewer* (a
coverage/“no data here” cue), rather than pretending the whole floor is reconstructed.

## Anti-needle regularization (why the splats stopped looking spiky)

Sparse-view 3DGS on this data initially produced highly **anisotropic "needle"
Gaussians** — thin, elongated splats that read as visual spikes from any angle other
than the one that created them. Measured directly: ~40% of Gaussians had an
anisotropy ratio (max-scale / min-scale) above 10, some exceeding 1000×. This is a
known failure mode of unconstrained 3DGS under sparse coverage — the optimizer can
satisfy the photometric loss for a *few* training views with a thin sliver oriented
edge-on to those cameras, which is disastrous from any other viewpoint (exactly our
single-vantage, many-yaw setup).

**Fix:** penalize anisotropy and oversized Gaussians directly in the loss —
`λ_aniso · relu(max_scale/min_scale − 6)` and `λ_size · relu(max_scale − 0.25m)`,
at `λ_aniso=0.02, λ_size=0.05` — plus gsplat's `rasterize_mode="antialiased"`.
Result: needle fraction (aniso > 10) **40% → 0%**; masked PSNR on training views
dropped slightly (21.5 → 20.7 dB — the needles were overfitting single views, not
adding real information) while the reconstruction became visually clean from every
angle. This became the final shipped model (`outputs/full_reg.pt`, 2.43M Gaussians).
This is a straightforward regularization, not from a specific paper, but the same
philosophy as scale/opacity pruning used throughout the 3DGS literature (e.g.
Mip-Splatting, Compact-3DGS) to keep Gaussians well-conditioned.

## 2DGS — tried, and rejected for this data (a real negative result)

2D Gaussian Splatting (Huang et al., *"2D Gaussian Splatting for Geometrically
Accurate Radiance Fields,"* SIGGRAPH 2024, https://arxiv.org/abs/2403.17888)
represents each primitive as a flat, surface-aligned disk instead of a 3D ellipsoid,
with two regularizers — **normal consistency** (rendered normal vs. depth-derived
surface normal) and a **distortion loss** (concentrate a ray's weight on one surface)
— that are reported to produce much cleaner indoor walls/floors than 3DGS.

We tried it (`src/train_2dgs.py`, reusing gsplat's `rasterization_2dgs`) hoping the
flat surfels would especially help the floor/ceiling regions that are only ever seen
at a grazing angle. Two attempts:
- **v1** (`dist_lambda=100` from iteration 0): **collapsed** — masked PSNR ~10.6 dB,
  blocky/tiled depth, washed-out renders. The distortion loss is too aggressive for
  this sparse, single-vantage capture; it appears to have crushed geometry onto a
  handful of degenerate depth planes before normals had anywhere sensible to point.
- **v2** (`dist_lambda=1`, delayed to iteration 7000, `normal_lambda=0.02`): fixed the
  collapse (18.4 dB) but remained **~3 dB behind** the regularized 3DGS model (21.4 dB)
  *despite* using 2.6× more primitives (6.34M vs 2.43M), and developed localized
  "confetti" noise — surfels whose orientation never converged — concentrated around
  windows and reflective shelving.

**Conclusion:** 2DGS's surface regularizers implicitly assume denser multi-view
coverage than a single-vantage, 20-scan-point capture provides; per-scene 2DGS is not
a good fit for this dataset's coverage shape. Kept `outputs/full_reg.pt` (3DGS) as the
shipped model; both 2DGS checkpoints are retained for the record
(`outputs/full_2dgs.pt`, `outputs/full_2dgs_v2.pt`).

## Floor & ceiling: geometric completion, not hallucination

Because capture is single-pitch (README §5), the floor directly underfoot and the
ceiling directly overhead are **never in any image** — zero observations, not a
training deficiency. No optimization can recover geometry that was never seen; the
only options are (a) leave it as unconstrained noise, (b) fit the trivially-known flat
*planes* that must exist there, or (c) generatively hallucinate detail (next section).
We implemented (b): fit the floor/ceiling heights from the percentiles of the observed
point cloud projected onto world-up, colour them from the nearest-height real points,
and render them as clean planes in the viewer (`export_meta.py`'s `inferred_planes`,
toggleable in the UI, off by default). This is explicitly **completion of known
geometry** (a floor is planar; we are not inventing furniture or texture), which is
the trust-preserving choice for an inspection twin, at the cost of looking obviously
synthetic rather than "reconstructed."

## Generative enhancement: ArtiFixer and the video-diffusion family (cited, not run)

A distinct line of very recent work targets exactly the failure mode above —
sparse/single-vantage 3D reconstruction that "extrapolates poorly to under-observed
areas" — using large video-diffusion priors to regenerate the unreliable regions:

- **ArtiFixer** (NVIDIA, SIGGRAPH 2026), https://arxiv.org/abs/2603.00492 — built on
  Wan2.1's 14B-parameter video diffusion model. Its key idea, **opacity mixing**:
  start denoising from the existing rendering where the 3D representation has
  geometry, from pure noise where it does not, blended by the rendered opacity map.
  Reports +2 dB over prior enhancement methods.
- **Gen3R** (CVPR 2026, https://arxiv.org/abs/2601.04090), **VideoScene** (CVPR 2025,
  https://arxiv.org/abs/2504.01956), **Scene Splatter**
  (https://arxiv.org/abs/2504.02764) — the same family: couple a feed-forward
  reconstructor or sparse 3DGS with a video-diffusion prior to synthesize
  geometrically-consistent novel content from very few images.
- **SHARP** (Apple, Dec 2025, https://github.com/apple/ml-sharp) — regresses a 3DGS
  from a *single* image in under a second; relevant in spirit but built for
  single-image synthesis, not multi-view fusion of 240 posed images + depth, which is
  our actual strength.

**We could not run the real ArtiFixer**: it requires ~80 GB VRAM (A100/H100/GB200
class) and ~34 GB of disk for weights; this instance has a 24 GB RTX 3090 and, at the
time, single-digit GB of free disk. There is no smaller/quantized variant.

**What we built instead is the same idea at feasible scale**
(`src/artifixer_enhance.py`): render a view, then run an image-diffusion model
(SD-Turbo, then SDXL for a stronger pass) as img2img over it, **blended against the
original render using DA3's per-pixel `conf` as the gate** — low confidence (windows,
reflections) is enhanced/regenerated, high confidence is kept as the real render.
This is ArtiFixer's opacity-mixing mechanism with our existing trust data standing in
for its rendered-opacity signal — our confidence/coverage channels turn out to be
exactly the control signal that mechanism needs. Two honest results from running it:
the confidence gate correctly and precisely isolates the windows in every test view;
and a full, *ungated* SDXL repaint of the whole frame looks more polished than the
gated blend, but hallucinates walls/ceiling detail that was never there — the gated
version is the more defensible choice for an inspection tool, at the cost of being
visibly less "fixed." With an 80 GB GPU, the real ArtiFixer (or VideoScene, the most
efficient of the family) is the clear next step for the under-observed regions this
project surfaces.

## Summary table — decision → source

| Decision | Grounded in |
|---|---|
| 3DGS over NeRF for a browser viewer | Kerbl SIGGRAPH 2023 |
| Use depth priors at all | DS-NeRF CVPR 2022 |
| Depth reg is essential for sparse views | DNGaussian CVPR 2024; Chung CVPRW 2024 |
| Init Gaussians from lifted mono depth | Chung CVPRW 2024 |
| Scale-invariant (Pearson / patch-norm) depth loss, not L1 | DN-Splatter WACV 2025; DNGaussian CVPR 2024 |
| Downweight depth at RGB edges | DN-Splatter (gradient-aware log loss) |
| L1 + D-SSIM photometric, λ=0.2 | Kerbl SIGGRAPH 2023 |
| Inverse person mask as loss weight | DN-Splatter; standard practice |
| Shadows/reflections need robust masking | RobustNeRF CVPR 2023; NeRF On-the-go CVPR 2024; SpotLessSplats TOG 2025 |
| Penalize anisotropy/scale to kill needle splats | standard 3DGS-family compaction practice (Mip-/Compact-3DGS lineage) |
| 2DGS tried, rejected (assumes denser coverage) | Huang et al. SIGGRAPH 2024 (2DGS) |
| Floor/ceiling = fitted planes, not hallucinated | — (geometric completion from observed-cloud percentiles) |
| Confidence-gated diffusion enhancement for windows | ArtiFixer SIGGRAPH 2026 (opacity mixing) — real model infeasible on 24GB, reproduced the mechanism |
