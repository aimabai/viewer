# Research grounding & design decisions

The reconstruction method here is **depth-supervised 3D Gaussian Splatting**, with
each design choice tied to a published method. This dataset sits squarely in the
*sparse / uneven-coverage* and *dynamic-scene* regimes those papers were built for:
20 scan-points clustered in a small bubble, with people walking through 67% of views.

> **Methodology caveat that applies to every number below.** All PSNR/SSIM figures
> in this document are **training-view (in-sample) metrics** — every one of the 240
> views was used in training; none were held out. Training-view PSNR measures fit to
> views the optimizer directly saw, not generalization to a novel viewpoint, so it
> can be inflated by anything that helps a model memorize its supervision (this
> matters concretely below for both the anti-needle result and pose refinement).
> This is the single biggest methodological gap in this project. The fix is a
> held-out split — e.g. exclude every 12th view from training (matching the
> `--eval-mode interval --eval-interval 12` convention used elsewhere on this
> dataset) and report PSNR only on the excluded views — which we did not build here
> for time reasons; see README1.md's "another week" section for the concrete plan
> and what we'd expect to change under it.

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
Result: needle fraction (aniso > 10) **40% → 0%**; masked **training-view** PSNR
dropped slightly (21.5 → 20.7 dB) while the reconstruction became visually clean
from every angle. This became the final shipped model (`outputs/full_reg.pt`,
2.43M Gaussians). This is a straightforward regularization, not from a specific
paper, but the same philosophy as scale/opacity pruning used throughout the 3DGS
literature (e.g. Mip-Splatting, Compact-3DGS) to keep Gaussians well-conditioned.

Our reading is that the needles were overfitting individual training views (a thin
sliver edge-on to one camera can locally satisfy that view's photometric loss
without corresponding to real geometry) rather than encoding real information, so
their removal should be a real quality improvement despite the training-view PSNR
drop — but that argument is itself made using only the training metric it's
critiquing, which is circular as *proof*, even though we think it's the right
read. A held-out split is the only way to actually settle whether the visually
cleaner model also generalizes better, and we haven't run one (see the caveat
at the top of this document).

## Multi-view consistency filtering — a floater remover, not real triangulation

`init_pointcloud.py --consistency-check` drops a depth-lifted candidate point if it
disagrees (beyond a relative tolerance) with an *adjacent-yaw view of the same
scan-point* that also has a valid, comparable pixel there (see the file's docstring
for the exact projection). Result: 3.7% of points dropped as inconsistent, used as
the initialization for `full_v3.pt`.

**We want to be precise about what this check actually verifies, because it's
weaker than "multi-view geometric verification" sounds.** All 12 yaws at one
scan-point share the *same camera position* — they're a rotation-only sweep of a
360° capture, with **zero baseline** between them. Zero baseline means zero
triangulation power: two rays from the same optical center can never disagree
about depth through parallax, only through the *monocular* depth model producing a
different estimate for the same physical surface when it's cropped into two
different perspective images. So this check is really: "does DA3's single monocular
model self-consistently agree with itself across two overlapping crops of nominally
the same 360° capture?" — a legitimate way to catch a subset of monocular-depth
outliers (and cheap, since we already have the adjacent-yaw images), but it is not
independent multi-view verification in the stereo sense, and it can't catch
errors the monocular model makes *consistently* across both crops.

Cross-scan-point checks would have genuine (if small) baseline — the 20 scan-points
spread across roughly a 1.5 m³ bubble — but that baseline is tiny relative to the
12×20 m room: triangulation precision degrades with baseline/depth ratio, and
sub-meter baseline against 12-20 m walls gives very poor depth resolution for
anything beyond a couple of metres. We use same-scan-point adjacent yaws (not
cross-scan-point pairs) specifically because they're cheap and still catch the
class of error we cared about (isolated single-frame depth blowups), but the
honest characterization is "floater/outlier remover via monocular self-consistency,"
not "multi-view stereo verification." We'd call it that plainly in any follow-up
rather than the more impressive-sounding "adjacent views should agree on depth"
framing the code comments originally used.

## Pose refinement — and why its measured gain is partly a training-fit artifact

`train.py --refine-poses` learns a small per-camera SE3 correction (axis-angle
rotation + translation, `pose_rot`/`pose_trans` in the checkpoint) on top of DA3's
given poses, with a 2000-iteration warm-up (let Gaussian geometry settle first —
same lesson as 2DGS's distortion loss above: coupling a second set of free
parameters to the optimization from iteration 0 destabilizes it) and an L2 penalty
keeping each correction small. Final magnitudes stayed modest: ~1.2° rotation,
~6.2 cm translation on average — plausible calibration-error scale, not runaway
drift.

**We initially mis-evaluated this and want to document the fix, not just the
number.** `render_eval.py` originally rendered from the *original* DA3 pose while
the model had been trained against the *corrected* pose for that camera — an
eval/train mismatch that looked like a ~6 dB *regression* until we applied the same
correction at eval time, after which training-view PSNR was ~20.8 dB (in line with
the unrefined baseline). We fixed this in both `render_eval.py` and `export.py`'s
provenance computation (the trust/coverage data would have had the same silent
mismatch otherwise).

That fix is correct, but the "~6 dB you'd lose by forgetting the correction" number
**cuts both ways and should not be read as a generalization gain.** A per-camera
learned correction is fit *to that specific training image* — it has no way to
generalize to a genuinely novel viewpoint, because there's no learned delta for a
pose that was never in the training set. So the measured PSNR recovery is, by
construction, partly (perhaps mostly) a training-fit artifact: we're partly
measuring "how well can the model reproduce a training view once we let its exact
camera also move to fit that view," which is a strictly easier target than novel-view
quality. What's plausibly real and separate from this: if DA3's poses carry a small
systematic error, correcting it during optimization should let the optimizer
converge to *more internally consistent 3D geometry* across all 240 views jointly
(the classic bundle-adjustment argument), which would benefit novel views too — but
that benefit, if it exists, wouldn't show up as "apply this view's specific
correction at eval time"; it would show up as a genuinely better underlying model
even when rendered from unrefined novel poses. We have not isolated that effect
from the training-fit effect, which is exactly what a held-out split (see the
caveat at the top of this document) would let us do: hold out every Nth view,
train pose corrections only on the remaining views, and check whether the held-out
views (which have *no* learned correction at all) render better than a no-pose-
refinement baseline. If they do, that's the genuine bundle-adjustment benefit,
isolated from the training-fit artifact.

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
  collapse (18.4 dB **training-view**) but remained **~3 dB behind** the regularized
  3DGS model (21.4 dB, same training-view metric) *despite* using 2.6× more
  primitives (6.34M vs 2.43M), and developed localized "confetti" noise — surfels
  whose orientation never converged — concentrated around windows and reflective
  shelving.

**Conclusion:** 2DGS's surface regularizers implicitly assume denser multi-view
coverage than a single-vantage, 20-scan-point capture provides; per-scene 2DGS is not
a good fit for this dataset's coverage shape. Kept `outputs/full_reg.pt` (3DGS) as the
shipped model; both 2DGS checkpoints are retained for the record
(`outputs/full_2dgs.pt`, `outputs/full_2dgs_v2.pt`).

## Floor & ceiling: three iterations, ending in generative fill baked as real Gaussians

Because capture is single-pitch (README §5), the floor directly underfoot and the
ceiling directly overhead are **never in any image** — zero observations outside the
coverage bubble, not a training deficiency. No optimization can recover geometry
that was never seen. We went through three designs here, each rejected by feedback
for a specific, defensible reason — worth tracing honestly rather than presenting
only the final one:

1. **Flat coloured plane** (fit floor/ceiling height from point-cloud percentiles,
   colour from nearest-height real points, render as a separate mesh). Geometrically
   honest (a floor is planar; we invented no texture) but looked like an obviously
   separate, flat, disconnected surface — rejected on sight.
2. **Classical-inpainting-textured plane** (project real colour onto a top-down
   "map view," extend it into the unobserved holes with OpenCV's Telea inpainting —
   no generative model, so nothing invented beyond direct extrapolation of nearby
   real pixels). More honest than (1) but still visually read as a separate flat
   plane, and the extrapolation itself looked streaky at hole boundaries.
3. **Shipped: generative fill baked directly into the Gaussian model.** Only the
   *unobserved* floor pixels (78.4% of the floor's top-down footprint) are completed
   via SDXL-inpainting (mask-conditioned on the real/hole split, same top-down
   projection as (2)); every hole pixel then becomes an individual small, flat,
   floor-plane-aligned Gaussian with that pixel's completed colour, concatenated
   directly into the exported splat set (146k synthetic Gaussians alongside 1.2M
   real ones) — not a separate mesh or render path. **Ceiling is not filled at all**
   (judged too irregular for the same top-down-projection approach to be
   defensible) and there is no plane object left in the viewer for either surface.

Provenance is preserved through this: synthetic floor Gaussians are tagged
`confidence=0, n_views=0, supervising_view=-1` in the trust data, so switching to
Coverage or Confidence mode still correctly shows this region as *generated*, not
observed — the one thing that's constant across all three iterations is that the
viewer never claims synthetic content is real.

This is the same img2img-plus-mask mechanism as `artifixer_enhance.py` below (see
that section for the precise distinction from ArtiFixer's actual opacity mixing) —
here scoped to hole pixels only rather than blended by confidence over a whole
frame, which sidesteps the "hallucinates walls that were never there" failure we
saw there, because there is no ambiguity about *where* to generate: the hole mask
already tells us exactly.

## Generative enhancement: ArtiFixer and the video-diffusion family (cited, not run)

A distinct line of very recent work targets exactly the failure mode above —
sparse/single-vantage 3D reconstruction that "extrapolates poorly to under-observed
areas" — using large video-diffusion priors to regenerate the unreliable regions:

- **ArtiFixer** (NVIDIA, SIGGRAPH 2026), https://arxiv.org/abs/2603.00492 — built on
  Wan2.1's 14B-parameter video diffusion model. Its key idea, **opacity mixing**:
  *at the noise-initialization stage of denoising*, start from the existing rendering
  where the 3D representation has geometry, from pure noise where it does not
  (blended by the rendered opacity map). This is what lets it *generate* plausible
  content in genuinely empty regions instead of just cleaning up existing pixels.
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
- **Difix3D+** (NVIDIA, CVPR 2025 Oral, https://arxiv.org/abs/2503.01774) — the
  correct reference class for what we actually built (see below): a single-step
  image-diffusion model that cleans up *rendered* views via **deterministic
  conditioning** on the existing image, not noise-init selection. Explicitly a
  different, lighter-weight mechanism than opacity mixing.

**We could not run the real ArtiFixer**: it requires ~80 GB VRAM (A100/H100/GB200
class) and ~34 GB of disk for weights; this instance has a 24 GB RTX 3090 and, at the
time, single-digit GB of free disk. There is no smaller/quantized variant.

**What we built (`src/artifixer_enhance.py`) is NOT opacity mixing — we want to be
precise about this rather than overclaim.** It renders a view, runs an
image-diffusion model (SD-Turbo, then SDXL) as **img2img over the already-rendered
image**, then blends the diffusion *output* against the original render using DA3's
per-pixel `conf` as a post-hoc alpha gate. Opacity mixing intervenes earlier and
differently — at the noise-initialization stage of denoising, before any image
exists — which is specifically what lets it generate coherent content in *empty*
regions rather than just conditioning on what's already there. What we built is
**confidence-gated deterministic conditioning**, the same family as Difix3D+ /
"Fixer"-style artifact cleanup, not ArtiFixer's mechanism. This distinction isn't
academic: the ArtiFixer paper explicitly notes that deterministic-conditioning
approaches "fail to reconstruct empty regions" — and that is *exactly* what we
independently observed. Two honest results from running it: the confidence gate
correctly and precisely isolates the windows in every test view (the gating logic
works); and a full, *ungated* SDXL repaint of the whole frame looks more polished
than the gated blend, but **hallucinates walls/ceiling detail that was never
there** — precisely the empty-region failure mode opacity mixing exists to prevent.
So rather than reproducing ArtiFixer's mechanism at smaller scale, we reproduced
the *baseline it improves on*, and then independently rediscovered the exact
failure mode motivating ArtiFixer's design. That's a real, if smaller, result: our
confidence/coverage channels are already the right control signal for an
opacity-mixing-style approach (they mark precisely the "empty" regions ArtiFixer's
opacity map would need), which is the concrete link to build next with the real
model. With an 80 GB GPU, the real ArtiFixer (or VideoScene, the most efficient of
the family) is the clear next step for the under-observed regions this
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
| Penalize anisotropy/scale to kill needle splats | standard 3DGS-family compaction practice (Mip-/Compact-3DGS lineage); result is training-view PSNR only |
| Adjacent-yaw depth consistency check for init | — (monocular self-consistency / floater removal; NOT multi-view triangulation — zero baseline within a scan-point) |
| Small per-camera pose correction during training | classic bundle-adjustment-style pose refinement; measured PSNR gain is entangled with training-fit, not isolated as generalization |
| 2DGS tried, rejected (assumes denser coverage) | Huang et al. SIGGRAPH 2024 (2DGS); comparison is training-view PSNR only |
| Floor = generative fill baked as real Gaussians (hole-only); ceiling not filled | img2img/deterministic-conditioning family (Difix3D+-like), not ArtiFixer's opacity mixing |
| Confidence-gated diffusion enhancement for windows | Difix3D+ CVPR'25 / "Fixer"-family deterministic conditioning — NOT ArtiFixer's opacity mixing (real ArtiFixer infeasible on 24GB); our confidence/coverage channels are the right control signal for a future opacity-mixing implementation |
