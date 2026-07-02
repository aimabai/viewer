# InfraScan 3D Walk-Through — Depth-Supervised Gaussian Splatting

Turns the DA3 output in `dataset/` into a photographic 3D reconstruction and a
browser walk-through viewer. The original task brief is preserved at
[`docs/TASK.md`](docs/TASK.md). See [`RESEARCH.md`](RESEARCH.md) for the
paper-grounded design decisions, the full method specification, and every
alternative that was tried and not shipped.

## What this is

- **Reconstruction:** depth-supervised 3D Gaussian Splatting, trained from
  scratch on [`gsplat`](https://github.com/nerfstudio-project/gsplat) — a
  custom training loop, not nerfstudio/splatfacto. Masked photometric + Pearson
  depth loss (confidence- and edge-weighted), confidence-adaptive anti-needle
  regularization, multi-view-consistency-filtered initialization, and learned
  per-camera pose corrections. See `RESEARCH.md` §1-2 for the exact recipe.
- **Viewer:** three.js + `@mkkellogg/gaussian-splats-3d`:
  - Real-time photographic splat rendering, first-person WASD + mouse-look
    (click the scene to lock the pointer, Esc to release).
  - Three trust-aware render modes — confidence heat-map, coverage (view-count,
    with a continuous-fade threshold slider), colour-by-supervising-view —
    each self-explaining in the on-screen status line.
  - A floor-plan minimap (a real top-down density image of the reconstruction)
    with a clickable, locally-adaptive walkable path.
  - Bidirectional click-to-inspect: click a pixel in the shown photo to
    ray-cast through its DA3 depth and drop a 3D marker; click a point on the
    3D reconstruction while walking to see which of the 240 source photos
    supervised that surface.
  - A generatively-completed floor — the single-pitch capture never directly
    observes the floor beyond the coverage bubble (or the ceiling at all), so
    unobserved floor pixels are completed via mask-conditioned SDXL inpainting
    of a top-down projection, then baked as real, flat, floor-aligned
    Gaussians merged directly into the model. Those synthetic points are
    tagged unsupervised in the trust data, so Coverage/Confidence mode still
    correctly shows this region as generated.
  - A standalone raw-splat viewer (`viewer/raw.html`) for opening an exported
    `.ply`/`.ksplat` directly.

## Setup (clean machine)

Requires: an NVIDIA GPU (developed on an RTX 3090, 24GB — CUDA 11.8 wheels are
used, so most driver versions from the last few years work), Python 3.10,
[`uv`](https://docs.astral.sh/uv/), and Node.js 18+.

### 1. Python environment (reconstruction pipeline)

```bash
cd viewer-project
bash setup_env.sh          # creates .venv, installs torch cu118 + prebuilt gsplat wheel
source .venv/bin/activate
```

`setup_env.sh` installs gsplat's prebuilt CUDA wheel rather than compiling from
source — no `nvcc`/system CUDA toolkit required, just a driver new enough for
CUDA 11.8.

### 2. Put the dataset in place

```
viewer-project/dataset/       # cameras.json, pointcloud.ply, views/, views_mask/, da3/
```

### 3. Run the reconstruction pipeline

```bash
cd src

# 1. sanity-check the dataset
python data.py ../dataset

# 2. depth-lift a colored point cloud. Two variants are used downstream:
#    the plain one for training (regularization handles outliers directly),
#    and a multi-view-consistency-filtered one for the floor top-down
#    projection (more sensitive to noise; ~3.7% of points dropped).
python init_pointcloud.py --dataset ../dataset --out ../prepared/init_pointcloud.ply
python init_pointcloud.py --dataset ../dataset --out ../prepared/init_pointcloud_consistent.ply \
  --consistency-check --consistency-tol 0.12

# 3. train (full run: 504x504 res, 30k iters, ~60-65 min on a 3090)
python train.py --dataset ../dataset --init-ply ../prepared/init_pointcloud.ply \
  --out ../outputs/full.pt --iters 30000 --sh-degree 3 --max-init-points 1000000 \
  --depth-lambda 0.2 --aniso-lambda 0.02 --size-lambda 0.05 \
  --refine-poses --pose-warmup 2000 --conf-gamma 1.5 \
  --conf-reg-gamma 2.0 --reg-conf-boost 4.0 --grad-edge-scale 8.0

# 4. evaluate against ground truth (masked PSNR + a GT|render|depth panel).
#    render_eval.py automatically applies the learned per-camera pose
#    correction stored in the checkpoint if --refine-poses was used.
python render_eval.py --ckpt ../outputs/full.pt --views 0,6,30,120

# 5. export scene metadata (world-up, floor basis, walkable path) and per-view
#    depth (for the photo-click backprojection) before the floor completion,
#    which reads both.
python export_meta.py --dataset ../dataset --init-ply ../prepared/init_pointcloud.ply \
  --out ../viewer/public/scene/scene_meta.json
python export_depth.py --dataset ../dataset --out-dir ../viewer/public/scene/depth \
  --meta ../viewer/public/scene/scene_meta.json

# 6. generatively complete the floor and produce the final scene.ply +
#    trust.bin (per-Gaussian confidence/coverage/supervising-view for the
#    trust-aware viewer modes) in one step.
python inpaint_planes.py --ckpt ../outputs/full.pt --dataset ../dataset \
  --init-ply ../prepared/init_pointcloud_consistent.ply \
  --out-scene ../viewer/public/scene/scene.ply \
  --out-trust ../viewer/public/scene/trust.bin \
  --meta ../viewer/public/scene/scene_meta.json

# 7. convert the exported .ply to gsplat-3d's compact .ksplat format
cd ../viewer
node scripts/ply2ksplat.mjs public/scene/scene.ply public/scene/scene.ksplat
```

A trained checkpoint is not included in the repo (large binary; regenerate with
the commands above). For a quick smoke test of the whole pipeline instead of
the full ~65-minute run, cut `--iters 2000 --max-init-points 200000` — quality
will be low but everything runs end-to-end in a couple of minutes.

**Re-exporting from an already-trained checkpoint** (e.g. after retraining with
different hyperparameters) without re-running the floor's diffusion inpainting:
`python redeploy_from_ckpt.py --ckpt ../outputs/full.pt` reuses the cached
`floor_texture.png` from a prior `inpaint_planes.py` run. Requires that a floor
texture already exists.

### 4. Run the viewer

```bash
cd viewer
npm install
npm run dev        # http://localhost:10100 (see viewer/vite.config.js)
```

Production build:
```bash
npm run build && npm run preview
```
This builds two pages: `/` (the full interactive viewer) and `/raw.html` (the
standalone raw-splat file viewer — drag and drop a `.ply`/`.ksplat`, or visit
`/raw.html?src=/scene/scene.ply`).

## Known limitations

See `RESEARCH.md` §4 for the full list with root-cause detail. Summary:

- All reported PSNR/SSIM numbers are training-view, not held-out — the single
  largest methodological gap in this project.
- Windows, mirrors, and reflective surfaces reconstruct as soft, low-detail
  regions. A confidence-gated image-diffusion enhancement pass exists as an
  offline comparison tool (`RESEARCH.md` §3.4, `src/flux_enhance.py`) — a
  partial, offline mitigation, not a fix to the reconstruction itself.
- Floor/ceiling beyond the ~9×6×11m coverage bubble are geometrically
  unconstrained by definition (single-vantage capture). The floor is
  generatively completed and tagged as such; the ceiling is not filled.
- One person (visible at yaw 240 across every scan-point) was missed entirely
  by the provided YOLOv8m-seg masks in one source frame — root-caused, not
  papered over.
- Floater-pruning by view-provenance is implemented but disabled by default —
  a fixed-distance version regressed on live testing (see `RESEARCH.md` §3.2).

## What I'd do with another week

**A genuine held-out-view evaluation.** Every comparison in `RESEARCH.md` is
training-view PSNR. Exclude every 12th view from training (20 held-out views
spread across all 20 scan-points), train identically otherwise, and re-run
each comparison on the excluded views. Expected direction of change is
documented in `RESEARCH.md` §4.

**A validated floater-pruning fix.** The density-adaptive version prototyped in
`RESEARCH.md` §3.2 needs verification beyond a single rendered comparison
before shipping — a proper before/after sweep across many views, not just the
one that reproduced the original complaint.

**Real ArtiFixer or an equivalent opacity-mixing model**, given an 80GB-class
GPU — the confidence/coverage channels already computed for the trust-aware
viewer are the right control signal for it (`RESEARCH.md` §3.4).

Smaller items: a second-opinion detector or per-scan-point temporal-consistency
check to catch upstream person-detection misses automatically; progressive/LOD
splat loading so the full-quality model doesn't require a ~100MB first load.
