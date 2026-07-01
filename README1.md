# InfraScan 3D Walk-Through — Depth-Supervised Gaussian Splatting

Turns the DA3 output in `dataset/` into a photographic 3D reconstruction and a
browser walk-through viewer. The original task brief is preserved at
[`docs/TASK.md`](docs/TASK.md); this file is the submission README (setup +
what-I'd-do-next, per the brief's §9 deliverables).

See [`RESEARCH.md`](RESEARCH.md) for the paper-grounded design decisions and every
experiment we ran — including two that didn't pan out (2DGS, full-diffusion floor
completion) and why. [`docs/pipeline.png`](docs/pipeline.png) has the pipeline figure.

---

## What this is

- **Reconstruction:** depth-supervised 3D Gaussian Splatting, trained from scratch on
  [`gsplat`](https://github.com/nerfstudio-project/gsplat) (our own training loop, not
  nerfstudio/splatfacto). Masked photometric + Pearson depth loss, confidence-weighted,
  anti-needle scale regularization, multi-view-consistency-filtered initialization, and
  small learned per-camera pose corrections.
- **Viewer:** three.js + `@mkkellogg/gaussian-splats-3d`:
  - real-time photographic splat rendering, first-person **WASD + mouse-look** movement
    (click the scene to lock the pointer, Esc to release),
  - three **trust-aware render modes** — confidence heat-map, coverage (view-count,
    with a threshold slider), and colour-by-supervising-view, each self-explaining in
    the on-screen status line,
  - a **floor-plan minimap** (a real top-down density image of the reconstruction) with
    a **clickable walkable path** along the room's principal axis,
  - **bidirectional click-to-inspect**: click a pixel in the shown photo to ray-cast
    through its DA3 depth and drop a 3D marker (the interaction named in the brief's
    §2 that a plain point-cloud viewer wouldn't have); click the crosshair on the 3D
    reconstruction while walking to see which of the 240 source photos supervised that
    surface,
  - a **generatively-filled floor** — the single-pitch capture never directly
    observes the floor beyond the coverage bubble (or the ceiling at all), so the
    unobserved floor pixels are completed via mask-conditioned SDXL inpainting of a
    top-down projection, then baked as real, flat, floor-aligned Gaussians merged
    directly into the model (not a separate textured plane); those synthetic points
    are tagged confidence=0/unsupervised in the trust data, so Coverage/Confidence
    mode still correctly shows this region as generated. Ceiling is left unfilled
    (judged too irregular for this approach). See RESEARCH.md for the precise
    (and initially overclaimed) relationship to NVIDIA's ArtiFixer,
  - a **standalone raw-splat viewer** (`viewer/raw.html`) for opening an exported
    `.ply`/`.ksplat` file directly, separate from the full interactive walk-through.

---

## Setup (clean machine)

Requires: an NVIDIA GPU (developed on an RTX 3090, 24 GB — CUDA 11.8 wheels are used,
so most driver versions from the last few years work), Python 3.10,
[`uv`](https://docs.astral.sh/uv/), and Node.js 18+.

### 1. Python environment (reconstruction pipeline)

```bash
cd viewer-project
bash setup_env.sh          # creates .venv, installs torch cu118 + prebuilt gsplat wheel
source .venv/bin/activate
```

`setup_env.sh` installs gsplat's **prebuilt** CUDA wheel rather than compiling from
source — no `nvcc`/system CUDA toolkit or matching host compiler required, just a
driver new enough for CUDA 11.8 (see the script for why this matters and the exact
index URLs — we hit and solved this exact problem during development).

### 2. Put the dataset in place

```
viewer-project/dataset/       # cameras.json, pointcloud.ply, views/, views_mask/, da3/
```

### 3. Run the reconstruction pipeline

```bash
cd src

# 1. sanity-check the dataset (matches the brief's §4)
python data.py ../dataset

# 2. depth-lift a colored point cloud, filtered for multi-view depth consistency
#    (adjacent yaws should agree on a surface's depth; drop points where they don't —
#    our single-vantage capture never cross-checks DA3's monocular noise otherwise)
python init_pointcloud.py --dataset ../dataset --out ../prepared/init_pointcloud.ply \
  --consistency-check --consistency-tol 0.12

# 3. train (full run: 504x504 res, 30k iters, ~60-65 min on a 3090 with pose refinement)
python train.py --dataset ../dataset --init-ply ../prepared/init_pointcloud.ply \
  --out ../outputs/full.pt --iters 30000 --sh-degree 3 --max-init-points 1000000 \
  --depth-lambda 0.2 --aniso-lambda 0.02 --size-lambda 0.05 \
  --refine-poses --pose-warmup 2000 --conf-gamma 1.5

# 4. evaluate against ground truth (masked PSNR + a GT|render|depth panel).
#    NOTE: if you trained with --refine-poses, render_eval.py automatically applies
#    the learned per-camera correction (it's stored in the checkpoint) — evaluating
#    without it under-reports quality by several dB (we hit this ourselves). NOTE:
#    this recovered number is still training-view PSNR and is partly a training-fit
#    artifact of the per-camera correction itself — see RESEARCH.md.
python render_eval.py --ckpt ../outputs/full.pt --views 0,6,30,120

# 5. export for the browser viewer: pruned/SH-controlled splat PLY, per-Gaussian trust
#    provenance (confidence / coverage / supervising view), scene metadata (poses,
#    floor-plan basis, walkable path, inferred-plane fit), per-view depth (for the
#    photo-click backprojection), and the floor's inpainted top-down texture.
python make_light_ply.py --ckpt ../outputs/full.pt --out ../viewer/public/scene/scene_light.ply \
  --keep-sh --max-aniso 12 --max-points 1200000
python export.py --ckpt ../outputs/full.pt --dataset ../dataset \
  --out ../outputs/standard.ply --trust-bin ../viewer/public/scene/trust.bin
python export_meta.py --dataset ../dataset --init-ply ../prepared/init_pointcloud.ply \
  --out ../viewer/public/scene/scene_meta.json
python export_depth.py --dataset ../dataset --out-dir ../viewer/public/scene/depth \
  --meta ../viewer/public/scene/scene_meta.json
python inpaint_planes.py --init-ply ../prepared/init_pointcloud.ply \
  --out-dir ../viewer/public/scene --meta ../viewer/public/scene/scene_meta.json

# 6. convert the exported .ply to gsplat-3d's compact .ksplat format (loads much
#    faster in the browser than the equivalent .ply)
cd ../viewer
node -e '
const fs=require("fs");
const m=require("@mkkellogg/gaussian-splats-3d");
const b=fs.readFileSync("public/scene/scene_light.ply");
const ab=b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength);
const sb=m.PlyParser.parseToUncompressedSplatBuffer(ab,5);
fs.writeFileSync("public/scene/scene.ksplat", Buffer.from(sb.bufferData));
'
```

A pre-trained checkpoint is not included in the repo (large binary; regenerate with
the commands above). For a quick smoke test of the whole pipeline instead of the full
~65-minute run, cut `--iters 2000 --max-init-points 200000` — quality will be low but
everything runs end-to-end in a couple of minutes.

### 4. Run the viewer

```bash
cd viewer
npm install
npm run dev        # http://localhost:5173
```

Production build (what we actually used for review):
```bash
npm run build && npm run preview
```
This builds two pages: `/` (the full interactive viewer) and `/raw.html` (the
standalone raw-splat file viewer — drag and drop a `.ply`/`.ksplat`, or visit
`/raw.html?src=/scene/scene.ply`).

---

## Known limitations (found and diagnosed, not glossed over)

- **Windows, mirrors, and other reflective/transparent surfaces** reconstruct as
  soft dark blobs. The provided person masks correctly exclude people, but *not*
  their shadows or reflections (the brief flags this — it's real). We prototyped a
  confidence-gated generative fix inspired by NVIDIA's ArtiFixer (SIGGRAPH 2026) —
  see `src/artifixer_enhance.py` and `RESEARCH.md` — but the actual ArtiFixer model
  needs ~80 GB VRAM we don't have here; ours is the same mechanism (DA3 confidence as
  the "where to regenerate" gate) at consumer-GPU scale, and it's a partial, offline
  improvement, not a live fix.
- **A specific reconstructed person, traced end-to-end.** We found a person visible
  from multiple angles (most obviously at yaw 240 across *every* scan-point) despite
  our masking. Root-caused it by backprojecting the offending pixel to its 3D point,
  looking up which source photo actually supervised it (we track this per-Gaussian
  already, for the trust-aware viewer modes), and inspecting that photo directly:
  the upstream YOLOv8m-seg detector supplied with the dataset simply missed this one
  person in one frame (`views/000003_pz000_y300_normal.jpg`, 0% mask coverage) — not
  a flaw in our masking logic, a miss in the provided labels we depend on and don't
  re-run ourselves (out of scope per the brief). Left as a documented, precisely
  diagnosed limitation rather than papered over.
- **Floor/ceiling beyond the coverage bubble** are geometrically unconstrained by
  definition (single-pitch, single-vantage capture — see `RESEARCH.md`). We
  generatively complete the floor (hole pixels only, tagged as unsupervised in the
  trust data) and leave the ceiling unfilled rather than fabricate detail with no
  principled way to gate *where* to invent it.
- **All reported PSNR/SSIM numbers are training-view, not held-out.** Every one of
  the 240 views was used in training. This is the most important caveat in this
  project — see "What I'd do with another week" for the concrete held-out-split
  plan and what we'd expect each comparison to do under it.
- **Our confidence-gated diffusion enhancement is not ArtiFixer's mechanism.**
  We initially described `artifixer_enhance.py` as reproducing ArtiFixer's opacity
  mixing; it doesn't — it's deterministic conditioning on an existing render (the
  Difix3D+/"Fixer" family), a distinction with teeth: our own ungated test
  hallucinated walls/ceiling that were never there, exactly the empty-region
  failure opacity mixing exists to prevent. See RESEARCH.md for the corrected
  characterization.
- **2DGS was tried and rejected** for this dataset (two tuning attempts; its surface
  regularizers assume denser multi-view coverage than a single-vantage capture
  provides) — a real negative result, documented in `RESEARCH.md` rather than hidden.

## What I'd do with another week

**First, and most important: a genuine held-out-view evaluation, concretely.**
Every PSNR/SSIM number in this project — the anti-needle result (21.5→20.7 dB), the
2DGS comparison (21.4 vs 18.4 dB), the pose-refinement recovery — is **training-view**
PSNR: all 240 views were used in training, none held out. That's the single biggest
methodological gap here, and I'd fix it exactly like this: exclude every 12th view
from training (`--eval-mode interval --eval-interval 12`, the same convention used
elsewhere on this dataset, giving 20 held-out views spread evenly across all 20
scan-points rather than clustered), train identically otherwise, and re-run each
comparison PSNR-on-held-out instead of PSNR-on-training. Concretely what I'd expect
to change: the anti-needle result should hold up or strengthen — needles overfitting
individual training views should generalize *worse*, not better, so I'd expect the
regularized model's held-out PSNR to beat the un-regularized one even though its
training-view PSNR is lower, which would make the argument non-circular for the
first time. Pose refinement is the one I expect to shrink: a per-camera correction
is fit to its own training image and has no learned delta for a held-out pose, so
most of the "~6 dB you lose by omitting it" is training-fit, not generalization —
I'd isolate the genuine bundle-adjustment benefit (if any) by training pose
corrections only on the non-held-out views and checking whether the held-out
renders (using their *original*, uncorrected poses) still improve over a
no-pose-refinement baseline. Also on the list: I'd re-derive the multi-view
consistency filter's actual claim — same-scan-point yaws share one optical center
(zero baseline), so it's a monocular self-consistency check, not real triangulation;
worth trying a version that also cross-checks the (small, but nonzero) baseline
between nearby scan-points to see if it catches anything the yaw-only version
misses.

**Second: close the reflective-surface gap properly, correctly scoped this time.**
We built a confidence-gated SDXL img2img pass and initially described it as
reproducing ArtiFixer's "opacity mixing" — it doesn't. Opacity mixing intervenes at
the noise-initialization stage of denoising (start from the render where geometry
exists, from noise where it doesn't) specifically so it can *generate* content in
empty regions; what we built is deterministic conditioning on an existing render (the
Difix3D+/"Fixer" family), which is why our own ungated test hallucinated walls/
ceiling that were never there — precisely the failure opacity mixing exists to
prevent. With real compute, the next step is either renting an 80 GB GPU to run
actual ArtiFixer (our confidence/coverage channels are already the right control
signal for its opacity map) or adding unsupervised distractor handling
(SpotLessSplats, TOG 2025) so shadows/reflections stop baking in without needing
masks at all.

Smaller items: an automated second-opinion pass to catch upstream person-detection
misses like the one documented above (a second segmentation model or a simple
flow/temporal-consistency check across a scan-point's 12 yaws would likely have
caught it), and progressive/LOD splat loading so the full-quality model doesn't
require a ~170 MB first load.
