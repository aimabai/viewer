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
  - an **inferred floor** — the single-pitch capture never directly observes the floor
    (or ceiling), so we complete it as a top-down texture via classical (non-generative)
    inpainting of the real observed pixels, explicitly labelled as completion, not
    invention; the ceiling is left as a flat fallback (judged too irregular to
    extrapolate honestly this way),
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
#    without it under-reports quality by several dB (we hit this ourselves).
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
  definition (single-pitch, single-vantage capture — see `RESEARCH.md`). We complete
  the floor honestly (extend real texture, no invention) and leave the ceiling flat
  rather than fabricate detail we have no basis for.
- **2DGS was tried and rejected** for this dataset (two tuning attempts; its surface
  regularizers assume denser multi-view coverage than a single-vantage capture
  provides) — a real negative result, documented in `RESEARCH.md` rather than hidden.

## What I'd do with another week

Two things would move the needle most. First, **close the reflective-surface gap
properly** — either rent an 80 GB GPU to run the real ArtiFixer (our confidence/
coverage channels are already exactly the control signal it needs), or add
unsupervised distractor handling (SpotLessSplats, TOG 2025) so shadows and
reflections stop baking in as static texture without needing masks at all. Second,
**a rigorous held-out-view evaluation** — everything we report is training-view PSNR;
setting up a proper eval split (holding out every Nth view, matching how prior
approaches to this dataset were scored) would give an honest generalization number
instead of a training-fit one, and might reveal that some of our regularization
choices (which were conservative by design) have a real, measurable payoff that
training-view PSNR alone doesn't show. Smaller items: an automated second-opinion
pass to catch upstream person-detection misses like the one documented above (a
second segmentation model or simple flow/temporal consistency check across the 12
yaws of a scan-point would likely have caught it), and progressive/LOD splat loading
so the full-quality model doesn't require a ~170 MB first load.
