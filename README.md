# Intern Take-Home — 3D Walk-Through Viewer

Welcome! This is a take-home task for the InfraScan team. **Budget about
5 days of calendar time.** The implementation is deliberately open-ended;
we care more about what you choose to ship, what you choose to skip, and
how you communicate the result than about how many boxes you tick. There
is no autograder.

---

## 1 · What InfraScan does

A field tech records the inside of a building — either with a 360° camera
(Insta360) or a tripod laser scanner. We turn that recording into a **3D
digital twin** in the browser:

- a coloured **point cloud** of the walls, floors, and ceilings,
- the **camera poses** at every position the operator stood at, and
- a **dense depth map** for each perspective image we sampled (where every
  pixel knows how far it is from the camera).

The depth maps and camera poses in this dataset come from
[**Depth Anything v3**](https://depth-anything.com) (DA3 for short) —
a monocular depth + pose estimator. That's what the `da3/` folder name
refers to throughout.

Other teams at InfraScan then build on top of that: object detection,
similarity search, higher-fidelity reconstruction. Your task is the
foundational layer they all sit on — turn this raw output into a 3D
reconstruction someone can actually walk through and inspect.

---

## 2 · Your task

> Take the DA3 output in `dataset/` and turn it into a **3D reconstruction
> of the scene plus a viewer that lets a human explore it.** What
> "reconstruction" means, what the viewer does, what functionality you
> layer on top — those are yours to pick. **Be creative.**

The floor is a working point-cloud renderer with some way to navigate
between the 20 scan-points. That alone is fine but unexciting. We want to
see what *you* think a good 3D scene viewer should do given this kind of
data.

**On approach — there is no required reconstruction method.**

You could:

- Render the existing dense point cloud nicely (a real LoD pipeline, EDL
  shading, etc.).
- Build a colored cloud yourself by lifting per-view depth maps into world
  coords and accumulating across views.
- Try TSDF fusion / Poisson reconstruction → textured mesh.
- Train a gaussian splat or a small NeRF.
- Something we haven't thought of.

**A few honest notes if you go the splat route.** We know `ns-train
splatfacto --data ...` and rendering the result is a ~30-minute task —
that on its own is below the bar. What we'd care about is the engineering
*around* it:

- **The scene is dynamic** (people walking around, see §5). This is the
  classic open problem for splats / NeRFs that assume a static scene.
  You'll need to mask out the people (`person_mask` is provided) and
  argue why your masking choice works. Shadows + reflections you won't
  catch with a person mask — flagging that you noticed is itself signal.
- **Coverage is uneven** (see §5). 20 scan-points cluster in ~1.5 m³ of a
  ~12 × 7 × 20 m cloud. Whatever you reconstruct will only be
  well-constrained inside that bubble. Be honest about where your
  reconstruction works and where it doesn't, ideally in the viewer
  itself (e.g. fade-out / "no data here" cue).
- **Integrate it.** A splat rendered as an offline video isn't a viewer.

**What you should ship — bare minimum:**

1. A way to see the 3D reconstruction from arbitrary viewpoints in the
   browser (or as a runnable native app).
2. The 20 perspective views referenced into the 3D scene somehow — "the
   camera stood *here* and was looking at *this*."
3. Setup instructions in a `README.md` that work on a clean laptop.

**Where to be creative** (ideas, not a checklist):

- Click on a perspective view → ray-cast through the depth map → drop a
  3D marker in the reconstruction.
- A floor-plan minimap synced bidirectionally with the 3D view.
- Smooth fly-through animation along the trajectory (or between any two
  scan-points).
- Surface normals / depth heat-map overlay; "confidence" rendering using
  the `conf` channel from the NPZs.
- A "what was reconstructed from what" overlay — colour points by which
  view supervised them.
- An obvious-in-retrospect interaction we'd never have specified.

Depth of work on one or two ideas is much more interesting to us than
checklist coverage of five.

**Out of scope** — don't sink time into:

- Login, multi-user, anything backend-side. Treat the dataset as static
  files served from disk.
- Re-running depth estimation yourself. The output is provided.
- Productionising your splat trainer / mesh builder. A research-grade
  result you can defend is more useful than a polished pipeline.

---

## 3 · What's in `dataset/`

```
dataset/
├── cameras.json           # 240 camera entries (20 scan-points × 12 yaws × 1 pitch)
├── pointcloud.ply         # 61 MB downsampled coloured cloud (THE WHOLE FLOOR, see §5)
├── views/                 # 240 perspective JPGs (504×504) — persons gaussian-blurred (see §5)
├── views_mask/            # 240 PNG masks (504×504, uint8 0/255) — 255 = was-person pixel
└── da3/
    ├── camera_poses.txt   # 240 rows × 16 floats — row-major 4×4 camera→world matrices
    ├── intrinsic.txt      # 240 rows × 4 floats — fx fy cx cy
    └── results_output/
        ├── frame_0.npz
        ├── frame_1.npz
        ...
        └── frame_239.npz  # per-view depth + image + intrinsics + person_mask (see below)
```

### `cameras.json` schema

JSON array, one entry per perspective view:

```jsonc
{
  "id":    0,                            // also the frame_N.npz index
  "pos":   [x, y, z],                    // camera origin in world coords (metres)
  "xy":    [x, z],                       // 2D projection on the floor plane
  "R":     [[r00,r01,r02], ...],         // 3×3 camera-to-world rotation (row-major)
  "pano":  "views/000000_pz000_y000_normal.jpg",
  "frame": 0,    "pitch": 0,    "yaw": 0
}
```

- **`pos` is in metres.** Origin is at the first scan-point.
- **`R` is camera-to-world**: a point in the camera frame `[X,Y,Z]` lands
  at `R @ [X,Y,Z] + pos` in world. Camera convention is OpenCV:
  **+Z forward, +X right, +Y down**.
- `pano` is a relative path inside `dataset/`.

### `frame_N.npz` schema

`np.load("frame_N.npz")` exposes:

| key | shape | dtype | meaning |
|---|---|---|---|
| `image`       | (504, 504, 3) | uint8   | the perspective image the depth was computed for (persons blurred — see §5) |
| `depth`       | (504, 504)    | float32 | metric depth per pixel, in metres (estimated by Depth Anything v3) |
| `conf`        | (504, 504)    | float32 | DA3's per-pixel confidence (higher = trust more) |
| `intrinsics`  | (3, 3)        | float32 | K matrix for this view |
| `person_mask` | (504, 504)    | uint8   | 255 where the original image had a person, 0 elsewhere |

### `views/` directory

Same images as `npz['image']` (redundant, pick whichever loader is easier).
Filename pattern: `{scanpoint:06d}_pz{pitch:03d}_y{yaw:03d}_normal.jpg`.

### What a "scan-point" is

The operator stood at a position, the 360° camera captured the whole
sphere, then we projected that sphere into 12 perspective views (every
30° of yaw). So scan-point 0 has 12 entries in `cameras.json` (yaw 0 to
330), all sharing the same `pos`.

---

## 4 · Sanity check before you start

Open `dataset/cameras.json` and confirm you can:

- Read all 240 entries.
- Group them into 20 scan-points by `frame`.
- For one scan-point, read `R` and confirm the 12 yaw views point in
  different directions (their forward axes — third column of `R` — span
  the unit circle on the floor plane).

If any of that surprises you, ask before you start coding.

---

## 5 · Known dataset caveats

So you don't blame your code:

- **The point cloud covers the whole floor (~12 m × 20 m); the 20 views
  cluster in roughly a 1.5 m³ region** at one end of it. Most of the
  cloud is correct geometry but was *not* seen by the views you have.
  This is fine for a viewer — just don't expect every part of the cloud
  to be lit by a view.
- **Single pitch (`pz000` only).** All 240 views look horizontal.
  Ceiling and floor have no dedicated coverage; depth there is whatever
  spilled in from the horizon views.
- **504×504 perspective resolution** is low by modern standards. Texture
  detail is what it is.
- **`depth` is camera-local z-depth** (top-to-bottom, matches the image),
  in metres. To project pixel `(u, v)` to 3D in the camera frame:
  `X = (u - cx) * depth / fx`, similar for `Y`, `Z = depth`. Then apply
  `R @ [X,Y,Z] + pos` to get the world point.
- **Persons in the views have been gaussian-blurred** for privacy (this
  is an active office). 67 % of the views contained at least one person.
  Masks came from YOLOv8m-seg (instance segmentation, not bounding box),
  dilated by 4 px for safety, then used both to drive the blur and as the
  shipped mask. Average masked region is ≈ 3.3 % of the frame — tight
  silhouettes, no padding waste — so background pixels next to where
  someone stood are still sharp.

  The masks ship in two redundant places (same data):
  - `views_mask/<same-filename>.png` — uint8 504×504, 255 = "this pixel
    was a person in the original capture", 0 elsewhere.
  - `frame_N.npz["person_mask"]` — same array inside the NPZ.

  How you'd use them:

  - **Any reconstruction method that assumes a static scene** (splats,
    NeRFs, TSDF fusion, multi-view stereo). Pass the *inverse* mask as a
    per-pixel loss / weight mask (i.e. `1 - person_mask`). The
    reconstruction is then supervised only on static scene pixels; the
    wall behind a person in one view is learned from other views where
    it's visible. Without masking, the blurred blobs get baked in as
    colored static artifacts. Most libraries (gsplat, nerfstudio,
    open3d's TSDF integrator) accept either an alpha channel in the
    image or a separate `mask.png`. Heads-up: shadows on the floor and
    reflections in monitors / glass are *not* masked — they'll be
    reconstructed as static texture. Noticing that in your write-up is
    signal we like.
  - **Tagging / detection UI.** Overlay the mask as a semi-transparent
    red layer so a user knows which pixels were anonymized vs. real
    texture — useful when explaining "why is this region blurry?"

---

## 6 · Where this fits (downstream context)

So you understand what your viewer feeds into:

- **Object tagging.** The viewer also acts as a labelling tool — a user
  clicks an object in a view, names it ("Vent — Type A"), and we store
  the visual embedding so the same kind of object can be auto-found in
  other views. A click-on-view hook that returns `(view_id, pixel_x,
  pixel_y)` (and optionally backprojects to world coords via `depth`) is
  the substrate this needs.
- **Auto-tagging / detection.** A vision model (currently FastSAM +
  DINOv2) runs over every view, proposes objects, indexes them in
  FAISS. The viewer's job is then to render those detections on the
  perspective image (boxes / masks) and on the 3D scene (markers at the
  backprojected positions).
- **Higher-fidelity reconstruction.** Today the digital twin is a
  point cloud. We're interested in upgrading to gaussian splat / NeRF /
  textured-mesh outputs so the walkthrough looks photographic instead
  of dotty — see §2 if you want to take a swing at it.

You don't need to implement any of these. They're here so you know what
"a good viewer" eventually has to support; design the foundations
accordingly.

[gsplat]: https://github.com/nerfstudio-project/gsplat
[nerfstudio]: https://docs.nerf.studio/

---

## 7 · What we're looking for

| Dimension | What good looks like |
|---|---|
| **It works** | Setup is documented. We can run it on a clean laptop without DM'ing you. The bare minimum from §2 works end-to-end. |
| **Code quality** | Idiomatic for whatever stack you pick. Small, focused, easy to follow. |
| **Judgement** | What did you choose to leave out, and why? A note on one trade-off you made is worth a screen of code. |
| **Communication** | A short Loom or written walkthrough at the end. What works, what's broken, what you'd build next with another week. |

**Anti-patterns** we've seen:

- All the time spent on visual polish, depth NPZs never opened.
- Hard-coding values that only work on this scan (path lengths, point
  cloud bounds, scanpoint count).
- Re-implementing a `.ply` loader instead of using `three.js`'s `PLYLoader`.
- Adding dependencies (auth, state libraries, build pipelines) the task
  doesn't need.

---

## 8 · Stack / tooling

**Free choice.** Most candidates pick `three.js` because the point-cloud
ecosystem is rich there, but `babylon.js`, `react-three-fiber`, a
WebGL/WGPU project, or even native (Open3D, Blender add-on) are all fine.
Be explicit about how we run whatever you build.

---

## 9 · Deliverables

- A git repo (zip is fine if you can't share access).
- `README.md` at the root: setup instructions + one paragraph on what
  you'd do with another week.
- A short video or markdown walkthrough.

Send to: chan@infrascan-ai.com — or the recruiter who briefed you.

Good luck. Have fun with it.

— Chan, InfraScan team
