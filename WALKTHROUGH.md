# Walkthrough

A depth-supervised 3D Gaussian Splat reconstruction of the office scan, with a
browser viewer built around one idea: don't just show the reconstruction, show
*how much to trust each part of it*. Setup instructions are in
[`README.md`](README.md); the full design rationale and every alternative
that didn't make the cut are in [`RESEARCH.md`](RESEARCH.md). This page is the
five-minute version.

## What works

- **Walk through it.** WASD + mouse-look, first-person, real-time. Click the
  scene to lock the pointer, Esc to release.
- **See what's trustworthy and what isn't.** Switch to Coverage mode and drag
  the slider — points seen by fewer of the 240 source photos fade out in real
  time, showing exactly where the reconstruction is well-constrained versus
  guessed. Confidence heat-map and colour-by-supervising-view work the same way, each
  self-explaining in the on-screen status line.
- **Click a point, see the photo that built it — both directions.** Click
  anywhere on the 3D reconstruction and the exact source photo that
  supervised that surface pops up. Click a pixel in that photo and it
  ray-casts through its depth map to drop a 3D marker back at that point.
- **A floor-plan minimap** with a real top-down density projection and a
  clickable walkable path — click anywhere on it to fly there.
- **The floor is complete, honestly.** The capture never observes the floor
  beyond a coverage bubble in the middle of the room. That region is
  generatively filled and baked into the model as real Gaussians — but tagged
  unsupervised in the trust data, so Coverage/Confidence mode still correctly
  shows it as generated, not observed.

## What's broken

- **Every quality number in this project is training-view, not held-out.**
  All 240 source views were used in training; none were excluded for
  evaluation. This is the biggest methodological gap here — see
  `RESEARCH.md` §4 for the concrete held-out-split plan that would fix it.
- **Windows and reflective surfaces are soft and low-detail** — DA3's depth
  confidence is genuinely low there, and a single-vantage capture gives no
  cross-view signal to resolve them. There's a confidence-gated image-diffusion
  pass that sharpens them as an offline comparison (`RESEARCH.md` §3.4), but
  it's cosmetic — it doesn't fix the underlying geometry, and it fabricates
  plausible-looking detail (like signage text) that isn't actually there.
- **One person slipped through masking** — visible at yaw 240 across every
  scan-point. Root-caused to a single upstream detection miss in the provided
  labels, not a bug in this project's masking logic (`RESEARCH.md` §4).
- **Floater-pruning is built but shipped disabled.** A first version — using
  data already computed for the trust viewer to prune ungrounded Gaussians —
  regressed on deployment (a fixed distance threshold over-pruned sparse but
  legitimate geometry far from the cameras, visibly breaking the far wall).
  Reverted immediately; a better version is prototyped but not yet validated
  enough to ship (`RESEARCH.md` §3.2).

## What I'd build next with another week

1. **A real held-out evaluation** — exclude every 12th view from training,
   re-run every comparison in `RESEARCH.md` on the excluded views only. This
   is the one thing that would turn "directionally suggestive" numbers into
   actual proof.
2. **Finish validating the floater-pruning fix** — a proper sweep across many
   views, not the single rendered comparison it has now.
3. **The real ArtiFixer** (or an equivalent opacity-mixing model), given an
   80GB-class GPU — the confidence/coverage data already computed for the
   trust viewer is the right control signal for it.
