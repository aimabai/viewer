"""Complete the FLOOR as a top-down texture via CLASSICAL (non-generative) inpainting.

Single-pitch capture means the floor is never directly observed (README §5) — zero
pixels, not a training deficiency (see RESEARCH.md). We project the real point
cloud's colour onto the floor plane (a top-down "map view"), mask the unobserved
holes, and extend the REAL nearby texture into them with OpenCV's classical
inpainting (Telea 2004) — no generative model, so nothing is invented that wasn't
directly extrapolated from adjacent real pixels. This is a deliberate downgrade from
an earlier SDXL-diffusion version of this script: diffusion produced a more polished
image but can hallucinate detail that was never there; for an inspection tool,
"visibly extrapolated from real data" was judged more trustworthy than "prettier but
invented." See RESEARCH.md for both attempts and the reasoning.

Ceiling is intentionally left alone (no texture, flat colour fallback in the viewer)
— it's less regular/predictable than a floor, so naive nearby-pixel extrapolation is
less defensible there; not attempted.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from plyfile import PlyData

from data import load_views  # noqa: F401 (kept for parity/future use)


def project_topdown(fp, col, e1, e2, up, height, band, bounds, res_u):
    """Rasterize points near `height` (along up) into a top-down RGB image + hole mask."""
    h = fp @ up
    sel = np.abs(h - height) < band
    pts, colp = fp[sel], col[sel]
    u = pts @ e1
    v = pts @ e2
    (umin, vmin), (umax, vmax) = bounds["min"], bounds["max"]
    res_v = max(1, int(res_u * (vmax - vmin) / (umax - umin)))

    ui = np.clip(((u - umin) / (umax - umin) * res_u).astype(int), 0, res_u - 1)
    vi = np.clip(((v - vmin) / (vmax - vmin) * res_v).astype(int), 0, res_v - 1)

    sum_rgb = np.zeros((res_v, res_u, 3), dtype=np.float64)
    count = np.zeros((res_v, res_u), dtype=np.int64)
    np.add.at(sum_rgb, (vi, ui), colp)
    np.add.at(count, (vi, ui), 1)

    hole = count < 2
    rgb = np.zeros((res_v, res_u, 3), dtype=np.uint8)
    seen = ~hole
    rgb[seen] = np.clip(sum_rgb[seen] / count[seen, None] * 255, 0, 255).astype(np.uint8)
    return rgb, hole


def classical_inpaint(rgb, hole, radius=12):
    """Extend the real (unmasked) pixels into the holes via OpenCV Telea inpainting —
    propagates nearby real colour/texture, invents nothing beyond that extrapolation."""
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    mask = (hole * 255).astype(np.uint8)
    # dilate slightly so the inpainter blends from a clean interior, not noisy edge pixels
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    out = cv2.inpaint(bgr, mask, radius, cv2.INPAINT_TELEA)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init-ply", default="../prepared/init_pointcloud.ply")
    ap.add_argument("--out-dir", default="../viewer/public/scene")
    ap.add_argument("--meta", default="../viewer/public/scene/scene_meta.json")
    ap.add_argument("--res", type=int, default=512, help="top-down projection resolution")
    ap.add_argument("--pad", type=float, default=1.15, help="expand bounds beyond measured cloud")
    ap.add_argument("--inpaint-radius", type=int, default=12)
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    up = np.array(meta["world_up"])
    e1 = np.array(meta["floor_basis"]["e1"])
    e2 = np.array(meta["floor_basis"]["e2"])
    floor_h = meta["inferred_planes"]["floor_height"]
    ceil_h = meta["inferred_planes"]["ceiling_height"]

    p = PlyData.read(args.init_ply)["vertex"]
    fp = np.stack([np.asarray(p["x"]), np.asarray(p["y"]), np.asarray(p["z"])], axis=1)
    col = np.stack([np.asarray(p["red"]), np.asarray(p["green"]), np.asarray(p["blue"])], axis=1) / 255.0

    rb = meta["reconstructed_bounds"]
    cu, cv_ = (rb["min"][0] + rb["max"][0]) / 2, (rb["min"][1] + rb["max"][1]) / 2
    half_u = (rb["max"][0] - rb["min"][0]) / 2 * args.pad
    half_v = (rb["max"][1] - rb["min"][1]) / 2 * args.pad
    bounds = {"min": [cu - half_u, cv_ - half_v], "max": [cu + half_u, cv_ + half_v]}
    band = (ceil_h - floor_h) * 0.06

    rgb, hole = project_topdown(fp, col, e1, e2, up, floor_h, band, bounds, args.res)
    print(f"floor: {hole.mean()*100:.1f}% unobserved, classical inpainting (Telea)…")
    completed = classical_inpaint(rgb, hole, args.inpaint_radius)

    from PIL import Image
    out_dir = Path(args.out_dir)
    Image.fromarray(completed).save(out_dir / "floor_texture.png")
    Image.fromarray(rgb).save(out_dir / "floor_texture_raw.png")
    # remove any stale diffusion-era ceiling texture so the viewer's flat fallback is used
    (out_dir / "ceiling_texture.png").unlink(missing_ok=True)
    print(f"wrote {out_dir}/floor_texture.png")

    meta["inferred_planes"]["plane_bounds"] = bounds
    meta["inferred_planes"]["floor_textured"] = True
    meta["inferred_planes"]["ceiling_textured"] = False
    meta["inferred_planes"].pop("textured", None)   # old combined flag, superseded
    Path(args.meta).write_text(json.dumps(meta, indent=2))
    print("updated scene_meta.json (floor_textured=true, ceiling_textured=false)")


if __name__ == "__main__":
    main()
