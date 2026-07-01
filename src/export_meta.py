"""Write viewer/public/scene/scene_meta.json — data for the floor-plan minimap and
camera-frustum overlay. Pure geometry from cameras.json + the point clouds.

The scan's coordinate frame is NOT axis-aligned (its vertical is ~31° off the Y axis),
so a top-down map can't just use x-z. We compute the true world-up from the camera
poses, build an orthonormal FLOOR BASIS (e1, e2) perpendicular to it, and express all
2D map coordinates as (p·e1, p·e2). The viewer projects the live camera the same way.

Contents:
  world_up          : unit up vector (for the 3D viewer)
  floor_basis       : {e1, e2} — 3D axes spanning the floor plane (for the minimap)
  floor_bounds      : 2D extent of the WHOLE floor (dataset/pointcloud.ply) — map frame
  reconstructed_bounds : 2D extent of what we reconstructed (init cloud)
  coverage_hull     : convex polygon (2D floor coords) of the reconstructed footprint
  scan_points[]     : 20 operator positions (3D pos + 2D floor xy) with their 12 yaws
  views[]           : 240 views with 3D pose, 2D floor xy, forward azimuth, intrinsics
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from plyfile import PlyData

from data import load_views


def floor_basis_from_up(up: np.ndarray):
    """Two orthonormal horizontal axes (e1, e2) spanning the plane perpendicular to up."""
    ref = np.array([1.0, 0.0, 0.0]) if abs(up[0]) < 0.9 else np.array([0.0, 0.0, 1.0])
    e1 = ref - up * (ref @ up)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(up, e1)
    e2 /= np.linalg.norm(e2)
    return e1, e2


def ply_floor_bounds(path: str, e1, e2):
    p = PlyData.read(path)["vertex"]
    pts = np.stack([np.asarray(p["x"]), np.asarray(p["y"]), np.asarray(p["z"])], axis=1)
    u, v = pts @ e1, pts @ e2
    return {"min": [float(u.min()), float(v.min())], "max": [float(u.max()), float(v.max())]}


def convex_hull_floor(path: str, e1, e2, sample: int = 200_000):
    """Monotone-chain convex hull of the cloud projected onto the floor plane (no scipy)."""
    p = PlyData.read(path)["vertex"]
    pts = np.stack([np.asarray(p["x"]), np.asarray(p["y"]), np.asarray(p["z"])], axis=1)
    if len(pts) > sample:
        pts = pts[np.random.default_rng(0).choice(len(pts), sample, replace=False)]
    pts2 = np.unique(np.stack([pts @ e1, pts @ e2], axis=1), axis=0)
    pts2 = pts2[np.lexsort((pts2[:, 1], pts2[:, 0]))]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for pt in pts2:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0:
            lower.pop()
        lower.append(pt)
    upper = []
    for pt in pts2[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0:
            upper.pop()
        upper.append(pt)
    hull = np.array(lower[:-1] + upper[:-1])
    return [[float(a), float(b)] for a, b in hull]


def write_floorplan(path, init_ply, e1, e2, bounds, res_u=340):
    """Top-down density image of the reconstructed cloud projected onto the floor.
    Walls (many points stacked vertically) read as bright lines, so it looks like an
    actual floor plan. Saved RGBA (transparent where empty) for the minimap background."""
    import matplotlib.cm as cm
    import imageio.v3 as iio
    p = PlyData.read(init_ply)["vertex"]
    pts = np.stack([np.asarray(p["x"]), np.asarray(p["y"]), np.asarray(p["z"])], axis=1)
    u, v = pts @ e1, pts @ e2
    (umin, vmin), (umax, vmax) = bounds["min"], bounds["max"]
    res_v = max(1, int(res_u * (vmax - vmin) / (umax - umin)))
    hist, _, _ = np.histogram2d(v, u, bins=[res_v, res_u], range=[[vmin, vmax], [umin, umax]])
    d = np.log1p(hist); d = d / (d.max() + 1e-9)
    rgba = cm.get_cmap("cividis")(d)
    rgba[..., 3] = np.clip(d * 1.7, 0, 1)            # alpha by density → empty = transparent
    iio.imwrite(path, (rgba * 255).astype(np.uint8))
    print(f"wrote {path} | floor-plan {res_u}x{res_v}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--init-ply", default="prepared/init_pointcloud.ply")
    ap.add_argument("--out", default="viewer/public/scene/scene_meta.json")
    args = ap.parse_args()

    views = load_views(args.dataset)

    # True world-up (cameras are pitch-0, so camera DOWN axis = world down), then a
    # floor basis perpendicular to it. All 2D map coords are (p·e1, p·e2).
    downs = np.array([np.asarray(v.c2w[:3, 1]) for v in views])
    up = -downs.mean(0); up /= np.linalg.norm(up)
    e1, e2 = floor_basis_from_up(up)

    def proj(p):
        p = np.asarray(p)
        return [float(p @ e1), float(p @ e2)]

    view_entries, by_frame = [], {}
    for v in views:
        fwd = v.c2w[:3, 2]
        pos = v.c2w[:3, 3]
        H, W = v.hw
        fx, fy, cx, cy = float(v.K[0, 0]), float(v.K[1, 1]), float(v.K[0, 2]), float(v.K[1, 2])
        fwd2 = proj(fwd)
        e = {"id": v.id, "frame": v.frame, "yaw": v.yaw,
             "pos": [float(c) for c in pos], "xy": proj(pos),
             "forward_xy": fwd2, "azimuth_deg": round(math.degrees(math.atan2(fwd2[1], fwd2[0])), 1),
             "R": [float(x) for x in v.c2w[:3, :3].reshape(-1)],
             "intr": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "w": int(W), "h": int(H)},
             "pano": v.pano}
        view_entries.append(e)
        by_frame.setdefault(v.frame, []).append(e)

    scan_points = []
    for frame in sorted(by_frame):
        grp = by_frame[frame]
        c = np.mean([g["pos"] for g in grp], axis=0)
        scan_points.append({
            "frame": frame, "pos": [float(x) for x in c], "xy": proj(c),
            "view_ids": [g["id"] for g in grp],
            "yaws": [{"id": g["id"], "yaw": g["yaw"], "azimuth_deg": g["azimuth_deg"]} for g in grp],
        })

    # Walkable path: principal axis of the reconstructed floor footprint, at eye height.
    # (Capture is a near-single vantage, so this is a synthetic route to glide the camera
    # through the room rather than a recorded trajectory.)
    pp = PlyData.read(args.init_ply)["vertex"]
    fp = np.stack([np.asarray(pp["x"]), np.asarray(pp["y"]), np.asarray(pp["z"])], axis=1)
    col = np.stack([np.asarray(pp["red"]), np.asarray(pp["green"]), np.asarray(pp["blue"])], axis=1) / 255.0
    uv = np.stack([fp @ e1, fp @ e2], axis=1)

    # Floor & ceiling are never directly seen (single horizontal pitch), so they render
    # as garbage. They are trivially-known PLANES, though: fit their height along world-up
    # from the observed geometry and their colour from the nearest-height points. The viewer
    # renders clean inferred planes there (marked as inferred — completion, not hallucination).
    h = fp @ up
    floor_h, ceil_h = float(np.percentile(h, 2)), float(np.percentile(h, 98))
    band = (ceil_h - floor_h) * 0.06
    floor_col = col[np.abs(h - floor_h) < band].mean(0) if (np.abs(h - floor_h) < band).any() else np.array([.4, .4, .4])
    ceil_col = col[np.abs(h - ceil_h) < band].mean(0) if (np.abs(h - ceil_h) < band).any() else np.array([.2, .2, .2])
    inferred_planes = {
        "floor_height": floor_h, "ceiling_height": ceil_h,
        "floor_color": [float(x) for x in floor_col], "ceiling_color": [float(x) for x in ceil_col],
    }
    c2 = uv.mean(0)
    evals, evecs = np.linalg.eigh((uv - c2).T @ (uv - c2))
    axis = evecs[:, -1]                                   # principal (longest) direction
    t = (uv - c2) @ axis
    lo, hi = np.percentile(t, [4, 96])
    eye = float(np.mean([np.asarray(sp["pos"]) @ up for sp in scan_points]))
    path = []
    for s in np.linspace(lo, hi, 16):
        p2 = c2 + axis * s
        p3 = e1 * p2[0] + e2 * p2[1] + up * eye
        path.append({"pos": [float(x) for x in p3], "xy": [float(p2[0]), float(p2[1])]})

    meta = {
        "world_up": [float(x) for x in up],
        "floor_basis": {"e1": [float(x) for x in e1], "e2": [float(x) for x in e2]},
        "inferred_planes": inferred_planes,
        "path": path,
        "floor_bounds": ply_floor_bounds(str(Path(args.dataset) / "pointcloud.ply"), e1, e2),
        "reconstructed_bounds": ply_floor_bounds(args.init_ply, e1, e2),
        "coverage_hull": convex_hull_floor(args.init_ply, e1, e2),
        "scan_points": scan_points,
        "views": view_entries,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2))
    write_floorplan(str(out.parent / "floorplan.png"), args.init_ply, e1, e2,
                    meta["reconstructed_bounds"])
    fb, rb = meta["floor_bounds"], meta["reconstructed_bounds"]
    print(f"wrote {out} | {len(scan_points)} scan-points, {len(view_entries)} views, "
          f"hull {len(meta['coverage_hull'])} pts")
    print(f"floor extent: {round(fb['max'][0]-fb['min'][0],1)} x {round(fb['max'][1]-fb['min'][1],1)} m | "
          f"reconstructed: {round(rb['max'][0]-rb['min'][0],1)} x {round(rb['max'][1]-rb['min'][1],1)} m")


if __name__ == "__main__":
    main()
