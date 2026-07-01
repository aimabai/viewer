"""Lift per-view DA3 depth maps into a single coloured world-space point cloud.

This serves two purposes:
  1. A standalone reconstruction deliverable (a denser, view-consistent cloud built
     only from the 20 scan-points we actually have coverage for).
  2. The initialization for gsplat training (well-placed Gaussians converge faster
     and float less than random init).

Pixels are dropped if they were a person (mask) or low-confidence (per-frame
quantile on DA3 `conf`), then backprojected with the camera intrinsics/pose and
voxel-downsampled so overlapping views don't pile up duplicate points.

Multi-view consistency filtering (--consistency-check): our capture is a single
rotating vantage (20 scan-points in a ~1.5m^3 bubble, no real translation/parallax),
so DA3's monocular depth noise is otherwise never cross-checked against anything.
Adjacent yaws (30 deg apart, same scan-point) have heavy FOV overlap; a genuine
surface point should get a similar depth estimate from both. We project each
candidate point into its scan-point's neighbouring yaw views and drop it only if a
neighbour has a valid, comparable pixel there AND actively disagrees beyond
tolerance (absence of a corroborating view — e.g. FOV edge — is not penalised).
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import open3d as o3d

from data import View, load_views


def backproject(view: View, conf_quantile: float, max_depth: float, stride: int):
    """Return (points_world Nx3, colors Nx3 float[0,1], depth_cam N, us N, vs N)."""
    H, W = view.hw
    fx, fy = view.K[0, 0], view.K[1, 1]
    cx, cy = view.K[0, 2], view.K[1, 2]

    vs, us = np.mgrid[0:H:stride, 0:W:stride]
    d = view.depth[vs, us]
    conf = view.conf[vs, us]
    keep = view.person_mask[vs, us] == 0          # not a person
    keep &= (d > 0) & (d < max_depth)             # valid, near-enough depth
    if conf_quantile > 0:
        valid = conf[keep] if keep.any() else conf.ravel()
        thr = np.quantile(valid, conf_quantile) if valid.size else 0.0
        keep &= conf >= thr                       # drop lowest-confidence pixels

    us, vs, d = us[keep], vs[keep], d[keep]
    x = (us - cx) * d / fx
    y = (vs - cy) * d / fy
    pts_cam = np.stack([x, y, d], axis=1)
    pts_world = pts_cam @ view.c2w[:3, :3].T + view.c2w[:3, 3]
    colors = view.image[vs, us, :3].astype(np.float32) / 255.0
    return pts_world.astype(np.float32), colors, d.astype(np.float32)


def _project_into(other: View, pts_world: np.ndarray, max_depth: float):
    """Project world points into `other`'s camera; return (depth_in_other, sampled_depth,
    valid_mask). depth_in_other = implied depth if this point is really there;
    sampled_depth = what other's own DA3 depth map says at that pixel."""
    cam = (pts_world - other.c2w[:3, 3]) @ other.c2w[:3, :3]     # world -> other's camera frame
    z = cam[:, 2]
    fx, fy = other.K[0, 0], other.K[1, 1]
    cx, cy = other.K[0, 2], other.K[1, 2]
    u = (fx * cam[:, 0] / np.maximum(z, 1e-6) + cx)
    v = (fy * cam[:, 1] / np.maximum(z, 1e-6) + cy)
    H, W = other.hw
    ui, vi = u.astype(int), v.astype(int)
    valid = (z > 0.05) & (z < max_depth) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    sampled = np.zeros_like(z)
    sampled[valid] = other.depth[vi[valid], ui[valid]]
    valid &= sampled > 0
    return z, sampled, valid


def consistency_filter(views, tol: float, max_depth: float, stride: int, conf_quantile: float):
    """Backproject every view, then drop points whose depth actively disagrees with an
    adjacent-yaw view of the same scan-point. Returns stacked (points, colors)."""
    by_frame = defaultdict(list)
    for v in views:
        by_frame[v.frame].append(v)
    neighbours = {}   # view.id -> [prev_view, next_view] (same scan-point, adjacent yaw)
    for frame, grp in by_frame.items():
        grp = sorted(grp, key=lambda v: v.yaw)
        n = len(grp)
        for i, v in enumerate(grp):
            neighbours[v.id] = [grp[(i - 1) % n], grp[(i + 1) % n]]

    kept_pts, kept_col = [], []
    n_before = n_after = 0
    for v in views:
        pts, col, d_self = backproject(v, conf_quantile, max_depth, stride)
        n_before += len(pts)
        keep = np.ones(len(pts), dtype=bool)
        for other in neighbours[v.id]:
            if other.id == v.id:
                continue
            z_other_frame, sampled, valid = _project_into(other, pts, max_depth)
            denom = np.maximum(np.maximum(z_other_frame, sampled), 1e-6)
            disagree = valid & (np.abs(z_other_frame - sampled) / denom > tol)
            keep &= ~disagree
        kept_pts.append(pts[keep]); kept_col.append(col[keep])
        n_after += int(keep.sum())
    print(f"consistency filter: {n_before:,} -> {n_after:,} pts "
          f"({100*(1-n_after/max(n_before,1)):.1f}% dropped as adjacent-view-inconsistent)")
    return np.concatenate(kept_pts), np.concatenate(kept_col)


def lift_all(views, conf_quantile, max_depth, stride):
    """Backproject every view; return stacked (points Nx3, colors Nx3) world arrays."""
    all_pts, all_col = [], []
    for v in views:
        p, c, _ = backproject(v, conf_quantile, max_depth, stride)
        all_pts.append(p)
        all_col.append(c)
    return np.concatenate(all_pts), np.concatenate(all_col)


def build(views, conf_quantile, max_depth, stride, voxel, consistency_check, consistency_tol):
    if consistency_check:
        pts, col = consistency_filter(views, consistency_tol, max_depth, stride, conf_quantile)
    else:
        pts, col = lift_all(views, conf_quantile, max_depth, stride)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(col)
    if voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)
    return pcd, len(pts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--out", default="prepared/init_pointcloud.ply")
    ap.add_argument("--conf-quantile", type=float, default=0.20,
                    help="drop the lowest-confidence fraction of pixels per frame")
    ap.add_argument("--max-depth", type=float, default=20.0)
    ap.add_argument("--stride", type=int, default=2, help="pixel subsampling")
    ap.add_argument("--voxel", type=float, default=0.02, help="downsample voxel (m); 0=off")
    ap.add_argument("--consistency-check", action="store_true",
                    help="drop points whose depth disagrees with an adjacent-yaw view")
    ap.add_argument("--consistency-tol", type=float, default=0.12, help="relative depth tolerance")
    args = ap.parse_args()

    views = load_views(args.dataset)
    pcd, n_raw = build(views, args.conf_quantile, args.max_depth, args.stride, args.voxel,
                       args.consistency_check, args.consistency_tol)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    o3d.io.write_point_cloud(str(out), pcd)
    print(f"{n_raw:,} raw pts -> {len(pcd.points):,} after voxel({args.voxel}m) -> {out}")


if __name__ == "__main__":
    main()
