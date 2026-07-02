"""Export a trained checkpoint to a viewer-ready Gaussian-splat PLY.

Two jobs:
  1. Write the standard INRIA/3DGS .ply layout (x,y,z, normals, f_dc_*, f_rest_*,
     opacity, scale_*, rot_*) so any gsplat/three.js loader can render it. Raw
     (pre-activation) values are stored, as viewers expect (sigmoid opacity,
     exp scale, normalize quat).
  2. Add THREE extra per-Gaussian fields driving the trust-aware viewer:
       - confidence       : DA3 conf at the best supervising view (0..1)
       - supervising_view  : id of that view (-1 if unseen)
       - n_views           : how many views see this Gaussian (coverage)
     These are derived post-hoc by projecting each Gaussian into all views and
     keeping only depth-consistent, non-person pixels (see RESEARCH.md — provenance
     can't be tracked through densification, so we recover it here).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

from data import load_views


def axis_angle_to_R_np(aa: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(aa) + 1e-8
    axis = aa / theta
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def refine_c2w(views, pose_rot: np.ndarray, pose_trans: np.ndarray):
    """Apply each view's trained pose correction (see train.py --refine-poses) to its
    c2w, so provenance/coverage is projected with the SAME cameras the Gaussians were
    actually fit against — otherwise even a few-cm/degree mismatch measurably corrupts
    which points count as 'seen' by which view (confirmed: cost ~6dB in eval before
    this fix)."""
    out = {}
    for i, v in enumerate(views):
        delta = np.eye(4, dtype=np.float64)
        delta[:3, :3] = axis_angle_to_R_np(pose_rot[i])
        delta[:3, 3] = pose_trans[i]
        viewmat_refined = delta @ np.linalg.inv(v.c2w.astype(np.float64))
        c2w_refined = np.linalg.inv(viewmat_refined)
        out[v.id] = c2w_refined.astype(np.float32)
    return out


@torch.no_grad()
def compute_provenance(means: torch.Tensor, views, max_depth: float,
                       depth_tol: float, device: str, refined_c2w: dict | None = None):
    """Return (confidence Nf32, supervising_view Ni32, n_views Ni32).

    A Gaussian is 'seen' by a view if it projects inside the image, in front of
    the camera, on a non-person pixel, and its camera-z agrees with DA3 depth to
    within `depth_tol` (relative). The supervising view is the highest-confidence
    such view; confidence is DA3 conf there.
    """
    N = means.shape[0]
    means = means.to(device)
    best_conf = torch.full((N,), -1.0, device=device)
    best_view = torch.full((N,), -1, dtype=torch.int32, device=device)
    n_views = torch.zeros(N, dtype=torch.int32, device=device)

    for v in views:
        H, W = v.hw
        c2w = refined_c2w[v.id] if refined_c2w is not None else v.c2w
        R = torch.tensor(c2w[:3, :3], device=device)
        t = torch.tensor(c2w[:3, 3], device=device)
        K = torch.tensor(v.K, device=device)
        depth = torch.tensor(v.depth, device=device)
        conf = torch.tensor(v.conf, device=device)
        person = torch.tensor(v.person_mask, device=device)

        cam = (means - t) @ R                       # world->camera (R orthonormal)
        z = cam[:, 2]
        u = (K[0, 0] * cam[:, 0] / z + K[0, 2])
        vv = (K[1, 1] * cam[:, 1] / z + K[1, 2])
        ui, vi = u.long(), vv.long()
        inb = (z > 0.05) & (z < max_depth) & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)

        ui_c = ui.clamp(0, W - 1); vi_c = vi.clamp(0, H - 1)
        d_gt = depth[vi_c, ui_c]
        consistent = (torch.abs(z - d_gt) / (d_gt + 1e-6)) < depth_tol
        not_person = person[vi_c, ui_c] == 0
        seen = inb & consistent & not_person & (d_gt > 0)

        n_views += seen.int()
        c = conf[vi_c, ui_c]
        better = seen & (c > best_conf)
        best_conf = torch.where(better, c, best_conf)
        best_view = torch.where(better, torch.full_like(best_view, v.id), best_view)

    conf_out = best_conf.clamp(min=0.0)
    # normalise confidence to 0..1 across the seen Gaussians for stable shading
    seen_mask = best_view >= 0
    if seen_mask.any():
        lo = conf_out[seen_mask].min()
        hi = conf_out[seen_mask].max()
        conf_out = ((conf_out - lo) / (hi - lo + 1e-8)).clamp(0, 1)
    return (conf_out.cpu().numpy(), best_view.cpu().numpy(), n_views.cpu().numpy())


def prune_floaters(means: np.ndarray, n_views: np.ndarray, dist_thresh: float) -> np.ndarray:
    """Boolean keep-mask that drops genuine floaters: Gaussians that are BOTH never
    confirmed by the depth-consistency check (n_views==0) AND spatially isolated
    from any confirmed surface (> dist_thresh from the nearest n_views>=1 point).

    Neither signal alone is a good floater test on its own — most n_views==0
    points sit within a few cm of real geometry (they just missed the strict
    depth-consistency tolerance) and pruning all of them would strip real detail;
    requiring BOTH "never confirmed" and "far from anything that was" is a much
    more targeted floater signal (see RESEARCH.md's floater-pruning section for
    the measured distance distribution that motivated the default threshold)."""
    seen = n_views >= 1
    if not seen.any() or seen.all():
        return np.ones(len(means), dtype=bool)
    tree = cKDTree(means[seen])
    d, _ = tree.query(means[~seen], k=1)
    isolated = np.zeros(len(means), dtype=bool)
    unseen_idx = np.nonzero(~seen)[0]
    isolated[unseen_idx[d > dist_thresh]] = True
    return ~isolated


def write_ply(path, means, sh0, shN, opacities, scales, quats):
    """Standard INRIA/3DGS .ply (no custom fields, for max loader compatibility).
    Provenance for the viewer lives in trust.bin instead."""
    n = means.shape[0]
    sh0 = sh0.reshape(n, 3)                       # (N,3) DC term
    shN = shN.reshape(n, -1, 3)                   # (N, K-1, 3)
    # INRIA f_rest ordering is channel-major: R's K-1 coeffs, then G, then B.
    f_rest = shN.transpose(0, 2, 1).reshape(n, -1)

    fields = [("x", "f4"), ("y", "f4"), ("z", "f4"),
              ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
              ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")]
    fields += [(f"f_rest_{i}", "f4") for i in range(f_rest.shape[1])]
    fields += [("opacity", "f4"), ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
               ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4")]

    arr = np.empty(n, dtype=fields)
    arr["x"], arr["y"], arr["z"] = means[:, 0], means[:, 1], means[:, 2]
    arr["nx"] = arr["ny"] = arr["nz"] = 0.0
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = sh0[:, 0], sh0[:, 1], sh0[:, 2]
    for i in range(f_rest.shape[1]):
        arr[f"f_rest_{i}"] = f_rest[:, i]
    arr["opacity"] = opacities
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = scales[:, 0], scales[:, 1], scales[:, 2]
    arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"] = (
        quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3])

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, "vertex")], byte_order="<").write(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/full.pt")
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--out", default="viewer/public/scene/scene.ply")
    ap.add_argument("--max-depth", type=float, default=20.0)
    ap.add_argument("--depth-tol", type=float, default=0.1, help="relative z agreement")
    ap.add_argument("--device", default="cuda", help="cuda or cpu (cpu avoids contending with a running train)")
    ap.add_argument("--min-opacity", type=float, default=0.02, help="prune Gaussians below this opacity")
    ap.add_argument("--max-scale", type=float, default=0.5, help="prune Gaussians with max-scale (m) above this")
    ap.add_argument("--floater-dist", type=float, default=0.0,
                    help="prune Gaussians with n_views==0 AND farther than this (m) "
                         "from the nearest depth-confirmed point — genuine floaters, "
                         "not just near-surface points that missed the consistency "
                         "check. 0 disables (default: the fixed-threshold version "
                         "over-prunes sparse-but-legitimate far geometry — see "
                         "RESEARCH.md — so it's opt-in until a density-adaptive "
                         "version replaces it)")
    ap.add_argument("--trust-bin", default="viewer/public/scene/trust.bin",
                    help="compact point cloud for trust-rendering modes ('' to skip)")
    ap.add_argument("--trust-points", type=int, default=400_000, help="max points in trust.bin")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    # Prune obvious junk before export: near-transparent or oversized Gaussians.
    # This both cleans floaters and shrinks the viewer PLY.
    op = torch.sigmoid(ck["opacities"]).numpy()
    maxsc = torch.exp(ck["scales"]).numpy().max(1)
    keep = (op >= args.min_opacity) & (maxsc <= args.max_scale)
    n0 = len(op)
    idx = np.nonzero(keep)[0]
    means = ck["means"][idx]
    sh0, shN = ck["sh0"][idx], ck["shN"][idx]
    opac, scales, quats = ck["opacities"][idx], ck["scales"][idx], ck["quats"][idx]
    print(f"{n0:,} Gaussians -> {len(idx):,} after prune "
          f"(opacity>={args.min_opacity}, scale<={args.max_scale}m)")

    views = load_views(args.dataset)
    refined_c2w = None
    if "pose_rot" in ck:
        refined_c2w = refine_c2w(views, ck["pose_rot"].numpy(), ck["pose_trans"].numpy())
        print("using trained per-camera pose corrections for provenance projection")
    print(f"computing provenance over {len(views)} views...")
    conf, sview, nv = compute_provenance(means, views, args.max_depth, args.depth_tol,
                                        args.device, refined_c2w)

    if args.floater_dist > 0:
        means_np = means.numpy()
        keep2 = prune_floaters(means_np, nv, args.floater_dist)
        n1 = len(means_np)
        means, sh0, shN = means[keep2], sh0[keep2], shN[keep2]
        opac, scales, quats = opac[keep2], scales[keep2], quats[keep2]
        conf, sview, nv = conf[keep2], sview[keep2], nv[keep2]
        print(f"floater prune: {n1:,} -> {len(means):,} "
              f"({n1 - len(means):,} isolated unconfirmed points removed, "
              f">{args.floater_dist*100:.0f}cm from the nearest confirmed surface)")

    write_ply(args.out, means.numpy(), sh0.numpy(), shN.numpy(),
              opac.numpy(), scales.numpy(), quats.numpy())
    pct_seen = 100.0 * (sview >= 0).mean()
    print(f"wrote {args.out} | {pct_seen:.1f}% Gaussians supervised | "
          f"median n_views {int(np.median(nv))}")

    if args.trust_bin:
        write_trust_bin(args.trust_bin, means.numpy(), sh0.numpy(), conf, sview, nv,
                        args.trust_points)


def write_trust_bin(path, means, sh0, conf, sview, nv, max_points):
    """Compact binary point cloud for the viewer's trust-rendering modes.

    Layout: int32 count, then contiguous arrays:
      pos  float32[count*3]
      rgb  uint8[count*3]     (DC colour)
      conf uint8[count]       (0..255 = confidence 0..1)
      nv   uint8[count]       (n_views, clamped to 255)
      view int16[count]       (supervising view id, -1 if unseen)
    """
    n = len(means)
    if n > max_points:
        sel = np.random.default_rng(0).choice(n, max_points, replace=False)
        means, sh0, conf, sview, nv = means[sel], sh0[sel], conf[sel], sview[sel], nv[sel]
    rgb = np.clip(0.5 + 0.28209479177387814 * sh0.reshape(-1, 3), 0, 1)
    buf = bytearray()
    buf += np.int32(len(means)).tobytes()
    buf += means.astype("<f4").tobytes()
    buf += (rgb * 255).astype(np.uint8).tobytes()
    buf += np.clip(conf * 255, 0, 255).astype(np.uint8).tobytes()
    buf += np.clip(nv, 0, 255).astype(np.uint8).tobytes()
    buf += sview.astype("<i2").tobytes()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(buf)
    print(f"wrote {path} | {len(means):,} trust points")


if __name__ == "__main__":
    main()
