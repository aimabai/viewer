"""Redeploy the viewer scene from a new checkpoint, reusing the ALREADY-GENERATED
floor completion (viewer/public/scene/floor_texture.png) instead of re-running
SDXL-inpainting — the floor hole/fill only depends on the point-cloud geometry
(same init_pointcloud.ply for every checkpoint), not on which trained checkpoint
supplies the real Gaussians. Saves a ~6.5GB model re-download for what would be
an identical floor image. See inpaint_planes.py for the original generation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from plyfile import PlyData

from data import load_views
from export import compute_provenance, refine_c2w, write_ply, write_trust_bin, prune_floaters
from inpaint_planes import project_topdown, make_floor_gaussians


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../outputs/full_v5_gradedge.pt")
    ap.add_argument("--dataset", default="../dataset")
    ap.add_argument("--init-ply", default="../prepared/init_pointcloud_consistent.ply")
    ap.add_argument("--out-scene", default="../viewer/public/scene/scene.ply")
    ap.add_argument("--out-trust", default="../viewer/public/scene/trust.bin")
    ap.add_argument("--meta", default="../viewer/public/scene/scene_meta.json")
    ap.add_argument("--floor-texture", default="../viewer/public/scene/floor_texture.png")
    ap.add_argument("--res", type=int, default=384)
    ap.add_argument("--pad", type=float, default=1.15)
    ap.add_argument("--min-opacity", type=float, default=0.02)
    ap.add_argument("--max-scale", type=float, default=0.5)
    ap.add_argument("--max-aniso", type=float, default=12.0)
    ap.add_argument("--floater-dist", type=float, default=0.0,
                    help="prune real Gaussians with n_views==0 AND farther than this "
                         "(m) from the nearest depth-confirmed point. 0 disables (default: "
                         "the fixed-threshold version is known to over-prune sparse-but-"
                         "legitimate geometry far from the scan-point cluster — see "
                         "RESEARCH.md's floater-pruning section — so it is opt-in, not "
                         "the shipped default, until a density-adaptive version replaces it)")
    ap.add_argument("--max-points", type=int, default=1_200_000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    meta = json.loads(Path(args.meta).read_text())
    up = np.array(meta["world_up"])
    e1 = np.array(meta["floor_basis"]["e1"])
    e2 = np.array(meta["floor_basis"]["e2"])
    floor_h = meta["inferred_planes"]["floor_height"]
    ceil_h = meta["inferred_planes"]["ceiling_height"]
    rb = meta["reconstructed_bounds"]
    cu, cv_ = (rb["min"][0] + rb["max"][0]) / 2, (rb["min"][1] + rb["max"][1]) / 2
    half_u = (rb["max"][0] - rb["min"][0]) / 2 * args.pad
    half_v = (rb["max"][1] - rb["min"][1]) / 2 * args.pad
    bounds = {"min": [cu - half_u, cv_ - half_v], "max": [cu + half_u, cv_ + half_v]}
    band = (ceil_h - floor_h) * 0.06

    p = PlyData.read(args.init_ply)["vertex"]
    fp = np.stack([np.asarray(p["x"]), np.asarray(p["y"]), np.asarray(p["z"])], axis=1)
    col = np.stack([np.asarray(p["red"]), np.asarray(p["green"]), np.asarray(p["blue"])], axis=1) / 255.0

    # recompute the SAME hole mask as the original run (geometry-only, no diffusion)
    _, hole, _ = project_topdown(fp, col, e1, e2, up, floor_h, band, bounds, args.res)
    completed = np.asarray(Image.open(args.floor_texture).convert("RGB"))
    assert completed.shape[:2] == hole.shape, \
        f"cached floor_texture.png {completed.shape[:2]} doesn't match recomputed hole grid {hole.shape} — rerun inpaint_planes.py instead"

    ck = torch.load(args.ckpt, map_location="cpu")
    sh_degree = ck["sh_degree"]
    f_means, f_sh0, f_shN, f_op, f_sc, f_q, px = make_floor_gaussians(
        completed, hole, bounds, e1, e2, up, floor_h, sh_degree)
    print(f"reused {len(f_means):,} synthetic floor Gaussians from cached floor_texture.png (pixel size {px*100:.1f}cm)")

    op = torch.sigmoid(ck["opacities"]).numpy()
    sc = torch.exp(ck["scales"]).numpy()
    aniso = sc.max(1) / np.maximum(sc.min(1), 1e-6)
    keep = (op >= args.min_opacity) & (sc.max(1) <= args.max_scale) & (aniso <= args.max_aniso)
    idx = np.nonzero(keep)[0]
    if len(idx) > args.max_points:
        idx = np.random.default_rng(0).choice(idx, args.max_points, replace=False)
    r_means = ck["means"][idx].numpy()
    r_sh0 = ck["sh0"][idx].numpy(); r_shN = ck["shN"][idx].numpy()
    r_op = ck["opacities"][idx].numpy(); r_sc = ck["scales"][idx].numpy(); r_q = ck["quats"][idx].numpy()
    print(f"{len(op):,} -> {len(idx):,} real Gaussians kept after opacity/scale/aniso pruning")

    views = load_views(args.dataset)
    refined_c2w = refine_c2w(views, ck["pose_rot"].numpy(), ck["pose_trans"].numpy()) if "pose_rot" in ck else None
    conf_r, sview_r, nv_r = compute_provenance(torch.tensor(r_means), views, 20.0, 0.1, args.device, refined_c2w)

    if args.floater_dist > 0:
        keep2 = prune_floaters(r_means, nv_r, args.floater_dist)
        n1 = len(r_means)
        r_means, r_sh0, r_shN = r_means[keep2], r_sh0[keep2], r_shN[keep2]
        r_op, r_sc, r_q = r_op[keep2], r_sc[keep2], r_q[keep2]
        conf_r, sview_r, nv_r = conf_r[keep2], sview_r[keep2], nv_r[keep2]
        print(f"floater prune: {n1:,} -> {len(r_means):,} "
              f"({n1 - len(r_means):,} isolated unconfirmed points removed, "
              f">{args.floater_dist*100:.0f}cm from the nearest confirmed surface)")

    means = np.concatenate([r_means, f_means])
    sh0 = np.concatenate([r_sh0, f_sh0]); shN = np.concatenate([r_shN, f_shN])
    opac = np.concatenate([r_op, f_op]); scales = np.concatenate([r_sc, f_sc]); quats = np.concatenate([r_q, f_q])
    write_ply(args.out_scene, means, sh0, shN, opac, scales, quats)
    print(f"wrote {args.out_scene} | {len(means):,} total Gaussians "
          f"({len(r_means):,} real + {len(f_means):,} synthetic floor fill)")

    conf = np.concatenate([conf_r, np.zeros(len(f_means), dtype=np.float32)])
    sview = np.concatenate([sview_r, np.full(len(f_means), -1, dtype=np.int32)])
    nv = np.concatenate([nv_r, np.zeros(len(f_means), dtype=np.int32)])
    write_trust_bin(args.out_trust, means, sh0, conf, sview, nv, 400_000)
    print(f"wrote {args.out_trust}")


if __name__ == "__main__":
    main()
