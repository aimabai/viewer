"""Make a lightweight scene.ply for fast browser loading.

The full export (SH degree 3, ~480 MB) is slow to stream/parse in a browser. This
drops the spherical-harmonic rest terms (keeping only the DC colour — view-independent,
fine for a mostly-diffuse indoor scene) and optionally decimates, cutting the file to
~60 MB so it loads in seconds. Geometry is unchanged; only view-dependent specular and
some density are sacrificed. Use the full export.py PLY when fidelity matters.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from export import write_ply


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../outputs/full.pt")
    ap.add_argument("--out", default="../viewer/public/scene/scene.ply")
    ap.add_argument("--min-opacity", type=float, default=0.05)
    ap.add_argument("--max-scale", type=float, default=0.5)
    ap.add_argument("--max-aniso", type=float, default=12.0, help="drop needle splats (max/min scale ratio)")
    ap.add_argument("--max-points", type=int, default=1_200_000)
    ap.add_argument("--keep-sh", action="store_true", help="keep full SH (deg 3); else DC-only (deg 0)")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    op = torch.sigmoid(ck["opacities"]).numpy()
    sc = torch.exp(ck["scales"]).numpy()
    maxsc = sc.max(1)
    aniso = maxsc / np.maximum(sc.min(1), 1e-6)
    keep = np.nonzero((op >= args.min_opacity) & (maxsc <= args.max_scale) &
                      (aniso <= args.max_aniso))[0]   # drop near-transparent, huge, and needle splats
    n0 = len(op)
    if len(keep) > args.max_points:
        keep = np.random.default_rng(0).choice(keep, args.max_points, replace=False)

    n = len(keep)
    shN = ck["shN"][keep].numpy() if args.keep_sh else np.zeros((n, 0, 3), dtype=np.float32)
    write_ply(args.out, ck["means"][keep].numpy(), ck["sh0"][keep].numpy(), shN,
              ck["opacities"][keep].numpy(), ck["scales"][keep].numpy(), ck["quats"][keep].numpy())
    mb = Path(args.out).stat().st_size / 1e6
    deg = 3 if args.keep_sh else 0
    print(f"{n0:,} -> {n:,} splats kept (SH deg {deg}, aniso<={args.max_aniso}) | {mb:.0f} MB")


if __name__ == "__main__":
    main()
