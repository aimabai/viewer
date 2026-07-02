"""Stage 1 of the Flux-based enhancement pipeline: render + cache what the Flux
stage needs (RGB render, alpha, DA3 confidence, GT), as plain files.

Split into two stages because the two halves need incompatible torch builds:
gsplat (this venv) is compiled against torch 2.1.2+cu118, while modern diffusers'
GGUF loading (flux_enhance.py, run from .venv-flux) needs torch 2.4+. They can't
share a Python environment, so we hand off through the filesystem instead of a
shared process. See RESEARCH.md's ArtiFixer section.

Run:  python render_for_enhance.py --ckpt ../outputs/full_v4_confreg.pt --views 0,120,132
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v3 as iio

from data import load_views
from gsplat import rasterization
from export import axis_angle_to_R_np


@torch.no_grad()
def render_view(ck, sh, v, view_idx, device, max_depth=20.0):
    H, W = v.hw
    K = torch.tensor(v.K, device=device)[None]
    viewmat = v.viewmat
    if "pose_rot" in ck:
        delta = np.eye(4, dtype=np.float64)
        delta[:3, :3] = axis_angle_to_R_np(ck["pose_rot"][view_idx].numpy())
        delta[:3, 3] = ck["pose_trans"][view_idx].numpy()
        viewmat = (delta @ viewmat.astype(np.float64)).astype(np.float32)
    vm = torch.tensor(viewmat, device=device)[None]
    out, alpha, _ = rasterization(
        ck["means"].to(device), F.normalize(ck["quats"].to(device), dim=-1),
        torch.exp(ck["scales"].to(device)), torch.sigmoid(ck["opacities"].to(device)),
        sh, vm, K, W, H, sh_degree=ck["sh_degree"], render_mode="RGB",
        far_plane=max_depth, rasterize_mode="antialiased")
    rgb = out[0].clamp(0, 1).cpu().numpy()
    a = alpha[0, ..., 0].cpu().numpy()
    return rgb, a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../outputs/full_v4_confreg.pt")
    ap.add_argument("--dataset", default="../dataset")
    ap.add_argument("--views", default="0,120,132")
    ap.add_argument("--out-dir", default="../outputs/enhance_cache")
    args = ap.parse_args()
    device = "cuda"

    ck = torch.load(args.ckpt, map_location="cpu")
    sh = torch.cat([ck["sh0"], ck["shN"]], dim=1).to(device)
    views = load_views(args.dataset)
    ids = [int(x) for x in args.views.split(",")]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in ids:
        v = views[i]
        rgb, alpha = render_view(ck, sh, v, i, device)
        gt = np.asarray(iio.imread(args.dataset + "/" + v.pano)).astype(np.float32) / 255.0
        H, W = rgb.shape[:2]
        if gt.shape[:2] != (H, W):
            gt = np.asarray(iio.imread(args.dataset + "/" + v.pano))
        iio.imwrite(out_dir / f"view{i}_render.png", (rgb * 255).astype(np.uint8))
        iio.imwrite(out_dir / f"view{i}_gt.png", (gt * 255).astype(np.uint8) if gt.max() <= 1.0 else gt.astype(np.uint8))
        np.save(out_dir / f"view{i}_alpha.npy", alpha.astype(np.float32))
        np.save(out_dir / f"view{i}_conf.npy", v.conf.astype(np.float32))
        print(f"cached view {i}: render {rgb.shape}, conf mean {v.conf.mean():.3f}")

    print(f"wrote render cache to {out_dir} — now run flux_enhance.py from .venv-flux")


if __name__ == "__main__":
    main()
