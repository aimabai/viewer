"""Render a trained checkpoint against ground truth for visual + metric eval.

Produces a panel image (rows = sampled views, cols = GT | render | depth) and
prints masked PSNR / L1 over the chosen views (people excluded via the mask).
Run:  python render_eval.py --ckpt ../outputs/full.pt --views 0,6,30,120
"""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v3 as iio
import matplotlib.cm as cm

from data import load_views
from gsplat import rasterization


def axis_angle_to_R_np(aa: np.ndarray) -> np.ndarray:
    theta = np.linalg.norm(aa) + 1e-8
    axis = aa / theta
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


@torch.no_grad()
def render_view(ck, sh, v, view_idx, scale, device, max_depth):
    H, W = v.hw
    nH, nW = int(H * scale), int(W * scale)
    K = v.K.copy(); K[:2] *= scale
    viewmat = v.viewmat
    if "pose_rot" in ck:
        # apply this view's trained pose correction (train.py --refine-poses) — without
        # this, eval compares against the WRONG camera and PSNR is meaningless-low even
        # though the trained geometry is fine (confirmed: ~6dB difference on this model)
        delta = np.eye(4, dtype=np.float64)
        delta[:3, :3] = axis_angle_to_R_np(ck["pose_rot"][view_idx].numpy())
        delta[:3, 3] = ck["pose_trans"][view_idx].numpy()
        viewmat = (delta @ viewmat.astype(np.float64)).astype(np.float32)
    vm = torch.tensor(viewmat, device=device)[None]
    Kt = torch.tensor(K, device=device)[None]
    out, _, _ = rasterization(
        ck["means"].to(device), F.normalize(ck["quats"].to(device), dim=-1),
        torch.exp(ck["scales"].to(device)), torch.sigmoid(ck["opacities"].to(device)),
        sh, vm, Kt, nW, nH, sh_degree=ck["sh_degree"],
        render_mode="RGB+ED", far_plane=max_depth)
    rgb = out[0, ..., :3].clamp(0, 1).cpu().numpy()
    d = out[0, ..., 3].cpu().numpy()
    gt = np.asarray(iio.imread("../dataset/" + v.pano)) / 255.0
    gt = F.interpolate(torch.tensor(gt).permute(2, 0, 1)[None].float(), (nH, nW),
                       mode="bilinear", align_corners=False)[0].permute(1, 2, 0).numpy()
    m = F.interpolate(torch.tensor(v.train_mask())[None, None], (nH, nW))[0, 0].numpy()
    dn = (d - d[d > 0].min()) / (d.max() - d[d > 0].min() + 1e-8)
    depth_rgb = cm.turbo(dn)[..., :3]
    mse = (((gt - rgb) ** 2) * m[..., None]).sum() / (m.sum() * 3 + 1e-8)
    psnr = -10 * math.log10(mse + 1e-12)
    l1 = (np.abs(gt - rgb) * m[..., None]).sum() / (m.sum() * 3 + 1e-8)
    panel = np.concatenate([gt, rgb, depth_rgb], axis=1)
    return panel, psnr, l1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../outputs/full.pt")
    ap.add_argument("--dataset", default="../dataset")
    ap.add_argument("--views", default="0,6,30,120", help="comma-separated view ids")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--max-depth", type=float, default=20.0)
    ap.add_argument("--out", default="../outputs/eval_render.png")
    args = ap.parse_args()

    device = "cuda"
    ck = torch.load(args.ckpt, map_location="cpu")
    sh = torch.cat([ck["sh0"], ck["shN"]], dim=1).to(device)
    views = load_views(args.dataset)
    ids = [int(x) for x in args.views.split(",")]

    panels, psnrs, l1s = [], [], []
    for i in ids:
        p, ps, l1 = render_view(ck, sh, views[i], i, args.scale, device, args.max_depth)
        panels.append(p); psnrs.append(ps); l1s.append(l1)
        print(f"view {i:3d}: masked PSNR {ps:5.2f} dB | L1 {l1:.4f}")
    panel = (np.concatenate(panels, axis=0) * 255).astype(np.uint8)
    iio.imwrite(args.out, panel)
    print(f"\nmean masked PSNR {np.mean(psnrs):.2f} dB | mean L1 {np.mean(l1s):.4f}")
    print(f"wrote {args.out}  (rows: views {ids} | cols: GT | render | depth)")


if __name__ == "__main__":
    main()
