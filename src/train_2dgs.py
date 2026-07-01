"""2D Gaussian Splatting variant (surfels) — for cleaner surface geometry than 3DGS.

Reuses the 3DGS data path and losses (train.py), but rasterizes with gsplat's
`rasterization_2dgs`, which represents each Gaussian as a flat oriented disk. This
adds 2DGS's two signature regularizers that make indoor surfaces clean:
  - normal consistency: rendered normals should match the depth-derived surface normals
  - distortion loss: concentrate each ray's weight on a single surface
Keeps the same masked photometric + Pearson depth loss + position-LR decay.
Reference: Huang et al., "2D Gaussian Splatting for Geometrically Accurate Radiance
Fields", SIGGRAPH 2024.
"""
from __future__ import annotations

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data import load_views
from gsplat import rasterization_2dgs
from gsplat.strategy import DefaultStrategy
from train import (ssim, gaussian_window, pearson_depth_loss, build_model,
                   make_optimizers, load_init_ply, save)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--init-ply", default="prepared/init_pointcloud.ply")
    ap.add_argument("--out", default="outputs/ckpt_2dgs.pt")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--res-scale", type=float, default=1.0)
    ap.add_argument("--max-init-points", type=int, default=1_000_000)
    ap.add_argument("--depth-lambda", type=float, default=0.2)
    ap.add_argument("--ssim-lambda", type=float, default=0.2)
    ap.add_argument("--normal-lambda", type=float, default=0.02)
    ap.add_argument("--dist-lambda", type=float, default=1.0)
    ap.add_argument("--dist-start", type=int, default=7000, help="iter to start distortion loss")
    ap.add_argument("--max-depth", type=float, default=20.0)
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda"

    views = load_views(args.dataset)
    pts, col = load_init_ply(args.init_ply, args.max_init_points)
    scene_scale = float(np.linalg.norm(pts - pts.mean(0), axis=1).max())
    print(f"2DGS | {len(views)} views | {len(pts):,} init | scene_scale {scene_scale:.2f}m")

    params = build_model(pts, col, args.sh_degree, device)
    optimizers = make_optimizers(params, scene_scale)
    means_sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / max(args.iters, 1)))
    strategy = DefaultStrategy(verbose=False, key_for_gradient="gradient_2dgs")
    state = strategy.initialize_state(scene_scale=scene_scale)
    win = gaussian_window(11, 1.5, device)

    def view_tensors(v):
        s = args.res_scale
        H, W = v.hw
        img = torch.tensor(v.image / 255.0, dtype=torch.float32)
        depth = torch.tensor(v.depth, dtype=torch.float32)
        conf = torch.tensor(v.conf, dtype=torch.float32)
        m = torch.tensor(v.train_mask(), dtype=torch.float32)
        K = v.K
        if s != 1.0:
            nH, nW = int(H * s), int(W * s)
            img = F.interpolate(img.permute(2, 0, 1)[None], (nH, nW), mode="bilinear", align_corners=False)[0].permute(1, 2, 0)
            depth = F.interpolate(depth[None, None], (nH, nW))[0, 0]
            conf = F.interpolate(conf[None, None], (nH, nW))[0, 0]
            m = F.interpolate(m[None, None], (nH, nW))[0, 0]
            K = v.K.copy(); K[:2] *= s
        return img, depth, conf, m, torch.tensor(K), torch.tensor(v.viewmat), img.shape[:2]

    cache = [view_tensors(v) for v in views]
    order = np.random.permutation(len(views)).tolist()
    pbar = tqdm(range(args.iters))
    for step in pbar:
        if not order:
            order = np.random.permutation(len(views)).tolist()
        vi = order.pop()
        img, depth, conf, m, K, viewmat, (H, W) = cache[vi]
        img = img.to(device); depth = depth.to(device); conf = conf.to(device)
        m = m.to(device); K = K.to(device)[None]; viewmat = viewmat.to(device)[None]

        sh = torch.cat([params["sh0"], params["shN"]], dim=1)
        cur_sh = min(args.sh_degree, step // 1000)
        renders, alphas, normals, surf_normals, distort, median, info = rasterization_2dgs(
            params["means"], F.normalize(params["quats"], dim=-1),
            torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
            sh, viewmat, K, W, H, sh_degree=cur_sh, packed=False,
            render_mode="RGB+ED", near_plane=0.01, far_plane=args.max_depth,
        )
        rgb = renders[..., :3].clamp(0, 1)          # (1,H,W,3)
        d_r = renders[..., 3]                        # (1,H,W)
        strategy.step_pre_backward(params, optimizers, state, step, info)

        mask = m[None, ..., None]
        gt = img[None]
        l1 = (torch.abs(rgb - gt) * mask).sum() / (mask.sum() * 3 + 1e-8)
        a = (rgb * mask).permute(0, 3, 1, 2)
        b = (gt * mask).permute(0, 3, 1, 2)
        dssim = 1.0 - ssim(a, b, win)
        photo = (1 - args.ssim_lambda) * l1 + args.ssim_lambda * dssim

        dweight = m * conf * ((depth > 0) & (depth < args.max_depth)).float()
        depth_loss = pearson_depth_loss(d_r[0], depth, dweight)

        # 2DGS surface regularizers (ramp distortion in after warm-up)
        normal_loss = (1 - (normals * surf_normals).sum(-1)).mean()
        dist_loss = distort.mean()
        dist_w = args.dist_lambda if step > args.dist_start else 0.0
        loss = photo + args.depth_lambda * depth_loss + \
            args.normal_lambda * normal_loss + dist_w * dist_loss

        loss.backward()
        if step % args.log_every == 0:
            pbar.set_description(
                f"L{loss.item():.4f} l1 {l1.item():.4f} dssim {dssim.item():.3f} "
                f"dep {depth_loss.item():.3f} nrm {normal_loss.item():.3f} N {params['means'].shape[0]//1000}k")
        for opt in optimizers.values():
            opt.step(); opt.zero_grad(set_to_none=True)
        means_sched.step()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=False)

    save(params, args.out, args.sh_degree, scene_scale)


if __name__ == "__main__":
    main()
