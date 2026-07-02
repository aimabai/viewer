"""Depth-supervised 3D Gaussian Splatting on gsplat.

Recipe (see RESEARCH.md for paper grounding):
  L = (0.8*L1 + 0.2*D-SSIM)          photometric, 3DGS (Kerbl SIGGRAPH'23)
    + lambda_d * (1 - Pearson(D_render, D_DA3))   scale-invariant depth,
                                       DN-Splatter (WACV'25) / DNGaussian (CVPR'24)
Every per-pixel term is multiplied by the inverse person-mask, and the depth
residual is additionally weighted by DA3's per-pixel `conf` (our extension —
most depth-supervised papers have no confidence signal; this dataset does).

Densification/pruning is handled by gsplat's DefaultStrategy. Gaussians are
initialized from the depth-lifted cloud (prepared/init_pointcloud.ply).
Per-Gaussian provenance (confidence / supervising view) is NOT tracked here — it
is derived post-hoc in export.py, because densification clones/splits Gaussians.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from plyfile import PlyData
from tqdm import tqdm

from data import load_views
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy

C0 = 0.28209479177387814  # SH band-0 constant


def rgb_to_sh(rgb: torch.Tensor) -> torch.Tensor:
    return (rgb - 0.5) / C0


def gaussian_window(size: int, sigma: float, device) -> torch.Tensor:
    coords = torch.arange(size, device=device) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum())
    return (g[:, None] * g[None, :])[None, None]  # (1,1,size,size)


def ssim(a: torch.Tensor, b: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    """SSIM on (1,3,H,W) images in [0,1]; mean over the map."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    win = win.expand(a.shape[1], 1, *win.shape[-2:])
    p = win.shape[-1] // 2
    mu_a = F.conv2d(a, win, padding=p, groups=a.shape[1])
    mu_b = F.conv2d(b, win, padding=p, groups=a.shape[1])
    mu_a2, mu_b2, mu_ab = mu_a * mu_a, mu_b * mu_b, mu_a * mu_b
    s_a = F.conv2d(a * a, win, padding=p, groups=a.shape[1]) - mu_a2
    s_b = F.conv2d(b * b, win, padding=p, groups=a.shape[1]) - mu_b2
    s_ab = F.conv2d(a * b, win, padding=p, groups=a.shape[1]) - mu_ab
    num = (2 * mu_ab + C1) * (2 * s_ab + C2)
    den = (mu_a2 + mu_b2 + C1) * (s_a + s_b + C2)
    return (num / den).mean()


def image_gradient_weight(img: torch.Tensor, scale: float) -> torch.Tensor:
    """exp(-scale * |grad(I)|), DN-Splatter's edge-downweighting term (RESEARCH.md):
    depth supervision is least reliable at RGB edges/occlusion boundaries (that's
    exactly where monocular depth over-smooths), so trust it less there and let the
    photometric loss — which sees the sharp edge directly — drive the geometry
    instead. `img` is (H,W,3) in [0,1]; returns (H,W) in (0,1]."""
    gray = img.mean(dim=-1)
    gx = torch.zeros_like(gray); gy = torch.zeros_like(gray)
    gx[:, :-1] = gray[:, 1:] - gray[:, :-1]
    gy[:-1, :] = gray[1:, :] - gray[:-1, :]
    grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    return torch.exp(-scale * grad_mag)


def pearson_depth_loss(d_render, d_gt, weight):
    """1 - weighted Pearson correlation between rendered and DA3 depth.

    Scale-and-shift invariant, so DA3's monocular scale error doesn't bias geometry.
    `weight` (>=0) is inverse-mask * conf; pixels with weight 0 are ignored.
    """
    w = weight.flatten()
    if w.sum() < 16:
        return d_render.new_zeros(())
    x = d_render.flatten()
    y = d_gt.flatten()
    wsum = w.sum()
    mx = (w * x).sum() / wsum
    my = (w * y).sum() / wsum
    xc, yc = x - mx, y - my
    cov = (w * xc * yc).sum()
    vx = (w * xc * xc).sum()
    vy = (w * yc * yc).sum()
    corr = cov / (torch.sqrt(vx * vy) + 1e-8)
    return 1.0 - corr


def axis_angle_to_R(aa: torch.Tensor) -> torch.Tensor:
    """Rodrigues' formula: small rotation vector (3,) -> rotation matrix (3,3)."""
    theta = torch.linalg.norm(aa) + 1e-8
    axis = aa / theta
    zero = torch.zeros((), device=aa.device, dtype=aa.dtype)
    K = torch.stack([
        torch.stack([zero, -axis[2], axis[1]]),
        torch.stack([axis[2], zero, -axis[0]]),
        torch.stack([-axis[1], axis[0], zero]),
    ])
    eye = torch.eye(3, device=aa.device, dtype=aa.dtype)
    return eye + torch.sin(theta) * K + (1 - torch.cos(theta)) * (K @ K)


def knn_scale_init(points: np.ndarray) -> np.ndarray:
    """Per-point isotropic log-scale = log(mean dist to 3 nearest neighbours)."""
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    d = np.asarray(pcd.compute_nearest_neighbor_distance())
    d = np.clip(d, 1e-4, None)
    return np.log(d)[:, None].repeat(3, axis=1).astype(np.float32)


def load_init_ply(path: str, max_points: int):
    p = PlyData.read(path)["vertex"]
    pts = np.stack([p["x"], p["y"], p["z"]], axis=1).astype(np.float32)
    col = np.stack([p["red"], p["green"], p["blue"]], axis=1).astype(np.float32) / 255.0
    if max_points and len(pts) > max_points:
        idx = np.random.default_rng(0).choice(len(pts), max_points, replace=False)
        pts, col = pts[idx], col[idx]
    return pts, col


def build_model(pts, col, sh_degree, device):
    n = len(pts)
    n_sh = (sh_degree + 1) ** 2
    means = torch.tensor(pts, device=device)
    scales = torch.tensor(knn_scale_init(pts), device=device)
    quats = torch.zeros(n, 4, device=device); quats[:, 0] = 1.0
    opac = torch.logit(torch.full((n,), 0.1, device=device))
    sh0 = rgb_to_sh(torch.tensor(col, device=device))[:, None, :]   # (n,1,3)
    shN = torch.zeros(n, n_sh - 1, 3, device=device)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(means),
        "scales": torch.nn.Parameter(scales),
        "quats": torch.nn.Parameter(quats),
        "opacities": torch.nn.Parameter(opac),
        "sh0": torch.nn.Parameter(sh0),
        "shN": torch.nn.Parameter(shN),
    }).to(device)
    return params


def make_optimizers(params, scene_scale):
    lrs = {"means": 1.6e-4 * scene_scale, "scales": 5e-3, "quats": 1e-3,
           "opacities": 5e-2, "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
    return {k: torch.optim.Adam([{"params": [params[k]], "lr": lr, "name": k}],
                                eps=1e-15) for k, lr in lrs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset")
    ap.add_argument("--init-ply", default="prepared/init_pointcloud.ply")
    ap.add_argument("--out", default="outputs/ckpt.pt")
    ap.add_argument("--iters", type=int, default=30000)
    ap.add_argument("--sh-degree", type=int, default=3)
    ap.add_argument("--res-scale", type=float, default=1.0, help="downscale images for speed")
    ap.add_argument("--max-init-points", type=int, default=1_000_000)
    ap.add_argument("--depth-lambda", type=float, default=0.2)
    ap.add_argument("--ssim-lambda", type=float, default=0.2)
    ap.add_argument("--aniso-lambda", type=float, default=0.02, help="anti-needle: penalize anisotropy")
    ap.add_argument("--size-lambda", type=float, default=0.05, help="penalize oversized Gaussians")
    ap.add_argument("--max-aniso", type=float, default=6.0, help="anisotropy ratio penalized above this")
    ap.add_argument("--max-scale", type=float, default=0.25, help="metre scale penalized above this")
    ap.add_argument("--max-depth", type=float, default=20.0)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--refine-poses", action="store_true",
                    help="learn a small per-camera SE3 correction on top of the given "
                         "DA3 poses — adjacent yaws overlap heavily, so any small pose "
                         "error shows up as blur/double-vision there; this lets the "
                         "optimizer correct it")
    ap.add_argument("--pose-rot-lr", type=float, default=2e-4, help="radians/iter scale")
    ap.add_argument("--pose-trans-lr", type=float, default=2e-4, help="x scene_scale, metres/iter scale")
    ap.add_argument("--pose-warmup", type=int, default=2000,
                    help="iters of fixed (unrefined) poses before pose optimization starts — "
                         "letting Gaussian geometry settle first avoids a coupled-drift "
                         "instability (same failure mode as 2DGS's distortion loss)")
    ap.add_argument("--pose-reg-lambda", type=float, default=1.0,
                    help="L2 penalty keeping pose deltas small (rotation in rad, translation in scene_scale units)")
    ap.add_argument("--conf-gamma", type=float, default=1.0,
                    help="sharpen per-pixel confidence weighting in the depth loss: "
                         "weight = conf**gamma (>1 pushes low-confidence pixels toward "
                         "zero weight harder, without hard-dropping whole frames)")
    ap.add_argument("--grad-edge-scale", type=float, default=8.0,
                    help="DN-Splatter edge-downweighting: depth-loss weight *= "
                         "exp(-scale * |grad(RGB)|), so sharp RGB edges (where "
                         "monocular depth over-smooths) get less depth supervision "
                         "and more freedom to be fit by the photometric loss instead. "
                         "0 disables (uniform weight 1, recovers prior behaviour)")
    ap.add_argument("--conf-reg-gamma", type=float, default=2.0,
                    help="sharpen the confidence-adaptive regularization weight: "
                         "distrust = (1-conf)**gamma sampled at each visible Gaussian's "
                         "projected pixel")
    ap.add_argument("--reg-conf-boost", type=float, default=4.0,
                    help="extra anti-needle/size regularization strength in low-confidence "
                         "regions, on top of the flat --aniso-lambda/--size-lambda base "
                         "(0 disables confidence-adaptivity, recovering the old uniform "
                         "per-Gaussian penalty)")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = "cuda"

    views = load_views(args.dataset)
    pts, col = load_init_ply(args.init_ply, args.max_init_points)
    scene_scale = float(np.linalg.norm(pts - pts.mean(0), axis=1).max())
    print(f"{len(views)} views | {len(pts):,} init Gaussians | scene_scale {scene_scale:.2f}m")

    params = build_model(pts, col, args.sh_degree, device)
    optimizers = make_optimizers(params, scene_scale)
    # Exponential decay of the POSITION learning rate to ~1% over training — the
    # standard 3DGS schedule. Without it, Gaussian positions never settle and the
    # scene melts over long runs (observed: 30k iters degraded vs the 400-iter smoke).
    means_sched = torch.optim.lr_scheduler.ExponentialLR(
        optimizers["means"], gamma=0.01 ** (1.0 / max(args.iters, 1)))
    strategy = DefaultStrategy(verbose=False)
    state = strategy.initialize_state(scene_scale=scene_scale)
    win = gaussian_window(11, 1.5, device)

    # Optional per-camera pose refinement, kept OUT of gsplat's `optimizers` dict (and
    # its own optimizer) so it never touches DefaultStrategy's densification bookkeeping,
    # which expects exactly the six Gaussian-parameter keys.
    pose_rot = pose_trans = pose_opt = None
    if args.refine_poses:
        pose_rot = torch.zeros(len(views), 3, device=device, requires_grad=True)
        pose_trans = torch.zeros(len(views), 3, device=device, requires_grad=True)
        pose_opt = torch.optim.Adam([
            {"params": [pose_rot], "lr": args.pose_rot_lr},
            {"params": [pose_trans], "lr": args.pose_trans_lr * scene_scale},
        ])

    # Pre-stage per-view tensors on CPU (moved to GPU per step to save VRAM).
    def view_tensors(v):
        s = args.res_scale
        H, W = v.hw
        img = torch.tensor(v.image / 255.0, dtype=torch.float32)
        depth = torch.tensor(v.depth, dtype=torch.float32)
        conf = torch.tensor(v.conf, dtype=torch.float32)
        m = torch.tensor(v.train_mask(), dtype=torch.float32)
        if s != 1.0:
            nH, nW = int(H * s), int(W * s)
            img = F.interpolate(img.permute(2, 0, 1)[None], (nH, nW), mode="bilinear",
                                align_corners=False)[0].permute(1, 2, 0)
            depth = F.interpolate(depth[None, None], (nH, nW))[0, 0]
            conf = F.interpolate(conf[None, None], (nH, nW))[0, 0]
            m = F.interpolate(m[None, None], (nH, nW))[0, 0]
            K = v.K.copy(); K[:2] *= s
        else:
            K = v.K
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
        m = m.to(device); K = K.to(device)[None]; viewmat = viewmat.to(device)

        refining = args.refine_poses and step >= args.pose_warmup
        if refining:
            R_delta = axis_angle_to_R(pose_rot[vi])
            delta = torch.eye(4, device=device, dtype=viewmat.dtype)
            delta = delta.clone()
            delta[:3, :3] = R_delta
            delta[:3, 3] = pose_trans[vi]
            viewmat = delta @ viewmat
        viewmat = viewmat[None]

        sh = torch.cat([params["sh0"], params["shN"]], dim=1)
        cur_sh = min(args.sh_degree, step // 1000)
        render, alpha, info = rasterization(
            params["means"], F.normalize(params["quats"], dim=-1),
            torch.exp(params["scales"]), torch.sigmoid(params["opacities"]),
            sh, viewmat, K, W, H, sh_degree=cur_sh, packed=True,
            render_mode="RGB+ED", near_plane=0.01, far_plane=args.max_depth,
            rasterize_mode="antialiased",
        )
        rgb = render[..., :3].clamp(0, 1)            # (1,H,W,3)
        d_r = render[..., 3]                          # (1,H,W)
        strategy.step_pre_backward(params, optimizers, state, step, info)

        mask = m[None, ..., None]                     # (1,H,W,1)
        gt = img[None]                                # (1,H,W,3)
        l1 = (torch.abs(rgb - gt) * mask).sum() / (mask.sum() * 3 + 1e-8)
        a = (rgb * mask).permute(0, 3, 1, 2)
        b = (gt * mask).permute(0, 3, 1, 2)
        dssim = 1.0 - ssim(a, b, win)
        photo = (1 - args.ssim_lambda) * l1 + args.ssim_lambda * dssim

        conf_w = conf.pow(args.conf_gamma) if args.conf_gamma != 1.0 else conf
        edge_w = image_gradient_weight(img, args.grad_edge_scale) if args.grad_edge_scale > 0 else 1.0
        dweight = m * conf_w * edge_w * ((depth > 0) & (depth < args.max_depth)).float()
        depth_loss = pearson_depth_loss(d_r[0], depth, dweight)

        # Anti-needle regularization: penalize elongated (anisotropic) and oversized
        # Gaussians, the source of the spike artifacts in this sparse-view scene.
        # Confidence-adaptive (extension beyond the flat penalty): only the Gaussians
        # actually visible in THIS view are regularized (packed rasterization gives
        # their global indices + projected pixel for free), and the penalty is boosted
        # where DA3 was unsure. Previously every Gaussian was pushed toward the same
        # cap regardless of how much we trusted the depth that placed it there — this
        # lets confident regions stay sharp/anisotropic while uncertain regions get
        # pushed harder toward small, round (conservative) geometry.
        sc = torch.exp(params["scales"])
        gids = info["gaussian_ids"]
        xy = info["means2d"]                                        # (nnz,2) pixel coords
        gx = (xy[:, 0] / max(W - 1, 1)) * 2 - 1
        gy = (xy[:, 1] / max(H - 1, 1)) * 2 - 1
        grid = torch.stack([gx, gy], dim=-1)[None, :, None, :]      # (1,nnz,1,2)
        conf_vis = F.grid_sample(conf[None, None], grid, mode="bilinear",
                                  align_corners=True, padding_mode="border")[0, 0, :, 0]
        distrust = (1.0 - conf_vis).clamp(0, 1).pow(args.conf_reg_gamma)
        reg_weight = 1.0 + args.reg_conf_boost * distrust             # >=1, higher where unsure

        sc_vis = sc[gids]
        smax_vis = sc_vis.max(dim=1).values
        smin_vis = sc_vis.min(dim=1).values
        reg_aniso = (reg_weight * torch.relu(smax_vis / (smin_vis + 1e-6) - args.max_aniso)).mean()
        reg_size = (reg_weight * torch.relu(smax_vis - args.max_scale)).mean()
        loss = photo + args.depth_lambda * depth_loss + \
            args.aniso_lambda * reg_aniso + args.size_lambda * reg_size
        if refining:
            # keep the correction "small" (that's the whole premise): L2 penalty on this
            # view's current delta, not summed over all 240 (would swamp the photo loss)
            pose_reg = pose_rot[vi].pow(2).sum() + (pose_trans[vi] / scene_scale).pow(2).sum()
            loss = loss + args.pose_reg_lambda * pose_reg

        loss.backward()
        if step % args.log_every == 0:
            pose_msg = (f" prot {pose_rot.norm(dim=1).mean().item():.4f}rad "
                       f"ptrans {pose_trans.norm(dim=1).mean().item()*100:.2f}cm") if args.refine_poses else ""
            pbar.set_description(
                f"L{loss.item():.4f} l1 {l1.item():.4f} dssim {dssim.item():.3f} "
                f"dep {depth_loss.item():.3f} N {params['means'].shape[0]//1000}k{pose_msg}")
        for opt in optimizers.values():
            opt.step(); opt.zero_grad(set_to_none=True)
        if refining:
            pose_opt.step(); pose_opt.zero_grad(set_to_none=True)
        means_sched.step()
        strategy.step_post_backward(params, optimizers, state, step, info, packed=True)

    save(params, args.out, args.sh_degree, scene_scale, pose_rot, pose_trans)


def save(params, out, sh_degree, scene_scale, pose_rot=None, pose_trans=None):
    out = Path(out); out.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {k: params[k].detach().cpu() for k in params}
    ckpt["sh_degree"] = sh_degree
    ckpt["scene_scale"] = scene_scale
    if pose_rot is not None:
        ckpt["pose_rot"] = pose_rot.detach().cpu()
        ckpt["pose_trans"] = pose_trans.detach().cpu()
    torch.save(ckpt, out)
    print(f"saved {params['means'].shape[0]:,} Gaussians -> {out}")


if __name__ == "__main__":
    main()
