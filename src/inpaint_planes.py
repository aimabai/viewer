"""Fill the FLOOR's unobserved regions with GENERATIVE (diffusion) content, baked
directly into the Gaussian model — not a separate textured plane.

Single-pitch capture means the floor is never directly observed outside the
coverage bubble (README §5) — zero pixels there, not a training deficiency (see
RESEARCH.md). Earlier iterations of this script tried (a) a flat coloured plane,
then (b) a classical-inpainting-textured plane; both were rejected as looking like
an obviously separate, flat, disconnected surface. This version instead:

  1. Projects the real point cloud's colour onto the floor plane (top-down "map
     view") and masks the holes — exactly as before.
  2. Runs SDXL-inpainting (mask-conditioned) to complete ONLY the hole pixels —
     the observed pixels are passed through untouched, matching the source image.
  3. Converts EVERY HOLE PIXEL (not the whole image) into a new, flat, floor-plane
     -aligned Gaussian — same SH/opacity/scale parameterisation as the trained
     model — and merges them directly into the exported Gaussian set. No plane
     mesh, no separate render path: the fill is literally part of the reconstruction.
  4. Synthetic points are marked confidence=0, n_views=0, supervising_view=-1 in
     the trust data, so the confidence/coverage viewer modes correctly show this
     region as ungrounded even though it looks like solid floor in photographic mode.

Ceiling is intentionally NOT filled (left as whatever the raw model produces there,
no plane, no fill) — judged too irregular for this approach to be defensible, and
the user asked for floor-only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from plyfile import PlyData
from scipy.spatial.transform import Rotation

from data import load_views
from export import compute_provenance, refine_c2w, write_ply, write_trust_bin
from train import rgb_to_sh, C0


def project_topdown(fp, col, e1, e2, up, height, band, bounds, res_u):
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
    mean_col = colp.mean(0) * 255 if len(colp) else np.array([128, 128, 128])
    rgb[hole] = mean_col.astype(np.uint8)   # sane starting image for the diffusion model
    return rgb, hole, res_v


def generative_inpaint(rgb, hole, model, prompt, size=1024, strength=0.99, steps=30, guidance=7.5):
    from PIL import Image
    h, w = rgb.shape[:2]
    scale = size / max(h, w)
    nw, nh = int(round(w * scale / 8) * 8), int(round(h * scale / 8) * 8)
    img = Image.fromarray(rgb).resize((nw, nh), Image.LANCZOS)
    mask = Image.fromarray((hole * 255).astype(np.uint8)).resize((nw, nh), Image.NEAREST)
    out = model(prompt=prompt, image=img, mask_image=mask, num_inference_steps=steps,
               strength=strength, guidance_scale=guidance).images[0]
    return np.asarray(out.resize((w, h), Image.LANCZOS))


def make_floor_gaussians(completed_rgb, hole, bounds, e1, e2, up, floor_h, sh_degree):
    """One flat Gaussian per HOLE pixel (never for observed pixels), aligned to the
    floor plane, coloured from the diffusion-completed image."""
    res_v, res_u = hole.shape
    vi, ui = np.nonzero(hole)
    n = len(vi)
    (umin, vmin), (umax, vmax) = bounds["min"], bounds["max"]
    pixel_w = (umax - umin) / res_u
    pixel_h = (vmax - vmin) / res_v
    px = pixel_size = float((pixel_w + pixel_h) / 2)

    u = umin + (ui + 0.5) / res_u * (umax - umin)
    v = vmin + (vi + 0.5) / res_v * (vmax - vmin)
    means = np.outer(u, e1) + np.outer(v, e2) + floor_h * up[None, :]

    # flat, floor-aligned orientation: local Z (thin axis) -> world up
    basis = np.stack([e1, e2, up], axis=1)          # columns = local X,Y,Z -> world
    quat_xyzw = Rotation.from_matrix(basis).as_quat()  # scipy: [x,y,z,w]
    quat_wxyz = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
    quats = np.tile(quat_wxyz, (n, 1)).astype(np.float32)

    scales = np.tile(np.log([pixel_size * 1.3, pixel_size * 1.3, pixel_size * 0.15]), (n, 1)).astype(np.float32)
    opacities = np.full(n, np.log(0.97 / 0.03), dtype=np.float32)   # logit(0.97), near-solid

    rgb = completed_rgb[vi, ui].astype(np.float32) / 255.0
    sh0 = ((torch.tensor(rgb) - 0.5) / C0).numpy()[:, None, :]      # (n,1,3), same encoding as train.py
    n_sh = (sh_degree + 1) ** 2
    shN = np.zeros((n, n_sh - 1, 3), dtype=np.float32)

    return means.astype(np.float32), sh0.astype(np.float32), shN, opacities, scales, quats, pixel_size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="../outputs/full_v3.pt")
    ap.add_argument("--dataset", default="../dataset")
    ap.add_argument("--init-ply", default="../prepared/init_pointcloud_consistent.ply")
    ap.add_argument("--out-scene", default="../viewer/public/scene/scene.ply")
    ap.add_argument("--out-trust", default="../viewer/public/scene/trust.bin")
    ap.add_argument("--meta", default="../viewer/public/scene/scene_meta.json")
    ap.add_argument("--res", type=int, default=384, help="top-down projection resolution")
    ap.add_argument("--pad", type=float, default=1.15)
    ap.add_argument("--min-opacity", type=float, default=0.02)
    ap.add_argument("--max-scale", type=float, default=0.5)
    ap.add_argument("--max-aniso", type=float, default=12.0)
    ap.add_argument("--max-points", type=int, default=1_200_000)
    ap.add_argument("--model", default="diffusers/stable-diffusion-xl-1.0-inpainting-0.1")
    ap.add_argument("--steps", type=int, default=30)
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

    rgb, hole, res_v = project_topdown(fp, col, e1, e2, up, floor_h, band, bounds, args.res)
    print(f"floor: {hole.mean()*100:.1f}% unobserved ({hole.sum()} pixels to fill generatively)")

    print(f"loading {args.model}…")
    import torch as T
    from diffusers import StableDiffusionXLInpaintPipeline
    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        args.model, torch_dtype=T.float16, variant="fp16", use_safetensors=True).to(args.device)
    pipe.set_progress_bar_config(disable=True)
    prompt = ("polished concrete office floor, seamless continuous surface, "
              "photorealistic, top-down view, consistent lighting, no objects")
    completed = generative_inpaint(rgb, hole, pipe, prompt, steps=args.steps)
    del pipe; T.cuda.empty_cache()

    from PIL import Image
    Image.fromarray(completed).save(Path(args.out_scene).parent / "floor_texture.png")
    Image.fromarray(rgb).save(Path(args.out_scene).parent / "floor_texture_raw.png")

    ck = torch.load(args.ckpt, map_location="cpu")
    sh_degree = ck["sh_degree"]
    f_means, f_sh0, f_shN, f_op, f_sc, f_q, px = make_floor_gaussians(
        completed, hole, bounds, e1, e2, up, floor_h, sh_degree)
    print(f"generated {len(f_means):,} synthetic floor Gaussians (pixel size {px*100:.1f}cm)")

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
    print(f"{len(op):,} -> {len(idx):,} real Gaussians kept after pruning")

    means = np.concatenate([r_means, f_means])
    sh0 = np.concatenate([r_sh0, f_sh0]); shN = np.concatenate([r_shN, f_shN])
    opac = np.concatenate([r_op, f_op]); scales = np.concatenate([r_sc, f_sc]); quats = np.concatenate([r_q, f_q])
    write_ply(args.out_scene, means, sh0, shN, opac, scales, quats)
    print(f"wrote {args.out_scene} | {len(means):,} total Gaussians "
          f"({len(r_means):,} real + {len(f_means):,} synthetic floor fill)")

    # provenance: real points get proper trust computation; synthetic fill is
    # marked unsupervised (confidence=0, n_views=0, view=-1) — honest labelling so
    # the trust-aware modes show this region as generated, not observed.
    views = load_views(args.dataset)
    refined_c2w = refine_c2w(views, ck["pose_rot"].numpy(), ck["pose_trans"].numpy()) if "pose_rot" in ck else None
    conf_r, sview_r, nv_r = compute_provenance(torch.tensor(r_means), views, 20.0, 0.1, args.device, refined_c2w)
    conf = np.concatenate([conf_r, np.zeros(len(f_means), dtype=np.float32)])
    sview = np.concatenate([sview_r, np.full(len(f_means), -1, dtype=np.int32)])
    nv = np.concatenate([nv_r, np.zeros(len(f_means), dtype=np.int32)])
    write_trust_bin(args.out_trust, means, sh0, conf, sview, nv, 400_000)

    meta["inferred_planes"]["floor_fill_gaussians"] = int(len(f_means))
    meta["inferred_planes"]["floor_textured"] = False   # superseded: fill is baked into the model now
    meta["inferred_planes"]["ceiling_textured"] = False
    Path(args.meta).write_text(json.dumps(meta, indent=2))
    print("updated scene_meta.json")


if __name__ == "__main__":
    main()
