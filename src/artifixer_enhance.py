"""ArtiFixer-inspired confidence-gated enhancement of rendered views.

The full ArtiFixer (NVIDIA, SIGGRAPH'26) uses a 14B video diffusion model to enhance
3D reconstructions, blending the rendering with generation via a rendered opacity map
("opacity mixing"). We can't run a 14B video model on one 3090, so this implements the
*idea* at feasible scale:

  - render a trained view from the Gaussian model,
  - enhance it with a small image diffusion model (SD-Turbo img2img),
  - BLEND render↔enhanced per-pixel using DA3 confidence as the gate — keep the render
    where the depth prior was trusted, let the model regenerate where it was not
    (windows, reflections, low-confidence regions).

This is the trust-aware analogue of ArtiFixer's opacity-mixing: OUR confidence channel
is the control signal that decides where to generate. Offline (enhances rendered
frames); it does not re-fit the 3DGS. Run:
  python artifixer_enhance.py --ckpt ../outputs/full_reg.pt --views 0,120,132
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
import imageio.v3 as iio

from data import load_views
from gsplat import rasterization


@torch.no_grad()
def render_view(ck, sh, v, device, max_depth=20.0):
    H, W = v.hw
    K = torch.tensor(v.K, device=device)[None]
    vm = torch.tensor(v.viewmat, device=device)[None]
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
    ap.add_argument("--ckpt", default="../outputs/full_reg.pt")
    ap.add_argument("--dataset", default="../dataset")
    ap.add_argument("--views", default="0,120,132")
    ap.add_argument("--out", default="../outputs/artifixer.png")
    ap.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--guidance", type=float, default=6.0)
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--conf-quantile", type=float, default=0.5,
                    help="gate midpoint: pixels below this conf quantile get enhanced")
    args = ap.parse_args()
    device = "cuda"

    ck = torch.load(args.ckpt, map_location="cpu")
    sh = torch.cat([ck["sh0"], ck["shN"]], dim=1).to(device)
    views = load_views(args.dataset)
    ids = [int(x) for x in args.views.split(",")]

    from diffusers import AutoPipelineForImage2Image
    print(f"loading {args.model} (downloads several GB on first run)…")
    try:
        pipe = AutoPipelineForImage2Image.from_pretrained(
            args.model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    except Exception:
        pipe = AutoPipelineForImage2Image.from_pretrained(args.model, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pipe.enable_attention_slicing()
    PROMPT = ("a sharp, clean, photorealistic photo of a modern office interior, "
              "large bright windows, clear glass, realistic reflections, high detail")

    rows = []
    for i in ids:
        v = views[i]
        rgb, alpha = render_view(ck, sh, v, device)
        H, W = rgb.shape[:2]

        # confidence gate: low DA3 conf (windows/reflections) -> enhance; high -> keep render
        conf = v.conf.astype(np.float32)
        thr = np.quantile(conf, args.conf_quantile)
        gate = np.clip((thr - conf) / (thr + 1e-6), 0, 1)        # 1 where low-conf
        gate *= (alpha > 0.3)                                    # only where the model rendered something
        gate = np.power(gate, 0.7)

        # diffusion enhancement (img2img)
        from PIL import Image
        img = Image.fromarray((rgb * 255).astype(np.uint8)).resize((args.size, args.size))
        enh = pipe(prompt=PROMPT, image=img, num_inference_steps=args.steps,
                   strength=args.strength, guidance_scale=args.guidance).images[0]
        enh = np.asarray(enh.resize((W, H))).astype(np.float32) / 255.0

        g3 = gate[..., None]
        blended = rgb * (1 - g3) + enh * g3

        gt = np.asarray(iio.imread(args.dataset + "/" + v.pano)).astype(np.float32) / 255.0
        if gt.shape[:2] != (H, W):
            gt = np.asarray(Image.fromarray((gt * 255).astype(np.uint8)).resize((W, H))) / 255.0
        gate_vis = np.repeat(gate[..., None], 3, axis=2)
        row = np.concatenate([gt, rgb, gate_vis, enh, blended], axis=1)
        rows.append(row)
        print(f"view {i}: enhanced (gate mean {gate.mean():.2f})")

    panel = (np.concatenate(rows, axis=0) * 255).astype(np.uint8)
    iio.imwrite(args.out, panel)
    print(f"wrote {args.out}  (rows: views {ids} | cols: GT | render | conf-gate | enhanced | blended)")


if __name__ == "__main__":
    main()
