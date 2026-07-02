"""Stage 2 of the Flux-based enhancement pipeline (run from .venv-flux, NOT the
main gsplat venv — see render_for_enhance.py for why they're split).

Same trust-aware idea as artifixer_enhance.py (SDXL/SD-Turbo version): render a
trained view, enhance it with a diffusion model, and blend render<->enhanced
per-pixel using DA3 confidence as the gate. This version swaps the enhancer for
quantized Flux.1-schnell (GGUF, city96) so the whole stack — 12B-param transformer
+ T5-XXL text encoder — fits in ~9GB instead of requiring the full bf16 weights
(~34GB) or a gated HF download. Components:
  - transformer : city96/FLUX.1-schnell-gguf, Q4_K_S (GGUF-quantized, ungated)
  - text_encoder_2 (T5-XXL) : city96/t5-v1_1-xxl-encoder-gguf, Q3_K_S (ungated)
  - text_encoder (CLIP-L) : openai/clip-vit-large-patch14 (ungated)
  - vae : community ungated mirror of Flux's ae.safetensors (black-forest-labs's
    own copy is gated; the weights are identical, just re-hosted)

Run:  .venv-flux/bin/python3 flux_enhance.py --cache-dir ../outputs/enhance_cache --views 0,120,132
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("HF_HOME", "/workspace/.hf_home")

import numpy as np
import torch
import imageio.v3 as iio
from PIL import Image


def build_pipeline(transformer_gguf, t5_gguf, vae_path, device="cuda"):
    from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig, AutoencoderKL, FluxImg2ImgPipeline
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from transformers import T5EncoderModel, T5TokenizerFast, CLIPTextModel, CLIPTokenizer

    print("loading Flux transformer (GGUF, Q4_K_S)…")
    # from_single_file still needs the (unquantized) architecture config to build the
    # module graph before loading GGUF weights into it. Its default source is the
    # ORIGINAL gated black-forest-labs repo, which we can't reach without a token — so
    # we point it at a local copy of that same (public, widely-documented) config.json
    # instead. Only architecture metadata (layer counts, dims), no weights, no gating.
    local_config_dir = os.path.join(os.path.dirname(__file__), "..", "configs", "flux_schnell_transformer_config")
    transformer = FluxTransformer2DModel.from_single_file(
        transformer_gguf, config=local_config_dir,
        quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
        torch_dtype=torch.bfloat16)

    print("loading T5-XXL text encoder (GGUF, Q3_K_S)…")
    text_encoder_2 = T5EncoderModel.from_pretrained(
        "city96/t5-v1_1-xxl-encoder-gguf", gguf_file=os.path.basename(t5_gguf), torch_dtype=torch.bfloat16)
    tokenizer_2 = T5TokenizerFast.from_pretrained("google/t5-v1_1-xxl")

    print("loading CLIP-L text encoder…")
    text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", torch_dtype=torch.bfloat16)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

    print("loading VAE…")
    # same problem as the transformer: auto-config defaults to standard SD's 4-channel
    # latent VAE, but Flux's VAE has 16 latent channels — supply the correct (public,
    # documented) config explicitly instead of guessing.
    local_vae_config_dir = os.path.join(os.path.dirname(__file__), "..", "configs", "flux_schnell_vae_config")
    vae = AutoencoderKL.from_single_file(vae_path, config=local_vae_config_dir, torch_dtype=torch.bfloat16)

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=1.0)

    pipe = FluxImg2ImgPipeline(
        transformer=transformer, text_encoder=text_encoder, tokenizer=tokenizer,
        text_encoder_2=text_encoder_2, tokenizer_2=tokenizer_2, vae=vae, scheduler=scheduler)
    pipe.enable_model_cpu_offload()   # 24GB card, several bf16 sub-models — offload keeps peak VRAM sane
    return pipe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="../outputs/enhance_cache")
    ap.add_argument("--views", default="0,120,132")
    ap.add_argument("--out", default="../outputs/artifixer_flux.png")
    ap.add_argument("--transformer-gguf", default=None, help="local path; auto-resolved from HF cache if omitted")
    ap.add_argument("--t5-gguf", default=None)
    ap.add_argument("--vae", default=None)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--strength", type=float, default=0.6)
    ap.add_argument("--steps", type=int, default=4, help="schnell is a few-step distilled model")
    ap.add_argument("--conf-quantile", type=float, default=0.5)
    args = ap.parse_args()

    from huggingface_hub import hf_hub_download
    transformer_gguf = args.transformer_gguf or hf_hub_download(
        "city96/FLUX.1-schnell-gguf", "flux1-schnell-Q4_K_S.gguf", cache_dir="/workspace/.hf_home/hub")
    t5_gguf = args.t5_gguf or hf_hub_download(
        "city96/t5-v1_1-xxl-encoder-gguf", "t5-v1_1-xxl-encoder-Q3_K_S.gguf", cache_dir="/workspace/.hf_home/hub")
    vae_path = args.vae or hf_hub_download(
        "sirorable/flux-ae-vae", "ae.safetensors", cache_dir="/workspace/.hf_home/hub")

    pipe = build_pipeline(transformer_gguf, t5_gguf, vae_path)
    PROMPT = ("a sharp, clean, photorealistic photo of a modern office interior, "
              "large bright windows, clear glass, realistic reflections, high detail")

    from pathlib import Path
    cache = Path(args.cache_dir)
    ids = [int(x) for x in args.views.split(",")]
    rows = []
    for i in ids:
        rgb = np.asarray(Image.open(cache / f"view{i}_render.png")).astype(np.float32) / 255.0
        gt = np.asarray(Image.open(cache / f"view{i}_gt.png")).astype(np.float32) / 255.0
        alpha = np.load(cache / f"view{i}_alpha.npy")
        conf = np.load(cache / f"view{i}_conf.npy")
        H, W = rgb.shape[:2]

        thr = np.quantile(conf, args.conf_quantile)
        gate = np.clip((thr - conf) / (thr + 1e-6), 0, 1)
        gate *= (alpha > 0.3)
        gate = np.power(gate, 0.7)

        img = Image.fromarray((rgb * 255).astype(np.uint8)).resize((args.size, args.size))
        enh = pipe(prompt=PROMPT, image=img, num_inference_steps=args.steps,
                   strength=args.strength, guidance_scale=0.0).images[0]  # schnell: no CFG
        enh = np.asarray(enh.resize((W, H))).astype(np.float32) / 255.0

        g3 = gate[..., None]
        blended = rgb * (1 - g3) + enh * g3
        gate_vis = np.repeat(gate[..., None], 3, axis=2)
        row = np.concatenate([gt, rgb, gate_vis, enh, blended], axis=1)
        rows.append(row)
        print(f"view {i}: enhanced (gate mean {gate.mean():.2f})")

    panel = (np.concatenate(rows, axis=0) * 255).astype(np.uint8)
    iio.imwrite(args.out, panel)
    print(f"wrote {args.out}  (rows: views {ids} | cols: GT | render | conf-gate | enhanced | blended)")


if __name__ == "__main__":
    main()
