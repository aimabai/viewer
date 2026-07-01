"""Export compact per-view depth maps for the viewer's photo -> 3D backprojection
(click a pixel in a captured photo, ray-cast through DA3 depth, drop a 3D marker —
the interaction named in the task brief §2 that we hadn't built yet).

Downsampled 2x and stored as raw uint16 millimetres (no PNG — browsers decode 2D
canvases to 8-bit, which would truncate depth precision; a header-free binary blob
read via fetch()+DataView is simpler and exact). One ~127KB file per view, fetched
lazily only for the view currently shown in the viewer.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data import load_views


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="../dataset")
    ap.add_argument("--out-dir", default="../viewer/public/scene/depth")
    ap.add_argument("--meta", default="../viewer/public/scene/scene_meta.json")
    ap.add_argument("--downsample", type=int, default=2)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    views = load_views(args.dataset)
    s = args.downsample
    for v in views:
        d = v.depth[::s, ::s]
        mm = np.clip(d * 1000.0, 0, 65535).astype("<u2")
        (out / f"{v.id}.bin").write_bytes(mm.tobytes())
    h, w = views[0].depth[::s, ::s].shape
    print(f"wrote {len(views)} depth maps to {out} | {w}x{h} uint16mm | "
          f"downsample={s} (apply to full-res intrinsics/pixels at read time)")

    meta_path = Path(args.meta)
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        meta["depth_export"] = {"width": int(w), "height": int(h), "downsample": s, "unit": "mm"}
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"updated {meta_path} with depth_export")


if __name__ == "__main__":
    main()
