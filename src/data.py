"""Load the InfraScan DA3 dataset into gsplat-ready tensors.

The DA3 output uses the OpenCV camera convention (+X right, +Y down, +Z forward),
which is exactly what gsplat expects — so camera-to-world rotations are used as-is,
with no axis flip. (The nerfstudio route needs a diag(1,-1,-1) flip; we don't.)

Conventions
-----------
- `cameras.json` gives `R` (3x3 camera->world, row-major) and `pos` (world, metres).
- A camera-frame point p_c maps to world as  p_w = R @ p_c + pos   (camera-to-world).
- gsplat wants `viewmat` = world-to-camera = inverse of [[R, pos], [0, 1]].
- `depth` (from frame_N.npz) is camera-local z-depth in metres; backprojection is
      X = (u - cx) * d / fx,  Y = (v - cy) * d / fy,  Z = d.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class View:
    """One perspective view with everything reconstruction needs."""
    id: int
    frame: int          # scan-point index (0..19)
    yaw: int            # degrees
    image: np.ndarray   # (H, W, 3) uint8
    depth: np.ndarray   # (H, W) float32, metric metres
    conf: np.ndarray    # (H, W) float32, DA3 per-pixel confidence
    person_mask: np.ndarray  # (H, W) uint8, 255 = was-person
    K: np.ndarray       # (3, 3) float32 intrinsics
    c2w: np.ndarray     # (4, 4) float32 camera-to-world
    pano: str           # relative image path

    @property
    def hw(self) -> tuple[int, int]:
        return self.image.shape[:2]

    @property
    def viewmat(self) -> np.ndarray:
        """World-to-camera (4x4), what gsplat.rasterization consumes."""
        return np.linalg.inv(self.c2w).astype(np.float32)

    def train_mask(self) -> np.ndarray:
        """(H, W) float32 in [0, 1]: 1 = supervise this pixel, 0 = ignore.

        People are excluded (inverse of person_mask). Shadows and reflections
        are NOT masked by DA3 and will bake in as static texture — a known limit.
        """
        return (self.person_mask == 0).astype(np.float32)


def load_views(dataset: str | Path) -> list[View]:
    """Load all perspective views from a dataset directory."""
    dataset = Path(dataset)
    cams = json.loads((dataset / "cameras.json").read_text())
    npz_dir = dataset / "da3" / "results_output"

    views: list[View] = []
    for c in sorted(cams, key=lambda e: e["id"]):
        npz = np.load(npz_dir / f"frame_{c['id']}.npz")
        c2w = np.eye(4, dtype=np.float32)
        c2w[:3, :3] = np.asarray(c["R"], dtype=np.float32)
        c2w[:3, 3] = np.asarray(c["pos"], dtype=np.float32)
        views.append(View(
            id=c["id"], frame=c["frame"], yaw=c["yaw"],
            image=npz["image"], depth=npz["depth"].astype(np.float32),
            conf=npz["conf"].astype(np.float32), person_mask=npz["person_mask"],
            K=npz["intrinsics"].astype(np.float32), c2w=c2w, pano=c["pano"],
        ))
    return views


def sanity_check(views: list[View]) -> dict:
    """README §4: confirm structure before training. Raises on surprises."""
    from collections import defaultdict

    by_frame: dict[int, list[View]] = defaultdict(list)
    for v in views:
        by_frame[v.frame].append(v)

    n_sp = len(by_frame)
    per_sp = sorted({len(g) for g in by_frame.values()})

    # Forward axis (3rd column of R) of one scan-point should span the floor circle.
    sp0 = sorted(by_frame[min(by_frame)], key=lambda v: v.yaw)
    fwd = np.array([v.c2w[:3, 2] for v in sp0])
    az = np.degrees(np.arctan2(fwd[:, 2], fwd[:, 0]))  # floor plane is x-z
    spread = float(az.max() - az.min())

    report = {
        "n_views": len(views),
        "n_scan_points": n_sp,
        "views_per_scan_point": per_sp,
        "scan_point0_yaw_azimuth_spread_deg": round(spread, 1),
        "depth_range_m": [float(min(v.depth.min() for v in views)),
                          float(max(v.depth.max() for v in views))],
    }
    assert len(views) == 240, f"expected 240 views, got {len(views)}"
    assert per_sp == [12], f"expected 12 views/scan-point, got {per_sp}"
    assert spread > 270, f"yaw forward axes don't span the circle ({spread:.0f} deg)"
    return report


if __name__ == "__main__":
    import sys
    vs = load_views(sys.argv[1] if len(sys.argv) > 1 else "dataset")
    print(json.dumps(sanity_check(vs), indent=2))
