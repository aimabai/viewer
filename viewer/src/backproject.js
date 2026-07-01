import * as THREE from 'three';

// Photo -> 3D: click a pixel in a captured photo, ray-cast through its DA3 depth,
// return the 3D world point. Depth is shipped downsampled (see src/export_depth.py)
// as a header-free uint16-millimetre binary, fetched lazily and cached per view.
const depthCache = new Map();

export async function loadDepth(viewId) {
  if (depthCache.has(viewId)) return depthCache.get(viewId);
  const buf = await (await fetch(`/scene/depth/${viewId}.bin`)).arrayBuffer();
  const arr = new Uint16Array(buf);           // already little-endian on all real browsers
  depthCache.set(viewId, arr);
  return arr;
}

// px, py: click position in FULL-RES image pixels (0..w, 0..h). view: a scene_meta
// views[] entry (has .intr, .R camera-to-world row-major 3x3, .pos). depthMeta: the
// scene_meta.depth_export block ({width, height}).
export function backprojectPixel(view, px, py, depthArr, depthMeta) {
  const { fx, fy, cx, cy, w, h } = view.intr;
  const { width: dw, height: dh } = depthMeta;
  const du = Math.min(dw - 1, Math.max(0, Math.round(px * dw / w)));
  const dv = Math.min(dh - 1, Math.max(0, Math.round(py * dh / h)));
  const mm = depthArr[dv * dw + du];
  if (!mm) return null;                       // 0 = invalid/no-depth pixel
  const depth = mm / 1000;
  const X = (px - cx) * depth / fx;
  const Y = (py - cy) * depth / fy;
  const Z = depth;
  const R = view.R;                            // camera-to-world, row-major 3x3
  return new THREE.Vector3(
    R[0] * X + R[1] * Y + R[2] * Z + view.pos[0],
    R[3] * X + R[4] * Y + R[5] * Z + view.pos[1],
    R[6] * X + R[7] * Y + R[8] * Z + view.pos[2],
  );
}

// A single reusable "dropped pin" marker.
export function makeMarker() {
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.05, 16, 16),
    new THREE.MeshBasicMaterial({ color: 0xffd23f }));
  mesh.visible = false;
  mesh.renderOrder = 999;
  return mesh;
}
