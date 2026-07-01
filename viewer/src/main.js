// Phase 3 — splat viewer + trust modes + floor-plan minimap (bidirectional sync).
import * as THREE from 'three';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';
import { makeTrustCloud } from './trustcloud.js';
import { Minimap } from './minimap.js';
import { WalkControls } from './controls.js';
import { loadDepth, backprojectPixel, makeMarker } from './backproject.js';

const statusEl = document.getElementById('status');
const setStatus = (t) => { if (statusEl) statusEl.textContent = t; };

let viewer = null, trust = null, minimap = null, meta = null, controls = null, inferredPlanes = null;
let mode = 'photographic', selectedFrame = null, marker = null, currentView = null;
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.06;

// Two planes (floor, ceiling) at the fitted heights, spanning the room footprint,
// oriented perpendicular to world-up. Completion of known geometry, not hallucination
// — textured with a diffusion-inpainted top-down completion when available
// (src/inpaint_planes.py), else a flat fallback colour.
function makeInferredPlanes(meta) {
  const ip = meta.inferred_planes;
  const e1 = new THREE.Vector3(...meta.floor_basis.e1);
  const e2 = new THREE.Vector3(...meta.floor_basis.e2);
  const up = new THREE.Vector3(...meta.world_up);
  // Basis matrix (e1, e2, up) as columns: local plane +X -> e1, +Y -> e2, +Z -> up.
  // (setFromUnitVectors alone only fixes +Z->up, leaving an undetermined twist about
  // that axis — wrong once the planes carry an oriented texture, not just flat colour.)
  const basis = new THREE.Matrix4().makeBasis(e1, e2, up);
  const q = new THREE.Quaternion().setFromRotationMatrix(basis);

  const rb = meta.reconstructed_bounds;
  let bounds = ip.plane_bounds;                // present once inpaint_planes.py has run
  if (!bounds) {                                // fallback: pad reconstructed_bounds about its centre
    const cu0 = (rb.min[0] + rb.max[0]) / 2, cw0 = (rb.min[1] + rb.max[1]) / 2;
    const hu = (rb.max[0] - rb.min[0]) / 2 * 1.15, hv = (rb.max[1] - rb.min[1]) / 2 * 1.15;
    bounds = { min: [cu0 - hu, cw0 - hv], max: [cu0 + hu, cw0 + hv] };
  }
  const cu = (bounds.min[0] + bounds.max[0]) / 2, cw = (bounds.min[1] + bounds.max[1]) / 2;
  const w = Math.max(0.5, bounds.max[0] - bounds.min[0]);
  const h = Math.max(0.5, bounds.max[1] - bounds.min[1]);

  const loader = new THREE.TextureLoader();
  const group = new THREE.Group();
  // Floor: classical (non-generative) inpainting extends the real observed texture
  // (src/inpaint_planes.py). Ceiling: intentionally left flat — less predictable than
  // a floor, so nearby-pixel extrapolation was judged not defensible there.
  const specs = [
    { height: ip.floor_height, color: ip.floor_color, tex: 'floor_texture.png', textured: ip.floor_textured },
    { height: ip.ceiling_height, color: ip.ceiling_color, tex: 'ceiling_texture.png', textured: ip.ceiling_textured },
  ];
  for (const { height, color, tex, textured } of specs) {
    const material = new THREE.MeshBasicMaterial({
      color: new THREE.Color(...color), side: THREE.DoubleSide });
    const mesh = new THREE.Mesh(new THREE.PlaneGeometry(w, h), material);
    mesh.position.copy(e1.clone().multiplyScalar(cu).addScaledVector(e2, cw).addScaledVector(up, height));
    mesh.quaternion.copy(q);
    group.add(mesh);
    if (textured) {
      loader.load('/scene/' + tex, (t) => {
        t.flipY = false;              // our raster's row0 = v-min; three.js default would invert it
        t.colorSpace = THREE.SRGBColorSpace;
        material.map = t; material.color.set(0xffffff); material.needsUpdate = true;
      });
    }
  }
  return group;
}
const clock = new THREE.Clock();
const sceneCenter = new THREE.Vector3();

async function main() {
  meta = await (await fetch('/scene/scene_meta.json')).json();

  const up = meta.world_up;
  const rb = meta.reconstructed_bounds, e1 = meta.floor_basis.e1, e2 = meta.floor_basis.e2;
  // centre of the reconstructed region, back in 3D via the floor basis + up
  const cu = (rb.min[0] + rb.max[0]) / 2, cw = (rb.min[1] + rb.max[1]) / 2;
  const center = [0, 1, 2].map((i) => e1[i] * cu + e2[i] * cw + up[i] * 0 + 0);
  // anchor height: average scan-point along up
  const meanUp = meta.scan_points.reduce((s, sp) => s + sp.pos[0] * up[0] + sp.pos[1] * up[1] + sp.pos[2] * up[2], 0) / meta.scan_points.length;
  for (let i = 0; i < 3; i++) center[i] += up[i] * meanUp;
  const diag = Math.hypot(rb.max[0] - rb.min[0], rb.max[1] - rb.min[1]);
  const pos = [0, 1, 2].map((i) => center[i] + up[i] * diag * 0.30 + e1[i] * diag * 0.85);
  sceneCenter.set(center[0], center[1], center[2]);

  viewer = new GaussianSplats3D.Viewer({
    cameraUp: up,
    initialCameraPosition: pos,
    initialCameraLookAt: center,
    sharedMemoryForWorkers: false,
    useBuiltInControls: false,        // we drive the camera with WalkControls
  });
  window.__viewer = viewer;

  setStatus('Loading full-quality splats (~168 MB)…');
  await viewer.addSplatScene('/scene/scene.ksplat', { showLoadingUI: true, splatAlphaRemovalThreshold: 5 });
  viewer.start();

  const buf = await (await fetch('/scene/trust.bin')).arrayBuffer();
  trust = makeTrustCloud(buf);
  trust.points.visible = false;
  viewer.threeScene.add(trust.points);

  marker = makeMarker();
  viewer.threeScene.add(marker);

  // Inferred floor + ceiling planes: the single horizontal pitch never sees them, so
  // they reconstruct as garbage. They are trivially-known planes — fill them cleanly.
  inferredPlanes = makeInferredPlanes(meta);
  viewer.threeScene.add(inferredPlanes);

  // first-person walk controls; start standing in the cluster looking into the room
  controls = new WalkControls(viewer.camera, viewer.renderer.domElement, up, e1, e2);
  controls.setPose(meta.scan_points[0].pos, sceneCenter.clone().sub(new THREE.Vector3(...meta.scan_points[0].pos)).toArray());
  controls.onLockChange = (locked) => {
    setStatus(locked ? MODE_HELP[mode] : 'Click the scene to walk · then WASD + mouse · Esc to release');
    document.getElementById('crosshair').style.display = locked ? '' : 'none';
  };

  minimap = new Minimap(document.getElementById('minimap'), meta, navigateTo);

  wireUI();
  wireClickHooks();
  drawLoop();
  setStatus('Click the scene to walk · WASD + mouse-look · Space/Shift up/down · click the path to jump');
}

function wireUI() {
  document.querySelectorAll('button.mode').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('button.mode').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      setMode(btn.dataset.mode);
    });
  });
  const slider = document.getElementById('cov-slider');
  slider.addEventListener('input', () => {
    document.getElementById('cov-val').textContent = slider.value;
    if (trust) trust.material.uniforms.uCovThresh.value = parseFloat(slider.value);
  });
  document.getElementById('toggle-planes').addEventListener('change', (e) => {
    if (inferredPlanes) inferredPlanes.visible = e.target.checked;
  });
}

const MODE_HELP = {
  photographic: 'Drag to orbit · scroll to zoom · click the path to walk through',
  confidence: 'Point colour = DA3 depth confidence (blue = low → red = high). Reds are where the depth prior was most trusted.',
  coverage: 'Point colour = how many of the 240 photos saw each point (red = many). The slider hides barely-seen points — what falls away is the low-trust region.',
  byview: 'Each colour = one of the 240 source photos; the colour of a surface = which photo reconstructed it. Because capture is one rotating vantage, you see the ~12 yaw "wedges" tiling the room.',
};

function setMode(m) {
  mode = m;
  const photographic = m === 'photographic';
  if (viewer && viewer.splatMesh) viewer.splatMesh.visible = photographic;
  if (trust) {
    trust.points.visible = !photographic;
    trust.material.uniforms.uMode.value = { confidence: 1, coverage: 2, byview: 3 }[m] ?? 0;
  }
  document.getElementById('cov').style.display = m === 'coverage' ? '' : 'none';
  setStatus(MODE_HELP[m] || '');
}

// map -> 3D: jump to a point on the walkable path, facing along the path (into the room)
function navigateTo(pos3, tangent3) {
  if (controls) controls.setPose(pos3, tangent3);
  // show the nearest real captured photo as a reference for where you are
  let best = null, bd = 1e9;
  for (const v of meta.views) {
    const d = (v.pos[0] - pos3[0]) ** 2 + (v.pos[1] - pos3[1]) ** 2 + (v.pos[2] - pos3[2]) ** 2;
    if (d < bd) { bd = d; best = v; }
  }
  if (best) showViewThumb(best);
}

function showViewThumb(v, label) {
  currentView = v;
  const box = document.getElementById('viewinfo');
  box.style.display = '';
  document.getElementById('viewinfo-label').textContent =
    label || `nearest capture · scan-point ${v.frame} · yaw ${v.yaw}°`;
  document.getElementById('viewinfo-img').src = '/' + v.pano;
}

// --- bidirectional click hook (task brief §2: "click a view -> ray-cast through the
// depth map -> drop a 3D marker", plus its natural inverse) ---

function placeMarker(worldPos) {
  marker.position.copy(worldPos);
  marker.visible = true;
}

// Photo -> 3D: click a pixel in the shown photo, ray-cast through its DA3 depth.
async function onPhotoClick(e) {
  if (!currentView || !meta.depth_export) return;
  const img = e.target;
  const px = (e.offsetX / img.clientWidth) * currentView.intr.w;
  const py = (e.offsetY / img.clientHeight) * currentView.intr.h;
  const depthArr = await loadDepth(currentView.id);
  const world = backprojectPixel(currentView, px, py, depthArr, meta.depth_export);
  if (world) {
    placeMarker(world);
    setStatus(`Marker dropped at (${world.x.toFixed(2)}, ${world.y.toFixed(2)}, ${world.z.toFixed(2)}) ` +
      `from scan-point ${currentView.frame} · yaw ${currentView.yaw}°`);
  } else {
    setStatus('No valid depth at that pixel (edge/invalid region) — try elsewhere in the photo.');
  }
}

// 3D -> photo: click a point on the reconstruction (crosshair, centre of screen — the
// pointer is locked while walking), find which view supervised it, show that photo.
function onSceneClick() {
  if (!controls.locked || !trust) return;
  raycaster.setFromCamera(new THREE.Vector2(0, 0), viewer.camera);   // screen-centre crosshair
  const hits = raycaster.intersectObject(trust.points, false);
  if (!hits.length) { setStatus('No reconstructed point at the crosshair.'); return; }
  const idx = hits[0].index;
  placeMarker(hits[0].point);
  const viewId = trust.view[idx];
  if (viewId < 0) { setStatus('This point was not clearly supervised by any single view.'); return; }
  const v = meta.views.find((x) => x.id === viewId);
  if (v) showViewThumb(v, `supervising view · scan-point ${v.frame} · yaw ${v.yaw}° (n_views=${trust.nviews[idx]})`);
}

function wireClickHooks() {
  document.getElementById('viewinfo-img').addEventListener('click', onPhotoClick);
  viewer.renderer.domElement.addEventListener('click', onSceneClick);
}

function drawLoop() {
  requestAnimationFrame(drawLoop);
  if (controls) controls.update(clock.getDelta());
  if (minimap && viewer && viewer.camera) minimap.draw(viewer.camera, selectedFrame);
}

main().catch((e) => {
  console.error(e);
  const msg = (e && e.message) ? e.message : String(e);
  setStatus('Error: ' + msg + (/webgl|context/i.test(msg) ? '\n\nQuit Chrome COMPLETELY and reopen.' : ''));
});
