// Phase 3 — splat viewer + trust modes + floor-plan minimap (bidirectional sync).
import * as THREE from 'three';
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';
import { makeTrustCloud } from './trustcloud.js';
import { Minimap } from './minimap.js';
import { WalkControls } from './controls.js';
import { loadDepth, backprojectPixel, makeMarker } from './backproject.js';

const statusEl = document.getElementById('status');
const setStatus = (t) => { if (statusEl) statusEl.textContent = t; };

let viewer = null, trust = null, minimap = null, meta = null, controls = null;
let mode = 'photographic', selectedFrame = null, marker = null, currentView = null;
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 0.09;   // wider net so there's usually >1 candidate to search

// NOTE: floor completion is no longer a separate mesh/plane object. It's baked
// directly into the exported Gaussian model as synthetic, floor-aligned splats
// (src/inpaint_planes.py) — generative (SDXL-inpainting) fill of ONLY the pixels
// with no real observation, merged into scene.ksplat itself. Marked confidence=0/
// n_views=0 in trust.bin so the confidence/coverage modes still show it as
// ungrounded. Ceiling is intentionally left unfilled.
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
  viewer.start();   // start the render loop immediately, splats stream in async below

  const buf = await (await fetch('/scene/trust.bin')).arrayBuffer();
  trust = makeTrustCloud(buf);
  trust.points.visible = false;
  viewer.threeScene.add(trust.points);

  marker = makeMarker();
  viewer.threeScene.add(marker);

  // first-person walk controls; start standing in the cluster looking into the room
  controls = new WalkControls(viewer.camera, viewer.renderer.domElement, up, e1, e2);
  controls.setPose(meta.scan_points[0].pos, sceneCenter.clone().sub(new THREE.Vector3(...meta.scan_points[0].pos)).toArray());
  controls.onLockChange = (locked) => {
    // Only announce the "unlocked" (escaped) state. Announcing "locked" here would
    // race with onSceneClick's result message on the SAME click (WalkControls also
    // requests pointer lock on an unlocked click) and clobber it a moment later.
    if (!locked) setStatus('Click the scene to walk · then WASD + mouse · Esc to release');
    document.getElementById('crosshair').style.display = locked ? '' : 'none';
  };

  minimap = new Minimap(document.getElementById('minimap'), meta, navigateTo);

  // Wire up interaction BEFORE the heavy splat load — WASD, minimap, and the
  // click-to-inspect hook only need `trust` (already loaded above), not the splat
  // mesh, so there's no reason to gate them behind a ~168 MB download.
  wireUI();
  wireClickHooks();
  drawLoop();
  setStatus('Loading full-quality splats (~168 MB)… (walking + click-to-inspect already work)');

  await viewer.addSplatScene('/scene/scene.ksplat', { showLoadingUI: true, splatAlphaRemovalThreshold: 5 });
  if (viewer.splatMesh) viewer.splatMesh.visible = (mode === 'photographic');
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
}

const MODE_HELP = {
  photographic: 'Drag to orbit · scroll to zoom · click the path to walk through',
  confidence: 'Point colour = DA3 depth confidence (blue = low → red = high). Reds are where the depth prior was most trusted.',
  coverage: 'Point colour = how many of the 240 photos saw each point (red = many). Drag the slider — barely-seen points dissolve out continuously as you raise the threshold, revealing the low-trust region.',
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
  box.style.display = 'block';   // CSS sets display:none by default; '' doesn't override that
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
  try {
    // No pointer-lock requirement: the crosshair always samples screen-centre
    // regardless of cursor/lock state, so raycasting doesn't need it.
    if (!trust) { setStatus('Trust data not loaded yet — try again in a moment.'); return; }
    raycaster.setFromCamera(new THREE.Vector2(0, 0), viewer.camera);   // screen-centre crosshair
    const hits = raycaster.intersectObject(trust.points, false);
    if (!hits.length) { setStatus('No reconstructed point at the crosshair.'); return; }
    // Take the closest hit for the MARKER (visually where you clicked), but for the
    // supervising-view lookup, walk outward past any unsupervised floaters — sparse
    // noise points sitting slightly in front of real geometry along the ray would
    // otherwise win every time even though the surface behind them is well-supervised.
    placeMarker(hits[0].point);
    const hit = hits.find((h) => trust.view[h.index] >= 0);
    if (!hit) { setStatus(`No clearly-supervised point near the crosshair (${hits.length} floater(s) only).`); return; }
    const idx = hit.index;
    const v = meta.views.find((x) => x.id === trust.view[idx]);
    if (v) showViewThumb(v, `supervising view · scan-point ${v.frame} · yaw ${v.yaw}° (n_views=${trust.nviews[idx]})`);
    else setStatus(`Hit view id ${trust.view[idx]} but couldn't find it in scene_meta.views — data mismatch.`);
  } catch (e) {
    console.error('[click] error:', e);
    setStatus('Click-to-inspect error: ' + (e.message || e));
  }
}

function wireClickHooks() {
  document.getElementById('viewinfo-img').addEventListener('click', onPhotoClick);
  // Listen at the document level rather than viewer.renderer.domElement directly —
  // gsplat-3d's Viewer may layer additional elements/canvases over the base one, and
  // a listener on the wrong specific element would silently never fire. Filter out
  // clicks that originated inside any of our own UI panels.
  document.addEventListener('click', (e) => {
    if (e.target.closest('#panel, #minimap-wrap, #viewinfo, #status')) return;
    onSceneClick();
  });
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
