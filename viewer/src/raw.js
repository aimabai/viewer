// Minimal standalone splat viewer — no trust modes, minimap, or click hooks. Just the
// plain three.js + gsplat-3d rasterizer, for opening an exported .ply/.ksplat file
// directly ("in an editor", separate from the full interactive walk-through).
import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

const statusEl = document.getElementById('status');
const setStatus = (t) => { if (statusEl) statusEl.textContent = t; };

const EXT_FORMAT = {
  ply: GaussianSplats3D.SceneFormat.Ply,
  splat: GaussianSplats3D.SceneFormat.Splat,
  ksplat: GaussianSplats3D.SceneFormat.KSplat,
  spz: GaussianSplats3D.SceneFormat.Spz,
};

async function loadScene(path, format) {
  document.getElementById('panel').style.display = 'none';
  setStatus('Loading ' + path + ' …');
  const viewer = new GaussianSplats3D.Viewer({ sharedMemoryForWorkers: false });
  window.__viewer = viewer;
  try {
    await viewer.addSplatScene(path, { showLoadingUI: true, format, splatAlphaRemovalThreshold: 5 });
    viewer.start();
    setStatus('Drag to orbit · scroll to zoom');
  } catch (e) {
    console.error(e);
    setStatus('Error: ' + (e.message || e));
  }
}

function extOf(name) {
  const m = /\.([a-z0-9]+)$/i.exec(name);
  return m ? m[1].toLowerCase() : '';
}

function wireDrop() {
  const drop = document.getElementById('drop');
  const input = document.getElementById('file-input');
  const openFile = (file) => {
    const ext = extOf(file.name);
    const format = EXT_FORMAT[ext];
    if (format === undefined) { setStatus('Unrecognized extension: .' + ext); return; }
    loadScene(URL.createObjectURL(file), format);
  };
  input.addEventListener('change', (e) => { if (e.target.files[0]) openFile(e.target.files[0]); });
  drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
  drop.addEventListener('dragleave', () => drop.classList.remove('over'));
  drop.addEventListener('drop', (e) => {
    e.preventDefault(); drop.classList.remove('over');
    if (e.dataTransfer.files[0]) openFile(e.dataTransfer.files[0]);
  });
}

const params = new URLSearchParams(location.search);
const src = params.get('src');
if (src) {
  loadScene(src, EXT_FORMAT[extOf(src)]);
} else {
  wireDrop();
}
