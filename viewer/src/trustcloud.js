import * as THREE from 'three';

// Parse trust.bin (see src/export.py write_trust_bin):
//   int32 count, then float32 pos[count*3], uint8 rgb[count*3],
//   uint8 conf[count], uint8 nviews[count], int16 view[count].
function parseTrustBin(buffer) {
  const dv = new DataView(buffer);
  const count = dv.getInt32(0, true);
  let off = 4;
  const pos = new Float32Array(buffer.slice(off, off + count * 12)); off += count * 12;
  const rgb = new Uint8Array(buffer.slice(off, off + count * 3)); off += count * 3;
  const conf = new Uint8Array(buffer.slice(off, off + count)); off += count;
  const nviews = new Uint8Array(buffer.slice(off, off + count)); off += count;
  const view = new Int16Array(buffer.slice(off, off + count * 2)); off += count * 2;
  return { count, pos, rgb, conf, nviews, view };
}

const VERT = /* glsl */`
  attribute vec3 aColor;     // photographic DC colour
  attribute float aConf;     // 0..1 confidence
  attribute float aCov;      // 0..1 normalised coverage (n_views)
  attribute float aNV;       // raw n_views (view count)
  attribute float aView;     // supervising view id (-1 = unseen)
  uniform int uMode;         // 0 rgb, 1 confidence, 2 coverage, 3 by-view
  uniform float uSize;
  uniform float uCovThresh;  // hide points with n_views below this (raw count)
  varying vec3 vColor;
  varying float vDrop;

  // compact turbo-ish blue->green->red ramp
  vec3 ramp(float t){
    t = clamp(t, 0.0, 1.0);
    return clamp(vec3(1.5 - abs(4.0*t - 3.0),
                      1.5 - abs(4.0*t - 2.0),
                      1.5 - abs(4.0*t - 1.0)), 0.0, 1.0);
  }
  vec3 hue(float h){
    h = fract(h);
    return clamp(abs(mod(h*6.0 + vec3(0.0,4.0,2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
  }

  void main(){
    vDrop = 0.0;
    if (uMode == 0)      vColor = aColor;
    else if (uMode == 1) vColor = ramp(aConf);
    else if (uMode == 2) vColor = ramp(aCov);
    else {
      if (aView < 0.0) { vColor = vec3(0.12); }
      else vColor = hue(aView * 0.61803398);
    }
    if (aView < 0.0 && uMode != 0) vDrop = 1.0;          // unseen → grey in trust modes
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_Position = projectionMatrix * mv;
    // coverage mode: cull points seen by fewer than the threshold # of views
    if (uMode == 2 && aNV < uCovThresh) { gl_Position = vec4(2.0, 2.0, 2.0, 1.0); return; }
    gl_PointSize = clamp(uSize / max(-mv.z, 0.1), 1.0, 4.0);
  }
`;

const FRAG = /* glsl */`
  precision mediump float;
  varying vec3 vColor;
  varying float vDrop;
  void main(){
    vec2 d = gl_PointCoord - vec2(0.5);
    if (dot(d, d) > 0.25) discard;                       // round points
    float a = vDrop > 0.5 ? 0.15 : 1.0;
    gl_FragColor = vec4(vColor, a);
  }
`;

export function makeTrustCloud(buffer, sizePx = 45) {
  const t = parseTrustBin(buffer);
  // normalise coverage by 95th-percentile n_views for stable colours
  const sorted = Float32Array.from(t.nviews).sort();
  const cov95 = Math.max(1, sorted[Math.floor(0.95 * sorted.length)]);

  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(t.pos, 3));
  const col = new Float32Array(t.count * 3);
  const cov = new Float32Array(t.count);
  const nvF = new Float32Array(t.count);
  const viewF = new Float32Array(t.count);
  const confF = new Float32Array(t.count);
  for (let i = 0; i < t.count; i++) {
    col[i*3] = t.rgb[i*3] / 255; col[i*3+1] = t.rgb[i*3+1] / 255; col[i*3+2] = t.rgb[i*3+2] / 255;
    confF[i] = t.conf[i] / 255;
    cov[i] = Math.min(1, t.nviews[i] / cov95);
    nvF[i] = t.nviews[i];
    viewF[i] = t.view[i];
  }
  geo.setAttribute('aColor', new THREE.BufferAttribute(col, 3));
  geo.setAttribute('aConf', new THREE.BufferAttribute(confF, 1));
  geo.setAttribute('aCov', new THREE.BufferAttribute(cov, 1));
  geo.setAttribute('aNV', new THREE.BufferAttribute(nvF, 1));
  geo.setAttribute('aView', new THREE.BufferAttribute(viewF, 1));

  const mat = new THREE.ShaderMaterial({
    uniforms: {
      uMode: { value: 1 }, uSize: { value: sizePx },
      uCovThresh: { value: 1 }, uNViewsRaw: { value: 0 },
    },
    vertexShader: VERT, fragmentShader: FRAG, transparent: true, depthWrite: true,
  });
  const points = new THREE.Points(geo, mat);
  points.frustumCulled = false;
  // raw per-point arrays exposed for lookups after raycasting (e.g. "which view
  // supervised the point I clicked?" — see main.js placeMarkerFrom3DClick)
  return { points, material: mat, count: t.count, view: t.view, nviews: t.nviews, conf: t.conf, pos: t.pos };
}
