import * as THREE from 'three';

// First-person "walk" controls (WASD + mouse-look via pointer lock), built around the
// scene's true world-up (the scan frame is tilted ~31° off Y). W/S/A/D move on the
// horizontal floor plane regardless of where you look; Space / Shift go up / down.
// No zoom — you move — which avoids the orbit "keep zooming out" problem.
export class WalkControls {
  constructor(camera, dom, worldUp, e1, e2, opts = {}) {
    this.cam = camera;
    this.dom = dom;
    this.up = new THREE.Vector3(...worldUp).normalize();
    this.e1 = new THREE.Vector3(...e1).normalize();   // floor axis (yaw = 0 reference)
    this.e2 = new THREE.Vector3(...e2).normalize();
    this.pos = camera.position.clone();
    this.yaw = 0; this.pitch = 0;
    this.speed = opts.speed ?? 2.2;                   // metres / second
    this.sens = opts.sens ?? 0.0022;
    this.keys = {};
    this.locked = false;
    this.cam.up.copy(this.up);
    this._initEvents();
  }

  _initEvents() {
    this.dom.style.cursor = 'pointer';
    this.dom.addEventListener('click', () => { if (!this.locked) this.dom.requestPointerLock(); });
    document.addEventListener('pointerlockchange', () => {
      this.locked = document.pointerLockElement === this.dom;
      if (this.onLockChange) this.onLockChange(this.locked);
    });
    document.addEventListener('mousemove', (e) => {
      if (!this.locked) return;
      this.yaw -= e.movementX * this.sens;
      this.pitch -= e.movementY * this.sens;
      const lim = Math.PI / 2 - 0.05;
      this.pitch = Math.max(-lim, Math.min(lim, this.pitch));
    });
    window.addEventListener('keydown', (e) => {
      this.keys[e.code] = true;
      if (['KeyW', 'KeyA', 'KeyS', 'KeyD', 'Space'].includes(e.code)) e.preventDefault();
    });
    window.addEventListener('keyup', (e) => { this.keys[e.code] = false; });
  }

  _horizForward() {
    return this.e1.clone().multiplyScalar(Math.cos(this.yaw)).addScaledVector(this.e2, Math.sin(this.yaw));
  }
  _forward() {
    return this._horizForward().multiplyScalar(Math.cos(this.pitch)).addScaledVector(this.up, Math.sin(this.pitch)).normalize();
  }

  // teleport to a position looking along a direction (used by the minimap path)
  setPose(pos, lookDir) {
    this.pos.set(pos[0], pos[1], pos[2]);
    const d = new THREE.Vector3(...lookDir).normalize();
    const vert = THREE.MathUtils.clamp(d.dot(this.up), -1, 1);
    this.pitch = Math.asin(vert);
    const horiz = d.clone().addScaledVector(this.up, -vert).normalize();
    this.yaw = Math.atan2(horiz.dot(this.e2), horiz.dot(this.e1));
  }

  update(dt) {
    const hf = this._horizForward();
    const right = hf.clone().cross(this.up).normalize();   // strafe axis (horizontal)
    const v = new THREE.Vector3();
    const k = this.keys;
    if (k['KeyW'] || k['ArrowUp']) v.add(hf);
    if (k['KeyS'] || k['ArrowDown']) v.sub(hf);
    if (k['KeyD'] || k['ArrowRight']) v.add(right);
    if (k['KeyA'] || k['ArrowLeft']) v.sub(right);
    if (k['Space']) v.add(this.up);
    if (k['ShiftLeft'] || k['ShiftRight'] || k['KeyC']) v.sub(this.up);
    const boost = (k['KeyW'] || k['KeyS'] || k['KeyA'] || k['KeyD']) && (k['ShiftLeft']) ? 1 : 1;
    if (v.lengthSq() > 0) { v.normalize().multiplyScalar(this.speed * boost * Math.min(dt, 0.1)); this.pos.add(v); }
    this.cam.position.copy(this.pos);
    this.cam.up.copy(this.up);
    this.cam.lookAt(this.pos.clone().add(this._forward()));
  }
}
