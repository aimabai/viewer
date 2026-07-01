import * as THREE from 'three';

// Top-down floor-plan minimap, bidirectionally synced with the 3D view.
// Background is floorplan.png — a top-down density image of the reconstruction, so it
// reads as an actual floor plan. The scan frame is tilted, so 2D coords are world
// points projected onto the floor basis (e1, e2); the map is framed on the
// reconstructed region (where there is data), not the whole empty floor.
export class Minimap {
  constructor(canvas, meta, onNavigate) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.meta = meta;
    this.onNavigate = onNavigate;        // (pos3 [x,y,z], tangent3 [x,y,z]) => void
    this.margin = 12;
    this.path = meta.path || [];
    // precompute 3D tangent at each waypoint (direction of travel along the path)
    this.tangents = this.path.map((wp, i) => {
      const a = this.path[Math.max(0, i - 1)].pos, b = this.path[Math.min(this.path.length - 1, i + 1)].pos;
      const t = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
      const n = Math.hypot(...t) || 1; return t.map((v) => v / n);
    });
    this.e1 = new THREE.Vector3(...meta.floor_basis.e1);
    this.e2 = new THREE.Vector3(...meta.floor_basis.e2);

    // frame on the reconstructed bounds with a little padding
    const b = meta.reconstructed_bounds;
    const padU = (b.max[0] - b.min[0]) * 0.05, padV = (b.max[1] - b.min[1]) * 0.05;
    this.bounds = { min: [b.min[0] - padU, b.min[1] - padV], max: [b.max[0] + padU, b.max[1] + padV] };
    // floorplan.png covers exactly the (unpadded) reconstructed bounds
    this.planBounds = b;
    this.plan = new Image();
    this.plan.src = '/scene/floorplan.png';

    canvas.addEventListener('click', (e) => this._onClick(e));
  }

  _proj(v) { return [v.dot(this.e1), v.dot(this.e2)]; }
  _toCanvas(u, w) {
    const { min, max } = this.bounds, m = this.margin, W = this.canvas.width, H = this.canvas.height;
    return [m + (u - min[0]) / (max[0] - min[0]) * (W - 2 * m),
            m + (w - min[1]) / (max[1] - min[1]) * (H - 2 * m)];
  }
  _fromCanvas(px, py) {
    const { min, max } = this.bounds, m = this.margin, W = this.canvas.width, H = this.canvas.height;
    return [min[0] + (px - m) / (W - 2 * m) * (max[0] - min[0]),
            min[1] + (py - m) / (H - 2 * m) * (max[1] - min[1])];
  }

  _onClick(e) {
    const r = this.canvas.getBoundingClientRect();
    const [u, w] = this._fromCanvas(e.clientX - r.left, e.clientY - r.top);
    // snap to the nearest path waypoint and travel along the path there
    let bi = -1, bd = 1e9;
    this.path.forEach((wp, i) => {
      const d = (wp.xy[0] - u) ** 2 + (wp.xy[1] - w) ** 2;
      if (d < bd) { bd = d; bi = i; }
    });
    if (bi >= 0) this.onNavigate(this.path[bi].pos, this.tangents[bi]);
  }

  draw(camera, selectedFrame = null) {
    const ctx = this.ctx, W = this.canvas.width, H = this.canvas.height;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#10151c'; ctx.fillRect(0, 0, W, H);

    // floor-plan density image, stretched to its (unpadded) bounds
    if (this.plan.complete && this.plan.naturalWidth) {
      const a = this._toCanvas(this.planBounds.min[0], this.planBounds.min[1]);
      const c = this._toCanvas(this.planBounds.max[0], this.planBounds.max[1]);
      ctx.drawImage(this.plan, a[0], a[1], c[0] - a[0], c[1] - a[1]);
    }

    // walkable path (click to travel along it)
    if (this.path.length) {
      ctx.strokeStyle = 'rgba(86,182,255,0.9)'; ctx.lineWidth = 2;
      ctx.beginPath();
      this.path.forEach((wp, i) => { const p = this._toCanvas(wp.xy[0], wp.xy[1]); i ? ctx.lineTo(p[0], p[1]) : ctx.moveTo(p[0], p[1]); });
      ctx.stroke();
      for (const wp of this.path) {
        const p = this._toCanvas(wp.xy[0], wp.xy[1]);
        ctx.beginPath(); ctx.arc(p[0], p[1], 2.5, 0, 7); ctx.fillStyle = '#56b6ff'; ctx.fill();
      }
    }

    // live camera position + heading
    const dir = new THREE.Vector3(); camera.getWorldDirection(dir);
    const [pu, pw] = this._proj(camera.position);
    const [du, dw] = this._proj(dir);
    const p = this._toCanvas(pu, pw);
    const ang = Math.atan2(dw, du);
    ctx.strokeStyle = '#3fb950'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(p[0], p[1]); ctx.lineTo(p[0] + Math.cos(ang) * 16, p[1] + Math.sin(ang) * 16); ctx.stroke();
    ctx.fillStyle = '#3fb950'; ctx.beginPath(); ctx.arc(p[0], p[1], 4.5, 0, 7); ctx.fill();
  }
}
