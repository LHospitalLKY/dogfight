/* Dogfight Web Observer: three.js 3D 场景 + HTML/Canvas HUD 覆盖层。
 * 服务端快照包含世界坐标 (x 东, y 北, z 上)；three.js 使用 (X=x, Y=z, Z=-y)。
 */

(() => {
  "use strict";

  // 声明式写在最前面：下面的 RGB/C 常量表在初始化时就要调用 clamp，
  // 若用 const 定义会落进暂时性死区（ReferenceError: Cannot access
  // 'clamp' before initialization），整个脚本在构建颜色表时就中断。
  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  // ---------- theme.py 颜色常量（前端照搬） ----------
  const RGB = {
    TEAM: [[255, 116, 96], [92, 178, 255]],
    TEXT: [238, 243, 250],
    SUB: [158, 170, 186],
    HUD_GREEN: [132, 236, 178],
    HUD_DIM: [74, 140, 106],
    HUD_MARK: [255, 214, 120],
    HUD_WARN: [255, 96, 84],
    ACCENT_RED: [255, 59, 59],
    AIR_GRID: [40, 43, 49],
    AIR_EDGE: [58, 62, 70],
    CLAY_LIT: [232, 234, 236],
    CLAY_AO: [110, 114, 120],
    CANOPY: [126, 196, 238],
    SCENE_BG: [16, 17, 20],
  };
  const C = {
    text: rgbHex(RGB.TEXT),
    sub: rgbHex(RGB.SUB),
    hudGreen: rgbHex(RGB.HUD_GREEN),
    hudDim: rgbHex(RGB.HUD_DIM),
    hudMark: rgbHex(RGB.HUD_MARK),
    hudWarn: rgbHex(RGB.HUD_WARN),
    team: RGB.TEAM.map(rgbHex),
    airGrid: rgbHex(RGB.AIR_GRID),
    airEdge: rgbHex(RGB.AIR_EDGE),
    hudDimRgb: "rgb(74,140,106)",
    greenRgb: "rgb(100,176,136)",
    ladderMajor: "rgb(104,190,146)",
    ladderMajorLabel: "rgb(92,168,126)",
    glassBg: "rgba(5,15,11,0.84)",
    valueBg: "rgba(8,22,16,0.925)",
  };
  const FONT = '"Fira Code", "SFMono-Regular", Consolas, monospace';
  const REF_LENGTH = 20.1;
  const TRAIL_FADE = 12.0;
  const TAU = Math.PI * 2;

  function rgbHex(rgb) {
    return "#" + rgb.map((v) => Math.round(clamp(v, 0, 255)).toString(16).padStart(2, "0")).join("");
  }

  function mix(a, b, t) {
    return [
      a[0] + (b[0] - a[0]) * t,
      a[1] + (b[1] - a[1]) * t,
      a[2] + (b[2] - a[2]) * t,
    ];
  }

  function rgba(rgb, alpha) {
    return `rgba(${Math.round(rgb[0])},${Math.round(rgb[1])},${Math.round(rgb[2])},${alpha})`;
  }

  const w2t = (p) => new THREE.Vector3(p[0], p[2], -p[1]);

  // ---------- three.js 场景 ----------
  const canvas = document.getElementById("scene");
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: false,
    powerPreference: "high-performance",
  });
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  renderer.setSize(window.innerWidth, window.innerHeight, false);

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(rgbHex(RGB.SCENE_BG));

  const camera = new THREE.PerspectiveCamera(
    50, window.innerWidth / window.innerHeight, 1, 1e7
  );

  const ambient = new THREE.AmbientLight(0xffffff, 0.50);
  scene.add(ambient);
  const keyLight = new THREE.DirectionalLight(0xdfe8ff, 1.05);
  keyLight.position.copy(w2t([-0.35, -0.55, 0.76]).multiplyScalar(1e6));
  scene.add(keyLight);
  const fillLight = new THREE.DirectionalLight(0xffd8c0, 0.32);
  fillLight.position.copy(w2t([0.42, 0.36, -0.80]).multiplyScalar(1e6));
  scene.add(fillLight);

  // 地面：8 km 一格的极淡网格 + 空域边界。
  const floorGroup = new THREE.Group();
  scene.add(floorGroup);
  let floorArena = 0;
  let floorZ = 0;

  function buildFloor(arena, zFloor) {
    if (arena === floorArena && zFloor === floorZ) return;
    floorArena = arena;
    floorZ = zFloor;
    while (floorGroup.children.length) {
      const o = floorGroup.children.pop();
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    }
    const half = arena / 2.0;
    const step = 8000.0;
    const gridPos = [];
    for (let x = -half; x <= half + 1e-6; x += step) {
      gridPos.push(x, zFloor, -half, x, zFloor, half);
    }
    for (let y = -half; y <= half + 1e-6; y += step) {
      gridPos.push(-half, zFloor, -y, half, zFloor, -y);
    }
    const gridGeo = new THREE.BufferGeometry();
    gridGeo.setAttribute("position", new THREE.Float32BufferAttribute(gridPos, 3));
    const gridMat = new THREE.LineBasicMaterial({
      color: C.airGrid, transparent: true, opacity: 0.95,
    });
    floorGroup.add(new THREE.LineSegments(gridGeo, gridMat));

    const edgePos = [
      -half, zFloor, -half, half, zFloor, -half,
      half, zFloor, -half, half, zFloor, half,
      half, zFloor, half, -half, zFloor, half,
      -half, zFloor, half, -half, zFloor, -half,
    ];
    const edgeGeo = new THREE.BufferGeometry();
    edgeGeo.setAttribute("position", new THREE.Float32BufferAttribute(edgePos, 3));
    floorGroup.add(new THREE.LineSegments(
      edgeGeo,
      new THREE.LineBasicMaterial({ color: C.airEdge, transparent: true, opacity: 0.9 })
    ));
  }

  // ---------- OBJ 解析（只处理 v / f / usemtl，足够本项目模型） ----------
  const MODEL_URLS = {
    "SU-57": "/api/models/su57.obj",
    "F-16V": "/api/models/f16.obj",
  };
  const MODEL_GEOS = {};
  const MODELS_READY = (async () => {
    const jobs = Object.entries(MODEL_URLS).map(async ([name, url]) => {
      const res = await fetch(url, { cache: "no-store" });
      const text = await res.text();
      MODEL_GEOS[name] = parseOBJ(text);
    });
    await Promise.all(jobs).catch((err) => console.error("模型加载失败", err));
  })();

  function parseOBJ(text) {
    const verts = [];
    const groups = {
      body: { pos: [], count: 0 },
      canopy: { pos: [], count: 0 },
    };
    let material = "body";
    const lines = text.split(/\r?\n/);
    for (const raw of lines) {
      if (raw.length < 2) continue;
      const kind = raw.slice(0, 2);
      if (kind === "v ") {
        const parts = raw.trim().split(/\s+/);
        verts.push([+parts[1], +parts[2], +parts[3]]);
      } else if (raw.startsWith("usemtl ")) {
        const name = raw.slice(7).trim();
        material = name === "canopy" ? "canopy" : "body";
      } else if (kind === "f ") {
        const idx = raw.trim().split(/\s+/).slice(1)
          .map((tok) => parseInt(tok.split("/")[0], 10) - 1);
        if (idx.length < 3) continue;
        const g = groups[material];
        for (let t = 1; t < idx.length - 1; t += 1) {
          for (const i of [idx[0], idx[t], idx[t + 1]]) {
            const p = verts[i];
            if (!p) continue;
            g.pos.push(p[0], p[1], p[2]);
            g.count += 1;
          }
        }
      }
    }
    const out = {};
    for (const key of ["body", "canopy"]) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(groups[key].pos, 3));
      geo.computeVertexNormals();
      geo.computeBoundingSphere();
      out[key] = geo;
    }
    return out;
  }

  const bodyMats = RGB.TEAM.map((team) => new THREE.MeshStandardMaterial({
    color: new THREE.Color(rgbHex(mix(RGB.CLAY_LIT, team, 0.10))),
    roughness: 0.70,
    metalness: 0.05,
    side: THREE.DoubleSide,
  }));
  const canopyMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(rgbHex(RGB.CANOPY)),
    roughness: 0.25,
    metalness: 0.15,
    transparent: true,
    opacity: 0.55,
    depthWrite: false,
    side: THREE.DoubleSide,
  });

  // 尾焰：单位锥体，绕 X 轴转成沿 +Z，运行时按机尾方向缩放。
  const coneOuterGeo = new THREE.ConeGeometry(0.22, 1.0, 8, 1, false);
  coneOuterGeo.rotateX(Math.PI / 2);
  const coneInnerGeo = new THREE.ConeGeometry(0.12, 1.0, 8, 1, false);
  coneInnerGeo.rotateX(Math.PI / 2);

  function bodyQuat(psi, theta, phi) {
    const sp = Math.sin(psi), cp = Math.cos(psi);
    const st = Math.sin(theta), ct = Math.cos(theta);
    const sf = Math.sin(phi), cf = Math.cos(phi);
    const xb = [ct * cp, ct * sp, st];
    const yb = [cf * sp + sf * st * cp, -cf * cp + sf * st * sp, -sf * ct];
    const zb = [-sf * sp + cf * st * cp, sf * cp + cf * st * sp, -cf * ct];
    // three.js 局部轴：X=机体 y，Y=-机体 z，Z=-机体 x（机头 -Z）。
    const vx = new THREE.Vector3(yb[0], yb[2], -yb[1]);
    const vy = new THREE.Vector3(-zb[0], -zb[2], zb[1]);
    const vz = new THREE.Vector3(-xb[0], -xb[2], xb[1]);
    const m = new THREE.Matrix4().makeBasis(vx, vy, vz);
    return new THREE.Quaternion().setFromRotationMatrix(m);
  }

  class AircraftView {
    constructor(team) {
      this.team = team;
      this.group = new THREE.Group();
      this.modelGroup = new THREE.Group();
      this.group.add(this.modelGroup);
      scene.add(this.group);
      this.name = "";
      this.bodyMesh = null;
      this.canopyMesh = null;
      this.flames = [];
      this.engineCount = -1;

      // 高度投影线与地面圈（不随姿态旋转）。
      this.groundGeo = new THREE.BufferGeometry();
      this.groundGeo.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(6), 3));
      this.groundLine = new THREE.Line(
        this.groundGeo,
        new THREE.LineBasicMaterial({ color: new THREE.Color(C.team[team]), transparent: true, opacity: 0.45 })
      );
      scene.add(this.groundLine);
      const ringPos = [];
      for (let k = 0; k <= 24; k += 1) {
        const a = TAU * k / 24.0;
        ringPos.push(Math.cos(a) * 2600.0, 0, Math.sin(a) * 2600.0);
      }
      this.ringGeo = new THREE.BufferGeometry();
      this.ringGeo.setAttribute("position", new THREE.Float32BufferAttribute(ringPos, 3));
      this.ring = new THREE.LineLoop(
        this.ringGeo,
        new THREE.LineBasicMaterial({ color: new THREE.Color(C.team[team]), transparent: true, opacity: 0.45 })
      );
      scene.add(this.ring);

      // 机炮包线：机体所在高度的两个水平圆（半径 gun_range 与 0.5×），
      // 与 render._draw_gun_envelopes 的画法一致，只受高度影响、不随机体滚转。
      const envPos = [];
      for (let k = 0; k < 96; k += 1) {
        const a = TAU * k / 96.0;
        envPos.push(Math.cos(a) * 1.0, 0, Math.sin(a) * 1.0);
      }
      this.envGeo = new THREE.BufferGeometry();
      this.envGeo.setAttribute("position", new THREE.Float32BufferAttribute(envPos, 3));
      this.envOuter = new THREE.LineLoop(this.envGeo, new THREE.LineBasicMaterial({
        color: new THREE.Color(C.team[team]), transparent: true, opacity: 0.16,
      }));
      this.envInner = new THREE.LineLoop(this.envGeo, new THREE.LineBasicMaterial({
        color: new THREE.Color(C.team[team]), transparent: true, opacity: 0.09,
      }));
      this.envOuter.frustumCulled = false;
      this.envInner.frustumCulled = false;
      scene.add(this.envOuter, this.envInner);
    }

    setModel(name, engines) {
      if (this.name === name && this.engineCount === engines.length) return;
      this.name = name;
      this.engineCount = engines.length;
      this.modelGroup.clear();
      this.bodyMesh = null;
      this.canopyMesh = null;
      this.flames = [];
      const geo = MODEL_GEOS[name];
      if (geo) {
        this.bodyMesh = new THREE.Mesh(geo.body, bodyMats[this.team]);
        this.canopyMesh = new THREE.Mesh(geo.canopy, canopyMat);
        this.modelGroup.add(this.bodyMesh, this.canopyMesh);
      }
      for (const eng of engines) {
        const outer = new THREE.Mesh(coneOuterGeo, new THREE.MeshBasicMaterial({
          color: 0xe6e8ec, transparent: true, opacity: 0.52,
          blending: THREE.AdditiveBlending, depthWrite: false,
        }));
        const inner = new THREE.Mesh(coneInnerGeo, new THREE.MeshBasicMaterial({
          color: 0xffece8, transparent: true, opacity: 0.72,
          blending: THREE.AdditiveBlending, depthWrite: false,
        }));
        outer.position.set(eng[0], eng[1], eng[2]);
        inner.position.set(eng[0], eng[1], eng[2]);
        this.modelGroup.add(outer, inner);
        this.flames.push({ outer, inner, base: eng });
      }
    }

    update(ac, cameraDepth) {
      this.group.position.copy(w2t(ac.pos));
      this.group.quaternion.copy(bodyQuat(ac.psi, ac.theta, ac.phi));
      const focal = (window.innerHeight / 2.0) / Math.tan((50.0 * Math.PI / 180.0) / 2.0);
      const targetPx = 46.0 * clamp(8000.0 / cameraDepth, 2.0, 3.0);
      const k = targetPx * cameraDepth / (focal * REF_LENGTH);
      this.group.scale.setScalar(k);
      this.group.visible = !!ac.alive;

      const level = ac.alive ? clamp(ac.flame, 0, 1) : 0;
      const len = ac.model_len * (0.06 + 0.20 * level);
      for (const f of this.flames) {
        f.outer.visible = ac.alive && level > 0.01;
        f.inner.visible = f.outer.visible;
        f.outer.scale.setScalar(len);
        f.inner.scale.setScalar(len * 0.55);
        f.outer.position.set(f.base[0], f.base[1], f.base[2] + len * 0.5);
        f.inner.position.set(f.base[0], f.base[1], f.base[2] + len * 0.275);
      }

      const floorY = snapshot ? snapshot.z_floor : 0;
      const pos = this.group.position;
      const gp = this.groundGeo.attributes.position.array;
      gp[0] = pos.x; gp[1] = pos.y; gp[2] = pos.z;
      gp[3] = pos.x; gp[4] = floorY; gp[5] = pos.z;
      this.groundGeo.attributes.position.needsUpdate = true;
      this.ring.position.set(pos.x, floorY, pos.z);
      this.groundLine.visible = this.ring.visible = !!ac.alive;

      const gunRange = ac.gun_range || 0.0;
      this.envOuter.position.set(pos.x, pos.y, pos.z);
      this.envInner.position.copy(this.envOuter.position);
      this.envOuter.scale.setScalar(gunRange);
      this.envInner.scale.setScalar(gunRange * 0.5);
      this.envOuter.visible = this.envInner.visible = !!ac.alive && gunRange > 0.0;
    }
  }

  const aircraftViews = [new AircraftView(0), new AircraftView(1)];

  // ---------- 航迹 ----------
  const trailViews = [0, 1].map((team) => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(0), 3));
    geo.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(0), 3));
    const mat = new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.92,
    });
    const line = new THREE.Line(geo, mat);
    line.frustumCulled = false;
    scene.add(line);

    // 航迹方向箭头：每 ARROW_EVERY 个采样点一个人字形箭标（与 render 的
    // ARROW_EVERY = 25 一致），单独一个 LineSegments 承载。
    const ageo = new THREE.BufferGeometry();
    ageo.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(0), 3));
    ageo.setAttribute("color", new THREE.Float32BufferAttribute(new Float32Array(0), 3));
    const arrowLine = new THREE.LineSegments(ageo, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.9,
    }));
    arrowLine.frustumCulled = false;
    scene.add(arrowLine);
    return {
      geo, line, team, count: 0,
      arrowGeo: ageo, arrowLine, arrowCount: 0,
    };
  });
  const ARROW_EVERY = 25;

  function updateTrails(snap) {
    for (let i = 0; i < 2; i += 1) {
      const src = snap.trails[i] || { points: [] };
      const pts = src.points || [];
      const view = trailViews[i];
      if (pts.length < 2) {
        view.line.visible = false;
        view.arrowLine.visible = false;
        continue;
      }
      view.line.visible = true;
      view.arrowLine.visible = true;
      if (view.count !== pts.length) {
        const pos = new Float32Array(pts.length * 3);
        const col = new Float32Array(pts.length * 3);
        view.geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
        view.geo.setAttribute("color", new THREE.BufferAttribute(col, 3));
        view.count = pts.length;
      }
      const pos = view.geo.attributes.position.array;
      const col = view.geo.attributes.color.array;
      const base = mix([198, 202, 208], RGB.TEAM[i], 0.25);
      const now = snap.t;
      for (let j = 0; j < pts.length; j += 1) {
        const p = pts[j];
        pos[j * 3] = p[0];
        pos[j * 3 + 1] = p[2];
        pos[j * 3 + 2] = -p[1];
        const age = now - (p[3] || now);
        const fade = Math.max(0.0, 1.0 - age / TRAIL_FADE);
        const f = 0.38 + 0.62 * fade;
        col[j * 3] = (base[0] / 255.0) * f;
        col[j * 3 + 1] = (base[1] / 255.0) * f;
        col[j * 3 + 2] = (base[2] / 255.0) * f;
      }
      view.geo.attributes.position.needsUpdate = true;
      view.geo.attributes.color.needsUpdate = true;

      // 方向箭头：与 render 一致，每 ARROW_EVERY 个采样点放一个人字形箭标，
      // 朝向沿该段航迹，尺寸按相机距离换算成约 12 px，远近观感稳定。
      const arrowBase = mix([236, 238, 242], RGB.TEAM[i], 0.18);
      const focal = (window.innerHeight / 2.0) / Math.tan((50.0 * Math.PI / 180.0) / 2.0);
      const camDist = (snap.camera && snap.camera.dist) || 6000.0;
      const size = clamp(12.0 * camDist / focal, 18.0, 420.0);
      const arrowPos = [];
      const arrowCol = [];
      for (let j = ARROW_EVERY; j < pts.length; j += ARROW_EVERY) {
        const A = w2t(pts[j - 1]);
        const B = w2t(pts[j]);
        const d = B.clone().sub(A);
        if (d.lengthSq() < 1e-9) continue;
        d.normalize();
        const side = new THREE.Vector3(d.z, 0, -d.x);
        if (side.lengthSq() < 1e-6) side.set(1, 0, 0);
        side.normalize();
        const tail = B.clone().addScaledVector(d, -size);
        const b1 = tail.clone().addScaledVector(side, size * 0.48);
        const b2 = tail.clone().addScaledVector(side, -size * 0.48);
        const age = now - (pts[j][3] || now);
        const fade = Math.max(0.0, 1.0 - age / TRAIL_FADE);
        const f = 0.38 + 0.62 * fade;
        const r = (arrowBase[0] / 255.0) * f;
        const g = (arrowBase[1] / 255.0) * f;
        const b = (arrowBase[2] / 255.0) * f;
        for (const P of [B, b1, B, b2]) {
          arrowPos.push(P.x, P.y, P.z);
          arrowCol.push(r, g, b);
        }
      }
      const nArrow = arrowPos.length / 3;
      if (view.arrowCount !== nArrow) {
        view.arrowGeo.setAttribute(
          "position", new THREE.BufferAttribute(new Float32Array(nArrow * 3), 3)
        );
        view.arrowGeo.setAttribute(
          "color", new THREE.BufferAttribute(new Float32Array(nArrow * 3), 3)
        );
        view.arrowCount = nArrow;
      }
      if (nArrow > 0) {
        view.arrowGeo.attributes.position.array.set(arrowPos);
        view.arrowGeo.attributes.color.array.set(arrowCol);
        view.arrowGeo.attributes.position.needsUpdate = true;
        view.arrowGeo.attributes.color.needsUpdate = true;
      }
    }
  }

  // ---------- 导弹 / 预测线 / 炮线 / 事件标记 / 干扰物 ----------
  const missileGroup = new THREE.Group();
  scene.add(missileGroup);
  const missileDotGeo = new THREE.SphereGeometry(1, 6, 6);

  const predictionGroup = new THREE.Group();
  scene.add(predictionGroup);
  const predEndGeo = new THREE.SphereGeometry(1, 6, 6);

  const gunfireGroup = new THREE.Group();
  scene.add(gunfireGroup);

  function makeSpriteTexture(draw, size = 64) {
    const cv = document.createElement("canvas");
    cv.width = cv.height = size;
    const ctx = cv.getContext("2d");
    draw(ctx, size);
    const tex = new THREE.CanvasTexture(cv);
    tex.needsUpdate = true;
    return tex;
  }

  const texFire = makeSpriteTexture((ctx, s) => {
    ctx.translate(s / 2, s / 2);
    ctx.beginPath();
    ctx.moveTo(0, -s * 0.22); ctx.lineTo(s * 0.22, 0);
    ctx.lineTo(0, s * 0.22); ctx.lineTo(-s * 0.22, 0);
    ctx.closePath();
    ctx.fillStyle = "#f2f2f4"; ctx.fill();
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 2; ctx.stroke();
  });

  const texStar = makeSpriteTexture((ctx, s) => {
    ctx.translate(s / 2, s / 2);
    ctx.beginPath();
    for (let k = 0; k < 10; k += 1) {
      const a = -Math.PI / 2 + k * Math.PI / 5;
      const r = k % 2 === 0 ? s * 0.30 : s * 0.135;
      ctx.lineTo(Math.cos(a) * r, Math.sin(a) * r);
    }
    ctx.closePath();
    ctx.fillStyle = "#f3efe6"; ctx.fill();
    ctx.strokeStyle = "#8b9096"; ctx.lineWidth = 2; ctx.stroke();
  });

  const texHighG = makeSpriteTexture((ctx, s) => {
    ctx.translate(s / 2, s / 2);
    ctx.beginPath();
    ctx.moveTo(0, -s * 0.28); ctx.lineTo(s * 0.28, 0);
    ctx.lineTo(0, s * 0.28); ctx.lineTo(-s * 0.28, 0);
    ctx.closePath();
    ctx.fillStyle = "#b9bec5"; ctx.fill();
    ctx.strokeStyle = "#6e7278"; ctx.lineWidth = 2; ctx.stroke();
  });

  const texRing = makeSpriteTexture((ctx, s) => {
    ctx.translate(s / 2, s / 2);
    ctx.beginPath();
    ctx.arc(0, 0, s * 0.28, 0, TAU);
    ctx.strokeStyle = "#ffffff"; ctx.lineWidth = 3; ctx.stroke();
    ctx.beginPath();
    ctx.arc(0, 0, s * 0.075, 0, TAU);
    ctx.fillStyle = "#ffffff"; ctx.fill();
  });

  function makePoints(texture, size, teamColor) {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(new Float32Array(0), 3));
    const mat = new THREE.PointsMaterial({
      size, sizeAttenuation: false, map: texture,
      transparent: true, alphaTest: 0.05, depthWrite: false,
      color: teamColor || 0xffffff,
    });
    const points = new THREE.Points(geo, mat);
    points.frustumCulled = false;
    return points;
  }

  const firePoints = [makePoints(texFire, 11), makePoints(texFire, 11)];
  scene.add(firePoints[0], firePoints[1]);
  const launchPoints = makePoints(texStar, 16);
  scene.add(launchPoints);
  const highgPoints = makePoints(texHighG, 16);
  scene.add(highgPoints);
  const predEndPoints = [makePoints(texRing, 15), makePoints(texRing, 15)];
  scene.add(predEndPoints[0], predEndPoints[1]);

  const chaffGeo = new THREE.BufferGeometry();
  const chaffPoints = new THREE.Points(chaffGeo, new THREE.PointsMaterial({
    size: 6, sizeAttenuation: false, color: 0xced0d6,
    transparent: true, opacity: 0.9, depthWrite: false,
  }));
  const flareGeo = new THREE.BufferGeometry();
  const flarePoints = new THREE.Points(flareGeo, new THREE.PointsMaterial({
    size: 8, sizeAttenuation: false, color: 0xffaa46,
    transparent: true, opacity: 0.95, depthWrite: false,
  }));
  scene.add(chaffPoints, flarePoints);

  function setPointsPositions(points, arr) {
    const geo = points.geometry;
    geo.setAttribute("position", new THREE.Float32BufferAttribute(arr, 3));
    geo.attributes.position.needsUpdate = true;
    points.visible = arr.length > 0;
  }

  function updateMissiles(snap) {
    while (missileGroup.children.length) {
      const o = missileGroup.children.pop();
      if (o.geometry && o.geometry !== missileDotGeo) o.geometry.dispose();
      if (o.material) o.material.dispose();
    }
    for (const m of snap.missiles || []) {
      const p = w2t(m.pos);
      const d = new THREE.Vector3(m.dir[0], m.dir[2], -m.dir[1]);
      if (d.lengthSq() < 1e-8) d.set(0, 0, 1);
      d.normalize();
      const color = new THREE.Color(rgbHex(mix(RGB.CLAY_LIT, RGB.TEAM[m.owner], 0.25)));
      if (!m.guided) color.set(0x969696);
      const len = clamp(m.length * 1.4, 2.0, 8.0);
      const mesh = new THREE.Mesh(
        new THREE.CylinderGeometry(0.10, 0.10, len, 5),
        new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.95 })
      );
      mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d);
      mesh.position.copy(p).addScaledVector(d, -len * 0.5);
      missileGroup.add(mesh);
      const dot = new THREE.Mesh(missileDotGeo, new THREE.MeshBasicMaterial({ color: 0xfafafc }));
      dot.scale.setScalar(0.18);
      dot.position.copy(p);
      missileGroup.add(dot);
      if (m.trail && m.trail.length > 1) {
        const g = new THREE.BufferGeometry();
        const arr = [];
        for (const tp of m.trail) {
          const v = w2t(tp);
          arr.push(v.x, v.y, v.z);
        }
        g.setAttribute("position", new THREE.Float32BufferAttribute(arr, 3));
        missileGroup.add(new THREE.Line(g, new THREE.LineBasicMaterial({
          color: 0x84888e, transparent: true, opacity: 0.58,
        })));
      }
      if (m.ignited && m.fuel > 0.0) {
        const tip = p.clone().addScaledVector(d, -len * 1.6);
        const g = new THREE.BufferGeometry().setFromPoints([p.clone().addScaledVector(d, -len), tip]);
        missileGroup.add(new THREE.Line(g, new THREE.LineBasicMaterial({
          color: 0xeceef2, transparent: true, opacity: 0.75,
        })));
      }
    }
  }

  function updatePredictions(snap) {
    while (predictionGroup.children.length) {
      const o = predictionGroup.children.pop();
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    }
    for (let i = 0; i < 2; i += 1) {
      const pts = (snap.predictions || [])[i] || [];
      const ends = [];
      if (pts.length < 2) {
        setPointsPositions(predEndPoints[i], []);
        continue;
      }
      const vidx = [];
      for (const p of pts) {
        const v = w2t(p);
        vidx.push(v.x, v.y, v.z);
      }
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(vidx, 3));
      predictionGroup.add(new THREE.Line(g, new THREE.LineBasicMaterial({
        color: new THREE.Color(rgbHex(mix(RGB.TEAM[i], RGB.SCENE_BG, 0.35))),
        transparent: true, opacity: 0.8,
      })));
      const end = w2t(pts[pts.length - 1]);
      ends.push(end.x, end.y, end.z);
      predEndPoints[i].material.color.set(C.team[i]);
      setPointsPositions(predEndPoints[i], ends);
    }
  }

  function updateGunfire(snap) {
    while (gunfireGroup.children.length) {
      const o = gunfireGroup.children.pop();
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    }
    const lines = Array.isArray(snap.gunfire)
      ? snap.gunfire
      : (snap.gunfire && snap.gunfire.lines) || [];
    for (const g of lines) {
      if (!g) continue;
      const a = w2t(g.nose);
      const b = w2t(g.end);
      const geo = new THREE.BufferGeometry().setFromPoints([a, b]);
      gunfireGroup.add(new THREE.Line(geo, new THREE.LineBasicMaterial({
        color: 0xff9a36, transparent: true, opacity: 0.9,
      })));
      const glow = new THREE.Line(geo.clone(), new THREE.LineBasicMaterial({
        color: 0xffecaa, transparent: true, opacity: 0.55,
      }));
      gunfireGroup.add(glow);
      const flash = makePoints(texFire, 20, 0xffd678);
      setPointsPositions(flash, [a.x, a.y, a.z]);
      gunfireGroup.add(flash);
    }
  }

  function updateMarks(snap) {
    const marks = snap.marks || {};
    for (let i = 0; i < 2; i += 1) {
      const arr = [];
      for (const p of marks.fire?.[i] || []) {
        const v = w2t(p);
        arr.push(v.x, v.y, v.z);
      }
      setPointsPositions(firePoints[i], arr);
    }
    const launch = [];
    for (const m of marks.launch || []) {
      const v = w2t(m.pos);
      launch.push(v.x, v.y, v.z);
    }
    setPointsPositions(launchPoints, launch);
    const highg = [];
    for (const m of marks.highg || []) {
      const v = w2t(m.pos);
      highg.push(v.x, v.y, v.z);
    }
    setPointsPositions(highgPoints, highg);

    const chaff = [];
    const flare = [];
    for (const cm of snap.counters || []) {
      const v = w2t(cm.pos);
      if (cm.kind === "chaff") chaff.push(v.x, v.y, v.z);
      else flare.push(v.x, v.y, v.z);
    }
    setPointsPositions(chaffPoints, chaff);
    setPointsPositions(flarePoints, flare);
  }

  // ---------- 姿态框、座舱 HUD 布局 ----------
  const STATION_MARGIN = 20;
  const STATION_TOP = 20;
  const ATT_W = 440;
  const WEAPON_H = 108;
  const HUD_STATUS_H = 24;
  const FS_TINY = 14;
  const FS_SMALL = 18;
  const FS_BODY = 22;

  const stationsEl = document.getElementById("stations");
  const attitudeEl = document.getElementById("attitude-panel");
  const missileStatusEl = document.getElementById("missile-status");
  const bottomBarEl = document.getElementById("bottom-bar");
  const bottomHintEl = document.getElementById("bottom-hint");
  const pausedEl = document.getElementById("paused-label");
  const resultEl = document.getElementById("result-overlay");

  function stationMetrics(w, h) {
    const avail = Math.max(320, Math.floor((w - ATT_W - 2 * STATION_MARGIN - 16) / 2));
    const weaponH = WEAPON_H;
    const statusH = HUD_STATUS_H;
    if (avail >= 520) {
      const clusterW = Math.min(avail, 760);
      const hudW = Math.min(456, Math.floor(clusterW * 0.58));
      const radarW = clusterW - hudW - 8;
      const hudH = Math.floor(clamp(hudW * 0.60, 168, 280)) + statusH;
      const radarH = Math.max(radarW, hudH + 8 + weaponH);
      return { clusterW, radarW, radarH, hudW, hudH, weaponH, stacked: false };
    }
    const radarW = avail;
    const radarH = Math.floor(clamp(Math.min(avail * 0.58, h * 0.30), 190, 240));
    const hudH = Math.floor(clamp(avail * 0.58, 200, 300)) + statusH;
    return { clusterW: avail, radarW, radarH, hudW: avail, hudH, weaponH, stacked: true };
  }

  function stationRects(i, w, h) {
    const m = stationMetrics(w, h);
    const top = STATION_TOP;
    const x0 = i === 0 ? STATION_MARGIN : w - STATION_MARGIN - m.clusterW;
    let radar, hud, weapon;
    if (m.stacked) {
      radar = { x: x0, y: top, w: m.radarW, h: m.radarH };
      hud = { x: x0, y: radar.y + radar.h + 6, w: m.hudW, h: m.hudH };
      weapon = { x: x0, y: hud.y + hud.h + 6, w: m.hudW, h: m.weaponH };
    } else {
      const radarX = i === 0 ? x0 : x0 + m.clusterW - m.radarW;
      radar = { x: radarX, y: top, w: m.radarW, h: m.radarH };
      const hudX = i === 0 ? x0 + m.radarW + 8 : x0;
      hud = { x: hudX, y: top, w: m.hudW, h: m.hudH };
      weapon = { x: hudX, y: hud.y + hud.h + 8, w: m.hudW, h: m.weaponH };
    }
    return { radar, hud, weapon };
  }

  function setBox(el, r) {
    el.style.left = `${r.x}px`;
    el.style.top = `${r.y}px`;
    el.style.width = `${r.w}px`;
    el.style.height = `${r.h}px`;
  }

  function fitCanvas(cv, w, h) {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const bw = Math.max(1, Math.round(w * dpr));
    const bh = Math.max(1, Math.round(h * dpr));
    if (cv.width !== bw || cv.height !== bh) {
      cv.width = bw;
      cv.height = bh;
    }
    cv._w = w;
    cv._h = h;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    return ctx;
  }

  function text(ctx, str, x, y, color, size = FS_SMALL) {
    ctx.font = `${size}px ${FONT}`;
    ctx.fillStyle = color;
    ctx.textBaseline = "top";
    ctx.fillText(str, x, y);
  }

  // 顶部两个姿态框。
  const attitudeParts = [0, 1].map((i) => {
    const half = document.createElement("div");
    half.className = `att-half ${i === 0 ? "left" : "right"} ${i === 0 ? "team-red" : "team-blue"}`;
    const name = document.createElement("div");
    name.className = "att-name";
    const cv = document.createElement("canvas");
    cv.className = "att-canvas";
    const rowThr = document.createElement("div");
    rowThr.className = "att-bar-row thr";
    rowThr.innerHTML = '<span class="att-label">THR</span><span class="att-bar thr"><i></i></span><span class="att-value"></span>';
    const rowFuel = document.createElement("div");
    rowFuel.className = "att-bar-row fuel";
    rowFuel.innerHTML = '<span class="att-label">FUEL</span><span class="att-bar fuel"><i></i></span><span class="att-value"></span>';
    half.append(name, cv, rowThr, rowFuel);
    attitudeEl.appendChild(half);
    return {
      half, name, cv,
      thrBar: rowThr.querySelector(".att-bar"),
      thrFill: rowThr.querySelector("i"),
      thrValue: rowThr.querySelector(".att-value"),
      fuelFill: rowFuel.querySelector("i"),
      fuelValue: rowFuel.querySelector(".att-value"),
    };
  });
  {
    const divider = document.createElement("div");
    divider.className = "divider";
    attitudeEl.appendChild(divider);
  }

  // 座舱三块面板。
  function makeStationPanel(teamIdx, cls) {
    const el = document.createElement("div");
    el.className = `station ${cls}`;
    el.style.setProperty("--team", C.team[teamIdx]);
    el.style.setProperty("--team-rgb", RGB.TEAM[teamIdx].join(","));
    stationsEl.appendChild(el);
    return el;
  }

  class StationView {
    constructor(i) {
      this.i = i;
      this.radar = makeStationPanel(i, "radar");
      this.radarTitle = document.createElement("div");
      this.radarTitle.className = "panel-title";
      this.radarState = document.createElement("div");
      this.radarState.className = "panel-state";
      this.radarHead = document.createElement("div");
      this.radarHead.className = "panel-headline";
      this.lockWrap = document.createElement("div");
      this.lockWrap.className = "lock-progress";
      this.lockBar = document.createElement("i");
      this.lockWrap.appendChild(this.lockBar);
      this.scopeCanvas = document.createElement("canvas");
      this.scopeCanvas.className = "scope";
      this.radar.append(this.radarTitle, this.radarState, this.radarHead, this.lockWrap, this.scopeCanvas);

      this.hud = makeStationPanel(i, "hud-panel");
      this.hudCanvas = document.createElement("canvas");
      this.hudCanvas.className = "hud-canvas";
      this.hud.appendChild(this.hudCanvas);

      this.weapon = makeStationPanel(i, "weapon");
      this.wpnTitle = div("wpn-title", "WPN");
      this.wpnStatus = div("wpn-status", "");
      this.wpnSeek = div("wpn-seek", "");
      this.silCanvas = document.createElement("canvas");
      this.silCanvas.className = "silhouette";
      this.wpnLeft = div("wpn-left", "");
      this.wpnRight = div("wpn-right", "");
      this.wpnFootL = div("wpn-foot left", "");
      this.wpnFootR = div("wpn-foot right", "");
      this.weapon.append(this.wpnTitle, this.wpnStatus, this.wpnSeek, this.silCanvas,
                        this.wpnLeft, this.wpnRight, this.wpnFootL, this.wpnFootR);
      this.groupHeadL = div("wpn-group-head", "AAM-R");
      this.groupNameL = div("wpn-group-name", "");
      this.groupHeadR = div("wpn-group-head", "AAM-IR");
      this.groupNameR = div("wpn-group-name", "");
      this.groupHeadL.style.position = "absolute";
      this.groupNameL.style.position = "absolute";
      this.groupHeadR.style.position = "absolute";
      this.groupNameR.style.position = "absolute";
      this.wpnLeft.append(this.groupHeadL, this.groupNameL);
      this.wpnRight.append(this.groupHeadR, this.groupNameR);
      this.rects = null;
      this.lastModel = "";
    }

    setRects(rs) {
      this.rects = rs;
      setBox(this.radar, rs.radar);
      setBox(this.hud, rs.hud);
      setBox(this.weapon, rs.weapon);
      const gap = clamp(rs.weapon.h - 84, 22, 30);
      this.groupHeadL.style.left = "0px";
      this.groupNameL.style.left = "0px";
      this.groupHeadR.style.right = "0px";
      this.groupNameR.style.right = "0px";
      this.groupHeadL.style.top = "22px";
      this.groupNameL.style.top = "36px";
      this.groupHeadR.style.top = `${22 + gap}px`;
      this.groupNameR.style.top = `${36 + gap}px`;
      this.wpnLeft.style.top = "0px";
      this.wpnRight.style.top = "0px";
    }

    update(ac, snap) {
      if (!this.rects) return;
      this.radarTitle.textContent = `${ac.name} ${ac.radar_on ? "RDR ON" : "RDR SIL"}`;
      this.radarState.textContent = ac.hud.lock_text;
      this.radarState.style.color = ac.hud.lock_color;
      if (ac.radar_lock_timer > 0.0 && !ac.radar_locked && ac.radar_lock_time > 0) {
        this.lockWrap.style.display = "block";
        this.lockBar.style.width = `${clamp(ac.radar_lock_timer / ac.radar_lock_time, 0, 1) * 100}%`;
        this.lockBar.style.background = ac.hud.lock_color;
      } else {
        this.lockWrap.style.display = "none";
      }
      drawScope(this.scopeCanvas, ac, this.rects.radar);
      drawFwdHud(this.hudCanvas, ac, this.rects.hud);
      this.updateWeapon(ac, snap);
    }

    updateWeapon(ac, snap) {
      const wp = ac.weapon;
      this.wpnStatus.textContent = wp.status;
      this.wpnStatus.style.color = wp.status_color;
      this.wpnSeek.textContent = wp.seek;
      this.wpnSeek.style.color = wp.seek_color;
      const r = wp.radar;
      const ir = wp.ir;
      this.groupHeadL.textContent = "AAM-R";
      this.groupNameL.textContent = r ? `${r.name} ${r.have}/${r.total}` : "-";
      this.groupNameL.className = `wpn-group-name${r && r.have === 0 ? " empty" : ""}`;
      this.groupHeadR.textContent = "AAM-IR";
      this.groupNameR.textContent = ir ? `${ir.name} ${ir.have}/${ir.total}` : "-";
      this.groupNameR.className = `wpn-group-name${ir && ir.have === 0 ? " empty" : ""}`;
      this.wpnFootL.textContent = `GUN ${wp.gun}`;
      this.wpnFootR.textContent = `CH ${wp.chaff}  FL ${wp.flares}`;
      const rw = this.rects.weapon;
      drawSilhouette(this.silCanvas, ac.name, ac.team, rw.w - 172, rw.h - 38);
    }
  }

  function div(cls, txt) {
    const d = document.createElement("div");
    d.className = cls;
    if (txt) d.textContent = txt;
    return d;
  }

  const stationViews = [new StationView(0), new StationView(1)];

  // ---------- Canvas HUD 绘制 ----------
  function drawScope(cv, ac, radarRect) {
    const w = Math.max(1, radarRect.w - 16);
    const h = Math.max(1, radarRect.h - 42);
    const ctx = fitCanvas(cv, w, h);
    const sc = ac.scope;
    ctx.fillStyle = "rgba(7,15,11,0.216)";
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = "rgb(34,88,58)";
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, w - 1, h - 1);

    const cx = w / 2.0;
    const cy = h / 2.0;
    const radius = Math.floor(Math.min(w, h) / 2) - 20;
    if (radius <= 2) return;

    for (const frac of sc.rings) {
      ctx.beginPath();
      ctx.arc(cx, cy, radius * frac, 0, TAU);
      ctx.strokeStyle = "rgb(34,88,58)";
      ctx.stroke();
    }
    const labels = sc.labels || [];
    for (let k = 0; k < labels.length; k += 1) {
      const frac = k === 0 ? 0.5 : 1.0;
      text(ctx, `${labels[k]}`, cx + 3, cy + radius * frac - 17, C.hudDim, FS_SMALL);
    }
    ctx.beginPath();
    ctx.moveTo(cx - radius, cy); ctx.lineTo(cx + radius, cy);
    ctx.moveTo(cx, cy - radius); ctx.lineTo(cx, cy + radius);
    ctx.strokeStyle = "rgb(34,88,58)";
    ctx.stroke();

    // 本机三角
    ctx.beginPath();
    ctx.moveTo(cx, cy - 8); ctx.lineTo(cx - 5, cy + 6); ctx.lineTo(cx + 5, cy + 6);
    ctx.closePath();
    ctx.fillStyle = C.team[ac.team];
    ctx.fill();

    const tgt = sc.target;
    if (sc.contact >= 2 && tgt && sc.foe_alive) {
      const rr = clamp(tgt.rr, 0, 0.96) * radius;
      const tx = cx + rr * Math.sin(tgt.az);
      const ty = cy - rr * Math.cos(tgt.az);
      ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(tx, ty);
      ctx.strokeStyle = sc.lock_color; ctx.lineWidth = 1; ctx.stroke();
      ctx.beginPath(); ctx.arc(tx, ty, 4, 0, TAU);
      ctx.strokeStyle = sc.lock_color; ctx.lineWidth = 1; ctx.stroke();
      ctx.beginPath(); ctx.arc(tx, ty, 2, 0, TAU);
      ctx.fillStyle = "rgb(7,15,11)"; ctx.fill();
      const tag = (tgt.d / 1000.0).toFixed(1).padStart(4, " ");
      const alt = `${(tgt.dz / 1000.0 >= 0 ? "+" : "")}${(tgt.dz / 1000.0).toFixed(1)}k`;
      ctx.font = `${FS_SMALL}px ${FONT}`;
      let lx = tx + 8;
      if (lx + ctx.measureText(tag).width > w - 4) {
        lx = tx - 8 - ctx.measureText(tag).width;
      }
      let ly = ty - 18;
      if (ly < 4) ly = ty + 6;
      text(ctx, tag, lx, ly, sc.lock_color, FS_SMALL);
      text(ctx, alt, lx, ly + 18, C.hudDim, FS_SMALL);
    } else if (sc.detect) {
      const az = sc.detect.az;
      const tcol = sc.detect.rwr ? "rgb(255,96,88)" : C.hudDim;
      for (const sgn of [-1.0, 1.0]) {
        const a = az + sgn * (4.0 * Math.PI / 180.0);
        ctx.beginPath();
        ctx.moveTo(cx + 14 * Math.sin(a), cy - 14 * Math.cos(a));
        ctx.lineTo(cx + radius * Math.sin(a), cy - radius * Math.cos(a));
        ctx.strokeStyle = tcol; ctx.stroke();
      }
      const tag = sc.detect.rwr ? "RWR" : "AZ";
      ctx.font = `${FS_SMALL}px ${FONT}`;
      const ex = cx + (radius - 2) * Math.sin(az);
      const ey = cy - (radius - 2) * Math.cos(az);
      text(ctx, tag, ex - ctx.measureText(tag).width / 2, ey - 18, tcol, FS_SMALL);
    } else {
      ctx.font = `${FS_SMALL}px ${FONT}`;
      const lab = "NO CONTACT";
      text(ctx, lab, cx - ctx.measureText(lab).width / 2, cy + radius - 4, C.hudDim, FS_SMALL);
    }

    let incomingSeen = false;
    for (const b of sc.blips || []) {
      const rr = clamp(b.rr, 0, 0.96) * radius;
      const bx = cx + rr * Math.sin(b.az);
      const by = cy - rr * Math.cos(b.az);
      const fx = Math.sin(b.hdg), fy = -Math.cos(b.hdg);
      const px = Math.cos(b.hdg), py = Math.sin(b.hdg);
      if (b.incoming) {
        incomingSeen = true;
        ctx.beginPath();
        ctx.moveTo(bx + 6 * fx, by + 6 * fy);
        ctx.lineTo(bx - 4 * fx + 3.5 * px, by - 4 * fy + 3.5 * py);
        ctx.lineTo(bx - 4 * fx - 3.5 * px, by - 4 * fy - 3.5 * py);
        ctx.closePath();
        ctx.fillStyle = "rgb(255,96,88)"; ctx.fill();
        text(ctx, (b.md / 1000).toFixed(1), bx + 8, by - 16, "rgb(255,96,88)", FS_SMALL);
      } else {
        ctx.beginPath(); ctx.arc(bx, by, 3, 0, TAU);
        ctx.strokeStyle = "rgb(116,232,150)"; ctx.stroke();
        ctx.beginPath(); ctx.moveTo(bx, by); ctx.lineTo(bx + 7 * fx, by + 7 * fy);
        ctx.strokeStyle = "rgb(116,232,150)"; ctx.stroke();
      }
    }
    if (sc.threat && !incomingSeen) {
      const az = sc.threat.az;
      for (const sgn of [-1.0, 1.0]) {
        const a = az + sgn * (3.0 * Math.PI / 180.0);
        ctx.beginPath();
        ctx.moveTo(cx + (radius - 9) * Math.sin(a), cy - (radius - 9) * Math.cos(a));
        ctx.lineTo(cx + radius * Math.sin(a), cy - radius * Math.cos(a));
        ctx.strokeStyle = "rgb(255,80,72)"; ctx.lineWidth = 2; ctx.stroke();
      }
    }
    if (sc.situation) {
      text(ctx, sc.situation, 3, h - 3 - FS_TINY, C.hudDim, FS_TINY);
    }
    if (sc.incoming) {
      ctx.font = `${FS_SMALL}px ${FONT}`;
      const lab = `IN ${sc.incoming}`;
      text(ctx, lab, w - 3 - ctx.measureText(lab).width, h - 3 - FS_SMALL,
           "rgb(255,96,88)", FS_SMALL);
    }
  }

  function drawValueBox(ctx, x, y, w, h, str, ticks) {
    ctx.font = `${FS_BODY}px ${FONT}`;
    const tw = ctx.measureText(str).width;
    const bw = Math.max(w, tw + 8);
    if (bw > w) x = x + w - bw;
    const ch = 2;
    ctx.beginPath();
    ctx.moveTo(x + ch, y);
    ctx.lineTo(x + bw, y);
    ctx.lineTo(x + bw, y + h - ch);
    ctx.lineTo(x + bw - ch, y + h);
    ctx.lineTo(x, y + h);
    ctx.lineTo(x, y + ch);
    ctx.closePath();
    ctx.fillStyle = C.valueBg;
    ctx.fill();
    ctx.strokeStyle = C.hudMark;
    ctx.lineWidth = 1;
    ctx.stroke();
    if (ticks) {
      const tx = 5;
      ctx.beginPath();
      ctx.moveTo(x + tx, y + h / 2 - 4); ctx.lineTo(x + tx, y + h / 2 + 4);
      ctx.moveTo(x + bw - tx, y + h / 2 - 4); ctx.lineTo(x + bw - tx, y + h / 2 + 4);
      ctx.strokeStyle = C.hudMark; ctx.stroke();
    }
    ctx.font = `${FS_BODY}px ${FONT}`;
    ctx.fillStyle = C.hudMark;
    ctx.textBaseline = "top";
    ctx.fillText(str, x + bw / 2 - tw / 2, y + h / 2 - FS_BODY / 2 - 1);
  }

  function drawFwdHud(cv, ac, hudRect) {
    const w = hudRect.w;
    const h = hudRect.h;
    const ctx = fitCanvas(cv, w, h);
    const hd = ac.hud;

    // 顶部状态带：HP 与 G（与 render._draw_hud_status 一致）。
    const hp = ac.alive ? Math.max(0, ac.hp) : 0;
    const hpFrac = ac.hp_frac;
    const hpCol = hpFrac < 0.25 ? C.hudWarn : C.hudGreen;
    ctx.font = `${FS_SMALL}px ${FONT}`;
    const hpLab = `HP ${String(Math.round(hp)).padStart(3, " ")}`;
    text(ctx, hpLab, 12, 3, hpCol, FS_SMALL);
    const labW = ctx.measureText(hpLab).width;
    const barX = 12 + labW + 8;
    const barY = 9;
    ctx.fillStyle = "rgb(26,40,34)";
    ctx.fillRect(barX, barY, 52, 5);
    if (hpFrac > 0) {
      ctx.fillStyle = hpCol;
      ctx.fillRect(barX, barY, Math.max(1, Math.round(52 * hpFrac)), 5);
    }
    const gLab = `G ${ac.g.toFixed(1).padStart(3, " ")}`;
    ctx.font = `${FS_SMALL}px ${FONT}`;
    text(ctx, gLab, w - 12 - ctx.measureText(gLab).width, 3, C.hudGreen, FS_SMALL);
    ctx.beginPath();
    ctx.moveTo(12, HUD_STATUS_H - 0.5); ctx.lineTo(w - 12, HUD_STATUS_H - 0.5);
    ctx.strokeStyle = "rgb(43,45,51)";
    ctx.lineWidth = 1; ctx.stroke();

    const fx = 5;
    const fy = HUD_STATUS_H;
    const fw = w - 10;
    const fh = h - HUD_STATUS_H - 6;
    if (fw < 40 || fh < 40) return;
    ctx.fillStyle = "rgba(14,30,44,0.376)";
    ctx.fillRect(fx, fy, fw, fh);

    const cx = fx + fw / 2.0;
    const cy = fy + fh / 2.0 + 3;
    const pxPerDeg = clamp(fh / 105.0, 1.7, 2.6);
    const half = fw * 0.5;
    const nom = Math.tan(26.0 * Math.PI / 180.0);
    const phi = hd.phi * Math.PI / 180.0;
    const dx = Math.cos(phi), dy = Math.sin(phi);
    const nx = -Math.sin(phi), ny = Math.cos(phi);
    const pitch = hd.theta;
    const ladderTop = fy + 54;
    const ladderBottom = fy + fh - 24;
    const ladderLeft = fx + 104;
    const ladderRight = fx + fw - 104;

    const hudPt = (az, below) => [
      cx + az * dx + below * pxPerDeg * nx,
      cy + az * dy + below * pxPerDeg * ny,
    ];
    const viewPt = (azDeg, elDeg) => hudPt(
      Math.tan(azDeg * Math.PI / 180.0) / nom * half * 0.82,
      pitch - elDeg
    );

    ctx.save();
    ctx.beginPath();
    ctx.rect(Math.max(0, ladderLeft), ladderTop,
             Math.max(1, ladderRight - ladderLeft), Math.max(1, ladderBottom - ladderTop));
    ctx.clip();
    const [hx, hy] = hudPt(0.0, pitch);
    const big = Math.max(fw, fh) * 3.0;
    ctx.beginPath();
    ctx.moveTo(hx - big * dx, hy - big * dy);
    ctx.lineTo(hx + big * dx, hy + big * dy);
    ctx.strokeStyle = C.hudGreen; ctx.lineWidth = 1; ctx.stroke();

    for (let p = -40; p <= 40; p += 5) {
      const [mx, my] = hudPt(0.0, pitch - p);
      if (!(my > ladderTop && my < ladderBottom)) continue;
      const major = p % 10 === 0;
      const ln = major ? 22.0 : 10.0;
      ctx.beginPath();
      ctx.moveTo(mx - ln * dx, my - ln * dy);
      ctx.lineTo(mx + ln * dx, my + ln * dy);
      ctx.strokeStyle = major ? C.ladderMajor : C.hudDim;
      ctx.lineWidth = 1; ctx.stroke();
      if (major && p !== 0) {
        if (Math.abs(ny) * pxPerDeg * 10.0 < 12.0) continue;
        const label = `${p > 0 ? "+" : ""}${p}`;
        ctx.font = `${FS_TINY}px ${FONT}`;
        const lw = ctx.measureText(label).width;
        const lh = FS_TINY;
        const labX = Math.round(mx - ln * dx) - 4 - lw;
        const labY = Math.round(my - ln * dy) - lh / 2;
        if (labY < ladderTop || labY + lh > ladderBottom
            || labX < ladderLeft + 2 || labX + lw > ladderRight - 2) continue;
        if (labX < cx + 48 && labX + lw > cx - 48
            && labY < cy + 12 && labY + lh > cy - 12) continue;
        text(ctx, label, labX, labY, C.ladderMajorLabel, FS_TINY);
      }
    }
    ctx.restore();

    // 固定水平标志
    ctx.strokeStyle = C.hudMark; ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(cx - 44, cy); ctx.lineTo(cx - 13, cy);
    ctx.moveTo(cx + 13, cy); ctx.lineTo(cx + 44, cy);
    ctx.moveTo(cx - 44, cy - 4); ctx.lineTo(cx - 44, cy + 4);
    ctx.moveTo(cx + 44, cy - 4); ctx.lineTo(cx + 44, cy + 4);
    ctx.stroke();

    // 速度 / 高度带
    ctx.save();
    ctx.beginPath();
    ctx.rect(fx + 2, ladderTop, Math.max(1, fw - 4), Math.max(1, ladderBottom - ladderTop));
    ctx.clip();
    drawSpeedTape(ctx, fx + 2, ladderTop, ladderBottom, hd.v, cy);
    drawAltitudeTape(ctx, fx + fw - 2, ladderTop, ladderBottom, hd.z, cy);
    ctx.restore();

    // 航向带
    drawHeadingTape(ctx, fx, fy, fw, fh, hd.heading);

    // 目标框 / 头瞄指引
    if (hd.foe_alive) {
      if (hd.contact >= 2) {
        if (Math.abs(hd.rel) < 52.0) {
          let [tx, ty] = viewPt(hd.rel, hd.elev);
          tx = clamp(tx, fx + 96, fx + fw - 120);
          ty = clamp(ty, fy + 56, fy + fh - 26);
          drawDiamond(ctx, tx, ty, 9, hd.lock_color);
          const lab = (hd.dist / 1000.0).toFixed(1).padStart(4, " ");
          ctx.font = `${FS_SMALL}px ${FONT}`;
          const lw = ctx.measureText(lab).width;
          let chipX = Math.round(tx - lw / 2 - 3);
          let chipY = Math.round(ty - 26);
          if (chipY < fy + 80) chipY = Math.round(Math.min(ty + 14, fy + fh - 18));
          const lo = fx + 94;
          const hi = fx + fw - 118 - (lw + 6);
          if (hi >= lo) chipX = Math.round(clamp(chipX, lo, hi));
          chipX = clamp(chipX, fx, fx + fw - lw - 6);
          ctx.fillStyle = "rgba(10,22,18,0.80)";
          ctx.fillRect(chipX, chipY, lw + 6, 16);
          text(ctx, lab, chipX + 3, chipY + 2, hd.lock_color, FS_SMALL);
        } else {
          // 出视场：头瞄指引贴在梯窗两侧
          const rightSide = hd.rel > 0.0;
          const side = rightSide ? fx + fw - 138 : ladderLeft + 4;
          ctx.beginPath();
          ctx.moveTo(side, cy - 7);
          ctx.lineTo(side, cy + 7);
          ctx.lineTo(side + (rightSide ? 10 : -10), cy);
          ctx.closePath();
          ctx.fillStyle = hd.lock_color; ctx.fill();
          const lab = "HMD";
          ctx.font = `${FS_SMALL}px ${FONT}`;
          text(ctx, lab, rightSide ? side - ctx.measureText(lab).width : side,
               cy + 10, hd.lock_color, FS_SMALL);
        }
      } else if (hd.contact === 1 || hd.rwr) {
        drawBearingMarker(ctx, fx, fy, fw, fh, hd.heading, hd.bearing ?? hd.heading);
      }
    }

    // 机炮准星
    if (Math.abs(hd.gun_az) < 52.0) {
      const [gx, gy] = viewPt(hd.gun_az, hd.gun_el);
      if (gx > fx + 96 && gx < fx + fw - 96 && gy > fy + 56 && gy < fy + fh - 26) {
        ctx.beginPath(); ctx.arc(gx, gy, 9, 0, TAU);
        ctx.strokeStyle = C.hudMark; ctx.lineWidth = 1; ctx.stroke();
        for (let k = 0; k < 4; k += 1) {
          const a = Math.PI * 0.5 * k;
          const ox = Math.cos(a), oy = Math.sin(a);
          ctx.beginPath();
          ctx.moveTo(gx + ox * 11, gy + oy * 11);
          ctx.lineTo(gx + ox * 15, gy + oy * 15);
          ctx.strokeStyle = C.hudMark; ctx.stroke();
        }
        ctx.beginPath(); ctx.arc(gx, gy, 2, 0, TAU);
        ctx.fillStyle = C.hudMark; ctx.fill();
      }
    }

    // 单位标注
    text(ctx, "SPD m/s", fx + 6, fy + fh - 19, C.hudDim, FS_SMALL);
    ctx.font = `${FS_SMALL}px ${FONT}`;
    const altCap = "ALT m";
    text(ctx, altCap, fx + fw - 6 - ctx.measureText(altCap).width,
         fy + fh - 19, C.hudDim, FS_SMALL);

    // 来袭导弹
    if (hd.threat) {
      const az = hd.threat.az;
      const wx = clamp(cx + Math.sin(az) * half * 0.8, fx + 96, fx + fw - 96);
      const wy = clamp(cy - Math.cos(az) * fh * 0.34, fy + 56, fy + fh - 26);
      ctx.beginPath(); ctx.arc(wx, wy, 5, 0, TAU);
      ctx.fillStyle = C.hudWarn; ctx.fill();
      const lab = `MSL ${String(Math.round(hd.threat.az * 180 / Math.PI)).padStart(4, " ")}  ${(hd.threat.d / 1000).toFixed(1)}km`;
      ctx.font = `${FS_SMALL}px ${FONT}`;
      const lw = ctx.measureText(lab).width;
      const bandX = cx - lw / 2 - 4;
      const bandY = fy + fh - 43;
      ctx.fillStyle = "rgba(10,22,18,0.90)";
      ctx.fillRect(bandX, bandY, lw + 8, 17);
      text(ctx, lab, bandX + 4, bandY + 2, C.hudWarn, FS_SMALL);
    }
  }

  function drawDiamond(ctx, x, y, r, color) {
    ctx.beginPath();
    ctx.moveTo(x, y - r); ctx.lineTo(x + r, y);
    ctx.lineTo(x, y + r); ctx.lineTo(x - r, y);
    ctx.closePath();
    ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.stroke();
  }

  function drawBearingMarker(ctx, fx, fy, fw, fh, hdg, brg) {
    const span = 200.0;
    const pxPerDeg = fw / span;
    const delta = ((brg - hdg + 180.0) % 360.0 + 360.0) % 360.0 - 180.0;
    const x = clamp(fx + fw / 2 + delta * pxPerDeg, fx + 12, fx + fw - 12);
    const top = fy + 3;
    ctx.beginPath();
    ctx.moveTo(x - 6, top); ctx.lineTo(x + 6, top); ctx.lineTo(x, top + 11);
    ctx.closePath();
    ctx.fillStyle = rgbHex(RGB.ACCENT_RED); ctx.fill();
  }

  function drawSpeedTape(ctx, x, top, bottom, v, cy) {
    const pxPerMs = 1.5;
    const lineX = x + 34;
    ctx.beginPath(); ctx.moveTo(lineX, top); ctx.lineTo(lineX, bottom);
    ctx.strokeStyle = C.hudDim; ctx.lineWidth = 1; ctx.stroke();
    const lo = v - (cy - top) / pxPerMs;
    const hi = v - (cy - bottom) / pxPerMs;
    for (let k = Math.floor(lo / 10.0) * 10.0; k <= hi; k += 10.0) {
      const y = cy - (k - v) * pxPerMs;
      const major = (Math.round(k) % 50) === 0;
      const ln = major ? 7 : 4;
      ctx.beginPath(); ctx.moveTo(lineX - ln, y); ctx.lineTo(lineX, y);
      ctx.strokeStyle = C.hudDim; ctx.stroke();
      if (major && Math.abs(y - cy) > 24 && y - 9 >= top && y + 4 <= bottom) {
        text(ctx, String(Math.round(k)).padStart(3, " "), x + 2, y - 9,
             C.greenRgb, FS_SMALL);
      }
    }
    const boxX = x + 46;
    const value = String(Math.round(v)).padStart(3, " ");
    drawValueBox(ctx, boxX, cy - 11, 44, 22, value, false);
    ctx.beginPath();
    ctx.moveTo(x + 34, cy); ctx.lineTo(x + 46, cy - 6); ctx.lineTo(x + 46, cy + 6);
    ctx.closePath();
    ctx.fillStyle = C.hudMark; ctx.fill();
  }

  function drawAltitudeTape(ctx, right, top, bottom, z, cy) {
    const pxPer100m = clamp((bottom - top) / 58.0, 2.6, 4.6);
    const labStep = pxPer100m * 5.0 >= 18.0 ? 500.0 : 1000.0;
    const lineX = right - 34;
    ctx.beginPath(); ctx.moveTo(lineX, top); ctx.lineTo(lineX, bottom);
    ctx.strokeStyle = C.hudDim; ctx.stroke();
    const span = (bottom - top) / pxPer100m * 100.0;
    const lo = z - span * 0.5;
    const hi = z + span * 0.5;
    for (let k = Math.floor(lo / 100.0) * 100.0; k <= hi; k += 100.0) {
      const y = cy - (k - z) / 100.0 * pxPer100m;
      const major = (Math.round(k) % 500) === 0;
      const ln = major ? 7 : 4;
      ctx.beginPath(); ctx.moveTo(lineX, y); ctx.lineTo(lineX + ln, y);
      ctx.strokeStyle = C.hudDim; ctx.stroke();
      if (major && Math.abs(y - cy) > 24 && (Math.round(k) % Math.round(labStep)) === 0
          && y - 9 >= top && y + 4 <= bottom) {
        ctx.font = `${FS_SMALL}px ${FONT}`;
        const lab = String(Math.round(k)).padStart(5, " ");
        text(ctx, lab, lineX - 9 - ctx.measureText(lab).width, y - 9,
             C.greenRgb, FS_SMALL);
      }
    }
    const value = String(Math.round(z)).padStart(5, " ");
    drawValueBox(ctx, right - 90, cy - 11, 44, 22, value, false);
    ctx.beginPath();
    ctx.moveTo(right - 34, cy); ctx.lineTo(right - 46, cy - 6);
    ctx.lineTo(right - 46, cy + 6);
    ctx.closePath();
    ctx.fillStyle = C.hudMark; ctx.fill();
  }

  function drawHeadingTape(ctx, fx, fy, fw, fh, hdg) {
    const y = fy + 26;
    const cx = fx + fw / 2;
    ctx.fillStyle = "rgba(8,22,30,0.47)";
    ctx.fillRect(fx + 2, fy + 2, fw - 4, 42);
    ctx.beginPath(); ctx.moveTo(fx + 6, y); ctx.lineTo(fx + fw - 6, y);
    ctx.strokeStyle = C.hudDim; ctx.lineWidth = 1; ctx.stroke();
    const hdgCap = "HDG";
    ctx.font = `${FS_SMALL}px ${FONT}`;
    const hdgCapW = ctx.measureText(hdgCap).width;
    const labelLeft = fx + 6 + hdgCapW + 6;
    const span = 200.0;
    const pxPerDeg = fw / span;
    for (let t = 0; t < 360; t += 5) {
      const delta = ((t - hdg + 180.0) % 360.0 + 360.0) % 360.0 - 180.0;
      if (Math.abs(delta) > span * 0.5 + 4.0) continue;
      const x = cx + delta * pxPerDeg;
      if (x < fx + 6 || x > fx + fw - 6) continue;
      if (Math.abs(x - cx) < 32.0) continue;
      const major = t % 30 === 0;
      const ln = major ? 9 : 5;
      ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x, y - ln);
      ctx.strokeStyle = C.hudDim; ctx.stroke();
      if (major) {
        ctx.font = `${FS_SMALL}px ${FONT}`;
        const lab = String(t).padStart(3, "0");
        const lw = ctx.measureText(lab).width;
        if (x - lw * 0.5 < labelLeft) continue;
        if (x + lw * 0.5 > fx + fw - 10) continue;
        if (Math.abs(x - cx) < 26.0 + lw * 0.5 + 4.0) continue;
        text(ctx, lab, x - lw / 2, y + 2, C.greenRgb, FS_SMALL);
      }
    }
    drawValueBox(ctx, cx - 26, y - 11, 52, 22,
                 String(Math.round(hdg)).padStart(3, "0"), true);
    text(ctx, hdgCap, fx + 6, y + 2, C.hudDim, FS_SMALL);
  }

  // ---------- 武器页俯视轮廓 ----------
  const SILHOUETTE_RIGHT = {
    "SU-57": [
      [0.50, 0.000], [0.43, 0.014], [0.26, 0.022], [0.08, 0.034],
      [-0.01, 0.081], [-0.06, 0.222], [-0.12, 0.322], [-0.20, 0.351],
      [-0.36, 0.298], [-0.27, 0.137], [-0.30, 0.060], [-0.44, 0.105],
      [-0.50, 0.085],
    ],
    "F-16V": [
      [0.50, 0.000], [0.44, 0.030], [0.28, 0.045], [0.10, 0.070],
      [0.01, 0.240], [-0.06, 0.330], [-0.14, 0.325],
      [-0.20, 0.200], [-0.22, 0.090], [-0.36, 0.200], [-0.47, 0.215],
      [-0.50, 0.130],
    ],
  };

  function silhouettePoints(name) {
    const right = SILHOUETTE_RIGHT[name] || SILHOUETTE_RIGHT["SU-57"];
    return right.concat(right.slice().reverse().map((p) => [p[0], -p[1]]));
  }

  function drawSilhouette(cv, name, team, w, h) {
    if (w <= 2 || h <= 2) return;
    const ctx = fitCanvas(cv, w, h);
    const pts = silhouettePoints(name);
    let minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
    for (const p of pts) {
      minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]);
      minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]);
    }
    const sx = (w - 12) / Math.max(1e-6, maxY - minY);
    const sy = (h - 8) / Math.max(1e-6, maxX - minX);
    const scale = Math.min(sx, sy);
    const cx = w / 2;
    const cy = h / 2;
    ctx.beginPath();
    pts.forEach((p, k) => {
      const x = cx + p[1] * scale;
      const y = cy - p[0] * scale;
      if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.closePath();
    ctx.fillStyle = rgba(mix(RGB.CLAY_LIT, RGB.TEAM[team], 0.22), 0.92);
    ctx.fill();
    ctx.strokeStyle = "rgba(232,234,236,0.75)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  // ---------- 顶部姿态框 ----------
  const ATT_VIEW = [0.85, -0.98, 0.55];
  const ATT_RIGHT = [
    ATT_VIEW[1] * 1 - ATT_VIEW[2] * 0, // cross((0,0,1), view)
    ATT_VIEW[2] * 0 - ATT_VIEW[0] * 1,
    ATT_VIEW[0] * 0 - ATT_VIEW[1] * 0,
  ];
  function norm3(v) {
    const n = Math.hypot(v[0], v[1], v[2]) || 1;
    return [v[0] / n, v[1] / n, v[2] / n];
  }
  const ATT_VIEW_N = norm3(ATT_VIEW);
  const ATT_RIGHT_N = norm3(ATT_RIGHT);
  const ATT_UP_N = norm3([
    ATT_VIEW_N[1] * ATT_RIGHT_N[2] - ATT_VIEW_N[2] * ATT_RIGHT_N[1],
    ATT_VIEW_N[2] * ATT_RIGHT_N[0] - ATT_VIEW_N[0] * ATT_RIGHT_N[2],
    ATT_VIEW_N[0] * ATT_RIGHT_N[1] - ATT_VIEW_N[1] * ATT_RIGHT_N[0],
  ]);

  function bodyAxes(psi, theta, phi) {
    const sp = Math.sin(psi), cp = Math.cos(psi);
    const st = Math.sin(theta), ct = Math.cos(theta);
    const sf = Math.sin(phi), cf = Math.cos(phi);
    return [
      [ct * cp, ct * sp, st],
      [cf * sp + sf * st * cp, -cf * cp + sf * st * sp, -sf * ct],
      [-sf * sp + cf * st * cp, sf * cp + cf * st * sp, -cf * ct],
    ];
  }

  function drawAttitude(part, ac) {
    const w = part.cv.clientWidth || 200;
    const h = part.cv.clientHeight || 134;
    const ctx = fitCanvas(part.cv, w, h);
    const rectBottom = h - 4;
    const zx = 12;
    ctx.beginPath();
    ctx.moveTo(4, rectBottom); ctx.lineTo(w - 4, rectBottom);
    ctx.strokeStyle = "rgb(34,44,58)"; ctx.lineWidth = 1; ctx.stroke();
    ctx.beginPath(); ctx.moveTo(zx, rectBottom - 2); ctx.lineTo(zx, 10);
    ctx.strokeStyle = "rgb(120,132,148)"; ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(zx - 3, 15); ctx.lineTo(zx, 9); ctx.lineTo(zx + 3, 15);
    ctx.strokeStyle = "rgb(120,132,148)"; ctx.stroke();
    text(ctx, "Z", zx + 4, 8, "rgb(150,162,178)", FS_SMALL);

    const psi = ac.psi, theta = ac.theta, phi = ac.phi;
    const [xb, yb, zb] = bodyAxes(psi, theta, phi);
    const name = ac.name;
    const length = ac.model_len;
    const scale = 100.0 / length;
    const cx = w / 2.0;
    const cy = h / 2.0 + 6;
    const project = (p) => {
      const wx = xb[0] * p[0] + yb[0] * p[1] + zb[0] * p[2];
      const wy = xb[1] * p[0] + yb[1] * p[1] + zb[1] * p[2];
      const wz = xb[2] * p[0] + yb[2] * p[1] + zb[2] * p[2];
      return [
        cx + (wx * ATT_RIGHT_N[0] + wy * ATT_RIGHT_N[1] + wz * ATT_RIGHT_N[2]) * scale,
        cy - (wx * ATT_UP_N[0] + wy * ATT_UP_N[1] + wz * ATT_UP_N[2]) * scale,
      ];
    };

    // 尾焰
    if (ac.alive && ac.flame > 0.01) {
      const tail = project([-length / 2.0, 0, 0]);
      const tip = project([-length / 2.0 - length * (0.10 + 0.34 * ac.flame), 0, 0]);
      const ddx = tip[0] - tail[0], ddy = tip[1] - tail[1];
      const n = Math.hypot(ddx, ddy) || 1;
      const ux = ddx / n, uy = ddy / n;
      const wid = 3.0 + 7.0 * ac.flame;
      ctx.beginPath();
      ctx.moveTo(tail[0] - uy * wid, tail[1] + ux * wid);
      ctx.lineTo(tip[0], tip[1]);
      ctx.lineTo(tail[0] + uy * wid, tail[1] - ux * wid);
      ctx.closePath();
      ctx.fillStyle = rgbHex(mix([214, 216, 220], [250, 250, 252], 0.35 + 0.45 * ac.ab));
      ctx.fill();
    }

    const pts = silhouettePoints(name);
    ctx.beginPath();
    pts.forEach((p, k) => {
      const q = project([p[0] * length, p[1] * length, 0]);
      if (k === 0) ctx.moveTo(q[0], q[1]); else ctx.lineTo(q[0], q[1]);
    });
    ctx.closePath();
    ctx.fillStyle = ac.alive ? rgbHex(mix(RGB.CLAY_LIT, RGB.TEAM[ac.team], 0.10)) : "rgb(110,114,120)";
    ctx.fill();
    ctx.strokeStyle = "rgb(34,36,40)";
    ctx.lineWidth = 1;
    ctx.stroke();
  }

  function updateAttitude(snap) {
    for (let i = 0; i < 2; i += 1) {
      const ac = snap.acs[i];
      const part = attitudeParts[i];
      part.name.textContent = ac.name;
      part.name.style.color = C.team[i];
      drawAttitude(part, ac);
      const thr = ac.alive ? clamp(ac.throttle, 0, 1) : 0;
      const ab = ac.alive ? clamp(ac.ab, 0, 1) : 0;
      part.thrFill.style.width = `${Math.round(thr * 100)}%`;
      part.thrBar.classList.toggle("ab", ab > 0.05);
      part.thrValue.textContent = `${Math.floor(thr * 100).toString().padStart(3, " ")}%${ab > 0.05 ? " AB" : ""}`;
      part.fuelFill.style.width = `${Math.round(ac.fuel_frac * 100)}%`;
      part.fuelValue.textContent = `${Math.floor(ac.fuel_frac * 100).toString().padStart(3, " ")}%`;
    }
  }

  // ---------- 布局 ----------
  let stationsVisible = false;

  function layoutAll() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    stationsVisible = w >= 1180 && h >= 700;
    stationsEl.style.display = stationsVisible ? "block" : "none";
    if (stationsVisible) {
      for (let i = 0; i < 2; i += 1) {
        stationViews[i].setRects(stationRects(i, w, h));
      }
    }
    const halfW = Math.floor((440 - 16) / 2);
    for (const part of attitudeParts) {
      part.cv.style.width = `${halfW}px`;
      part.cv.style.height = "134px";
    }
  }

  // ---------- 快照 → 场景 / HUD ----------
  let snapshot = null;
  let cameraReady = false;
  let localYaw = 0.0;
  let localPitch = 0.0;
  let localZoom = 1.0;
  let camPos = new THREE.Vector3();
  let camTarget = new THREE.Vector3();

  function updateScene(snap) {
    buildFloor(snap.arena, snap.z_floor);
    const cameraPos = w2t(snap.camera.eye);
    for (let i = 0; i < 2; i += 1) {
      const ac = snap.acs[i];
      const view = aircraftViews[i];
      view.setModel(ac.model, ac.engines || []);
      const depth = Math.max(1.0, cameraPos.distanceTo(w2t(ac.pos)));
      view.update(ac, depth);
    }
    updateTrails(snap);
    updateMissiles(snap);
    updatePredictions(snap);
    updateGunfire(snap);
    updateMarks(snap);
  }

  function updateHud(snap) {
    const msl = snap.missiles || [];
    if (msl.length) {
      missileStatusEl.classList.remove("hidden");
      let html = "";
      for (const m of msl) {
        const tgt = snap.acs[m.target] ? snap.acs[m.target].name : "-";
        html += `<div class="ms-row"><span class="dot" style="background:${C.team[m.owner]}"></span>`
          + `<span>${m.name} → ${tgt}   ${String(Math.round(m.speed)).padStart(4, " ")} m/s</span></div>`;
      }
      missileStatusEl.innerHTML = html;
    } else {
      missileStatusEl.classList.add("hidden");
    }
    bottomBarEl.textContent = snap.bottom.text;
    bottomHintEl.textContent = "SPACE pause    R reset    F follow    drag / wheel camera";
    pausedEl.classList.toggle("hidden", !snap.paused);
    if (snap.result) {
      resultEl.classList.remove("hidden");
      resultEl.innerHTML = `<div class="result-title" style="color:${snap.result.color}">${snap.result.title}</div>`
        + `<div class="result-detail">${snap.result.detail}</div>`
        + '<div class="result-tip">press R to restart</div>';
    } else {
      resultEl.classList.add("hidden");
    }
    if (stationsVisible) {
      for (let i = 0; i < 2; i += 1) {
        stationViews[i].update(snap.acs[i], snap);
      }
    }
    updateAttitude(snap);
    updateControls(snap);
  }

  function updateControls(snap) {
    const btnPause = document.getElementById("btn-pause");
    btnPause.textContent = snap.paused ? "RESUME (SPACE)" : "PAUSE (SPACE)";
    const follow = snap.camera.follow;
    let label = "FOLLOW: OFF";
    if (follow === 0 || follow === 1) {
      label = `FOLLOW: ${snap.acs[follow].name}`;
    }
    document.getElementById("btn-follow").textContent = label;
  }

  function updateCameraFrame() {
    if (!snapshot) return;
    const c = snapshot.camera;
    const target = new THREE.Vector3(c.cx, c.cz, -c.cy);
    const pitch = clamp(c.pitch + localPitch, 0.05, 1.45);
    const yaw = c.yaw + localYaw;
    const dist = c.dist * localZoom;
    const cp = Math.cos(pitch), sp = Math.sin(pitch);
    const eye = new THREE.Vector3(
      target.x + dist * cp * Math.cos(yaw),
      target.y + dist * sp,
      target.z - dist * cp * Math.sin(yaw)
    );
    if (!cameraReady) {
      camPos.copy(eye);
      camTarget.copy(target);
      cameraReady = true;
    } else {
      camPos.lerp(eye, 0.12);
      camTarget.lerp(target, 0.12);
    }
    camera.position.copy(camPos);
    camera.lookAt(camTarget);
  }

  // ---------- 事件 / 控件 ----------
  let drag = null;
  canvas.addEventListener("pointerdown", (e) => {
    drag = { x: e.clientX, y: e.clientY };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!drag) return;
    const dx = e.clientX - drag.x;
    const dy = e.clientY - drag.y;
    drag = { x: e.clientX, y: e.clientY };
    localYaw -= dx * 0.005;
    localPitch += dy * 0.004;
  });
  canvas.addEventListener("pointerup", (e) => {
    drag = null;
    try { canvas.releasePointerCapture(e.pointerId); } catch (_) { /* ignore */ }
  });
  canvas.addEventListener("pointercancel", () => { drag = null; });
  canvas.addEventListener("wheel", (e) => {
    e.preventDefault();
    localZoom = clamp(localZoom * Math.exp(e.deltaY * 0.0012), 0.15, 8.0);
  }, { passive: false });

  async function postControl(body) {
    try {
      await fetch("/api/control", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
    } catch (err) {
      console.error("控制请求失败", err);
    }
  }

  document.getElementById("btn-pause").addEventListener("click", () => {
    if (!snapshot) return;
    postControl({ action: snapshot.paused ? "resume" : "pause" });
  });
  document.getElementById("btn-reset").addEventListener("click", () => {
    localYaw = localPitch = 0;
    localZoom = 1.0;
    cameraReady = false;
    postControl({ action: "reset" });
  });
  document.getElementById("btn-follow").addEventListener("click", () => {
    let follow = null;
    if (snapshot) {
      const cur = snapshot.camera.follow;
      if (cur === null || cur === undefined) follow = 0;
      else if (cur === 0) follow = 1;
      else follow = null;
    }
    localYaw = localPitch = 0;
    cameraReady = false;
    postControl({ action: "follow", follow });
  });
  window.addEventListener("keydown", (e) => {
    if (e.repeat) return;
    if (e.code === "Space") {
      e.preventDefault();
      if (snapshot) postControl({ action: snapshot.paused ? "resume" : "pause" });
    } else if (e.code === "KeyR") {
      localYaw = localPitch = 0;
      localZoom = 1.0;
      cameraReady = false;
      postControl({ action: "reset" });
    } else if (e.code === "KeyF") {
      let follow = null;
      if (snapshot) {
        const cur = snapshot.camera.follow;
        if (cur === null || cur === undefined) follow = 0;
        else if (cur === 0) follow = 1;
        else follow = null;
      }
      localYaw = localPitch = 0;
      cameraReady = false;
      postControl({ action: "follow", follow });
    }
  });
  window.addEventListener("resize", () => {
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    layoutAll();
    if (snapshot) updateHud(snapshot);
  });

  // ---------- SSE ----------
  const stream = new EventSource("/api/stream");
  stream.onmessage = (ev) => {
    try {
      snapshot = JSON.parse(ev.data);
      MODELS_READY.then(() => {
        updateScene(snapshot);
        updateHud(snapshot);
      });
    } catch (err) {
      console.error("快照解析失败", err);
    }
  };
  stream.onerror = () => {
    // EventSource 会自动重连；这里只留一条日志，避免控制台刷屏。
    console.warn("SSE 连接断开，正在重连…");
  };

  // ---------- 渲染循环 ----------
  function animate() {
    requestAnimationFrame(animate);
    updateCameraFrame();
    renderer.render(scene, camera);
  }

  layoutAll();
  animate();
})();
