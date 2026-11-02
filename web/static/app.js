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
  // 高过载特效阈值；干扰物寿命与 sim.py 的 CHAFF_LIFE / FLARE_LIFE 一致，
  // 前端只用于视觉包络，不参与物理。
  const HIGH_G_ON = 4.5;
  // 暂时关闭高 G 蒸汽特效（翼尖/尾段雾带），改为 true 即可恢复。
  const HIGH_G_EFFECT = false;
  const CHAFF_LIFE = 4.0;
  const FLARE_LIFE = 2.5;

  // ISA 标准大气下的当地声速，用来判断飞机是否从亚音速突破音速。
  function speedOfSound(alt) {
    const h = clamp(alt, 0.0, 32000.0);
    let temp;
    if (h <= 11000.0) temp = 288.15 - 0.0065 * h;
    else if (h <= 20000.0) temp = 216.65;
    else temp = 216.65 + 0.001 * (h - 20000.0);
    return Math.sqrt(1.4 * 287.053 * temp);
  }
  // 迷彩每次加载随机生成一套花纹（同一页面内保持稳定）。
  const CAMO_SEED = Math.floor(Math.random() * 1e9);

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

  // ---------- 1080p 离屏/固定分辨率采集 ----------
  // 用法：/?capture=1920x1080&frames=90&fps=30&delay=5&name=j1
  // 采集模式下 WebGL 画布按固定分辨率渲染（与面板像素无关），
  // 渲染完成后把画面合成到离屏 2D 画布再 POST 给服务端，
  // 由 /api/capture 落盘、/api/capture/encode 用 ffmpeg 编码 MP4。
  const CAPTURE = (() => {
    const q = new URLSearchParams(window.location.search);
    const raw = q.get("capture") || q.get("res");
    if (!raw || raw === "0") return null;
    const m = /^(\d+)[xX×](\d+)$/.exec(raw);
    const w = m ? +m[1] : 1920;
    const h = m ? +m[2] : 1080;
    if (!(w > 0 && h > 0)) return null;
    return {
      w, h,
      delay: Math.max(0.0, parseFloat(q.get("delay") || "0") || 0.0),
      frames: Math.max(1, parseInt(q.get("frames") || "1", 10) || 1),
      fps: Math.max(1.0, parseFloat(q.get("fps") || "30") || 30.0),
      name: (q.get("name") || "shot").replace(/[^\w.-]+/g, "_"),
      encode: q.get("encode") !== "0",
    };
  })();

  // ---------- three.js 场景 ----------
  const canvas = document.getElementById("scene");
  const renderer = new THREE.WebGLRenderer({
    canvas,
    antialias: true,
    alpha: false,
    powerPreference: "high-performance",
  });
  const VIEW_W = CAPTURE ? CAPTURE.w : window.innerWidth;
  const VIEW_H = CAPTURE ? CAPTURE.h : window.innerHeight;
  renderer.setPixelRatio(CAPTURE ? 1 : Math.min(2, window.devicePixelRatio || 1));
  renderer.setSize(VIEW_W, VIEW_H, false);
  if (CAPTURE) canvas.classList.add("capture-mode");

  const scene = new THREE.Scene();
  scene.background = new THREE.Color(rgbHex(RGB.SCENE_BG));

  const camera = new THREE.PerspectiveCamera(
    50, VIEW_W / VIEW_H, 1, 1e7
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
  // 导弹模型由服务端按真实弹长导出：弹头在局部原点、弹体沿 +Z 向弹尾延伸。
  const MISSILE_URLS = {
    "R-73": "/api/models/r73.obj",
    "AIM-9X": "/api/models/aim9.obj",
    "R-77-1": "/api/models/r77.obj",
    "AIM-120C": "/api/models/aim120.obj",
  };
  const MODEL_GEOS = {};
  const MISSILE_GEOS = {};
  const MODELS_READY = (async () => {
    // 模型 URL 带上服务端导出版本号：MODEL_TAG 一升级，浏览器必定重新取文件，
    // 不会因为旧缓存继续显示旧模型（例如已修复的透明导弹）。
    let ver = "";
    try {
      const meta = await (await fetch("/api/models/meta.json", { cache: "no-store" })).json();
      if (meta && meta.version) ver = `?v=${encodeURIComponent(meta.version)}`;
    } catch (_) {
      /* 元数据缺失时退回无参数 URL，由服务端 no-store 兜底 */
    }
    const jobs = [
      ...Object.entries(MODEL_URLS).map(async ([name, url]) => {
        const res = await fetch(url + ver, { cache: "no-store" });
        const text = await res.text();
        MODEL_GEOS[name] = parseOBJ(text);
      }),
      ...Object.entries(MISSILE_URLS).map(async ([name, url]) => {
        const res = await fetch(url + ver, { cache: "no-store" });
        const text = await res.text();
        const geo = parseOBJ(text);
        assignMissileUV(geo.body);
        missileShared.add(geo.body);
        missileShared.add(geo.canopy);
        MISSILE_GEOS[name] = geo;
      }),
    ];
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

  // 导弹贴图用圆柱坐标：u 为周向角、v 为沿弹轴的归一化位置（0 弹头、1 弹尾）。
  // 弹体本身沿 +Z 生长，包围盒即可给出弹长，无需额外元数据。
  function assignMissileUV(geo) {
    geo.computeBoundingBox();
    const bb = geo.boundingBox;
    const len = Math.max(1e-6, bb.max.z - bb.min.z);
    const pos = geo.attributes.position;
    const uv = new Float32Array(pos.count * 2);
    for (let i = 0; i < pos.count; i += 1) {
      const x = pos.getX(i);
      const y = pos.getY(i);
      const z = pos.getZ(i);
      uv[i * 2] = (Math.atan2(y, x) + Math.PI) / TAU;
      uv[i * 2 + 1] = clamp((z - bb.min.z) / len, 0.0, 1.0);
    }
    geo.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  }

  // ---------- 机体迷彩 ----------
  // OBJ 没有 UV，用物体空间三平面投影在片元着色器里采样迷彩贴图；
  // 花纹参考 assets/flighter 的示意图：F-16V 为空优灰蓝软边双色迷彩，
  // SU-57 为浅灰底 + 蓝灰数码像素迷彩。贴图按机型、随机种子生成，
  // 机头雷达罩单独压深，保留机体识别特征。
  const CAMO_TEX_SIZE = 512;

  function camoRng(seed) {
    let s = (seed >>> 0) || 1;
    return () => {
      s = (s * 1664525 + 1013904223) >>> 0;
      return s / 4294967296;
    };
  }

  function camoCanvas() {
    const cv = document.createElement("canvas");
    cv.width = CAMO_TEX_SIZE;
    cv.height = CAMO_TEX_SIZE;
    return cv;
  }

  // 贴图边缘无缝平铺：同一笔在 3x3 邻域重复绘制。
  function wrapDraw(ctx, x, y, fn) {
    for (const dx of [-CAMO_TEX_SIZE, 0, CAMO_TEX_SIZE]) {
      for (const dy of [-CAMO_TEX_SIZE, 0, CAMO_TEX_SIZE]) {
        fn(x + dx, y + dy);
      }
    }
  }

  function f16CamoTexture(seed) {
    const rnd = camoRng(seed);
    const cv = camoCanvas();
    const g = cv.getContext("2d");
    const S = CAMO_TEX_SIZE;
    g.fillStyle = "#c2cad1";
    g.fillRect(0, 0, S, S);
    // 蒙皮细碎色差
    for (let i = 0; i < 5000; i += 1) {
      const v = 150 + Math.floor(rnd() * 76);
      g.fillStyle = `rgba(${v},${v + 4},${v + 9},${0.05 + rnd() * 0.08})`;
      const w = 1 + rnd() * 5;
      g.fillRect(rnd() * S, rnd() * S, w, w);
    }
    // 深灰蓝大斑块：若干软边圆叠出不规则软边，符合空优双色迷彩
    for (let i = 0; i < 15; i += 1) {
      const cx = rnd() * S;
      const cy = rnd() * S;
      const scale = 62 + rnd() * 112;
      const lobes = 5 + Math.floor(rnd() * 5);
      for (let k = 0; k < lobes; k += 1) {
        const ox = (rnd() - 0.5) * scale * 1.5;
        const oy = (rnd() - 0.5) * scale * 1.5;
        const r = scale * (0.34 + rnd() * 0.36);
        wrapDraw(g, cx, cy, (x, y) => {
          const grd = g.createRadialGradient(x + ox, y + oy, r * 0.5, x + ox, y + oy, r);
          grd.addColorStop(0, "rgba(60, 74, 88, 0.95)");
          grd.addColorStop(0.70, "rgba(63, 77, 91, 0.82)");
          grd.addColorStop(1, "rgba(66, 80, 94, 0)");
          g.fillStyle = grd;
          g.beginPath();
          g.arc(x + ox, y + oy, r, 0, TAU);
          g.fill();
        });
      }
    }
    // 少量更浅的补块，避免整片深灰
    for (let i = 0; i < 8; i += 1) {
      const cx = rnd() * S;
      const cy = rnd() * S;
      const r = 36 + rnd() * 78;
      wrapDraw(g, cx, cy, (x, y) => {
        const grd = g.createRadialGradient(x, y, r * 0.45, x, y, r);
        grd.addColorStop(0, "rgba(224, 230, 235, 0.55)");
        grd.addColorStop(1, "rgba(224, 230, 235, 0)");
        g.fillStyle = grd;
        g.beginPath();
        g.arc(x, y, r, 0, TAU);
        g.fill();
      });
    }
    // 极淡的蒙皮分块线
    g.strokeStyle = "rgba(58, 70, 82, 0.16)";
    g.lineWidth = 1;
    for (let i = 0; i < 26; i += 1) {
      const x = rnd() * S;
      const y = rnd() * S;
      const len = 30 + rnd() * 90;
      g.beginPath();
      g.moveTo(x, y);
      g.lineTo(x + (rnd() < 0.5 ? len : 0), y + (rnd() < 0.5 ? len : 0));
      g.stroke();
    }
    return cv;
  }

  function su57CamoTexture(seed) {
    const rnd = camoRng(seed);
    const cv = camoCanvas();
    const g = cv.getContext("2d");
    const S = CAMO_TEX_SIZE;
    g.fillStyle = "#d0d5da";
    g.fillRect(0, 0, S, S);
    for (let i = 0; i < 4200; i += 1) {
      const v = 165 + Math.floor(rnd() * 64);
      g.fillStyle = `rgba(${v},${v + 4},${v + 8},${0.04 + rnd() * 0.07})`;
      const w = 1 + rnd() * 4;
      g.fillRect(rnd() * S, rnd() * S, w, w);
    }
    // 数码像素迷彩：椭圆范围内按 8 px 网格铺色块，边缘呈阶梯状
    const shades = ["#a9b3be", "#8d99a6", "#6f7b89"];
    for (let i = 0; i < 24; i += 1) {
      const cx = rnd() * S;
      const cy = rnd() * S;
      const rx = 48 + rnd() * 98;
      const ry = 42 + rnd() * 86;
      g.fillStyle = shades[Math.floor(rnd() * shades.length)];
      const step = 8;
      for (let px = -rx; px <= rx; px += step) {
        for (let py = -ry; py <= ry; py += step) {
          const d = (px * px) / (rx * rx) + (py * py) / (ry * ry);
          if (d > 1.0) continue;
          if (rnd() > 0.94 - d * 0.62) continue;
          const w = step * (rnd() < 0.22 ? 2 : 1);
          wrapDraw(g, cx, cy, (x, y) => {
            g.fillRect(x + px, y + py, w, step);
          });
        }
      }
    }
    // 散落的单像素块，让边缘更像真实数码迷彩
    for (let i = 0; i < 500; i += 1) {
      g.fillStyle = shades[Math.floor(rnd() * shades.length)];
      g.globalAlpha = 0.35 + rnd() * 0.5;
      const s = 4 + Math.floor(rnd() * 3) * 4;
      g.fillRect(Math.floor(rnd() * S), Math.floor(rnd() * S), s, s);
    }
    g.globalAlpha = 1;
    return cv;
  }

  const CAMO_SPECS = {
    "F-16V": { draw: f16CamoTexture, tile: 7.4, len: 15.06, seed: 3,
      nose: [64, 70, 78], noseAmt: 1.0, tint: 0.12 },
    "SU-57": { draw: su57CamoTexture, tile: 9.2, len: 20.1, seed: 17,
      nose: [148, 156, 165], noseAmt: 0.85, tint: 0.12 },
  };
  const camoTextures = {};
  const camoMaterials = {};

  function camoTexture(name) {
    if (!camoTextures[name]) {
      const spec = CAMO_SPECS[name];
      const tex = new THREE.CanvasTexture(spec.draw(CAMO_SEED + spec.seed));
      tex.wrapS = THREE.RepeatWrapping;
      tex.wrapT = THREE.RepeatWrapping;
      tex.anisotropy = 4;
      tex.needsUpdate = true;
      camoTextures[name] = tex;
    }
    return camoTextures[name];
  }

  function camoMaterial(team, name) {
    const key = `${team}|${name}`;
    const cached = camoMaterials[key];
    if (cached) return cached;
    const part = CAMO_SPECS[name] || CAMO_SPECS["SU-57"];
    const mat = new THREE.MeshStandardMaterial({
      color: new THREE.Color(rgbHex(mix([255, 255, 255], RGB.TEAM[team], part.tint))),
      roughness: 0.66,
      metalness: 0.08,
      side: THREE.DoubleSide,
    });
    const noseVec = new THREE.Vector3(
      part.nose[0] / 255, part.nose[1] / 255, part.nose[2] / 255
    );
    mat.onBeforeCompile = (shader) => {
      shader.uniforms.uCamoTex = { value: camoTexture(name) };
      shader.uniforms.uCamoScale = { value: 1.0 / part.tile };
      shader.uniforms.uCamoLen = { value: part.len };
      shader.uniforms.uNoseColor = { value: noseVec };
      shader.uniforms.uNoseAmt = { value: part.noseAmt };
      shader.vertexShader = `varying vec3 vCamoPos;
varying vec3 vCamoNrm;
${shader.vertexShader}`;
      shader.vertexShader = shader.vertexShader.replace(
        "#include <begin_vertex>",
        `#include <begin_vertex>
  vCamoPos = position;
  vCamoNrm = normal;`
      );
      shader.fragmentShader = `varying vec3 vCamoPos;
varying vec3 vCamoNrm;
uniform sampler2D uCamoTex;
uniform float uCamoScale;
uniform float uCamoLen;
uniform vec3 uNoseColor;
uniform float uNoseAmt;
${shader.fragmentShader}`;
      shader.fragmentShader = shader.fragmentShader.replace(
        "#include <map_fragment>",
        `#include <map_fragment>
  vec3 camoN = abs(normalize(vCamoNrm));
  vec3 camoN6 = camoN * camoN;
  camoN6 = camoN6 * camoN6 * camoN6;
  vec3 camoW = camoN6 / (camoN6.x + camoN6.y + camoN6.z + 1e-4);
  vec3 camoP = vCamoPos * uCamoScale;
  vec3 camoCol = texture2D(uCamoTex, camoP.zy).rgb * camoW.x
    + texture2D(uCamoTex, camoP.xz).rgb * camoW.y
    + texture2D(uCamoTex, camoP.xy).rgb * camoW.z;
  diffuseColor.rgb *= camoCol;
  float camoFz = vCamoPos.z / uCamoLen;
  float camoNose = (1.0 - smoothstep(-0.42, -0.35, camoFz)) * uNoseAmt;
  diffuseColor.rgb = mix(diffuseColor.rgb, uNoseColor, camoNose);`
      );
    };
    camoMaterials[key] = mat;
    return mat;
  }

  const canopyMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(rgbHex(RGB.CANOPY)),
    roughness: 0.25,
    metalness: 0.15,
    transparent: true,
    opacity: 0.55,
    depthWrite: false,
    side: THREE.DoubleSide,
  });
  const deadBodyMat = new THREE.MeshStandardMaterial({
    color: new THREE.Color(rgbHex(RGB.CLAY_AO)),
    roughness: 0.80,
    metalness: 0.05,
    side: THREE.DoubleSide,
  });

  // ---------- 导弹涂装 ----------
  // 实弹涂装惯例：整弹浅灰/白色，导引头罩（雷达罩/红外罩）压成深灰；
  // 战斗部与发动机段用黄褐色识别环，另加黑色编号条，尾喷口为金属深灰。
  const MISSILE_LIVERY = {
    "R-73": { base: "#d3d5d7", nose: "#3d4248", band: "#b59a5c",
      bandAt: [0.34, 0.375], stripeAt: [0.62, 0.645] },
    "AIM-9X": { base: "#d8dbde", nose: "#33383e", band: "#c9a24e",
      bandAt: [0.28, 0.32], stripeAt: [0.55, 0.57] },
    "R-77-1": { base: "#cfd2d5", nose: "#3a3f45", band: "#b89a55",
      bandAt: [0.19, 0.23], stripeAt: [0.50, 0.52] },
    "AIM-120C": { base: "#cdd1d4", nose: "#343940", band: "#c7a04d",
      bandAt: [0.17, 0.21], stripeAt: [0.46, 0.48] },
  };
  const MISSILE_TEX_W = 128;
  const MISSILE_TEX_H = 512;
  const missileTextures = {};

  function missileTexture(name) {
    if (missileTextures[name]) return missileTextures[name];
    const spec = MISSILE_LIVERY[name] || MISSILE_LIVERY["R-73"];
    const cv = document.createElement("canvas");
    cv.width = MISSILE_TEX_W;
    cv.height = MISSILE_TEX_H;
    const g = cv.getContext("2d");
    const W = MISSILE_TEX_W;
    const H = MISSILE_TEX_H;
    g.fillStyle = spec.base;
    g.fillRect(0, 0, W, H);
    // 纵向漆面磨损（贴图 x 为周向、y 为轴向，竖线即纵向条痕）
    const rnd = camoRng(CAMO_SEED + 977 + name.length * 131);
    for (let i = 0; i < 80; i += 1) {
      const light = rnd() < 0.5;
      g.fillStyle = light
        ? `rgba(255,255,255,${0.03 + rnd() * 0.05})`
        : `rgba(36,40,46,${0.03 + rnd() * 0.06})`;
      g.fillRect(rnd() * W, rnd() * H, 1 + rnd() * 2, 40 + rnd() * 220);
    }
    for (let i = 0; i < 1600; i += 1) {
      const v = 150 + Math.floor(rnd() * 90);
      g.fillStyle = `rgba(${v},${v + 2},${v + 4},${0.05 + rnd() * 0.08})`;
      g.fillRect(rnd() * W, rnd() * H, 1, 1 + rnd() * 3);
    }
    // 导引头罩
    g.fillStyle = spec.nose;
    g.fillRect(0, 0, W, H * 0.11);
    g.fillStyle = "rgba(30,34,39,0.9)";
    g.fillRect(0, H * 0.11, W, H * 0.012);
    // 识别环与黑色编号条
    g.fillStyle = spec.band;
    g.fillRect(0, H * spec.bandAt[0], W, H * (spec.bandAt[1] - spec.bandAt[0]));
    g.fillStyle = "#2c3035";
    g.fillRect(0, H * spec.stripeAt[0], W, H * (spec.stripeAt[1] - spec.stripeAt[0]));
    // 尾喷口金属段
    g.fillStyle = "#585c61";
    g.fillRect(0, H * 0.905, W, H * 0.095);
    g.fillStyle = "rgba(28,31,36,0.85)";
    g.fillRect(0, H * 0.895, W, H * 0.008);
    const tex = new THREE.CanvasTexture(cv);
    tex.flipY = false;
    tex.wrapS = THREE.ClampToEdgeWrapping;
    tex.wrapT = THREE.ClampToEdgeWrapping;
    tex.anisotropy = 4;
    tex.needsUpdate = true;
    missileTextures[name] = tex;
    return tex;
  }

  // 导弹几何与材质跨帧共享，不能随每帧重建的场景节点一起 dispose。
  const missileMats = {};
  const missileDeadMat = new THREE.MeshStandardMaterial({
    color: 0x9a9ea4,
    roughness: 0.62,
    metalness: 0.22,
    side: THREE.DoubleSide,
  });
  const missileShared = new Set([missileDeadMat]);

  function missileMaterial(team, name) {
    const key = `${team}|${name}`;
    let mat = missileMats[key];
    if (mat) return mat;
    mat = new THREE.MeshStandardMaterial({
      map: missileTexture(name),
      color: new THREE.Color(rgbHex(mix([255, 255, 255], RGB.TEAM[team], 0.10))),
      roughness: 0.46,
      metalness: 0.32,
      side: THREE.DoubleSide,
    });
    missileMats[key] = mat;
    missileShared.add(mat);
    return mat;
  }

  // 尾焰：屏幕朝向的带状面片 + 程序化着色（与 flame_demo 同一套羽流）。
  // 面片两端对齐喷口与羽流末端，宽度在视空间内展开，任何视角下都是柔和
  // 光带；加力时走冷蓝，不开加力时走暖黄。
  const plumeVert = `
    attribute float aSide;
    attribute float aV;
    uniform float uLen;
    uniform float uW0;
    uniform float uW1;
    uniform float uSpread;
    uniform float uScale;
    varying vec2 vUv;
    void main() {
      vUv = vec2((aSide + 1.0) * 0.5, aV);
      vec4 p0 = modelViewMatrix * vec4(0.0, 0.0, 0.0, 1.0);
      vec4 p1 = modelViewMatrix * vec4(0.0, 0.0, uLen, 1.0);
      vec2 d = p1.xy - p0.xy;
      float dl = length(d);
      vec2 dir = dl > 1e-5 ? d / dl : vec2(0.0, 1.0);
      vec2 perp = vec2(-dir.y, dir.x);
      vec4 mv = mix(p0, p1, aV);
      mv.xy += perp * (aSide * 0.5 * mix(uW0, uW1, aV) * uSpread * uScale);
      gl_Position = projectionMatrix * mv;
    }
  `;

  const plumeFrag = `
    precision highp float;
    varying vec2 vUv;
    uniform float uTime;
    uniform float uLevel;
    uniform float uSeed;
    uniform float uGain;
    uniform float uCoreMix;
    uniform float uAb;

    vec3 rampCold(float v) {
      vec3 white  = vec3(0.92, 0.99, 1.00);
      vec3 cyan   = vec3(0.20, 0.82, 1.00);
      vec3 blue   = vec3(0.16, 0.40, 1.00);
      vec3 violet = vec3(0.62, 0.20, 1.00);
      if (v < 0.18) return mix(white, cyan, v / 0.18);
      if (v < 0.62) return mix(cyan, blue, (v - 0.18) / 0.44);
      if (v < 0.80) return mix(blue, violet, (v - 0.62) / 0.18);
      return mix(violet, vec3(0.82, 0.97, 0.56), clamp((v - 0.80) / 0.20, 0.0, 1.0));
    }

    vec3 rampWarm(float v) {
      vec3 white  = vec3(1.00, 0.98, 0.90);
      vec3 yellow = vec3(1.00, 0.82, 0.30);
      vec3 orange = vec3(1.00, 0.54, 0.12);
      vec3 amber  = vec3(0.92, 0.34, 0.06);
      vec3 ember  = vec3(0.70, 0.18, 0.03);
      if (v < 0.18) return mix(white, yellow, v / 0.18);
      if (v < 0.62) return mix(yellow, orange, (v - 0.18) / 0.44);
      if (v < 0.82) return mix(orange, amber, (v - 0.62) / 0.20);
      return mix(amber, ember, clamp((v - 0.82) / 0.18, 0.0, 1.0));
    }

    void main() {
      float v = vUv.y;
      float u = (vUv.x - 0.5) * 2.0;

      float core = exp(-pow(abs(u) * 2.80, 2.0));
      float halo = exp(-pow(abs(u) * 1.00, 2.0));
      float edge = 1.0 - smoothstep(0.52, 1.0, abs(u));

      float alongCore = 1.0 - smoothstep(0.42, 1.0, v);
      float alongHalo = 1.0 - smoothstep(0.26, 1.0, v);
      float head = smoothstep(0.0, 0.045, v);
      float hot = smoothstep(0.42, 0.0, v);

      float diamond = 0.62 + 0.38 * cos(v * 27.0 - uTime * 2.0 + uSeed * 4.0);
      diamond = mix(1.0, diamond, smoothstep(0.62, 0.06, v) * (0.35 + 0.65 * uAb));

      float turb = 0.84
        + 0.10 * sin(u * 7.0 + uTime * 4.0 + uSeed * 9.0)
        + 0.06 * sin(v * 16.0 - uTime * 6.0 + uSeed * 5.0);

      float a = (core * uCoreMix * alongCore + halo * 0.34 * alongHalo)
        * head * diamond * turb * edge;
      a *= uLevel * uGain * (1.0 + 0.25 * hot);
      a = clamp(a, 0.0, 1.0);

      vec3 colCore = mix(rampWarm(v), rampCold(v), uAb);
      vec3 colHalo = mix(vec3(1.00, 0.46, 0.10), vec3(1.00, 0.28, 0.82), uAb);
      vec3 col = colCore * core * uCoreMix * 1.15 + colHalo * halo * 0.30;
      vec3 hotCol = mix(vec3(1.00, 0.93, 0.74), vec3(0.84, 0.97, 1.00), uAb);
      col = mix(col, hotCol, 0.55 * hot);
      float tip = smoothstep(0.56, 0.98, v);
      vec3 tipCol = mix(vec3(1.00, 0.66, 0.20), vec3(0.88, 0.98, 0.58), uAb);
      col = mix(col, tipCol, tip * (0.45 + 0.40 * halo));

      gl_FragColor = vec4(col, a);
    }
  `;

  const plumeGeo = new THREE.BufferGeometry();
  plumeGeo.setAttribute("position", new THREE.Float32BufferAttribute(
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 3));
  plumeGeo.setAttribute("aSide", new THREE.Float32BufferAttribute([-1, 1, -1, 1], 1));
  plumeGeo.setAttribute("aV", new THREE.Float32BufferAttribute([0, 0, 1, 1], 1));
  plumeGeo.setIndex([0, 1, 2, 2, 1, 3]);

  function makeGlowTexture() {
    const s = 256;
    const cv = document.createElement("canvas");
    cv.width = cv.height = s;
    const ctx = cv.getContext("2d");
    const grd = ctx.createRadialGradient(s / 2, s / 2, 0, s / 2, s / 2, s / 2);
    grd.addColorStop(0.00, "rgba(235,250,255,1.00)");
    grd.addColorStop(0.12, "rgba(150,225,255,0.85)");
    grd.addColorStop(0.28, "rgba(110,140,255,0.55)");
    grd.addColorStop(0.48, "rgba(230,90,220,0.28)");
    grd.addColorStop(0.72, "rgba(150,40,140,0.10)");
    grd.addColorStop(1.00, "rgba(0,0,0,0)");
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, s, s);
    return new THREE.CanvasTexture(cv);
  }
  const glowTex = makeGlowTexture();

  const PLUME_HOT_WARM = new THREE.Color(0xffd27a);
  const PLUME_HOT_COLD = new THREE.Color(0xbfe4ff);
  const PLUME_HALO_WARM = new THREE.Color(0xff8a2a);
  const PLUME_HALO_COLD = new THREE.Color(0xc257ff);
  const PLUME_LIGHT_WARM = new THREE.Color(0xffb050);
  const PLUME_LIGHT_COLD = new THREE.Color(0x74c8ff);

  function plumeMesh(seed, spread, coreMix, gain) {
    const uniforms = {
      uTime: { value: 0 },
      uLevel: { value: 1.0 },
      uSeed: { value: seed },
      uGain: { value: gain },
      uCoreMix: { value: coreMix },
      uSpread: { value: spread },
      uLen: { value: 6.6 },
      uW0: { value: 0.62 },
      uW1: { value: 1.60 },
      uScale: { value: 1.0 },
      uAb: { value: 1.0 },
    };
    const mat = new THREE.ShaderMaterial({
      vertexShader: plumeVert,
      fragmentShader: plumeFrag,
      uniforms,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    const m = new THREE.Mesh(plumeGeo, mat);
    m.frustumCulled = false;
    return m;
  }

  // 每台发动机两层羽流 + 喷口高热辉光 + 中段大光晕 + 照亮机体的点光。
  function makePlume(seed, modelLen) {
    const sf = modelLen / REF_LENGTH;
    const group = new THREE.Group();
    const core = plumeMesh(seed, 1.0, 1.0, 1.70);
    const halo = plumeMesh(seed + 0.4, 3.2, 0.0, 0.50);
    group.add(core, halo);

    const glowHot = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTex, color: PLUME_HOT_WARM, transparent: true,
      blending: THREE.AdditiveBlending, depthWrite: false,
      opacity: 0.55,
    }));
    glowHot.position.set(0, 0, 0.30 * sf);
    glowHot.scale.setScalar(2.2 * sf);
    group.add(glowHot);

    const glowHalo = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTex, color: PLUME_HALO_WARM, transparent: true,
      blending: THREE.AdditiveBlending, depthWrite: false,
      opacity: 0.26,
    }));
    glowHalo.position.set(0, 0, 4.2 * sf);
    glowHalo.scale.setScalar(8.2 * sf);
    group.add(glowHalo);

    const light = new THREE.PointLight(0xffb050, 1.6, 34.0 * sf, 1.8);
    light.position.set(0, 0, 0.9 * sf);
    group.add(light);

    group.userData = {
      core: core.material.uniforms,
      halo: halo.material.uniforms,
      glowHot, glowHalo, light,
    };
    return group;
  }

  function setPlumeState(plume, level, ab, len, scale, sf) {
    const g = plume.userData;
    for (const u of [g.core, g.halo]) {
      u.uLevel.value = level;
      u.uAb.value = ab;
      u.uLen.value = len;
      u.uW0.value = 0.62 * sf;
      u.uW1.value = (1.15 + 0.50 * level) * sf;
      u.uScale.value = scale;
      u.uGain.value = 1.35 + 0.35 * ab;
    }
    g.glowHot.material.color.copy(PLUME_HOT_WARM).lerp(PLUME_HOT_COLD, ab);
    g.glowHot.material.opacity = 0.10 + 0.45 * level;
    g.glowHot.position.z = 0.30 * sf;
    g.glowHalo.material.color.copy(PLUME_HALO_WARM).lerp(PLUME_HALO_COLD, ab);
    g.glowHalo.material.opacity = 0.08 + 0.22 * level;
    g.glowHalo.position.z = 0.42 * len;
    g.glowHalo.scale.setScalar((6.4 + 1.8 * level) * sf);
    g.light.color.copy(PLUME_LIGHT_WARM).lerp(PLUME_LIGHT_COLD, ab);
    g.light.intensity = (0.55 + 1.05 * level) * (1.0 + 0.30 * ab);
  }

  // 导弹尾焰：单层小羽流 + 喷口辉光，只在点火且仍有推进剂时显示。
  function makeMissilePlume(seed) {
    const group = new THREE.Group();
    const core = plumeMesh(seed, 1.0, 1.0, 1.9);
    group.add(core);
    const glow = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTex,
      color: PLUME_HOT_WARM,
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      opacity: 0.55,
    }));
    group.add(glow);
    group.userData = { core: core.material.uniforms, glow };
    return group;
  }

  function setMissilePlumeState(plume, len, level, time, scale) {
    const u = plume.userData.core;
    u.uTime.value = time;
    u.uLevel.value = level;
    u.uAb.value = 0.0;
    u.uLen.value = len;
    u.uW0.value = 0.10;
    u.uW1.value = 0.30;
    u.uScale.value = scale;
    u.uGain.value = 1.9;
    plume.userData.glow.material.opacity = 0.30 + 0.40 * level;
    plume.userData.glow.scale.setScalar(0.75 + 0.45 * level);
  }

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

  // 高过载时翼尖拉出的白色蒸汽/涡流。复用尾焰的屏幕朝向带状顶点着色器，
  // 只是把长度、宽度和颜色换成淡淡的冷凝雾。
  const vaporFrag = `
    precision highp float;
    varying vec2 vUv;
    uniform float uAlpha;
    uniform float uTime;
    uniform float uSeed;
    uniform vec3 uColor;
    void main() {
      float v = vUv.y;
      float u = (vUv.x - 0.5) * 2.0;
      float core = exp(-pow(abs(u) * 1.8, 2.0));
      float edge = 1.0 - smoothstep(0.55, 1.0, abs(u));
      float head = smoothstep(0.0, 0.16, v);
      float trail = 1.0 - smoothstep(0.30, 1.0, v);
      float turb = 0.74
        + 0.14 * sin(v * 15.0 - uTime * 5.0 + uSeed * 7.0)
        + 0.12 * sin(u * 9.0 + uTime * 3.2 + uSeed * 3.0);
      float a = core * edge * (head * 0.85 + trail * 0.75) * turb * uAlpha;
      vec3 col = mix(uColor, vec3(1.0), 0.35 * core);
      gl_FragColor = vec4(col, clamp(a, 0.0, 1.0));
    }
  `;

  function makeVaporMaterial(seed, color) {
    return new THREE.ShaderMaterial({
      vertexShader: plumeVert,
      fragmentShader: vaporFrag,
      uniforms: {
        uLen: { value: 1.0 },
        uW0: { value: 0.04 },
        uW1: { value: 0.15 },
        uSpread: { value: 1.0 },
        uScale: { value: 1.0 },
        uAlpha: { value: 0.0 },
        uTime: { value: 0.0 },
        uSeed: { value: seed },
        uColor: { value: new THREE.Color(color) },
      },
      transparent: true,
      blending: THREE.NormalBlending,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
  }

  const texBoomCloud = makeSpriteTexture((ctx, s) => {
    const rnd = camoRng(0x9c41);
    ctx.clearRect(0, 0, s, s);
    for (let i = 0; i < 10; i += 1) {
      const x = s * 0.5 + (rnd() - 0.5) * s * 0.48;
      const y = s * 0.5 + (rnd() - 0.5) * s * 0.48;
      const r = s * (0.14 + rnd() * 0.18);
      const grd = ctx.createRadialGradient(x, y, 0, x, y, r);
      grd.addColorStop(0, `rgba(255,255,255,${0.22 + rnd() * 0.18})`);
      grd.addColorStop(0.50, "rgba(236,246,255,0.10)");
      grd.addColorStop(1.00, "rgba(220,236,255,0)");
      ctx.fillStyle = grd;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, TAU);
      ctx.fill();
    }
  }, 128);

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
      this.modelLen = REF_LENGTH;
      // 发动机中心相对机体中心的偏移（模型局部 y 向下为负、z 向机尾为正），
      // 用于把飘带下移到喷口高度并后移到喷口位置；数值在 setModel 时给出。
      this.engineY = 0.0;
      this.engineZ = 0.0;

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

      // 高 G 蒸汽与音爆云挂在机体 group 上，跟着姿态和视觉缩放一起走。
      this.effectGroup = new THREE.Group();
      this.group.add(this.effectGroup);
      this.vaporStreaks = [];
      this.vaporHalos = [];
      this.highGLevel = 0.0;
      this.machPrev = null;
      this.boomStart = -1e9;
      this.effectLastT = null;
      this.boomCloud = null;
      this.boomSmoke = [];
      this.boomPuffs = [];
    }

    setModel(name, engines, modelLen) {
      const ml = modelLen || REF_LENGTH;
      if (this.name === name && this.engineCount === engines.length && this.modelLen === ml) return;
      this.name = name;
      this.engineCount = engines.length;
      this.modelLen = ml;
      this.engineY = engines.length
        ? engines.reduce((sum, e) => sum + e[1], 0) / engines.length
        : 0.0;
      this.engineZ = engines.length
        ? engines.reduce((sum, e) => sum + e[2], 0) / engines.length
        : 0.0;
      this.modelGroup.clear();
      this.bodyMesh = null;
      this.canopyMesh = null;
      this.flames = [];
      const geo = MODEL_GEOS[name];
      if (geo) {
        this.bodyMesh = new THREE.Mesh(geo.body, camoMaterial(this.team, name));
        this.canopyMesh = new THREE.Mesh(geo.canopy, canopyMat);
        this.modelGroup.add(this.bodyMesh, this.canopyMesh);
      }
      engines.forEach((eng, idx) => {
        const plume = makePlume(idx === 0 ? 0.35 : 1.75, ml);
        plume.position.set(eng[0], eng[1], eng[2]);
        // 双发轻微外张，单发保持正后方。
        if (engines.length > 1) plume.rotation.y = eng[0] > 0 ? 0.05 : -0.05;
        this.modelGroup.add(plume);
        this.flames.push(plume);
      });
      this.buildFlightEffects(ml);
    }

    // 翼尖蒸汽与音爆云都放在机体局部坐标系里，随 group 的统一视觉缩放
    // 一起放大，因此所有长度按 modelLen 的比例给出即可。
    buildFlightEffects(ml) {
      for (const obj of [...this.effectGroup.children]) {
        obj.traverse((n) => {
          if (n.material && n.material.dispose) n.material.dispose();
          if (n.isMesh && n.geometry && n.geometry !== plumeGeo) n.geometry.dispose();
        });
      }
      this.effectGroup.clear();
      this.vaporStreaks = [];
      this.vaporHalos = [];
      this.boomCloud = null;
      this.boomSmoke = [];
      this.boomPuffs = [];

      const geo = MODEL_GEOS[this.name];
      let halfSpan = ml * 0.46;
      let wingY = 0.0;
      let wingZ = ml * 0.24;
      if (geo && geo.body) {
        geo.body.computeBoundingBox();
        const bb = geo.body.boundingBox;
        halfSpan = Math.max(halfSpan, Math.abs(bb.min.x), Math.abs(bb.max.x));
        wingY = bb.min.y * 0.18;
        // 蒸汽起点取真实翼尖顶点的平均位置：后掠翼的翼尖在弦向靠后，
        // 按机体中点摆放会出现“蒸汽从机翼前方冒出”的错位。
        const pos = geo.body.attributes.position;
        const tipX = halfSpan * 0.95;
        let sy = 0.0, sz = 0.0, cnt = 0;
        for (let i = 0; i < pos.count; i += 1) {
          if (Math.abs(pos.getX(i)) >= tipX) {
            sy += pos.getY(i);
            sz += pos.getZ(i);
            cnt += 1;
          }
        }
        if (cnt > 0) {
          wingY = sy / cnt;
          wingZ = sz / cnt;
        }
      }
      const wingX = halfSpan * 0.92;

      for (const sgn of [-1.0, 1.0]) {
        // 每个翼尖两层：外层宽而淡，内层窄而实，叠出参考图里的白色
        // 冷凝雾带而不是一条发光线。
        for (let layer = 0; layer < 2; layer += 1) {
          const mat = makeVaporMaterial(0.35 + sgn * 0.11 + layer * 0.7, 0xf4f9ff);
          const streak = new THREE.Mesh(plumeGeo, mat);
          streak.position.set(
            sgn * wingX * (1.0 - layer * 0.10),
            wingY,
            wingZ + layer * ml * 0.035,
          );
          streak.frustumCulled = false;
          streak.visible = false;
          streak.userData.widthScale = layer === 0 ? 1.0 : 0.55;
          streak.userData.lengthScale = layer === 0 ? 1.0 : 0.78;
          this.effectGroup.add(streak);
          this.vaporStreaks.push(streak);
        }

        // 水平尾翼/后机身也拉出较窄的蒸汽，贴近参考图里翼尖与尾段
        // 同时起雾的样子。
        const tail = new THREE.Mesh(plumeGeo, makeVaporMaterial(0.9 + sgn * 0.2, 0xf4f9ff));
        tail.position.set(sgn * ml * 0.17, wingY * 0.35, ml * 0.30);
        tail.frustumCulled = false;
        tail.visible = false;
        tail.userData.widthScale = 0.62;
        tail.userData.lengthScale = 0.62;
        this.effectGroup.add(tail);
        this.vaporStreaks.push(tail);

        const halo = new THREE.Sprite(new THREE.SpriteMaterial({
          map: glowTex,
          color: 0xf4faff,
          transparent: true,
          blending: THREE.AdditiveBlending,
          depthWrite: false,
          opacity: 0.0,
        }));
        halo.position.set(sgn * wingX, wingY, wingZ + ml * 0.06);
        halo.scale.setScalar(ml * 0.085);
        halo.visible = false;
        this.effectGroup.add(halo);
        this.vaporHalos.push(halo);
      }

      // 音爆云：参照真实跨音速照片，白色凝结雾从翼尖、翼根和机尾向后
      // 喷出，并在机身后段聚成软云团；不再画几何锥壳，避免出现生硬的
      // 三角轮廓。
      const cloud = new THREE.Group();
      cloud.visible = false;
      this.boomSmoke = [];
      this.boomPuffs = [];
      const smokeDefs = [
        [wingX, wingY, wingZ + ml * 0.02, 1.55, 1.00, 0.15],
        [-wingX, wingY, wingZ + ml * 0.02, 1.55, 1.00, 0.85],
        [wingX * 0.52, wingY, wingZ + ml * 0.12, 1.25, 0.84, 0.45],
        [-wingX * 0.52, wingY, wingZ + ml * 0.12, 1.25, 0.84, 1.10],
        [ml * 0.12, -ml * 0.03, ml * 0.31, 1.45, 0.95, 0.30],
        [-ml * 0.12, -ml * 0.03, ml * 0.31, 1.45, 0.95, 1.35],
      ];
      for (const [x, y, z, wScale, lScale, seed] of smokeDefs) {
        const mat = makeVaporMaterial(seed, 0xf6fbff);
        const streak = new THREE.Mesh(plumeGeo, mat);
        streak.position.set(x, y, z);
        streak.frustumCulled = false;
        streak.visible = false;
        streak.userData.widthScale = wScale;
        streak.userData.lengthScale = lScale;
        cloud.add(streak);
        this.boomSmoke.push(streak);
      }
      for (let k = 0; k < 8; k += 1) {
        const mat = new THREE.SpriteMaterial({
          map: texBoomCloud,
          color: 0xf2f9ff,
          transparent: true,
          opacity: 0.0,
          depthWrite: false,
          blending: THREE.NormalBlending,
        });
        const sp = new THREE.Sprite(mat);
        const ang = (k / 8.0) * TAU + 0.35;
        const rad = ml * (0.07 + 0.10 * Math.random());
        sp.position.set(
          Math.cos(ang) * rad,
          Math.sin(ang) * rad - ml * 0.04,
          -ml * 0.16 + (k / 8.0) * ml * 0.58,
        );
        const baseW = ml * (0.30 + 0.12 * Math.random());
        const baseH = ml * (0.20 + 0.08 * Math.random());
        sp.scale.set(baseW, baseH, 1.0);
        sp.userData.baseW = baseW;
        sp.userData.baseH = baseH;
        sp.userData.alphaScale = 0.55 + Math.random() * 0.35;
        cloud.add(sp);
        this.boomPuffs.push(sp);
      }
      this.effectGroup.add(cloud);
      this.boomCloud = cloud;
    }

    updateFlightEffects(ac, t, k) {
      const ml = this.modelLen || REF_LENGTH;
      // 4.5G 触发后立刻给一个可见的起始强度（约 1/3），再随过载增大
      // 平滑增强；否则 4.6G 时 target 只有 0.07，几乎看不见。
      const over = HIGH_G_EFFECT && ac.alive
        ? clamp((ac.g - HIGH_G_ON) / 1.2, 0.0, 1.0)
        : 0.0;
      const target = over > 0.0 ? 0.34 + 0.66 * over : 0.0;
      this.highGLevel += (target - this.highGLevel) * 0.30;
      const g = this.highGLevel;

      for (const streak of this.vaporStreaks) {
        streak.visible = g > 0.01;
        if (!streak.visible) continue;
        const u = streak.material.uniforms;
        const wScale = streak.userData.widthScale || 1.0;
        const lScale = streak.userData.lengthScale || 1.0;
        u.uAlpha.value = 0.60 * g;
        u.uTime.value = t;
        u.uLen.value = ml * (0.55 + 0.28 * g) * lScale;
        u.uW0.value = (0.75 + 0.45 * g) * wScale;
        u.uW1.value = (2.00 + 1.00 * g) * wScale * (ml / REF_LENGTH);
        u.uScale.value = k;
      }
      for (const halo of this.vaporHalos) {
        halo.visible = g > 0.02;
        if (!halo.visible) continue;
        halo.material.opacity = 0.10 * g
          * (0.82 + 0.18 * Math.sin(t * 9.0 + halo.position.x * 2.0));
      }

      const mach = ac.alive ? ac.speed / speedOfSound(ac.alt) : 0.0;
      const crossed = this.machPrev !== null
        && this.machPrev < 1.0 && mach >= 1.0
        && (t - this.boomStart) > 1.5;
      if (crossed) this.boomStart = t;
      this.machPrev = mach;

      const age = t - this.boomStart;
      const p = clamp(age / 0.95, 0.0, 1.0);
      const active = ac.alive && age >= 0.0 && p < 1.0;
      const rise = clamp(age / 0.08, 0.0, 1.0);
      const env = rise * Math.pow(1.0 - p, 1.1);
      if (this.boomCloud) {
        this.boomCloud.visible = active;
        if (active) {
          for (const streak of this.boomSmoke) {
            const u = streak.material.uniforms;
            const wScale = streak.userData.widthScale || 1.0;
            const lScale = streak.userData.lengthScale || 1.0;
            u.uAlpha.value = 0.80 * env;
            u.uTime.value = t;
            u.uLen.value = ml * (0.38 + 0.34 * p) * lScale;
            u.uW0.value = 0.34 * wScale;
            u.uW1.value = (1.20 + 0.35 * p) * wScale * (ml / REF_LENGTH);
            u.uScale.value = k;
          }
          for (const sp of this.boomPuffs) {
            const s = 0.90 + 0.55 * p;
            sp.scale.set(
              sp.userData.baseW * s,
              sp.userData.baseH * s,
              1.0,
            );
            sp.material.opacity = 0.40 * env * sp.userData.alphaScale;
          }
        }
      }
    }

    update(ac, cameraDepth, simT) {
      this.group.position.copy(w2t(ac.pos));
      this.group.quaternion.copy(bodyQuat(ac.psi, ac.theta, ac.phi));
      const focal = (window.innerHeight / 2.0) / Math.tan((50.0 * Math.PI / 180.0) / 2.0);
      const targetPx = 46.0 * clamp(8000.0 / cameraDepth, 2.0, 3.0);
      const k = targetPx * cameraDepth / (focal * REF_LENGTH);
      this.group.scale.setScalar(k);
      this.group.visible = !!ac.alive;

      const t = simT || 0.0;
      if (this.effectLastT !== null && t < this.effectLastT) {
        this.highGLevel = 0.0;
        this.machPrev = null;
        this.boomStart = -1e9;
      }
      this.effectLastT = t;
      this.updateFlightEffects(ac, t, k);

      const level = ac.alive ? clamp(ac.flame, 0, 1) : 0;
      const ab = ac.alive ? clamp(ac.ab, 0, 1) : 0;
      const ml = this.modelLen || REF_LENGTH;
      const len = ml * (0.10 + 0.18 * level + 0.20 * ab);
      const phase = (performance.now() / 1000.0) % 8.0;
      const sf = ml / REF_LENGTH;
      for (const f of this.flames) {
        f.visible = ac.alive && level > 0.01;
        if (!f.visible) continue;
        f.userData.core.uTime.value = (phase / 8.0) * TAU;
        f.userData.halo.uTime.value = (phase / 8.0) * TAU;
        setPlumeState(f, level, ab, len, k, sf);
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

  // ---------- 航迹：渐变宽飘带 ----------
  // 复刻 render._draw_trail_ribbon：飘带宽度沿机体 y 轴展开、屏幕厚度约
  // 2×5px，横向中心亮、两边缘暗，并按航迹年龄整体变暗。飘带面片每个端点
  // 两个顶点，因此渐变完全由 GPU 插值，不会出现 8 条窄带的色阶。
  const trailVert = `
    attribute float aSide;
    attribute float aFade;
    attribute float aShade;
    varying float vSide;
    varying float vFade;
    varying float vShade;
    void main() {
      vSide = aSide;
      vFade = aFade;
      vShade = aShade;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `;
  const trailFrag = `
    precision highp float;
    uniform vec3 uColor;
    varying float vSide;
    varying float vFade;
    varying float vShade;
    void main() {
      float u = abs(vSide);
      // 整条飘带是“带”而不是“线”：中心亮度平缓下降，边缘再柔化成
      // 羽化边，加色混合下旧段自然变暗而不会压住背景。
      float core = exp(-pow(u * 1.7, 2.0));
      float soft = 1.0 - 0.45 * u * u;
      float a = clamp((0.30 + 0.70 * core) * soft * vFade, 0.0, 1.0);
      vec3 col = uColor * vShade * (0.80 + 0.40 * core);
      gl_FragColor = vec4(min(col, vec3(1.0)), a);
    }
  `;
  // 屏幕半宽（px）：飘带是发光丝带，目标约 12 px 总宽；边缘不再迅速衰减，
  // 让转弯、俯冲时都能看出“带”而不是一条细线。
  const TRAIL_HALF_PX = 6.0;
  const TRAIL_LIGHT = new THREE.Vector3(-0.35, -0.55, 0.76);
  const TRAIL_SUB = 3;

  // Catmull-Rom 平滑：后端每 60 m 才采一个点，硬转弯会形成生硬的折角。
  // 飘带中心线按样条细分，转弯处自然变圆，起点仍然精确落在机尾。
  function trailSpline(p0, p1, p2, p3, u) {
    const u2 = u * u, u3 = u2 * u;
    const f = (a, b, c, d) => 0.5 * (
      2.0 * b + (c - a) * u
      + (2.0 * a - 5.0 * b + 4.0 * c - d) * u2
      + (3.0 * b - a - 3.0 * c + d) * u3
    );
    return new THREE.Vector3(f(p0.x, p1.x, p2.x, p3.x),
                              f(p0.y, p1.y, p2.y, p3.y),
                              f(p0.z, p1.z, p2.z, p3.z));
  }

  const trailViews = [0, 1].map((team) => {
    const geo = new THREE.BufferGeometry();
    const mat = new THREE.ShaderMaterial({
      vertexShader: trailVert,
      fragmentShader: trailFrag,
      uniforms: {
        uColor: { value: new THREE.Color(rgbHex(mix([198, 202, 208], RGB.TEAM[team], 0.25))) },
      },
      transparent: true,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.frustumCulled = false;
    mesh.visible = false;
    scene.add(mesh);

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
      geo, mesh, team, segCount: 0,
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
        view.mesh.visible = false;
        view.arrowLine.visible = false;
        continue;
      }
      view.mesh.visible = true;
      view.arrowLine.visible = true;

      const now = snap.t;
      const axes = src.axes || [];
      const ac = snap.acs ? snap.acs[i] : null;
      const ml = (ac && ac.alive) ? (ac.model_len || REF_LENGTH) : REF_LENGTH;
      const eye = w2t(snap.camera.eye);
      const center = new THREE.Vector3(snap.camera.cx, snap.camera.cz, -snap.camera.cy);
      const forward = center.clone().sub(eye).normalize();
      const focal = (window.innerHeight / 2.0)
        / Math.tan((snap.fov || 50.0) * Math.PI / 360.0);
      // 后端每 60 m 才记录一个航迹采样点，末点可能在机体后方很远。
      // 历史点保持真实飞行轨迹，只把末尾追加的“当前点”放到视觉机尾；
      // 若把所有历史点都按切线后移，转弯时整条飘带会扭成穿过机体的折线。
      const centerAc = (ac && ac.alive) ? w2t(ac.pos) : new THREE.Vector3();
      const depthAc = Math.max(1.0, eye.distanceTo(centerAc));
      const targetPx = 46.0 * clamp(8000.0 / depthAc, 2.0, 3.0);
      const k = targetPx * depthAc / (focal * REF_LENGTH);
      const back = ml * 0.5 * k;
      const P = pts.map((p) => w2t(p));
      const axisW = axes.map((ax) => w2t(ax));
      const times = pts.map((p) => (p[3] === undefined ? now : p[3]));
      // 机体轴向单位向量（xb 指向机头），末点即使飞机已坠毁也用它做平移。
      const sp = ac ? Math.sin(ac.psi) : 0.0;
      const cp = ac ? Math.cos(ac.psi) : 1.0;
      const st = ac ? Math.sin(ac.theta) : 0.0;
      const ct = ac ? Math.cos(ac.theta) : 1.0;
      const sf = ac ? Math.sin(ac.phi) : 0.0;
      const cf = ac ? Math.cos(ac.phi) : 1.0;
      const xb = [ct * cp, ct * sp, st];
      const yb = [
        cf * sp + sf * st * cp,
        -cf * cp + sf * st * sp,
        -sf * ct,
      ];
      // 当前帧的飘带头部直接挂在当前视觉机尾上，从尾部伸出。
      if (ac && ac.alive) {
        P.push(w2t([
          ac.pos[0] - xb[0] * back,
          ac.pos[1] - xb[1] * back,
          ac.pos[2] - xb[2] * back,
        ]));
        axisW.push(w2t(yb));
        times.push(now);
      }
      // 整条飘带沿机体向后（机体轴 -xb）平移：距离取机体中心到发动机中心
      // 在机体轴向上的分量（模型局部 +z），按当前视觉缩放 k 换算成世界坐标，
      // 机体显示大小随镜头距离变化，所以每帧用当前的 k。
      const backShift = Math.max(0.0, aircraftViews[i].engineZ || 0.0) * k;
      if (backShift > 0.0) {
        for (const p of P) {
          p.x -= xb[0] * backShift;
          p.y -= xb[2] * backShift;
          p.z += xb[1] * backShift;
        }
      }
      // 飘带整体下移到喷口高度：模型局部 y 向下为负，机体中心到发动机
      // 中心的纵向间距同样按当前 k 换算。
      const drop = Math.max(0.0, -(aircraftViews[i].engineY || 0.0)) * k;
      if (drop > 0.0) {
        for (const p of P) p.y -= drop;
      }
      // Catmull-Rom 细分：60 m 一个后端采样点，直接连折线会在转弯处出现
      // 生硬折角。中心线按 TRAIL_SUB 细分，展向/时间同步插值；端点复制一份
      // 作切线控制点，首点仍精确落在机尾、末点仍是最后一个真实采样点。
      const S = [];
      const SA = [];
      const ST = [];
      const nP = P.length;
      for (let j = 0; j < nP - 1; j += 1) {
        const p0 = P[Math.max(0, j - 1)];
        const p1 = P[j];
        const p2 = P[j + 1];
        const p3 = P[Math.min(nP - 1, j + 2)];
        const a0 = axisW[j] || new THREE.Vector3(0, 1, 0);
        const a1 = axisW[j + 1] || a0;
        const t0 = times[j];
        const t1 = times[j + 1];
        for (let s = 0; s < TRAIL_SUB; s += 1) {
          const uu = s / TRAIL_SUB;
          S.push(trailSpline(p0, p1, p2, p3, uu));
          SA.push(a0.clone().lerp(a1, uu).normalize());
          ST.push(t0 + (t1 - t0) * uu);
        }
      }
      S.push(P[nP - 1].clone());
      SA.push(axisW[nP - 1] || new THREE.Vector3(0, 1, 0));
      ST.push(times[nP - 1]);

      const depths = S.map((p) => Math.max(1.0, p.clone().sub(eye).dot(forward)));

      const segs = S.length - 1;
      if (view.segCount !== segs) {
        const nv = segs * 4;
        const pos = new Float32Array(nv * 3);
        const side = new Float32Array(nv);
        const fade = new Float32Array(nv);
        const shade = new Float32Array(nv);
        const idx = new Uint16Array(segs * 6);
        for (let s = 0; s < segs; s += 1) {
          const b = s * 4;
          idx.set([b, b + 1, b + 2, b + 2, b + 1, b + 3], s * 6);
        }
        view.geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
        view.geo.setAttribute("aSide", new THREE.BufferAttribute(side, 1));
        view.geo.setAttribute("aFade", new THREE.BufferAttribute(fade, 1));
        view.geo.setAttribute("aShade", new THREE.BufferAttribute(shade, 1));
        view.geo.setIndex(new THREE.BufferAttribute(idx, 1));
        view.segCount = segs;
      }
      const pos = view.geo.attributes.position.array;
      const sideA = view.geo.attributes.aSide.array;
      const fadeA = view.geo.attributes.aFade.array;
      const shadeA = view.geo.attributes.aShade.array;

      for (let j = 1; j < S.length; j += 1) {
        const A = S[j - 1];
        const B = S[j];
        const b = (j - 1) * 4;
        const T = B.clone().sub(A);
        const axis = SA[j] || new THREE.Vector3(0, 1, 0);
        let span = null;
        if (T.lengthSq() > 1e-9) {
          T.normalize();
          span = axis.clone().addScaledVector(T, -axis.dot(T));
        }
        if (!span || span.lengthSq() <= 1e-9) {
          // 退化段收成零面积，避免在原点拖出一个错误四边形。
          for (let v = 0; v < 4; v += 1) {
            pos[(b + v) * 3] = A.x;
            pos[(b + v) * 3 + 1] = A.y;
            pos[(b + v) * 3 + 2] = A.z;
            sideA[b + v] = v % 2 === 0 ? -1.0 : 1.0;
            fadeA[b + v] = 0.62;
            shadeA[b + v] = 0.8;
          }
          continue;
        }
        span.normalize();

        // 法向 = 切线 × 展向，按固定天光做单面着色（倒扣时更亮）。
        const normal = T.clone().cross(span);
        const lit = normal.dot(TRAIL_LIGHT);
        const shade = 0.80 + 0.20 * Math.max(0.0, lit);

        const age = now - ST[j];
        const life = Math.max(0.0, 1.0 - age / TRAIL_FADE);
        const fade = 0.62 + 0.38 * life;
        const half = TRAIL_HALF_PX * 0.5
          * (depths[j - 1] + depths[j]) / focal;

        const verts = [
          A.clone().addScaledVector(span, -half),
          A.clone().addScaledVector(span, half),
          B.clone().addScaledVector(span, -half),
          B.clone().addScaledVector(span, half),
        ];
        for (let v = 0; v < 4; v += 1) {
          pos[(b + v) * 3] = verts[v].x;
          pos[(b + v) * 3 + 1] = verts[v].y;
          pos[(b + v) * 3 + 2] = verts[v].z;
          sideA[b + v] = v % 2 === 0 ? -1.0 : 1.0;
          fadeA[b + v] = fade;
          shadeA[b + v] = shade;
        }
      }
      for (const name of ["position", "aSide", "aFade", "aShade"]) {
        view.geo.attributes[name].needsUpdate = true;
      }

      // 方向箭头：与 render 一致，每 ARROW_EVERY 个采样点放一个人字形箭标，
      // 朝向沿该段航迹，尺寸按相机距离换算成约 12 px，远近观感稳定。
      const arrowBase = mix([236, 238, 242], RGB.TEAM[i], 0.18);
      const camDist = (snap.camera && snap.camera.dist) || 6000.0;
      const size = clamp(12.0 * camDist / focal, 18.0, 420.0);
      const arrowPos = [];
      const arrowCol = [];
      for (let j = ARROW_EVERY; j < pts.length; j += ARROW_EVERY) {
        const A = P[j - 1];
        const B = P[j];
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
  // 尾焰羽流池单独挂一个组，避免每帧清空导弹节点时把着色器材质一起释放。
  const missileFlameGroup = new THREE.Group();
  scene.add(missileFlameGroup);
  const missileFlames = [];

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

  // ---------- 干扰物动效：箔条云 / 热诱弹 ----------
  // 快照只给存活干扰物的位置和类型；前端按位置做帧间匹配，自行维护
  // 每个诱饵的出生时间、膨胀和拖尾。干扰物显示时锚定在机体的“视觉
  // 机尾”上（机体被放大渲染，机尾离机位中心约半个屏幕机长），
  // 否则诱饵会从模型内部出现，看起来像贴在图中央。
  const cmGroup = new THREE.Group();
  scene.add(cmGroup);

  const texChaffCloud = makeSpriteTexture((ctx, s) => {
    const rnd = camoRng(0x51a7);
    ctx.clearRect(0, 0, s, s);
    for (let i = 0; i < 16; i += 1) {
      const x = s * 0.5 + (rnd() - 0.5) * s * 0.46;
      const y = s * 0.5 + (rnd() - 0.5) * s * 0.46;
      const r = s * (0.12 + rnd() * 0.20);
      const grd = ctx.createRadialGradient(x, y, 0, x, y, r);
      grd.addColorStop(0, `rgba(255,255,255,${0.14 + rnd() * 0.12})`);
      grd.addColorStop(0.55, `rgba(232,238,246,${0.06 + rnd() * 0.06})`);
      grd.addColorStop(1, "rgba(210,220,232,0)");
      ctx.fillStyle = grd;
      ctx.beginPath();
      ctx.arc(x, y, r, 0, TAU);
      ctx.fill();
    }
  }, 128);

  const texFlareCore = makeSpriteTexture((ctx, s) => {
    const grd = ctx.createRadialGradient(s * 0.5, s * 0.5, 0, s * 0.5, s * 0.5, s * 0.5);
    grd.addColorStop(0.00, "rgba(255,255,250,1.00)");
    grd.addColorStop(0.14, "rgba(255,246,196,0.95)");
    grd.addColorStop(0.36, "rgba(255,176,64,0.62)");
    grd.addColorStop(0.66, "rgba(255,92,18,0.22)");
    grd.addColorStop(1.00, "rgba(255,50,0,0)");
    ctx.fillStyle = grd;
    ctx.fillRect(0, 0, s, s);
  }, 128);

  const CM_TRAIL_MAX = 48;
  const CM_MATCH_R2 = 60.0 * 60.0;
  const cmEffects = [];
  let cmLastT = -1.0;

  // 把真实干扰物位置映射到机体的视觉机尾坐标系。
  function cmDisplayPos(eff, snap, eye, focal) {
    const ac = snap.acs ? snap.acs[eff.owner] : null;
    if (!ac || !ac.alive) return eff.raw.clone();
    const center = w2t(ac.pos);
    const depth = Math.max(1.0, eye.distanceTo(center));
    const targetPx = 46.0 * clamp(8000.0 / depth, 2.0, 3.0);
    const k = targetPx * depth / (focal * REF_LENGTH);
    const ml = ac.model_len || REF_LENGTH;
    const sp = Math.sin(ac.psi), cp = Math.cos(ac.psi);
    const st = Math.sin(ac.theta), ct = Math.cos(ac.theta);
    const xb = [ct * cp, ct * sp, st];
    const back = ml * 0.5 * k;
    const tail = new THREE.Vector3(
      center.x - xb[0] * back,
      center.y - xb[2] * back,
      center.z + xb[1] * back,
    );
    return tail.add(eff.raw.clone().sub(center));
  }

  function baseCounterEffect(kind, live, born) {
    return {
      kind,
      owner: live.owner,
      raw: live.pos.clone(),
      born,
      last: born,
      seed: Math.random() * 1000.0,
      objects: [],
      materials: [],
      puffMat: null,
      glintMat: null,
      coreMat: null,
      haloMat: null,
      smokeMat: null,
      lineGeo: null,
      linePos: null,
      lineCol: null,
      history: [],
    };
  }

  function attachCounterObject(eff, obj) {
    cmGroup.add(obj);
    eff.objects.push(obj);
    if (obj.material && !eff.materials.includes(obj.material)) {
      eff.materials.push(obj.material);
    }
  }

  function spawnChaffEffect(live, born) {
    const eff = baseCounterEffect("chaff", live, born);
    const mat = new THREE.SpriteMaterial({
      map: texChaffCloud, color: 0xd2d8e0, transparent: true,
      opacity: 0, depthWrite: false, blending: THREE.NormalBlending,
    });
    eff.puffMat = mat;
    eff.materials.push(mat);
    eff.puffs = [];
    for (let k = 0; k < 4; k += 1) {
      const sp = new THREE.Sprite(mat);
      const ang = Math.random() * TAU;
      const y = (Math.random() - 0.5) * 0.55;
      const radial = Math.sqrt(Math.max(0.0, 1.0 - y * y));
      sp.visible = false;
      attachCounterObject(eff, sp);
      eff.puffs.push({
        sprite: sp,
        dir: new THREE.Vector3(Math.cos(ang) * radial, y, Math.sin(ang) * radial),
        off: 0.12 + Math.random() * 0.50,
        size: 0.42 + Math.random() * 0.34,
      });
    }
    const gmat = new THREE.SpriteMaterial({
      map: texStar, color: 0xeef4fb, transparent: true,
      opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    eff.glintMat = gmat;
    eff.materials.push(gmat);
    eff.glints = [];
    for (let k = 0; k < 3; k += 1) {
      const sp = new THREE.Sprite(gmat);
      const ang = Math.random() * TAU;
      const y = (Math.random() - 0.5) * 0.6;
      const radial = Math.sqrt(Math.max(0.0, 1.0 - y * y));
      sp.visible = false;
      attachCounterObject(eff, sp);
      eff.glints.push({
        sprite: sp,
        dir: new THREE.Vector3(Math.cos(ang) * radial, y, Math.sin(ang) * radial),
        phase: Math.random() * TAU,
      });
    }
    return eff;
  }

  function spawnFlareEffect(live, born) {
    const eff = baseCounterEffect("flare", live, born);
    const coreMat = new THREE.SpriteMaterial({
      map: texFlareCore, color: 0xffffff, transparent: true,
      opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    const haloMat = new THREE.SpriteMaterial({
      map: texFlareCore, color: 0xff8c30, transparent: true,
      opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending,
    });
    const smokeMat = new THREE.SpriteMaterial({
      map: texChaffCloud, color: 0x8e8a84, transparent: true,
      opacity: 0, depthWrite: false, blending: THREE.NormalBlending,
    });
    eff.coreMat = coreMat;
    eff.haloMat = haloMat;
    eff.smokeMat = smokeMat;
    eff.materials.push(coreMat, haloMat, smokeMat);
    eff.core = new THREE.Sprite(coreMat);
    eff.halo = new THREE.Sprite(haloMat);
    eff.smoke = new THREE.Sprite(smokeMat);
    for (const sp of [eff.smoke, eff.halo, eff.core]) {
      sp.visible = false;
      attachCounterObject(eff, sp);
    }

    // 燃烧拖尾：位置来自快照历史，颜色从新到旧变暗。
    eff.linePos = new Float32Array(CM_TRAIL_MAX * 3);
    eff.lineCol = new Float32Array(CM_TRAIL_MAX * 3);
    eff.lineGeo = new THREE.BufferGeometry();
    eff.lineGeo.setAttribute("position", new THREE.BufferAttribute(eff.linePos, 3));
    eff.lineGeo.setAttribute("color", new THREE.BufferAttribute(eff.lineCol, 3));
    eff.lineGeo.setDrawRange(0, 0);
    const line = new THREE.Line(eff.lineGeo, new THREE.LineBasicMaterial({
      vertexColors: true, transparent: true, opacity: 0.85,
      blending: THREE.AdditiveBlending, depthWrite: false,
    }));
    line.frustumCulled = false;
    cmGroup.add(line);
    eff.objects.push(line);
    eff.materials.push(line.material);
    return eff;
  }

  function disposeCounterEffect(eff) {
    for (const o of eff.objects) cmGroup.remove(o);
    for (const m of eff.materials) m.dispose();
    if (eff.lineGeo) eff.lineGeo.dispose();
  }

  function clearCounterEffects() {
    for (const eff of cmEffects) disposeCounterEffect(eff);
    cmEffects.length = 0;
    cmLastT = -1.0;
  }

  function updateChaffEffect(eff, age, snap, eye, focal) {
    const center = cmDisplayPos(eff, snap, eye, focal);
    const pxWorld = Math.max(1.0, eye.distanceTo(center)) / focal;
    const p = clamp(age / CHAFF_LIFE, 0, 1);
    const fadeIn = clamp(age / 0.16, 0, 1);
    const fadeOut = clamp((1.0 - p) / 0.30, 0, 1);
    eff.puffMat.opacity = 0.38 * fadeIn * fadeOut;
    const cloudPx = 16.0 + 68.0 * (1.0 - Math.exp(-age * 1.5));
    for (const puff of eff.puffs) {
      puff.sprite.visible = true;
      puff.sprite.position.copy(center).addScaledVector(puff.dir, cloudPx * 0.45 * pxWorld * puff.off);
      puff.sprite.scale.setScalar(cloudPx * puff.size * pxWorld);
    }
    eff.glintMat.opacity = 0.62 * fadeIn * fadeOut;
    for (let k = 0; k < eff.glints.length; k += 1) {
      const gl = eff.glints[k];
      gl.sprite.visible = true;
      gl.sprite.position.copy(center).addScaledVector(gl.dir, cloudPx * 0.35 * pxWorld);
      const glint = 3.0 + 2.0 * Math.abs(Math.sin(age * 11.0 + gl.phase));
      gl.sprite.scale.setScalar(glint * pxWorld * 2.0);
    }
  }

  function updateFlareEffect(eff, age, snap, eye, focal) {
    const center = cmDisplayPos(eff, snap, eye, focal);
    const pxWorld = Math.max(1.0, eye.distanceTo(center)) / focal;
    const p = clamp(age / FLARE_LIFE, 0, 1);
    const fadeOut = clamp((1.0 - p) / 0.32, 0, 1);
    const flash = Math.min(1.0, age / 0.10);
    const flicker = 0.86 + 0.14 * Math.sin(age * 26.0 + eff.seed);

    eff.core.visible = true;
    eff.core.position.copy(center);
    eff.core.scale.setScalar((10.0 + 8.0 * flash) * flicker * pxWorld);
    eff.coreMat.opacity = 0.95 * fadeOut;

    eff.halo.visible = true;
    eff.halo.position.copy(center);
    eff.halo.scale.setScalar((30.0 + 34.0 * Math.min(1.0, age / 0.5)) * flicker * pxWorld);
    eff.haloMat.opacity = 0.40 * fadeOut;

    eff.smoke.visible = true;
    eff.smoke.position.copy(center);
    eff.smoke.scale.setScalar((26.0 + 36.0 * Math.min(1.0, age / 1.2)) * pxWorld);
    eff.smokeMat.opacity = 0.16 * fadeOut * (1.0 - 0.4 * p);

    // 记录拖尾点；只在移动足够远或时间超过 60 ms 时采样，最多 48 点。
    const hist = eff.history;
    const last = hist.length ? hist[hist.length - 1] : null;
    if (!last || last.p.distanceTo(center) > 6.0 * pxWorld || snap.t - last.t > 0.06) {
      hist.push({ p: center.clone(), t: snap.t });
    }
    while (hist.length && snap.t - hist[0].t > 1.5) hist.shift();
    if (hist.length > CM_TRAIL_MAX) hist.splice(0, hist.length - CM_TRAIL_MAX);

    const n = hist.length;
    for (let k = 0; k < n; k += 1) {
      const hp = hist[k];
      eff.linePos[k * 3] = hp.p.x;
      eff.linePos[k * 3 + 1] = hp.p.y;
      eff.linePos[k * 3 + 2] = hp.p.z;
      const f = Math.max(0.16, 1.0 - (snap.t - hp.t) / 1.5);
      eff.lineCol[k * 3] = 1.00 * f;
      eff.lineCol[k * 3 + 1] = 0.60 * f;
      eff.lineCol[k * 3 + 2] = 0.18 * f;
    }
    eff.lineGeo.setDrawRange(0, n);
    eff.lineGeo.attributes.position.needsUpdate = true;
    eff.lineGeo.attributes.color.needsUpdate = true;
  }

  function syncCounterEffects(snap) {
    if (cmLastT >= 0.0 && snap.t < cmLastT) clearCounterEffects();
    cmLastT = snap.t;

    const live = (snap.counters || []).map((cm) => ({
      kind: cm.kind,
      owner: cm.owner,
      pos: w2t(cm.pos),
    }));
    const used = new Array(live.length).fill(false);
    let write = 0;
    for (const eff of cmEffects) {
      let best = -1;
      let bestD = CM_MATCH_R2;
      for (let k = 0; k < live.length; k += 1) {
        if (used[k] || live[k].kind !== eff.kind) continue;
        const d = eff.raw.distanceToSquared(live[k].pos);
        if (d < bestD) { bestD = d; best = k; }
      }
      if (best >= 0) {
        used[best] = true;
        eff.raw.copy(live[best].pos);
        eff.owner = live[best].owner;
        eff.last = snap.t;
        cmEffects[write] = eff;
        write += 1;
      } else if (snap.t - eff.last <= 0.30) {
        cmEffects[write] = eff;
        write += 1;
      } else {
        disposeCounterEffect(eff);
      }
    }
    cmEffects.length = write;
    for (let k = 0; k < live.length; k += 1) {
      if (used[k]) continue;
      const eff = live[k].kind === "chaff"
        ? spawnChaffEffect(live[k], snap.t)
        : spawnFlareEffect(live[k], snap.t);
      cmEffects.push(eff);
    }

    const eye = w2t(snap.camera.eye);
    const focal = (window.innerHeight / 2.0)
      / Math.tan((snap.fov || 50.0) * Math.PI / 360.0);
    for (const eff of cmEffects) {
      const age = Math.max(0.0, snap.t - eff.born);
      if (eff.kind === "chaff") updateChaffEffect(eff, age, snap, eye, focal);
      else updateFlareEffect(eff, age, snap, eye, focal);
    }
  }

  function missileDir(m) {
    const d = new THREE.Vector3(m.dir[0], m.dir[2], -m.dir[1]);
    if (d.lengthSq() < 1e-8) d.set(0, 0, 1);
    return d.normalize();
  }

  function updateMissiles(snap) {
    while (missileGroup.children.length) {
      const o = missileGroup.children.pop();
      if (o.geometry && !missileShared.has(o.geometry)) o.geometry.dispose();
      if (o.material && !missileShared.has(o.material)) o.material.dispose();
    }
    const ms = snap.missiles || [];
    for (const f of missileFlames) f.visible = false;
    const focal = (window.innerHeight / 2.0)
      / Math.tan(camera.fov * Math.PI / 360.0);

    ms.forEach((m, idx) => {
      const p = w2t(m.pos);
      const d = missileDir(m);
      const geo = MISSILE_GEOS[m.name];
      // 导弹尺寸直接复用载机的可视化缩放系数 k：机体为看得清会被放大到
      // 92–138 px，导弹用同一个 k 放大，弹长/机长就严格等于真实比例。
      // k 的算法与 AircraftView.update 完全一致（同一视深基准）。
      const eye = (snap.camera && snap.camera.eye) ? w2t(snap.camera.eye) : camera.position;
      const ac = snap.acs ? snap.acs[m.owner] : null;
      const refPos = ac ? w2t(ac.pos) : p;
      const depth = Math.max(1.0, eye.distanceTo(refPos));
      const targetPx = 46.0 * clamp(8000.0 / depth, 2.0, 3.0);
      const s = targetPx * depth / (focal * REF_LENGTH);
      if (geo) {
        // 模型按真实弹长导出：弹头在局部原点、弹尾朝 +Z，直接把弹头放在
        // 快照位置，弹体沿 -d 向后延伸。
        const mat = m.guided ? missileMaterial(m.owner & 1, m.name) : missileDeadMat;
        const mesh = new THREE.Mesh(geo.body, mat);
        mesh.position.copy(p);
        mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, -1), d);
        mesh.scale.setScalar(s);
        missileGroup.add(mesh);
      } else {
        // 模型未就绪时退化为一段短柱，保证弹道仍然可见。
        const mesh = new THREE.Mesh(
          new THREE.CylinderGeometry(0.10, 0.10, m.length, 5),
          new THREE.MeshBasicMaterial({ color: 0xd8dade })
        );
        mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), d);
        mesh.position.copy(p).addScaledVector(d, -m.length * 0.5);
        mesh.scale.setScalar(s);
        missileGroup.add(mesh);
      }
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
      // 点火阶段：弹尾挂一支小羽流，燃料烧完即熄灭。
      if (m.ignited && m.fuel > 0.0) {
        let flame = missileFlames[idx];
        if (!flame) {
          flame = makeMissilePlume(2.1 + idx * 0.9);
          missileFlameGroup.add(flame);
          missileFlames[idx] = flame;
        }
        flame.visible = true;
        flame.position.copy(p).addScaledVector(d, -m.length * s);
        flame.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, -1), d);
        flame.scale.setScalar(s);
        const phase = (performance.now() / 1000.0 + idx * 0.7) % 8.0;
        // 羽流长度按真实弹长给，scale 再按屏幕尺度放大。
        setMissilePlumeState(flame, m.length * 0.75, 1.0, (phase / 8.0) * TAU, s);
      }
    });
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
    syncCounterEffects(snap);
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
    const glCv = document.createElement("canvas");
    glCv.className = "att-canvas att-gl";
    const cv = document.createElement("canvas");
    cv.className = "att-canvas";
    const rowThr = document.createElement("div");
    rowThr.className = "att-bar-row thr";
    rowThr.innerHTML = '<span class="att-label">THR</span><span class="att-bar thr"><i></i></span><span class="att-value"></span>';
    const rowFuel = document.createElement("div");
    rowFuel.className = "att-bar-row fuel";
    rowFuel.innerHTML = '<span class="att-label">FUEL</span><span class="att-bar fuel"><i></i></span><span class="att-value"></span>';
    half.append(name, glCv, cv, rowThr, rowFuel);
    attitudeEl.appendChild(half);
    return {
      half, name, cv, glCanvas: glCv, gl: null,
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
      const sw = rw.w - 172;
      const sh = rw.h - 38;
      const padBottom = Math.max(14, sh * 0.30);
      drawSilhouette(this.silCanvas, ac.name, ac.team, sw, sh, padBottom);
      drawWeaponMissiles(this.silCanvas, sw, sh, padBottom);
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

  const SIL_IMAGES = {};
  for (const [key, file] of Object.entries({
    "SU-57": "su57_top.png", "F-16V": "f16_top.png",
  })) {
    const img = new Image();
    img.src = `/assets/silhouette/${file}`;
    SIL_IMAGES[key] = img;
  }

  function drawSilhouette(cv, name, team, w, h, padBottom = 0) {
    if (w <= 2 || h <= 2) return;
    const ctx = fitCanvas(cv, w, h);
    // padBottom 给机尾下方的两枚小弹留位置，机体按剩余高度居中。
    const bh = Math.max(2, h - padBottom);
    const img = SIL_IMAGES[name];
    if (img && img.complete && img.naturalWidth) {
      const s = Math.min((w - 8) / img.naturalWidth, (bh - 8) / img.naturalHeight);
      const dw = img.naturalWidth * s;
      const dh = img.naturalHeight * s;
      const dx = (w - dw) / 2.0;
      const dy = (bh - dh) / 2.0;
      ctx.globalAlpha = 0.92;
      ctx.drawImage(img, dx, dy, dw, dh);
      // 透明度只作用于轮廓像素：用队伍色轻轻染色，与 2D 版观感一致。
      ctx.globalCompositeOperation = "source-atop";
      ctx.globalAlpha = 0.16;
      ctx.fillStyle = C.team[team];
      ctx.fillRect(dx, dy, dw, dh);
      ctx.globalCompositeOperation = "source-over";
      ctx.globalAlpha = 1.0;
      return;
    }
    const pts = silhouettePoints(name);
    let minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
    for (const p of pts) {
      minX = Math.min(minX, p[0]); maxX = Math.max(maxX, p[0]);
      minY = Math.min(minY, p[1]); maxY = Math.max(maxY, p[1]);
    }
    const sx = (w - 12) / Math.max(1e-6, maxY - minY);
    const sy = (bh - 8) / Math.max(1e-6, maxX - minX);
    const scale = Math.min(sx, sy);
    const cx = w / 2;
    const cy = bh / 2;
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

  // 武器面板的导弹轮廓（参考图 2）：左右两枚大弹、机尾下方两枚小弹，
  // 绿色霓虹描边；与机体轮廓一起构成“挂载示意”，不抢文字读数。
  function missileOutlinePath(ctx, cx, cy, len, wid) {
    const nose = cy - len * 0.5;
    const tail = cy + len * 0.5;
    const hw = wid * 0.5;
    const shoulder = nose + len * 0.24;
    ctx.beginPath();
    ctx.moveTo(cx, nose);
    ctx.quadraticCurveTo(cx - hw, shoulder - len * 0.12, cx - hw, shoulder);
    ctx.lineTo(cx - hw, tail - len * 0.16);
    ctx.lineTo(cx - hw * 1.9, tail - len * 0.05);
    ctx.lineTo(cx - hw, tail - len * 0.02);
    ctx.lineTo(cx - hw * 0.6, tail);
    ctx.lineTo(cx + hw * 0.6, tail);
    ctx.lineTo(cx + hw, tail - len * 0.02);
    ctx.lineTo(cx + hw * 1.9, tail - len * 0.05);
    ctx.lineTo(cx + hw, tail - len * 0.16);
    ctx.lineTo(cx + hw, shoulder);
    ctx.quadraticCurveTo(cx + hw, shoulder - len * 0.12, cx, nose);
    ctx.closePath();
  }

  function drawWeaponMissiles(cv, w, h, padBottom) {
    if (w <= 8 || h <= 8) return;
    const ctx = cv.getContext("2d");
    const bh = Math.max(2, h - padBottom);
    ctx.save();
    ctx.lineWidth = 1.1;
    ctx.strokeStyle = "rgba(152,246,184,0.92)";
    ctx.fillStyle = "rgba(118,232,162,0.08)";
    ctx.shadowColor = "rgba(110,240,160,0.85)";
    ctx.shadowBlur = 5;
    // 两侧大弹：长度约占机体高度的 0.9，位于机体轮廓外侧。
    const bigLen = bh * 0.86;
    const bigWid = Math.max(2.8, bigLen * 0.11);
    for (const x of [w * 0.12, w * 0.88]) {
      missileOutlinePath(ctx, x, bh * 0.46, bigLen, bigWid);
      ctx.fill();
      ctx.stroke();
    }
    // 机尾下方两枚小弹：靠近中轴，弹头朝上。
    const smallLen = bh * 0.34;
    const smallWid = Math.max(1.8, smallLen * 0.15);
    for (const x of [w * 0.44, w * 0.56]) {
      missileOutlinePath(ctx, x, bh + padBottom * 0.5, smallLen, smallWid);
      ctx.fill();
      ctx.stroke();
    }
    ctx.restore();
  }

  // ---------- 顶部姿态框 ----------
  const ATT_VIEW = [0.85, -0.98, 0.55];
  // 与 render.py 的 _attitude_basis 完全一致：right = cross((0,0,1), view)。
  // 这里若写成 cross(view, z) 会让 up 反向，姿态框里的飞机整体上下颠倒。
  const ATT_RIGHT = [-ATT_VIEW[1], ATT_VIEW[0], 0];
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

  // 姿态框里的飞机用 three.js 独立小场景渲染同一套 OBJ 模型：
  // 相机固定沿 ATT_VIEW 方向，模型按机体姿态旋转，投影关系与原 2D 版一致。
  function ensureAttitude3D(part) {
    if (part.gl) return part.gl;
    const renderer = new THREE.WebGLRenderer({
      canvas: part.glCanvas, antialias: true, alpha: true,
    });
    renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    renderer.setClearColor(0x000000, 0);
    const sc = new THREE.Scene();
    // 姿态框要看清机体明暗，光照比主场景克制：环境光只托底，主光偏侧上。
    sc.add(new THREE.AmbientLight(0xffffff, 0.25));
    const key = new THREE.DirectionalLight(0xdfe8ff, 0.58);
    key.position.copy(w2t(ATT_VIEW_N)).multiplyScalar(200.0);
    sc.add(key);
    const fill = new THREE.DirectionalLight(0xffd8c0, 0.18);
    fill.position.copy(w2t(ATT_UP_N)).multiplyScalar(200.0);
    sc.add(fill);
    const cam = new THREE.PerspectiveCamera(30.0, 1.0, 0.1, 1000.0);
    const group = new THREE.Group();
    sc.add(group);
    const gl = {
      renderer, scene: sc, camera: cam, group,
      name: "", modelLen: REF_LENGTH, engineCount: -1,
      body: null, canopy: null, flames: [],
    };
    part.gl = gl;
    return gl;
  }

  function buildAttitudeModel(gl, ac) {
    gl.group.clear();
    gl.flames = [];
    gl.body = null;
    gl.canopy = null;
    const geo = MODEL_GEOS[ac.name];
    const ml = ac.model_len || REF_LENGTH;
    if (geo) {
      gl.body = new THREE.Mesh(geo.body, ac.alive ? camoMaterial(ac.team, ac.name) : deadBodyMat);
      gl.canopy = new THREE.Mesh(geo.canopy, canopyMat);
      gl.canopy.visible = !!ac.alive;
      gl.group.add(gl.body, gl.canopy);
    }
    const engines = ac.engines || [];
    engines.forEach((eng, idx) => {
      const plume = makePlume(idx === 0 ? 0.35 : 1.75, ml);
      plume.position.set(eng[0], eng[1], eng[2]);
      if (engines.length > 1) plume.rotation.y = eng[0] > 0 ? 0.05 : -0.05;
      gl.group.add(plume);
      gl.flames.push(plume);
    });
    gl.name = ac.name;
    gl.modelLen = ml;
    gl.engineCount = engines.length;
  }

  function renderAttitude3D(part, ac, w, h) {
    if (!MODEL_GEOS[ac.name]) return;
    const gl = ensureAttitude3D(part);
    gl.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    gl.renderer.setSize(w, h, false);
    gl.camera.aspect = w / Math.max(1.0, h);
    const ml = ac.model_len || REF_LENGTH;
    // 让机身长度投影到约 100 px，与原 2D 投影尺度一致。
    const dist = (h / 2.0) / (Math.tan((30.0 * Math.PI / 180.0) / 2.0) * (100.0 / ml));
    gl.camera.position.copy(w2t(ATT_VIEW_N)).multiplyScalar(dist);
    gl.camera.up.copy(w2t(ATT_UP_N));
    gl.camera.lookAt(0, 0, 0);
    gl.camera.updateProjectionMatrix();

    const engines = ac.engines || [];
    if (gl.name !== ac.name || gl.modelLen !== ml || gl.engineCount !== engines.length) {
      buildAttitudeModel(gl, ac);
    }
    if (gl.body) gl.body.material = ac.alive ? camoMaterial(ac.team, ac.name) : deadBodyMat;
    if (gl.canopy) gl.canopy.visible = !!ac.alive;
    gl.group.quaternion.copy(bodyQuat(ac.psi, ac.theta, ac.phi));

    const level = ac.alive ? clamp(ac.flame, 0, 1) : 0;
    const ab = ac.alive ? clamp(ac.ab, 0, 1) : 0;
    const len = ml * (0.10 + 0.18 * level + 0.20 * ab);
    const sf = ml / REF_LENGTH;
    const phase = (performance.now() / 1000.0) % 8.0;
    for (const plume of gl.flames) {
      plume.visible = ac.alive && level > 0.01;
      if (!plume.visible) continue;
      plume.userData.core.uTime.value = (phase / 8.0) * TAU;
      plume.userData.halo.uTime.value = (phase / 8.0) * TAU;
      setPlumeState(plume, level, ab, len, 1.0, sf);
    }
    gl.renderer.render(gl.scene, gl.camera);
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

    renderAttitude3D(part, ac, w, h);
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

  // ---------- 屏幕空间标签：导弹名称 / 机体旁的战术与机动 ----------
  const labelsCanvas = document.getElementById("labels");
  const labelsCtx = labelsCanvas.getContext("2d");
  let labelsW = 0;
  let labelsH = 0;
  let labelsDpr = 1;

  function resizeLabels() {
    labelsDpr = Math.min(2, window.devicePixelRatio || 1);
    labelsW = window.innerWidth;
    labelsH = window.innerHeight;
    labelsCanvas.width = Math.round(labelsW * labelsDpr);
    labelsCanvas.height = Math.round(labelsH * labelsDpr);
  }
  resizeLabels();

  function projectScreen(p) {
    const v = p.clone().project(camera);
    if (v.z < -1.0 || v.z > 1.0) return null;
    if (v.x < -1.05 || v.x > 1.05 || v.y < -1.05 || v.y > 1.05) return null;
    return { x: (v.x + 1.0) * 0.5 * labelsW, y: (1.0 - v.y) * 0.5 * labelsH };
  }

  function drawTag(str, sx, sy, color, size, flip) {
    const ctx = labelsCtx;
    ctx.font = `${size}px ${FONT}`;
    const w = ctx.measureText(str).width;
    const x = flip ? sx - w : sx;
    // 先画一层深色底影，任何背景亮度下都读得清。
    ctx.fillStyle = "rgba(6,10,14,0.88)";
    ctx.fillText(str, x + 1, sy + 1);
    ctx.fillStyle = color;
    ctx.fillText(str, x, sy);
  }

  function drawLabels() {
    labelsCtx.setTransform(labelsDpr, 0, 0, labelsDpr, 0, 0);
    labelsCtx.clearRect(0, 0, labelsW, labelsH);
    if (!snapshot) return;

    for (let i = 0; i < 2; i += 1) {
      const ac = snapshot.acs ? snapshot.acs[i] : null;
      if (!ac || !ac.alive) continue;
      const s = projectScreen(w2t(ac.pos));
      if (!s) continue;
      const hud = ac.hud || {};
      const parts = [hud.tactic, hud.primitive].filter((v) => v);
      if (!parts.length) continue;
      // 标签贴在机体右侧；靠右边缘时自动翻到左侧，避免被窗口裁掉。
      const flip = s.x > labelsW - 230;
      drawTag(parts.join(" · "), s.x + (flip ? -30 : 30), s.y - 24, C.team[i], 13, flip);
    }

    for (const m of snapshot.missiles || []) {
      const s = projectScreen(w2t(m.pos));
      if (!s) continue;
      drawTag(m.name, s.x + 9, s.y - 6, m.guided ? C.text : C.sub, 12, false);
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
      part.glCanvas.style.width = `${halfW}px`;
      part.glCanvas.style.height = "134px";
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
      view.setModel(ac.model, ac.engines || [], ac.model_len);
      const depth = Math.max(1.0, cameraPos.distanceTo(w2t(ac.pos)));
      view.update(ac, depth, snap.t);
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
    if (CAPTURE) return;  // 固定分辨率采集时不受面板缩放影响
    renderer.setSize(window.innerWidth, window.innerHeight, false);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    resizeLabels();
    layoutAll();
    if (snapshot) updateHud(snapshot);
  });

  // ---------- SSE ----------
  const stream = new EventSource("/api/stream");
  stream.onmessage = (ev) => {
    try {
      snapshot = JSON.parse(ev.data);
      capReady = true;
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

  // ---------- 1080p 采集 ----------
  // 每帧渲染完成后按目标帧率把 WebGL 画面拷贝到离屏 2D 画布，
  // 以 PNG 二进制 POST 给服务端；帧序列由服务端 ffmpeg 统一编码成 MP4。
  // 采集只包含 3D 场景（HUD 是 DOM，不参与离屏合成）。
  let capReady = false;
  const capState = { t0: 0, next: 0, i: 0, done: false };

  function captureTick() {
    if (!capReady || capState.done) return;
    const now = performance.now();
    if (capState.t0 === 0) {
      capState.t0 = now + CAPTURE.delay * 1000;
      capState.next = capState.t0;
      if (CAPTURE.delay > 0) return;
    }
    if (now < capState.next) return;
    capState.next += 1000.0 / CAPTURE.fps;
    if (capState.next < now - 500) capState.next = now;  // 掉帧时不无限追赶

    const off = document.createElement("canvas");
    off.width = renderer.domElement.width;
    off.height = renderer.domElement.height;
    off.getContext("2d").drawImage(renderer.domElement, 0, 0);
    const idx = String(capState.i).padStart(4, "0");
    off.toBlob((blob) => {
      if (!blob) return;
      fetch(`/api/capture?name=${encodeURIComponent(CAPTURE.name)}&i=${idx}`, {
        method: "POST",
        body: blob,
      }).catch((err) => console.error("帧上传失败", err));
    }, "image/png");

    capState.i += 1;
    if (capState.i >= CAPTURE.frames) {
      capState.done = true;
      if (CAPTURE.frames > 1 && CAPTURE.encode) {
        // 等最后几帧传完再请求编码
        setTimeout(() => {
          fetch("/api/capture/encode", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name: CAPTURE.name, fps: CAPTURE.fps }),
          }).then((r) => r.json())
            .then((d) => console.log("采集完成", d))
            .catch((err) => console.error("编码请求失败", err));
        }, 1600);
      }
    }
  }

  // ---------- 渲染循环 ----------
  function animate() {
    requestAnimationFrame(animate);
    updateCameraFrame();
    renderer.render(scene, camera);
    drawLabels();
    if (CAPTURE) captureTick();
  }

  layoutAll();
  animate();
})();
