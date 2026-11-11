/* 材质与几何工具：给仓库里的 OBJ 增加更真实的 PBR 表面。
 * 纯前端演示，不修改工程代码；模型仍是 dogfight/web/static/models 下的导出文件。
 */
(function () {
  "use strict";

  // 十六进制颜色按 sRGB 解释并转到线性空间：r149 的 ColorManagement 在部分
  // 构建里不生效，显式转换可以保证材质颜色不被“洗白”。
  function srgb(hex) {
    return new THREE.Color(hex).convertSRGBToLinear();
  }

  // ---------------------------------------------------------------- 几何
  // 服务端导出的 OBJ 每个三角形都是独立顶点，直接 computeVertexNormals 会得到
  // 全平面着色（一块块棱面）。这里先按 1 mm 量化焊接顶点，再按夹角阈值平滑，
  // 既保住进气道唇口/翼前缘这类硬边，又让机身曲面连续。
  function weldAndSmooth(rawGeo, angleDeg) {
    const pos = rawGeo.attributes.position.array;
    const n = pos.length / 3;
    const q = 1e3; // 1 mm
    const map = new Map();
    const verts = [];
    const index = new Int32Array(n);
    for (let i = 0; i < n; i += 1) {
      const x = pos[i * 3], y = pos[i * 3 + 1], z = pos[i * 3 + 2];
      const key = `${Math.round(x * q)}|${Math.round(y * q)}|${Math.round(z * q)}`;
      let id = map.get(key);
      if (id === undefined) {
        id = verts.length / 3;
        map.set(key, id);
        verts.push(x, y, z);
      }
      index[i] = id;
    }
    const triCount = n / 3;
    const fn = new Float32Array(triCount * 3);
    for (let t = 0; t < triCount; t += 1) {
      const a = index[t * 3] * 3, b = index[t * 3 + 1] * 3, c = index[t * 3 + 2] * 3;
      const ax = verts[b] - verts[a], ay = verts[b + 1] - verts[a + 1], az = verts[b + 2] - verts[a + 2];
      const bx = verts[c] - verts[a], by = verts[c + 1] - verts[a + 1], bz = verts[c + 2] - verts[a + 2];
      let nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
      const len = Math.hypot(nx, ny, nz) || 1;
      fn[t * 3] = nx / len; fn[t * 3 + 1] = ny / len; fn[t * 3 + 2] = nz / len;
    }
    // 顶点 → 相邻面
    const adj = new Map();
    for (let t = 0; t < triCount; t += 1) {
      for (let k = 0; k < 3; k += 1) {
        const v = index[t * 3 + k];
        let list = adj.get(v);
        if (!list) { list = []; adj.set(v, list); }
        list.push(t);
      }
    }
    const cosLim = Math.cos((angleDeg * Math.PI) / 180);
    const outPos = new Float32Array(n * 3);
    const outNrm = new Float32Array(n * 3);
    for (let t = 0; t < triCount; t += 1) {
      const fx = fn[t * 3], fy = fn[t * 3 + 1], fz = fn[t * 3 + 2];
      for (let k = 0; k < 3; k += 1) {
        const vi = index[t * 3 + k];
        const list = adj.get(vi);
        let sx = 0, sy = 0, sz = 0;
        for (let j = 0; j < list.length; j += 1) {
          const a = list[j];
          const ax = fn[a * 3], ay = fn[a * 3 + 1], az = fn[a * 3 + 2];
          if (ax * fx + ay * fy + az * fz >= cosLim) { sx += ax; sy += ay; sz += az; }
        }
        const len = Math.hypot(sx, sy, sz) || 1;
        const o = (t * 3 + k) * 3;
        outPos[o] = verts[vi * 3];
        outPos[o + 1] = verts[vi * 3 + 1];
        outPos[o + 2] = verts[vi * 3 + 2];
        outNrm[o] = sx / len;
        outNrm[o + 1] = sy / len;
        outNrm[o + 2] = sz / len;
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.BufferAttribute(outPos, 3));
    geo.setAttribute("normal", new THREE.BufferAttribute(outNrm, 3));
    geo.computeBoundingBox();
    geo.computeBoundingSphere();
    return geo;
  }

  // 按三角形重心剔除几何：用于在机身座舱位置开一个口，避免玻璃罩与蒙皮
  // 互相穿插产生锯齿状的深度接缝（原模型没有为座舱盖做开口）。
  function cutRegion(geo, keepCentroid) {
    const pos = geo.attributes.position.array;
    const nrm = geo.attributes.normal.array;
    const triCount = pos.length / 9;
    const p = [];
    const n = [];
    for (let t = 0; t < triCount; t += 1) {
      const o = t * 9;
      const cx = (pos[o] + pos[o + 3] + pos[o + 6]) / 3;
      const cy = (pos[o + 1] + pos[o + 4] + pos[o + 7]) / 3;
      const cz = (pos[o + 2] + pos[o + 5] + pos[o + 8]) / 3;
      if (!keepCentroid(cx, cy, cz)) continue;
      for (let i = 0; i < 9; i += 1) { p.push(pos[o + i]); n.push(nrm[o + i]); }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
    g.setAttribute("normal", new THREE.Float32BufferAttribute(n, 3));
    g.computeBoundingSphere();
    return g;
  }

  // 按三角形重心 z（或任意轴）切分几何：机头雷达罩、尾喷口单独给材质。
  function splitBy(geo, axis, ranges) {
    const pos = geo.attributes.position.array;
    const nrm = geo.attributes.normal.array;
    const triCount = pos.length / 9;
    const buckets = ranges.map(() => ({ p: [], n: [] }));
    for (let t = 0; t < triCount; t += 1) {
      const o = t * 9;
      const c = (pos[o + axis] + pos[o + 3 + axis] + pos[o + 6 + axis]) / 3;
      for (let r = 0; r < ranges.length; r += 1) {
        if (c >= ranges[r][0] && c < ranges[r][1]) {
          const b = buckets[r];
          for (let i = 0; i < 9; i += 1) { b.p.push(pos[o + i]); b.n.push(nrm[o + i]); }
          break;
        }
      }
    }
    return buckets.map((b, i) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(b.p, 3));
      g.setAttribute("normal", new THREE.Float32BufferAttribute(b.n, 3));
      g.computeBoundingSphere();
      g.userData.range = ranges[i];
      return g;
    });
  }

  // 座舱盖专用：保留 y 以上的光滑穹顶，并把开口边界向下拉出一圈竖直裙边，
  // 让玻璃罩有一个干净的水密边缘（原模型边缘是一圈尖三角形，会在交界处
  // 留下锯齿轮廓）。
  function domeAndSkirt(geo, yCut, yBottom) {
    const pos = geo.attributes.position.array;
    const nrm = geo.attributes.normal.array;
    const triCount = pos.length / 9;
    const keepP = [];
    const keepN = [];
    const edges = new Map(); // 边 → 出现次数 + 端点
    const key = (x, y, z) => `${Math.round(x * 1e3)}|${Math.round(y * 1e3)}|${Math.round(z * 1e3)}`;
    for (let t = 0; t < triCount; t += 1) {
      const o = t * 9;
      const cy = (pos[o + 1] + pos[o + 4] + pos[o + 7]) / 3;
      if (cy < yCut) continue;
      for (let i = 0; i < 9; i += 1) { keepP.push(pos[o + i]); keepN.push(nrm[o + i]); }
      for (let k = 0; k < 3; k += 1) {
        const a = o + k * 3;
        const b = o + ((k + 1) % 3) * 3;
        const ka = key(pos[a], pos[a + 1], pos[a + 2]);
        const kb = key(pos[b], pos[b + 1], pos[b + 2]);
        const ek = ka < kb ? ka + "#" + kb : kb + "#" + ka;
        const hit = edges.get(ek);
        if (hit) hit.n += 1;
        else edges.set(ek, { n: 1, a: [pos[a], pos[a + 1], pos[a + 2]], b: [pos[b], pos[b + 1], pos[b + 2]] });
      }
    }
    const dome = new THREE.BufferGeometry();
    dome.setAttribute("position", new THREE.Float32BufferAttribute(keepP, 3));
    dome.setAttribute("normal", new THREE.Float32BufferAttribute(keepN, 3));

    // 边界边（只被一个保留三角形引用）→ 向下拉裙边
    const skP = [];
    edges.forEach((e) => {
      if (e.n !== 1) return;
      const a = e.a;
      const b = e.b;
      const a2 = [a[0], yBottom, a[2]];
      const b2 = [b[0], yBottom, b[2]];
      skP.push(...a, ...b, ...b2, ...a, ...b2, ...a2);
    });
    const skirt = new THREE.BufferGeometry();
    skirt.setAttribute("position", new THREE.Float32BufferAttribute(skP, 3));
    skirt.computeVertexNormals();
    dome.computeBoundingSphere();
    skirt.computeBoundingSphere();
    return { dome, skirt };
  }

  // 座舱盖重建：原模型的座舱组边缘是一圈尖三角形，和机身相交会留下锯齿。
  // 这里按模型自身的轮廓（每个 z 切片的半宽与最高点）重新放样出一个干净的
  // 穹顶，边缘下沉到机身表面以下，形成水密、平滑的玻璃罩。
  function canopyShell(geo, opts) {
    const o = Object.assign({ baseY: 0.03, topDrop: 0, slices: 72, ring: 34, pow: 0.62, xPow: 0.80 }, opts || {});
    const pos = geo.attributes.position.array;
    let zmin = Infinity;
    let zmax = -Infinity;
    for (let i = 2; i < pos.length; i += 3) {
      if (pos[i] < zmin) zmin = pos[i];
      if (pos[i] > zmax) zmax = pos[i];
    }
    // 先粗采样，再用 Catmull-Rom 插值：原始网格边缘有细碎尖刺，直接逐层取
    // 极值会让镜面高光出现台阶状波纹。
    const coarse = 14;
    const cdz = (zmax - zmin) / coarse;
    const cp = [];
    for (let i = 0; i <= coarse; i += 1) {
      const z = zmin + cdz * i;
      let hw = 0;
      let top = -1e9;
      for (let k = 0; k < pos.length; k += 3) {
        if (Math.abs(pos[k + 2] - z) > cdz * 0.9) continue;
        const ax = Math.abs(pos[k]);
        if (ax > hw) hw = ax;
        if (pos[k + 1] > top) top = pos[k + 1];
      }
      cp.push({ hw, top });
    }
    // 两端口径收拢，避免座舱罩出现生硬的开边
    cp[0].hw *= 0.35;
    cp[cp.length - 1].hw *= 0.30;

    const catmull = (arr, u) => {
      const n = arr.length - 1;
      const f = Math.min(Math.max(u, 0), 1) * n;
      const i = Math.min(Math.floor(f), n - 1);
      const t = f - i;
      const p0 = arr[Math.max(i - 1, 0)];
      const p1 = arr[i];
      const p2 = arr[i + 1];
      const p3 = arr[Math.min(i + 2, n)];
      const t2 = t * t;
      const t3 = t2 * t;
      return 0.5 * ((2 * p1) + (-p0 + p2) * t
        + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
        + (-p0 + 3 * p1 - 3 * p2 + p3) * t3);
    };

    const steps = o.slices;
    const rings = [];
    for (let i = 0; i <= steps; i += 1) {
      const u = i / steps;
      rings.push({
        z: zmin + (zmax - zmin) * u,
        hw: Math.max(0.006, catmull(cp.map((p) => p.hw), u)),
        top: Math.max(o.baseY + 0.02, catmull(cp.map((p) => p.top), u) - o.topDrop),
      });
    }
    const K = o.ring;
    const verts = [];
    rings.forEach((r) => {
      for (let j = 0; j < K; j += 1) {
        const th = (Math.PI * j) / (K - 1);
        // 超椭圆截面：侧壁更直、顶部更平，接近真实座舱盖的梯形轮廓
        const cx = Math.cos(th);
        const sx = Math.sign(cx) * Math.pow(Math.abs(cx), o.xPow);
        verts.push(
          r.hw * sx,
          o.baseY + (r.top - o.baseY) * Math.pow(Math.sin(th), o.pow),
          r.z,
        );
      }
    });
    const idx = [];
    for (let i = 0; i < rings.length - 1; i += 1) {
      for (let j = 0; j < K - 1; j += 1) {
        const a = i * K + j;
        const b = a + 1;
        const c = a + K;
        const d = c + 1;
        idx.push(a, c, b, b, c, d);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setIndex(idx);
    g.setAttribute("position", new THREE.Float32BufferAttribute(verts, 3));
    g.computeVertexNormals();
    g.computeBoundingSphere();
    g.userData.rings = rings.map((r, i) => verts.slice(i * K * 3, (i + 1) * K * 3));
    g.userData.ringCount = K;
    return g;
  }

  // 从一块开放曲面的边界向下拉出竖直裙边：给座舱罩补一圈干净的“舱框”，
  // 把玻璃罩与机身蒙皮那条参差的相交线挡住。
  function rimSkirt(geo, yBottom) {
    const pos = geo.attributes.position.array;
    const idx = geo.index ? geo.index.array : null;
    const triCount = idx ? idx.length / 3 : pos.length / 9;
    const at = (t, k) => (idx ? idx[t * 3 + k] : t * 3 + k);
    const edges = new Map();
    for (let t = 0; t < triCount; t += 1) {
      for (let k = 0; k < 3; k += 1) {
        const a = at(t, k);
        const b = at(t, (k + 1) % 3);
        const key = a < b ? `${a}#${b}` : `${b}#${a}`;
        const hit = edges.get(key);
        if (hit) hit.n += 1;
        else edges.set(key, { n: 1, a, b });
      }
    }
    const P = [];
    edges.forEach((e) => {
      if (e.n !== 1) return;
      const a = [pos[e.a * 3], pos[e.a * 3 + 1], pos[e.a * 3 + 2]];
      const b = [pos[e.b * 3], pos[e.b * 3 + 1], pos[e.b * 3 + 2]];
      const a2 = [a[0], yBottom, a[2]];
      const b2 = [b[0], yBottom, b[2]];
      P.push(...a, ...b, ...b2, ...a, ...b2, ...a2);
    });
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(P, 3));
    g.computeVertexNormals();
    g.computeBoundingSphere();
    return g;
  }

  // 按算出来的面法线切分几何。座舱盖不做任何形状/位置改动：法线朝上的面
  // 归为玻璃穹顶，朝外/朝下的下缘归为舱框，交界处由几何本身决定。
  function splitByNormal(geo, keep) {
    const pos = geo.attributes.position.array;
    const nrm = geo.attributes.normal.array;
    const tri = pos.length / 9;
    const bins = [{ p: [], n: [] }, { p: [], n: [] }];
    for (let t = 0; t < tri; t += 1) {
      const o = t * 9;
      const nx = nrm[o] + nrm[o + 3] + nrm[o + 6];
      const ny = nrm[o + 1] + nrm[o + 4] + nrm[o + 7];
      const nz = nrm[o + 2] + nrm[o + 5] + nrm[o + 8];
      const len = Math.hypot(nx, ny, nz) || 1;
      const b = bins[keep(nx / len, ny / len, nz / len) ? 0 : 1];
      for (let i = 0; i < 9; i += 1) { b.p.push(pos[o + i]); b.n.push(nrm[o + i]); }
    }
    return bins.map((b) => {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(b.p, 3));
      g.setAttribute("normal", new THREE.Float32BufferAttribute(b.n, 3));
      g.computeBoundingSphere();
      return g;
    });
  }

  // ---------------------------------------------------------------- 着色注入
  const NOISE = `
    float hash31(vec3 p){
      p = fract(p * 0.1031);
      p += dot(p, p.yzx + 33.33);
      return fract((p.x + p.y) * p.z);
    }
    float vnoise(vec3 p){
      vec3 i = floor(p), f = fract(p);
      f = f * f * (3.0 - 2.0 * f);
      float a = mix(mix(hash31(i), hash31(i + vec3(1,0,0)), f.x),
                    mix(hash31(i + vec3(0,1,0)), hash31(i + vec3(1,1,0)), f.x), f.y);
      float b = mix(mix(hash31(i + vec3(0,0,1)), hash31(i + vec3(1,0,1)), f.x),
                    mix(hash31(i + vec3(0,1,1)), hash31(i + vec3(1,1,1)), f.x), f.y);
      return mix(a, b, f.z);
    }
    float fbm3(vec3 p){
      return 0.55 * vnoise(p) + 0.28 * vnoise(p * 2.03) + 0.14 * vnoise(p * 4.01);
    }
    float seam(float c, float period, float width){
      float d = abs(fract(c / period + 0.5) - 0.5) * period;
      return 1.0 - smoothstep(width * 0.35, width, d);
    }
  `;

  // 机体蒙皮：低可视度哑光金属漆 + 分块蒙皮线 + 细微做旧 + 雷达罩过渡。
  // 全部按物体空间三平面坐标计算，不依赖 UV（OBJ 没有 UV）。
  // 雷达罩/尾喷口不再按三角形切材质（会在曲面处留下锯齿接缝），而是在同一个
  // 着色器里按 z 平滑过渡颜色/金属度/粗糙度。
  function skinMaterial(opts) {
    const o = Object.assign({
      color: 0x474e56,
      metalness: 0.55,
      roughness: 0.55,
      seamPeriodZ: 1.55,
      seamPeriodX: 2.10,
      seamWidth: 0.022,
      seamDark: 0.22,
      mottle: 0.025,
      roughSpread: 0.12,
      clearcoat: 0.22,
      clearcoatRoughness: 0.60,
      envMapIntensity: 0.85,
      noseZ: null,
      radomeColor: 0x353b42,
      radomeRough: 0.62,
      nozzleZ: null,
      nozzleColor: 0x4a4642,
    }, opts || {});

    const noseExpr = (o.noseZ === null)
      ? "0.0"
      : `(1.0 - smoothstep(${(o.noseZ - 0.55).toFixed(3)}, ${(o.noseZ + 0.55).toFixed(3)}, P.z))`;
    const nozzleExpr = (o.nozzleZ === null)
      ? "0.0"
      : `smoothstep(${(o.nozzleZ - 0.40).toFixed(3)}, ${(o.nozzleZ + 0.55).toFixed(3)}, P.z)`;

    const mat = new THREE.MeshPhysicalMaterial({
      color: srgb(o.color),
      metalness: o.metalness,
      roughness: o.roughness,
      clearcoat: o.clearcoat,
      clearcoatRoughness: o.clearcoatRoughness,
      envMapIntensity: o.envMapIntensity,
      side: THREE.DoubleSide,
    });
    mat.onBeforeCompile = (shader) => {
      shader.vertexShader = shader.vertexShader
        .replace("#include <common>", "#include <common>\nvarying vec3 vObjP;")
        .replace("#include <begin_vertex>", "#include <begin_vertex>\n  vObjP = position;");
      shader.fragmentShader = shader.fragmentShader
        .replace("#include <common>", "#include <common>\nvarying vec3 vObjP;\n" + NOISE + `
          float panel_seam(vec3 P){
            float lines = max(seam(P.z, ${o.seamPeriodZ.toFixed(3)}, ${o.seamWidth.toFixed(4)}),
                              seam(P.x, ${o.seamPeriodX.toFixed(3)}, ${o.seamWidth.toFixed(4)}));
            // 用低频噪声把连续的分块线打断，避免机翼上出现规整的“方格纸”
            lines *= mix(0.25, 1.0, smoothstep(0.30, 0.62, vnoise(P * 0.55 + 11.0)));
            return lines;
          }
        `)
        .replace("#include <map_fragment>", `
          {
            vec3 P = vObjP;
            float lines = panel_seam(P);
            float dirt = lines * (0.7 + 0.3 * fbm3(P * 6.0));
            diffuseColor.rgb *= (1.0 - ${o.seamDark.toFixed(3)} * dirt);
            float blot = fbm3(P * 0.33 + 3.7);
            diffuseColor.rgb *= 1.0 - ${o.mottle.toFixed(3)} * (blot - 0.5) * 2.0;
            float rn = ${noseExpr};
            float nz = ${nozzleExpr};
            diffuseColor.rgb = mix(diffuseColor.rgb, vec3(${hexToVec(o.radomeColor)}), rn);
            diffuseColor.rgb = mix(diffuseColor.rgb, vec3(${hexToVec(o.nozzleColor)}), nz);
          }
        `)
        .replace("#include <metalnessmap_fragment>", `
          float metalnessFactor = metalness;
          {
            vec3 P = vObjP;
            float rn = ${noseExpr};
            float nz = ${nozzleExpr};
            metalnessFactor = mix(metalnessFactor, 0.05, rn);
            metalnessFactor = mix(metalnessFactor, 1.0, nz);
          }
        `)
        .replace("#include <roughnessmap_fragment>", `
          float roughnessFactor = roughness;
          {
            vec3 P = vObjP;
            float lines = panel_seam(P);
            roughnessFactor *= 0.94 + ${o.roughSpread.toFixed(3)} * (fbm3(P * 0.9) - 0.5);
            roughnessFactor = mix(roughnessFactor, roughnessFactor + 0.12, lines * 0.6);
            float rn = ${noseExpr};
            float nz = ${nozzleExpr};
            roughnessFactor = mix(roughnessFactor, ${o.radomeRough.toFixed(3)}, rn);
            roughnessFactor = mix(roughnessFactor, 0.34, nz);
          }
          roughnessFactor = clamp(roughnessFactor, 0.05, 1.0);
        `);
    };
    // 着色器里没有 uniform 名字冲突，按参数缓存即可
    mat.customProgramCacheKey = () => "skin" + JSON.stringify(o);
    return mat;
  }

  // 十六进制颜色 → GLSL vec3（线性空间），供注入代码使用
  function hexToVec(hex) {
    const c = srgb(hex);
    return `${c.r.toFixed(4)}, ${c.g.toFixed(4)}, ${c.b.toFixed(4)}`;
  }

  // 座舱盖玻璃：薄壳透射 + 清漆高光，带一点点 ITO 镀膜的金色吸收。
  function glassMaterial(opts) {
    const o = Object.assign({
      tint: 0xe8e2cf,
      gold: 0xc9a86a,
      roughness: 0.055,
      thickness: 0.22,
      ior: 1.52,
      transmission: 0.96,
      attenuationColor: 0xbfa76e,
      attenuationDistance: 0.9,
      clipY: null,
      doubleSide: true,
    }, opts || {});
    const mat = new THREE.MeshPhysicalMaterial({
      color: srgb(o.tint),
      metalness: 0.0,
      roughness: o.roughness,
      transmission: o.transmission,
      thickness: o.thickness,
      ior: o.ior,
      attenuationColor: srgb(o.attenuationColor),
      attenuationDistance: o.attenuationDistance,
      clearcoat: 1.0,
      clearcoatRoughness: 0.03,
      transparent: true,
      side: o.doubleSide ? THREE.DoubleSide : THREE.FrontSide,
      envMapIntensity: o.envMapIntensity === undefined ? 2.0 : o.envMapIntensity,
      depthWrite: false,
      polygonOffset: true,
      polygonOffsetFactor: -2,
      polygonOffsetUnits: -2,
    });
    if (o.clipY !== null || o.clipYMax !== undefined) {
      const lo = o.clipY === null || o.clipY === undefined ? null : o.clipY;
      const hi = o.clipYMax === undefined ? null : o.clipYMax;
      const discard = [
        lo === null ? "" : `if (vObjP.y < ${lo.toFixed(3)}) discard;`,
        hi === null ? "" : `if (vObjP.y > ${hi.toFixed(3)}) discard;`,
      ].join(" ");
      mat.onBeforeCompile = (shader) => {
        shader.vertexShader = shader.vertexShader
          .replace("#include <common>", "#include <common>\nvarying vec3 vObjP;")
          .replace("#include <begin_vertex>", "#include <begin_vertex>\n  vObjP = position;");
        shader.fragmentShader = shader.fragmentShader
          .replace("#include <common>", "#include <common>\nvarying vec3 vObjP;")
          .replace("#include <clipping_planes_fragment>",
            `#include <clipping_planes_fragment>\n  ${discard}`);
      };
      mat.customProgramCacheKey = () => `glass${lo}:${hi}`;
    }
    return mat;
  }

  // 雷达罩/介质材料：非金属，稍高的粗糙度与更低的反射。
  function dielectricMaterial(color, roughness) {
    return new THREE.MeshPhysicalMaterial({
      color: srgb(color),
      metalness: 0.0,
      roughness: roughness === undefined ? 0.58 : roughness,
      clearcoat: 0.25,
      clearcoatRoughness: 0.5,
      envMapIntensity: 0.9,
      side: THREE.DoubleSide,
    });
  }

  // 座舱框/密封条：深色哑光，用来遮住模型自带座舱裙边的锯齿轮廓。
  function trimMaterial(color) {
    return new THREE.MeshStandardMaterial({
      color: srgb(color === undefined ? 0x15181b : color),
      metalness: 0.35,
      roughness: 0.62,
      side: THREE.DoubleSide,
    });
  }

  // 尾喷口：高温合金，带一点热变色（紫/蓝）与更高的金属度。
  function nozzleMaterial() {
    const m = skinMaterial({
      color: 0x4a4a4e, metalness: 1.0, roughness: 0.34,
      seamPeriodZ: 0.34, seamPeriodX: 0.6, seamDark: 0.45,
      clearcoat: 0.0, mottle: 0.03, roughSpread: 0.26,
      envMapIntensity: 1.5,
    });
    return m;
  }

  // ---------------------------------------------------------------- 模型装配
  function loadObj(url) {
    return fetch(url).then((r) => r.text()).then(parseOBJ);
  }

  function parseOBJ(text) {
    const verts = [];
    const groups = { body: [], canopy: [], stores: [] };
    let material = "body";
    const lines = text.split(/\r?\n/);
    for (let li = 0; li < lines.length; li += 1) {
      const raw = lines[li];
      if (raw.length < 2) continue;
      const kind = raw.slice(0, 2);
      if (kind === "v ") {
        const p = raw.trim().split(/\s+/);
        verts.push(+p[1], +p[2], +p[3]);
      } else if (raw.startsWith("usemtl ")) {
        const name = raw.slice(7).trim();
        material = (name === "canopy" || name === "stores") ? name : "body";
      } else if (kind === "f ") {
        const tok = raw.trim().split(/\s+/).slice(1);
        const idx = [];
        for (let i = 0; i < tok.length; i += 1) idx.push(parseInt(tok[i].split("/")[0], 10) - 1);
        for (let t = 1; t < idx.length - 1; t += 1) {
          const tri = [idx[0], idx[t], idx[t + 1]];
          for (let k = 0; k < 3; k += 1) {
            const i = tri[k];
            groups[material].push(verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2]);
          }
        }
      }
    }
    const out = {};
    for (const key of ["body", "canopy", "stores"]) {
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.Float32BufferAttribute(groups[key], 3));
      out[key] = g;
    }
    return out;
  }

  window.MAT = {
    srgb, weldAndSmooth, splitBy, splitByNormal, cutRegion, domeAndSkirt, canopyShell, rimSkirt, skinMaterial, glassMaterial,
    dielectricMaterial, nozzleMaterial, trimMaterial, loadObj, parseOBJ,
  };
})();
