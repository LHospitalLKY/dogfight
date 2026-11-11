/* 独立材质预览场景（不接入工程代码）。
 * 用途：给 j20 / f16 / f15 / su57 与导弹 OBJ 换一套更真实的 PBR 材质并出静态渲染图。
 * 用法：index.html?shot=j20_hero|j20_canopy|f16_hero|f16_canopy|...|missiles&w=1600&h=900
 */
(function () {
  "use strict";

  const q = new URLSearchParams(location.search);
  const shot = q.get("shot") || "hero";
  const W = parseInt(q.get("w") || "1600", 10);
  const H = parseInt(q.get("h") || "900", 10);

  // 材质库必须是第一个可用依赖：下面的环境贴图/背景在文件顶层就会用到 MAT.srgb。
  const MAT = window.MAT;

  THREE.ColorManagement.legacyMode = false;

  const renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
  renderer.setPixelRatio(1);
  renderer.setSize(W, H);
  renderer.outputEncoding = THREE.sRGBEncoding;
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.0;
  renderer.shadowMap.enabled = true;
  renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  document.body.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(30, W / H, 0.1, 800);

  // ------------------------------------------------------------ 环境与背景
  function gradientDome() {
    const geo = new THREE.SphereGeometry(120, 48, 32);
    const mat = new THREE.ShaderMaterial({
      side: THREE.BackSide,
      depthWrite: false,
      uniforms: {
        top: { value: MAT.srgb(0x1d2530) },
        mid: { value: MAT.srgb(0x10151b) },
        bot: { value: MAT.srgb(0x05070a) },
      },
      vertexShader: "varying vec3 vP; void main(){ vP = position; gl_Position = projectionMatrix * modelViewMatrix * vec4(position,1.0); }",
      fragmentShader: `
        varying vec3 vP; uniform vec3 top, mid, bot;
        void main(){
          float h = normalize(vP).y;
          vec3 c = h > 0.0 ? mix(mid, top, pow(h, 0.75)) : mix(mid, bot, pow(-h, 0.55));
          gl_FragColor = vec4(c, 1.0);
        }`,
    });
    return new THREE.Mesh(geo, mat);
  }

  function softbox(w, h, pos, rot, rgb) {
    const m = new THREE.Mesh(
      new THREE.PlaneGeometry(w, h),
      new THREE.MeshBasicMaterial({ color: new THREE.Color().setRGB(rgb[0], rgb[1], rgb[2]), side: THREE.DoubleSide }),
    );
    m.position.set(pos[0], pos[1], pos[2]);
    m.rotation.set(rot[0], rot[1], rot[2]);
    return m;
  }

  function studioEnvironment() {
    const env = new THREE.Scene();
    env.add(gradientDome());
    // 顶部长条柔光箱：金属表面最重要的高光来源
    env.add(softbox(34, 12, [0, 26, 2], [Math.PI / 2, 0, 0], [2.4, 2.45, 2.55]));
    // 左前主光
    env.add(softbox(4.2, 26, [-24, 8, -12], [0, Math.PI / 2, 0], [1.5, 1.5, 1.55]));
    // 右侧冷色补光
    env.add(softbox(3.0, 20, [24, 6, 4], [0, -Math.PI / 2, 0], [0.7, 0.95, 1.3]));
    // 后方暖色轮廓光条
    env.add(softbox(24, 2.6, [2, 9, 26], [0, Math.PI, 0], [1.4, 1.33, 1.20]));
    // 地面反弹（很暗）
    env.add(softbox(40, 40, [0, -6, 0], [-Math.PI / 2, 0, 0], [0.05, 0.06, 0.08]));
    const pmrem = new THREE.PMREMGenerator(renderer);
    const tex = pmrem.fromScene(env, 0.02).texture;
    pmrem.dispose();
    return tex;
  }

  function backdrop() {
    const cv = document.createElement("canvas");
    cv.width = 1024;
    cv.height = 512;
    const g = cv.getContext("2d");
    const grad = g.createLinearGradient(0, 0, 0, 512);
    grad.addColorStop(0.0, "#1b222b");
    grad.addColorStop(0.42, "#141a21");
    grad.addColorStop(0.56, "#0d1116");
    grad.addColorStop(1.0, "#05070a");
    g.fillStyle = grad;
    g.fillRect(0, 0, 1024, 512);
    // 地平线附近的一层薄雾，避免背景纯死黑
    const fog = g.createRadialGradient(512, 300, 20, 512, 300, 460);
    fog.addColorStop(0, "rgba(70,88,110,0.16)");
    fog.addColorStop(1, "rgba(70,88,110,0)");
    g.fillStyle = fog;
    g.fillRect(0, 0, 1024, 512);
    const tex = new THREE.CanvasTexture(cv);
    tex.mapping = THREE.EquirectangularReflectionMapping;
    tex.encoding = THREE.sRGBEncoding;
    return tex;
  }

  function floorRig() {
    const group = new THREE.Group();
    // 只接阴影，不要可见的地面：避免出现生硬的圆盘/地平线边
    const shadow = new THREE.Mesh(
      new THREE.CircleGeometry(26, 128),
      new THREE.ShadowMaterial({ color: 0x000000, opacity: 0.20 }),
    );
    shadow.rotation.x = -Math.PI / 2;
    shadow.position.y = -2.05;
    shadow.receiveShadow = true;
    if (!(q.get("dbg") || "").includes("nopad")) group.add(shadow);
    return group;
  }

  const envTex = studioEnvironment();
  scene.environment = (q.get("dbg") || "").includes("noenv") ? null : envTex;
  scene.background = backdrop();
  if (!(q.get("dbg") || "").includes("nofog")) scene.fog = new THREE.Fog(0x080b0f, 55, 165);

  if (shot !== "missiles") scene.add(floorRig());

  // 背景柔光板（放在机体后方很远处，只做很淡的舞台光晕）
  function backdropGlow() {
    const cv = document.createElement("canvas");
    cv.width = 512; cv.height = 512;
    const g = cv.getContext("2d");
    const rad = g.createRadialGradient(256, 256, 10, 256, 256, 250);
    rad.addColorStop(0, "rgba(120,150,180,0.55)");
    rad.addColorStop(1, "rgba(120,150,180,0)");
    g.fillStyle = rad;
    g.fillRect(0, 0, 512, 512);
    const tex = new THREE.CanvasTexture(cv);
    tex.encoding = THREE.sRGBEncoding;
    const m = new THREE.Mesh(
      new THREE.PlaneGeometry(110, 70),
      new THREE.MeshBasicMaterial({ map: tex, transparent: true, depthWrite: false, opacity: 0.55 }),
    );
    m.position.set(6, 6, -70);
    return m;
  }
  if (shot !== "missiles") scene.add(backdropGlow());

  // ------------------------------------------------------------ 灯光
  function lights(castShadow) {
    const key = new THREE.DirectionalLight(0xf6f9ff, 1.7);
    key.position.set(-16, 22, -14);
    if (castShadow) {
      key.castShadow = true;
      key.shadow.mapSize.set(2048, 2048);
      const c = key.shadow.camera;
      c.left = -20; c.right = 20; c.top = 20; c.bottom = -20;
      c.near = 1; c.far = 80;
      key.shadow.bias = -0.0008;
      key.shadow.normalBias = 0.05;
      key.shadow.radius = 3;
    }
    scene.add(key);

    const fill = new THREE.DirectionalLight(0xcfe2ff, 1.5);
    fill.position.set(24, 6, 10);
    scene.add(fill);

    const rimWarm = new THREE.DirectionalLight(0xffe9d6, 0.5);
    rimWarm.position.set(12, 11, 22);
    scene.add(rimWarm);

    const rimCool = new THREE.DirectionalLight(0xa9c8f5, 1.5);
    rimCool.position.set(-8, 6, 20);
    rimCool.position.set(-14, 5, 18);
    scene.add(rimCool);

    // 极弱的天穹补光，避免机腹完全死黑
    const hemi = new THREE.HemisphereLight(0x2b3644, 0x101317, 0.55);
    scene.add(hemi);

    // 尾部补光：后视机位时给垂尾/喷口一点轮廓
    const backFill = new THREE.DirectionalLight(0xdce8ff, 0.85);
    backFill.position.set(4, 9, 30);
    scene.add(backFill);

    if (((q.get("dbg") || "")).includes("amb")) {
      scene.add(new THREE.AmbientLight(0xffffff, 1.2));
    }
  }

  // ------------------------------------------------------------ 材质套装
  const MODELS = "/dogfight/web/static/models";

  // 各机型的少数可调参数：机体色与蒙皮分块周期（其余参数共用一个真实感基线）
  const AIRCRAFT = {
    j20: { file: "j20", label: "J-20", body: 0x454e58, seamZ: 1.55, seamX: 2.10 },
    f16: { file: "f16", label: "F-16C", body: 0x4b535c, seamZ: 1.20, seamX: 1.50 },
    f15: { file: "f15ex", label: "F-15EX", body: 0x49525b, seamZ: 1.45, seamX: 1.95 },
    su57: { file: "su57", label: "Su-57", body: 0x47505b, seamZ: 1.50, seamX: 2.05 },
  };

  function aircraftMaterials(mode, spec) {
    if (mode === "legacy") {
      return {
        body: new THREE.MeshStandardMaterial({
          color: MAT.srgb(0x6a7178), roughness: 0.82, metalness: 0.15, side: THREE.DoubleSide,
        }),
        canopy: new THREE.MeshStandardMaterial({
          color: MAT.srgb(0x9fb3bd), roughness: 0.25, metalness: 0.15,
          transparent: true, opacity: 0.55, depthWrite: false, side: THREE.DoubleSide,
        }),
      };
    }
    return {
      // 低可视度哑光金属漆：金属度中等（金属漆而不是镜面铝），粗糙度偏高；
      // 雷达罩/尾喷口在同一着色器里按 z 平滑过渡，避免按三角形切材质留下锯齿。
      body: MAT.skinMaterial({
        color: spec.body, metalness: 0.34, roughness: 0.62,
        seamPeriodZ: spec.seamZ, seamPeriodX: spec.seamX, seamWidth: 0.024,
        seamDark: 0.20, mottle: 0.025, roughSpread: 0.12,
        clearcoat: 0.06, clearcoatRoughness: 0.85, envMapIntensity: 0.45,
        noseZ: spec.noseZ, radomeColor: 0x2b3137, radomeRough: 0.64,
        nozzleZ: spec.nozzleZ, nozzleColor: 0x4b4640,
      }),
      // 镀膜座舱玻璃：沿用模型自带的座舱分组，只换材质，不动它的位置与形状。
      canopy: MAT.glassMaterial({
        tint: 0xe9e5db, roughness: 0.02, thickness: 0.10,
        transmission: 0.94, attenuationColor: 0xcfae74, attenuationDistance: 0.9,
        envMapIntensity: 1.15, doubleSide: true, clipY: null,
      }),
    };
  }

  function buildAircraft(geoBody, geoCanopy, mode) {
    const group = new THREE.Group();
    const body = MAT.weldAndSmooth(geoBody, 42);
    body.computeBoundingBox();
    const bb = body.boundingBox;
    const len = bb.max.z - bb.min.z;
    const spec = Object.assign({}, AIRCRAFT[shot.split("_")[0]], {
      // 机头雷达罩、尾喷口按各机自身长度比例定位，换机型不用手调
      noseZ: bb.min.z + len * 0.165,
      nozzleZ: bb.max.z - len * 0.115,
    });
    const mats = aircraftMaterials(mode, spec);

    const meshes = {};
    const m = new THREE.Mesh(body, mats.body);
    m.castShadow = true;
    m.receiveShadow = false;
    meshes.body = m;
    if (!(q.get("dbg") || "").includes("bodyless")) group.add(m);

    let cbb = null;
    if (geoCanopy && geoCanopy.attributes.position.count) {
      // 座舱：直接用模型自带的 canopy 分组（不做重放样、不开孔、不改位置）
      const cg = MAT.weldAndSmooth(geoCanopy, 34);
      cg.computeBoundingBox();
      cbb = cg.boundingBox;
      const cm = new THREE.Mesh(cg, mats.canopy);
      cm.castShadow = false;
      cm.receiveShadow = false;
      cm.renderOrder = 4;
      group.add(cm);
      meshes.canopy = cm;
    }

    group.userData.meshes = meshes;
    group.userData.bb = bb;
    group.userData.cbb = cbb;
    return group;
  }

  function missileMaterials(name) {
    const schemes = {
      "PL-15": { base: 0xaeb4b9, band: 0xa8834a },
      "PL-10": { base: 0xb2b7bb, band: 0xa8834a },
      "AIM-120C": { base: 0xa9afb4, band: 0xbe9a4d },
      "AIM-9X": { base: 0xb4b9bd, band: 0xbfa04e },
      "R-77-1": { base: 0xacb1b6, band: 0xb09355 },
      "R-73": { base: 0xb0b4b8, band: 0xac925c },
    };
    const s = schemes[name] || schemes["R-73"];
    return {
      body: MAT.skinMaterial({
        color: s.base, metalness: 0.38, roughness: 0.36,
        seamPeriodZ: 0.30, seamPeriodX: 0.34, seamWidth: 0.006,
        seamDark: 0.42, mottle: 0.03, roughSpread: 0.16,
        clearcoat: 0.45, clearcoatRoughness: 0.22,
      }),
      nose: MAT.dielectricMaterial(0x2f343a, 0.52),
      nozzle: MAT.nozzleMaterial(),
      band: new THREE.MeshPhysicalMaterial({
        color: MAT.srgb(s.band), roughness: 0.45, metalness: 0.25,
        clearcoat: 0.3, clearcoatRoughness: 0.35,
      }),
    };
  }

  function buildMissile(geo, name) {
    const g = new THREE.Group();
    const body = MAT.weldAndSmooth(geo.body, 38);
    body.computeBoundingBox();
    const bb = body.boundingBox;
    const len = bb.max.z - bb.min.z;
    const cuts = [
      [-1e9, bb.min.z + len * 0.155],             // 雷达/红外罩
      [bb.min.z + len * 0.155, bb.min.z + len * 0.20],   // 识别环
      [bb.min.z + len * 0.20, bb.min.z + len * 0.962],   // 弹体
      [bb.min.z + len * 0.962, 1e9],               // 尾喷口
    ];
    const mats = missileMaterials(name);
    const parts = MAT.splitBy(body, 2, cuts);
    const keys = ["nose", "band", "body", "nozzle"];
    parts.forEach((p, i) => {
      if (!p.attributes.position.count) return;
      const m = new THREE.Mesh(p, mats[keys[i]]);
      m.castShadow = true;
      m.receiveShadow = true;
      g.add(m);
    });
    // 把几何重心挪到原点：各弹的导出原点不一致，否则并排摆放会错位
    const box = new THREE.Box3().setFromObject(g);
    const c = box.getCenter(new THREE.Vector3());
    g.children.forEach((m) => m.position.sub(c));
    return g;
  }

  // ------------------------------------------------------------ 镜头
  function frame(target, dir, radius, fov, margin) {
    camera.fov = fov;
    camera.updateProjectionMatrix();
    const dist = (radius / Math.sin(THREE.MathUtils.degToRad(fov * 0.5))) * (margin || 1.0);
    camera.position.copy(target).addScaledVector(new THREE.Vector3(dir[0], dir[1], dir[2]).normalize(), dist);
    camera.lookAt(target);
    camera.near = Math.max(0.1, dist * 0.02);
    camera.far = dist * 20;
    camera.updateProjectionMatrix();
  }

  function labelSprite(text) {
    const cv = document.createElement("canvas");
    cv.width = 512; cv.height = 128;
    const g = cv.getContext("2d");
    g.fillStyle = "rgba(10,13,17,0.72)";
    g.fillRect(0, 0, 512, 128);
    g.strokeStyle = "rgba(120,170,205,0.55)";
    g.lineWidth = 3;
    g.strokeRect(1.5, 1.5, 509, 125);
    g.fillStyle = "#cfe0ee";
    g.font = "600 56px 'Helvetica Neue', Arial, sans-serif";
    g.textAlign = "center";
    g.textBaseline = "middle";
    g.fillText(text, 256, 68);
    const tex = new THREE.CanvasTexture(cv);
    tex.encoding = THREE.sRGBEncoding;
    const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false }));
    sp.scale.set(1.6, 0.4, 1);
    return sp;
  }

  // ------------------------------------------------------------ 组装
  function loadAircraft(key, mode) {
    const a = AIRCRAFT[key] || AIRCRAFT.j20;
    return MAT.loadObj(`${MODELS}/${a.file}.obj`).then((geo) => {
      const model = buildAircraft(geo.body, geo.canopy, mode);
      scene.add(model);
      lights(true);
      return model;
    });
  }

  function shootAircraft(key, view, mode) {
    loadAircraft(key, mode).then((model) => {
      const bb = model.userData.bb;
      const cbb = model.userData.cbb;
      const size = new THREE.Vector3();
      if (view === "canopy" && cbb) {
        const c = new THREE.Vector3();
        cbb.getCenter(c);
        cbb.getSize(size);
        frame(c, [-0.46, 0.52, -0.72], Math.max(size.x, size.y, size.z) * 0.85, 30, 1.0);
      } else {
        const c = new THREE.Vector3();
        bb.getCenter(c);
        bb.getSize(size);
        const half = size.length() / 2;
        if (view === "rear") {
          frame(c, [0.55, 0.20, 0.80], half * 0.50, 32, 0.98);
        } else if (view === "detail") {
          frame(c, [-0.42, 0.30, -0.86], half * 0.40, 30, 1.0);
        } else {
          frame(c, [-0.64, 0.34, -0.70], half * 0.55, 31, 1.0);
        }
      }
      render();
    });
  }

  function shootMissiles() {
    const list = [
      ["PL-15", "pl15"], ["PL-10", "pl10"], ["AIM-120C", "aim120"],
      ["AIM-9X", "aim9"], ["R-77-1", "r77"], ["R-73", "r73"],
    ];
    lights(true);
    const jobs = list.map(([name, file], i) => MAT.loadObj(`${MODELS}/${file}.obj`)
      .then((geo) => ({ name, model: buildMissile(geo, name), i })));
    Promise.all(jobs).then((items) => {
      const group = new THREE.Group();
      const spacing = 0.95;
      const platform = new THREE.Mesh(
        new THREE.BoxGeometry(5.0, 0.12, spacing * list.length + 0.6),
        new THREE.MeshLambertMaterial({ color: MAT.srgb(0x6b747e) }),
      );
      platform.position.set(0, -0.02, 0);
      platform.receiveShadow = true;
      group.add(platform);

      items.forEach(({ name, model, i }) => {
        const z = (i - (list.length - 1) / 2) * spacing;
        // 弹轴 +Z，转到沿 -X 平放，弹头朝 -X
        model.rotation.y = -Math.PI / 2;
        model.position.set(0, 0.16, z);
        group.add(model);

        const lb = labelSprite(name);
        lb.position.set(-2.5, 0.52, z);
        lb.scale.set(1.15, 0.30, 1);
        group.add(lb);
      });
      scene.add(group);
      frame(new THREE.Vector3(0.05, 0.30, 0.0), [0.82, 0.42, 0.20], 2.7, 20, 1.0);
      render();
    });
  }

  // ------------------------------------------------------------ 出图
  let ready = false;
  function render() {
    renderer.render(scene, camera);
    if (!ready) {
      ready = true;
      // 第二帧：透射材质需要先有一帧把背景渲染进 transmission 缓冲
      requestAnimationFrame(() => {
        renderer.render(scene, camera);
        window.__shotReady = true;
      });
    }
  }

  window.addEventListener("error", (e) => {
    window.__shotError = String(e.message || e);
  });

  if (shot === "missiles") {
    shootMissiles();
  } else {
    const parts = shot.split("_");
    const key = parts[0];
    const view = parts[1] || "hero";
    if (key === "j20" && view === "legacy") scene.environment = null;
    shootAircraft(key, view, view === "legacy" ? "legacy" : "pbr");
  }
})();
