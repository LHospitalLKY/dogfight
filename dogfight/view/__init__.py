"""three.js 观战端共享的纯几何层：尺度常量、姿态与模型转换/抽稀。

模型统一来自 configs/assets/models.yaml 的 web 段（assets/aircraft 原始 OBJ）。
浏览器导出、俯视轮廓与资产兜底都从这里取同一份几何，只在需要时做体素抽稀。
"""
from __future__ import annotations

import math
from pathlib import Path

from dogfight.assets import load_catalog

# 轨迹采样间距：前端按距离抽点，避免几千段线拖慢每帧。
TRAIL_SPACING = 60.0
# 开火点抽稀：同一个点射不必每一帧都插一个标记。
SHOT_SPACING = 120.0
# 过载超过人体可持续耐受（6 g）即标注；可用过载不足时按 75% 也标注。
HIGH_G_LOAD = 6.0
HIGH_G_REL = 0.75
HIGH_G_COOLDOWN = 1.5
# 导弹的等效 RCS：弹体只有 3–4 m 长、0.2 m 粗，迎头回波约 0.05 m² 量级。
MISSILE_RCS = 0.05
FOV = math.radians(50.0)

_CATALOG = load_catalog()
# 屏幕尺寸统一参考长度：取注册表里最长的机体，其他机型按真实比例缩放。
REF_LENGTH = max(asset.length_m for asset in _CATALOG.aircraft.values())


def attitude_of(ac) -> tuple[float, float, float]:
    """视图用姿态 (psi, theta, phi)。点质量内核用航迹角与滚转角还原。"""
    if getattr(ac, "rigid", False):
        return ac.psi, ac.theta, ac.phi
    # 3DOF 的 bank>0 为左滚，而机体 phi>0 为右滚，故取负号。
    return ac.psi, ac.gamma, -ac.bank


def model_length(name: str) -> float:
    asset = _CATALOG.aircraft.get(name)
    if asset is not None:
        return asset.length_m
    return REF_LENGTH


def read_obj(path: str | Path, with_uv: bool = False):
    """读取 OBJ 的顶点与面（面记录为 (索引元组, 材质名)）。

    with_uv=True 时按 (顶点表, 面表, 角点 UV) 返回；角点 UV 与面表一一对应，
    每一项是该面各角点的 (u, v)，没有 vt 的模型为 None。
    """
    verts = []
    uvs = []
    faces = []
    face_uv = []
    material = "body"
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                verts.append((float(x), float(y), float(z)))
            elif line.startswith("vt "):
                parts = line.split()
                uvs.append((float(parts[1]), float(parts[2]) if len(parts) > 2 else 0.0))
            elif line.startswith("usemtl "):
                material = line.split(None, 1)[1].strip()
            elif line.startswith("f "):
                toks = line.split()[1:]
                idx = tuple(int(tok.split("/")[0]) - 1 for tok in toks)
                if len(idx) < 3:
                    continue
                faces.append((idx, material))
                if not with_uv:
                    continue
                corner = []
                for tok in toks:
                    bits = tok.split("/")
                    ti = int(bits[1]) - 1 if len(bits) > 1 and bits[1] else -1
                    corner.append(uvs[ti] if 0 <= ti < len(uvs) else None)
                face_uv.append(tuple(corner))
    if with_uv:
        return verts, faces, face_uv
    return verts, faces


def bbox(verts):
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))


def _quadric_positions(points, faces, cells, voxel):
    """用二次误差（QEM）决定每个体素聚类的代表点。

    纯体素平均（质心）会把一块平板上的顶点往格子中心拽，相邻格子错开，
    远看就是一层层台阶和凹坑。这里按 Garland-Heckbert 的思路，把每个面
    的平面二次型（按面积加权）累加到它三个顶点所在格子里，再解
    argmin xᵀAx + 2bᵀx，得到的点落在局部主平面上，平板保持平、曲面误差更小。
    """
    # 每个格子累加：A 对称矩阵 6 项 + b 3 项（quadric 为 4x4 的对称部分）
    quad: dict[tuple[int, int, int], list[float]] = {}
    for idx, _material in faces:
        if len(idx) < 3:
            continue
        ax, ay, az = points[idx[0]]
        bx, by, bz = points[idx[1]]
        cx, cy, cz2 = points[idx[2]]
        ux, uy, uz = bx - ax, by - ay, bz - az
        vx, vy, vz = cx - ax, cy - ay, cz2 - az
        nx = uy * vz - uz * vy
        ny = uz * vx - ux * vz
        nz = ux * vy - uy * vx
        area2 = nx * nx + ny * ny + nz * nz
        if area2 <= 1e-12:
            continue
        # 面积加权：叉积长度 = 2×面积，|n|·unit(n)/2 = n/2 即面积加权的平面向量
        nx, ny, nz = nx * 0.5, ny * 0.5, nz * 0.5
        w = -(nx * ax + ny * ay + nz * az)
        a, b, c = nx, ny, nz
        aa, ab, ac = a * a, a * b, a * c
        bb, bc, cc = b * b, b * c, c * c
        aw, bw, cw = a * w, b * w, c * w
        ww = w * w
        for i in idx:
            p = points[i]
            key = (round(p[0] / voxel), round(p[1] / voxel), round(p[2] / voxel))
            slot = quad.get(key)
            if slot is None:
                quad[key] = [aa, ab, ac, bb, bc, cc, aw, bw, cw, ww]
            else:
                slot[0] += aa; slot[1] += ab; slot[2] += ac
                slot[3] += bb; slot[4] += bc; slot[5] += cc
                slot[6] += aw; slot[7] += bw; slot[8] += cw
                slot[9] += ww
    pts = []
    for key, slot in cells.items():
        q = quad.get(key)
        cx = slot[0] / slot[3]
        cy = slot[1] / slot[3]
        cz = slot[2] / slot[3]
        if q is None:
            pts.append((cx, cy, cz))
            continue
        aa, ab, ac, bb, bc, cc, aw, bw, cw, _ww = q
        # 轻微 Tikhonov 正则：纯平面让 A 奇异，正则后解退化到“沿平面
        # 最靠近质心”的点，正好是想要的结果。
        lam = (aa + bb + cc) * 1e-4 + 1e-9
        a11, a12, a13 = aa + lam, ab, ac
        a22, a23 = bb + lam, bc
        a33 = cc + lam
        det = (a11 * (a22 * a33 - a23 * a23)
               - a12 * (a12 * a33 - a23 * a13)
               + a13 * (a12 * a23 - a22 * a13))
        if abs(det) < 1e-12:
            pts.append((cx, cy, cz))
            continue
        # Cramer：解 A x = -b
        b1, b2, b3 = -aw, -bw, -cw
        x1 = ((b1 * (a22 * a33 - a23 * a23)
               - a12 * (b2 * a33 - a23 * b3)
               + a13 * (b2 * a23 - a22 * b3)) / det)
        x2 = ((a11 * (b2 * a33 - a23 * b3)
               - b1 * (a12 * a33 - a13 * a23)
               + a13 * (a12 * b3 - b2 * a13)) / det)
        x3 = ((a11 * (a22 * b3 - b2 * a23)
               - a12 * (a12 * b3 - b2 * a13)
               + b1 * (a12 * a23 - a22 * a13)) / det)
        # 病态时解会飞出去：偏离质心超过一格半就退回质心。
        lim = voxel * 1.5
        if (abs(x1 - cx) > lim or abs(x2 - cy) > lim or abs(x3 - cz) > lim):
            pts.append((cx, cy, cz))
        else:
            pts.append((x1, x2, x3))
    # cells 的插入顺序就是 index 的顺序
    return pts


def cluster(points, faces, voxel, canopy_faces=frozenset(),
            store_faces=frozenset(), quadric: bool = False, uv_faces=None):
    """体素聚类抽稀：同格顶点合并到平均值，退化/重复面直接丢弃。

    faces 为 (索引元组, 材质名)；返回 (新顶点表, [(面索引, 面类别)], 角点 UV)，
    面类别为 "body" / "canopy" / "stores"（外挂导弹、副油箱等）。
    quadric=True 时聚类代表点按二次误差取在局部主平面上，避免平板台阶。
    uv_faces 与 faces 一一对应（来自 read_obj(with_uv=True)）；给出时返回的
    第三个元素与输出面一一对应，聚类合并掉的角点会一并去掉。
    """
    cells: dict[tuple[int, int, int], list[float]] = {}
    for x, y, z in points:
        key = (round(x / voxel), round(y / voxel), round(z / voxel))
        slot = cells.get(key)
        if slot is None:
            cells[key] = [x, y, z, 1]
        else:
            slot[0] += x
            slot[1] += y
            slot[2] += z
            slot[3] += 1
    index = {key: i for i, key in enumerate(cells)}
    if quadric:
        pts = _quadric_positions(points, faces, cells, voxel)
    else:
        pts = [(slot[0] / slot[3], slot[1] / slot[3], slot[2] / slot[3])
               for slot in cells.values()]
    remap = [index[(round(x / voxel), round(y / voxel), round(z / voxel))]
             for x, y, z in points]
    out = []
    out_uv = []
    seen = set()
    for n, (idx, _material) in enumerate(faces):
        corner = uv_faces[n] if uv_faces is not None else None
        poly = []
        poly_uv = []
        for k, i in enumerate(idx):
            c = remap[i]
            if c in poly:
                continue
            poly.append(c)
            if corner is not None and k < len(corner):
                poly_uv.append(corner[k])
        if len(poly) < 3:
            continue
        sig = tuple(sorted(poly))
        if sig in seen:
            continue
        seen.add(sig)
        # 座舱盖面同时也在外挂物集合里（玻璃属于非蒙皮材质），必须先判定，
        # 否则抽稀后座舱盖会被并进 stores，导出成不透明外挂物。
        if n in canopy_faces:
            kind = "canopy"
        elif n in store_faces:
            kind = "stores"
        else:
            kind = "body"
        out.append((tuple(poly), kind))
        if uv_faces is not None:
            complete = len(poly_uv) == len(poly) and all(uv is not None for uv in poly_uv)
            out_uv.append(tuple(poly_uv) if complete else None)
    return pts, out, (out_uv if uv_faces is not None else None)


def _convert_su57(verts, faces, length: float) -> dict:
    """SU-57：OBJ 纵轴 X、机头 -X；输出机体系（x 前、y 右、z 下）。"""
    (x0, x1), (y0, y1), (z0, z1) = bbox(verts)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    s = length / (x1 - x0)

    def conv(v):
        return (-(v[0] - cx) * s, -v[2] * s, -(v[1] - cy) * s)

    pts = [conv(v) for v in verts]
    engines = []
    # 双发喷口：尾部 x≈1.7–1.8、|z|≈0.18 的两个出口环。
    for sign in (1.0, -1.0):
        ring = [v for v in verts
                if 1.45 <= v[0] <= 1.90 and 0.10 <= sign * v[2] <= 0.30
                and -0.02 <= v[1] <= 0.15]
        if not ring:
            continue
        x_exit = max(v[0] for v in ring)
        tip = [v for v in ring if v[0] >= x_exit - 0.02]
        ey = sum(v[1] for v in tip) / len(tip)
        ez = sum(v[2] for v in tip) / len(tip)
        engines.append(list(conv((x_exit, ey, ez))))
    # 座舱盖只有 glass.001；screen / Glowing_green 是仪表面片，仍归 body。
    canopy = {n for n, (_idx, material) in enumerate(faces) if material == "glass.001"}
    return {"pts": pts, "faces": faces, "engines": engines,
            "canopy": canopy, "stores": set()}


def _convert_f16(verts, faces, length: float) -> dict:
    """F-16：OBJ 纵轴 Z、机头 +Z；输出机体系（x 前、y 右、z 下）。"""
    (x0, x1), (y0, y1), (z0, z1) = bbox(verts)
    cx, cy, cz = (x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0
    s = length / (z1 - z0)

    def conv(v):
        return ((v[2] - cz) * s, -(v[0] - cx) * s, -(v[1] - cy) * s)

    pts = [conv(v) for v in verts]
    exhaust = set()
    for idx, material in faces:
        if material == "EXHAUST":
            exhaust.update(idx)
    engines = []
    if exhaust:
        ep = [verts[i] for i in exhaust]
        z_exit = min(p[2] for p in ep)
        ring = [p for p in ep if p[2] <= z_exit + 0.01]
        ex = sum(p[0] for p in ring) / len(ring)
        ey = sum(p[1] for p in ring) / len(ring)
        engines.append(list(conv((ex, ey, z_exit))))
    # F-16 的玻璃材质混了整颗翼下导弹：GLASS2 全部、glass.002 的 ±1.9 侧块。
    # 只有靠近机体中线（|x|<0.8）的座舱盖归 canopy，其余玻璃面片留在 body，
    # 否则前端会把导弹当成透明座舱盖渲染成玻璃。
    canopy = set()
    for n, (idx, material) in enumerate(faces):
        if "glass" not in material.lower():
            continue
        if max(abs(verts[i][0] - cx) for i in idx) < 0.8:
            canopy.add(n)
    return {"pts": pts, "faces": faces, "engines": engines,
            "canopy": canopy, "stores": set()}


def _convert_j20(verts, faces, length: float) -> dict:
    """J-20：OBJ 纵轴 Z、机头 +Z、+Y 向上；输出机体系（x 前、y 右、z 下）。

    原始模型没有座舱/喷口分组：座舱盖按机背上的几何区域选取，双发喷口
    取发动机材质在尾端的两个出口环。J-20 无外挂（导弹在弹舱内）。
    """
    (x0, x1), (y0, y1), (z0, z1) = bbox(verts)
    cx, cy, cz = (x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0
    span_x, span_y, span_z = x1 - x0, y1 - y0, z1 - z0
    s = length / span_z

    def conv(v):
        return ((v[2] - cz) * s, -(v[0] - cx) * s, -(v[1] - cy) * s)

    pts = [conv(v) for v in verts]
    # 双发喷口：mat0 是尾段/喷管材质（z 192–240），出口环取该材质的最小 z。
    nozzle = set()
    for idx, material in faces:
        if material == "mat0":
            nozzle.update(idx)
    engines = []
    if nozzle:
        z_exit = min(verts[i][2] for i in nozzle)
        ring = [verts[i] for i in nozzle
                if verts[i][2] <= z_exit + 0.005 * span_z]
        for left in (True, False):
            side = [p for p in ring if (p[0] < cx) == left]
            if not side:
                continue
            ex = sum(p[0] for p in side) / len(side)
            ey = sum(p[1] for p in side) / len(side)
            engines.append(list(conv((ex, ey, z_exit))))
    # 座舱盖：按三视图量取，机头起约 17%–31%（即尾部起 69%–82.7%）、
    # 机背 y ≥ 50.8%、中线附近 ±8.5% 翼展的蒙皮面。
    canopy = set()
    z_lo, z_hi = z0 + 0.690 * span_z, z0 + 0.827 * span_z
    y_lo = y0 + 0.508 * span_y
    x_lim = 0.085 * span_x
    for n, (idx, _material) in enumerate(faces):
        poly = [verts[i] for i in idx]
        cz_f = sum(p[2] for p in poly) / len(poly)
        cy_f = sum(p[1] for p in poly) / len(poly)
        if (z_lo <= cz_f <= z_hi and cy_f >= y_lo
                and max(abs(p[0] - cx) for p in poly) <= x_lim):
            canopy.add(n)
    return {"pts": pts, "faces": faces, "engines": engines,
            "canopy": canopy, "stores": set()}


def _convert_f15(verts, faces, length: float) -> dict:
    """F-15C / F-15E / F-15EX 共用：OBJ 纵轴 Z、机头 +Z、+Y 向上。

    原始模型只按贴图分材质，没有座舱、导弹、副油箱的分组：
    - 机体蒙皮占绝大多数面（F-15C 为 db77f96e_dds，F-15EX 为 cba3dd14_dds），
      其余材质的面（AIM-120 级中距弹、翼尖 AIM-9）一律归为外挂物 stores；
    - 座舱盖按机背上的几何区域（机头起 24.4%–33.4%，即尾部起 66.2%–75.3%；
      y ≥ 51.9%、中线 ±0.6 m）选取；
    - 双发喷口取机腹后段两个出口环。
    """
    (x0, x1), (y0, y1), (z0, z1) = bbox(verts)
    cx, cy, cz = (x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0
    span_x, span_y, span_z = x1 - x0, y1 - y0, z1 - z0
    s = length / span_z

    def conv(v):
        return ((v[2] - cz) * s, -(v[0] - cx) * s, -(v[1] - cy) * s)

    pts = [conv(v) for v in verts]
    # 面数最多的材质是机体蒙皮，其余材质是挂载（导弹/挂架/吊舱）。
    counts: dict[str, int] = {}
    for _idx, material in faces:
        counts[material] = counts.get(material, 0) + 1
    body_material = max(counts, key=counts.get) if counts else "body"
    stores = {n for n, (_idx, material) in enumerate(faces)
              if material != body_material}

    # 双发喷口：机腹后段（下 30% 高度、中线 ±10% 翼展）最靠后的两个出口环。
    # 横向不能放宽：再宽一点就会把平尾根部的后缘当成"最靠后的出口"。
    engines = []
    body_verts = set()
    for n, (idx, _material) in enumerate(faces):
        if n not in stores:
            body_verts.update(idx)
    rear = [verts[i] for i in body_verts
            if abs(verts[i][0] - cx) <= 0.10 * span_x
            and verts[i][1] <= y0 + 0.30 * span_y]
    if rear:
        z_exit = min(p[2] for p in rear)
        ring = [p for p in rear if p[2] <= z_exit + 0.005 * span_z]
        for left in (True, False):
            side = [p for p in ring if (p[0] < cx) == left]
            if not side:
                continue
            ex = sum(p[0] for p in side) / len(side)
            ey = sum(p[1] for p in side) / len(side)
            engines.append(list(conv((ex, ey, z_exit))))

    # 座舱盖：按三视图量取，机头起 24.4%–33.4%（即尾部起 66.2%–75.3%）、
    # 机背 y ≥ 51.9%、中线 ±0.6 m 的蒙皮面。
    canopy = set()
    z_lo, z_hi = z0 + 0.662 * span_z, z0 + 0.753 * span_z
    y_lo = y0 + 0.519 * span_y
    x_lim = 0.031 * span_z
    for n, (idx, _material) in enumerate(faces):
        if n in stores:
            continue
        poly = [verts[i] for i in idx]
        cz_f = sum(p[2] for p in poly) / len(poly)
        cy_f = sum(p[1] for p in poly) / len(poly)
        if (z_lo <= cz_f <= z_hi and cy_f >= y_lo
                and max(abs(p[0] - cx) for p in poly) <= x_lim):
            canopy.add(n)
    return {"pts": pts, "faces": faces, "engines": engines,
            "canopy": canopy, "stores": stores}


def _convert_flanker(verts, faces, length: float) -> dict:
    """J-16 / J-15T（侧卫家族）：OBJ 纵轴 X、机头 +X、+Y 向上。

    输出机体系（x 前、y 右、z 下）。两个模型都带完整挂载（导弹、吊舱、挂架），
    机体蒙皮是面数最多的材质，其余材质一律归 stores；座舱盖按机背上
    机头起 12%–28%、中线附近的蒙皮/玻璃面选取。
    """
    (x0, x1), (y0, y1), (z0, z1) = bbox(verts)
    cx, cy, cz = (x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0
    span_x, span_y, span_z = x1 - x0, y1 - y0, z1 - z0
    s = length / span_x

    def conv(v):
        return ((v[0] - cx) * s, -(v[2] - cz) * s, -(v[1] - cy) * s)

    pts = [conv(v) for v in verts]
    counts: dict[str, int] = {}
    for _idx, material in faces:
        counts[material] = counts.get(material, 0) + 1
    body_material = max(counts, key=counts.get) if counts else "body"
    stores = {n for n, (_idx, material) in enumerate(faces)
              if material != body_material}

    # 双发喷口：在机尾的发动机带（横向 4%–17% 翼展、竖向靠下）里，
    # 从尾端往前扫 0.5 m 窗口，第一个装满足够顶点的窗口就是喷口环；
    # 取该窗口内顶点的 (y, z) 质心作为喷口中心。不能再用“机尾最前点 ±
    # 18% 翼展”估算——侧卫的尾梁伸在喷口之后，那个点落在尾梁上，喷口会
    # 偏到发动机舱外侧（此前 J-16 就是这样）。
    engines = []
    body_verts = set()
    for n, (idx, _material) in enumerate(faces):
        if n not in stores:
            body_verts.update(idx)
    win = 0.5
    need = 250
    for sign in (1.0, -1.0):
        band = sorted(
            (verts[i] for i in body_verts
             if 0.04 * span_z <= sign * (verts[i][2] - cz) <= 0.17 * span_z
             and y0 + 0.05 * span_y <= verts[i][1] <= y0 + 0.40 * span_y),
            key=lambda p: p[0],
        )
        x_exit = None
        j = 0
        for i, p in enumerate(band):
            j = max(j, i)
            while j + 1 < len(band) and band[j + 1][0] <= p[0] + win:
                j += 1
            if j - i + 1 >= need:
                x_exit = p[0]
                break
        if x_exit is None:
            continue
        slab = [p for p in band if x_exit <= p[0] <= x_exit + win]
        ey = sum(p[1] for p in slab) / len(slab)
        ez = sum(p[2] for p in slab) / len(slab)
        engines.append(list(conv((x_exit, ey, ez))))
    engines.sort(key=lambda e: e[1])

    # 座舱盖：机头起 12%–28%、机背 y ≥ 50% 高度、中线 ±5% 翼展的玻璃面。
    # 这批模型的座舱区域里蒙皮与玻璃是分开的材质，只挑玻璃面，避免把
    # 机背蒙皮也渲染成透明件。
    canopy = set()
    x_hi, x_lo = x1 - 0.12 * span_x, x1 - 0.28 * span_x
    y_lo = y0 + 0.50 * span_y
    z_lim = 0.05 * span_z
    for n, (idx, material) in enumerate(faces):
        if "glass" not in material:
            continue
        poly = [verts[i] for i in idx]
        cx_f = sum(p[0] for p in poly) / len(poly)
        cy_f = sum(p[1] for p in poly) / len(poly)
        if (x_lo <= cx_f <= x_hi and cy_f >= y_lo
                and max(abs(p[2] - cz) for p in poly) <= z_lim):
            canopy.add(n)
    return {"pts": pts, "faces": faces, "engines": engines,
            "canopy": canopy, "stores": stores, "body_material": body_material}


def _convert_default(verts, faces, length: float, forward: str = "+x") -> dict:
    """新增机型的默认换算：把指定轴当作纵轴，输出机体系（x 前、y 右、z 下）。

    喷口与座舱暂留空，待对应 OBJ 到位后在专属 converter 里按材质名与几何
    特征补全。forward 可选 "+x" / "-x" / "+z" / "-z"。
    """
    (x0, x1), (y0, y1), (z0, z1) = bbox(verts)
    cx, cy, cz = (x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0
    if forward in ("+x", "-x"):
        span = x1 - x0
        s = length / span
        sign = 1.0 if forward == "+x" else -1.0
        def conv(v):
            return (sign * (v[0] - cx) * s, -(v[2] - cz) * s, -(v[1] - cy) * s)
    else:
        span = z1 - z0
        s = length / span
        sign = 1.0 if forward == "+z" else -1.0
        def conv(v):
            return (sign * (v[2] - cz) * s, -(v[0] - cx) * s, -(v[1] - cy) * s)
    pts = [conv(v) for v in verts]
    return {"pts": pts, "faces": faces, "engines": [],
            "canopy": set(), "stores": set()}


# 原始 OBJ 的轴向约定不统一，按注册表里的 converter 名分派（新增机型在此登记）。
CONVERTERS = {
    "su57": _convert_su57,
    "f16": _convert_f16,
    "j20": _convert_j20,
    "f15": _convert_f15,
    "j16": _convert_flanker,
    "j15t": _convert_flanker,
}


def convert_model(path: str | Path, converter: str, length: float) -> dict:
    """读取原始 OBJ 并换算到机体系；返回 length / pts / faces / engines / canopy。"""
    try:
        convert = CONVERTERS[converter]
    except KeyError as exc:
        raise KeyError(f"未知 converter: {converter}") from exc
    verts, faces, face_uv = read_obj(path, with_uv=True)
    geo = convert(verts, faces, length)
    geo["length"] = length
    geo["uv_faces"] = face_uv
    return geo


def load_model(path: str | Path, converter: str, length: float, voxel: float,
               keep_materials: set | None = None) -> dict:
    """换算后按体素抽稀，供资产缺失时的 Web 兜底与俯视轮廓使用。

    外挂物（导弹/副油箱等）不参与简化：俯视轮廓只画机体，挂载由前端
    单独渲染；因此 stores 面在这里直接剔除。
    """
    geo = convert_model(path, converter, length)
    stores = geo.get("stores", set())
    faces = []
    canopy = set()
    for n, (idx, material) in enumerate(geo["faces"]):
        if n in stores:
            continue
        if keep_materials is not None and material not in keep_materials:
            continue
        if n in geo["canopy"]:
            canopy.add(len(faces))
        faces.append((idx, material))
    pts, clustered, _uv = cluster(geo["pts"], faces, voxel, canopy)
    return {
        "length": length,
        "faces": [([pts[c] for c in poly], kind) for poly, kind in clustered],
        "engines": geo["engines"],
    }


def _build_models() -> dict[str, dict]:
    models: dict[str, dict] = {}
    for name, asset in _CATALOG.aircraft.items():
        # 新增机型的 OBJ 尚未就位时跳过，前端会退回通用兜底模型；
        # 避免因单个缺失文件导致整个观战端导入失败。
        if not asset.web.source.exists():
            continue
        models[name] = load_model(
            asset.web.source, asset.web.converter, asset.length_m,
            asset.voxel_m, asset.keep_materials,
        )
    return models


MODEL_BY_NAME = _build_models()
# 场景里的机型名未登记到资产表时，退回第一架已注册机体，避免观战端直接崩。
GENERIC_MODEL_NAME = next(iter(MODEL_BY_NAME))
GENERIC_MODEL = MODEL_BY_NAME[GENERIC_MODEL_NAME]
