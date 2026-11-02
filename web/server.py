"""Dogfight Web 观战服务：复用现有内核，用 HTTP + SSE 把快照推给浏览器。

只依赖标准库（http.server / socketserver / threading）与仓库内已有的
main / render / sim / theme 模块；three.js 由前端本地 vendor 文件提供。

启动：
    /home/lky/anaconda3/envs/dogfight/bin/python web/server.py
    python web/server.py --host 127.0.0.1 --port 8765 --seconds 300 --seed 7
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = Path(__file__).resolve().parent
STATIC = WEB / "static"
MODEL_DIR = STATIC / "models"
# 前端固定分辨率采集（/?capture=1920x1080）上传的 PNG 帧与编码后的 MP4。
CAPTURE_DIR = WEB / "captures"
sys.path.insert(0, str(ROOT))

# 导入 render 会读取 OBJ、做体素简化（启动花几秒属正常）；这里只取模型
# 数据与少量纯函数，不创建 Renderer、不初始化 pygame 窗口。
import main  # noqa: E402
import render  # noqa: E402
import theme  # noqa: E402
from sim import (  # noqa: E402
    CONTACT_DETECT, CONTACT_TRACK, G, RADAR_DETECT_FOV, Action,
    air_density, body_axes, clamp, gun_direction, nose_direction,
    predict_position, rcs_scale, world_velocity, wrap_pi,
)

SNAPSHOT_HZ = 24.0
REF_LENGTH = render.REF_LENGTH
MISSILE_RCS = render.MISSILE_RCS


# ---------------------------------------------------------------------------
# Web 模型导出：直接读取 assets 下的原始 OBJ，转到 three.js 机体系
# （机头 -Z、上 +Y），只服务 Web 观战端；pygame 继续用 render.py 的模型。
# ---------------------------------------------------------------------------

ASSET_ROOT = ROOT / "assets"
# 导出版本号：改动坐标换算/抽稀参数时递增，旧的 web/static/models 会自动重建。
MODEL_TAG = "web-obj-v3"

AIRCRAFT_EXPORTS = (
    # (机体名, 源文件, 真实机长 m)
    ("SU-57", ASSET_ROOT / "flighter" / "SU57.obj", 20.1),
    ("F-16V", ASSET_ROOT / "flighter" / "F16.obj", 15.06),
)
MISSILE_EXPORTS = (
    # (导弹名, 源文件, 真实弹长 m, 导出文件名)
    ("R-73", ASSET_ROOT / "missle" / "R73.obj", 2.90, "r73.obj"),
    ("AIM-9X", ASSET_ROOT / "missle" / "AIM9.obj", 3.02, "aim9.obj"),
    ("R-77-1", ASSET_ROOT / "missle" / "R77.obj", 3.60, "r77.obj"),
    ("AIM-120C", ASSET_ROOT / "missle" / "AIM120.obj", 3.66, "aim120.obj"),
)


def _three_vertex(p):
    """机体系点 → three.js 坐标：X = y, Y = -z, Z = -x。"""
    return (p[1], -p[2], -p[0])


def _export_model(model: dict, path: Path, name: str) -> None:
    """旧版兜底：render 的体素模型 → web OBJ（原始资产缺失时才会用到）。"""
    lines = [f"# dogfight web export: {name}", f"o {name}"]
    vcount = 0
    for poly, kind, _ao in model["faces"]:
        lines.append(f"usemtl {kind}")
        base = vcount + 1
        for p in poly:
            x, y, z = _three_vertex(p)
            lines.append(f"v {x:.4f} {y:.4f} {z:.4f}")
            vcount += 1
        lines.append("f " + " ".join(str(base + k) for k in range(len(poly))))
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_obj(path: Path):
    """读取 OBJ 的顶点与面（面记录为 (索引元组, 材质名)）。"""
    verts = []
    faces = []
    material = "body"
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                verts.append((float(x), float(y), float(z)))
            elif line.startswith("usemtl "):
                material = line.split(None, 1)[1].strip()
            elif line.startswith("f "):
                idx = tuple(int(tok.split("/")[0]) - 1 for tok in line.split()[1:])
                if len(idx) >= 3:
                    faces.append((idx, material))
    return verts, faces


def _cluster(verts, voxel, faces):
    """体素聚类抽稀：同格顶点合并到平均值，退化/重复面直接丢弃。"""
    cells = {}
    for x, y, z in verts:
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
    pts = [(slot[0] / slot[3], slot[1] / slot[3], slot[2] / slot[3])
           for slot in cells.values()]
    remap = [index[(round(x / voxel), round(y / voxel), round(z / voxel))]
             for x, y, z in verts]
    out = []
    seen = set()
    for idx, material in faces:
        polys = []
        for i in idx:
            c = remap[i]
            if c not in polys:
                polys.append(c)
        if len(polys) < 3:
            continue
        sig = tuple(sorted(polys))
        if sig in seen:
            continue
        seen.add(sig)
        out.append((tuple(polys), material))
    return pts, out


def _write_obj(path: Path, pts, faces, name: str, canopy_faces=None) -> None:
    """写 three.js 用 OBJ；canopy_faces 是 faces 中要归入 canopy 的面序号。"""
    body = []
    canopy = []
    for n, (idx, _material) in enumerate(faces):
        if canopy_faces and n in canopy_faces:
            canopy.append(idx)
        else:
            body.append(idx)
    lines = [f"# dogfight web {MODEL_TAG}", f"o {name}"]
    for x, y, z in pts:
        lines.append(f"v {x:.3f} {y:.3f} {z:.3f}")
    for kind, group in (("body", body), ("canopy", canopy)):
        if not group:
            continue
        lines.append(f"usemtl {kind}")
        for idx in group:
            lines.append("f " + " ".join(str(i + 1) for i in idx))
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _bbox(verts):
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    zs = [v[2] for v in verts]
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))


def _export_su57(path: Path):
    """SU-57：OBJ 纵轴 X、机头 -X；换算后机头朝 three 的 -Z。"""
    verts, faces = _read_obj(path)
    (x0, x1), (y0, y1), (z0, z1) = _bbox(verts)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    s = 20.1 / (x1 - x0)

    def conv(v):
        return (-v[2] * s, (v[1] - cy) * s, (v[0] - cx) * s)

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
    return pts, faces, engines, canopy


def _export_f16(path: Path):
    """F-16：OBJ 纵轴 Z、机头 +Z；换算后机头朝 three 的 -Z。"""
    verts, faces = _read_obj(path)
    (x0, x1), (y0, y1), (z0, z1) = _bbox(verts)
    cx, cy, cz = (x0 + x1) / 2.0, (y0 + y1) / 2.0, (z0 + z1) / 2.0
    s = 15.06 / (z1 - z0)

    def conv(v):
        return (-(v[0] - cx) * s, (v[1] - cy) * s, -(v[2] - cz) * s)

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
    return pts, faces, engines, canopy


def _export_missile(path: Path, length: float):
    """导弹：长轴弹头落到局部原点、弹尾朝 +Z，按真实弹长缩放并抽稀。

    平面网格太密（AIM-120 有 76 万面），按弹长 1/300 的体素聚类，保留
    弹头锥度、翼面与尾喷口的轮廓即可。
    """
    verts, faces = _read_obj(path)
    bb = _bbox(verts)
    axis = max(range(3), key=lambda k: bb[k][1] - bb[k][0])
    lo, hi = bb[axis]
    others = [k for k in range(3) if k != axis]
    centers = [(bb[k][0] + bb[k][1]) / 2.0 for k in others]

    def radial(p):
        return sum((p[k] - centers[j]) ** 2 for j, k in enumerate(others)) ** 0.5

    # 两端各取 3% 的薄片：最大径向半径更小的一端是弹头（尖锥），
    # 半径大的另一端是尾喷口/尾翼。
    slab = (hi - lo) * 0.03
    r_lo = max(radial(v) for v in verts if v[axis] <= lo + slab)
    r_hi = max(radial(v) for v in verts if v[axis] >= hi - slab)
    nose_hi = r_hi < r_lo
    scale = length / (hi - lo)

    def conv(v):
        along = (hi - v[axis]) if nose_hi else (v[axis] - lo)
        z = along * scale
        return tuple((v[k] - centers[j]) * scale for j, k in enumerate(others)) + (z,)

    pts = [conv(v) for v in verts]
    pts, faces = _cluster(pts, max(0.004, length / 300.0), faces)
    return pts, faces


def _exported_ok(path: Path) -> bool:
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.readline().strip() == f"# dogfight web {MODEL_TAG}"
    except OSError:
        return False


def export_models() -> dict:
    """导出/复用 assets 原始模型，返回前端所需元数据。"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    meta_path = MODEL_DIR / "meta.json"
    required = [MODEL_DIR / "su57.obj", MODEL_DIR / "f16.obj"] + [
        MODEL_DIR / spec[3] for spec in MISSILE_EXPORTS
    ]
    sources = [spec[1] for spec in AIRCRAFT_EXPORTS] + [spec[1] for spec in MISSILE_EXPORTS]
    if all(spec.exists() for spec in sources):
        if meta_path.exists() and all(_exported_ok(p) for p in required):
            try:
                return json.loads(meta_path.read_text(encoding="utf-8"))["models"]
            except (OSError, ValueError, KeyError):
                pass
        aircraft = {}
        for name, source, length in AIRCRAFT_EXPORTS:
            if name == "SU-57":
                pts, faces, engines, canopy = _export_su57(source)
            else:
                pts, faces, engines, canopy = _export_f16(source)
            _write_obj(MODEL_DIR / ("su57.obj" if name == "SU-57" else "f16.obj"),
                       pts, faces, name.lower().replace("-", ""), canopy)
            aircraft[name] = {
                "url": f"/api/models/{'su57' if name == 'SU-57' else 'f16'}.obj",
                "length": length,
                "engines": engines,
            }
        missiles = {}
        for name, source, length, filename in MISSILE_EXPORTS:
            pts, faces = _export_missile(source, length)
            _write_obj(MODEL_DIR / filename, pts, faces, filename[:-4])
            missiles[name] = {
                "url": f"/api/models/{filename}",
                "length": length,
            }
        meta = {"version": MODEL_TAG, "models": {"aircraft": aircraft,
                                                 "missiles": missiles}}
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        return meta["models"]

    # 原始资产缺失时退回 render 的体素模型，保证服务仍能启动。
    for filename, model, name in (
        ("su57.obj", render.SU57_MODEL, "su57"),
        ("f16.obj", render.F16_MODEL, "f16"),
    ):
        _export_model(model, MODEL_DIR / filename, name)

    def engines(model):
        return [list(_three_vertex(p)) for p in model.get("engines", [])]

    return {
        "aircraft": {
            "SU-57": {
                "url": "/api/models/su57.obj",
                "length": render.SU57_MODEL["length"],
                "engines": engines(render.SU57_MODEL),
            },
            "F-16V": {
                "url": "/api/models/f16.obj",
                "length": render.F16_MODEL["length"],
                "engines": engines(render.F16_MODEL),
            },
        },
        "missiles": {},
    }


# ---------------------------------------------------------------------------
# 航迹 / 事件记录：完全复用 render 的采样与门限规则
# ---------------------------------------------------------------------------

class TrailRecorder:
    def __init__(self, engine) -> None:
        self.engine = engine
        self.reset()

    def reset(self) -> None:
        self.trails = [[], []]
        self.axes = [[], []]
        self.ts = [[], []]
        self.fire_marks = [[], []]
        self.launch_marks = []
        self.highg_marks = []
        self._seen_missiles = set()
        self._highg_state = [False, False]
        self._last_highg = [-1e9, -1e9]
        self._last_sample = [None, None]
        self._last_shot = [None, None]
        self.missile_trails = {}

    def record(self, world, agents) -> None:
        for i, ac in enumerate(world.acs):
            if not ac.alive:
                continue
            p = (ac.x, ac.y, ac.z)
            last = self._last_sample[i]
            if last is None or math.dist(last, p) >= render.TRAIL_SPACING:
                self.trails[i].append(p)
                _, yb, _ = body_axes(*render.attitude_of(ac))
                self.axes[i].append(yb)
                self.ts[i].append(world.t)
                self._last_sample[i] = p
            if ac.firing:
                last_shot = self._last_shot[i]
                if last_shot is None or math.dist(last_shot, p) >= render.SHOT_SPACING:
                    self.fire_marks[i].append(p)
                    self._last_shot[i] = p

        for m in world.missiles:
            if m.alive and id(m) not in self._seen_missiles:
                self._seen_missiles.add(id(m))
                self.launch_marks.append((m.owner, (m.x, m.y, m.z)))

        for i, ac in enumerate(world.acs):
            primitive = getattr(agents[i], "primitive", "") if agents else ""
            n_avail = ac.spec.n_available(ac.v, ac.z, ac.mass) if ac.alive else 0.0
            hi = ac.alive and (
                ac.n >= render.HIGH_G_LOAD
                or (primitive.startswith("HIGH_G")
                    and ac.n >= render.HIGH_G_REL * n_avail)
            )
            if (hi and not self._highg_state[i]
                    and world.t - self._last_highg[i] > render.HIGH_G_COOLDOWN):
                self._last_highg[i] = world.t
                self.highg_marks.append((i, (ac.x, ac.y, ac.z)))
            self._highg_state[i] = hi

        live_ids = {id(m) for m in world.missiles if m.alive}
        for key in [k for k in self.missile_trails if k not in live_ids]:
            del self.missile_trails[key]
        for m in world.missiles:
            if not m.alive:
                continue
            pts = self.missile_trails.setdefault(id(m), [])
            p = (m.x, m.y, m.z)
            if not pts or math.dist(pts[-1], p) >= 25.0:
                pts.append(p)
                if len(pts) > 160:
                    del pts[0]


# ---------------------------------------------------------------------------
# 轨道相机：复刻 render.Renderer.update_camera 的平滑 / 跟随规则
# ---------------------------------------------------------------------------

class CameraState:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.cam = [0.0, 0.0, 1000.0]
        self.yaw = math.pi / 2.0
        self.pitch = math.radians(26.0)
        self.dist = 6000.0
        self.eye = (0.0, 6000.0, 3000.0)
        self.follow = None
        self.follow_dist = 1800.0
        self.span_factor = 1.25
        self.ready = False

    def update(self, world) -> None:
        follow_ac = None if self.follow is None else world.acs[self.follow]
        alive = [ac for ac in world.acs if ac.alive] or world.acs
        if follow_ac is not None:
            cx, cy, cz = follow_ac.x, follow_ac.y, follow_ac.z
            target_dist = self.follow_dist
        else:
            k = len(alive)
            cx = sum(ac.x for ac in alive) / k
            cy = sum(ac.y for ac in alive) / k
            cz = sum(ac.z for ac in alive) / k
            a, b = world.acs
            sep_xy = math.hypot(b.x - a.x, b.y - a.y)
            span = max(sep_xy, abs(b.z - a.z) * 2.0, 1400.0)
            target_dist = clamp(span * self.span_factor, 2200.0, 320000.0)
        if self.ready:
            self.dist += (target_dist - self.dist) * 0.06
            self.cam[0] += (cx - self.cam[0]) * 0.10
            self.cam[1] += (cy - self.cam[1]) * 0.10
            self.cam[2] += (cz - self.cam[2]) * 0.08
        else:
            self.cam = [cx, cy, cz]
            self.dist = target_dist
            self.ready = True

        if follow_ac is not None:
            desired = follow_ac.psi + math.pi
            self.yaw += clamp(wrap_pi(desired - self.yaw), -0.06, 0.06)
            self.pitch += (math.radians(16.0) - self.pitch) * 0.02
        else:
            a, b = world.acs
            sep_xy = math.hypot(b.x - a.x, b.y - a.y)
            if sep_xy > 500.0:
                desired = math.atan2(b.y - a.y, b.x - a.x) + math.pi / 2.0 - 0.35
                self.yaw += clamp(wrap_pi(desired - self.yaw), -0.03, 0.03)

        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        self.eye = (
            self.cam[0] + self.dist * cp * math.cos(self.yaw),
            self.cam[1] + self.dist * cp * math.sin(self.yaw),
            self.cam[2] + self.dist * sp,
        )

    def snapshot(self) -> dict:
        return {
            "cx": round(self.cam[0], 1),
            "cy": round(self.cam[1], 1),
            "cz": round(self.cam[2], 1),
            "yaw": round(self.yaw, 5),
            "pitch": round(self.pitch, 5),
            "dist": round(self.dist, 1),
            "eye": [round(v, 1) for v in self.eye],
            "fov": math.degrees(render.FOV),
            "follow": self.follow,
            "follow_dist": self.follow_dist,
        }


# ---------------------------------------------------------------------------
# 快照构建
# ---------------------------------------------------------------------------

def _hex(rgb) -> str:
    return "#%02x%02x%02x" % tuple(int(max(0, min(255, c))) for c in rgb)


def _mix(a, b, t):
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def _p(p, digits=1):
    return [round(float(v), digits) for v in p]


def _lock_state(ac):
    red, yellow, green = (255, 80, 72), (255, 212, 96), (116, 232, 150)
    if not ac.alive:
        return red, "LOST"
    if ac.radar_locked:
        return red, "LOCK"
    if ac.radar_lock_timer > 0.0:
        return yellow, "LOCKING"
    if ac.contact >= CONTACT_TRACK:
        return green, "TRACK"
    if ac.contact == CONTACT_DETECT or ac.rwr:
        return green, "DETECT"
    return (86, 170, 118), "SEARCH"


def _gun_fall_point(world, i):
    """复刻 render._gun_fall_point：当前机头指向下的机炮弹道落点。"""
    ac = world.acs[i]
    foe = world.acs[1 - i]
    spec = ac.spec
    if foe.alive:
        dist = math.sqrt(
            (foe.x - ac.x) ** 2 + (foe.y - ac.y) ** 2 + (foe.z - ac.z) ** 2
        )
    else:
        dist = spec.gun_range
    dist = clamp(dist, 300.0, spec.gun_range)
    nx, ny, nz = nose_direction(ac)
    vx, vy, vz = world_velocity(ac)
    area = math.pi * (0.5 * spec.bullet_diameter) ** 2
    drag_k = air_density(ac.z) * 0.30 * area / (2.0 * spec.bullet_mass)
    bx_ = ac.x + nx * spec.length * 0.5
    by_ = ac.y + ny * spec.length * 0.5
    bz_ = ac.z + nz * spec.length * 0.5
    bvx = vx + spec.muzzle_speed * nx
    bvy = vy + spec.muzzle_speed * ny
    bvz = vz + spec.muzzle_speed * nz
    dt = 1.0 / 120.0
    for _ in range(240):
        sp = math.sqrt(bvx * bvx + bvy * bvy + bvz * bvz)
        bvx -= drag_k * sp * bvx * dt
        bvy -= drag_k * sp * bvy * dt
        bvz -= (drag_k * sp * bvz + G) * dt
        bx_ += bvx * dt
        by_ += bvy * dt
        bz_ += bvz * dt
        r2 = (bx_ - ac.x) ** 2 + (by_ - ac.y) ** 2 + (bz_ - ac.z) ** 2
        if r2 >= dist * dist:
            break
    return (bx_, by_, bz_)


def _missile_threat(world, i):
    ac = world.acs[i]
    best = None
    for m in world.missiles:
        if not m.alive or m.target != i:
            continue
        dx, dy, dz = m.x - ac.x, m.y - ac.y, m.z - ac.z
        d = math.sqrt(dx * dx + dy * dy + dz * dz)
        az = wrap_pi(math.atan2(dy, dx) - ac.psi)
        if best is None or d < best[1]:
            best = (az, d)
    return best


def _scope(world, i, agent) -> dict:
    ac = world.acs[i]
    foe = world.acs[1 - i]
    lock_color, lock_text = _lock_state(ac)
    d = math.dist((ac.x, ac.y, ac.z), (foe.x, foe.y, foe.z))
    if ac.contact >= CONTACT_TRACK and d < 1e8:
        rng = clamp(d * 1.35, 4000.0, max(ac.spec.radar_range, 40000.0))
    else:
        rng = max(ac.spec.radar_range, 40000.0) * 0.6

    rel = wrap_pi(math.atan2(foe.y - ac.y, foe.x - ac.x) - ac.psi)
    target = None
    if ac.contact >= CONTACT_TRACK and foe.alive:
        target = {
            "az": rel,
            "rr": clamp(d / rng, 0.0, 0.96),
            "d": d,
            "dz": foe.z - ac.z,
        }
    detect = None
    if ac.contact == CONTACT_DETECT or ac.rwr:
        detect = {
            "az": ac.rwr_az if ac.rwr else rel,
            "rwr": bool(ac.rwr),
        }

    blips = []
    if ac.radar_on and ac.spec.radar_range > 0.0:
        nnx, nny, nnz = nose_direction(ac)
        reach = ac.spec.radar_range * rcs_scale(MISSILE_RCS)
        for m in world.missiles:
            if not m.alive:
                continue
            mx, my, mz = m.x - ac.x, m.y - ac.y, m.z - ac.z
            md = math.sqrt(mx * mx + my * my + mz * mz)
            if md < 1e-6 or md > reach:
                continue
            if math.acos(clamp((mx * nnx + my * nny + mz * nnz) / md, -1.0, 1.0)) \
                    > RADAR_DETECT_FOV:
                continue
            blips.append({
                "az": wrap_pi(math.atan2(my, mx) - ac.psi),
                "rr": clamp(md / rng, 0.0, 0.96),
                "hdg": wrap_pi(math.atan2(m.vy, m.vx) - ac.psi),
                "md": md,
                "incoming": m.target == i,
            })

    threat = _missile_threat(world, i)
    incoming = sum(1 for m in world.missiles if m.alive and m.target == i)
    return {
        "rng": rng,
        "rings": [0.25, 0.5, 0.75, 1.0],
        "labels": [round(rng * f / 1000.0) for f in (0.5, 1.0)],
        "rel": rel,
        "contact": ac.contact,
        "foe_alive": foe.alive,
        "target": target,
        "detect": detect,
        "blips": blips,
        "lock_color": _hex(lock_color),
        "lock_text": lock_text,
        "fov": math.degrees(RADAR_DETECT_FOV),
        "threat": None if threat is None else {"az": threat[0], "d": threat[1]},
        "incoming": incoming,
        "situation": getattr(agent, "situation", "") if agent else "",
    }


def _hud(world, i, agent) -> dict:
    ac = world.acs[i]
    foe = world.acs[1 - i]
    lock_color, lock_text = _lock_state(ac)
    psi, theta, phi = render.attitude_of(ac)
    dx, dy, dz = foe.x - ac.x, foe.y - ac.y, foe.z - ac.z
    dist = math.sqrt(dx * dx + dy * dy + dz * dz)
    rel = wrap_pi(math.atan2(dy, dx) - ac.psi)
    elev = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
    threat = _missile_threat(world, i)
    fall = _gun_fall_point(world, i)
    rx, ry, rz = fall[0] - ac.x, fall[1] - ac.y, fall[2] - ac.z
    g_az = math.degrees(wrap_pi(math.atan2(ry, rx) - ac.psi))
    g_el = math.degrees(math.atan2(rz, math.hypot(rx, ry)))
    brg = None
    if ac.contact == CONTACT_DETECT or ac.rwr:
        brg = (math.degrees(ac.psi) + math.degrees(rel)) % 360.0
    return {
        "heading": math.degrees(ac.psi) % 360.0,
        "v": ac.v,
        "z": ac.z,
        "theta": math.degrees(theta),
        "phi": math.degrees(phi),
        "rel": math.degrees(rel),
        "rel_rad": rel,
        "elev": elev,
        "dist": dist,
        "foe_alive": foe.alive,
        "contact": ac.contact,
        "rwr": bool(ac.rwr),
        "rwr_az": ac.rwr_az,
        "lock_color": _hex(lock_color),
        "lock_text": lock_text,
        "bearing": brg,
        "gun_az": g_az,
        "gun_el": g_el,
        "threat": None if threat is None else {
            "az": threat[0], "d": threat[1],
        },
        "primitive": getattr(agent, "primitive", "") if agent else "",
        "tactic": getattr(agent, "tactic", "") if agent else "",
    }


def _weapon(world, i) -> dict:
    ac = world.acs[i]
    inflight = sum(1 for m in world.missiles if m.alive and m.owner == i)
    threat = _missile_threat(world, i)
    if threat is not None:
        status, status_color = f"MSL {threat[1] / 1000:.1f}", theme.HUD_WARN
    elif inflight:
        status, status_color = f"IN {inflight}", theme.HUD_MARK
    else:
        status, status_color = "CLEAR", (104, 202, 150)
    msl = ac.spec.missile
    if msl is not None:
        if ac.locked:
            seek, seek_color = "LOCK", (255, 214, 96)
        elif ac.lock_timer > 0.0:
            pct = 100.0 * ac.lock_timer / max(msl.lock_time, 1e-6)
            seek, seek_color = f"SEEK {pct:3.0f}%", (214, 202, 142)
        else:
            seek, seek_color = "SEEK", (140, 178, 158)
    else:
        seek, seek_color = "", (140, 178, 158)
    return {
        "status": status,
        "status_color": _hex(status_color),
        "seek": seek,
        "seek_color": _hex(seek_color),
        "radar": None if ac.spec.radar_missile is None else {
            "name": ac.spec.radar_missile.name,
            "have": ac.radar_missiles,
            "total": ac.spec.radar_missile_count,
        },
        "ir": None if msl is None else {
            "name": msl.name,
            "have": ac.missiles,
            "total": ac.spec.missile_count,
        },
        "gun": int(ac.ammo),
        "chaff": ac.chaff,
        "flares": ac.flares,
    }


def _result(world):
    if world.result is None:
        return None
    if world.result == "draw":
        return {
            "title": "DRAW",
            "color": _hex(theme.TEXT),
            "detail": ("mutual kill" if world.reason_code == "kill"
                       else "time-up, HP even"),
        }
    idx = 0 if world.result == "red" else 1
    loser = world.acs[1 - idx].spec.name
    detail = {
        "kill": f"{loser} shot down",
        "boundary": f"{loser} out of bounds",
        "timeout": "time-up, HP advantage",
    }.get(world.reason_code, "")
    return {
        "title": f"{world.acs[idx].spec.name} WINS",
        "color": _hex(theme.TEAM_COLORS[idx]),
        "detail": detail,
    }


def _ac_snapshot(world, i, agent, engines, recorder) -> dict:
    ac = world.acs[i]
    psi, theta, phi = render.attitude_of(ac)
    model = render.MODEL_BY_NAME.get(ac.spec.name, render.GENERIC_MODEL)
    inflight = sum(1 for m in world.missiles if m.alive and m.owner == i)
    threat = _missile_threat(world, i)
    lock_color, lock_text = _lock_state(ac)
    hp_frac = clamp(ac.hp / max(ac.spec.hp_max, 1.0), 0.0, 1.0) if ac.alive else 0.0
    fuel_frac = (clamp(ac.fuel / ac.spec.fuel_capacity, 0.0, 1.0)
                 if ac.spec.fuel_capacity > 0.0 and ac.alive else 1.0)
    thrust_frac = ac.thrust / max(ac.spec.thrust_max, 1.0) if ac.alive else 0.0
    # 尾焰强度与 render._update_engines 同一算法；这里只给前端一个 0..1 值。
    flame = (0.10 + 0.90 * clamp(thrust_frac, 0.0, 1.0)) if ac.alive else 0.0
    return {
        "i": i,
        "name": ac.spec.name,
        "team": i,
        "team_color": _hex(theme.TEAM_COLORS[i]),
        "alive": bool(ac.alive),
        "pos": _p((ac.x, ac.y, ac.z)),
        "psi": psi,
        "theta": theta,
        "phi": phi,
        "hp": round(ac.hp, 1),
        "hp_max": ac.spec.hp_max,
        "speed": round(ac.v, 1),
        "alt": round(ac.z, 1),
        "g": round(ac.n, 2),
        "fuel": round(ac.fuel, 1),
        "fuel_cap": round(ac.spec.fuel_capacity, 1),
        "fuel_frac": fuel_frac,
        "throttle": clamp(ac.throttle, 0.0, 1.0),
        "ab": clamp(getattr(ac, "ab", 0.0), 0.0, 1.0),
        "thrust": round(ac.thrust, 1),
        "flame": round(flame, 3),
        "radar_on": bool(ac.radar_on),
        "contact": ac.contact,
        "rwr": bool(ac.rwr),
        "rwr_az": ac.rwr_az,
        "locked": bool(ac.locked),
        "lock_text": lock_text,
        "lock_color": _hex(lock_color),
        "lock_timer": ac.lock_timer,
        "lock_time": (ac.spec.missile.lock_time if ac.spec.missile else 0.0),
        "radar_locked": bool(ac.radar_locked),
        "radar_lock_timer": ac.radar_lock_timer,
        "radar_lock_time": (ac.spec.radar_missile.lock_time
                            if ac.spec.radar_missile else 0.0),
        "chaff": ac.chaff,
        "flares": ac.flares,
        "ammo": ac.ammo,
        "guns": int(ac.ammo),
        "gun_range": ac.spec.gun_range,
        "radar_range": ac.spec.radar_range,
        "model": ac.spec.name if ac.spec.name in ("SU-57", "F-16V") else "SU-57",
        "model_len": model["length"],
        "engines": engines,
        "missiles": [
            {
                "name": (ac.spec.radar_missile.name if ac.spec.radar_missile else "-"),
                "count": ac.radar_missiles,
                "total": ac.spec.radar_missile_count,
                "inflight": inflight,
                "threat": None if threat is None else round(threat[1], 1),
                "kind": "radar",
            },
            {
                "name": (ac.spec.missile.name if ac.spec.missile else "-"),
                "count": ac.missiles,
                "total": ac.spec.missile_count,
                "inflight": inflight,
                "threat": None if threat is None else round(threat[1], 1),
                "kind": "ir",
            },
        ],
        "hud": _hud(world, i, agent),
        "scope": _scope(world, i, agent),
        "weapon": _weapon(world, i),
        "hp_frac": hp_frac,
    }


class SimEngine:
    """独立线程推进内核，维护最近一帧快照与版本号。"""

    def __init__(self, seconds: float, seed: int | None, baseline: bool = False,
                 dof: int = 6, f16_radar: str = "always",
                 su57_radar: str = "always", awacs: bool = True) -> None:
        self.seconds = seconds
        self.seed = seed
        self.baseline = baseline
        self.dof = dof
        self.f16_radar = f16_radar
        self.su57_radar = su57_radar
        self.awacs = awacs
        self.lock = threading.Condition()
        self.version = 0
        self.snapshot = {}
        self.paused = False
        self.follow = None
        self.stop = False
        self.recorder = TrailRecorder(self)
        self.camera = CameraState()
        self._disp = {}
        with self.lock:
            self._new_world()
            self.snapshot = self._build_snapshot(0.0)
            self.version += 1
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def _new_world(self) -> None:
        self.world = main.build_world(self.seconds, self.dof, awacs=self.awacs)
        self.agents = main.build_agents(
            self.baseline, seed=self.seed,
            f16_radar=self.f16_radar, su57_radar=self.su57_radar,
        )
        self.actions = [Action(), Action()]
        self.accumulator = 0.0
        self.steps = 0
        self.recorder.reset()
        self.camera.reset()
        self.camera.follow = self.follow
        self._disp.clear()

    def _smooth(self, key, target, dt):
        prev = self._disp.get(key)
        if prev is None:
            self._disp[key] = target
            return target
        k = 1.0 - math.exp(-dt / theme.NUM_TAU)
        val = prev + (target - prev) * k
        self._disp[key] = val
        return val

    def _build_snapshot(self, dt: float) -> dict:
        world = self.world
        a, b = world.acs
        dist = math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        d_alt = b.z - a.z
        d_e = a.energy - b.energy
        dist_s = self._smooth("dist", dist, dt)
        dalt_s = self._smooth("dalt", d_alt, dt)
        models = export_meta["aircraft"]
        acs = [
            _ac_snapshot(world, i, self.agents[i],
                         models.get(world.acs[i].spec.name,
                                    models["SU-57"])["engines"],
                         self.recorder)
            for i in range(2)
        ]
        trails = []
        for i in range(2):
            pts = self.recorder.trails[i]
            axes = self.recorder.axes[i]
            ts = self.recorder.ts[i]
            trails.append({
                "points": [_p(p, 1) + [round(ts[j], 2)] for j, p in enumerate(pts)],
                "axes": [_p(ax, 3) for ax in axes],
            })
        missiles = []
        for m in world.missiles:
            if not m.alive:
                continue
            speed = m.speed
            d = (m.vx / speed, m.vy / speed, m.vz / speed) if speed > 1e-6 else (0, 0, 0)
            missiles.append({
                "owner": m.owner,
                "target": m.target,
                "pos": _p((m.x, m.y, m.z)),
                "dir": _p(d, 4),
                "alive": True,
                "speed": round(speed, 1),
                "guided": bool(m.guided),
                "ignited": bool(m.ignited),
                "fuel": round(m.fuel, 1),
                "length": m.spec.length,
                "name": m.spec.name,
                "trail": [_p(p, 1) for p in self.recorder.missile_trails.get(id(m), [])],
            })
        predictions = []
        for i, ac in enumerate(world.acs):
            if not ac.alive:
                predictions.append([])
                continue
            nx, ny, nz = nose_direction(ac)
            half = 0.5 * ac.spec.length
            pts = [(ac.x + nx * half, ac.y + ny * half, ac.z + nz * half)]
            for k in range(1, 5):
                pts.append(predict_position(ac, 1.0 * k / 4.0, steps=4))
            predictions.append([_p(p) for p in pts])
        gunfire_lines = []
        for i, ac in enumerate(world.acs):
            if not ac.alive or not ac.firing:
                gunfire_lines.append(None)
                continue
            psi, theta, phi = render.attitude_of(ac)
            model = render.MODEL_BY_NAME.get(ac.spec.name, render.GENERIC_MODEL)
            # 与 render 的屏幕模型缩放无关：炮线是纯世界坐标，从机头中心沿炮
            # 方向伸出 2.6 倍机长。
            xb, yb, zb = body_axes(psi, theta, phi)
            nose = (
                ac.x + xb[0] * model["length"] * 0.5,
                ac.y + xb[1] * model["length"] * 0.5,
                ac.z + xb[2] * model["length"] * 0.5,
            )
            gx, gy, gz = gun_direction(ac)
            length = model["length"] * 2.6
            gunfire_lines.append({
                "owner": i,
                "nose": _p(nose),
                "end": _p((nose[0] + gx * length,
                           nose[1] + gy * length,
                           nose[2] + gz * length)),
            })
        counters = [
            {
                "kind": cm.kind,
                "owner": cm.owner,
                "pos": _p((cm.x, cm.y, cm.z)),
            }
            for cm in world.counters if cm.alive
        ]
        fire_marks = [[_p(p) for p in marks] for marks in self.recorder.fire_marks]
        launch_marks = [
            {"owner": owner, "pos": _p(p)}
            for owner, p in self.recorder.launch_marks
        ]
        highg_marks = [
            {"owner": owner, "pos": _p(p)}
            for owner, p in self.recorder.highg_marks
        ]
        attitude = [
            {
                key: acs[i][key]
                for key in ("name", "team", "alive", "psi", "theta", "phi",
                            "flame", "throttle", "ab", "fuel_frac", "model_len",
                            "hp_frac")
            }
            for i in range(2)
        ]
        return {
            "t": round(world.t, 3),
            "paused": bool(self.paused),
            "running": world.result is None,
            "result": _result(world),
            "reason": world.reason,
            "arena": world.arena,
            "z_floor": world.z_floor,
            "camera": self.camera.snapshot(),
            "fov": math.degrees(render.FOV),
            "ref_len": REF_LENGTH,
            "acs": acs,
            "trails": trails,
            "missiles": missiles,
            "predictions": predictions,
            "gunfire": {
                "lines": gunfire_lines,
                "fire_marks": fire_marks,
                "launch_marks": launch_marks,
                "highg_marks": highg_marks,
            },
            "marks": {
                "fire": fire_marks,
                "launch": launch_marks,
                "highg": highg_marks,
            },
            "attitude": attitude,
            "counters": counters,
            "bottom": {
                "t": round(world.t, 1),
                "dist": round(dist_s, 0),
                "d_alt": round(dalt_s, 0),
                "d_e": round(d_e, 0),
                "text": (
                    f"T {world.t:5.1f}s    DIST {dist_s:5.0f}m    "
                    f"dALT {dalt_s:+5.0f}m    dE {d_e:6.0f} J/kg"
                ),
            },
        }

    def _run(self) -> None:
        last = time.monotonic()
        next_snap = last
        while not self.stop:
            now = time.monotonic()
            frame_dt = now - last
            last = now
            with self.lock:
                world = self.world
                if not self.paused and world.result is None:
                    self.accumulator += min(frame_dt, 0.1)
                    while self.accumulator >= main.PHYSICS_DT:
                        if self.steps % main.STEPS_PER_DECISION == 0:
                            self.actions = [
                                ag.act(world.observe(i))
                                for i, ag in enumerate(self.agents)
                            ]
                        world.step(self.actions)
                        self.recorder.record(world, self.agents)
                        self.accumulator -= main.PHYSICS_DT
                        self.steps += 1
                        if world.result is not None:
                            break
                self.camera.update(world)
                if now >= next_snap:
                    self.snapshot = self._build_snapshot(
                        min(max(frame_dt, 0.0), 0.10)
                    )
                    self.version += 1
                    self.lock.notify_all()
                    next_snap = now + 1.0 / SNAPSHOT_HZ
            time.sleep(0.004)

    def set_paused(self, paused: bool) -> None:
        with self.lock:
            self.paused = bool(paused)
            self.snapshot = self._build_snapshot(0.0)
            self.version += 1
            self.lock.notify_all()

    def set_follow(self, follow) -> None:
        with self.lock:
            self.follow = follow
            self.camera.follow = follow
            self.snapshot = self._build_snapshot(0.0)
            self.version += 1
            self.lock.notify_all()

    def reset(self) -> None:
        with self.lock:
            self._new_world()
            self.paused = False
            self.snapshot = self._build_snapshot(0.0)
            self.version += 1
            self.lock.notify_all()

    def latest(self) -> tuple[int, dict]:
        with self.lock:
            return self.version, self.snapshot


# ---------------------------------------------------------------------------
# 1080p 帧采集 / 编码
# ---------------------------------------------------------------------------

def _safe_name(raw) -> str:
    name = re.sub(r"[^\w.-]+", "_", str(raw or ""))[:64].strip("._")
    return name or "shot"


def _ffmpeg_bin():
    """优先用 imageio-ffmpeg 自带的二进制，其次找系统 ffmpeg。"""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return shutil.which("ffmpeg")


# ---------------------------------------------------------------------------
# HTTP / SSE
# ---------------------------------------------------------------------------

export_meta = export_models()
ENGINE: SimEngine | None = None


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        # 默认每个请求一行日志在 24 Hz SSE 下也只在连接/断开时打印，保留但简化。
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_file(self, path: Path, content_type: str, cache: str = "no-cache") -> None:
        try:
            data = path.read_bytes()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(data)

    def _json(self, obj, status: int = 200) -> None:
        data = json.dumps(obj, separators=(",", ":"), ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/" or path == "/index.html":
            self._send_file(STATIC / "index.html", "text/html; charset=utf-8")
            return
        if path == "/api/state":
            _v, snap = ENGINE.latest()
            self._json(snap)
            return
        if path == "/api/stream":
            self._stream()
            return
        if path.startswith("/assets/font/"):
            name = os.path.basename(path)
            self._send_file(ROOT / "assets" / "font" / name, "font/ttf")
            return
        if path.startswith("/assets/silhouette/"):
            name = os.path.basename(path)
            self._send_file(ROOT / "assets" / "silhouette" / name, "image/png")
            return
        if path.startswith("/api/models/"):
            name = os.path.basename(path)
            # 模型是重新导出的生成物，禁止浏览器缓存，避免继续用旧模型。
            self._send_file(MODEL_DIR / name, "text/plain; charset=utf-8", "no-store")
            return
        if path.startswith("/captures/"):
            name = os.path.basename(path)
            ctype = "video/mp4" if name.endswith(".mp4") else "image/png"
            self._send_file(CAPTURE_DIR / name, ctype, "no-cache")
            return
        if path.startswith("/static/"):
            rel = path[len("/static/"):]
            target = (STATIC / rel).resolve()
            if STATIC.resolve() in target.parents and target.is_file():
                ctype = {
                    ".js": "application/javascript; charset=utf-8",
                    ".css": "text/css; charset=utf-8",
                    ".html": "text/html; charset=utf-8",
                    ".obj": "text/plain; charset=utf-8",
                }.get(target.suffix, "application/octet-stream")
                # 开发期前端资源同样不缓存，改完刷新即生效。
                self._send_file(target, ctype, "no-store")
                return
        self.send_error(404)

    def _stream(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        last_version = -1
        last_write = time.monotonic()
        try:
            while True:
                version, snap = ENGINE.latest()
                if version != last_version:
                    last_version = version
                    payload = json.dumps(
                        snap, separators=(",", ":"), ensure_ascii=False
                    ).encode()
                    self.wfile.write(b"data: " + payload + b"\n\n")
                    self.wfile.flush()
                    last_write = time.monotonic()
                else:
                    # 等待新快照，同时定期发心跳：暂停/结算后版本号不再变化，
                    # 只有真正写一次才能发现客户端已断开并退出线程。
                    if time.monotonic() - last_write >= 1.0:
                        self.wfile.write(b": ping\n\n")
                        self.wfile.flush()
                        last_write = time.monotonic()
                    time.sleep(0.01)
        except (BrokenPipeError, ConnectionResetError, OSError):
            return

    def _capture_frame(self) -> None:
        """接收一帧 PNG（binary body）并按 name_0000.png 落盘。"""
        query = self.path.split("?", 1)[1] if "?" in self.path else ""
        params = dict((k, v[0]) for k, v in urllib.parse.parse_qs(query).items())
        name = _safe_name(params.get("name", "shot"))
        try:
            idx = max(0, int(params.get("i", "0")))
        except ValueError:
            self._json({"ok": False, "error": "bad index"}, 400)
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        if length <= 0 or length > 40_000_000:
            self._json({"ok": False, "error": "bad length"}, 400)
            return
        data = self.rfile.read(length)
        CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        target = CAPTURE_DIR / f"{name}_{idx:04d}.png"
        target.write_bytes(data)
        self._json({"ok": True, "path": f"/captures/{target.name}", "bytes": len(data)})

    def _capture_encode(self) -> None:
        """把 name_*.png 帧序列编码成 MP4（服务端 ffmpeg，恒定帧率）。"""
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._json({"ok": False, "error": "invalid json"}, 400)
            return
        name = _safe_name(body.get("name", "shot"))
        try:
            fps = max(1.0, float(body.get("fps", 30.0)))
        except (TypeError, ValueError):
            fps = 30.0
        frames = sorted(CAPTURE_DIR.glob(f"{name}_*.png"))
        if not frames:
            self._json({"ok": False, "error": "no frames"}, 400)
            return
        ffmpeg = _ffmpeg_bin()
        if not ffmpeg:
            self._json({"ok": False, "error": "ffmpeg not found"}, 500)
            return
        out = CAPTURE_DIR / f"{name}.mp4"
        cmd = [
            ffmpeg, "-y", "-framerate", f"{fps:.3f}",
            "-start_number", "0",
            "-i", str(CAPTURE_DIR / f"{name}_%04d.png"),
            "-c:v", "libx264", "-preset", "medium", "-crf", "18",
            "-pix_fmt", "yuv420p", str(out),
        ]
        try:
            subprocess.run(cmd, check=True, timeout=300,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except (OSError, subprocess.SubprocessError) as exc:
            self._json({"ok": False, "error": str(exc)}, 500)
            return
        self._json({
            "ok": True,
            "path": f"/captures/{out.name}",
            "frames": len(frames),
            "fps": fps,
        })

    def do_POST(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/capture":
            self._capture_frame()
            return
        if path == "/api/capture/encode":
            self._capture_encode()
            return
        if path != "/api/control":
            self.send_error(404)
            return
        length = int(self.headers.get("Content-Length", "0") or 0)
        try:
            body = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError):
            self._json({"ok": False, "error": "invalid json"}, 400)
            return
        action = body.get("action")
        try:
            if action == "pause":
                ENGINE.set_paused(True)
            elif action == "resume":
                ENGINE.set_paused(False)
            elif action == "reset":
                ENGINE.reset()
            elif action == "follow":
                value = body.get("follow", None)
                if value is not None:
                    value = int(value)
                    if value not in (0, 1):
                        raise ValueError("follow 只能是 0 / 1 / null")
                ENGINE.set_follow(value)
            else:
                raise ValueError("unknown action")
        except (TypeError, ValueError) as exc:
            self._json({"ok": False, "error": str(exc)}, 400)
            return
        self._json({"ok": True})


def main_entry() -> None:
    global ENGINE
    parser = argparse.ArgumentParser(description="Dogfight Web 观战前端")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--seconds", type=float, default=300.0,
                        help="对局时长上限（默认 300 s）")
    parser.add_argument("--seed", type=int, default=None,
                        help="固定开局随机占位的随机种子（默认每次启动随机）")
    parser.add_argument("--baseline", action="store_true",
                        help="双方使用纯追踪基线")
    parser.add_argument(
        "--dof", type=int, choices=(2, 3, 6), default=6,
        help="内核自由度：默认 6（刚体）；2/3 仅用于 A/B 对照实验",
    )
    radar_choices = ("always", "silent", "auto")
    parser.add_argument(
        "--f16-radar", choices=radar_choices, default="always",
        help="F-16V 雷达策略：常开 / 全程静默 / 静默+短时开机（默认 always）",
    )
    parser.add_argument(
        "--su57-radar", choices=radar_choices, default="always",
        help="Su-57 雷达策略（默认 always）",
    )
    parser.add_argument(
        "--awacs", action=argparse.BooleanOptionalAction, default=True,
        help="是否启用预警机（默认开启；--no-awacs 关闭）",
    )
    args = parser.parse_args()

    ENGINE = SimEngine(
        args.seconds, args.seed, baseline=args.baseline, dof=args.dof,
        f16_radar=args.f16_radar, su57_radar=args.su57_radar,
        awacs=args.awacs,
    )
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    httpd.daemon_threads = True
    print(f"Dogfight web observer: http://{args.host}:{args.port}/")
    print("模型已导出到 web/static/models/")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        ENGINE.stop = True
        httpd.server_close()


if __name__ == "__main__":
    main_entry()
