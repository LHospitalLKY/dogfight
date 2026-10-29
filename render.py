"""M2 观战窗口：pygame 透视 3D 渲染 + 机体模型 + 机动标签，只读 World 状态。

相机围绕两机连线的中点做轨道飞行，俯角约 26°，地面网格和竖直投影线
用来给出高度的直观参照；机体用对应的三维模型表示，姿态直接取内核的
psi/theta/phi（点质量内核用 gamma/-bank 还原），开火时叠加炮口焰与炮线。
"""
from __future__ import annotations

import math
import os
from array import array

import pygame

from sim import World, body_axes, clamp, gun_direction, wrap_pi

BG = (18, 20, 24)
GRID = (31, 35, 41)
BORDER = (72, 78, 88)
TEXT = (231, 234, 238)
SUB = (148, 155, 165)
TEAM_COLORS = ((232, 92, 80), (86, 150, 255))
# 轨迹采样间距：整场都保留，但按距离抽点，避免几千段线拖慢每帧。
TRAIL_SPACING = 60.0
# 开火点抽稀：同一个点射不必每一帧都插一个标记。
SHOT_SPACING = 120.0
# 每隔多少个采样点放一个方向箭头。
ARROW_EVERY = 25

# ---- 轨迹上的事件标记（实时画在 3D 航迹线上，不是事后出图） ----
# 金色星标 = 导弹发射点；橙色菱形 = 高 G 机动点。
LAUNCH_MARK = (255, 209, 102)
HIGH_G_MARK_COLOR = (255, 122, 69)
# 过载超过人体可持续耐受（6 g）即标注；高空升力受限时（9–12 km、300 m/s
# 量级）可用过载本身只有 4–5 g，因此执行高 G 原语且拉到当地可用过载的
# 75% 也标注，否则航迹上几乎看不到高 G 点（与 main.py 一致）。
HIGH_G_LOAD = 6.0
HIGH_G_REL = 0.75
HIGH_G_COOLDOWN = 1.5

NEAR = 1.0
FOV = math.radians(50.0)
# 屏幕尺寸统一参考长度：用 Su-57 的真实机长做基准，F-16 自动缩到它的 0.75，
# 两机在画面上呈现真实的大小比例关系。
REF_LENGTH = 20.1
# 固定方向光，给机体多边形做明暗，避免纯色剪影。
LIGHT = (-0.35, -0.55, 0.76)


# ---------- GPU（OpenGL 3.3 core）渲染 ----------
# 3D 场景（地面网格、航迹、机体、姿态框模型）交给 OpenGL 绘制，pygame 仍然
# 负责窗口/输入与 2D HUD：每帧把 HUD 画在透明 Surface 上，再作为纹理叠到 GL 画面。
LIT_VS = """#version 330
uniform mat4 u_mvp;
uniform mat4 u_rot;
uniform vec3 u_light;
in vec3 in_pos;
in vec3 in_nrm;
in vec3 in_col;
out vec3 v_col;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    vec3 n = normalize(mat3(u_rot) * in_nrm);
    v_col = in_col * (0.50 + 0.50 * abs(dot(n, normalize(u_light))));
}
"""

LIT_FS = """#version 330
in vec3 v_col;
out vec4 f_col;
void main() { f_col = vec4(v_col, 1.0); }
"""

FLAT_VS = """#version 330
uniform mat4 u_mvp;
in vec3 in_pos;
in vec3 in_col;
out vec3 v_col;
void main() {
    gl_Position = u_mvp * vec4(in_pos, 1.0);
    v_col = in_col;
}
"""

HUD_VS = """#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    gl_Position = vec4(in_pos, 0.0, 1.0);
    v_uv = in_uv;
}
"""

HUD_FS = """#version 330
uniform sampler2D u_tex;
in vec2 v_uv;
out vec4 f_col;
void main() { f_col = texture(u_tex, v_uv); }
"""


def _mat4_flat(rows) -> tuple:
    """行主序 4x4 → OpenGL 需要的列主序 16 元组（moderngl 按原始字节上传）。"""
    return tuple(rows[r][c] for c in range(4) for r in range(4))


def _mat4_mul(a, b) -> list:
    """行主序 4x4 矩阵相乘。"""
    return [
        [sum(a[r][k] * b[k][c] for k in range(4)) for c in range(4)]
        for r in range(4)
    ]


def _mix(a, b, t: float) -> tuple[int, int, int]:
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


# ---------- OBJ 机体模型 ----------

SU57_OBJ_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "I154-SU-57战机",
    "I154-SU-57战机_OBJ.obj",
)
# F-16C 的 OBJ 在 f_16c_model 目录下，原始单位就是米。
F16_OBJ_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "f_16c_model",
    "f_16c.obj",
)
# 座舱内部、仪表和起落架灯从外部视角看不到，读入时直接丢弃。
_DROP_MATERIALS = {
    # "pkfa-cockpit",
    # "pkfa-cockpit-Instr-glass",
    # "pkfa-Instr-glass",
    # "pkfa-hud",
    # "pkfa-cockpit-digits",
    # "pkfa-cockpit-ADIM",
    # "pkfa-landinglights",
}
_CANOPY_MATERIALS = {"pkfa-cockpit-glass"}
# 模型是起落架放下状态；机体系下低于 -0.10 的面属于起落架（机身最低约 -0.096）。
_GEAR_Y = -0.10

# F-16C 的 OBJ 带了一大堆外部挂载（副油箱、LITENING/Sniper 吊舱、GPU-5A 机炮吊舱、
# 各种挂架和座椅内饰）。只保留机体蒙皮与座舱玻璃，其余全部丢弃：既符合对局里的
# 武器挂载，又能把 28 万面降到可实时渲染的量级。
F16_KEEP_MATERIALS = {
    "f_16c_block_50_d_a_n_d_l_c_d_m_c_b",
    "dynamic_glass_chrome_d_o_0.17_c_d_g_c",
    "dynamic_glass_chrome_d_o_0.75_c_d_g_c",
    "dynamic_glass_chrome_d_o_0.2_c_d_g_c",
    "dynamic_glass_chrome_d_o_0.15_c_d_g_c",
    "dynamic_glass_chrome_d_o_0.1_c_d_g_c",
}
F16_CANOPY_MATERIALS = {
    "dynamic_glass_chrome_d_o_0.17_c_d_g_c",
    "dynamic_glass_chrome_d_o_0.75_c_d_g_c",
}
# 机体最低点约 -0.95 m，再往下是放下的起落架；切掉这些面让飞行姿态更干净。
F16_GEAR_Y = -0.95


def _simple_model(length: float, twin: bool = False) -> dict:
    """OBJ 缺失时的替代机体：用方块按真实尺寸比例拼一架简化战斗机。

    仓库不带 3D 模型（.gitignore 忽略了模型目录），远程/CI 环境拿不到 OBJ。
    这里按机体长度生成低面数几何，保证观战与录制仍能出画面；有模型时
    仍然走原来的体素简化路径，两者接口一致。
    """
    half = length / 2.0
    w = 0.055 * length       # 机身半宽
    h = 0.050 * length       # 机身半高
    thin = 0.008 * length    # 翼面厚度
    faces: list[tuple[list[tuple[float, float, float]], str]] = []

    def box(x0, x1, y0, y1, z0, z1, kind="body"):
        v = [
            (x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0),
            (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1),
        ]
        for a, b, c, d in (
            (0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1),
            (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0),
        ):
            faces.append(([v[a], v[b], v[c], v[d]], kind))

    # 机身 + 锥形机头。
    box(-0.62 * half, 0.60 * half, -w, w, -h, h)
    box(0.60 * half, half, -0.35 * w, 0.35 * w, -0.55 * h, 0.55 * h)
    # 主翼：后掠直角三角形按矩形近似，展长按公开翼展取。
    box(-0.24 * half, 0.10 * half, -0.34 * length, 0.34 * length, -thin, thin, "wing")
    # 平尾。
    box(-0.98 * half, -0.72 * half, -0.20 * length, 0.20 * length, -thin, thin, "tail")
    # 垂尾（z 向下为正，因此向负 z 竖起）+ 腹鳍。
    box(-0.98 * half, -0.66 * half, -0.02 * length, 0.02 * length, -h - 0.10 * length, -h, "tail")
    # 座舱。
    box(0.06 * half, 0.46 * half, -0.032 * length, 0.032 * length, -h - 0.035 * length, -h, "canopy")
    # 进气道与喷口。
    box(-0.10 * half, 0.20 * half, -w - 0.022 * length, -w, -0.45 * h, 0.35 * h, "intake")
    box(-0.10 * half, 0.20 * half, w, w + 0.022 * length, -0.45 * h, 0.35 * h, "intake")
    if twin:
        for y in (-0.030 * length, 0.030 * length):
            box(-0.70 * half, -0.62 * half, y - 0.022 * length,
                y + 0.022 * length, -0.030 * length, 0.030 * length, "nozzle")
    else:
        box(-0.70 * half, -0.62 * half, -0.028 * length, 0.028 * length,
            -0.032 * length, 0.032 * length, "nozzle")
    return {"length": length, "faces": faces, "cull": False, "outline": False}


def _load_obj_model(
    path: str,
    length: float,
    voxel: float,
    keep_materials: set | None = None,
    canopy_materials: set | None = None,
    gear_y: float | None = None,
    twin: bool = False,
) -> dict:
    """读取 OBJ 并做体素聚类简化，返回机体坐标系下的面列表。

    原始 OBJ 坐标：x 朝机头、z 为展向、y 向上，原点不在机身中心。
    这里统一缩放/平移成 (x 前, y 右, z 下)，与内核和渲染的机体系一致。
    文件不存在时回退到 `_simple_model`，不阻断观战/录制。
    """
    if not os.path.exists(path):
        return _simple_model(length, twin)

    canopy_set = _CANOPY_MATERIALS if canopy_materials is None else canopy_materials
    verts = []
    faces = []
    material = None
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if line.startswith("v "):
                px, py, pz = line.split()[1:4]
                verts.append((float(px), float(py), float(pz)))
            elif line.startswith("usemtl "):
                material = line.split(None, 1)[1].strip()
            elif line.startswith("f "):
                if material in _DROP_MATERIALS:
                    continue
                if keep_materials is not None and material not in keep_materials:
                    continue
                idx = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                if gear_y is not None and min(verts[i][1] for i in idx) < gear_y:
                    continue
                faces.append((idx, material))

    used = sorted({i for idx, _ in faces for i in idx})
    xs = [verts[i][0] for i in used]
    ys = [verts[i][1] for i in used]
    zs = [verts[i][2] for i in used]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    cz = (min(zs) + max(zs)) / 2.0
    scale = length / (max(xs) - min(xs))

    # 体素聚类：同一格内的顶点合并到平均位置，面数从 4 万降到几千。
    buckets = {}
    for i in used:
        x, y, z = verts[i]
        key = (round(x / voxel), round(y / voxel), round(z / voxel))
        slot = buckets.get(key)
        if slot is None:
            buckets[key] = [x, y, z, 1]
        else:
            slot[0] += x
            slot[1] += y
            slot[2] += z
            slot[3] += 1

    remap = {}
    pts = []
    for k, (key, slot) in enumerate(buckets.items()):
        remap[key] = k
        pts.append((slot[0] / slot[3], slot[1] / slot[3], slot[2] / slot[3]))

    out = []
    seen = set()
    for idx, material in faces:
        cluster = []
        for i in idx:
            x, y, z = verts[i]
            c = remap[(round(x / voxel), round(y / voxel), round(z / voxel))]
            if c not in cluster:
                cluster.append(c)
        if len(cluster) < 3:
            continue
        sig = tuple(sorted(cluster))
        if sig in seen:
            continue
        seen.add(sig)
        poly = [
            ((pts[c][0] - cx) * scale,
             (pts[c][2] - cz) * scale,
             -(pts[c][1] - cy) * scale)
            for c in cluster
        ]
        kind = "canopy" if material in canopy_set else "body"
        out.append((poly, kind))
    return {"length": length, "faces": out, "cull": False, "outline": False}


# Su-57 直接用 SU-57 文件夹里的 OBJ，体素 0.05（约 0.5 m）：约 1500 面，
# 游戏里机体最多约 100 px，这个精度足够，还能保住 60 fps 以上。
# 长度取公开数据 20.1 m；F-16 取 15.06 m，两者屏幕上保持真实尺寸比例。
SU57_MODEL = _load_obj_model(SU57_OBJ_PATH, 20.1, 0.05, gear_y=_GEAR_Y)
# 双发尾喷口位置（机体系，x 向后、y 右、z 下），用于画尾焰。
SU57_MODEL["engines"] = [(-9.4, 0.7, 0.0), (-9.4, -0.7, 0.0)]

F16_MODEL = _load_obj_model(
    F16_OBJ_PATH, 15.06, 0.40,
    keep_materials=F16_KEEP_MATERIALS,
    canopy_materials=F16_CANOPY_MATERIALS,
    gear_y=F16_GEAR_Y,
)
F16_MODEL["engines"] = [(-7.30, 0.0, 0.0)]

MODEL_BY_NAME = {"SU-57": SU57_MODEL, "F-16V": F16_MODEL}
GENERIC_MODEL = SU57_MODEL

# 机体各部件的配色键 → 相对队伍色的明暗系数；座舱/进气道/喷口用独立颜色。
PART_SHADE = {
    "body": 1.0,
    "wing": 0.86,
    "tail": 0.76,
    "canopy": None,
    "intake": None,
    "nozzle": None,
}
CANOPY_COLOR = (126, 196, 238)
INTAKE_COLOR = (52, 56, 62)
NOZZLE_COLOR = (92, 88, 84)


def attitude_of(ac) -> tuple[float, float, float]:
    """渲染用姿态 (psi, theta, phi)。点质量内核用航迹角与滚转角还原。"""
    if getattr(ac, "rigid", False):
        return ac.psi, ac.theta, ac.phi
    # 3DOF 的 bank>0 为左滚，而机体 phi>0 为右滚，故取负号。
    return ac.psi, ac.gamma, -ac.bank


def _attitude_basis():
    """姿态框的固定等距视角基向量（与 _draw_attitude_model 保持一致）。"""
    view = _norm((0.85, -0.98, 0.55))
    right = _norm(_cross((0.0, 0.0, 1.0), view))
    return view, right, _cross(view, right)



def _dot(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b) -> tuple[float, float, float]:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _norm(a) -> tuple[float, float, float]:
    n = math.sqrt(_dot(a, a)) or 1.0
    return (a[0] / n, a[1] / n, a[2] / n)


class Renderer:
    def __init__(
        self,
        size: tuple[int, int] = (1280, 800),
        follow: int | None = None,
        follow_dist: float = 1800.0,
        gpu: bool = True,
    ) -> None:
        pygame.init()
        self.w, self.h = size
        # 3D 场景优先交给 OpenGL；没有可用 GL 上下文（远程会话/软件驱动）时
        # 自动回退到原来的 pygame 软件渲染，HUD 与交互完全一致。
        self.ctx = None
        self.mgl = None
        self.gpu = gpu and self._init_gpu(size)
        if not self.gpu:
            self.screen = pygame.display.set_mode(size)
        pygame.display.set_caption("Dogfight M2")
        self.font = pygame.font.Font(None, 22)
        self.font_small = pygame.font.Font(None, 18)
        self.font_big = pygame.font.Font(None, 64)

        # 轨道相机：yaw 绕世界 z 轴，pitch 抬高视线，dist 控制取景距离。
        self.cam = [0.0, 0.0, 1000.0]
        # 初始方位与“垂直于两机连线”的稳态一致，避免开局相机绕 180°。
        self.yaw = math.pi / 2.0
        self.pitch = math.radians(26.0)
        self.dist = 6000.0
        self.eye = (0.0, 6000.0, 3000.0)
        self.right = (1.0, 0.0, 0.0)
        self.up = (0.0, 0.0, 1.0)
        self.forward = (0.0, 1.0, 0.0)
        self.focal = (self.h / 2.0) / math.tan(FOV / 2.0)

        self.trails: list[list[tuple[float, float, float]]] = [[], []]
        self.fire_marks: list[list[tuple[float, float, float]]] = [[], []]
        # 轨迹事件：导弹发射点 (队别, 位置) 与高 G 机动点 (队别, 位置)。
        self.launch_marks: list[tuple[int, tuple[float, float, float]]] = []
        self.highg_marks: list[tuple[int, tuple[float, float, float]]] = []
        self._seen_missiles: set[int] = set()
        self._highg_state = [False, False]
        self._last_highg = [-1e9, -1e9]
        self._last_sample: list[tuple[float, float, float] | None] = [None, None]
        self._last_shot: list[tuple[float, float, float] | None] = [None, None]
        # 在飞格斗弹的尾迹：按 id 记账，导弹结束后清掉。
        self.missile_trails: dict[int, list[tuple[float, float, float]]] = {}
        # 尾焰强度跟着真实推力走，0=收油门、1=满加力。
        self._flame = [0.35, 0.35]
        self.overlay = pygame.Surface(size, pygame.SRCALPHA)
        # 姿态框的屏幕矩形：GPU 模式下由 GL 在对应区域补画三维模型。
        self._attitude_rects: list[pygame.Rect] = []
        self._cam_ready = False
        # 跟随模式：相机锁定某架飞机（默认关闭，仍取两机中点）。
        self.follow = follow
        self.follow_dist = follow_dist

    # ---------- 相机与投影 ----------

    def update_camera(self, world: World) -> None:
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
            target_dist = clamp(span * 2.1, 2200.0, 60000.0)
        if self._cam_ready:
            self.dist += (target_dist - self.dist) * 0.06
            self.cam[0] += (cx - self.cam[0]) * 0.10
            self.cam[1] += (cy - self.cam[1]) * 0.10
            self.cam[2] += (cz - self.cam[2]) * 0.08
        else:
            # 首帧直接对焦，避免开局相机从原点一路飞过去。
            self.cam = [cx, cy, cz]
            self.dist = target_dist
            self._cam_ready = True

        if follow_ac is not None:
            # 机尾后方跟拍：视线方向跟着 SU-57 的航向转。
            desired = follow_ac.psi + math.pi
            self.yaw += clamp(wrap_pi(desired - self.yaw), -0.06, 0.06)
            self.pitch += (math.radians(16.0) - self.pitch) * 0.02
        elif sep_xy > 500.0:
            # 视线不严格垂直于两机连线，留约 20° 夹角，避免机翼在正对头时完全侧视消失。
            desired = (
                math.atan2(b.y - a.y, b.x - a.x) + math.pi / 2.0 - 0.35
            )
            self.yaw += clamp(wrap_pi(desired - self.yaw), -0.03, 0.03)

        cp, sp = math.cos(self.pitch), math.sin(self.pitch)
        self.eye = (
            self.cam[0] + self.dist * cp * math.cos(self.yaw),
            self.cam[1] + self.dist * cp * math.sin(self.yaw),
            self.cam[2] + self.dist * sp,
        )
        self.forward = _norm(
            (
                self.cam[0] - self.eye[0],
                self.cam[1] - self.eye[1],
                self.cam[2] - self.eye[2],
            )
        )
        self.right = _norm(_cross(self.forward, (0.0, 0.0, 1.0)))
        self.up = _cross(self.right, self.forward)

    def _view(self, p) -> tuple[float, float, float]:
        """世界坐标 → 相机坐标 (右, 上, 前)。"""
        rx, ry, rz = p[0] - self.eye[0], p[1] - self.eye[1], p[2] - self.eye[2]
        return (
            rx * self.right[0] + ry * self.right[1] + rz * self.right[2],
            rx * self.up[0] + ry * self.up[1] + rz * self.up[2],
            rx * self.forward[0] + ry * self.forward[1] + rz * self.forward[2],
        )

    def _screen_of(self, c) -> tuple[float, float]:
        return (
            self.w / 2.0 + self.focal * c[0] / c[2],
            self.h / 2.0 - self.focal * c[1] / c[2],
        )

    def to_screen(self, p) -> tuple[float, float] | None:
        c = self._view(p)
        if c[2] <= NEAR:
            return None
        return self._screen_of(c)

    def _clip_segment(self, p0, p1):
        """线段跨越近裁剪面时先截断，避免相机后方的东西被画到屏幕上。"""
        a, b = self._view(p0), self._view(p1)
        if a[2] <= NEAR and b[2] <= NEAR:
            return None
        if a[2] <= NEAR:
            t = (NEAR - a[2]) / (b[2] - a[2])
            a = tuple(a[i] + (b[i] - a[i]) * t for i in range(3))
        elif b[2] <= NEAR:
            t = (NEAR - b[2]) / (a[2] - b[2])
            b = tuple(b[i] + (a[i] - b[i]) * t for i in range(3))
        return self._screen_of(a), self._screen_of(b)

    def reset(self) -> None:
        self.trails = [[], []]
        self.fire_marks = [[], []]
        self.launch_marks = []
        self.highg_marks = []
        self._seen_missiles = set()
        self._highg_state = [False, False]
        self._last_highg = [-1e9, -1e9]
        self._last_sample = [None, None]
        self._last_shot = [None, None]
        self.missile_trails = {}
        self._flame = [0.35, 0.35]
        self._attitude_rects = []

    def _record_trail(self, world: World, agents=None) -> None:
        """整局航迹都保留；按固定间距抽点，线段数量不会随帧率失控。

        同时记录两类事件，直接画在 3D 航迹线上：
        - 导弹发射点：第一次看到某个在飞弹 id 的时刻即发射瞬间；
        - 高 G 机动点：过载达到门限的上升沿，带 1.5 s 冷却避免连成一片。
        """
        for i, ac in enumerate(world.acs):
            if not ac.alive:
                continue
            p = (ac.x, ac.y, ac.z)
            last = self._last_sample[i]
            if last is None or math.dist(last, p) >= TRAIL_SPACING:
                self.trails[i].append(p)
                self._last_sample[i] = p
            if ac.firing:
                last_shot = self._last_shot[i]
                if last_shot is None or math.dist(last_shot, p) >= SHOT_SPACING:
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
                ac.n >= HIGH_G_LOAD
                or (
                    primitive.startswith("HIGH_G")
                    and ac.n >= HIGH_G_REL * n_avail
                )
            )
            if (
                hi
                and not self._highg_state[i]
                and world.t - self._last_highg[i] > HIGH_G_COOLDOWN
            ):
                self._last_highg[i] = world.t
                self.highg_marks.append((i, (ac.x, ac.y, ac.z)))
            self._highg_state[i] = hi

    def _update_engines(self, world: World) -> None:
        """尾焰强度直接跟随当前推力：军推一档、加力一档，油尽自动掉回军推。"""
        for i, ac in enumerate(world.acs):
            if not ac.alive:
                target = 0.0
            else:
                frac = ac.thrust / max(ac.spec.thrust_max, 1.0)
                target = 0.10 + 0.90 * clamp(frac, 0.0, 1.0)
            self._flame[i] += (target - self._flame[i]) * 0.18

    # ---------- 绘制 ----------

    def draw(self, world: World, paused: bool = False, agents=None) -> None:
        self.update_camera(world)
        self._record_trail(world, agents)
        self._update_engines(world)
        if self.gpu:
            # 2D 层先画在透明 HUD 面上，3D 场景由 GPU 渲染后再叠回 HUD。
            self.hud.fill((0, 0, 0, 0))
            self._draw_trails()
            self._draw_gun_envelopes(world)
            self._draw_aircraft(world, agents)
            self._draw_missiles(world)
            self._draw_counters(world)
            self._draw_gunfire(world)
            self._draw_altitude_ladder(world)
            self._draw_hud(world, paused, agents)
            if world.result is not None:
                self._draw_result(world)
            self._gpu_scene(world)
            self._gpu_hud()
            self._gpu_attitude(world)
            pygame.display.flip()
            return

        self.screen.fill(BG)
        self._draw_floor(world.arena, world.z_floor)
        self._draw_trails()
        self._draw_gun_envelopes(world)
        self._draw_aircraft(world, agents)
        self._draw_missiles(world)
        self._draw_counters(world)
        self._draw_gunfire(world)
        self._draw_altitude_ladder(world)
        self._draw_hud(world, paused, agents)

        if world.result is not None:
            self._draw_result(world)
        pygame.display.flip()

    def _draw_floor(self, arena: float, z_floor: float = 0.0) -> None:
        """最低高度层网格 + 空域边界：高度读数的水平参照。"""
        half = arena / 2.0
        step = 1000.0
        x = -half
        while x <= half:
            seg = self._clip_segment((x, -half, z_floor), (x, half, z_floor))
            if seg:
                pygame.draw.line(self.screen, GRID, seg[0], seg[1], 1)
            x += step
        y = -half
        while y <= half:
            seg = self._clip_segment((-half, y, z_floor), (half, y, z_floor))
            if seg:
                pygame.draw.line(self.screen, GRID, seg[0], seg[1], 1)
            y += step

        corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
        for i in range(4):
            p0 = (*corners[i], z_floor)
            p1 = (*corners[(i + 1) % 4], z_floor)
            seg = self._clip_segment(p0, p1)
            if seg:
                pygame.draw.line(self.screen, BORDER, seg[0], seg[1], 1)

    def _draw_trails(self) -> None:
        for i, trail in enumerate(self.trails):
            color = tuple(int(c * 0.60) for c in TEAM_COLORS[i])
            arrow_color = tuple(int(c * 0.95) for c in TEAM_COLORS[i])
            if len(trail) >= 2:
                # GPU 模式下航迹线由 OpenGL 画，这里只保留箭头与事件标记。
                if not self.gpu:
                    run: list[tuple[float, float]] = []
                    for p0, p1 in zip(trail, trail[1:]):
                        seg = self._clip_segment(p0, p1)
                        if seg is None:
                            if len(run) >= 2:
                                pygame.draw.lines(self.screen, color, False, run, 2)
                            run = []
                            continue
                        if not run:
                            run.append(seg[0])
                        run.append(seg[1])
                    if len(run) >= 2:
                        pygame.draw.lines(self.screen, color, False, run, 2)

                # 方向箭头：沿航迹等距放箭标，整局走线在远看时也能读出方向。
                for j in range(ARROW_EVERY, len(trail), ARROW_EVERY):
                    s0 = self.to_screen(trail[j - 1])
                    s1 = self.to_screen(trail[j])
                    if s0 is not None and s1 is not None:
                        self._draw_arrow(s0, s1, arrow_color)

        # 开火点：每个点射位置留一个亮色菱形，和航迹一起构成完整战况记录。
        for marks in self.fire_marks:
            for p in marks:
                s = self.to_screen(p)
                if s is None:
                    continue
                x, y = s
                pygame.draw.polygon(
                    self.screen, (255, 190, 66),
                    [(x, y - 5), (x + 5, y), (x, y + 5), (x - 5, y)],
                )
                pygame.draw.line(
                    self.screen, (255, 246, 214), (x, y - 3), (x, y + 3), 1
                )

        # 导弹发射点：金色五角星 + 队伍色描边，位置就在发射瞬间的航迹点上。
        for owner, p in self.launch_marks:
            s = self.to_screen(p)
            if s is None:
                continue
            edge = tuple(int(c * 0.55) for c in TEAM_COLORS[owner])
            self._draw_star(s, 7.5, LAUNCH_MARK, edge)

        # 高 G 机动点：橙色菱形（比开火点大一号），同样落在 3D 航迹线上。
        for i, p in self.highg_marks:
            s = self.to_screen(p)
            if s is None:
                continue
            x, y = s
            pygame.draw.polygon(
                self.screen, HIGH_G_MARK_COLOR,
                [(x, y - 7), (x + 7, y), (x, y + 7), (x - 7, y)],
            )
            pygame.draw.polygon(
                self.screen, TEAM_COLORS[i],
                [(x, y - 4), (x + 4, y), (x, y + 4), (x - 4, y)], 1,
            )

    def _draw_star(self, s, size: float, fill, edge) -> None:
        """五角星：10 个顶点交替取外/内半径，用于导弹发射点。"""
        pts = []
        for k in range(10):
            ang = -math.pi / 2.0 + k * math.pi / 5.0
            r = size if k % 2 == 0 else size * 0.45
            pts.append((s[0] + r * math.cos(ang), s[1] + r * math.sin(ang)))
        pygame.draw.polygon(self.screen, fill, pts)
        pygame.draw.polygon(self.screen, edge, pts, 1)

    def _draw_arrow(self, s0, s1, color, size: float = 6.0) -> None:
        """在 s1 处画一个沿 s0→s1 的箭头。"""
        dx, dy = s1[0] - s0[0], s1[1] - s0[1]
        n = math.hypot(dx, dy)
        if n < 8.0:
            return
        ux, uy = dx / n, dy / n
        left = (s1[0] - ux * size - uy * size * 0.6,
                s1[1] - uy * size + ux * size * 0.6)
        right = (s1[0] - ux * size + uy * size * 0.6,
                 s1[1] - uy * size - ux * size * 0.6)
        pygame.draw.lines(self.screen, color, False, [left, s1, right], 2)

    def _draw_gun_envelopes(self, world: World) -> None:
        """机炮包线画成机体所在高度的水平圆环，透视后自然成椭圆。"""
        self.overlay.fill((0, 0, 0, 0))
        for i, ac in enumerate(world.acs):
            if not ac.alive:
                continue
            color = TEAM_COLORS[i]
            radius = ac.spec.gun_range
            for scale, alpha in ((1.0, 40), (0.5, 24)):
                ring: list[tuple[float, float]] = []
                for step in range(49):
                    ang = 2.0 * math.pi * step / 48.0
                    p = (
                        ac.x + radius * scale * math.cos(ang),
                        ac.y + radius * scale * math.sin(ang),
                        ac.z,
                    )
                    s = self.to_screen(p)
                    if s is not None:
                        ring.append(s)
                if len(ring) >= 2:
                    pygame.draw.lines(self.overlay, (*color, alpha), True, ring, 1)
        self.screen.blit(self.overlay, (0, 0))

    # ---------- 机体模型与开火特效 ----------

    def _model_frame(self, ac):
        """返回 (模型, 缩放系数, 姿态, 相机深度)。缩放按屏幕像素长度反推。

        两机共用同一个参考长度 REF_LENGTH 反算屏幕尺寸，于是 F-16（15.06 m）
        在画面上就是 Su-57（20.1 m）的 0.75 倍，保持真实尺寸比例。
        """
        depth = self._view((ac.x, ac.y, ac.z))[2]
        if depth <= NEAR:
            return None
        model = MODEL_BY_NAME.get(ac.spec.name, GENERIC_MODEL)
        # 远距离时保持一个最小可见尺寸，近距也不至于铺满屏幕。
        # 55000 m 的空域尺度下开局两机距离可达 45 km 以上，下限提到 2.0 倍
        # 才能保证远处仍能看清机体轮廓（约 80 px），近距约 120 px。
        target_px = 40.0 * clamp(8000.0 / depth, 2.0, 3.0)
        k = target_px * depth / (self.focal * REF_LENGTH)
        return model, k, attitude_of(ac), depth

    def _model_world(self, ac, att, p, k):
        """机体系点 → 世界坐标，按当前姿态旋转并平移到机体位置。"""
        xb, yb, zb = body_axes(*att)
        u, v, w = p[0] * k, p[1] * k, p[2] * k
        return (
            ac.x + xb[0] * u + yb[0] * v + zb[0] * w,
            ac.y + xb[1] * u + yb[1] * v + zb[1] * w,
            ac.z + xb[2] * u + yb[2] * v + zb[2] * w,
        )

    def _draw_model(self, ac, team: int, frame) -> None:
        model, k, att, _depth = frame
        cull = model.get("cull", True)
        faces = []
        for pts, kind in model["faces"]:
            wp = [self._model_world(ac, att, p, k) for p in pts]
            vp = [self._view(p) for p in wp]
            if any(p[2] <= NEAR for p in vp):
                continue
            poly = [self._screen_of(p) for p in vp]
            normal = _norm(
                _cross(
                    (wp[1][0] - wp[0][0], wp[1][1] - wp[0][1], wp[1][2] - wp[0][2]),
                    (wp[2][0] - wp[0][0], wp[2][1] - wp[0][1], wp[2][2] - wp[0][2]),
                )
            )
            light = 0.50 + 0.50 * abs(_dot(normal, LIGHT))
            # 程序化部件都是封闭体，统一剔除背面，避免远近侧面互相覆盖；
            # OBJ 模型的绕序不统一，按双面绘制（远面先画、近面覆盖）。
            if cull:
                cx = sum(p[0] for p in wp) / len(wp)
                cy = sum(p[1] for p in wp) / len(wp)
                cz = sum(p[2] for p in wp) / len(wp)
                to_eye = (self.eye[0] - cx, self.eye[1] - cy, self.eye[2] - cz)
                if _dot(normal, to_eye) <= 0.0:
                    continue
            if kind == "canopy":
                base = CANOPY_COLOR
            elif kind == "intake":
                base = INTAKE_COLOR
            elif kind == "nozzle":
                base = NOZZLE_COLOR
            else:
                base = tuple(c * PART_SHADE[kind] for c in TEAM_COLORS[team])
            color = tuple(int(clamp(c * light, 0, 255)) for c in base)
            depth = sum(p[2] for p in vp) / len(vp)
            faces.append((depth, poly, color))

        # 尾焰和机体一起参与深度排序，近面能盖住离镜头更远的焰流。
        faces.extend(self._flame_faces(ac, model, k, att, team, _depth))

        # 画家算法：远的先画，近的覆盖上去。
        faces.sort(key=lambda f: -f[0])
        outline = model.get("outline", True)
        for _d, poly, color in faces:
            pygame.draw.polygon(self.screen, color, poly)
            if not outline:
                continue
            # OBJ 模型简化后单个面只有零点几像素，再描边会糊成一团黑，只给大面描边。
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            if max(max(xs) - min(xs), max(ys) - min(ys)) >= 6.0:
                pygame.draw.polygon(self.screen, (10, 12, 16), poly, 1)

    def _flame_faces(self, ac, model, k, att, team, depth) -> list:
        """尾焰：沿机尾方向铺开的锥形，长度跟随机体加速度（加速变长、减速变短）。"""
        engines = model.get("engines")
        if not engines:
            return []
        nose = self._model_world(ac, att, (model["length"] / 2.0, 0.0, 0.0), k)
        ns = self.to_screen(nose)
        if ns is None:
            return []

        level = self._flame[team]
        ab = clamp(getattr(ac, "ab", 0.0), 0.0, 1.0)
        # 加力时焰色从橙黄偏向蓝白，和内芯一起形成明显不同的两档外观。
        outer = _mix((255, 138, 44), (168, 192, 255), 0.55 * ab)
        inner = _mix((255, 238, 196), (238, 244, 255), 0.75 * ab)
        screen_len = k * model["length"] * self.focal / max(depth, NEAR)
        length = max(3.0, screen_len * (0.06 + 0.20 * level))
        width = max(1.4, length * 0.22)

        out = []
        for bx, by, bz in engines:
            tail = self._model_world(ac, att, (bx, by, bz), k)
            ts = self.to_screen(tail)
            if ts is None:
                continue
            dx, dy = ts[0] - ns[0], ts[1] - ns[1]
            n = math.hypot(dx, dy)
            if n < 1e-3:
                continue
            ux, uy = dx / n, dy / n
            px, py = -uy, ux
            p0 = (ts[0] + ux * 1.5, ts[1] + uy * 1.5)
            mid = (p0[0] + ux * length * 0.35, p0[1] + uy * length * 0.35)
            tip = (p0[0] + ux * length, p0[1] + uy * length)
            plume_depth = self._view(tail)[2]
            out.append((
                plume_depth,
                [
                    (p0[0] + px * width * 0.5, p0[1] + py * width * 0.5),
                    (mid[0] + px * width, mid[1] + py * width),
                    tip,
                    (mid[0] - px * width, mid[1] - py * width),
                    (p0[0] - px * width * 0.5, p0[1] - py * width * 0.5),
                ],
                outer,
            ))

            # 焰芯更短更窄，颜色接近白蓝，形成内外两层。
            ilen = length * 0.55
            imid = (p0[0] + ux * ilen * 0.35, p0[1] + uy * ilen * 0.35)
            itip = (p0[0] + ux * ilen, p0[1] + uy * ilen)
            iw = width * 0.55
            out.append((
                plume_depth - 0.01,
                [
                    (p0[0] + px * iw * 0.5, p0[1] + py * iw * 0.5),
                    (imid[0] + px * iw, imid[1] + py * iw),
                    itip,
                    (imid[0] - px * iw, imid[1] - py * iw),
                    (p0[0] - px * iw * 0.5, p0[1] - py * iw * 0.5),
                ],
                inner,
            ))
        return out

    def _draw_missiles(self, world: World) -> None:
        """在飞格斗弹：弹体 + 尾迹 + 助推尾焰。命中与否只看内核动力学。"""
        live = [m for m in world.missiles if m.alive]
        if not live:
            self.missile_trails.clear()
            return
        live_ids = {id(m) for m in live}
        for key in [k for k in self.missile_trails if k not in live_ids]:
            del self.missile_trails[key]

        self.overlay.fill((0, 0, 0, 0))
        for m in live:
            pts = self.missile_trails.setdefault(id(m), [])
            p = (m.x, m.y, m.z)
            if not pts or math.dist(pts[-1], p) >= 25.0:
                pts.append(p)
                if len(pts) > 160:
                    del pts[0]

            color = TEAM_COLORS[m.owner]
            dim = tuple(int(c * 0.55) for c in color)
            for a, b in zip(pts, pts[1:]):
                seg = self._clip_segment(a, b)
                if seg is None:
                    continue
                pygame.draw.line(self.overlay, (*dim, 150), seg[0], seg[1], 1)

            s = self.to_screen(p)
            if s is None:
                continue
            depth = self._view(p)[2]
            speed = m.speed
            if speed < 1.0:
                continue
            ux, uy, uz = m.vx / speed, m.vy / speed, m.vz / speed
            ahead = self.to_screen(
                (m.x + ux * 60.0, m.y + uy * 60.0, m.z + uz * 60.0)
            )
            if ahead is None:
                continue
            dx, dy = ahead[0] - s[0], ahead[1] - s[1]
            norm = math.hypot(dx, dy)
            if norm < 0.5:
                continue
            sx, sy = dx / norm, dy / norm

            # 弹体长度按型号缩放：中距弹（R-27R/AIM-7M）明显比格斗弹长。
            size = clamp(
                26.0 * (m.spec.length / 3.0) * 7500.0 / max(depth, NEAR),
                6.0, 42.0,
            )
            tail = (s[0] - sx * size, s[1] - sy * size)
            body_color = color if m.guided else (150, 150, 150)
            pygame.draw.line(
                self.overlay, (*body_color, 255), tail, s, max(2, int(size * 0.16))
            )
            pygame.draw.circle(self.overlay, (255, 245, 210, 255), s, max(1, int(size * 0.10)))

            if m.fuel > 0.0:
                flame = size * 0.9
                tip = (tail[0] - sx * flame, tail[1] - sy * flame)
                pygame.draw.line(
                    self.overlay, (255, 176, 70, 220), tail, tip, max(2, int(size * 0.2))
                )
                pygame.draw.line(
                    self.overlay, (255, 244, 200, 235), tail,
                    (tail[0] - sx * flame * 0.45, tail[1] - sy * flame * 0.45), 1
                )
        self.screen.blit(self.overlay, (0, 0))

    def _draw_gunfire(self, world: World) -> None:
        """开火特效：炮口焰 + 沿真实炮线方向的曳光轨迹。"""
        self.overlay.fill((0, 0, 0, 0))
        tick = pygame.time.get_ticks() * 0.001
        for i, ac in enumerate(world.acs):
            if not ac.alive or not ac.firing:
                continue
            frame = self._model_frame(ac)
            if frame is None:
                continue
            model, k, att, _depth = frame
            nose = self._model_world(
                ac, att, (model["length"] / 2.0, 0.0, 0.0), k
            )
            ns = self.to_screen(nose)
            if ns is None:
                continue
            gx, gy, gz = gun_direction(ac)
            length = k * model["length"] * 2.6
            es = self.to_screen(
                (nose[0] + gx * length, nose[1] + gy * length, nose[2] + gz * length)
            )
            if es is None:
                continue

            # 曳光轨迹：外层辉光 + 亮芯。
            pygame.draw.line(self.overlay, (*TEAM_COLORS[i], 46), ns, es, 7)
            pygame.draw.line(self.overlay, (255, 154, 54, 170), ns, es, 3)
            pygame.draw.line(self.overlay, (255, 236, 170, 235), ns, es, 1)

            # 炮口焰：脉动光晕 + 星芒。
            pulse = 0.8 + 0.2 * math.sin(tick * 34.0)
            for radius, alpha in ((12, 36), (8, 80), (5, 160), (3, 235)):
                pygame.draw.circle(
                    self.overlay, (255, 214, 128, alpha), ns,
                    max(1, int(radius * pulse)),
                )
            dx, dy = es[0] - ns[0], es[1] - ns[1]
            norm = math.hypot(dx, dy) or 1.0
            ux, uy = dx / norm, dy / norm
            for span, alpha in ((16, 150), (10, 220)):
                tip = (ns[0] + ux * span, ns[1] + uy * span)
                pygame.draw.line(
                    self.overlay, (255, 245, 200, alpha), ns, tip, 2
                )
                for sign in (1.0, -1.0):
                    side = (
                        ns[0] - uy * span * 0.55 * sign,
                        ns[1] + ux * span * 0.55 * sign,
                    )
                    pygame.draw.line(
                        self.overlay, (255, 200, 120, alpha), ns, side, 1
                    )
        self.screen.blit(self.overlay, (0, 0))

    def _draw_aircraft(self, world: World, agents=None) -> None:
        for i, ac in enumerate(world.acs):
            if not ac.alive:
                continue
            pos = self.to_screen((ac.x, ac.y, ac.z))
            if pos is None:
                continue
            frame = self._model_frame(ac)
            if frame is None:
                continue
            size = clamp(7500.0 / frame[3], 0.55, 2.40)
            color = TEAM_COLORS[i]

            # 竖直投影线 + 地面影子：高度差一眼可见。
            ground = self.to_screen((ac.x, ac.y, world.z_floor))
            if ground is not None:
                dim = tuple(int(c * 0.45) for c in color)
                pygame.draw.line(self.screen, dim, pos, ground, 1)
                pygame.draw.circle(self.screen, dim, ground, max(2, int(4 * size)), 1)

            # 机体本身：按 psi/theta/phi 实时旋转，Su-57 与 F-16V 用各自模型。
            # GPU 模式下机体由 OpenGL 画，2D 层只补屏幕空间的尾焰。
            if self.gpu:
                self._draw_flames(ac, i, frame)
            else:
                self._draw_model(ac, i, frame)

            # 机动原语标签随飞机移动：从机体引一条短线；靠近屏幕边缘时翻到
            # 机体另一侧，不再做贴边夹取（夹取会让标签脱开飞机停在边框上）。
            primitive = getattr(agents[i], "primitive", "PURE") if agents else "PURE"
            label = self.font_small.render(f"{primitive} {ac.z:.0f}m", True, color)
            tw, th = label.get_size()
            gap = 20.0 * size
            ty = int(pos[1] + (-34.0 if i == 0 else 8.0))
            if ty < 4:
                ty = int(pos[1] + 10.0)
            # 默认挂在机体右侧，只有右边放不下时才翻到左侧，
            # 避免标签在飞机越过画面中线时来回跳。
            if pos[0] + gap + tw <= self.w - 8:
                tx = int(pos[0] + gap)
                ax = tx - 6.0
            else:
                tx = int(pos[0] - gap - tw)
                ax = tx + tw + 6.0
            ay = ty + th / 2.0
            pygame.draw.line(
                self.screen, color, (int(pos[0]), int(pos[1])), (ax, ay), 1
            )
            pygame.draw.rect(self.screen, BG, pygame.Rect(tx - 3, ty - 2, tw + 6, th + 4))
            self.screen.blit(label, (tx, ty))

    def _draw_flames(self, ac, team: int, frame) -> None:
        """GPU 模式下尾焰仍画在 2D 层：屏幕空间锥形，叠在机体之上。"""
        model, k, att, depth = frame
        for _d, poly, color in self._flame_faces(ac, model, k, att, team, depth):
            pygame.draw.polygon(self.screen, color, poly)

    def _draw_altitude_ladder(self, world: World) -> None:
        """右侧高度尺：两台飞机在垂直方向的相对位置。"""
        x = self.w - 30
        top, bottom = 150, self.h - 150
        pygame.draw.line(self.screen, BORDER, (x, top), (x, bottom), 1)

        zmin, zmax = world.z_floor, world.z_ceiling
        # 高空场景下 1 km 一格太密，按总高度范围自动换挡。
        step = 1000.0 if zmax - zmin <= 8000.0 else 5000.0
        z = math.ceil(zmin / step) * step
        while z <= zmax:
            sy = bottom - (z - zmin) / (zmax - zmin) * (bottom - top)
            pygame.draw.line(self.screen, GRID, (x - 5, sy), (x + 5, sy), 1)
            label = self.font_small.render(f"{z / 1000:.0f}k", True, SUB)
            self.screen.blit(label, (x - 30, sy - 8))
            z += step

        cap = self.font_small.render("ALT", True, SUB)
        self.screen.blit(cap, (x - 18, top - 20))

        for i, ac in enumerate(world.acs):
            if not ac.alive:
                continue
            sy = bottom - (ac.z - zmin) / (zmax - zmin) * (bottom - top)
            # 红蓝左右错开，高度重合时也能看出是两个点。
            mx = x - 7 if i == 0 else x + 7
            pygame.draw.circle(self.screen, TEAM_COLORS[i], (mx, sy), 4)

    def _draw_hud(self, world: World, paused: bool, agents=None) -> None:
        for i, ac in enumerate(world.acs):
            agent = agents[i] if agents else None
            incoming = sum(
                1 for m in world.missiles if m.alive and m.target == i
            )
            self._draw_aircraft_panel(
                ac, agent, TEAM_COLORS[i], right=(i == 1), incoming=incoming
            )

        a, b = world.acs
        dist = math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        line = (
            f"T {world.t:5.1f}s    DIST {dist:5.0f}m    "
            f"dALT {b.z - a.z:+5.0f}m    dE {a.energy - b.energy:6.0f} J/kg"
        )
        surf = self.font.render(line, True, TEXT)
        self.screen.blit(surf, (self.w / 2 - surf.get_width() / 2, self.h - 34))

        hint = "ESC/Q quit    SPACE pause    R reset"
        hs = self.font_small.render(hint, True, SUB)
        self.screen.blit(hs, (self.w / 2 - hs.get_width() / 2, self.h - 58))

        self._draw_missile_status(world)
        self._draw_attitude_panel(world)

        if paused:
            ps = self.font.render("PAUSED", True, TEXT)
            self.screen.blit(ps, (self.w / 2 - ps.get_width() / 2, 18))

    def _draw_missile_status(self, world: World) -> None:
        """在飞导弹状态：型号、制导状态、速度、到目标距离与三维位置。"""
        live = [m for m in world.missiles if m.alive]
        if not live:
            return
        width, row_h = 476, 15
        height = 8 + row_h * len(live)
        x = self.w // 2 - width // 2
        y = 236
        pygame.draw.rect(self.screen, (26, 29, 34), pygame.Rect(x, y, width, height))
        pygame.draw.rect(self.screen, (88, 94, 104), pygame.Rect(x, y, width, height), 1)

        for k, m in enumerate(live):
            target = world.acs[m.target] if 0 <= m.target < len(world.acs) else None
            d = (
                math.dist((m.x, m.y, m.z), (target.x, target.y, target.z))
                if target is not None else 0.0
            )
            color = TEAM_COLORS[m.owner]
            if not m.guided:
                state = "BALLISTIC"
            elif m.decoy_t > 0.0:
                state = "DECOY"
            elif m.spec.is_arh:
                # 主动弹：中段靠载机雷达/数据链，末段自主截获后发射后不管。
                state = "ACTIVE" if m.active else "MIDCRS"
            elif m.spec.is_sarh:
                state = "ILLUM"
            else:
                state = "GUIDE"
            text = (
                f"{m.spec.name} -> {target.spec.name if target else '-':7s} {state:9s} "
                f"v={m.speed:5.0f}  d={d:6.0f}  "
                f"x={m.x / 1000.0:+6.1f} y={m.y / 1000.0:+6.1f} km  alt={m.z:5.0f} m"
            )
            pygame.draw.circle(
                self.screen, color, (x + 10, y + 8 + k * row_h + 5), 3
            )
            surf = self.font_small.render(text, True, TEXT)
            self.screen.blit(surf, (x + 20, y + 3 + k * row_h))

    def _draw_counters(self, world: World) -> None:
        """在飞干扰物：箔条画白色云团，热诱弹画橙色亮点。"""
        for cm in world.counters:
            if not cm.alive:
                continue
            pos = cm.position
            s = self.to_screen(pos)
            if s is None:
                continue
            depth = max(self._view(pos)[2], NEAR)
            r = clamp(self.focal * 2.6 / depth, 1.5, 6.0)
            if cm.kind == "chaff":
                pygame.draw.circle(self.screen, (206, 208, 214), s, int(r))
                pygame.draw.circle(self.screen, (150, 154, 160), s, int(r), 1)
            else:
                pygame.draw.circle(self.screen, (255, 170, 70), s, int(r * 1.3))

    def _draw_attitude_panel(self, world: World) -> None:
        """顶部中央的姿态框：实时显示双方三维模型、姿态、油门与来袭方向。

        姿态坐标系以地面垂直向上为 Z 轴；机体主体色按队伍区分，被导弹威胁时
        机体表面沿“威胁来向”做队伍色 → 告警红的渐变，越靠近来袭一侧越红。
        油门以滑动条 + 百分比给出，尾焰长度与当前推力（油门/加速度）同步。
        """
        width, height = 440, 205
        px = (self.w - width) // 2
        py = 20
        self._attitude_rects = []
        pygame.draw.rect(self.screen, (26, 29, 34), pygame.Rect(px, py, width, height))
        pygame.draw.rect(self.screen, BORDER, pygame.Rect(px, py, width, height), 1)
        pygame.draw.line(
            self.screen, BORDER, (px + width // 2, py + 6),
            (px + width // 2, py + height - 6), 1,
        )

        half = width // 2
        for i, ac in enumerate(world.acs):
            hx = px + i * half
            color = TEAM_COLORS[i]
            name = self.font_small.render(ac.spec.name, True, color)
            self.screen.blit(name, (hx + 10, py + 5))

            rect = pygame.Rect(hx + 8, py + 22, half - 16, 140)
            self._attitude_rects.append(rect)
            # GPU 模式下机体由 GL 补画，这里只留地面线、Z 轴和尾焰。
            self._draw_attitude_model(
                rect, world, i, ac, color, bodies=not self.gpu
            )

            # 油门滑动条：填充比例即当前功率指令；加力时整条变成白蓝色。
            bx, by, bw, bh = hx + 46, py + 173, 104, 9
            thr = clamp(ac.throttle, 0.0, 1.0) if ac.alive else 0.0
            ab = clamp(getattr(ac, "ab", 0.0), 0.0, 1.0) if ac.alive else 0.0
            pygame.draw.rect(self.screen, (44, 48, 55), pygame.Rect(bx, by, bw, bh))
            pygame.draw.rect(
                self.screen,
                (236, 244, 255) if ab > 0.05 else (255, 168, 64),
                pygame.Rect(bx, by, int(bw * thr), bh),
            )
            pygame.draw.rect(self.screen, SUB, pygame.Rect(bx, by, bw, bh), 1)
            self.screen.blit(self.font_small.render("THR", True, SUB), (hx + 10, by - 2))
            self.screen.blit(
                self.font_small.render(
                    f"{thr * 100:3.0f}%{' AB' if ab > 0.05 else '   '}",
                    True, TEXT,
                ),
                (bx + bw + 8, by - 2),
            )

            # 燃油条：加力油耗远高于军推，条长直接给出剩余油量。
            cap = ac.spec.fuel_capacity
            frac = clamp(ac.fuel / cap, 0.0, 1.0) if cap > 0.0 and ac.alive else 1.0
            fy = py + 192
            pygame.draw.rect(self.screen, (44, 48, 55), pygame.Rect(bx, fy, bw, bh))
            pygame.draw.rect(
                self.screen, (118, 196, 148), pygame.Rect(bx, fy, int(bw * frac), bh)
            )
            pygame.draw.rect(self.screen, SUB, pygame.Rect(bx, fy, bw, bh), 1)
            self.screen.blit(self.font_small.render("FUEL", True, SUB), (hx + 10, fy - 2))
            self.screen.blit(
                self.font_small.render(f"{frac * 100:3.0f}%", True, TEXT),
                (bx + bw + 8, fy - 2),
            )

    def _draw_attitude_model(self, rect, world: World, i: int, ac, color,
                             bodies: bool = True) -> None:
        """在姿态框里画一架迷你 3D 机体：固定等距视角 + 正交投影。

        bodies=False 时只画地面线、Z 轴和尾焰（GPU 模式下机体交给 GL 渲染）。
        """
        model = MODEL_BY_NAME.get(ac.spec.name, GENERIC_MODEL)
        length = model["length"]
        att = attitude_of(ac) if ac.alive else (0.0, 0.0, 0.0)
        xb, yb, zb = body_axes(*att)

        # 固定观察方向（世界系），坐标系以地面 Z 轴向上为基准。
        view, right, up = _attitude_basis()
        cx, cy = rect.centerx, rect.centery + 6
        scale = 108.0 / length

        def to_world(p):
            return (
                xb[0] * p[0] + yb[0] * p[1] + zb[0] * p[2],
                xb[1] * p[0] + yb[1] * p[1] + zb[1] * p[2],
                xb[2] * p[0] + yb[2] * p[1] + zb[2] * p[2],
            )

        def project(p):
            wx = xb[0] * p[0] + yb[0] * p[1] + zb[0] * p[2]
            wy = xb[1] * p[0] + yb[1] * p[1] + zb[1] * p[2]
            wz = xb[2] * p[0] + yb[2] * p[1] + zb[2] * p[2]
            return (
                cx + (wx * right[0] + wy * right[1] + wz * right[2]) * scale,
                cy - (wx * up[0] + wy * up[1] + wz * up[2]) * scale,
                wx * view[0] + wy * view[1] + wz * view[2],
            )

        # 地面参考线与 Z 轴（地面垂直向上），让姿态可读。
        pygame.draw.line(
            self.screen, GRID, (rect.left + 4, rect.bottom - 4),
            (rect.right - 4, rect.bottom - 4), 1,
        )
        zx = rect.left + 12
        pygame.draw.line(
            self.screen, (120, 132, 148), (zx, rect.bottom - 6), (zx, rect.top + 10), 1
        )
        pygame.draw.lines(
            self.screen, (120, 132, 148), False,
            [(zx - 3, rect.top + 15), (zx, rect.top + 9), (zx + 3, rect.top + 15)], 1,
        )
        self.screen.blit(self.font_small.render("Z", True, (150, 162, 178)),
                         (zx + 4, rect.top + 8))

        # 来袭方向：最近一枚仍在制导、指向本机的导弹，投影到机体水平面。
        threat = None
        best_d = 1e18
        for m in world.missiles:
            if not m.alive or m.target != i or not m.guided:
                continue
            d = (m.x - ac.x) ** 2 + (m.y - ac.y) ** 2
            if d < best_d:
                best_d, threat = d, m
        fx = fy = 0.0
        if threat is not None:
            dx, dy = threat.x - ac.x, threat.y - ac.y
            fx = dx * xb[0] + dy * xb[1]
            fy = dx * yb[0] + dy * yb[1]
            n = math.hypot(fx, fy)
            if n > 1e-6:
                fx, fy = fx / n, fy / n

        base_color = color if ac.alive else (96, 100, 108)
        # 告警色取黄橙：红蓝两队本体上用纯红都容易糊在一起，黄橙在两种底色
        # 上都能形成清晰的方向渐变。
        red = (255, 210, 90)

        # 尾焰先画，长度随油门/推力变化，机体会自然盖住焰根。
        if ac.alive and model.get("engines"):
            level = clamp(self._flame[i], 0.0, 1.0)
            ab = clamp(getattr(ac, "ab", 0.0), 0.0, 1.0)
            tail = project((-length / 2.0, 0.0, 0.0))
            tip = project((
                -length / 2.0 - length * (0.10 + 0.34 * level), 0.0, 0.0
            ))
            dx, dy = tip[0] - tail[0], tip[1] - tail[1]
            n = math.hypot(dx, dy)
            if n > 1e-3:
                ux, uy = dx / n, dy / n
                wid = 3.0 + 7.0 * level
                tip2 = (tip[0], tip[1])
                pygame.draw.polygon(
                    self.screen, _mix((255, 138, 44), (168, 192, 255), 0.55 * ab), [
                        (tail[0] - uy * wid, tail[1] + ux * wid),
                        tip2,
                        (tail[0] + uy * wid, tail[1] - ux * wid),
                    ])
                inner = (tail[0] + ux * n * 0.55, tail[1] + uy * n * 0.55)
                pygame.draw.polygon(
                    self.screen, _mix((255, 238, 196), (238, 244, 255), 0.75 * ab), [
                        (tail[0] - uy * wid * 0.4, tail[1] + ux * wid * 0.4),
                        inner,
                        (tail[0] + uy * wid * 0.4, tail[1] - ux * wid * 0.4),
                    ])

        # GPU 模式：机体交给 GL 渲染，2D 层到此为止。
        if not bodies:
            return

        faces = []
        for poly, kind in model["faces"]:
            wp = [to_world(p) for p in poly]
            sp = [project(p) for p in poly]
            center = (
                sum(p[0] for p in wp) / len(wp),
                sum(p[1] for p in wp) / len(wp),
                sum(p[2] for p in wp) / len(wp),
            )
            depth = center[0] * view[0] + center[1] * view[1] + center[2] * view[2]

            # 面法线做固定方向光照，避免纯色剪影。
            e1 = (wp[1][0] - wp[0][0], wp[1][1] - wp[0][1], wp[1][2] - wp[0][2])
            e2 = (wp[2][0] - wp[0][0], wp[2][1] - wp[0][1], wp[2][2] - wp[0][2])
            normal = _norm(_cross(e1, e2))
            light = 0.52 + 0.48 * abs(_dot(normal, LIGHT))

            if kind == "canopy":
                base = CANOPY_COLOR
            elif kind == "intake":
                base = INTAKE_COLOR
            elif kind == "nozzle":
                base = NOZZLE_COLOR
            else:
                base = base_color
            # 威胁渐变：面心在机体受威胁方向上的投影决定红色混入比例。
            if threat is not None:
                bx_ = sum(p[0] for p in poly) / len(poly)
                by_ = sum(p[1] for p in poly) / len(poly)
                t = clamp(
                    0.5 + 0.5 * (bx_ * fx + by_ * fy) / max(length * 0.5, 1e-6),
                    0.0, 1.0,
                )
                mix = 0.85 * t
                base = (
                    base[0] + (red[0] - base[0]) * mix,
                    base[1] + (red[1] - base[1]) * mix,
                    base[2] + (red[2] - base[2]) * mix,
                )
            color_ = tuple(int(clamp(c * light, 0, 255)) for c in base)
            faces.append((depth, [(p[0], p[1]) for p in sp], color_))

        # 画家算法：远的先画，近面覆盖。
        for _depth, pts, color_ in sorted(faces, key=lambda f: f[0]):
            pygame.draw.polygon(self.screen, color_, pts)

    def _draw_aircraft_panel(self, ac, agent, color, right: bool, incoming: int = 0) -> None:
        width, height = 236, 194
        x = self.w - width - 20 if right else 20
        y = 20
        pygame.draw.rect(self.screen, (26, 29, 34), pygame.Rect(x, y, width, height))
        pygame.draw.rect(self.screen, color, pygame.Rect(x, y, width, height), 1)
        pygame.draw.rect(self.screen, color, pygame.Rect(x, y, 4, height))

        name = self.font.render(ac.spec.name, True, color)
        self.screen.blit(name, (x + 14, y + 8))

        line = f"SPD {ac.v:5.1f} m/s   G {ac.n:4.1f}"
        self.screen.blit(self.font_small.render(line, True, TEXT), (x + 14, y + 29))

        line = f"ALT {ac.z:5.0f} m   GAM {math.degrees(ac.gamma):+5.1f}"
        self.screen.blit(self.font_small.render(line, True, SUB), (x + 14, y + 47))

        # 姿态读数与 3D 模型同步，来自同一套 psi/theta/phi。
        _psi, theta, phi = attitude_of(ac)
        line = f"PITCH {math.degrees(theta):+5.1f}   ROLL {math.degrees(phi):+6.1f}"
        self.screen.blit(self.font_small.render(line, True, TEXT), (x + 14, y + 65))

        # 格斗弹余量与导引头状态
        msl = ac.spec.missile
        if msl is not None:
            if ac.locked:
                status, st_color = "LOCK", (255, 214, 96)
            elif ac.lock_timer > 0.0:
                pct = 100.0 * ac.lock_timer / msl.lock_time
                status, st_color = f"SEEK {pct:3.0f}%", (214, 202, 142)
            else:
                status, st_color = "SEEK", SUB
            line = f"MSL {ac.missiles}/{ac.spec.missile_count}  {status}"
            self.screen.blit(self.font_small.render(line, True, st_color), (x + 14, y + 83))
            if incoming:
                warn = self.font_small.render(f"IN {incoming}", True, (255, 122, 88))
                self.screen.blit(
                    warn, (x + width - 14 - warn.get_width(), y + 83)
                )

        # 中距弹余量与雷达跟踪状态（主动弹末段截获前需要载机跟踪）
        rsl = ac.spec.radar_missile
        if rsl is not None:
            if ac.radar_locked:
                rstatus, rcolor = "LOCK", (255, 214, 96)
            elif ac.radar_lock_timer > 0.0:
                pct = 100.0 * ac.radar_lock_timer / rsl.lock_time
                rstatus, rcolor = f"TRK {pct:3.0f}%", (214, 202, 142)
            else:
                rstatus, rcolor = "RDR", SUB
            line = f"RDR {ac.radar_missiles}/{ac.spec.radar_missile_count}  {rstatus}"
            self.screen.blit(self.font_small.render(line, True, rcolor), (x + 14, y + 101))

        # 机动原语已改为跟随飞机的浮动标签，面板这里保留态势与干扰物余量。
        line = f"CH {ac.chaff:2d}  FL {ac.flares:2d}"
        self.screen.blit(self.font_small.render(line, True, SUB), (x + 14, y + 119))
        situation = getattr(agent, "situation", "") if agent else ""
        sit = self.font_small.render(f"SIT {situation}", True, SUB)
        self.screen.blit(sit, (x + width - 14 - sit.get_width(), y + 119))

        # 燃油与功率档位：AB = 加力，MIL = 军推；油尽后只能军推。
        cap = ac.spec.fuel_capacity
        fuel_frac = clamp(ac.fuel / cap, 0.0, 1.0) if cap > 0.0 else 1.0
        ab_on = ac.ab > 0.05
        line = f"FUEL {fuel_frac * 100:3.0f}%  {'AB' if ab_on else 'MIL'}"
        self.screen.blit(
            self.font_small.render(
                line, True, (255, 168, 64) if ab_on else (150, 200, 160)
            ),
            (x + 14, y + 135),
        )

        # HP 条 + 数值
        bx, by, bw, bh = x + 14, y + 161, 132, 12
        pygame.draw.rect(self.screen, (44, 48, 55), pygame.Rect(bx, by, bw, bh))
        ratio = max(0.0, ac.hp / ac.spec.hp_max)
        pygame.draw.rect(self.screen, color, pygame.Rect(bx, by, bw * ratio, bh))

        hp_text = self.font_small.render(f"HP {ac.hp:5.1f}", True, TEXT)
        self.screen.blit(hp_text, (bx + bw + 8, by + bh / 2 - hp_text.get_height() / 2))

    def _draw_result(self, world: World) -> None:
        self.overlay.fill((8, 9, 11, 190))
        self.screen.blit(self.overlay, (0, 0))

        if world.result == "draw":
            title, color = "DRAW", TEXT
            detail_text = (
                "mutual kill" if world.reason_code == "kill"
                else "time-up, HP even"
            )
        else:
            idx = 0 if world.result == "red" else 1
            loser = world.acs[1 - idx].spec.name
            title, color = f"{world.acs[idx].spec.name} WINS", TEAM_COLORS[idx]
            detail_text = {
                "kill": f"{loser} shot down",
                "boundary": f"{loser} out of bounds",
                "timeout": "time-up, HP advantage",
            }.get(world.reason_code, "")

        surf = self.font_big.render(title, True, color)
        self.screen.blit(surf, (self.w / 2 - surf.get_width() / 2, self.h / 2 - 60))

        detail = self.font.render(detail_text, True, TEXT)
        self.screen.blit(detail, (self.w / 2 - detail.get_width() / 2, self.h / 2 + 10))

        tip = self.font_small.render("press R to restart", True, SUB)
        self.screen.blit(tip, (self.w / 2 - tip.get_width() / 2, self.h / 2 + 42))

    # ---------- GPU 渲染（OpenGL 3.3 core / moderngl） ----------

    def _init_gpu(self, size: tuple[int, int]) -> bool:
        """尝试建立 GL 上下文并初始化渲染资源；失败返回 False。"""
        try:
            import moderngl
        except Exception:
            return False
        try:
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
            pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
            pygame.display.gl_set_attribute(
                pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
            )
            pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
            pygame.display.gl_set_attribute(pygame.GL_DOUBLEBUFFER, 1)
            pygame.display.set_mode(size, pygame.OPENGL | pygame.DOUBLEBUF)
            ctx = moderngl.create_context()
        except Exception:
            # 没有可用的 GL（远程会话 / 纯软件驱动）：重开显示，交给 CPU 路径。
            pygame.display.quit()
            pygame.display.init()
            return False

        self.mgl = moderngl
        self.ctx = ctx
        self.lit_prog = ctx.program(vertex_shader=LIT_VS, fragment_shader=LIT_FS)
        self.flat_prog = ctx.program(vertex_shader=FLAT_VS, fragment_shader=LIT_FS)
        self.hud_prog = ctx.program(vertex_shader=HUD_VS, fragment_shader=HUD_FS)
        self.lit_prog["u_light"].value = LIGHT
        self.hud_prog["u_tex"].value = 0

        # HUD 合成用的全屏四边形：翻转上传纹理，使 v=0 对应画面底部。
        quad = array("f", (
            -1.0, -1.0, 0.0, 0.0,  1.0, -1.0, 1.0, 0.0,  1.0, 1.0, 1.0, 1.0,
            -1.0, -1.0, 0.0, 0.0,  1.0, 1.0, 1.0, 1.0,  -1.0, 1.0, 0.0, 1.0,
        )).tobytes()
        self._hud_vbo = ctx.buffer(quad)
        self.hud_vao = ctx.vertex_array(
            self.hud_prog, [(self._hud_vbo, "2f 2f", "in_pos", "in_uv")]
        )
        # 姿态框模型先渲到离屏纹理，再按面板矩形贴回来。
        self._panel_vbo = ctx.buffer(reserve=6 * 4 * 4)
        self._panel_vao = ctx.vertex_array(
            self.hud_prog, [(self._panel_vbo, "2f 2f", "in_pos", "in_uv")]
        )
        self.hud_tex = ctx.texture(size, 4)
        self.hud_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)

        self._meshes: dict = {}
        self._trail_buf: list = [None, None]
        self._floor = None
        self._panel_fbo = None
        self._panel_tex = None
        self._panel_size = None
        # GPU 模式下 self.screen 指向透明 HUD 面，所有 2D 绘制都落在这里。
        self.screen = self.hud = pygame.Surface(size, pygame.SRCALPHA)
        return True

    def _proj_mat(self, near: float, far: float):
        """透视投影：与 CPU 的 focal 一致，屏幕坐标可直接对齐。

        采用 +z 朝前的相机系，深度在 [near, far] 上映射到 [-1, 1]。
        """
        a = 2.0 * self.focal / self.w
        b = 2.0 * self.focal / self.h
        return (
            (a, 0.0, 0.0, 0.0),
            (0.0, b, 0.0, 0.0),
            (0.0, 0.0, (far + near) / (far - near),
             -2.0 * far * near / (far - near)),
            (0.0, 0.0, 1.0, 0.0),
        )

    def _view_mat(self):
        e, r, u, fw = self.eye, self.right, self.up, self.forward
        return (
            (r[0], r[1], r[2], -_dot(r, e)),
            (u[0], u[1], u[2], -_dot(u, e)),
            (fw[0], fw[1], fw[2], -_dot(fw, e)),
            (0.0, 0.0, 0.0, 1.0),
        )

    def _floor_vao(self, arena: float, z_floor: float):
        """最低高度层网格 + 空域边界，按 (arena, z_floor) 缓存一次。"""
        key = (arena, z_floor)
        if self._floor is not None:
            if self._floor[3] == key:
                return self._floor[0]
            self._floor[0].release()
            self._floor[1].release()
        half = arena / 2.0
        step = 1000.0
        grid = tuple(c / 255.0 for c in GRID)
        edge = tuple(c / 255.0 for c in BORDER)
        verts: list[float] = []

        def add(p, col):
            verts.extend((p[0], p[1], p[2], col[0], col[1], col[2]))

        n = int(round(arena / step))
        for k in range(n + 1):
            t = -half + k * step
            add((t, -half, z_floor), grid)
            add((t, half, z_floor), grid)
            add((-half, t, z_floor), grid)
            add((half, t, z_floor), grid)
        corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
        for i in range(4):
            add((*corners[i], z_floor), edge)
            add((*corners[(i + 1) % 4], z_floor), edge)

        buf = self.ctx.buffer(array("f", verts).tobytes())
        vao = self.ctx.vertex_array(
            self.flat_prog, [(buf, "3f 3f", "in_pos", "in_col")]
        )
        self._floor = (vao, buf, len(verts) // 6, key)
        return vao

    def _trail_vao(self, i: int, trail):
        """航迹折线：只在采样点增加时重建缓冲。"""
        if len(trail) < 2:
            return None
        slot = self._trail_buf[i]
        if slot is not None and slot[2] == len(trail):
            return slot[0]
        if slot is not None:
            slot[0].release()
            slot[1].release()
        col = tuple(c / 255.0 * 0.60 for c in TEAM_COLORS[i])
        verts: list[float] = []
        for p in trail:
            verts.extend((p[0], p[1], p[2], col[0], col[1], col[2]))
        buf = self.ctx.buffer(array("f", verts).tobytes())
        vao = self.ctx.vertex_array(
            self.flat_prog, [(buf, "3f 3f", "in_pos", "in_col")]
        )
        self._trail_buf[i] = (vao, buf, len(trail))
        return vao

    def _mesh_vao(self, model, team: int):
        """把机体模型烘成三角网格：顶点色 + 面法线，按 (模型, 队伍) 缓存。"""
        key = (id(model), team)
        hit = self._meshes.get(key)
        if hit is not None:
            return hit[0]
        verts: list[float] = []
        for pts, kind in model["faces"]:
            if len(pts) < 3:
                continue
            if kind == "canopy":
                base = CANOPY_COLOR
            elif kind == "intake":
                base = INTAKE_COLOR
            elif kind == "nozzle":
                base = NOZZLE_COLOR
            else:
                shade = PART_SHADE.get(kind, 1.0)
                base = tuple(int(c * shade) for c in TEAM_COLORS[team])
            rgb = (base[0] / 255.0, base[1] / 255.0, base[2] / 255.0)
            e1 = (pts[1][0] - pts[0][0], pts[1][1] - pts[0][1],
                  pts[1][2] - pts[0][2])
            e2 = (pts[2][0] - pts[0][0], pts[2][1] - pts[0][1],
                  pts[2][2] - pts[0][2])
            nx, ny, nz = _cross(e1, e2)
            ln = math.sqrt(nx * nx + ny * ny + nz * nz)
            if ln < 1e-9:
                nx, ny, nz = 0.0, 0.0, 1.0
            else:
                nx, ny, nz = nx / ln, ny / ln, nz / ln
            # 多边形按扇形三角化，OBJ 的绕序不统一，统一双面绘制由法线取绝对值处理。
            for j in range(1, len(pts) - 1):
                for p in (pts[0], pts[j], pts[j + 1]):
                    verts.extend((
                        p[0], p[1], p[2], nx, ny, nz, rgb[0], rgb[1], rgb[2],
                    ))
        if not verts:
            return None
        buf = self.ctx.buffer(array("f", verts).tobytes())
        vao = self.ctx.vertex_array(
            self.lit_prog, [(buf, "3f 3f 3f", "in_pos", "in_nrm", "in_col")]
        )
        self._meshes[key] = (vao, buf)
        return vao

    def _model_matrix(self, ac, att, scale: float):
        """机体系 → 世界系：机体三轴按姿态旋转、整体缩放并平移到机体位置。"""
        xb, yb, zb = body_axes(*att)
        return (
            (scale * xb[0], scale * yb[0], scale * zb[0], ac.x),
            (scale * xb[1], scale * yb[1], scale * zb[1], ac.y),
            (scale * xb[2], scale * yb[2], scale * zb[2], ac.z),
            (0.0, 0.0, 0.0, 1.0),
        )

    def _gpu_scene(self, world: World) -> None:
        """3D 场景：地面网格、双方航迹、两架机体。"""
        ctx, mgl = self.ctx, self.mgl
        ctx.viewport = (0, 0, self.w, self.h)
        ctx.clear(BG[0] / 255.0, BG[1] / 255.0, BG[2] / 255.0, 1.0)
        ctx.enable(mgl.DEPTH_TEST)
        ctx.disable(mgl.CULL_FACE)
        ctx.disable(mgl.BLEND)

        # 近裁剪面跟着取景距离走：几十公里外看机体时，近平面太小会让
        # 24 位深度在这个距离上几乎全部量化到同一个值（面片互相穿插）。
        near = max(20.0, self.dist * 0.02)
        far = max(self.dist * 8.0, world.arena * 1.6)
        pv = _mat4_mul(self._proj_mat(near, far), self._view_mat())

        self.flat_prog["u_mvp"].value = _mat4_flat(pv)
        self._floor_vao(world.arena, world.z_floor).render(mgl.LINES)

        for i, trail in enumerate(self.trails):
            vao = self._trail_vao(i, trail)
            if vao is not None:
                vao.render(mgl.LINE_STRIP)

        for i, ac in enumerate(world.acs):
            if not ac.alive:
                continue
            frame = self._model_frame(ac)
            if frame is None:
                continue
            model, k, att, _depth = frame
            vao = self._mesh_vao(model, i)
            if vao is None:
                continue
            xb, yb, zb = body_axes(*att)
            rot = (
                (xb[0], yb[0], zb[0], 0.0),
                (xb[1], yb[1], zb[1], 0.0),
                (xb[2], yb[2], zb[2], 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
            self.lit_prog["u_mvp"].value = _mat4_flat(
                _mat4_mul(pv, self._model_matrix(ac, att, k))
            )
            self.lit_prog["u_rot"].value = _mat4_flat(rot)
            vao.render(mgl.TRIANGLES)

    def _gpu_hud(self) -> None:
        """把 pygame 画好的 HUD 面作为 RGBA 纹理叠在 3D 画面上。"""
        ctx, mgl = self.ctx, self.mgl
        ctx.viewport = (0, 0, self.w, self.h)
        ctx.disable(mgl.DEPTH_TEST)
        ctx.enable(mgl.BLEND)
        ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
        self.hud_tex.write(pygame.image.tobytes(self.hud, "RGBA", True))
        self.hud_tex.use(0)
        self.hud_vao.render(mgl.TRIANGLES)

    def _panel_target(self, size):
        """姿态框用的离屏缓冲：自带深度，避免和主场景的深度缓冲互相干扰。"""
        if self._panel_fbo is not None and self._panel_size == size:
            return self._panel_fbo
        if self._panel_fbo is not None:
            self._panel_fbo.release()
            self._panel_tex.release()
        tex = self.ctx.texture(size, 4)
        tex.filter = (self.mgl.NEAREST, self.mgl.NEAREST)
        fbo = self.ctx.framebuffer(
            color_attachments=[tex],
            depth_attachment=self.ctx.depth_texture(size),
        )
        self._panel_tex, self._panel_fbo, self._panel_size = tex, fbo, size
        return fbo

    def _gpu_attitude(self, world: World) -> None:
        """在顶部姿态框里补画两架迷你机体（固定等距视角的正交投影）。"""
        if not self._attitude_rects:
            return
        ctx, mgl = self.ctx, self.mgl
        view, right, up = _attitude_basis()
        view_mat = (
            (right[0], right[1], right[2], 0.0),
            (up[0], up[1], up[2], 0.0),
            (view[0], view[1], view[2], 0.0),
            (0.0, 0.0, 0.0, 1.0),
        )
        for i, rect in enumerate(self._attitude_rects):
            ac = world.acs[i]
            if not ac.alive:
                continue
            model = MODEL_BY_NAME.get(ac.spec.name, GENERIC_MODEL)
            vao = self._mesh_vao(model, i)
            if vao is None:
                continue
            att = attitude_of(ac)
            scale = 108.0 / model["length"]
            xb, yb, zb = body_axes(*att)
            rot = (
                (xb[0], yb[0], zb[0], 0.0),
                (xb[1], yb[1], zb[1], 0.0),
                (xb[2], yb[2], zb[2], 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
            proj = (
                (2.0 / rect.width, 0.0, 0.0, 0.0),
                (0.0, 2.0 / rect.height, 0.0, 0.0),
                (0.0, 0.0, 1.0 / 2000.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
            # 与 2D 版本一致：模型中心在面板中心下方 6 px。
            shift = (
                (1.0, 0.0, 0.0, 0.0),
                (0.0, 1.0, 0.0, -6.0),
                (0.0, 0.0, 1.0, 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
            model_mat = (
                (scale * xb[0], scale * yb[0], scale * zb[0], 0.0),
                (scale * xb[1], scale * yb[1], scale * zb[1], 0.0),
                (scale * xb[2], scale * yb[2], scale * zb[2], 0.0),
                (0.0, 0.0, 0.0, 1.0),
            )
            mvp = _mat4_mul(proj, _mat4_mul(shift, _mat4_mul(view_mat, model_mat)))

            fbo = self._panel_target((rect.width, rect.height))
            fbo.use()
            ctx.viewport = (0, 0, rect.width, rect.height)
            ctx.clear(0.0, 0.0, 0.0, 0.0, 1.0)
            ctx.enable(mgl.DEPTH_TEST)
            ctx.disable(mgl.CULL_FACE)
            ctx.disable(mgl.BLEND)
            self.lit_prog["u_mvp"].value = _mat4_flat(mvp)
            self.lit_prog["u_rot"].value = _mat4_flat(rot)
            vao.render(mgl.TRIANGLES)

            # 贴回主画面：面板矩形换算成 NDC，纹理 v=0 在矩形底边。
            ctx.screen.use()
            ctx.viewport = (0, 0, self.w, self.h)
            ctx.disable(mgl.DEPTH_TEST)
            ctx.enable(mgl.BLEND)
            ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
            x0 = 2.0 * rect.left / self.w - 1.0
            x1 = 2.0 * rect.right / self.w - 1.0
            y0 = 1.0 - 2.0 * rect.bottom / self.h
            y1 = 1.0 - 2.0 * rect.top / self.h
            quad = array("f", (
                x0, y0, 0.0, 0.0,  x1, y0, 1.0, 0.0,  x1, y1, 1.0, 1.0,
                x0, y0, 0.0, 0.0,  x1, y1, 1.0, 1.0,  x0, y1, 0.0, 1.0,
            )).tobytes()
            self._panel_vbo.write(quad)
            self._panel_tex.use(0)
            self._panel_vao.render(mgl.TRIANGLES)
