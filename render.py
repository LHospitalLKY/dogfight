"""M2 观战窗口：pygame 透视 3D 渲染 + 机体模型 + 机动标签，只读 World 状态。

相机围绕两机连线的中点做轨道飞行，俯角约 26°，地面网格和竖直投影线
用来给出高度的直观参照；机体用对应的三维模型表示，姿态直接取内核的
psi/theta/phi（点质量内核用 gamma/-bank 还原），开火时叠加炮口焰与炮线。
"""
from __future__ import annotations

import math
import os
import time
from array import array

import numpy as np
import pygame
from pygame import gfxdraw

from sim import (
    CONTACT_DETECT,
    CONTACT_TRACK,
    G,
    RADAR_DETECT_FOV,
    World,
    air_density,
    body_axes,
    clamp,
    gun_direction,
    nose_direction,
    predict_position,
    rcs_scale,
    world_velocity,
    wrap_pi,
)

# 配色全部收敛到 theme.py：背景压深压冷，文字、机体和航迹统一提高明度，
# 红方偏橙红、蓝方偏亮青蓝，与 HUD 绿、告警黄和地形图的青灰都不会混色。
from theme import (  # noqa: E402
    ACCENT_RED,
    AIR_EDGE,
    AIR_GRID,
    AMB,
    BG,
    BORDER,
    CANOPY_COLOR,
    CLAY_AO,
    CLAY_CANOPY,
    CLAY_EDGE,
    CLAY_INTAKE,
    CLAY_LIT,
    CLAY_NOZZLE,
    FONT_NUM,
    FONT_UI,
    FS_BODY,
    FS_H1,
    FS_SMALL,
    FS_TINY,
    GLASS_ALPHA_MID,
    GLASS_BAR_ALPHA,
    GLASS_BLUR_DIV,
    GLASS_EDGE_ALPHA,
    GLASS_HIGHLIGHT_ALPHA,
    GLASS_TINT,
    GRID,
    HUD_DIM,
    HUD_GREEN,
    HUD_MARK,
    HUD_WARN,
    ICON_STROKE,
    INTAKE_COLOR,
    KEY,
    NOZZLE_COLOR,
    NUM_TAU,
    PANEL_EDGE,
    PANEL_FILL,
    RADIUS_CHIP,
    RADIUS_PANEL,
    RIM,
    SCENE_BG,
    SCENE_STRIPE,
    STATIC_SCALE,
    STATION_FILL,
    STRIPE_ALPHA,
    STRIPE_STEP,
    SUB,
    TEAM_COLORS,
    TEAM_TINT,
    TEXT,
    VIGNETTE_ALPHA,
    VIGNETTE_INNER,
)

# 轨迹采样间距：整场都保留，但按距离抽点，避免几千段线拖慢每帧。
TRAIL_SPACING = 60.0
# 航迹飘带半宽（屏幕像素）：按相机深度换算成世界尺度，保证镜头拉近拉远时
# 飘带都有稳定的可见厚度，而不是一条 1–2 px 的细线。
TRAIL_HALF_PX = 5.0
# 横截面明暗：中心最亮、向两侧连续变暗，两侧再分一点受光差异（+y 侧比
# -y 侧略亮），飞机滚转时展向翻转，明暗方向随之交换。按 TRAIL_BANDS 拆窄带
# 插值，条数越多台阶越不明显，取 8 条在 720p～2K 下都看不出色阶。
TRAIL_CENTER_LIT = 1.14
TRAIL_EDGE_LIT = 0.98
TRAIL_EDGE_DARK = 0.82
TRAIL_BANDS = 8
# 航迹飘带的“天光”方向（右上前方）：飘带法向与它越接近越亮，
# 飞机滚转、倒扣时法向随之转动，飘带就出现明暗起伏。
TRAIL_LIGHT = (-0.35, -0.55, 0.76)
# 飘带按“年龄”渐变的时间窗口：最近这么久以内的段保持高亮，更旧的逐渐
# 淡出。窗口越短，越像一条会拖尾的发光航迹，机动动作的先后顺序越清楚。
TRAIL_FADE_TIME = 12.0
# 开火点抽稀：同一个点射不必每一帧都插一个标记。
SHOT_SPACING = 120.0
# 每隔多少个采样点放一个方向箭头。
ARROW_EVERY = 25

# ---- 轨迹上的事件标记（实时画在 3D 航迹线上，不是事后出图） ----
# 展示风格里红色只留给人眼需要立刻反应的告警：发射点用米白星标、
# 高 G 点用浅灰菱形，都不抢红色的语义。
LAUNCH_MARK = (243, 239, 230)
HIGH_G_MARK_COLOR = (185, 190, 197)
# 过载超过人体可持续耐受（6 g）即标注；高空升力受限时（9–12 km、300 m/s
# 量级）可用过载本身只有 4–5 g，因此执行高 G 原语且拉到当地可用过载的
# 75% 也标注，否则航迹上几乎看不到高 G 点（与 main.py 一致）。
HIGH_G_LOAD = 6.0
HIGH_G_REL = 0.75
HIGH_G_COOLDOWN = 1.5

# 导弹的等效 RCS：弹体只有 3–4 m 长、0.2 m 粗，迎头回波约 0.05 m² 量级。
# 机载雷达探测距离按 RCS 的四次方根缩放，因此对弹的发现距离只有对 1 m²
# 目标时的约 0.47 倍（Su-57 约 75 km、F-16V 约 33 km）。
MISSILE_RCS = 0.05

NEAR = 1.0
FOV = math.radians(50.0)
# 屏幕尺寸统一参考长度：用 Su-57 的真实机长做基准，F-16 自动缩到它的 0.75，
# 两机在画面上呈现真实的大小比例关系。
REF_LENGTH = 20.1
# 固定方向光，给机体多边形做明暗，避免纯色剪影。
LIGHT = (-0.35, -0.55, 0.76)
# 补光：右下偏暖的弱光，与主光（左上冷白）一起把“半黑半白”的机体拉出层次。
# 软件路径把它折进同一次点乘，零额外开销。
FILL_LIGHT = (0.42, 0.36, -0.80)
FILL_KEY = 0.20
# 大气透视：逐面颜色按相机距离向场景底色混合，8 km 开始、40 km 全混。
# 机体在远距时自然“退”进背景，近距保持对比，纵深一眼可读（零额外开销）。
HAZE_NEAR_PX = 8000.0
HAZE_FAR_PX = 40000.0
HAZE_MAX = 0.34
# 机体包围盒超采样倍率：软件路径没有 MSAA，模型与背景之间只能靠“画大再缩”
# 得到边缘抗锯齿。1.5× 约等于 2.25 个子采样，成本与观感之间比较平衡。
AIRCRAFT_SS = 1.5
# 机体包围盒外扩（px）：尾焰沿机尾再伸出约 0.26 倍机长，留够余量免得被裁掉。
AIRCRAFT_MARGIN = 42
# 面板外投影：向下偏移 4 px、约 5 层圆角矩形近似高斯，2× 预渲染后降采样即得
# 柔和阴影；阴影与面板同属一份缓存，逐帧只是把贴图范围放大 16 px。
PANEL_SHADOW_PAD = 8
PANEL_SHADOW_DROP = 4
PANEL_SHADOW_ALPHA = 88
PANEL_SHADOW_STEPS = 5
# 面板底衬竖向渐变：顶亮底暗的 10% 亮度差，让大面板不再是一块死色。
PANEL_GRAD_BOT = 0.90
# 读数框上下的渐隐遮罩高度（px）：刻度从框后滚过时不会突兀截断。
VALUE_FADE_H = 6


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
    v_col = in_col * (0.62 + 0.38 * abs(dot(n, normalize(u_light))));
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


def _bake_ao(faces: list, radius: float) -> tuple[list, list]:
    """给每个面烘焙环境光遮蔽（0.35–1.0），返回 (面表, 法线表)。

    面表元素是 (poly, kind, ao)；法线表是与之等长的 (nx, ny, nz)，已在
    机体系下算好，逐帧渲染时直接把光照方向转回机体系点乘，省掉每帧对
    每个面重新做叉积和归一化。

    一次性开销：用点云式半球遮挡代替逐面射线——对每个面把邻近面片按
    “朝向它的程度 × 距离衰减”累加，再按全体均值做相对归一化，因此结果与
    模型尺度无关。机腹、翼根、进气道这些凹处会自然压暗，翼尖与机背保持亮。
    """
    if not faces:
        return faces, []
    n = len(faces)
    cents = np.empty((n, 3), dtype=np.float32)
    nrms = np.empty((n, 3), dtype=np.float32)
    flips: list[bool] = []
    for i, (poly, _kind) in enumerate(faces):
        m = len(poly)
        cx = sum(p[0] for p in poly) / m
        cy = sum(p[1] for p in poly) / m
        cz = sum(p[2] for p in poly) / m
        ex1, ey1, ez1 = (
            poly[1][0] - poly[0][0], poly[1][1] - poly[0][1], poly[1][2] - poly[0][2]
        )
        ex2, ey2, ez2 = (
            poly[2][0] - poly[0][0], poly[2][1] - poly[0][1], poly[2][2] - poly[0][2]
        )
        nx = ey1 * ez2 - ez1 * ey2
        ny = ez1 * ex2 - ex1 * ez2
        nz = ex1 * ey2 - ey1 * ex2
        ln = math.sqrt(nx * nx + ny * ny + nz * nz) or 1.0
        nx, ny, nz = nx / ln, ny / ln, nz / ln
        # OBJ 的绕序不统一：法线统一翻到背离模型中心的一侧，并同步反转顶点
        # 顺序，让所有面的绕序都与外法线一致，渲染时才能做背面剔除。
        flip = nx * cx + ny * cy + nz * cz < 0.0
        if flip:
            nx, ny, nz = -nx, -ny, -nz
        flips.append(flip)
        cents[i] = (cx, cy, cz)
        nrms[i] = (nx, ny, nz)

    occ = np.zeros(n, dtype=np.float32)
    # 分块算，避免 n×n×3 的中间数组在大模型上吃掉几百 MB 内存。
    for lo in range(0, n, 256):
        hi = min(lo + 256, n)
        diff = cents[None, :, :] - cents[lo:hi, None, :]
        dist = np.sqrt((diff ** 2).sum(axis=2)) + 1e-6
        cosw = -(diff * nrms[lo:hi, None, :]).sum(axis=2) / dist
        w = np.clip(cosw, 0.0, None) * np.clip(1.0 - dist / radius, 0.0, None)
        occ[lo:hi] = w.sum(axis=1)
    mean = float(occ.mean()) or 1.0
    ao = np.clip(1.0 / (1.0 + 0.45 * occ / mean), 0.35, 1.0)
    normals = [
        (float(nrms[i, 0]), float(nrms[i, 1]), float(nrms[i, 2]))
        for i in range(n)
    ]
    return [
        (list(reversed(poly)) if flips[i] else poly, kind, float(ao[i]))
        for i, (poly, kind) in enumerate(faces)
    ], normals


def _load_obj_model(
    path: str,
    length: float,
    voxel: float,
    keep_materials: set | None = None,
    canopy_materials: set | None = None,
    gear_y: float | None = None,
    mini_voxel: float | None = None,
) -> dict:
    """读取 OBJ 并做体素聚类简化，返回机体坐标系下的面列表。

    原始 OBJ 坐标：x 朝机头、z 为展向、y 向上，原点不在机身中心。
    这里统一缩放/平移成 (x 前, y 右, z 下)，与内核和渲染的机体系一致。
    """
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
    # 同一个 OBJ 只解析一次，姿态框用的粗模型按更大的体素重新聚类。
    # 返回 (面表, 顶点表)：面表用顶点索引，渲染时可以一次性把整张顶点表
    # 投影到屏幕，而不是每个面的每个角点都重新算一遍矩阵。
    def cluster(vox: float) -> tuple[list, list]:
        buckets = {}
        for i in used:
            x, y, z = verts[i]
            key = (round(x / vox), round(y / vox), round(z / vox))
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

        cells_out = []
        seen = set()
        for idx, material in faces:
            cells = []
            for i in idx:
                x, y, z = verts[i]
                c = remap[(round(x / vox), round(y / vox), round(z / vox))]
                if c not in cells:
                    cells.append(c)
            if len(cells) < 3:
                continue
            sig = tuple(sorted(cells))
            if sig in seen:
                continue
            seen.add(sig)
            kind = "canopy" if material in canopy_set else "body"
            cells_out.append((tuple(cells), kind))
        return cells_out, pts

    def build_mesh(cells_faces: list, pts: list) -> dict:
        """把索引面表烘成两套数据：旧格式面表（GPU/姿态框用）与索引网格。

        索引网格额外带顶点法线和顶点 AO：顶点法线用于大面积的三角化渐变，
        顶点 AO 让环境项在相邻面之间连续过渡，不再出现一格一格的明暗块。
        """
        polys = [
            ([((pts[c][0] - cx) * scale,
               (pts[c][2] - cz) * scale,
               -(pts[c][1] - cy) * scale) for c in cells], kind)
            for cells, kind in cells_faces
        ]
        baked, normals = _bake_ao(polys, radius=max(1.0, length * 0.25))
        nv = len(pts)
        vnx = np.zeros(nv, dtype=np.float64)
        vny = np.zeros(nv, dtype=np.float64)
        vnz = np.zeros(nv, dtype=np.float64)
        vao_n = np.zeros(nv, dtype=np.float64)
        vao_d = np.zeros(nv, dtype=np.float64)
        fitems = []
        for (cells, _kind0), (poly, kind, ao), nrm in zip(
            cells_faces, baked, normals
        ):
            # 面积权重：细长小面对顶点法线的贡献本来就小，否则翼尖会被
            # 一堆退化三角形带偏。
            area = 0.0
            for t in range(1, len(poly) - 1):
                e1 = (poly[t][0] - poly[0][0], poly[t][1] - poly[0][1],
                      poly[t][2] - poly[0][2])
                e2 = (poly[t + 1][0] - poly[0][0], poly[t + 1][1] - poly[0][1],
                      poly[t + 1][2] - poly[0][2])
                area += 0.5 * math.sqrt(
                    (e1[1] * e2[2] - e1[2] * e2[1]) ** 2
                    + (e1[2] * e2[0] - e1[0] * e2[2]) ** 2
                    + (e1[0] * e2[1] - e1[1] * e2[0]) ** 2
                )
            area = max(area, 1e-9)
            nx, ny, nz = nrm
            for c in cells:
                vnx[c] += nx * area
                vny[c] += ny * area
                vnz[c] += nz * area
                vao_n[c] += ao * area
                vao_d[c] += area
            fitems.append((cells, kind, ao, nx, ny, nz))
        for c in range(nv):
            if vao_d[c] > 0.0:
                vnx[c] /= vao_d[c]
                vny[c] /= vao_d[c]
                vnz[c] /= vao_d[c]
                vao_n[c] /= vao_d[c]
            ln = math.sqrt(vnx[c] ** 2 + vny[c] ** 2 + vnz[c] ** 2) or 1.0
            vnx[c] /= ln
            vny[c] /= ln
            vnz[c] /= ln
        # 顶点坐标转到渲染用的机体系（x 前、y 右、z 下）。
        vx = np.array([(p[0] - cx) * scale for p in pts], dtype=np.float32)
        vy = np.array([(p[2] - cz) * scale for p in pts], dtype=np.float32)
        vz = np.array([-(p[1] - cy) * scale for p in pts], dtype=np.float32)
        return {
            "faces": baked, "normals": normals, "fitems": fitems,
            "vx": vx, "vy": vy, "vz": vz,
            "vnx": vnx.astype(np.float32),
            "vny": vny.astype(np.float32),
            "vnz": vnz.astype(np.float32),
            "vao": vao_n.astype(np.float32),
        }

    cells_faces, pts = cluster(voxel)
    mesh = build_mesh(cells_faces, pts)
    # OBJ 的绕序不统一（约 4 成面的绕序朝内），按面心相对模型中心的启发式
    # 翻转并不能可靠地判断“外表面”，做背面剔除会在翼面/机身内侧留下灰斑。
    # 因此仍按双面绘制、远面先画近面覆盖，只把烘焙出的法线留下给着色用。
    model = {
        "length": length, "faces": mesh["faces"], "normals": mesh["normals"],
        "cull": False, "outline": False,
        "fitems": mesh["fitems"], "vx": mesh["vx"], "vy": mesh["vy"],
        "vz": mesh["vz"], "vnx": mesh["vnx"], "vny": mesh["vny"],
        "vnz": mesh["vnz"], "vao": mesh["vao"],
    }
    if mini_voxel is not None:
        # 姿态框只有约 108 px 长，用稍粗的一档聚类；比主模型省面，但仍保留
        # 机身、翼面与尾翼的完整轮廓（远高于旧版约 120 面的粗模）。
        mini_cells, mini_pts = cluster(mini_voxel)
        mini = build_mesh(mini_cells, mini_pts)
        model["mini_faces"] = mini["faces"]
        model["mini_normals"] = mini["normals"]
    return model


# 面数按“主视图最多约 140 px”反推：0.03 m / 0.28 m 一格保留约 2900 / 3700 面，
# 140 px 下单个面接近 1 px，翼身与进气道轮廓比 0.06 / 0.35 m 明显更完整。
# 姿态框另有 0.06 / 0.40 m 的 mini 面表，约 1100 / 1900 面：108 px 下已看不到
# 体素台阶，观感与主视图同级。
# 长度取公开数据 20.1 m；F-16 取 15.06 m，两者屏幕上保持真实尺寸比例。
SU57_MODEL = _load_obj_model(
    SU57_OBJ_PATH, 20.1, 0.03, gear_y=_GEAR_Y, mini_voxel=0.06,
)
# 双发尾喷口位置（机体系，x 向后、y 右、z 下），用于画尾焰。
SU57_MODEL["engines"] = [(-9.4, 0.7, 0.0), (-9.4, -0.7, 0.0)]

F16_MODEL = _load_obj_model(
    F16_OBJ_PATH, 15.06, 0.28,
    keep_materials=F16_KEEP_MATERIALS,
    canopy_materials=F16_CANOPY_MATERIALS,
    gear_y=F16_GEAR_Y,
    mini_voxel=0.40,
)
F16_MODEL["engines"] = [(-7.30, 0.0, 0.0)]

MODEL_BY_NAME = {"SU-57": SU57_MODEL, "F-16V": F16_MODEL}
GENERIC_MODEL = SU57_MODEL

# ---------- 外部贴图资源（scripts/build_*.py 生成） ----------
_ASSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
# 城市地图（CityMapStudio 风格，覆盖 200 km × 200 km，与 main.py 的 arena 对齐）。
TERRAIN_ORTHO_PATH = os.path.join(_ASSET_DIR, "terrain", "city_ortho.png")
TERRAIN_SIZE_M = 200000.0
# 地面透视贴图的降采样步长（像素）：8 像素一格，放大后再做平滑。步长越大
# 逐像素射线越少（1600×1000 下从 4.4 万降到 2.5 万），贴图放大后仍是平滑的。
GROUND_STEP = 8
# 武器页的机体俯视轮廓（由 OBJ 正交投影生成，机头朝上、透明底）。
SILHOUETTE_PATH = {
    "SU-57": os.path.join(_ASSET_DIR, "silhouette", "su57_top.png"),
    "F-16V": os.path.join(_ASSET_DIR, "silhouette", "f16_top.png"),
}

# 机体各部件的配色键 → 相对队伍色的明暗系数；座舱/进气道/喷口用独立颜色。
PART_SHADE = {
    "body": 1.0,
    "wing": 0.86,
    "tail": 0.76,
    "canopy": None,
    "intake": None,
    "nozzle": None,
}

# 武器页中央的本机俯视轮廓：只给右半侧（x 向前、y 向右，单位 = 机长），
# 左半侧在 aircraft_silhouette() 里镜像拼接。y 的最大值 = 翼展 / 机长 / 2，
# 因此轮廓的宽高比与真实机体一致（Su-57 约 14.1/20.1，F-16 约 9.96/15.06）。
SILHOUETTE_RIGHT = {
    "SU-57": (
        (0.50, 0.000), (0.43, 0.014), (0.26, 0.022), (0.08, 0.034),
        (-0.01, 0.081), (-0.06, 0.222), (-0.12, 0.322), (-0.20, 0.351),
        (-0.36, 0.298), (-0.27, 0.137), (-0.30, 0.060), (-0.44, 0.105),
        (-0.50, 0.085),
    ),
    "F-16V": (
        (0.50, 0.000), (0.44, 0.030), (0.28, 0.045), (0.10, 0.070),
        (0.01, 0.240), (-0.06, 0.330), (-0.14, 0.325),
        (-0.20, 0.200), (-0.22, 0.090), (-0.36, 0.200), (-0.47, 0.215),
        (-0.50, 0.130),
    ),
}


def aircraft_silhouette(name: str) -> list[tuple[float, float]]:
    """返回机型的闭合俯视轮廓（机头朝 +x），右半侧镜像出左半侧。"""
    right = SILHOUETTE_RIGHT.get(name, SILHOUETTE_RIGHT["SU-57"])
    return list(right) + [(p[0], -p[1]) for p in reversed(right)]


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


# 白模配色缓存：明暗量按 1/16 分档，颜色算一次复用，逐面只剩查表。
_CLAY_CACHE: dict[tuple, tuple] = {}


def _clay(kind: str, shade: float, team: int) -> tuple:
    """白模材质取色：暗部是 AO 灰，亮部是接近纯白的基色。

    shade = AMB·ao + KEY·|n·l| + RIM·(1-|n·v|)²，已在调用处算好；
    这里把 shade 分档后做“AO 灰 → 基色”的插值，再按队伍色轻染色。
    """
    bucket = int(clamp(shade, 0.0, 1.0) * 16.0) / 16.0
    key = (kind, bucket, team)
    color = _CLAY_CACHE.get(key)
    if color is None:
        if kind == "canopy":
            base = CLAY_CANOPY
        elif kind == "intake":
            base = CLAY_INTAKE
        elif kind == "nozzle":
            base = CLAY_NOZZLE
        else:
            base = _mix(CLAY_LIT, TEAM_COLORS[team], TEAM_TINT)
        color = tuple(
            int(clamp(CLAY_AO[i] + (base[i] - CLAY_AO[i]) * bucket, 0, 255))
            for i in range(3)
        )
        _CLAY_CACHE[key] = color
    return color


class _CachedFont:
    """pygame.freetype 字体包装，保持 Font.render(text, antialias, color) 接口。

    freetype 带 hinting，文字比内置位图字体锐利；按 (文字, 颜色) 缓存字形
    Surface，避免每帧重复栅格化——这也是把 HUD 压进 4 ms 预算的关键。

    tabular=True 时数字按“定宽槽位”渲染：把 0–9 各自居中放进同一宽度（全字形
    平均步进）的槽里，等价于排版里的 tabular figures，读数逐帧变化时整行宽度
    不抖动；Fira Code 本身已是等宽字体，槽宽恰好等于自然步进，因此换字体的
    情况下这条约束依然成立，其余字符仍走自然宽度。
    """

    MAX_CACHE = 4096

    def __init__(self, name: str, size: int, tabular: bool = False) -> None:
        pygame.freetype.init()
        path = name if os.path.isfile(name) else pygame.font.match_font(name)
        self._font = (
            pygame.freetype.Font(path, size) if path
            else pygame.freetype.SysFont(name, size)
        )
        self._font.antialiased = True
        self.tabular = tabular
        self._cell = 0
        # 逐字形合成（定宽数字与加字距标题）都按基线落笔，所有字体统一打开
        # origin：只给 tabular 打开时，font_small 的标题走 _render_spaced 会
        # 把字形画到行盒之外，栅格结果只剩 9 px 高，“SU-57 RDR ON”“WPN”
        # 在成片里直接被削掉上半截。
        self._font.origin = True
        self._ascent = self._font.get_sized_ascender()
        if tabular:
            adv = [self._font.get_metrics(str(d))[0][4] for d in range(10)]
            self._cell = int(round(sum(adv) / len(adv)))
        self._cache: dict[tuple, pygame.Surface] = {}
        # 单字形缓存：定宽数字每帧都在变（速度/高度/航向），字符串缓存命中率低，
        # 真正贵的是逐字符栅格化；按 (字符, 颜色) 缓存栅格结果后，合成就只剩 blit。
        self._glyphs: dict[tuple, tuple] = {}

    def advance(self, ch: str) -> int:
        """单字符的水平步进：数字取定宽槽位，其余字符取字体自然步进。"""
        if self.tabular and ch.isdigit():
            return self._cell
        m = self._font.get_metrics(ch)
        return int(round(m[0][4])) if m and m[0] else 0

    def _glyph(self, ch: str, color) -> tuple:
        """栅格化单个字符，返回 (墨迹面, 墨迹相对行盒左上角的 x/y, 自然步进)。"""
        key = (ch, tuple(color))
        glyph = self._glyphs.get(key)
        if glyph is not None:
            return glyph
        natural = int(round(self._font.get_metrics(ch)[0][4]))
        tmp = pygame.Surface(
            (max(1, natural), self._font.get_sized_height() + 8), pygame.SRCALPHA
        )
        self._font.render_to(tmp, (0, self._ascent), ch, fgcolor=color)
        ink = tmp.get_bounding_rect()
        if ink.width > 0 and ink.height > 0:
            glyph = (tmp.subsurface(ink).copy(), ink.x, ink.y, natural)
        else:
            glyph = (None, 0, 0, natural)
        if len(self._glyphs) > 2048:
            self._glyphs.clear()
        self._glyphs[key] = glyph
        return glyph

    def _render_tabular(self, text: str, color) -> pygame.Surface:
        width = sum(self.advance(ch) for ch in text)
        height = self._font.get_sized_height() + 4
        canvas = pygame.Surface((max(1, width), height), pygame.SRCALPHA)
        pen = 0
        for ch in text:
            step = self.advance(ch)
            if not ch.isspace():
                surf, gx, gy, natural = self._glyph(ch, color)
                if surf is not None:
                    # 槽内居中，窄字形（如 1）不会偏在槽的左侧；所有字形都按
                    # “行盒左上角”对齐，基线自然一致。
                    canvas.blit(surf, (pen + (step - natural) // 2 + gx, gy))
            pen += step
        ink = canvas.get_bounding_rect()
        if ink.width > 0 and ink.height > 0:
            canvas = canvas.subsurface(ink).copy()
        return canvas

    def _render_spaced(self, text: str, color, tracking: int) -> pygame.Surface:
        """逐字加字距渲染（面板标题用 +1 px）：字形走缓存，只多累加步进。"""
        width = sum(self.advance(ch) for ch in text) + tracking * max(0, len(text) - 1)
        height = self._font.get_sized_height() + 4
        canvas = pygame.Surface((max(1, width), height), pygame.SRCALPHA)
        pen = 0
        for ch in text:
            step = self.advance(ch)
            if not ch.isspace():
                surf, gx, gy, natural = self._glyph(ch, color)
                if surf is not None:
                    canvas.blit(surf, (pen + (step - natural) // 2 + gx, gy))
            pen += step + tracking
        ink = canvas.get_bounding_rect()
        if ink.width > 0 and ink.height > 0:
            canvas = canvas.subsurface(ink).copy()
        return canvas

    def render(self, text: str, antialias: bool = True,
               color=(255, 255, 255), tracking: int = 0) -> pygame.Surface:
        key = (text, tuple(color), tracking)
        surf = self._cache.get(key)
        if surf is None:
            if not text:
                surf = pygame.Surface((1, 1), pygame.SRCALPHA)
            elif tracking:
                if len(self._cache) >= self.MAX_CACHE:
                    self._cache.clear()
                surf = self._render_spaced(text, color, tracking)
            elif self.tabular:
                if len(self._cache) >= self.MAX_CACHE:
                    self._cache.clear()
                surf = self._render_tabular(text, color)
            else:
                if len(self._cache) >= self.MAX_CACHE:
                    self._cache.clear()
                surf, _rect = self._font.render(text, fgcolor=color)
                if surf.get_bitsize() < 32:
                    surf = surf.convert_alpha()
            self._cache[key] = surf
        return surf


class Renderer:
    def __init__(
        self,
        size: tuple[int, int] = (1920, 1080),
        follow: int | None = None,
        follow_dist: float = 1800.0,
        gpu: bool = True,
        span_factor: float = 1.25,
        terrain: str = "slab",
    ) -> None:
        pygame.init()
        self.w, self.h = size
        # 3D 场景优先交给 OpenGL；没有可用 GL 上下文（远程会话/软件驱动）时
        # 自动回退到原来的 pygame 软件渲染，HUD 与交互完全一致。
        self.ctx = None
        self.mgl = None
        self.gpu = gpu and self._init_gpu(size)
        if not self.gpu:
            self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        pygame.display.set_caption("Dogfight M2")
        # 字体：统一 Fira Code；读数走定宽数字槽位并按固定宽度格式化，
        # 数字变化时布局不抖动。
        self.font = _CachedFont(FONT_UI, FS_BODY)
        self.font_small = _CachedFont(FONT_UI, FS_SMALL)
        self.font_tiny = _CachedFont(FONT_UI, FS_TINY)
        self.font_big = _CachedFont(FONT_UI, FS_H1)
        self.font_num = _CachedFont(FONT_NUM, FS_BODY, tabular=True)
        self.font_num_small = _CachedFont(FONT_NUM, FS_SMALL, tabular=True)

        # 帧剖析：DOGFIGHT_PROFILE=1 时按阶段计时，默认关闭避免影响帧率。
        self.profile = bool(os.environ.get("DOGFIGHT_PROFILE"))
        self.timings: dict[str, float] = {}
        self.frame_ms = 0.0
        self._samples: dict[str, list[float]] = {}
        # 数值显示插值：速度/高度/距离走一阶平滑，告警与锁定保持即时。
        self._disp: dict[str, float] = {}
        self._last_draw_t: float | None = None
        self._frame_dt = 0.0

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
        # 每个航迹采样点的机体横向轴（机体 y 轴 = 翼展方向），用来把航迹
        # 画成一条与飞机机体 xy 平面一致的宽飘带，而不是屏幕上的细线。
        self.trail_axes: list[list[tuple[float, float, float]]] = [[], []]
        # 每个采样点的时刻，用于让飘带按“年龄”做渐变（最近的段最亮）。
        self.trail_t: list[list[float]] = [[], []]
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
        # 地面风格：slab=暗场 + 极淡网格（默认，已去掉灰色展台），texture=城市地图贴图。
        self.terrain = terrain
        self.overlay = pygame.Surface(size, pygame.SRCALPHA)
        # P0 背景面：底色 + 45° 斜纹 + 径向暗角，按窗口尺寸缓存，每帧一次 blit。
        self._backdrop: pygame.Surface | None = None
        # 机体超采样离屏面：按包围盒增长复用，避免每帧新建 Surface。
        self._model_surf: pygame.Surface | None = None
        # 航迹飘带的半分辨率离屏面：先按 0.5× 画窄带，再双线性放大回 1×。
        self._trail_surf: pygame.Surface | None = None
        # 半透明叠加层：每类图元（机炮包线/导弹/开火特效）各用一层，各自只清
        # 上一次画过的脏矩形、只贴这一块，免掉整屏 fill + 整屏 alpha 混合
        # （1600×1000 上每次约 0.9 ms，一帧原本要做 3–4 次）。
        self._overlay_surfs: dict[str, pygame.Surface] = {}
        self._overlay_dirty: dict[str, pygame.Rect | None] = {}
        # 姿态框的屏幕矩形：GPU 模式下由 GL 在对应区域补画三维模型。
        self._attitude_rects: list[pygame.Rect] = []
        # 地面贴图：图像数组与逐帧投影缓存。
        self._ortho_arr = None
        self._ortho_tried = False
        # 武器页轮廓图（按机型和队伍预先着色、缩放后缓存）。
        self._sil_raw: dict[str, pygame.Surface | None] = {}
        self._sil_cache: dict[tuple, pygame.Surface] = {}
        # 面板底/边框的静态层缓存：键 = 尺寸 + 配色 + 强调条，每帧只 blit。
        self._chrome_cache: dict[tuple, pygame.Surface] = {}
        # HUD 读数框（底衬 + 切角 + 描边）与上下渐隐遮罩，按尺寸缓存。
        self._vbox_cache: dict[tuple, pygame.Surface] = {}
        self._vbox_fade: dict[int, tuple[pygame.Surface, pygame.Surface]] = {}
        # 纯色半透明底衬缓存（HUD 玻璃底、雷达屏底、武器页底）。
        self._alpha_cache: dict[tuple, pygame.Surface] = {}
        # 磨砂玻璃的圆角遮罩与压暗渐变：只与面板尺寸有关，按尺寸缓存。
        self._glass_cache: dict[tuple, pygame.Surface] = {}
        self._cam_ready = False
        # 跟随模式：相机锁定某架飞机（默认关闭，仍取两机中点）。
        self.follow = follow
        self.follow_dist = follow_dist
        # 中点视角的取景倍率：1.25 让两机在 50° 视场内留约 6% 边距；
        # 演示视频可以再调小一点，把两机拍得更近。
        self.span_factor = span_factor

    # ---------- 窗口尺寸与背景 ----------

    def _build_backdrop(self) -> pygame.Surface:
        """P0 背景面：近黑底色 + 45° 极淡斜纹 + 径向暗角。

        只在窗口尺寸变化时重建：斜纹与暗角都画在这张整屏面上，之后每帧
        只需一次 blit，代价与原来的 `screen.fill` 同量级。
        """
        w, h = max(2, self.w), max(2, self.h)
        surf = pygame.Surface((w, h))
        surf.fill(SCENE_BG)
        # 45° 斜纹：一条从底边到顶边的等距线阵，间距 STRIPE_STEP。
        stripe = pygame.Surface((w, h), pygame.SRCALPHA)
        stripe.fill((0, 0, 0, 0))
        color = (SCENE_STRIPE[0], SCENE_STRIPE[1], SCENE_STRIPE[2], STRIPE_ALPHA)
        step = max(8, int(STRIPE_STEP))
        x = -h
        while x < w:
            pygame.draw.line(stripe, color, (x, h), (x + h, 0), 1)
            x += step
        surf.blit(stripe, (0, 0))
        # 径向暗角：先在 64×64 的小图上算好 alpha，再平滑放大到全屏。
        n = 64
        yy, xx = np.mgrid[0:n, 0:n]
        nx = (xx + 0.5) / n * 2.0 - 1.0
        ny = (yy + 0.5) / n * 2.0 - 1.0
        dist = np.sqrt(nx * nx + ny * ny) / math.sqrt(2.0)
        falloff = np.clip((dist - VIGNETTE_INNER) / (1.0 - VIGNETTE_INNER), 0.0, 1.0)
        alpha = (falloff ** 2 * VIGNETTE_ALPHA).astype(np.uint8)
        rgba = np.zeros((n, n, 4), dtype=np.uint8)
        rgba[:, :, 3] = alpha
        vig = pygame.image.frombuffer(rgba.tobytes(), (n, n), "RGBA")
        surf.blit(pygame.transform.smoothscale(vig, (w, h)), (0, 0))
        return surf

    def resize(self, size: tuple[int, int]) -> None:
        """窗口尺寸变化：按新尺寸重建显示面、背景与所有尺寸相关的缓存。"""
        w, h = max(640, int(size[0])), max(480, int(size[1]))
        if (w, h) == (self.w, self.h):
            return
        self.w, self.h = w, h
        if not self.gpu:
            self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
        else:
            self.screen = self.hud = pygame.Surface((w, h), pygame.SRCALPHA)
        self.overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        self.focal = (h / 2.0) / math.tan(FOV / 2.0)
        self._overlay_surfs.clear()
        self._overlay_dirty.clear()
        self._alpha_cache.clear()
        self._chrome_cache.clear()
        self._glass_cache.clear()
        self._sil_cache.clear()
        self._backdrop = self._build_backdrop()

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
            # 取景距离直接按两机间距缩放：120 km 开局时旧的上限 60 km 会把
            # 双方都推出画面，这里改成随间距线性放大（FOV 50° 下 1.25 倍间距
            # 留约 6% 边距），上限 320 km 覆盖 200 km 空域的对角。
            target_dist = clamp(span * self.span_factor, 2200.0, 320000.0)
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

    @staticmethod
    def _clip_line_to_rect(p0, p1, w: int, h: int):
        """Liang-Barsky：把屏幕线段裁到视口内，返回 None 表示整条都在视口外。

        近裁剪面附近的端点投影后会落到几千万像素外（地面网格最远能到
        8.7e7 px），pygame 仍按整条线做光栅化，单条就要几十毫秒。
        先在屏幕空间裁掉视口外的部分再画，方向不变，开销几乎为零。
        """
        x0, y0 = p0
        x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0
        t0, t1 = 0.0, 1.0
        for p, q in (
            (-dx, x0), (dx, w - 1 - x0),
            (-dy, y0), (dy, h - 1 - y0),
        ):
            if p == 0.0:
                if q < 0.0:
                    return None
                continue
            r = q / p
            if p < 0.0:
                if r > t1:
                    return None
                if r > t0:
                    t0 = r
            else:
                if r < t0:
                    return None
                if r < t1:
                    t1 = r
        if t0 >= t1:
            return None
        return (
            (x0 + t0 * dx, y0 + t0 * dy),
            (x0 + t1 * dx, y0 + t1 * dy),
        )

    def _clipped_line(self, surface, color, p0, p1, width: int = 1) -> None:
        """画屏幕线段前先裁到 surface 视口，避免超长线拖慢软件光栅化。"""
        sw, sh = surface.get_size()
        seg = self._clip_line_to_rect(p0, p1, sw, sh)
        if seg is not None:
            pygame.draw.line(surface, color, seg[0], seg[1], width)

    def reset(self) -> None:
        self.trails = [[], []]
        self.trail_axes = [[], []]
        self.trail_t = [[], []]
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
        self._disp.clear()
        self._last_draw_t = None

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
                # 记录当时的机翼方向：飘带宽度沿机体 y 轴展开，
                # 因此飞机滚转/俯仰时飘带会跟着倾斜。
                _, yb, _ = body_axes(*attitude_of(ac))
                self.trail_axes[i].append(yb)
                self.trail_t[i].append(world.t)
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
        self.now = world.t
        # 显示插值的时间步：取两次绘制之间的仿真时间差，暂停时冻结。
        dt = 0.0
        if self._last_draw_t is not None:
            dt = clamp(world.t - self._last_draw_t, 0.0, 0.10)
        self._last_draw_t = world.t
        self._frame_dt = 0.0 if paused else dt
        self._record_trail(world, agents)
        self._update_engines(world)
        prof = self.profile
        t0 = time.perf_counter() if prof else 0.0

        def mark(name: str) -> None:
            nonlocal t0
            dt = (time.perf_counter() - t0) * 1000.0
            prev = self.timings.get(name)
            self.timings[name] = dt if prev is None else prev * 0.9 + dt * 0.1
            samples = self._samples.setdefault(name, [])
            samples.append(dt)
            if len(samples) > 600:
                del samples[0]
            t0 = time.perf_counter()

        if self.gpu:
            # 2D 层先画在透明 HUD 面上，3D 场景由 GPU 渲染后再叠回 HUD。
            self.hud.fill((0, 0, 0, 0))
            self._draw_trails()
            self._draw_gun_envelopes(world)
            self._draw_aircraft(world, agents)
            self._draw_predictions(world)
            self._draw_missiles(world)
            self._draw_counters(world)
            self._draw_gunfire(world)
            self._draw_hud(world, paused, agents)
            if world.result is not None:
                self._draw_result(world)
            self._gpu_scene(world)
            self._gpu_hud()
            self._gpu_attitude(world)
            pygame.display.flip()
            return

        if self._backdrop is None:
            self._backdrop = self._build_backdrop()
        self.screen.blit(self._backdrop, (0, 0))
        if prof:
            mark("clear")
        if self.terrain == "texture":
            self._draw_ground(world)
        if prof:
            mark("ground")
        self._draw_floor(world.arena, world.z_floor)
        self._draw_trails()
        if prof:
            mark("trails")
        self._draw_gun_envelopes(world)
        self._draw_aircraft(world, agents)
        if prof:
            mark("aircraft")
        self._draw_predictions(world)
        self._draw_missiles(world)
        if prof:
            mark("missiles")
        self._draw_counters(world)
        self._draw_gunfire(world)
        if prof:
            mark("effects")
        self._draw_hud(world, paused, agents)
        if prof:
            mark("hud")

        if world.result is not None:
            self._draw_result(world)
        if prof:
            mark("result")
        pygame.display.flip()
        if prof:
            mark("flip")
            self.frame_ms = sum(self.timings.values())

    def _ortho_image(self):
        """加载城市地图正交投影贴图，并预生成几级 mip，供远景采样用。

        远景时一个屏幕像素对应地面几百米，若直接从 2048 原图做最近邻采样，
        山体阴影会退化成满屏噪点；这里按地面采样密度选合适的 mip 层级。
        """
        if self._ortho_tried:
            return self._ortho_arr
        self._ortho_tried = True
        try:
            surf = pygame.image.load(TERRAIN_ORTHO_PATH).convert()
        except Exception:
            return None
        # surfarray 是 (宽, 高, 3)；转成 (高, 宽, 3)，行 0 对应地形图北侧。
        base = pygame.surfarray.array3d(surf).transpose(1, 0, 2).astype(np.uint8)
        mips = [base]
        while mips[-1].shape[0] >= 512:
            cur = mips[-1]
            h, w = cur.shape[:2]
            mips.append(
                cur.reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3)).astype(np.uint8)
            )
        self._ortho_arr = mips
        return self._ortho_arr

    def _ground_image(self, world: World, step: int = GROUND_STEP,
                      opaque: bool = False):
        """把城市地图按当前相机透视投影到地面（z = z_floor）上。

        地面是平面，因此逐像素从屏幕发射线求与地面的交点，再换算成地形图
        UV 即可完成透视贴图。CPU 直接画时用低分辨率算完再放大，避免
        每帧对 160 万像素做 Python 循环。

        返回 (贴图, 地平线行号)。opaque=True 时直接把边缘羽化底色与场景背景
        预先合成成不透明 RGB，CPU 路径可以走快得多的无 alpha blit。
        """
        mips = self._ortho_image()
        if mips is None:
            return None, 0
        gw = max(2, self.w // step)
        gh = max(2, self.h // step)
        xs = (np.arange(gw, dtype=np.float64) + 0.5) * step
        ys = (np.arange(gh, dtype=np.float64) + 0.5) * step
        dxs = (xs - self.w * 0.5) / self.focal
        dys = (self.h * 0.5 - ys) / self.focal
        fx, fy, fz = self.forward
        rx, ry, rz = self.right
        ux, uy, uz = self.up

        dir_x = fx + dxs[None, :] * rx + dys[:, None] * ux
        dir_y = fy + dxs[None, :] * ry + dys[:, None] * uy
        dir_z = fz + dxs[None, :] * rz + dys[:, None] * uz
        depth = np.where(dir_z < -1e-9, np.minimum(dir_z, -1e-9), -1e-9)
        t = (float(world.z_floor) - self.eye[2]) / depth
        valid = (dir_z < -1e-9) & (t > 0.0)

        px = self.eye[0] + t * dir_x
        py = self.eye[1] + t * dir_y
        half = TERRAIN_SIZE_M * 0.5
        u = (px + half) / TERRAIN_SIZE_M
        v = (half - py) / TERRAIN_SIZE_M
        valid &= (u >= 0.0) & (u < 1.0) & (v >= 0.0) & (v < 1.0)

        # 按地面采样密度选 mip：一个屏幕像素对应的地面米数越大，用越粗的一级。
        span = float(np.percentile(t[valid], 70)) if bool(valid.any()) else 0.0
        base_texel = TERRAIN_SIZE_M / mips[0].shape[0]
        target = max(span / self.focal, base_texel)
        level = int(round(math.log2(target / base_texel)))
        level = max(0, min(level, len(mips) - 1))
        arr = mips[level]
        n = arr.shape[0]

        ix = np.clip((u * n).astype(np.int32), 0, n - 1)
        iy = np.clip((v * n).astype(np.int32), 0, n - 1)
        rgb = arr[iy, ix].astype(np.float32)
        # 远处地面混入冷灰雾，透视感更像真实航拍而不是硬贴一张图。
        haze = np.clip(t / 240000.0, 0.0, 0.45)[:, :, None]
        rgb = rgb * (1.0 - haze) + np.array((48.0, 66.0, 82.0)) * haze
        # 场景本身没有光照，补一点增益让地形细节从 HUD 底色里透出来。
        rgb = rgb * 1.12

        # 地形图边缘做一小段羽化，避免 200 km 图幅边界出现硬切边。
        edge = np.minimum(np.minimum(u, 1.0 - u), np.minimum(v, 1.0 - v))
        fade = np.clip(edge / 0.02, 0.0, 1.0) * np.where(valid, 1.0, 0.0)
        rows = np.flatnonzero(fade.any(axis=1))
        top_src = int(rows[0]) if bool(rows.size) else gh
        if opaque:
            # 场景底色是纯 BG，先在低分辨率上做一次合成，之后就能走不透明
            # blit：1600×450 的 RGBA 混合约 7 ms，不透明只剩 1–2 ms。
            a = fade[:, :, None]
            bg = np.array(BG, dtype=np.float32)
            blended = np.clip(rgb, 0, 255) * a + bg * (1.0 - a)
            return (
                pygame.image.frombuffer(
                    blended.astype(np.uint8).tobytes(), (gw, gh), "RGB"
                ),
                top_src,
            )
        rgba = np.zeros((gh, gw, 4), dtype=np.uint8)
        rgba[:, :, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        rgba[:, :, 3] = (fade * 255.0).astype(np.uint8)
        return pygame.image.frombuffer(rgba.tobytes(), (gw, gh), "RGBA"), top_src

    def _draw_ground(self, world: World) -> None:
        """把城市地图地面层铺到 3D 场景最底层（地面网格仍画在它上面）。"""
        surf, top_src = self._ground_image(world, opaque=True)
        if surf is None:
            return
        gw, gh = surf.get_size()
        if top_src >= gh:
            return
        # 地平线以上没有地面：只把有内容的行放大后贴上去，省掉一次全屏混合。
        cropped = surf.subsurface(pygame.Rect(0, top_src, gw, gh - top_src))
        top = top_src * GROUND_STEP
        self.screen.blit(
            pygame.transform.smoothscale(cropped, (self.w, self.h - top)), (0, top)
        )

    def _clip_poly_near(self, points):
        """把世界坐标多边形裁到近裁剪面 z=NEAR，返回相机坐标顶点。

        Sutherland–Hodgman：只按前向深度做一次半平面裁剪，视口外的部分
        仍由 pygame 光栅化裁掉，因此不必再做四边裁剪。
        """
        cam = [self._view(p) for p in points]
        out = []
        n = len(cam)
        for i in range(n):
            a, b = cam[i], cam[(i + 1) % n]
            ain, bin_ = a[2] > NEAR, b[2] > NEAR
            if ain:
                out.append(a)
            if ain != bin_:
                t = (NEAR - a[2]) / (b[2] - a[2])
                out.append(tuple(a[k] + (b[k] - a[k]) * t for k in range(3)))
        return out

    def _draw_floor(self, arena: float, z_floor: float = 0.0) -> None:
        """最低高度层网格 + 空域边界：去掉展台后唯一的地面参照。

        网格压到 `AIR_GRID`（近黑灰），只在俯角足够时才有存在感；贴图模式
        保留原来的 4 km 网格与边界色。
        """
        half = arena / 2.0
        slab = self.terrain != "texture"
        step = 8000.0 if slab else 4000.0
        grid_color = AIR_GRID if slab else GRID
        edge_color = AIR_EDGE if slab else BORDER
        x = -half
        while x <= half:
            seg = self._clip_segment((x, -half, z_floor), (x, half, z_floor))
            if seg:
                self._clipped_line(self.screen, grid_color, seg[0], seg[1], 1)
            x += step
        y = -half
        while y <= half:
            seg = self._clip_segment((-half, y, z_floor), (half, y, z_floor))
            if seg:
                self._clipped_line(self.screen, grid_color, seg[0], seg[1], 1)
            y += step

        corners = [(-half, -half), (half, -half), (half, half), (-half, half)]
        for i in range(4):
            p0 = (*corners[i], z_floor)
            p1 = (*corners[(i + 1) % 4], z_floor)
            seg = self._clip_segment(p0, p1)
            if seg:
                self._clipped_line(self.screen, edge_color, seg[0], seg[1], 1)

    def _draw_trails(self) -> None:
        # 屏幕坐标缓存：飘带层与方向箭头共用同一次投影，同一航迹点不再算两遍。
        self._trail_cam = []
        self._trail_scr = []
        for trail in self.trails:
            cams = [self._view(p) for p in trail]
            scr = [self._screen_of(c) if c[2] > NEAR else None for c in cams]
            self._trail_cam.append(cams)
            self._trail_scr.append(scr)
        # GPU 模式下航迹线由 OpenGL 画，这里只保留箭头与事件标记。
        if not self.gpu:
            self._draw_trail_layer()

        for i, trail in enumerate(self.trails):
            # 飘带走灰白，队伍色只做 25% 染色，暗场里像一条浅色尾迹。
            arrow_color = _mix((236, 238, 242), TEAM_COLORS[i], 0.18)
            # 方向箭头：沿航迹等距放箭标，整局走线在远看时也能读出方向。
            ts = self.trail_t[i]
            scr = self._trail_scr[i]
            now = getattr(self, "now", 0.0)
            for j in range(ARROW_EVERY, len(trail), ARROW_EVERY):
                s0, s1 = scr[j - 1], scr[j]
                if s0 is not None and s1 is not None:
                    age = now - (ts[j] if j < len(ts) else now)
                    fade = max(0.0, 1.0 - age / TRAIL_FADE_TIME)
                    a_col = tuple(
                        int(c * (0.38 + 0.62 * fade)) for c in arrow_color
                    )
                    self._draw_arrow(s0, s1, a_col)

        # 开火点：每个点射位置留一个亮色菱形，和航迹一起构成完整战况记录。
        for marks in self.fire_marks:
            for p in marks:
                s = self.to_screen(p)
                if s is None:
                    continue
                x, y = s
                self._aa_poly(
                    (242, 242, 244),
                    [(x, y - 5), (x + 5, y), (x, y + 5), (x - 5, y)],
                )
                pygame.draw.line(
                    self.screen, (255, 255, 255), (x, y - 3), (x, y + 3), 1
                )

        # 导弹发射点：米白五角星 + 灰边，位置就在发射瞬间的航迹点上。
        for owner, p in self.launch_marks:
            s = self.to_screen(p)
            if s is None:
                continue
            edge = _mix((120, 124, 130), TEAM_COLORS[owner], 0.35)
            self._draw_star(s, 7.5, LAUNCH_MARK, edge)

        # 高 G 机动点：浅灰菱形（比开火点大一号），同样落在 3D 航迹线上。
        for i, p in self.highg_marks:
            s = self.to_screen(p)
            if s is None:
                continue
            x, y = s
            self._aa_poly(
                HIGH_G_MARK_COLOR,
                [(x, y - 7), (x + 7, y), (x, y + 7), (x - 7, y)],
            )
            self._aa_poly(
                _mix((110, 114, 120), TEAM_COLORS[i], 0.3),
                [(x, y - 4), (x + 4, y), (x, y + 4), (x - 4, y)], width=1,
            )

    def _aa_poly(self, color, pts, width: int = 0) -> None:
        """小图元抗锯齿：gfxdraw 只在边界做 AA，填充仍是整块，成本可忽略。

        width=0 表示实心；width>0 表示只描 1 px 抗锯齿边线。
        """
        ip = [(int(round(x)), int(round(y))) for x, y in pts]
        if width <= 0:
            gfxdraw.filled_polygon(self.screen, ip, color)
        gfxdraw.aapolygon(self.screen, ip, color)

    def _aa_circle(self, color, center, radius: int, width: int = 0) -> None:
        cx, cy = int(round(center[0])), int(round(center[1]))
        radius = max(1, int(round(radius)))
        if width <= 0:
            gfxdraw.filled_circle(self.screen, cx, cy, radius, color)
        gfxdraw.aacircle(self.screen, cx, cy, radius, color)

    def _draw_star(self, s, size: float, fill, edge) -> None:
        """五角星：10 个顶点交替取外/内半径，用于导弹发射点。"""
        pts = []
        for k in range(10):
            ang = -math.pi / 2.0 + k * math.pi / 5.0
            r = size if k % 2 == 0 else size * 0.45
            pts.append((s[0] + r * math.cos(ang), s[1] + r * math.sin(ang)))
        self._aa_poly(fill, pts)
        self._aa_poly(edge, pts, width=1)

    def _draw_trail_layer(self) -> None:
        """把所有飘带画进 0.5× 离屏面，再双线性放大回 1×。

        软件渲染没有 MSAA，8 条横向窄带在 1× 下会留下可见色阶，斜向段的
        边缘也有台阶。先在半分辨率面上画（过滤掉高频台阶），再放大回来，
        色阶与锯齿都被平滑掉；飘带的合成本身只占一次 smoothscale 的时间。
        """
        xs0 = ys0 = 1e9
        xs1 = ys1 = -1e9
        pad = 2.0 * TRAIL_HALF_PX + 6.0
        # 近裁剪面附近的点投影后会跑到屏幕外几百万像素，先把它们夹回屏幕
        # 边上再求包围盒，否则离屏面会白白放大到整屏、smoothscale 也变贵。
        xlo, xhi = -pad, self.w + pad
        ylo, yhi = -pad, self.h + pad
        for scr in self._trail_scr:
            for s in scr:
                if s is None:
                    continue
                sx = xlo if s[0] < xlo else (xhi if s[0] > xhi else s[0])
                sy = ylo if s[1] < ylo else (yhi if s[1] > yhi else s[1])
                if sx < xs0:
                    xs0 = sx
                if sx > xs1:
                    xs1 = sx
                if sy < ys0:
                    ys0 = sy
                if sy > ys1:
                    ys1 = sy
        if xs1 <= xs0 or ys1 <= ys0:
            return
        x0 = max(0, int(xs0 - pad))
        y0 = max(0, int(ys0 - pad))
        x1 = min(self.w, int(xs1 + pad) + 1)
        y1 = min(self.h, int(ys1 + pad) + 1)
        scale = 0.5
        bw = max(1, int((x1 - x0) * scale))
        bh = max(1, int((y1 - y0) * scale))
        surf = self._trail_surf
        if surf is None or surf.get_size() != (bw, bh):
            surf = pygame.Surface((bw, bh), pygame.SRCALPHA)
            self._trail_surf = surf
        surf.fill((0, 0, 0, 0), pygame.Rect(0, 0, bw, bh))

        now = getattr(self, "now", 0.0)
        for i, trail in enumerate(self.trails):
            if len(trail) < 2:
                continue
            # 飘带走灰白，队伍色只做 25% 染色，暗场里像一条浅色尾迹。
            base = _mix((198, 202, 208), TEAM_COLORS[i], 0.25)
            axes = self.trail_axes[i]
            ts = self.trail_t[i]
            cams = self._trail_cam[i]
            for j in range(1, len(trail)):
                axis = axes[j] if j < len(axes) else (0.0, 1.0, 0.0)
                age = now - (ts[j] if j < len(ts) else now)
                fade = max(0.0, 1.0 - age / TRAIL_FADE_TIME)
                self._draw_trail_ribbon(
                    trail[j - 1], trail[j], axis, base, fade,
                    surf=surf, ox=x0, oy=y0, scale=scale,
                    cam=(cams[j - 1], cams[j]),
                )

        full = pygame.transform.smoothscale(surf, (x1 - x0, y1 - y0))
        self.screen.blit(full, (x0, y0))

    def _draw_trail_ribbon(
        self, p0, p1, y_axis, base: tuple[int, int, int], fade: float,
        surf: pygame.Surface | None = None, ox: float = 0.0, oy: float = 0.0,
        scale: float = 1.0, cam=None,
    ) -> None:
        """画一段航迹飘带：四边形宽度沿机体 y 轴，平面与机体 xy 平面一致。

        先把两端做近裁剪面裁剪；宽度方向取机体 y 轴在“垂直于航迹”的
        平面上的分量，再按当前相机深度换算成世界尺度，使屏幕厚度稳定在
        约 2×TRAIL_HALF_PX 像素。飘带侧对镜头时自然变窄，保留立体感。

        ``surf`` 为空时直接画到屏幕；否则画到离屏面，屏幕坐标按
        ``(x-ox)*scale`` 换算（``_draw_trail_layer`` 用 0.5× 层做抗锯齿）。

        颜色做两层处理：
        - 渐变：``fade`` 从 0（最旧）到 1（最新），旧段变暗淡出；
        - 光影：飘带法向（机体 z 轴方向）与固定光求点积，飞机滚转/倒扣时
          法向随之转动，飘带亮度起伏，机动姿态因此更醒目。中心再叠一条
          更亮的高光芯，形成“宽边 + 亮芯”的立体观感。
        """
        if cam is None:
            a, b = self._view(p0), self._view(p1)
        else:
            a, b = cam
        if a[2] <= NEAR and b[2] <= NEAR:
            return
        if a[2] <= NEAR:
            t = (NEAR - a[2]) / (b[2] - a[2])
            a = tuple(a[k] + (b[k] - a[k]) * t for k in range(3))
        elif b[2] <= NEAR:
            t = (NEAR - b[2]) / (a[2] - b[2])
            b = tuple(b[k] + (a[k] - b[k]) * t for k in range(3))

        tx, ty, tz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
        tl = math.sqrt(tx * tx + ty * ty + tz * tz)
        if tl < 1e-6:
            return
        tx, ty, tz = tx / tl, ty / tl, tz / tl
        d = y_axis[0] * tx + y_axis[1] * ty + y_axis[2] * tz
        ux, uy, uz = (
            y_axis[0] - tx * d, y_axis[1] - ty * d, y_axis[2] - tz * d,
        )
        ul = math.sqrt(ux * ux + uy * uy + uz * uz)
        if ul < 1e-6:
            return
        ux, uy, uz = ux / ul, uy / ul, uz / ul

        # 飘带法向 = 切线 × 展向（约等于机体 z 轴）。用固定方向的“天光”
        # 做单面着色：法向朝天（倒扣）时最亮，平飞时偏暗。
        nx = ty * uz - tz * uy
        ny = tz * ux - tx * uz
        nz = tx * uy - ty * ux
        lit = nx * TRAIL_LIGHT[0] + ny * TRAIL_LIGHT[1] + nz * TRAIL_LIGHT[2]
        # 单面受光：倒扣（法向朝上）时明显更亮，平飞时偏暗。破 S 先滚转
        # 180° 再俯冲，飘带亮度会先暗后亮，滚转和倒扣一眼就能读出来。
        shade = 0.80 + 0.20 * max(0.0, lit)
        bright = (0.62 + 0.38 * fade) * shade

        # 方向在世界系，先转到相机坐标系再按深度定宽。
        u_cam = (
            ux * self.right[0] + uy * self.right[1] + uz * self.right[2],
            ux * self.up[0] + uy * self.up[1] + uz * self.up[2],
            ux * self.forward[0] + uy * self.forward[1] + uz * self.forward[2],
        )
        depth = 0.5 * (a[2] + b[2])
        half = TRAIL_HALF_PX * depth / self.focal
        ox, oy, oz = (u_cam[0] * half, u_cam[1] * half, u_cam[2] * half)
        ap = (a[0] + ox, a[1] + oy, a[2] + oz)
        am = (a[0] - ox, a[1] - oy, a[2] - oz)
        bp = (b[0] + ox, b[1] + oy, b[2] + oz)
        bm = (b[0] - ox, b[1] - oy, b[2] - oz)
        if min(ap[2], am[2], bp[2], bm[2]) <= NEAR:
            return
        # 横向按亮度渐变：中心最亮、向两侧连续变暗，不再是两块纯色。按
        # TRAIL_BANDS 拆成窄带，每条取该段中点的亮度；带宽只有几像素，
        # 视觉上是连续渐变。飞机滚转时展向跟着翻转，暗边/亮边的位置随之
        # 交换——这正是“带子翻面”的视觉线索。
        s_am, s_ap = self._screen_of(am), self._screen_of(ap)
        s_bm, s_bp = self._screen_of(bm), self._screen_of(bp)
        target = self.screen if surf is None else surf

        def sp(q) -> tuple[float, float]:
            """屏幕坐标 → 离屏面坐标（0.5× 层时坐标减偏移再减半）。"""
            if surf is None:
                return q
            return ((q[0] - ox) * scale, (q[1] - oy) * scale)

        s_am, s_ap, s_bm, s_bp = sp(s_am), sp(s_ap), sp(s_bm), sp(s_bp)

        def edge_pt(p, q, u: float) -> tuple[float, float]:
            """把端部横截面的暗边 → 亮边按 u∈[0,1] 插值。"""
            return (p[0] + (q[0] - p[0]) * u, p[1] + (q[1] - p[1]) * u)

        for k in range(TRAIL_BANDS):
            u0 = k / TRAIL_BANDS
            u1 = (k + 1) / TRAIL_BANDS
            u = 0.5 * (u0 + u1)
            # t = 0 在中心、1 在边缘；两侧边缘各有自己的受光亮度。
            t = abs(2.0 * u - 1.0)
            edge = TRAIL_EDGE_LIT if u > 0.5 else TRAIL_EDGE_DARK
            f = TRAIL_CENTER_LIT + (edge - TRAIL_CENTER_LIT) * t
            col = tuple(min(255, int(c * bright * f)) for c in base)
            pygame.draw.polygon(
                target, col,
                [edge_pt(s_am, s_ap, u0), edge_pt(s_bm, s_bp, u0),
                 edge_pt(s_bm, s_bp, u1), edge_pt(s_am, s_ap, u1)],
            )
        # 高光芯：中心一条细亮线，只比中心窄带再亮一档，整条带子保持
        # “两边缘 → 中心”的连续渐变，而不是两根亮线夹一块暗色。
        s_a, s_b = sp(self._screen_of(a)), sp(self._screen_of(b))
        core = tuple(min(255, int(c * min(1.28, bright * 1.28))) for c in base)
        pygame.draw.line(target, core, s_a, s_b, 2 if surf is None else 1)

    def _draw_arrow(self, s0, s1, color, size: float = 10.0) -> None:
        """在 s1 处画一个沿 s0→s1 的实心箭标（尾部内凹的细长箭头）。"""
        dx, dy = s1[0] - s0[0], s1[1] - s0[1]
        n = math.hypot(dx, dy)
        if n < 8.0:
            return
        ux, uy = dx / n, dy / n
        # 箭尖在航迹点，箭身向后收；两侧翼点略靠前、尾部中点内凹，
        # 屏幕上就是一个细长的实心箭头，和宽飘带一起读作“带方向的航迹”。
        tip = s1
        back = (s1[0] - ux * size * 2.2, s1[1] - uy * size * 2.2)
        left = (s1[0] - ux * size * 1.1 - uy * size * 0.62,
                s1[1] - uy * size * 1.1 + ux * size * 0.62)
        right = (s1[0] - ux * size * 1.1 + uy * size * 0.62,
                 s1[1] - uy * size * 1.1 - ux * size * 0.62)
        self._aa_poly(color, [tip, left, back, right])

    def _draw_gun_envelopes(self, world: World) -> None:
        """机炮包线画成机体所在高度的水平圆环，透视后自然成椭圆。"""
        surf = self._overlay_begin("envelopes")
        box = None
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
                    r = pygame.draw.lines(surf, (*color, alpha), True, ring, 1)
                    box = r if box is None else box.union(r)
        self._overlay_flush("envelopes", surf, box)

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
        # 才能保证远处仍能看清机体轮廓（约 92 px），近距约 138 px；模型放大
        # 后超采样带来的边缘平滑与面片明暗也更容易被看见。
        target_px = 46.0 * clamp(8000.0 / depth, 2.0, 3.0)
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
        model, k, att, depth = frame
        # 机体系 → 相机系 的合成矩阵：机体旋转、屏幕缩放、相机旋转一次性
        # 折进 3×3 矩阵，逐顶点只剩 9 次乘加，不再逐点调用 body_axes/_view。
        xb, yb, zb = body_axes(*att)
        rx, uy, fz = self.right, self.up, self.forward
        eye = self.eye
        pos = (ac.x, ac.y, ac.z)
        m0 = (k * _dot(xb, rx), k * _dot(yb, rx), k * _dot(zb, rx))
        m1 = (k * _dot(xb, uy), k * _dot(yb, uy), k * _dot(zb, uy))
        m2 = (k * _dot(xb, fz), k * _dot(yb, fz), k * _dot(zb, fz))
        t0 = _dot((pos[0] - eye[0], pos[1] - eye[1], pos[2] - eye[2]), rx)
        t1 = _dot((pos[0] - eye[0], pos[1] - eye[1], pos[2] - eye[2]), uy)
        t2 = _dot((pos[0] - eye[0], pos[1] - eye[1], pos[2] - eye[2]), fz)
        # 光照方向与视线方向先转回机体系：面的法线在机体系里是常量，逐面
        # 只做两次点乘，不必每帧重新叉积、归一化。
        lbx, lby, lbz = _dot(xb, LIGHT), _dot(yb, LIGHT), _dot(zb, LIGHT)
        fbx, fby, fbz = (
            _dot(xb, FILL_LIGHT), _dot(yb, FILL_LIGHT), _dot(zb, FILL_LIGHT),
        )
        vbx, vby, vbz = _dot(xb, fz), _dot(yb, fz), _dot(zb, fz)
        w2, h2, focal = self.w / 2.0, self.h / 2.0, self.focal

        # 顶点一次投影：整张顶点表用 numpy 算完，面循环里只查索引。
        # 原实现每个面的每个角点都重算一遍矩阵，顶点被重复投影 3–4 次。
        vx, vy, vz = model["vx"], model["vy"], model["vz"]
        cx_ = m0[0] * vx + m0[1] * vy + m0[2] * vz + t0
        cy_ = m1[0] * vx + m1[1] * vy + m1[2] * vz + t1
        cz_ = m2[0] * vx + m2[1] * vy + m2[2] * vz + t2
        ok = cz_ > NEAR
        if not ok.any():
            return
        inv = np.divide(focal, cz_, out=np.zeros_like(cz_), where=ok)
        sx = w2 + cx_ * inv
        sy = h2 - cy_ * inv

        # 超采样包围盒：软件路径没有 MSAA，把机体画到 1.5× 的离屏面再缩回来，
        # 轮廓与面片交界的硬台阶就会被双线性重采样抹平。外扩余量给尾焰。
        smin_x = float(sx[ok].min()) - AIRCRAFT_MARGIN
        smin_y = float(sy[ok].min()) - AIRCRAFT_MARGIN
        smax_x = float(sx[ok].max()) + AIRCRAFT_MARGIN
        smax_y = float(sy[ok].max()) + AIRCRAFT_MARGIN
        box = pygame.Rect(
            int(math.floor(smin_x)), int(math.floor(smin_y)),
            int(math.ceil(smax_x - smin_x)), int(math.ceil(smax_y - smin_y)),
        ).clip(self.screen.get_rect())
        if box.width < 2 or box.height < 2:
            return
        ss = AIRCRAFT_SS
        bw = int(math.ceil(box.width * ss))
        bh = int(math.ceil(box.height * ss))
        surf = self._model_surf
        if surf is None or surf.get_width() < bw or surf.get_height() < bh:
            surf = pygame.Surface((max(bw, 32), max(bh, 32)), pygame.SRCALPHA)
            self._model_surf = surf
        surf.fill((0, 0, 0, 0), pygame.Rect(0, 0, bw, bh))

        sxl = ((sx - box.x) * ss).tolist()
        syl = ((sy - box.y) * ss).tolist()
        zl = cz_.tolist()
        lit_norm = 1.0 / (1.0 + FILL_KEY)
        # 大气透视：远距机体向场景底色退一点，近景保持满对比。
        haze = clamp(
            (depth - HAZE_NEAR_PX) / max(HAZE_FAR_PX - HAZE_NEAR_PX, 1.0),
            0.0, 1.0,
        ) * HAZE_MAX
        faces = []
        for cells, kind, ao, nx, ny, nz in model["fitems"]:
            poly = []
            zsum = 0.0
            near = False
            for idx in cells:
                zi = zl[idx]
                if zi <= NEAR:
                    near = True
                    break
                poly.append((sxl[idx], syl[idx]))
                zsum += zi
            if near:
                continue
            face_depth = zsum / len(cells)
            # 白模着色：环境项乘烘焙 AO、主光取 |n·l|（双面绘制）、补光把背光
            # 侧抬起一档、边缘光按视线余弦提亮轮廓。双光与大气透视都折进这次
            # 点乘，不新增逐面计算。
            cl = abs(nx * lbx + ny * lby + nz * lbz)
            fl = abs(nx * fbx + ny * fby + nz * fbz)
            facing = abs(nx * vbx + ny * vby + nz * vbz)
            shade = (
                AMB * ao
                + KEY * (cl + FILL_KEY * fl) * lit_norm
                + RIM * (1.0 - facing) ** 2
            )
            color = _clay(kind, shade, team)
            if haze > 0.002:
                color = _mix(color, SCENE_BG, haze)
            faces.append((face_depth, poly, color))

        # 尾焰和机体一起参与深度排序，近面能盖住离镜头更远的焰流。
        flame_scale = ss
        for fd, poly, color in self._flame_faces(ac, model, k, att, team, depth):
            faces.append((
                fd,
                [((x - box.x) * flame_scale, (y - box.y) * flame_scale)
                 for x, y in poly],
                color,
            ))

        # 画家算法：远的先画，近的覆盖上去。
        faces.sort(key=lambda f: -f[0])
        outline = model.get("outline", True)
        edge_w = 1
        for _d, poly, color in faces:
            pygame.draw.polygon(surf, color, poly)
            if not outline:
                continue
            # OBJ 模型简化后单个面只有零点几像素，再描边会糊成一团黑，只给大面描边。
            xs = [p[0] for p in poly]
            ys = [p[1] for p in poly]
            if max(max(xs) - min(xs), max(ys) - min(ys)) >= 6.0 * ss:
                pygame.draw.polygon(surf, CLAY_EDGE, poly, edge_w)

        scaled = pygame.transform.smoothscale(
            surf.subsurface((0, 0, bw, bh)), box.size
        )
        self.screen.blit(scaled, box.topleft)

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
        # 展示风格：尾焰主体是灰白，加力时更亮；焰芯用唯一的那点红。
        outer = _mix((214, 216, 220), (250, 250, 252), 0.35 + 0.45 * ab)
        inner = _mix(ACCENT_RED, (255, 236, 232), 0.25)
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

        surf = self._overlay_begin("missiles")
        box = None
        for m in live:
            pts = self.missile_trails.setdefault(id(m), [])
            p = (m.x, m.y, m.z)
            if not pts or math.dist(pts[-1], p) >= 25.0:
                pts.append(p)
                if len(pts) > 160:
                    del pts[0]

            # 弹体与尾迹都用白模调的浅灰，队伍色只做很轻的染色。
            color = _mix(CLAY_LIT, TEAM_COLORS[m.owner], 0.25)
            dim = (132, 136, 142)
            for a, b in zip(pts, pts[1:]):
                seg = self._clip_segment(a, b)
                if seg is None:
                    continue
                r = pygame.draw.line(surf, (*dim, 150), seg[0], seg[1], 1)
                box = r if box is None else box.union(r)

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
            r = pygame.draw.line(
                surf, (*body_color, 255), tail, s, max(2, int(size * 0.16))
            )
            box = r if box is None else box.union(r)
            r = pygame.draw.circle(
                surf, (250, 250, 252, 255), s, max(1, int(size * 0.10))
            )
            box = box.union(r)

            # 预点火段（RCS 摆姿态）主发动机没点燃，不画尾焰。
            if m.ignited and m.fuel > 0.0:
                flame = size * 0.9
                tip = (tail[0] - sx * flame, tail[1] - sy * flame)
                r = pygame.draw.line(
                    surf, (236, 238, 242, 210), tail, tip, max(2, int(size * 0.2))
                )
                box = box.union(r)
                r = pygame.draw.line(
                    surf, (*ACCENT_RED, 235), tail,
                    (tail[0] - sx * flame * 0.45, tail[1] - sy * flame * 0.45), 1
                )
                box = box.union(r)
        self._overlay_flush("missiles", surf, box)

    def _draw_gunfire(self, world: World) -> None:
        """开火特效：炮口焰 + 沿真实炮线方向的曳光轨迹。"""
        surf = self._overlay_begin("gunfire")
        box = None
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
            r = pygame.draw.line(surf, (*TEAM_COLORS[i], 46), ns, es, 7)
            r = r.union(pygame.draw.line(surf, (255, 154, 54, 170), ns, es, 3))
            r = r.union(pygame.draw.line(surf, (255, 236, 170, 235), ns, es, 1))
            box = r if box is None else box.union(r)

            # 炮口焰：脉动光晕 + 星芒。
            pulse = 0.8 + 0.2 * math.sin(tick * 34.0)
            for radius, alpha in ((12, 36), (8, 80), (5, 160), (3, 235)):
                r = pygame.draw.circle(
                    surf, (255, 214, 128, alpha), ns,
                    max(1, int(radius * pulse)),
                )
                box = box.union(r)
            dx, dy = es[0] - ns[0], es[1] - ns[1]
            norm = math.hypot(dx, dy) or 1.0
            ux, uy = dx / norm, dy / norm
            for span, alpha in ((16, 150), (10, 220)):
                tip = (ns[0] + ux * span, ns[1] + uy * span)
                r = pygame.draw.line(
                    surf, (255, 245, 200, alpha), ns, tip, 2
                )
                box = box.union(r)
                for sign in (1.0, -1.0):
                    side = (
                        ns[0] - uy * span * 0.55 * sign,
                        ns[1] + ux * span * 0.55 * sign,
                    )
                    r = pygame.draw.line(
                        surf, (255, 200, 120, alpha), ns, side, 1
                    )
                    box = box.union(r)
        self._overlay_flush("gunfire", surf, box)

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
                self._clipped_line(self.screen, dim, pos, ground, 1)
                pygame.draw.circle(self.screen, dim, ground, max(2, int(4 * size)), 1)

            # 机体本身：按 psi/theta/phi 实时旋转，Su-57 与 F-16V 用各自模型。
            # GPU 模式下机体由 OpenGL 画，2D 层只补屏幕空间的尾焰。
            if self.gpu:
                self._draw_flames(ac, i, frame)
            else:
                self._draw_model(ac, i, frame)

            # 机动原语标签随飞机移动：从机体引一条短线；靠近屏幕边缘时翻到
            # 机体另一侧，不再做贴边夹取（夹取会让标签脱开飞机停在边框上）。
            # 结算后不再画：标签会透过半透明的结算遮罩，和结果文字叠成
            # “SU-57 shot down 0112m”这种读不通的句子。
            if world.result is not None:
                continue
            primitive = getattr(agents[i], "primitive", "PURE") if agents else "PURE"
            tactic = getattr(agents[i], "tactic", "") if agents else ""
            text = f"{tactic}·{primitive} {ac.z:.0f}m" if tactic else f"{primitive} {ac.z:.0f}m"
            label = self.font_small.render(text, True, color)
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

    def _draw_predictions(self, world: World) -> None:
        """机头前方的轨迹预测点：按当前转弯/爬升率积分，1 s 后的位置。

        连线从机头尖端连到预测点，中间按 0.25 s 采样，所以目标在回旋时
        这条线会跟着弯，而不是像速度切线那样笔直地伸出去。直线飞行时
        预测线自然退化成一条与机头方向重合的短线。
        """
        horizon = 1.0
        samples = 4
        for i, ac in enumerate(world.acs):
            if not ac.alive:
                continue
            nx, ny, nz = nose_direction(ac)
            half = 0.5 * ac.spec.length
            pts = [(ac.x + nx * half, ac.y + ny * half, ac.z + nz * half)]
            for k in range(1, samples + 1):
                pts.append(predict_position(ac, horizon * k / samples, steps=4))
            screen = [self.to_screen(p) for p in pts]
            color = TEAM_COLORS[i]
            dim = _mix(color, BG, 0.35)
            for a, b in zip(screen, screen[1:]):
                if a is None or b is None:
                    continue
                self._clipped_line(self.screen, dim, a, b, 1)
                pygame.draw.circle(self.screen, color, (int(b[0]), int(b[1])), 2)
            end = screen[-1]
            if end is None:
                continue
            # 1 s 后的落点：外圈 + 中心点，和航迹上的事件标记区分开。
            pygame.draw.circle(
                self.screen, color, (int(end[0]), int(end[1])), 7, 1,
            )
            pygame.draw.circle(
                self.screen, color, (int(end[0]), int(end[1])), 2,
            )

    def _draw_flames(self, ac, team: int, frame) -> None:
        """GPU 模式下尾焰仍画在 2D 层：屏幕空间锥形，叠在机体之上。"""
        model, k, att, depth = frame
        for _d, poly, color in self._flame_faces(ac, model, k, att, team, depth):
            pygame.draw.polygon(self.screen, color, poly)

    def _draw_hud(self, world: World, paused: bool, agents=None) -> None:
        a, b = world.acs
        dist = math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)
        d_alt = b.z - a.z
        line = (
            f"T {world.t:5.1f}s    DIST {self._smooth('dist', dist):5.0f}m    "
            f"dALT {self._smooth('dalt', d_alt):+5.0f}m    "
            f"dE {a.energy - b.energy:6.0f} J/kg"
        )
        surf = self.font_num.render(line, True, TEXT)
        self.screen.blit(surf, (self.w / 2 - surf.get_width() / 2, self.h - 34))

        hint = "ESC/Q quit    SPACE pause    R reset"
        hs = self.font_small.render(hint, True, SUB)
        self.screen.blit(hs, (self.w / 2 - hs.get_width() / 2, self.h - 58))

        self._draw_missile_status(world)
        self._draw_attitude_panel(world)
        self._draw_fighter_stations(world, agents)

        if paused:
            ps = self.font.render("PAUSED", True, TEXT)
            self.screen.blit(ps, (self.w / 2 - ps.get_width() / 2, 18))

    def _draw_missile_status(self, world: World) -> None:
        """在飞导弹状态：弹型 → 目标、当前速度，窄条居中，不挤占主视区。

        原来只写目标名和速度，两枚弹追同一目标时会读成同一个目标同时在
        以 311 和 423 m/s 飞行；现在给出弹型与目标，一眼能分清是哪两枚弹。
        """
        live = [m for m in world.missiles if m.alive]
        if not live:
            return
        # 面板宽度按最长一行实测（“AIM-120C → SU-57  586 m/s”约 270 px），
        # 不再写死：换成更长的弹型/目标名时会从右边溢出。
        rows = []
        for m in live:
            target = world.acs[m.target] if 0 <= m.target < len(world.acs) else None
            target_name = target.spec.name if target is not None else "-"
            rows.append((
                TEAM_COLORS[m.owner],
                self.font_num_small.render(
                    f"{m.spec.name} → {target_name}   {m.speed:4.0f} m/s",
                    True, TEXT,
                ),
            ))
        row_h = 16
        width = max(s.get_width() for _c, s in rows) + 34
        height = 10 + row_h * len(rows)
        x = self.w // 2 - width // 2
        # 姿态框底边在 y=238（py=20 + 高 218），这条状态条原来从 236 起笔，
        # 正好压在姿态框的下边框上。留 8 px 间隙放到下面。
        y = 246
        self._panel_bg(
            pygame.Rect(x, y, width, height), PANEL_FILL, edge=PANEL_EDGE
        )

        for k, (color, surf) in enumerate(rows):
            # 队伍色圆点表示是谁发射的。
            pygame.draw.circle(
                self.screen, color, (x + 10, y + 7 + k * row_h + 6), 3
            )
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
        width, height = 440, 218
        px = (self.w - width) // 2
        py = 20
        self._attitude_rects = []
        self._glass_panel(
            pygame.Rect(px, py, width, height), edge=BORDER
        )
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
            # 血量不在这里重复：它在本机 HUD 的顶部状态带里（见 _draw_hud_status）。

            rect = pygame.Rect(hx + 8, py + 22, half - 16, 134)
            self._attitude_rects.append(rect)
            # GPU 模式下机体由 GL 补画，这里只留地面线、Z 轴和尾焰。
            self._draw_attitude_model(
                rect, world, i, ac, color, bodies=not self.gpu
            )

            # 油门滑动条：填充比例即当前功率指令；加力时整条变成白蓝色。
            # 条长留出右侧读数位：读数右对齐到本半幅边缘，加力（“AB”）出现时
            # 也不会挤到相邻半幅的标签上。
            bx, by, bw, bh = hx + 62, py + 174, 64, 9
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
            txt = self.font_num_small.render(
                f"{thr * 100:3.0f}%{' AB' if ab > 0.05 else ''}", True, TEXT
            )
            self.screen.blit(txt, (hx + half - 12 - txt.get_width(), by - 2))

            # 燃油条：加力油耗远高于军推，条长直接给出剩余油量。
            cap = ac.spec.fuel_capacity
            frac = clamp(ac.fuel / cap, 0.0, 1.0) if cap > 0.0 and ac.alive else 1.0
            fy = py + 194
            pygame.draw.rect(self.screen, (44, 48, 55), pygame.Rect(bx, fy, bw, bh))
            pygame.draw.rect(
                self.screen, (118, 196, 148), pygame.Rect(bx, fy, int(bw * frac), bh)
            )
            pygame.draw.rect(self.screen, SUB, pygame.Rect(bx, fy, bw, bh), 1)
            self.screen.blit(self.font_small.render("FUEL", True, SUB), (hx + 10, fy - 2))
            txt = self.font_num_small.render(f"{frac * 100:3.0f}%", True, TEXT)
            self.screen.blit(txt, (hx + half - 12 - txt.get_width(), fy - 2))

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

        # 白模基色只混入少量队伍色，让两队仍可辨认，但不破坏展台白模的统一质感。
        base_color = (
            _mix(CLAY_LIT, TEAM_COLORS[i], TEAM_TINT)
            if ac.alive else (110, 114, 120)
        )
        # 告警色只在受威胁方向上做渐变，统一用 ACCENT_RED。
        red = ACCENT_RED

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
                    self.screen, _mix((214, 216, 220), (250, 250, 252), 0.35 + 0.45 * ab), [
                        (tail[0] - uy * wid, tail[1] + ux * wid),
                        tip2,
                        (tail[0] + uy * wid, tail[1] - ux * wid),
                    ])
                inner = (tail[0] + ux * n * 0.55, tail[1] + uy * n * 0.55)
                pygame.draw.polygon(
                    self.screen, _mix(ACCENT_RED, (255, 236, 232), 0.25), [
                        (tail[0] - uy * wid * 0.4, tail[1] + ux * wid * 0.4),
                        inner,
                        (tail[0] + uy * wid * 0.4, tail[1] - ux * wid * 0.4),
                    ])

        # GPU 模式：机体交给 GL 渲染，2D 层到此为止。
        if not bodies:
            return

        faces = []
        # 机体系 → 姿态框屏幕系：与 _draw_model 同理，把机体旋转和等距投影
        # 折成一个 3×3，逐顶点只做乘加，不再逐点建世界坐标再投影两次。
        def row(axis):
            return (
                axis[0] * xb[0] + axis[1] * xb[1] + axis[2] * xb[2],
                axis[0] * yb[0] + axis[1] * yb[1] + axis[2] * yb[2],
                axis[0] * zb[0] + axis[1] * zb[1] + axis[2] * zb[2],
            )

        rr, uu, vv = row(right), row(up), row(view)
        r0, r1, r2 = rr
        u0, u1, u2 = uu
        v0, v1, v2 = vv
        # 法线在机体系里是常量：把光照/视线方向转回机体系后逐面只做点乘。
        lbx, lby, lbz = _dot(xb, LIGHT), _dot(yb, LIGHT), _dot(zb, LIGHT)
        vbx, vby, vbz = _dot(xb, view), _dot(yb, view), _dot(zb, view)
        mini = model.get("mini_faces", model["faces"])
        mini_normals = model.get("mini_normals", model["normals"])
        for (poly, kind, ao), (nx, ny, nz) in zip(mini, mini_normals):
            pts = []
            dsum = 0.0
            for px, py, pz in poly:
                sr = r0 * px + r1 * py + r2 * pz
                su = u0 * px + u1 * py + u2 * pz
                sd = v0 * px + v1 * py + v2 * pz
                pts.append((cx + sr * scale, cy - su * scale))
                dsum += sd
            depth = dsum / len(poly)

            # 固定方向光照：环境项乘烘焙 AO，主光取 |n·l|，边缘光抬轮廓。
            cl = abs(nx * lbx + ny * lby + nz * lbz)
            facing = abs(nx * vbx + ny * vby + nz * vbz)
            shade = AMB * ao + KEY * cl + RIM * (1.0 - facing) ** 2

            if kind == "canopy":
                base = CLAY_CANOPY
            elif kind == "intake":
                base = CLAY_INTAKE
            elif kind == "nozzle":
                base = CLAY_NOZZLE
            else:
                base = base_color
            # 威胁渐变：面心在机体受威胁方向上的投影决定红色混入比例，
            # 这是全图唯一允许出现红色的场景元素（告警语义）。
            if threat is not None:
                bx_ = sum(p[0] for p in poly) / len(poly)
                by_ = sum(p[1] for p in poly) / len(poly)
                t = clamp(
                    0.5 + 0.5 * (bx_ * fx + by_ * fy) / max(length * 0.5, 1e-6),
                    0.0, 1.0,
                )
                mix = 0.55 * t
                base = (
                    base[0] + (red[0] - base[0]) * mix,
                    base[1] + (red[1] - base[1]) * mix,
                    base[2] + (red[2] - base[2]) * mix,
                )
            color_ = tuple(int(clamp(c, 0, 255)) for c in (
                CLAY_AO[0] + (base[0] - CLAY_AO[0]) * shade,
                CLAY_AO[1] + (base[1] - CLAY_AO[1]) * shade,
                CLAY_AO[2] + (base[2] - CLAY_AO[2]) * shade,
            ))
            faces.append((depth, pts, color_))

        # 画家算法：远的先画，近面覆盖。
        for _depth, pts, color_ in sorted(faces, key=lambda f: f[0]):
            pygame.draw.polygon(self.screen, color_, pts)

    # ---------- 机载座舱显示：雷达屏 + 抬头/头瞄 + 武器显示 ----------
    # 三块仪表贴屏幕上沿独立成组（不再挤在一列里）：宽屏时雷达在角落、
    # HUD 在它内侧并排；窄屏（录制用的 1280×800）改成竖排，雷达占满簇宽，
    # 保证雷达屏始终是能读出距离环的方块，而不是一条细长窗口。
    STATION_MARGIN = 20
    STATION_TOP = 20
    ATT_W = 440                # 顶部中央三维姿态框宽度
    WEAPON_H = 108
    HUD_STATUS_H = 24          # HUD 顶部状态带（左血量、右过载）

    def _stations_visible(self) -> bool:
        """小窗口放不下左右座舱面板时隐藏，避免与主视区、底部读数互相压住。"""
        return self.w >= 1180 and self.h >= 700

    def _station_metrics(self) -> tuple[int, int, int, int, int, int, bool]:
        """三块仪表的尺寸：(簇宽, 雷达宽, 雷达高, HUD 宽, HUD 高, 武器高, 是否竖排)。"""
        margin = self.STATION_MARGIN
        # 中间要给三维姿态框留出宽度和他的两侧间隙：按 (w - ATT_W) // 2 直接
        # 算可用宽会让左右两簇各压住姿态框 12 px，成片里姿态框的圆角边线
        # 正好切在本机 HUD 的外沿上。
        avail = max(320, (self.w - self.ATT_W - 2 * margin - 16) // 2)
        weapon_h = self.WEAPON_H
        status_h = self.HUD_STATUS_H
        if avail >= 520:
            # 并排：簇宽按可用宽度给足（1080p 约 712 px），HUD 占不到六成。
            cluster_w = min(avail, 760)
            hud_w = min(456, int(cluster_w * 0.58))
            radar_w = cluster_w - hud_w - 8
            # HUD 高度按自身宽度定（约 0.60），不再由雷达宽度反推：否则
            # 雷达一窄，HUD 就被压成 188 px 高的横条，俯仰梯只够显示 ±20°，
            # ±10° 的角度标注还会互相压住，速度/高度刻度也跟着挤成一团。
            hud_h = int(clamp(hud_w * 0.60, 168, 280)) + status_h
            radar_h = max(radar_w, hud_h + 8 + weapon_h)
            return cluster_w, radar_w, radar_h, hud_w, hud_h, weapon_h, False
        # 竖排：雷达占满簇宽，压成 190–240 px 高的横屏也比细长条好读。
        radar_w = avail
        radar_h = int(clamp(min(avail * 0.58, self.h * 0.30), 190, 240))
        # HUD 高度同样按宽度给（宽高比约 1.56），1280×800 下也能容下
        # 航向带、俯仰梯与两侧刻度带，不再压成 192 px 高的横条。
        hud_h = int(clamp(avail * 0.58, 200, 300)) + status_h
        return avail, radar_w, radar_h, avail, hud_h, weapon_h, True

    def _station_panels(self, i: int) -> tuple[pygame.Rect, pygame.Rect, pygame.Rect]:
        """第 i 架的三块仪表矩形（红方贴左上、蓝方贴右上，雷达始终靠外）。"""
        cluster_w, radar_w, radar_h, hud_w, hud_h, weapon_h, stacked = (
            self._station_metrics()
        )
        top = self.STATION_TOP
        x0 = (
            self.STATION_MARGIN if i == 0
            else self.w - self.STATION_MARGIN - cluster_w
        )
        if stacked:
            radar = pygame.Rect(x0, top, radar_w, radar_h)
            hud = pygame.Rect(x0, radar.bottom + 6, hud_w, hud_h)
            weapon = pygame.Rect(x0, hud.bottom + 6, hud_w, weapon_h)
            return radar, hud, weapon
        radar_x = x0 if i == 0 else x0 + cluster_w - radar_w
        radar = pygame.Rect(radar_x, top, radar_w, radar_h)
        hud_x = x0 + radar_w + 8 if i == 0 else x0
        hud = pygame.Rect(hud_x, top, hud_w, hud_h)
        weapon = pygame.Rect(hud_x, hud.bottom + 8, hud_w, weapon_h)
        return radar, hud, weapon

    def _draw_fighter_stations(self, world: World, agents=None) -> None:
        """红方座舱贴左上、蓝方贴右上，每块是雷达屏 / 头瞄视野 / 武器显示三块独立面板。

        目标颜色统一按火控状态：绿=已跟踪但未锁定，黄=正在锁定（可发射），
        红=已完成锁定。这样雷达屏与头瞄里的同一个目标颜色始终一致。
        """
        if not self._stations_visible():
            return
        for i in range(len(world.acs)):
            radar, hud, weapon = self._station_panels(i)
            self._draw_station(
                world, i, radar, hud, weapon,
                agents[i] if agents else None,
            )

    def _lock_state(self, ac) -> tuple[tuple[int, int, int], str]:
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

    def _missile_threat(self, world: World, i: int):
        """最近一枚来袭弹的 (相对机头方位, 距离)；没有威胁返回 None。"""
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

    def _alpha_rect(
        self, rect: pygame.Rect, color: tuple[int, int, int], alpha: int
    ) -> None:
        """半透明底：座舱区压在 3D 场景上，不能把后面的空战画面盖死。

        纯色半透明块按 (尺寸, 颜色, 透明度) 缓存成静态面，每帧只 blit，
        避免逐帧新建 Surface + fill（HUD 里这类底衬有 8 块以上）。
        """
        key = (rect.w, rect.h, tuple(color), alpha, RADIUS_CHIP)
        surf = self._alpha_cache.get(key)
        if surf is None:
            surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            pygame.draw.rect(
                surf, (color[0], color[1], color[2], alpha), surf.get_rect(),
                border_radius=RADIUS_CHIP,
            )
            self._alpha_cache[key] = surf
        self.screen.blit(surf, rect.topleft)

    def _overlay_begin(self, key: str) -> pygame.Surface:
        """取用一层半透明叠加面，只清它上一次画过的脏矩形，不整屏重清。"""
        surf = self._overlay_surfs.get(key)
        if surf is None:
            surf = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
            self._overlay_surfs[key] = surf
            self._overlay_dirty[key] = None
        else:
            dirty = self._overlay_dirty[key]
            if dirty is not None:
                surf.fill((0, 0, 0, 0), dirty)
        return surf

    def _overlay_flush(self, key: str, surf: pygame.Surface,
                       box: pygame.Rect | None) -> None:
        """把叠加层贴回屏幕，只贴 box 覆盖的区域（各图元返回矩形的并集）。"""
        if box is None:
            self._overlay_dirty[key] = None
            return
        r = box.inflate(6, 6).clip(self.screen.get_rect())
        if r.width <= 0 or r.height <= 0:
            self._overlay_dirty[key] = None
            return
        self.screen.blit(surf, r.topleft, r)
        self._overlay_dirty[key] = r

    def _panel_bg(
        self, rect: pygame.Rect, fill=None, *, alpha: int | None = None,
        edge=None, radius: int = RADIUS_PANEL,
        bar: tuple[str, tuple, int] | None = None,
    ) -> None:
        """统一的静态面板底：外投影 + 竖向渐变底衬 + 描边 + 内高光 + 强调条。

        fill=None 时只画投影、描边与高光，中心保持透明，供磨砂玻璃面板
        在模糊底图上叠一层“玻璃边框”。其余情况仍是原来的不透明渐变底衬。

        按 STATIC_SCALE 预渲染再 smoothscale：静态件在 1× 下得到真正的
        1 px 发丝线，且同尺寸/同配色的面板直接复用缓存，每帧只做一次 blit。
        外投影与底衬渐变都烘焙在同一份缓存里，逐帧成本为零；阴影让面板
        从场景上“浮”起来，顶亮底暗的底衬则避免大块纯色显得死板。
        """
        key = (rect.w, rect.h, None if fill is None else tuple(fill),
               alpha, edge, radius, bar, STATIC_SCALE)
        surf = self._chrome_cache.get(key)
        if surf is None:
            s = STATIC_SCALE
            m = PANEL_SHADOW_PAD * s
            w, h = rect.w * s, rect.h * s
            big = pygame.Surface((w + 2 * m, h + 2 * m), pygame.SRCALPHA)
            # 1) 外投影：从大到小叠多层半透明圆角矩形，每层都带一点向下偏移，
            # 2× 降采样后边缘自然变柔，比单层硬阴影更像真实投影。
            step_a = max(1, PANEL_SHADOW_ALPHA // PANEL_SHADOW_STEPS)
            for k in range(PANEL_SHADOW_STEPS, 0, -1):
                grow = k * s
                pygame.draw.rect(
                    big, (0, 0, 0, step_a),
                    pygame.Rect(
                        m - grow, m - grow + PANEL_SHADOW_DROP * s,
                        w + 2 * grow, h + 2 * grow,
                    ),
                    border_radius=radius * s + grow,
                )
            # 2) 底衬：竖向渐变按行填充，再用圆角蒙版裁出圆角。
            #    fill=None 时跳过填充，只保留上沿高光给玻璃边框用。
            hi = (255, 255, 255, GLASS_HIGHLIGHT_ALPHA)
            if fill is not None:
                a = alpha if alpha is not None else (
                    fill[3] if len(fill) > 3 else 255
                )
                c_top = tuple(int(v) for v in fill[:3])
                c_bot = tuple(int(v * PANEL_GRAD_BOT) for v in fill[:3])
                grad = pygame.Surface((w, h), pygame.SRCALPHA)
                for row in range(h):
                    t = row / max(1, h - 1)
                    col = tuple(
                        int(c_top[i] + (c_bot[i] - c_top[i]) * t)
                        for i in range(3)
                    )
                    pygame.draw.line(grad, (*col, a), (0, row), (w, row))
                mask = pygame.Surface((w, h), pygame.SRCALPHA)
                pygame.draw.rect(
                    mask, (255, 255, 255, 255), mask.get_rect(),
                    border_radius=radius * s,
                )
                grad.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
                big.blit(grad, (m, m))
                hi = tuple(
                    min(255, c_top[i] + max(6, int(c_top[i] * 0.18)))
                    for i in range(3)
                ) + (a,)
            # 3) 描边：队伍色/中性色 2 px（1× 下即 1 px）。
            if edge is not None:
                pygame.draw.rect(
                    big, edge, pygame.Rect(m, m, w, h), 2,
                    border_radius=radius * s,
                )
            # 4) 内高光：顶边内侧一条比底衬亮 12% 的实色线，模拟受光的上沿；
            # 这里不用半透明白叠画，因为 pygame.draw 是覆写而非混合，会打穿
            # 底衬的 alpha。
            pygame.draw.line(
                big, hi, (m + radius * s, m + 2 * s),
                (m + w - radius * s, m + 2 * s), s,
            )
            if bar is not None:
                side, color, thick = bar
                if side == "left":
                    pygame.draw.rect(big, color, pygame.Rect(m, m, thick * s, h))
                else:
                    pygame.draw.rect(
                        big, color, pygame.Rect(m, m, w, thick * s)
                    )
            surf = pygame.transform.smoothscale(big, (rect.w + 2 * PANEL_SHADOW_PAD,
                                                      rect.h + 2 * PANEL_SHADOW_PAD))
            self._chrome_cache[key] = surf
        self.screen.blit(surf, (rect.x - PANEL_SHADOW_PAD,
                                rect.y - PANEL_SHADOW_PAD))

    def _glass_panel(
        self, rect: pygame.Rect, *, edge, bar=None,
        radius: int = RADIUS_PANEL, alpha: int = GLASS_ALPHA_MID,
    ) -> None:
        """透明磨砂玻璃面板：模糊面板后方的场景，再压暗、圆角、描边。

        做法是取屏幕对应区域 → 缩到 1/GLASS_BLUR_DIV → 放大回原尺寸，
        用降采样-升采样近似高斯模糊（比逐像素卷积便宜一个量级），
        然后叠一层顶亮底暗的暗色渐变，最后用圆角遮罩裁边。
        GPU 模式下 HUD 画在透明面、拿不到场景像素，退化为半透明底衬。
        """
        if self.gpu:
            self._panel_bg(
                rect, GLASS_TINT, alpha=alpha, edge=edge,
                radius=radius, bar=bar,
            )
            return
        clip = rect.clip(self.screen.get_rect())
        if clip.w < 8 or clip.h < 8:
            return
        region = self.screen.subsurface(clip)
        div = GLASS_BLUR_DIV
        small = pygame.transform.smoothscale(
            region, (max(2, clip.w // div), max(2, clip.h // div))
        )
        # 亮部软压缩：面板背后常有导弹尾焰、机体高光经过，直接模糊会在玻璃
        # 上糊出一块灰白污渍（实测 t≈245 s 的 HUD 底部就被擦过面板的机体
        # 盖住）。指数映射保留暗部（空域网格仍然透得出来），同时把接近纯白
        # 的亮部压到 46 左右，亮物只剩一层很淡的雾。在小图上算，成本可忽略。
        arr = pygame.surfarray.pixels3d(small)
        arr[:] = (
            46.0 * (1.0 - np.exp(-arr.astype(np.float32) / 40.0))
        ).astype(np.uint8)
        del arr
        blur = pygame.Surface(clip.size, pygame.SRCALPHA)
        # 先平滑放大到一半尺寸，再用最近邻放大到原尺寸：1/14 小图直接最近邻
        # 放大后每格 14 px，会看出马赛克；平滑一次到半分辨率把格子缩到 2 px，
        # 模糊底本来就低频，再最近邻翻倍已经看不出块状，成本又只有全尺寸
        # 平滑的一半。
        mid = (max(2, clip.w // 2), max(2, clip.h // 2))
        blur.blit(
            pygame.transform.scale(
                pygame.transform.smoothscale(small, mid), clip.size
            ), (0, 0),
        )
        blur.blit(self._glass_overlay(clip.w, clip.h, alpha), (0, 0))
        blur.blit(
            self._glass_mask(clip.w, clip.h, radius), (0, 0),
            special_flags=pygame.BLEND_RGBA_MULT,
        )
        self.screen.blit(blur, clip.topleft)
        # 投影、描边、上沿高光与队伍色强调条仍走静态 chrome。
        self._panel_bg(rect, None, edge=edge, radius=radius, bar=bar)

    def _glass_overlay(self, w: int, h: int, alpha: int) -> pygame.Surface:
        """把 tint、反光与颗粒预合成一张，逐帧只做一次 blit。"""
        key = ("overlay", w, h, alpha)
        surf = self._glass_cache.get(key)
        if surf is None:
            surf = self._glass_tint(w, h, alpha).copy()
            surf.blit(self._glass_sheen(w, h), (0, 0))
            surf.blit(self._glass_noise(w, h), (0, 0))
            self._glass_cache[key] = surf
        return surf

    def _glass_tint(self, w: int, h: int, alpha: int) -> pygame.Surface:
        """玻璃的压暗渐变：顶部略浅、底部略深，模拟玻璃厚度与受光。"""
        key = ("tint", w, h, alpha)
        surf = self._glass_cache.get(key)
        if surf is None:
            surf = pygame.Surface((w, h), pygame.SRCALPHA)
            a0 = max(0, alpha - 22)
            a1 = min(255, alpha + 22)
            for row in range(h):
                t = row / max(1, h - 1)
                a = int(a0 + (a1 - a0) * t)
                pygame.draw.line(
                    surf, (*GLASS_TINT, a), (0, row), (w, row)
                )
            self._glass_cache[key] = surf
        return surf

    def _glass_sheen(self, w: int, h: int) -> pygame.Surface:
        """玻璃反光：顶部一抹由亮到暗的高光带 + 一条斜向反光。"""
        key = ("sheen", w, h)
        surf = self._glass_cache.get(key)
        if surf is None:
            surf = pygame.Surface((w, h), pygame.SRCALPHA)
            band = max(6, int(h * 0.42))
            for row in range(band):
                t = row / max(1, band - 1)
                a = int(34 * (1.0 - t))
                pygame.draw.line(
                    surf, (255, 255, 255, a), (0, row), (w, row)
                )
            pygame.draw.polygon(
                surf, (255, 255, 255, 13),
                [(int(w * 0.62), 0), (int(w * 0.86), 0),
                 (int(w * 0.30), h), (int(w * 0.06), h)],
            )
            self._glass_cache[key] = surf
        return surf

    def _glass_noise(self, w: int, h: int) -> pygame.Surface:
        """磨砂颗粒：低透明度噪点，让大面积玻璃不至于像一块纯色渐变。"""
        key = ("noise", w, h)
        surf = self._glass_cache.get(key)
        if surf is None:
            rng = np.random.default_rng(7)
            gray = rng.integers(128, 192, (h, w, 1), dtype=np.uint8)
            alpha = rng.integers(5, 12, (h, w, 1), dtype=np.uint8)
            rgba = np.concatenate((gray, gray, gray, alpha), axis=2)
            surf = pygame.image.frombuffer(
                np.ascontiguousarray(rgba).tobytes(), (w, h), "RGBA"
            )
            self._glass_cache[key] = surf
        return surf

    def _glass_mask(self, w: int, h: int, radius: int) -> pygame.Surface:
        """圆角遮罩：2× 生成再降采样，1× 下边缘也有抗锯齿。"""
        key = ("mask", w, h, radius)
        surf = self._glass_cache.get(key)
        if surf is None:
            s = STATIC_SCALE
            big = pygame.Surface((w * s, h * s), pygame.SRCALPHA)
            pygame.draw.rect(
                big, (255, 255, 255, 255), big.get_rect(),
                border_radius=radius * s,
            )
            surf = pygame.transform.smoothscale(big, (w, h))
            self._glass_cache[key] = surf
        return surf

    def _smooth(self, key: str, target: float) -> float:
        """一阶插值（τ=NUM_TAU）：速度/高度/距离平滑过渡。

        告警、锁定、余量这类必须即时刷新的数值不要走这里。
        """
        prev = self._disp.get(key)
        if prev is None:
            self._disp[key] = target
            return target
        k = 1.0 - math.exp(-self._frame_dt / NUM_TAU)
        val = prev + (target - prev) * k
        self._disp[key] = val
        return val

    def _draw_station(self, world: World, i: int, radar: pygame.Rect,
                      hud: pygame.Rect, weapon: pygame.Rect,
                      agent=None) -> None:
        """一架飞机的三块独立面板：雷达屏、抬头显示仪、武器栏。

        三块各带自己的圆角底衬与队伍色边线，可以分开摆放而不是塞进一个竖排
        大框：雷达贴屏幕角落所以能做大方块，HUD 与武器栏在它内侧纵排。
        """
        ac = world.acs[i]
        team = TEAM_COLORS[i]
        lock_color, lock_text = self._lock_state(ac)

        # ---- 雷达屏 ----
        # 页头一行交代“谁在看 / 雷达开没开 / 火控到了哪一步”，锁定过程用页头
        # 下方一条细进度条表示，不额外占一行高度。三块面板统一用磨砂玻璃底，
        # 只在边线与顶部强调条上带队伍色，看起来是一套仪表而不是三张卡片。
        glass_edge = (*_mix(team, (255, 255, 255), 0.40), GLASS_EDGE_ALPHA)
        self._glass_panel(
            radar, edge=glass_edge,
            bar=("top", (*team, GLASS_BAR_ALPHA), 3),
        )
        title = self.font_small.render(
            f"{ac.spec.name} {'RDR ON' if ac.radar_on else 'RDR SIL'}",
            True, team, tracking=1,
        )
        self.screen.blit(title, (radar.x + 12, radar.y + 6))
        # 标题下的 1 px 分隔线：把页头与下方的雷达画面明确分层。
        pygame.draw.line(
            self.screen, _mix(PANEL_EDGE, STATION_FILL, 0.4),
            (radar.x + 12, radar.y + 24), (radar.right - 12, radar.y + 24), 1,
        )
        # 火控状态用 FS_TINY：窄屏雷达只有 220 px 宽，18 px 会直接撞上标题。
        st = self.font_tiny.render(lock_text, True, lock_color)
        self.screen.blit(st, (radar.right - 12 - st.get_width(), radar.y + 8))
        if ac.radar_lock_timer > 0.0 and not ac.radar_locked:
            rspec = ac.spec.radar_missile
            frac = (
                clamp(ac.radar_lock_timer / max(rspec.lock_time, 1e-6), 0.0, 1.0)
                if rspec is not None else 0.0
            )
            bx, by, bw, bh = radar.x + 12, radar.y + 27, radar.w - 24, 3
            pygame.draw.rect(self.screen, (44, 48, 55), pygame.Rect(bx, by, bw, bh))
            pygame.draw.rect(
                self.screen, lock_color, pygame.Rect(bx, by, int(bw * frac), bh)
            )
        scope = pygame.Rect(radar.x + 8, radar.y + 34, radar.w - 16, radar.h - 42)
        self._draw_scope(world, i, scope, agent)

        # ---- 抬头显示仪 ----
        # 顶部一条状态带放血量与过载，下面是航向/俯仰/速度/高度。
        self._glass_panel(hud, edge=glass_edge)
        self._draw_hud_status(
            ac, pygame.Rect(hud.x, hud.y, hud.w, self.HUD_STATUS_H)
        )
        self._draw_fwd_hud(
            world, i,
            pygame.Rect(hud.x + 5, hud.y + self.HUD_STATUS_H,
                        hud.w - 10, hud.h - self.HUD_STATUS_H - 6),
        )

        # ---- 武器栏 ----
        self._draw_weapon_rows(world, i, weapon)

    def _draw_hud_status(self, ac, rect: pygame.Rect) -> None:
        """HUD 顶部状态带：左侧血量（数值 + 条），右侧当前过载。

        血量原先只在上方姿态框的机名行里，座舱面板看不到；过载原先贴在速度
        带下方，会压住滚动刻度。这两个“必须随时能瞟一眼”的量合成一条放在
        HUD 最上沿，左右各一个，且每项全界面只出现一次。
        """
        hp = max(0.0, ac.hp) if ac.alive else 0.0
        frac = clamp(hp / max(ac.spec.hp_max, 1.0), 0.0, 1.0)
        col = HUD_WARN if frac < 0.25 else HUD_GREEN
        y = rect.y + 3
        lab = self.font_num_small.render(f"HP {hp:3.0f}", True, col)
        self.screen.blit(lab, (rect.x + 12, y))
        bar = pygame.Rect(rect.x + 12 + lab.get_width() + 8, y + 6, 52, 5)
        pygame.draw.rect(self.screen, (26, 40, 34), bar)
        if frac > 0.0:
            pygame.draw.rect(
                self.screen, col,
                pygame.Rect(bar.x, bar.y, max(1, int(bar.w * frac)), bar.h),
            )
        g = self.font_num_small.render(f"G {ac.n:3.1f}", True, HUD_GREEN)
        self.screen.blit(g, (rect.right - 12 - g.get_width(), y))
        pygame.draw.line(
            self.screen, PANEL_EDGE, (rect.x + 12, rect.bottom),
            (rect.right - 12, rect.bottom), 1,
        )

    def _draw_scope(self, world: World, i: int, rect: pygame.Rect,
                    agent=None) -> None:
        """本机机头朝上的雷达屏：只有拿到距离级接触才画目标光点。"""
        ac = world.acs[i]
        foe = world.acs[1 - i]
        team = TEAM_COLORS[i]
        lock_color, _ = self._lock_state(ac)
        bg = (7, 15, 11)
        grid = (34, 88, 58)
        dim = (86, 170, 118)
        self._alpha_rect(rect, bg, 55)
        pygame.draw.rect(self.screen, grid, rect, 1)

        cx = rect.centerx
        cy = rect.centery
        radius = min(rect.w, rect.h) // 2 - 20

        # 距离档：有稳定跟踪时缩放到当前距离的 1.35 倍；没有接触时用常用档。
        d = math.dist((ac.x, ac.y, ac.z), (foe.x, foe.y, foe.z))
        if ac.contact >= CONTACT_TRACK and d < 1e8:
            rng = clamp(d * 1.35, 4000.0, max(ac.spec.radar_range, 40000.0))
        else:
            rng = max(ac.spec.radar_range, 40000.0) * 0.6

        for frac in (0.25, 0.5, 0.75, 1.0):
            pygame.draw.circle(
                self.screen, grid, (cx, cy), int(radius * frac), 1
            )
        # 距离档标注放在圆环下缘：目标通常出现在机头方向（画面上方）。
        for frac in (0.5, 1.0):
            label = self.font_num_small.render(f"{rng * frac / 1000:.0f}", True, dim)
            self.screen.blit(label, (cx + 3, cy + int(radius * frac) - 17))
        pygame.draw.line(self.screen, grid, (cx - radius, cy), (cx + radius, cy), 1)
        pygame.draw.line(self.screen, grid, (cx, cy - radius), (cx, cy + radius), 1)

        # 本机符号：机头朝上的小三角
        self._aa_poly(team, [(cx, cy - 8), (cx - 5, cy + 6), (cx + 5, cy + 6)])

        rel = wrap_pi(math.atan2(foe.y - ac.y, foe.x - ac.x) - ac.psi)
        if ac.contact >= CONTACT_TRACK and foe.alive:
            rr = clamp(d / rng, 0.0, 0.96) * radius
            tx = cx + rr * math.sin(rel)
            ty = cy - rr * math.cos(rel)
            pygame.draw.line(self.screen, lock_color, (cx, cy), (tx, ty), 1)
            self._aa_circle(lock_color, (tx, ty), 4, width=1)
            self._aa_circle(bg, (tx, ty), 2)
            tag = self.font_num_small.render(f"{d / 1000:4.1f}", True, lock_color)
            dz = foe.z - ac.z
            alt = self.font_num_small.render(f"{dz / 1000:+4.1f}k", True, dim)
            # 标签贴着光点放，但避开边框，避免压住距离环刻度。
            lx = tx + 8
            if lx + tag.get_width() > rect.right - 4:
                lx = tx - 8 - tag.get_width()
            ly = ty - 18
            if ly < rect.y + 4:
                ly = ty + 6
            self.screen.blit(tag, (lx, ly))
            self.screen.blit(alt, (lx, ly + tag.get_height()))
        elif ac.contact == CONTACT_DETECT or ac.rwr:
            az = ac.rwr_az if ac.rwr else rel
            tcol = (255, 96, 88) if ac.rwr else dim
            for sgn in (-1.0, 1.0):
                a = az + sgn * math.radians(4.0)
                pygame.draw.line(
                    self.screen, tcol,
                    (cx + 14 * math.sin(a), cy - 14 * math.cos(a)),
                    (cx + radius * math.sin(a), cy - radius * math.cos(a)), 1,
                )
            tag = "RWR" if ac.rwr else "AZ"
            lab = self.font_small.render(tag, True, tcol)
            ex = cx + (radius - 2) * math.sin(az)
            ey = cy - (radius - 2) * math.cos(az)
            self.screen.blit(lab, (ex - lab.get_width() / 2, ey - 18))
        else:
            lab = self.font_small.render("NO CONTACT", True, dim)
            self.screen.blit(lab, (cx - lab.get_width() / 2, cy + radius - 4))

        # ---- 导弹回波 ----
        # 机载雷达同样能扫到弹体，只是弹体 RCS 小，发现距离按四次方根缩水。
        # 只在雷达开机、弹在机头探测锥内、且距离在探测范围内时显示。
        incoming_seen = False
        if ac.radar_on and ac.spec.radar_range > 0.0:
            nnx, nny, nnz = nose_direction(ac)
            msl_reach = ac.spec.radar_range * rcs_scale(MISSILE_RCS)
            for m in world.missiles:
                if not m.alive:
                    continue
                mx, my, mz = m.x - ac.x, m.y - ac.y, m.z - ac.z
                md = math.sqrt(mx * mx + my * my + mz * mz)
                if md < 1e-6 or md > msl_reach:
                    continue
                if math.acos(clamp(
                    (mx * nnx + my * nny + mz * nnz) / md, -1.0, 1.0
                )) > RADAR_DETECT_FOV:
                    continue
                az = wrap_pi(math.atan2(my, mx) - ac.psi)
                rr = clamp(md / rng, 0.0, 0.96) * radius
                bx = cx + rr * math.sin(az)
                by = cy - rr * math.cos(az)
                # 弹的速度矢量方向（屏幕上机头朝上），用来画指向小三角。
                hdg = wrap_pi(math.atan2(m.vy, m.vx) - ac.psi)
                fx, fy = math.sin(hdg), -math.cos(hdg)
                px, py = math.cos(hdg), math.sin(hdg)
                if m.target == i:
                    incoming_seen = True
                    col = (255, 96, 88)
                    self._aa_poly(
                        col,
                        [(bx + 6 * fx, by + 6 * fy),
                         (bx - 4 * fx + 3.5 * px, by - 4 * fy + 3.5 * py),
                         (bx - 4 * fx - 3.5 * px, by - 4 * fy - 3.5 * py)],
                    )
                    tag = self.font_num_small.render(f"{md / 1000:.1f}", True, col)
                    self.screen.blit(tag, (bx + 8, by - 16))
                else:
                    # 己方在飞弹：浅绿空心点 + 一小段速度矢量（不用队伍色，
                    # 否则红方的己方弹会和来袭弹的红色三角混在一起）。
                    own_msl = (116, 232, 150)
                    pygame.draw.circle(
                        self.screen, own_msl, (int(bx), int(by)), 3, 1
                    )
                    pygame.draw.line(
                        self.screen, own_msl,
                        (bx, by), (bx + 7 * fx, by + 7 * fy), 1,
                    )

        # 来袭导弹：雷达没扫到（关机 / 在探测锥外 / 太远）时只在边缘画红色
        # 测向楔形，不给距离——这时的方位来自 RWR，而不是雷达回波。
        threat = self._missile_threat(world, i)
        if threat is not None and not incoming_seen:
            az = threat[0]
            for sgn in (-1.0, 1.0):
                a = az + sgn * math.radians(3.0)
                pygame.draw.line(
                    self.screen, (255, 80, 72),
                    (cx + (radius - 9) * math.sin(a), cy - (radius - 9) * math.cos(a)),
                    (cx + radius * math.sin(a), cy - radius * math.cos(a)), 2,
                )

        # ---- 底部状态行：左侧态势、右侧来袭弹计数 ----
        # 态势（SEPARATED / NEUTRAL / DEFENSIVE …）原在左右信息卡里，这里收进
        # 雷达屏底边，字号降到 FS_TINY，不参与主视区读数。
        situation = getattr(agent, "situation", "") if agent else ""
        if situation:
            sit = self.font_tiny.render(situation, True, dim)
            self.screen.blit(
                sit, (rect.x + 3, rect.bottom - 3 - sit.get_height())
            )
        incoming = sum(1 for m in world.missiles if m.alive and m.target == i)
        if incoming:
            warn = self.font_small.render(f"IN {incoming}", True, (255, 96, 88))
            self.screen.blit(
                warn, (rect.right - 3 - warn.get_width(),
                       rect.bottom - 3 - warn.get_height())
            )

    def _draw_fwd_hud(self, world: World, i: int, rect: pygame.Rect) -> None:
        """抬头/头瞄视野：航向带、速度/高度刻度带、俯仰梯、目标与炮线。

        版式按真实战机抬头显示：
        - 顶部航向带（相对地球坐标系），当前航向锁在中央方框里，刻度滚动；
        - 左侧速度带、右侧高度带，当前值同样用方框框住并随数值滚动，
          高度带上另有一个三角标志指向当前高度；
        - 中间的俯仰梯代表机头仰角，整条梯子作为刚体随滚转旋转，中央另有
          一个固定不转的水平标志，用它与梯子的夹角读横滚；
        - 目标框为菱形，机炮准星为圆形，圆心落在外弹道预测的落点上。
        """
        ac = world.acs[i]
        foe = world.acs[1 - i]
        lock_color, _ = self._lock_state(ac)
        _psi, theta, phi = attitude_of(ac)
        cx, cy = rect.centerx, rect.centery + 3
        px_per_deg = clamp(rect.h / 105.0, 1.7, 2.6)
        half = rect.w * 0.5
        nom = math.tan(math.radians(26.0))
        # 地平线与整条俯仰梯当作一个刚体，绕 HUD 中心旋转：
        #   d —— 横线自身的方向（随滚转 φ 转过同样的角）
        #   n —— 与横线垂直、指向屏幕下方的方向，各格沿它等距排布
        # 任意点 = 中心 + 横向偏移·d + 俯仰偏移·n。这样相邻横线的端点连线
        # 永远与横线垂直，滚转时整个梯子保持长方形，不会退化成平行四边形。
        d_x, d_y = math.cos(phi), math.sin(phi)
        n_x, n_y = -math.sin(phi), math.cos(phi)
        # 地平线按俯仰姿态（theta）走，而不是航迹角：大迎角时两者会差几度。
        pitch = math.degrees(theta)
        # 上部留给航向带（含方位尖标），底部留给速度/高度单位标注。
        ladder_top = rect.top + 54
        ladder_bottom = rect.bottom - 24
        # 地平线、天空/地面色块与俯仰梯只落在中间的“梯窗”里：两侧的速度/
        # 高度刻度带与上下单位标注不能被地平线斜着切过。大坡度时原来那条
        # 贯穿全幅的地平线正好从 “SPD m/s”“150”“ALT m” 上划过去，
        # 俯仰梯短线也会压到航向带的刻度数字上。
        ladder_left = rect.left + 104
        ladder_right = rect.right - 104

        def hud_pt(az: float, below: float) -> tuple[float, float]:
            """HUD 坐标 → 屏幕像素：az 沿横线方向，below 是相对机头基准向下的度数。"""
            return (
                cx + az * d_x + below * px_per_deg * n_x,
                cy + az * d_y + below * px_per_deg * n_y,
            )

        def view_pt(az_deg: float, el_deg: float) -> tuple[float, float]:
            """方位角/仰角（度）→ 面板坐标，与俯仰梯共用同一套旋转基准。"""
            return hud_pt(
                math.tan(math.radians(az_deg)) / nom * half * 0.82,
                pitch - el_deg,
            )

        self.screen.set_clip(rect)
        try:
            self._alpha_rect(rect, (14, 30, 44), 96)
            ladder_win = pygame.Rect(
                ladder_left, ladder_top,
                max(1, ladder_right - ladder_left),
                max(1, ladder_bottom - ladder_top),
            )
            self.screen.set_clip(ladder_win)
            # 地平线本体：只画在梯窗内，不横切两侧刻度带。
            hx, hy = hud_pt(0.0, pitch)
            big = max(rect.w, rect.h) * 3.0
            hl = (hx - big * d_x, hy - big * d_y)
            hr = (hx + big * d_x, hy + big * d_y)
            pygame.draw.line(self.screen, HUD_GREEN, hl, hr, 1)
            # 原来的天地色块已经去掉：它按地平线分成两种色，裁剪到梯窗后会在
            # 窗口左右/下沿留下直角接缝；不裁又会斜着切过速度高度读数。暗场里
            # 这条地平线本身已经够读，不需要再加一层低透明度色块。
            # 俯仰梯：5° 一短线、10° 一长线并标角度，0° 与地平线重合
            for p in range(-40, 45, 5):
                mx, my = hud_pt(0.0, pitch - p)
                if not ladder_top < my < ladder_bottom:
                    continue
                major = p % 10 == 0
                ln = 22.0 if major else 10.0
                col = (104, 190, 146) if major else HUD_DIM
                pygame.draw.line(
                    self.screen, col,
                    (mx - ln * d_x, my - ln * d_y),
                    (mx + ln * d_x, my + ln * d_y), 1,
                )
                if major and p != 0:
                    # 坡度接近 90° 时，相邻 10° 俯仰格在屏幕上只隔几像素，
                    # 角度标注会叠成一团（80° 坡度下 +20/+30/+40 已经连在一起）；
                    # 这时只保留梯线，不画数字。
                    if abs(n_y) * px_per_deg * 10.0 < 12.0:
                        continue
                    lab = self.font_tiny.render(f"{p:+d}", True, (92, 168, 126))
                    # 角度标注贴在线段左端：右端要留给高度带的五位数字，
                    # 否则 +10/+20 会直接压到 9500/10000 上。字号取 FS_TINY：
                    # 大坡度时相邻 10° 标注的竖向间距会被压到 14 px 左右，
                    # 18 px 字高会互相咬住。
                    lw, lh = lab.get_size()
                    lab_x = int(mx - ln * d_x) - 4 - lw
                    lab_y = int(my - ln * d_y) - lh / 2
                    # 梯窗上下/左右沿都会裁掉半个字：贴边的标注直接不画，
                    # 否则航向带下沿会留一道被切断的 “+40”，大坡度时梯子
                    # 左端还会露出只剩尾数的 “0”（完整标注是 “-10”）。
                    if (lab_y < ladder_top or lab_y + lh > ladder_bottom
                            or lab_x < ladder_left + 2
                            or lab_x + lw > ladder_right - 2):
                        continue
                    # 固定水平标志画在最后且不随滚转，梯子转过一定角度后
                    # 标注会正好落在它那对短线上被横穿；让标注避开中央区域。
                    if pygame.Rect(lab_x, lab_y, lw, lh).colliderect(
                        pygame.Rect(cx - 48, cy - 12, 96, 24)
                    ):
                        continue
                    self.screen.blit(lab, (lab_x, lab_y))

            # 水平标志：固定不随横滚转动，它与俯仰梯的夹角就是当前坡度。
            # 中间留空，避免和机炮准星/目标框挤在一起，只留两端的长线加端标。
            pygame.draw.line(self.screen, HUD_MARK, (cx - 44, cy), (cx - 13, cy), 2)
            pygame.draw.line(self.screen, HUD_MARK, (cx + 13, cy), (cx + 44, cy), 2)
            pygame.draw.line(self.screen, HUD_MARK, (cx - 44, cy - 4), (cx - 44, cy + 4), 2)
            pygame.draw.line(self.screen, HUD_MARK, (cx + 44, cy - 4), (cx + 44, cy + 4), 2)

            # 速度/高度刻度带只允许出现在梯窗的纵向范围内：滚动刻度越过
            # 上下边界时原来会画进顶部航向带，速度数字与 “HDG” 标签叠在
            # 一起，高度带上端的 “13000” 也会挤到航向刻度下方。读数方框
            # 在梯窗内，裁剪不影响它。
            self.screen.set_clip(
                pygame.Rect(rect.left + 2, ladder_top, rect.w - 4,
                            ladder_bottom - ladder_top)
            )
            self._speed_tape(rect.left + 2, ladder_top, ladder_bottom, ac.v, cy)
            self._altitude_tape(rect.right - 2, ladder_top, ladder_bottom, ac.z, cy)
            self.screen.set_clip(rect)
            self._heading_tape(rect, math.degrees(ac.psi) % 360.0)

            # 目标框：视场内按真实方位/仰角投影，菱形框；出视场退化为头瞄指引。
            # 若只拿到方位（CONTACT_DETECT，例如雷达被干扰、仅 RWR 告警），
            # 就在航向带上用一个红色小箭头指向对应刻度——不给距离框，也不给
            # 方位读数文字，避免把“只知道方向”画成“已经跟踪”。
            if foe.alive:
                dx, dy, dz = foe.x - ac.x, foe.y - ac.y, foe.z - ac.z
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                rel = wrap_pi(math.atan2(dy, dx) - ac.psi)
                if ac.contact >= CONTACT_TRACK:
                    elev = math.degrees(math.atan2(dz, math.hypot(dx, dy)))
                    if abs(rel) < math.radians(52.0):
                        tx, ty = view_pt(math.degrees(rel), elev)
                        # 右端让出高度值方框（宽约 63 px、贴着右边缘），
                        # 否则菱形会直接压在读数上。
                        tx = clamp(tx, rect.left + 96, rect.right - 120)
                        ty = clamp(ty, rect.top + 56, rect.bottom - 26)
                        self._diamond(tx, ty, 9, lock_color)
                        lab = self.font_num_small.render(f"{dist / 1000:4.1f}", True, lock_color)
                        # 距离读数贴菱形放：优先放上缘，但航向带（含刻度数字）
                        # 到 top+80 一带不能压，塞不下就改放在菱形下方。
                        lw = lab.get_width()
                        chip = pygame.Rect(int(tx - lw / 2 - 3), int(ty - 26), lw + 6, 16)
                        if chip.y < rect.top + 80:
                            chip.y = int(min(ty + 14, rect.bottom - 18))
                        # 左右贴边时读数会横跨到速度/高度数字列：夹在两列之间，
                        # 右边界留出高度读数框（五位数时约 63 px 宽）。
                        lo = rect.left + 94
                        hi = rect.right - 118 - chip.w
                        if hi >= lo:
                            chip.x = int(clamp(chip.x, lo, hi))
                        chip.clamp_ip(rect)
                        self._alpha_rect(chip, (10, 22, 18), 205)
                        self.screen.blit(lab, (chip.x + 3, chip.y + 2))
                    else:
                        # 出视场时的头瞄指引贴在梯窗两侧，不能落在速度/高度
                        # 刻度带上：原来贴着面板边画，箭头正好压在速度读数
                        # “150”上，HMD 标签也会盖住高度数字。右侧按五位高度
                        # 读数框的实际宽度退让（读数框会从 44 px 往左扩到约
                        # 73 px，“10467”），否则箭头会直接压在最低位上。
                        # 箭头朝外（目标在哪一侧就指哪一侧），不要朝面板中心。
                        side = (
                            rect.right - 138 if rel > 0.0 else ladder_left + 4
                        )
                        tri = (
                            [(side, cy - 7), (side, cy + 7),
                             (side + 10 if rel > 0.0 else side - 10, cy)]
                        )
                        self._aa_poly(lock_color, tri)
                        lab = self.font_small.render("HMD", True, lock_color)
                        # 标签贴着箭头放：原来固定在面板顶部（rect.top+58），
                        # 箭头却在面板中线，成片里 “HMD” 孤零零浮在航向带
                        # 下面，看不出和箭头是一组。
                        self.screen.blit(
                            lab, (side - lab.get_width() if rel > 0.0 else side,
                                  cy + 10)
                        )
                elif ac.contact == CONTACT_DETECT or ac.rwr:
                    brg = (math.degrees(ac.psi) + math.degrees(rel)) % 360.0
                    self._bearing_marker(rect, math.degrees(ac.psi) % 360.0, brg)

            # 机炮准星：圆形框，圆心是弹丸外弹道前推到目标距离后的落点。
            # 圆框压住目标菱形，就说明炮线真的压在目标上。
            fall = self._gun_fall_point(world, i)
            if fall is not None:
                rx, ry, rz = fall[0] - ac.x, fall[1] - ac.y, fall[2] - ac.z
                g_az = math.degrees(wrap_pi(math.atan2(ry, rx) - ac.psi))
                g_el = math.degrees(math.atan2(rz, math.hypot(rx, ry)))
                if abs(g_az) < 52.0:
                    gx_, gy_ = view_pt(g_az, g_el)
                    if (rect.left + 96 < gx_ < rect.right - 96
                            and rect.top + 56 < gy_ < rect.bottom - 26):
                        self._aa_circle(HUD_MARK, (gx_, gy_), 9, width=1)
                        for k in range(4):
                            ang = math.pi * 0.5 * k
                            ox, oy = math.cos(ang), math.sin(ang)
                            pygame.draw.line(
                                self.screen, HUD_MARK,
                                (int(gx_ + ox * 11), int(gy_ + oy * 11)),
                                (int(gx_ + ox * 15), int(gy_ + oy * 15)), 1,
                            )
                        self._aa_circle(HUD_MARK, (gx_, gy_), 2)

            # 底部单位标注
            cap = self.font_small.render("SPD m/s", True, HUD_DIM)
            self.screen.blit(cap, (rect.left + 6, rect.bottom - 19))
            cap = self.font_small.render("ALT m", True, HUD_DIM)
            self.screen.blit(cap, (rect.right - 6 - cap.get_width(), rect.bottom - 19))

            # 来袭导弹：红色标记落在相对本机的方位上，底部给方位读数。
            # 读数压在俯仰梯最下面两格上，所以先垫一条深色横幅，避免和
            # “-30 / -40”角度标注叠字。
            threat = self._missile_threat(world, i)
            if threat is not None:
                az, td = threat
                wx = clamp(cx + math.sin(az) * half * 0.8,
                           rect.left + 96, rect.right - 96)
                wy = clamp(cy - math.cos(az) * rect.h * 0.34,
                           rect.top + 56, rect.bottom - 26)
                pygame.draw.circle(self.screen, HUD_WARN, (int(wx), int(wy)), 5)
                lab = self.font_num_small.render(
                    f"MSL {math.degrees(az):+04.0f}  {td / 1000:4.1f}km",
                    True, HUD_WARN,
                )
                band = pygame.Rect(
                    int(cx - lab.get_width() / 2 - 4), rect.bottom - 43,
                    lab.get_width() + 8, 17,
                )
                self._alpha_rect(band, (10, 22, 18), 230)
                self.screen.blit(lab, (band.x + 4, band.y + 2))
        finally:
            self.screen.set_clip(None)

    # ---------- 抬头显示器的刻度带与读数框 ----------

    def _value_box(self, box: pygame.Rect, text: str, ticks: bool = False) -> None:
        """当前值方框：深底 + 亮框 + 四角切角 + 上下渐隐，刻度带从它后面滚过去。

        读数宽度按实际文字量算：五位高度值比固定 44 px 的框宽，扩展时保持
        右边线与箭头对齐、向左加宽，避免数字压在框线上。底衬按尺寸预渲染
        缓存（切角 / 内侧深影 / 描边 / 侧向竖线都在 2× 下画好再降采样），
        逐帧只剩两次 blit。
        """
        t = self.font_num.render(text.strip(), True, HUD_MARK)
        w = max(box.w, t.get_width() + 8)
        if w > box.w:
            box = pygame.Rect(box.right - w, box.y, w, box.h)
        bg = self._vbox_cache.get(box.size)
        if bg is None:
            s = STATIC_SCALE
            bw, bh = box.w * s, box.h * s
            big = pygame.Surface((bw, bh), pygame.SRCALPHA)
            big.fill((8, 22, 16, 236))
            ch = 2 * s
            # 四角切 2 px 斜角：比纯矩形更像仪表窗。pygame.draw 是覆写像素，
            # 用全透明色画三角形即可真正挖掉 alpha。
            for x, y, dx, dy in (
                (0, 0, 1, 1), (bw - 1, 0, -1, 1),
                (0, bh - 1, 1, -1), (bw - 1, bh - 1, -1, -1),
            ):
                pygame.draw.polygon(
                    big, (0, 0, 0, 0),
                    [(x, y), (x + dx * ch, y), (x, y + dy * ch)],
                )
            # 内侧 1 px 深影：贴在上沿，让框看起来是凹进去的窗口。
            pygame.draw.line(
                big, (4, 14, 10, 236), (ch, s), (bw - ch, s), s
            )
            pygame.draw.line(big, HUD_MARK, (ch, 0), (bw - ch, 0), s)
            pygame.draw.line(big, HUD_MARK, (ch, bh - s), (bw - ch, bh - s), s)
            pygame.draw.line(big, HUD_MARK, (0, ch), (0, bh - ch), s)
            pygame.draw.line(big, HUD_MARK, (bw - s, ch), (bw - s, bh - ch), s)
            if ticks:
                # 中央航向框左右各一条短竖线，强化“读数锁在正中”的读法。
                tx = 5 * s
                top, bot = bh // 2 - 4 * s, bh // 2 + 4 * s
                pygame.draw.line(big, HUD_MARK, (tx, top), (tx, bot), s)
                pygame.draw.line(big, HUD_MARK, (bw - tx, top), (bw - tx, bot), s)
            bg = pygame.transform.smoothscale(big, box.size)
            self._vbox_cache[box.size] = bg

        fades = self._vbox_fade.get(box.w)
        if fades is None:
            down = pygame.Surface((box.w, VALUE_FADE_H), pygame.SRCALPHA)
            up = pygame.Surface((box.w, VALUE_FADE_H), pygame.SRCALPHA)
            for i in range(VALUE_FADE_H):
                a = int(150 * (1.0 - i / max(1, VALUE_FADE_H - 1)))
                pygame.draw.line(down, (8, 22, 16, a), (0, i), (box.w, i))
                pygame.draw.line(
                    up, (8, 22, 16, a), (0, VALUE_FADE_H - 1 - i),
                    (box.w, VALUE_FADE_H - 1 - i),
                )
            fades = (down, up)
            self._vbox_fade[box.w] = fades
        self.screen.blit(fades[1], (box.x, box.y - VALUE_FADE_H))
        self.screen.blit(fades[0], (box.x, box.bottom))
        self.screen.blit(bg, box.topleft)
        self.screen.blit(
            t, (box.centerx - t.get_width() / 2, box.centery - t.get_height() / 2)
        )

    def _bearing_marker(self, rect: pygame.Rect, hdg: float, brg: float) -> None:
        """仅拿到目标方位时，在航向带正上方画一个红色小箭头指向对应刻度。

        方位角按“相对当前航向”换算成横向像素；目标越接近正前方，箭头越靠
        近中央读数框。原来那行 “TGT BRG 123” 文字已去掉，读数直接由箭头
        所指的刻度给出。箭头画在刻度线的上方窄带里，不压中央读数方框。
        """
        span = 200.0
        px_per_deg = rect.w / span
        delta = ((brg - hdg + 180.0) % 360.0) - 180.0
        x = clamp(rect.centerx + delta * px_per_deg, rect.left + 12, rect.right - 12)
        top = rect.top + 3
        self._aa_poly(ACCENT_RED, [(x - 6, top), (x + 6, top), (x, top + 11)])

    def _heading_tape(self, rect: pygame.Rect, hdg: float) -> None:
        """顶部航向带：0° = 世界 +x 轴，顺时针增大（相对地球坐标系）。"""
        y = rect.top + 26
        cx = rect.centerx
        hdg_cap = self.font_small.render("HDG", True, HUD_DIM)
        # 左端要留出 “HDG” 标签的宽度：origin 修复后标签比旧版宽约 4 px，
        # 原来固定 36 px 的门限会让第一个航向数字（如 210）贴着标签起笔，
        # 成片里读成 “HDG210”。按实际字宽算，再多留 6 px 间隔。
        label_left = rect.left + 6 + hdg_cap.get_width() + 6
        # 面板上沿多留一条窄带，给“仅方位”目标的尖标用（见 _bearing_marker）。
        self._alpha_rect(
            pygame.Rect(rect.left + 2, rect.top + 2, rect.w - 4, 42), (8, 22, 30), 120
        )
        pygame.draw.line(self.screen, HUD_DIM, (rect.left + 6, y), (rect.right - 6, y), 1)
        span = 200.0                       # 面板横向覆盖的航向范围（度）
        px_per_deg = rect.w / span
        for t in range(0, 360, 5):
            delta = ((t - hdg + 180.0) % 360.0) - 180.0
            if abs(delta) > span * 0.5 + 4.0:
                continue
            x = cx + delta * px_per_deg
            if x < rect.left + 6 or x > rect.right - 6:
                continue
            if abs(x - cx) < 32.0:         # 中央被读数框压住，不再画刻度
                continue
            major = t % 30 == 0
            ln = 9 if major else 5
            pygame.draw.line(self.screen, HUD_DIM, (x, y), (x, y - ln), 1)
            if major:
                lab = self.font_num_small.render(f"{t:03.0f}", True, (100, 176, 136))
                # 左右两端留出“HDG”标签的位置，避免第一个航向数字压在上面
                if x - lab.get_width() * 0.5 < label_left:
                    continue
                if x + lab.get_width() * 0.5 > rect.right - 10:
                    continue
                # 中央读数框宽 52 px，刻度数字要完整让开它（原先只跳过刻度线，
                # 数字仍会画进框里，出现 “352” 和 “000” 叠字）。
                if abs(x - cx) < 26.0 + lab.get_width() * 0.5 + 4.0:
                    continue
                self.screen.blit(lab, (x - lab.get_width() / 2, y + 2))
        # 航向是三位读数：小于 100° 时必须补零（092、017），否则读数框
        # 只有两位，和旁边的刻度标签（030/060）格式对不上。
        self._value_box(pygame.Rect(cx - 26, y - 11, 52, 22), f"{hdg:03.0f}", ticks=True)
        self.screen.blit(hdg_cap, (rect.left + 6, y + 2))

    def _speed_tape(self, x: int, top: int, bottom: int, v: float, cy: int) -> None:
        """左侧速度带：刻度随速度连续滚动，当前值锁在方框里（滚条式读数）。"""
        px_per_ms = 1.5
        line_x = x + 34
        pygame.draw.line(self.screen, HUD_DIM, (line_x, top), (line_x, bottom), 1)
        lo = v - (cy - top) / px_per_ms
        hi = v - (cy - bottom) / px_per_ms
        k = math.floor(lo / 10.0) * 10.0
        while k <= hi:
            y = cy - (k - v) * px_per_ms
            major = int(k) % 50 == 0
            ln = 7 if major else 4
            pygame.draw.line(self.screen, HUD_DIM, (line_x - ln, y), (line_x, y), 1)
            # 当前值方框附近不标数字：v 正好是 50 的整数倍时（300 m/s），
            # 刻度标签会紧挨着读数框画出同样的数，成片里读成 “300 300”。
            if (major and abs(y - cy) > 24
                    and y - 9 >= top and y + 4 <= bottom):
                lab = self.font_num_small.render(f"{k:3.0f}", True, (100, 176, 136))
                self.screen.blit(lab, (x + 2, y - 9))
            k += 10.0
        box = pygame.Rect(x + 46, cy - 11, 44, 22)
        self._value_box(box, f"{v:3.0f}")
        self._aa_poly(HUD_MARK, [(x + 34, cy), (x + 46, cy - 6), (x + 46, cy + 6)])

    def _altitude_tape(self, right: int, top: int, bottom: int, z: float, cy: int) -> None:
        """右侧高度带：刻度随高度滚动，当前值方框 + 三角标志指向当前高度。"""
        px_per_100m = clamp((bottom - top) / 58.0, 2.6, 4.6)
        # 五位读数宽约 55 px、字高约 20 px：矮 HUD 上 500 m 一档只有 13 px，
        # 相邻数字会叠成一团，这时只标 1000 m 一档（刻度短线仍保留 500 m）。
        lab_step = 500.0 if px_per_100m * 5.0 >= 18.0 else 1000.0
        line_x = right - 34
        pygame.draw.line(self.screen, HUD_DIM, (line_x, top), (line_x, bottom), 1)
        span = (bottom - top) / px_per_100m * 100.0
        lo, hi = z - span * 0.5, z + span * 0.5
        k = math.floor(lo / 100.0) * 100.0
        while k <= hi:
            y = cy - (k - z) / 100.0 * px_per_100m
            major = int(k) % 500 == 0
            ln = 7 if major else 4
            pygame.draw.line(self.screen, HUD_DIM, (line_x, y), (line_x + ln, y), 1)
            if major:
                # 当前值方框压住的那几行不再标数字（方框里已经是精确读数）。
                # 门限要含字形高度：只排除 |y-cy|≤16 时，紧邻方框的读数
                # 仍会露出半个字，看起来像方框里多了一位。
                if (abs(y - cy) > 24 and int(k) % int(lab_step) == 0
                        and y - 9 >= top and y + 4 <= bottom):
                    lab = self.font_num_small.render(f"{k:5.0f}", True, (100, 176, 136))
                    # 标在刻度线左侧：右侧只剩 34 px 到面板边，五位数字摆不下
                    self.screen.blit(lab, (line_x - 9 - lab.get_width(), y - 9))
            k += 100.0
        box = pygame.Rect(right - 90, cy - 11, 44, 22)
        self._value_box(box, f"{z:5.0f}")
        self._aa_poly(
            HUD_MARK,
            [(right - 34, cy), (right - 46, cy - 6), (right - 46, cy + 6)],
        )

    def _diamond(self, x: float, y: float, r: float, color) -> None:
        """雷达/头瞄锁定菱形框（抗锯齿描边，与其余图标统一）。"""
        self._aa_poly(
            color,
            [(x, y - r), (x + r, y), (x, y + r), (x - r, y)], width=ICON_STROKE,
        )

    def _gun_fall_point(
        self, world: World, i: int,
    ) -> tuple[float, float, float] | None:
        """机炮弹丸在当前机头指向下的预计落点（世界坐标）。

        与 sim._fire_guns / _integrate_bullets 走同一套外弹道：出膛速度取
        “载机速度 + 炮口初速沿机头”，之后只受二次阻力与重力；阻力系数按
        出膛高度的大气密度冻结（弹丸在机炮有效射程内只飞 1 s 量级，高度
        变化对密度的影响可以忽略）。前推距离取当前目标斜距，没有目视接触
        时按机炮最大射程。
        """
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
        for _ in range(240):          # 最多前推 2 s，已超出机炮有效射程
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

    def _silhouette_raw(self, name: str):
        """加载 OBJ 正交俯视轮廓图（透明底、机头朝上），失败返回 None。"""
        if name in self._sil_raw:
            return self._sil_raw[name]
        surf = None
        path = SILHOUETTE_PATH.get(name)
        if path:
            try:
                surf = pygame.image.load(path).convert_alpha()
            except Exception:
                surf = None
        self._sil_raw[name] = surf
        return surf

    def _aircraft_top_view(self, ac, i: int, box: pygame.Rect) -> None:
        """武器页中央的本机俯视轮廓（由 OBJ 正交投影图着色，机头朝上）。"""
        raw = self._silhouette_raw(ac.spec.name)
        if raw is not None:
            key = (ac.spec.name, i, box.w, box.h)
            sprite = self._sil_cache.get(key)
            if sprite is None:
                k = min(box.w / raw.get_width(), box.h / raw.get_height()) * 0.96
                size = (max(1, int(raw.get_width() * k)),
                        max(1, int(raw.get_height() * k)))
                sprite = pygame.transform.smoothscale(raw, size)
                # 白色填充按队伍色着色，深色描边保持不变。
                sprite.fill(TEAM_COLORS[i], special_flags=pygame.BLEND_RGBA_MULT)
                self._sil_cache[key] = sprite
            self.screen.blit(
                sprite,
                (box.centerx - sprite.get_width() // 2,
                 box.centery - sprite.get_height() // 2),
            )
            return

        # 贴图缺失时退回手绘轮廓，保证没有资源也能显示。
        pts = aircraft_silhouette(ac.spec.name)
        ys = [p[1] for p in pts]
        span = 2.0 * max(abs(min(ys)), abs(max(ys)))
        scale = min(box.w / 1.12, box.h / (span + 0.12))
        cx, cy = box.centerx, box.centery
        poly = [(cx + p[1] * scale, cy - p[0] * scale) for p in pts]
        fill = tuple(int(c * 0.24) for c in TEAM_COLORS[i])
        pygame.draw.polygon(self.screen, fill, poly)
        pygame.draw.polygon(self.screen, TEAM_COLORS[i], poly, 1)
        # 机身中线 + 座舱盖，给俯视轮廓一个朝向参照
        pygame.draw.line(
            self.screen, TEAM_COLORS[i],
            (cx, int(cy - 0.48 * scale)), (cx, int(cy + 0.48 * scale)), 1,
        )
        pygame.draw.circle(
            self.screen, CANOPY_COLOR,
            (cx, int(cy - 0.12 * scale)), max(2, int(0.045 * scale)), 1,
        )
        # 挂点：先按站位求出机翼在该处的弦长，短线只落在弦内，
        # 避免按固定前后位置画时短线戳出机身轮廓。
        for f in (-0.66, -0.32, 0.32, 0.66):
            sy = f * span * 0.5
            xs: list[float] = []
            for a, b in zip(pts, pts[1:] + pts[:1]):
                if (a[1] - sy) * (b[1] - sy) > 0.0 or a[1] == b[1]:
                    continue
                t = (sy - a[1]) / (b[1] - a[1])
                xs.append(a[0] + t * (b[0] - a[0]))
            if len(xs) < 2:
                continue
            mx = 0.5 * (min(xs) + max(xs))
            sx = cx + sy * scale
            pygame.draw.line(
                self.screen, (198, 226, 210),
                (sx, cy - (mx - 0.045) * scale),
                (sx, cy - (mx + 0.045) * scale), 1,
            )

    def _draw_missile_icon(self, x: float, y: float, color) -> None:
        """4×12 的小弹体剪影（鼻锥 + 弹体 + 尾翼），武器页的弹种 logo。"""
        xi, yi = int(x), int(y)
        pygame.draw.polygon(
            self.screen, color, [(xi + 2, yi), (xi + 4, yi + 4), (xi, yi + 4)]
        )
        pygame.draw.rect(self.screen, color, pygame.Rect(xi, yi + 4, 4, 5))
        pygame.draw.line(self.screen, color, (xi, yi + 9), (xi + 4, yi + 9), 1)
        pygame.draw.line(self.screen, color, (xi, yi + 12), (xi + 2, yi + 9), 1)
        pygame.draw.line(self.screen, color, (xi + 4, yi + 12), (xi + 2, yi + 9), 1)

    def _draw_weapon_rows(self, world: World, i: int, rect: pygame.Rect) -> None:
        """武器显示页（MFD 风格）：中央本机俯视轮廓 + 两侧挂点余量。

        版式参考机载武器页：黑底亮线，中间画本机俯视轮廓（Su-57 / F-16
        各自按真实俯视图的轮廓），左右各列一栏弹药，底部一行机炮与干扰物。
        每栏只占两行（口径标签 + “弹名 余量/总量”），行距随面板高度收敛，
        这样 1280×800 的小面板里也不会溢出到底部行。
        """
        ac = world.acs[i]
        team = TEAM_COLORS[i]
        self._glass_panel(
            rect, edge=(*_mix(team, (255, 255, 255), 0.40), GLASS_EDGE_ALPHA),
            bar=("top", (*team, GLASS_BAR_ALPHA), 2),
        )
        self.screen.blit(
            self.font_small.render("WPN", True, (104, 202, 150), tracking=1),
            (rect.x + 12, rect.y + 4),
        )
        inflight = sum(1 for m in world.missiles if m.alive and m.owner == i)
        threat = self._missile_threat(world, i)
        if threat is not None:
            tag, tcol = f"MSL {threat[1] / 1000:.1f}", HUD_WARN
        elif inflight:
            tag, tcol = f"IN {inflight}", HUD_MARK
        else:
            tag, tcol = "CLEAR", (104, 202, 150)
        t = self.font_small.render(tag, True, tcol)
        self.screen.blit(t, (rect.right - 12 - t.get_width(), rect.y + 4))

        # 格斗弹导引头状态（原来在信息卡的 MSL 行）：持续留在视场内读满时间
        # 才转 LOCK。放在页头中段，和右侧的导弹/CLEAR 状态错开。
        msl = ac.spec.missile
        if msl is not None:
            if ac.locked:
                seek, seek_col = "LOCK", (255, 214, 96)
            elif ac.lock_timer > 0.0:
                pct = 100.0 * ac.lock_timer / max(msl.lock_time, 1e-6)
                seek, seek_col = f"SEEK {pct:3.0f}%", (214, 202, 142)
            else:
                seek, seek_col = "SEEK", (140, 178, 158)
            s = self.font_small.render(seek, True, seek_col)
            self.screen.blit(s, (rect.centerx - s.get_width() / 2, rect.y + 4))

        self._aircraft_top_view(
            ac, i,
            pygame.Rect(rect.x + 86, rect.y + 18, rect.w - 172, rect.h - 38),
        )

        dim = (140, 178, 158)
        empty = (196, 156, 104)
        name_col = (208, 232, 218)

        # 左栏：中距弹；右栏：格斗弹
        groups = (
            ("AAM-R", ac.spec.radar_missile, ac.radar_missiles,
             ac.spec.radar_missile_count, True),
            ("AAM-IR", ac.spec.missile, ac.missiles,
             ac.spec.missile_count, False),
        )
        # 两栏的起始行距要同时避开底部机炮/干扰物行：每栏约占 32 px，
        # 行距随面板高度增加，但绝不能把第二栏的名字行推到底部行上。
        row_gap = clamp(rect.h - 76, 24, 34)
        for k, (title, spec, have, total, left) in enumerate(groups):
            if spec is None:
                continue
            y = rect.y + 22 + k * row_gap
            x = rect.x + 12 if left else rect.right - 12
            head = self.font_small.render(title, True, dim)
            self.screen.blit(head, (x if left else x - head.get_width(), y))
            # 弹体 logo：一排小弹体剪影，已挂载的亮、已打掉的压暗，
            # 让“还剩几枚”在一眼扫过时就能读出来。
            slots = min(int(total), 4)
            icons_w = slots * 6 - 2
            iy = y + 1
            if left:
                ix = x + head.get_width() + 6
            else:
                ix = x - head.get_width() - 6 - icons_w
            for s in range(slots):
                col = team if s < int(have) else (72, 92, 84)
                self._draw_missile_icon(ix + s * 6, iy, col)
            tag = self.font_small.render(f"{spec.name} {have}/{total}", True,
                                         name_col if have else empty)
            self.screen.blit(tag, (x if left else x - tag.get_width(), y + 14))

        by = rect.bottom - 22
        gun = self.font_small.render(f"GUN {int(ac.ammo)}", True, dim)
        self.screen.blit(gun, (rect.x + 8, by))
        cm = self.font_small.render(
            f"CH {ac.chaff}  FL {ac.flares}", True, dim
        )
        self.screen.blit(cm, (rect.right - 8 - cm.get_width(), by))


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
        # 地面城市地图：逐帧按相机投影生成低分辨率图层，再由 GL 线性放大。
        self._ground_tex = None

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
        # 与 CPU 版 _draw_floor 保持一致：贴了城市地图后网格放到 4 km。
        step = 4000.0
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
        col = tuple(c / 255.0 * 0.92 for c in TEAM_COLORS[i])
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
        team_lit = _mix(CLAY_LIT, TEAM_COLORS[team], TEAM_TINT)
        for pts, kind, ao in model["faces"]:
            if len(pts) < 3:
                continue
            if kind == "canopy":
                base = CLAY_CANOPY
            elif kind == "intake":
                base = CLAY_INTAKE
            elif kind == "nozzle":
                base = CLAY_NOZZLE
            else:
                base = team_lit
            base = tuple(
                int(CLAY_AO[c] + (base[c] - CLAY_AO[c]) * ao) for c in range(3)
            )
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

    def _gpu_ground(self, world: World) -> None:
        """把城市地图地面层作为全屏纹理画在 3D 场景最底层。"""
        surf, _top = self._ground_image(world)
        if surf is None:
            return
        ctx, mgl = self.ctx, self.mgl
        size = surf.get_size()
        if self._ground_tex is None or tuple(self._ground_tex.size) != size:
            if self._ground_tex is not None:
                self._ground_tex.release()
            self._ground_tex = ctx.texture(size, 4)
            self._ground_tex.filter = (mgl.LINEAR, mgl.LINEAR)
        self._ground_tex.write(pygame.image.tobytes(surf, "RGBA", True))
        ctx.disable(mgl.DEPTH_TEST)
        ctx.enable(mgl.BLEND)
        ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
        self._ground_tex.use(0)
        self.hud_prog["u_tex"].value = 0
        self.hud_vao.render(mgl.TRIANGLES)

    def _gpu_scene(self, world: World) -> None:
        """3D 场景：地面网格、双方航迹、两架机体。"""
        ctx, mgl = self.ctx, self.mgl
        ctx.viewport = (0, 0, self.w, self.h)
        ctx.clear(BG[0] / 255.0, BG[1] / 255.0, BG[2] / 255.0, 1.0)
        self._gpu_ground(world)
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
