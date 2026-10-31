"""机载多普勒雷达探测能力感知图（俯视极坐标）。

物理口径与 sim.py 的多普勒盲区模型一致：

- 脉冲多普勒雷达靠目标（相对地面杂波）的径向速度 |v_r| 把回波从杂波里
  挑出来；|v_r| < MDV 时回波落进主瓣杂波凹口，距离再近也看不见
  （正侧方横穿即经典的 notch / 三九机动原理）；
- 探测距离还受 RCS（四次方根）、天线阵面增益、下视杂波影响。

图中把目标固定在中心飞机的 +X 方向航向上匀速平飞，于是“不同方位”等价于
“不同交会角”：东西向（迎头/尾追）径向速度最大，南北向（横穿）落进盲区。

输出：results/radar/radar_detection_map.png

用法：
    python scripts/radar_detection_map.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Wedge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import F16V_SPEC, SU57_SPEC  # noqa: E402
from sim import (  # noqa: E402
    LOOKDOWN_MDV_GAIN,
    MDV_RADAR,
    NOTCH_BAND,
    rcs_scale,
)

OUT = ROOT / "results" / "radar"
INK = "#2b2b2b"
GRID = "#e7e7e7"
BLUE = "#1981fe"
ORANGE = "#f28b1a"
SLATE = "#6b7f95"
CMAP = LinearSegmentedColormap.from_list(
    "radar", ["#ffffff", "#e2ebf9", "#a8c4f0", "#5b97f2", "#1981fe"]
)

# 雷达标称探测距离：与 sim.launch_envelope 同一口径（导引头/雷达包线 × RCS 缩放）。
R_NOM = SU57_SPEC.radar_missile.seeker_range * rcs_scale(F16V_SPEC.rcs) / 1000.0
V_BASE = 300.0          # m/s，基准目标速度
V_SLOW = 200.0          # m/s，慢速横穿目标
LOOKDOWN_DZ = 6000.0    # m，下视高差
LOOKDOWN_RANGE_FRAC = 0.72

# 阵面增益：前向 ±60° 接近满增益，侧向阵面次级覆盖，尾部只剩旁瓣。
ANG = np.array([0.0, 60.0, 90.0, 120.0, 150.0, 180.0])
GAIN = np.array([1.0, 0.86, 0.50, 0.43, 0.37, 0.32])


def antenna_gain(theta: np.ndarray) -> np.ndarray:
    """按偏离机头方位角给阵面增益（0..1）。"""
    a = np.degrees(np.abs((theta + np.pi) % (2.0 * np.pi) - np.pi))
    return np.interp(a, ANG, GAIN)


def visibility(theta: np.ndarray, speed: float, mdv: float) -> np.ndarray:
    """目标沿 +X 平飞时的多普勒可见度：|v_r| = speed·|cosθ|。"""
    radial = speed * np.abs(np.cos(theta))
    return np.clip((radial - mdv) / NOTCH_BAND, 0.0, 1.0)


def boundary(theta: np.ndarray, speed: float, mdv: float,
             range_km: float) -> np.ndarray:
    """该方位上仍可探测的最大距离 km（可见度×阵面增益×标称距离）。"""
    return range_km * visibility(theta, speed, mdv) * antenna_gain(theta)


# 俯视机体轮廓（归一化：机头 +1、机尾 -1、半翼展 1），按 Su-57 平面形状近似。
BODY_HALF = [
    (1.00, 0.000), (0.86, 0.030), (0.62, 0.048), (0.38, 0.062),
    (0.16, 0.085), (-0.10, 0.100), (-0.42, 0.112), (-0.72, 0.128),
    (-0.92, 0.150), (-1.00, 0.125),
]
WING_HALF = [
    (0.22, 0.085), (-0.08, 0.400), (-0.60, 0.860), (-0.80, 0.860),
    (-0.92, 0.120), (0.05, 0.098),
]
STAB_HALF = [
    (-0.62, 0.150), (-0.84, 0.500), (-1.00, 0.500), (-1.00, 0.150),
]
NACELLE_HALF = [
    (-0.34, 0.120), (-0.34, 0.300), (-0.97, 0.300), (-1.04, 0.240),
    (-1.04, 0.120),
]


def _mirror(points):
    return [(x, -y) for x, y in points]


def draw_aircraft(ax, scale_km: float) -> None:
    """在原点画俯视机体：白色光晕 + 深蓝填充、浅灰蒙皮的机体。"""
    ax.add_patch(Circle((0.0, 0.0), 1.14 * scale_km, facecolor="#ffffff",
                        edgecolor="#d5e2f5", linewidth=1.2, zorder=7))
    ax.add_patch(Circle((0.0, 0.0), 1.14 * scale_km, facecolor="none",
                        edgecolor="#eaf1fb", linewidth=4.0, zorder=7))
    layers = [
        (NACELLE_HALF, "#4c5a6d", "#26313f", 0.55),
        (STAB_HALF, "#6a7c93", "#26313f", 0.20),
        (WING_HALF, "#7f93ad", "#26313f", 0.15),
        (BODY_HALF, "#aebfce", "#1f2b39", 0.0),
    ]
    for half, face, edge, z in layers:
        poly = [(x * scale_km, y * scale_km) for x, y in half]
        poly += [(x * scale_km, y * scale_km) for x, y in _mirror(half)][::-1]
        ax.add_patch(Polygon(poly, closed=True, facecolor=face,
                             edgecolor=edge, linewidth=0.7, zorder=8 + z))
    # 座舱盖
    ax.add_patch(Polygon(
        [(-0.10 * scale_km, -0.045 * scale_km), (0.42 * scale_km, -0.020 * scale_km),
         (0.42 * scale_km, 0.020 * scale_km), (-0.10 * scale_km, 0.045 * scale_km)],
        closed=True, facecolor="#2b3d55", edgecolor="#1a2431",
        linewidth=0.6, alpha=0.85, zorder=9,
    ))


def build(out_path: Path) -> Path:
    plt.rcParams.update({
        "font.family": ["Noto Sans CJK SC", "Noto Sans SC", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "figure.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
    })

    theta = np.linspace(0.0, 2.0 * np.pi, 721)

    mdv_lookdown = MDV_RADAR * (1.0 + LOOKDOWN_MDV_GAIN)
    cases = [
        ("基准 300 m/s（平飞）", V_BASE, MDV_RADAR, R_NOM, BLUE, "-", 2.2),
        ("慢速 200 m/s（横穿更久）", V_SLOW, MDV_RADAR, R_NOM, ORANGE,
         (0, (5.0, 2.0)), 1.7),
        (f"下视 {LOOKDOWN_DZ/1000:.0f} km（杂波抬高 MDV）",
         V_BASE, mdv_lookdown, R_NOM * LOOKDOWN_RANGE_FRAC, SLATE, ":", 1.6),
    ]

    fig = plt.figure(figsize=(9.6, 10.4))
    ax = fig.add_axes([0.055, 0.115, 0.89, 0.775])
    ax.set_aspect("equal")
    ax.axis("off")
    lim = R_NOM * 1.12
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # ---- 距离环与方位线 ----
    rings = np.arange(5.0, R_NOM + 0.1, 5.0)
    for r in rings:
        ax.add_patch(Circle((0.0, 0.0), r, facecolor="none",
                            edgecolor=GRID, linewidth=0.8, zorder=1))
        ax.text(r * math.cos(math.radians(122)), r * math.sin(math.radians(122)),
                f"{r:.0f}", color="#9a9a9a", fontsize=7.0, ha="center",
                va="center", zorder=3,
                bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
    for deg in range(0, 360, 15):
        a = math.radians(deg)
        major = deg % 45 == 0
        ax.plot([0.0, 1.075 * R_NOM * math.cos(a)],
                [0.0, 1.075 * R_NOM * math.sin(a)],
                color="#dfe3e8" if major else "#f0f2f4",
                linewidth=0.9 if major else 0.5, zorder=0)
    for deg in range(0, 360, 30):
        a = math.radians(deg)
        ax.text(1.105 * R_NOM * math.cos(a), 1.105 * R_NOM * math.sin(a),
                f"{deg}°", fontsize=7.5, color="#8a8a8a",
                ha="center", va="center", zorder=3)

    # ---- 探测概率场（笛卡尔网格，避免极坐标栅格的摩尔纹）----
    n = 720
    gx = np.linspace(-R_NOM, R_NOM, n)
    X, Y = np.meshgrid(gx, gx)
    Rg = np.hypot(X, Y)
    Th = np.arctan2(Y, X)
    gate = visibility(Th, V_BASE, MDV_RADAR)
    nominal = R_NOM * antenna_gain(Th)
    pg = gate / (1.0 + np.exp((Rg - nominal) / (0.05 * R_NOM)))
    pg = np.where(Rg <= R_NOM * 1.004, pg, np.nan)
    mesh = ax.imshow(pg, extent=[-R_NOM, R_NOM, -R_NOM, R_NOM],
                     origin="lower", cmap=CMAP, vmin=0.0, vmax=1.0,
                     alpha=0.55, interpolation="bilinear", zorder=2)

    # ---- 三条包线 ----
    handles = []
    for label, speed, mdv, range_km, color, ls, lw in cases:
        rb = boundary(theta, speed, mdv, range_km)
        ax.plot(rb * np.cos(theta), rb * np.sin(theta), color=color,
                linewidth=lw, linestyle=ls, zorder=6)
        handles.append(Line2D([], [], color=color, lw=lw, ls=ls, label=label))

    # ---- 径向射线（感知图纹理）----
    rb0 = boundary(theta, V_BASE, MDV_RADAR, R_NOM)
    for a in theta[::15]:
        i = np.searchsorted(theta, a)
        if rb0[i] <= 0.10 * R_NOM:
            continue
        rad = a
        ax.plot([0.10 * R_NOM * math.cos(rad), 0.985 * rb0[i] * math.cos(rad)],
                [0.10 * R_NOM * math.sin(rad), 0.985 * rb0[i] * math.sin(rad)],
                color=BLUE, alpha=0.13, linewidth=0.6, zorder=4)

    # ---- 多普勒盲区楔形 ----
    # 可见度归零的方位：|cosθ| ≤ MDV/V，楔形半角 = 90° − acos(MDV/V)。
    half = 90.0 - math.degrees(math.acos(min(MDV_RADAR / V_BASE, 1.0)))
    for ctr in (90.0, 270.0):
        ax.add_patch(Wedge((0.0, 0.0), R_NOM * 1.005, ctr - half, ctr + half,
                           facecolor="#f6d9c8", edgecolor="#e9b896",
                           linewidth=0.7, alpha=0.55, zorder=3))
    ax.annotate(
        f"多普勒盲区\n|v_r| < {MDV_RADAR:.0f} m/s（±{half:.0f}°）",
        xy=(0.0, R_NOM * 0.55), xytext=(R_NOM * 0.30, R_NOM * 0.86),
        ha="center", va="center", fontsize=8.0, color="#9a5a2e",
        arrowprops=dict(arrowstyle="->", color="#c98a5e", lw=0.9,
                        connectionstyle="arc3,rad=-0.25"), zorder=10,
    )

    # ---- 中心机体与标注 ----
    draw_aircraft(ax, R_NOM * 0.068)
    ax.annotate("机头 0°", xy=(R_NOM * 0.075, 0.0), xytext=(R_NOM * 0.30, -R_NOM * 0.115),
                fontsize=8.0, color=INK,
                arrowprops=dict(arrowstyle="->", color="#8a8a8a", lw=0.8), zorder=10)
    ax.text(0.0, -R_NOM * 0.135, "Su-57 机载雷达", fontsize=8.5, color=INK,
            ha="center", va="top", zorder=10)
    ax.annotate("", xy=(R_NOM * 1.055, R_NOM * 0.04), xytext=(R_NOM * 0.94, R_NOM * 0.04),
                arrowprops=dict(arrowstyle="-|>", color="#5c6b7a", lw=1.0), zorder=10)
    ax.text(R_NOM * 0.995, R_NOM * 0.075, "目标航向", fontsize=7.5,
            color="#5c6b7a", ha="center", zorder=10)

    ax.set_title("机载多普勒雷达探测能力感知图（俯视）", fontsize=15,
                 fontweight="bold", color=INK, pad=30)
    ax.text(0.5, 1.012,
            f"标称探测 {R_NOM:.1f} km @ RCS {F16V_SPEC.rcs:.1f} m² · 目标沿 +X 平飞 · "
            f"MDV {MDV_RADAR:.0f} m/s",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=9.5, color="#5f6b78")

    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.005, 0.005),
              fontsize=8.2, frameon=False, handlelength=2.4)

    cax = fig.add_axes([0.32, 0.072, 0.36, 0.012])
    cb = fig.colorbar(mesh, cax=cax, orientation="horizontal")
    cb.set_label("单次扫描探测概率 $P_d$", fontsize=8.0, color=INK, labelpad=2)
    cb.ax.tick_params(labelsize=7.0, colors="#666666")
    cb.outline.set_edgecolor("#cccccc")

    fig.text(0.5, 0.020,
             f"阴影楔：径向速度落进主瓣杂波凹口（过渡带 {NOTCH_BAND:.0f} m/s）；"
             f"下视按等效 MDV {mdv_lookdown:.0f} m/s、可用距离 ×{LOOKDOWN_RANGE_FRAC:.2f} 折算。",
             ha="center", fontsize=7.6, color="#7a7a7a")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    path = build(OUT / "radar_detection_map.png")
    print(f"wrote {path}")
