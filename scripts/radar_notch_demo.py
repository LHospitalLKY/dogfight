"""多普勒雷达盲区（notch）可视化演示（俯视态势 + 径向速度曲线）。

物理口径直接复用 sim.py 的多普勒盲区模型，不另写一套近似：

- ``radar_visibility()`` 给出目标相对地面杂波的可见度（含下视杂波抬高 MDV）；
- ``launch_envelope()`` 给出雷达制导中距弹的有效跟踪距离（随 RCS 与可见度缩放），
  锁定计时直接按它的结果累加/清零，与对局里的 ``_update_lock`` 完全一致。

演示场景（同高 9 km，本机坐标系）：

- 本机沿 +X 以 220 m/s 平飞，目标从右前方 17 km 处以 250 m/s 横向穿越；
- 目标多普勒径向速度 |v_r| 随视线角变化，横穿瞬间趋近 0，
  回波落进主瓣杂波凹口（|v_r| < MDV = 40 m/s），雷达丢失；
- 目标穿出凹口后径向速度重新抬起，1.5 s 稳定跟踪后重新锁定。

输出：out/radar/radar_notch_demo.mp4

用法：
    python scripts/radar_notch_demo.py
    python scripts/radar_notch_demo.py --fps 24 --seconds 34
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, Polygon, Rectangle, Wedge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import F16V_SPEC, SU57_SPEC  # noqa: E402
from sim import (  # noqa: E402
    LOOKDOWN_MDV_GAIN,
    LOOKDOWN_REF_DZ,
    MDV_RADAR,
    NOTCH_BAND,
    World,
    clamp,
    launch_envelope,
    radar_visibility,
    world_velocity,
    wrap_pi,
)

OUT_DIR = ROOT / "out" / "radar"

INK = "#2b2b2b"
MUTED = "#6b7f95"
GRID = "#e7e7e7"
PANEL = "#f7f9fc"
BLUE = "#1981fe"
ORANGE = "#f28b1a"
GREEN = "#2f9e44"
AMBER = "#f59f00"
RED = "#e03131"
PURPLE = "#9c36b5"

# ---- 场景几何 ----
OWN0 = (-4400.0, 0.0)   # 本机开局位置 m
OWN_V = 220.0           # 本机速度 m/s，航向 +X
TGT0 = (12000.0, -5000.0)  # 目标开局位置 m（右前方约 17 km）
TGT_V = 250.0           # 目标速度 m/s，航向 +Y（横穿）
Z0 = 9000.0             # 同高度
VIEW_KM = 23.0          # 俯视图半幅
T_END = 34.0            # 演示时长 s
RATE = 120.0            # 内部积分频率 Hz

# ---- 俯视机体外形（机头朝 +Y，单位长度归一） ----
JET = np.array([
    [0.00, 1.05], [0.14, 0.30], [1.00, -0.12], [1.00, -0.34],
    [0.18, -0.26], [0.32, -0.82], [0.55, -0.98], [-0.55, -0.98],
    [-0.32, -0.82], [-0.18, -0.26], [-1.00, -0.34], [-1.00, -0.12],
    [-0.14, 0.30],
])
HEAD = np.array([[0.0, 1.0], [-0.45, 0.25], [0.45, 0.25]])


def _rotate(pts: np.ndarray, ang: float) -> np.ndarray:
    c, s = math.cos(ang), math.sin(ang)
    return np.column_stack([pts[:, 0] * c - pts[:, 1] * s,
                            pts[:, 0] * s + pts[:, 1] * c])


def _shape(pts: np.ndarray, center, heading: float, scale: float) -> np.ndarray:
    """把指向 +Y 的局部外形旋转到相对航向 heading（俯视，顺时针为正）。"""
    return _rotate(pts, -heading) * scale + np.asarray(center, dtype=float)


# ---------------------------------------------------------------------------
# 预计算：真实模型逐点采样
# ---------------------------------------------------------------------------

def build_history() -> dict:
    world = World(
        [SU57_SPEC, F16V_SPEC], dt=1.0 / RATE, arena=1.0e9, time_limit=1.0e9,
        z0=Z0, z_floor=0.0, z_ceiling=45000.0, dof=3, y_offset=0.0, x_offset=0.0,
    )
    own, tgt = world.acs
    own.psi, own.gamma, own.v = 0.0, 0.0, OWN_V
    tgt.psi, tgt.gamma, tgt.v = math.pi / 2.0, 0.0, TGT_V
    rspec = own.spec.radar_missile
    lock_time = rspec.lock_time

    steps = int(round(T_END * RATE)) + 1
    ts = np.arange(steps, dtype=float) / RATE
    relx = np.zeros(steps)
    rely = np.zeros(steps)
    dist = np.zeros(steps)
    bearing = np.zeros(steps)
    vr = np.zeros(steps)
    vis = np.zeros(steps)
    reach = np.zeros(steps)
    gate = np.zeros(steps)
    timer = np.zeros(steps)
    locked = np.zeros(steps, dtype=bool)
    inside = np.zeros(steps, dtype=bool)

    lt = 0.0
    dt = 1.0 / RATE
    for i, t in enumerate(ts):
        ox = OWN0[0] + OWN_V * t
        oy = OWN0[1]
        tx = TGT0[0]
        ty = TGT0[1] + TGT_V * t
        own.x, own.y, own.z = ox, oy, Z0
        tgt.x, tgt.y, tgt.z = tx, ty, Z0

        rx, ry = tx - ox, ty - oy
        d = math.hypot(rx, ry)
        los = math.atan2(ry, rx)
        vx, vy, vz = world_velocity(tgt)
        vr[i] = abs((vx * rx + vy * ry + vz * 0.0) / max(d, 1e-9))
        vis[i] = radar_visibility(own, tgt)

        lo, hi = launch_envelope(own, rspec, tgt)
        reach[i] = hi
        ang = abs(wrap_pi(los - own.psi))
        inside[i] = (lo <= d <= hi and ang <= rspec.seeker_fov)

        lookdown = LOOKDOWN_MDV_GAIN * clamp(
            (own.z - tgt.z) / LOOKDOWN_REF_DZ, 0.0, 1.0)
        gate[i] = MDV_RADAR * (1.0 + lookdown)

        lt = min(lt + dt, lock_time) if inside[i] else 0.0
        timer[i] = lt
        locked[i] = lt >= lock_time

        relx[i] = rx
        rely[i] = ry
        dist[i] = d
        bearing[i] = los

    return {
        "ts": ts, "relx": relx, "rely": rely, "dist": dist,
        "bearing": bearing, "vr": vr, "vis": vis, "reach": reach,
        "gate": gate, "timer": timer, "locked": locked, "inside": inside,
        "lock_time": lock_time,
    }


def _runs(codes: np.ndarray, ts: np.ndarray):
    out, start = [], 0
    for i in range(1, len(codes) + 1):
        if i == len(codes) or codes[i] != codes[start]:
            out.append((int(codes[start]), float(ts[start]),
                        float(ts[min(i, len(ts) - 1)])))
            start = i
    return out


# ---------------------------------------------------------------------------
# 绘图
# ---------------------------------------------------------------------------

STATUS_COLOR = {0: RED, 1: AMBER, 2: GREEN}
STATUS_TEXT = {0: "盲区 · 雷达丢失", 1: "过渡带 · 跟踪不稳", 2: "稳定跟踪"}


def build_figure(h: dict, fps: int):
    plt.rcParams["font.family"] = [
        "Noto Sans SC", "Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.edgecolor"] = "#d9dfe8"

    fig = plt.figure(figsize=(16.0, 8.8), dpi=100, facecolor="white")
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.0, 1.12], height_ratios=[1.22, 1.0],
        left=0.045, right=0.975, top=0.865, bottom=0.095,
        wspace=0.16, hspace=0.38,
    )
    ax_map = fig.add_subplot(gs[:, 0])
    ax_vr = fig.add_subplot(gs[0, 1])
    sub = gs[1, 1].subgridspec(2, 1, height_ratios=[0.85, 1.15], hspace=0.62)
    ax_tl = fig.add_subplot(sub[0])
    ax_st = fig.add_subplot(sub[1])

    fig.suptitle("多普勒雷达盲区：目标横向穿越 → 落入主瓣杂波凹口 → 重新截获",
                 fontsize=16, fontweight="bold", color=INK, y=0.965)
    fig.text(0.5, 0.915,
             "同高 9 km ｜ 本机 220 m/s 平飞 ｜ 目标 250 m/s 横穿 ｜ "
             "可见度与锁定计时直接调用 sim.radar_visibility / launch_envelope",
             ha="center", va="center", fontsize=10.5, color=MUTED)

    # ---------------- 左：俯视态势 ----------------
    ax = ax_map
    ax.set_facecolor(PANEL)
    ax.set_xlim(-VIEW_KM, VIEW_KM)
    ax.set_ylim(-VIEW_KM, VIEW_KM)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("俯视态势（本机航向朝上）", fontsize=12, color=INK, pad=10)

    for r in (5, 10, 15, 20):
        ax.add_patch(Circle((0, 0), r, fill=False, lw=1.0, ec=GRID, zorder=1))
        ax.text(-0.9, r, f"{r} km", fontsize=8.5, color="#9aa8b8",
                ha="right", va="center")
    ax.plot([0, 0], [-VIEW_KM, VIEW_KM], color=GRID, lw=0.8, zorder=1)
    ax.plot([-VIEW_KM, VIEW_KM], [0, 0], color=GRID, lw=0.8, zorder=1)

    ax.add_patch(Wedge((0, 0), VIEW_KM * 1.5, 30.0, 150.0, facecolor=BLUE,
                       alpha=0.045, edgecolor=BLUE, lw=1.0,
                       linestyle=(0, (6, 4)), zorder=1))
    ax.text(0.04, 0.945, "淡蓝扇区 = ±60° 导引头视场", fontsize=8.5,
            color=BLUE, ha="left", va="top", alpha=0.75,
            transform=ax.transAxes)

    # 目标全航迹（相对坐标里是直线）
    ax.plot(h["rely"] / 1000.0, h["relx"] / 1000.0, color="#cfd8e3",
            lw=1.4, ls=":", zorder=2)
    trail, = ax.plot([], [], color=ORANGE, lw=1.2, alpha=0.6, zorder=3)

    reach_ring = Circle((0, 0), 1.0, fill=False, ec=AMBER, lw=1.4,
                        ls="--", zorder=2, alpha=0.9)
    ax.add_patch(reach_ring)

    los_line, = ax.plot([], [], lw=2.4, zorder=4)

    own_poly = Polygon(_shape(JET, (0.0, 0.0), 0.0, 0.85), closed=True,
                       facecolor=BLUE, edgecolor="white", lw=1.6, zorder=6)
    ax.add_patch(own_poly)
    tgt_poly = Polygon(_shape(JET, (0.0, 0.0), math.pi / 2.0, 0.85),
                       closed=True, facecolor=ORANGE, edgecolor="white",
                       lw=1.6, zorder=6)
    ax.add_patch(tgt_poly)

    vel_line, = ax.plot([], [], color=ORANGE, lw=2.0, alpha=0.85, zorder=5)
    vel_head = Polygon(_shape(HEAD, (0.0, 0.0), math.pi / 2.0, 0.55),
                       closed=True, facecolor=ORANGE, edgecolor="none",
                       alpha=0.85, zorder=5)
    ax.add_patch(vel_head)
    ax.text(VIEW_KM * 0.86, -VIEW_KM * 0.9, "橙色箭头 = 目标速度矢量",
            fontsize=9, color=ORANGE, ha="right", va="bottom")

    hud = ax.text(0.025, 0.035, "", transform=ax.transAxes, fontsize=10.5,
                  color=INK, ha="left", va="bottom", linespacing=1.55,
                  bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#d9dfe8",
                            alpha=0.92))
    pill = ax.text(0.5, 0.952, "", transform=ax.transAxes, fontsize=13,
                   fontweight="bold", color="white", ha="center", va="center",
                   bbox=dict(boxstyle="round,pad=0.45", fc=GREEN, ec="none"))

    # ---------------- 右上：径向速度 ----------------
    av = ax_vr
    av.set_facecolor("white")
    av.set_xlim(0.0, T_END)
    av.set_ylim(0.0, 115.0)
    av.set_title("目标多普勒径向速度 |v_r| 与检测门限", fontsize=12,
                 color=INK, pad=8, loc="left")
    av.set_ylabel("|v_r| (m/s)", fontsize=10, color=INK)
    av.grid(True, color=GRID, lw=0.8)
    av.set_axisbelow(True)
    for sp in ("top", "right"):
        av.spines[sp].set_visible(False)

    av.axhspan(0.0, MDV_RADAR, color=RED, alpha=0.13, zorder=0)
    av.axhspan(MDV_RADAR, MDV_RADAR + NOTCH_BAND, color=AMBER, alpha=0.13,
               zorder=0)
    av.axhline(MDV_RADAR, color=RED, lw=1.2, ls="--", zorder=1)
    av.axhline(MDV_RADAR * (1.0 + LOOKDOWN_MDV_GAIN), color=PURPLE, lw=1.1,
               ls=":", zorder=1)
    av.text(0.4, 108.0, "红带：|v_r| < MDV 40 m/s（主瓣杂波凹口）",
            fontsize=9, color=RED, va="top", ha="left")
    av.text(0.4, 101.0, "紫点线：下视 4 km 时门限抬高到 62 m/s",
            fontsize=9, color=PURPLE, va="top", ha="left")

    vr_curve, = av.plot([], [], color=BLUE, lw=2.0, zorder=3)
    vr_dot, = av.plot([], [], "o", color=BLUE, ms=6, zorder=4)
    vr_cursor = av.axvline(0.0, color="#9aa8b8", lw=1.0, ls="-", zorder=2)

    # ---------------- 右下上：状态时间条 ----------------
    at = ax_tl
    at.set_xlim(0.0, T_END)
    at.set_ylim(0.0, 1.0)
    at.set_title("雷达跟踪状态", fontsize=11, color=INK, pad=8, loc="left")
    at.set_yticks([])
    at.tick_params(axis="x", labelsize=8.5, colors=MUTED, length=3)
    at.set_xlabel("时间 (s)", fontsize=9, color=MUTED, labelpad=2)
    for sp in ("top", "left", "right"):
        at.spines[sp].set_visible(False)

    codes = np.where(h["vis"] <= 0.0, 0, np.where(h["vis"] >= 1.0, 2, 1))
    for code, t0, t1 in _runs(codes, h["ts"]):
        at.broken_barh([(t0, max(t1 - t0, 0.03))], (0.2, 0.6),
                       facecolors=STATUS_COLOR[code], edgecolors="none",
                       alpha=0.75)
    at_cursor = at.axvline(0.0, color=INK, lw=1.6)
    for x, label, color in ((0.52, "稳定跟踪", GREEN), (0.68, "过渡带", AMBER),
                            (0.80, "盲区", RED)):
        at.text(x, 0.99, f"■ {label}", transform=at.transAxes, fontsize=8.5,
                color=color, ha="left", va="bottom")

    # ---------------- 右下下：状态条 ----------------
    ast = ax_st
    ast.set_xlim(0.0, 1.0)
    ast.set_ylim(0.0, 1.0)
    ast.axis("off")

    rows = []
    for y, label in ((0.80, "多普勒可见度"), (0.50, "雷达锁定进度")):
        ast.text(0.0, y, label, fontsize=10.5, color=INK, va="center")
        ast.add_patch(Rectangle((0.30, y - 0.07), 0.60, 0.14, facecolor="#eef1f6",
                                edgecolor="#d9dfe8", lw=0.8))
        fill = Rectangle((0.30, y - 0.07), 0.0, 0.14, facecolor=GREEN,
                         edgecolor="none")
        ast.add_patch(fill)
        val = ast.text(0.925, y, "", fontsize=10.5, color=INK, va="center",
                       ha="left")
        rows.append((fill, val))

    ast.text(0.0, 0.14, "", fontsize=11.5, color=INK, va="center")
    launch_txt = ast.texts[-1]

    def update(frame: int) -> None:
        idx = min(int(frame * RATE / fps), len(h["ts"]) - 1)
        t = h["ts"][idx]
        rx, ry = h["relx"][idx], h["rely"][idx]
        px, py = ry / 1000.0, rx / 1000.0
        d = h["dist"][idx] / 1000.0
        bdeg = math.degrees(h["bearing"][idx])
        v = h["vis"][idx]
        locked = bool(h["locked"][idx])

        trail.set_data(h["rely"][:idx + 1] / 1000.0,
                       h["relx"][:idx + 1] / 1000.0)
        los_line.set_data([0.0, px], [0.0, py])
        los_line.set_color(RED if v <= 0.0 else (AMBER if v < 1.0 else GREEN))
        los_line.set_linestyle("--" if v <= 0.0 else "-")

        ang = math.pi / 2.0
        tgt_poly.set_xy(_shape(JET, (px, py), ang, 0.85))
        reach_ring.set_radius(max(h["reach"][idx] / 1000.0, 0.001))

        vdx, vdy = math.sin(ang), math.cos(ang)
        tip = (px + 3.4 * vdx, py + 3.4 * vdy)
        vel_line.set_data([px, tip[0]], [py, tip[1]])
        vel_head.set_xy(_shape(HEAD, tip, ang, 0.55))

        code = 0 if v <= 0.0 else (1 if v < 1.0 else 2)
        pill.set_text(STATUS_TEXT[code] + ("  ·  已锁定" if locked else ""))
        pill.set_bbox(dict(boxstyle="round,pad=0.45", fc=STATUS_COLOR[code],
                           ec="none"))

        hud.set_text(
            f"t = {t:5.1f} s\n"
            f"距离      {d:5.1f} km\n"
            f"方位      {bdeg:+5.1f}°\n"
            f"|v_r|     {h['vr'][idx]:5.0f} m/s\n"
            f"可见度    {v:4.2f}"
        )

        vr_curve.set_data(h["ts"][:idx + 1], h["vr"][:idx + 1])
        vr_dot.set_data([t], [h["vr"][idx]])
        vr_cursor.set_xdata([t, t])

        at_cursor.set_xdata([t, t])

        fill_vis, val_vis = rows[0]
        fill_vis.set_width(0.60 * v)
        fill_vis.set_facecolor(STATUS_COLOR[code])
        val_vis.set_text(f"{v:4.2f}")

        fill_lock, val_lock = rows[1]
        prog = min(h["timer"][idx] / h["lock_time"], 1.0)
        fill_lock.set_width(0.60 * prog)
        fill_lock.set_facecolor(GREEN if locked else AMBER)
        val_lock.set_text(f"{h['timer'][idx]:3.1f} / {h['lock_time']:.1f} s")

        if locked:
            launch_txt.set_text("R-77-1 发射条件：满足（已建立雷达锁定）")
            launch_txt.set_color(GREEN)
        elif v <= 0.0:
            launch_txt.set_text("R-77-1 发射条件：不满足（目标进入杂波凹口）")
            launch_txt.set_color(RED)
        else:
            launch_txt.set_text("R-77-1 发射条件：不满足（锁定计时清零中）")
            launch_txt.set_color(AMBER)

    return fig, update


def main() -> None:
    global T_END
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fps", type=int, default=20, help="视频帧率")
    ap.add_argument("--seconds", type=float, default=T_END, help="演示时长 s")
    ap.add_argument("--out", type=str,
                    default=str(OUT_DIR / "radar_notch_demo.mp4"))
    args = ap.parse_args()

    T_END = args.seconds

    h = build_history()
    fig, update = build_figure(h, args.fps)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int(round(T_END * args.fps)) + 1
    writer = imageio.get_writer(str(out), fps=args.fps, codec="libx264",
                                quality=7, pixelformat="yuv420p",
                                macro_block_size=8)
    try:
        for k in range(n_frames):
            update(k)
            fig.canvas.draw()
            writer.append_data(np.asarray(fig.canvas.buffer_rgba())[..., :3])
            if k % 60 == 0:
                print(f"  渲染 {k}/{n_frames} 帧 (t={k / args.fps:5.1f} s)")
    finally:
        writer.close()
        plt.close(fig)

    # 控制台摘要：便于核对关键事件时刻
    ts = h["ts"]
    locked = h["locked"]
    first = None
    lost = None
    back = None
    for i in range(1, len(ts)):
        if locked[i] and first is None:
            first = ts[i]
        elif first is not None and lost is None and not locked[i] and ts[i] > first:
            lost = ts[i]
        elif lost is not None and back is None and locked[i]:
            back = ts[i]
    print(f"视频输出：{out}")
    print(f"  首次锁定 t≈{first:.1f}s，丢失 t≈{lost:.1f}s，重新锁定 t≈{back:.1f}s"
          if first is not None and lost is not None and back is not None
          else "  锁定事件不足，请检查场景参数")


if __name__ == "__main__":
    main()
