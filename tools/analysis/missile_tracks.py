"""比例导引弹道图：中距弹 + 目标机（含最近点/脱靶量标注）。

复用 ``missile_g_analysis.fly()`` 跑同一套物理，逐拍取弹与目标的水平位置，
按时间渐变画出平面弹道。默认扫描若干 seed，并自动挑出四种典型情况：

- 命中（引信起爆且 PK 判定通过）
- 常规近失（需用过载始终远低于结构极限）
- 需用过载爆表的近失（PN 末端发散、被钳位后仍近失）
- 近距发射（发射距离最短、飞行时间最短的一发）

输出：out/missile_g/<weapon>_tracks.png

用法（在仓库根目录）：
    python -m tools.analysis.missile_tracks
    python -m tools.analysis.missile_tracks --seeds 2 6 0 12
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap

from dogfight.app.main import resolve_scenario
from tools.analysis.missile_g_analysis import OUT_DIR, fly, summarize

RED_SEQ = LinearSegmentedColormap.from_list("mis", ["#ffb3b3", "#c92a2a", "#7a0d0d"])
BLUE_SEQ = LinearSegmentedColormap.from_list("tgt", ["#a5d8ff", "#1c7ed6", "#0b3d70"])
INK = "#343a40"
MUTED = "#868e96"


def classify(info: dict, tr: dict) -> dict:
    """给一发弹打标签，供自动选案例用。"""
    outcome = info["outcome"]
    is_hit = ("命中" in outcome) and ("近失" not in outcome)
    return {
        "hit": is_hit,
        "near": "近失" in outcome,
        "spike": info["max_req_g"] > 100.0,
        "launch_km": tr["dist"][0],
        "flight_s": info["flight_s"],
    }


def scan(sc, seeds, weapon, shooter):
    out = []
    for seed in seeds:
        world, tr, ev, n = fly(sc, seed, weapon, shooter)
        if tr is None:
            continue
        info = summarize(tr, ev, world, n)
        out.append({
            "seed": seed, "world": world, "tr": tr, "info": info,
            **classify(info, tr),
        })
    return out


def pick_cases(cases):
    """四类各挑一发；挑不到就跳过，别硬凑。"""
    picks = []
    for label, pred in (
        ("命中", lambda c: c["hit"]),
        ("常规近失", lambda c: c["near"] and not c["spike"]),
        ("需用过载爆表", lambda c: c["near"] and c["spike"]),
    ):
        got = next((c for c in cases if pred(c)), None)
        if got is not None:
            picks.append((label, got))
            cases = [c for c in cases if c is not got]
    if cases:
        picks.append(("近距发射", min(cases, key=lambda c: c["launch_km"])))
    return picks


def gradient(ax, x, y, cmap, lw, z):
    segs = np.stack([np.column_stack([x[:-1], y[:-1]]),
                     np.column_stack([x[1:], y[1:]])], axis=1)
    lc = LineCollection(segs, cmap=cmap, linewidths=lw, zorder=z)
    lc.set_array((np.arange(len(segs)) + 1) / max(len(segs), 1))
    lc.set_clim(0.0, 1.0)
    ax.add_collection(lc)


def draw_case(ax, case, label):
    tr, info, world = case["tr"], case["info"], case["world"]
    tgt_name = world.acs[tr["tgt"]].spec.name
    mx = np.asarray(tr["mx"]) / 1e3
    my = np.asarray(tr["my"]) / 1e3
    tx = np.asarray(tr["tx"]) / 1e3
    ty = np.asarray(tr["ty"]) / 1e3
    t = np.asarray(tr["t"])

    gradient(ax, tx, ty, BLUE_SEQ, 2.2, 3)
    gradient(ax, mx, my, RED_SEQ, 1.8, 4)

    ax.plot(tx[0], ty[0], "o", ms=5, mfc="white", mec="#1c7ed6", mew=1.4, zorder=6)
    ax.plot(mx[0], my[0], "o", ms=5, mfc="white", mec="#c92a2a", mew=1.4, zorder=6)

    i = int(np.argmin(tr["dist"]))
    ax.plot(mx[i], my[i], "x", ms=6, color="#c92a2a", zorder=7)
    ax.plot(tx[i], ty[i], "o", ms=4, mfc="#1c7ed6", mec="#1c7ed6", zorder=7)
    ax.plot([mx[i], tx[i]], [my[i], ty[i]], ls=":", lw=1.0, color=MUTED, zorder=5)
    ax.annotate("最近点", xy=((mx[i] + tx[i]) / 2, (my[i] + ty[i]) / 2),
                xytext=(10, -16), textcoords="offset points", fontsize=8,
                color=INK, zorder=8,
                arrowprops=dict(arrowstyle="-", lw=0.7, color=MUTED))

    # 沿轨迹打时间刻度，方便看“谁先到”
    step = max(1, len(t) // 5)
    for k in range(step, len(t), step):
        ax.plot(mx[k], my[k], ".", ms=3, color="#7a0d0d", zorder=6)
        ax.plot(tx[k], ty[k], ".", ms=3, color="#0b3d70", zorder=6)

    ax.set_aspect("equal", adjustable="box")
    cx = np.concatenate([mx, tx])
    cy = np.concatenate([my, ty])
    ax.set_xlim(cx.min() - 0.07 * (cx.max() - cx.min()),
                cx.max() + 0.07 * (cx.max() - cx.min()))
    ax.set_ylim(cy.min() - 0.07 * (cy.max() - cy.min()),
                cy.max() + 0.07 * (cy.max() - cy.min()))
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_xlabel("x 东  km", fontsize=9, color=MUTED, labelpad=1)
    ax.tick_params(labelsize=8, colors=MUTED)

    ax.set_title(f"{label} · seed {case['seed']}\n"
                 f"发射距离 {case['launch_km']:.1f} km（{tgt_name} 为目标）",
                 fontsize=10.5, color=INK, pad=7)
    ax.text(0.03, 0.965,
            f"{info['outcome']}\n飞行 {info['flight_s']:.1f} s · 最大需用 {info['max_req_g']:.0f} g\n"
            f"高度 {min(tr['mz']) / 1e3:.1f}–{max(tr['mz']) / 1e3:.1f} km",
            transform=ax.transAxes, fontsize=8, color=INK, va="top", linespacing=1.35,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#e9ecef", alpha=0.92))


def span_of(case):
    """画面上（含 7% 边距）的横向/纵向跨度，单位 km，用来定面板长宽比。"""
    x = np.concatenate([np.asarray(case["tr"]["mx"]),
                        np.asarray(case["tr"]["tx"])]) / 1e3
    y = np.concatenate([np.asarray(case["tr"]["my"]),
                        np.asarray(case["tr"]["ty"])]) / 1e3
    return max(x.max() - x.min(), 1.0), max(y.max() - y.min(), 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="bvr-1v1")
    ap.add_argument("--weapon", default="R-77-1")
    ap.add_argument("--shooter", default="SU-57")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="指定 seed 列表；不给则自动扫描挑选")
    ap.add_argument("--scan", type=int, default=15, help="自动挑选时的扫描局数")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    plt.rcParams["font.family"] = ["Noto Sans SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.edgecolor"] = "#d9dfe8"

    sc = resolve_scenario(args.scenario)
    if args.seeds:
        picks = [(f"seed {s}", c) for s, c in enumerate(
            scan(sc, args.seeds, args.weapon, args.shooter))]
    else:
        picks = pick_cases(scan(sc, range(args.scan), args.weapon, args.shooter))
    if not picks:
        print("没有可画的弹道")
        return

    n = len(picks)
    rows = 2 if n > 2 else 1
    cols = int(math.ceil(n / rows))

    # 等比例坐标会让面板高矮由数据跨度决定，这里直接按「英寸」排版：
    # 面板宽固定，高 = 宽 × 纵/横跨度比，行高再留出标题与 x 轴标签的位置。
    fig_w = 11.6 if cols > 1 else 6.4
    left_in, right_in, gap_x = 0.85, 0.45, 0.75
    pw = (fig_w - left_in - right_in - (cols - 1) * gap_x) / cols
    title_in, xlabel_in, gap_y, head_in, tail_in = 0.62, 0.50, 0.30, 1.35, 0.25

    row_h, panel_h = [], []
    for r in range(rows):
        chunk = picks[r * cols:(r + 1) * cols]
        ph = [pw * span_of(c)[1] / span_of(c)[0] for _, c in chunk]
        while len(ph) < cols:
            ph.append(ph[-1])
        panel_h.append(ph)
        row_h.append(max(ph) + title_in + xlabel_in)

    fig_h = head_in + sum(row_h) + (rows - 1) * gap_y + tail_in
    fig = plt.figure(figsize=(fig_w, fig_h), dpi=150, facecolor="white")

    y_cursor = head_in
    for r in range(rows):
        for c in range(cols):
            k = r * cols + c
            if k >= n:
                continue
            label, case = picks[k]
            h = panel_h[r][c]
            ax = fig.add_axes([
                (left_in + c * (pw + gap_x)) / fig_w,
                1.0 - (y_cursor + h) / fig_h,
                pw / fig_w, h / fig_h,
            ])
            draw_case(ax, case, label)
            if c == 0:
                ax.set_ylabel("y 北  km", fontsize=9, color=MUTED, labelpad=1)
        y_cursor += row_h[r] + gap_y

    fig.text(0.5, 1.0 - 0.38 / fig_h,
             f"{args.shooter} 发射 {args.weapon}：比例导引弹道（{args.scenario}）",
             ha="center", va="center", fontsize=13, color=INK)
    fig.text(0.5, 1.0 - 0.78 / fig_h,
             "红 = 导弹（浅→深为时间推移）   蓝 = 目标机   白圈 = 发射时刻   "
             "× = 最近点   灰虚线 = 脱靶距离",
             ha="center", va="center", fontsize=9, color=MUTED)

    out = Path(args.out) if args.out else (
        OUT_DIR / f"{args.weapon.lower()}_tracks.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print("\n".join(
        f"{label}: seed {c['seed']} {c['info']['outcome']}" for label, c in picks))
    print("saved:", out)


if __name__ == "__main__":
    main()
