"""单发导弹的需用过载 / 可用过载统计（PN 指令 vs 结构+动压上限）。

需用过载：PN 制导律算出的原始横向加速度指令（钳位、舵机滞后之前），
          含追踪项与 loft，逐拍写入 ``Missile.g_req``；
可用过载：``min(结构极限 max_g, 动压×cn_max/mass + 燃气舵直接力)``，写入
          ``Missile.g_avail``。两者都在 ``core/sim.py::_update_missiles`` 里更新，
本工具只负责逐拍采样与作图，不另写一套制导近似。

跑一局 ``bvr-1v1``（Su-57 vs F-16V），挑一枚由指定机型发射的中距雷达弹，
从点火统计到命中/近失/自毁，输出「过载-时间 + 弹目距离/马赫」双栏图。

输出：out/missile_g/<weapon>_overload.png

用法（在仓库根目录）：
    python -m tools.analysis.missile_g_analysis
    python -m tools.analysis.missile_g_analysis --seed 12 --weapon R-77-1
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dogfight.app.main import build_agents, build_world, resolve_scenario

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "out" / "missile_g"

INK = "#343a40"
RED = "#e03131"
BLUE = "#1c7ed6"
GREEN = "#2b8a3e"
AMBER = "#f08c00"
GREY = "#868e96"


def air_sound(alt: float) -> float:
    """ISA 标准大气下的当地声速，仅用于把弹速换算成马赫数。"""
    h = min(max(alt, 0.0), 32000.0)
    if h <= 11000.0:
        temp = 288.15 - 0.0065 * h
    elif h <= 20000.0:
        temp = 216.65
    else:
        temp = 216.65 + 0.001 * (h - 20000.0)
    return math.sqrt(1.4 * 287.053 * temp)


def fly(scenario, seed: int, weapon: str, shooter: str):
    """跑一局并逐拍采集指定机型发射的同型弹弹道。"""
    world = build_world(scenario=scenario)
    agents = build_agents(seed=seed, scenario=scenario)
    shooter_idx = next(
        (i for i, ac in enumerate(world.acs) if ac.spec.name == shooter), None
    )
    if shooter_idx is None:
        raise SystemExit(f"场景里没有 {shooter}")

    traces = {}
    end_events = {}
    seen = 0
    for _ in range(int(scenario.time_limit / world.dt)):
        actions = [ag.act(world.observe(k)) for k, ag in enumerate(agents)]
        world.step(actions)
        new_events, seen = world.events[seen:], len(world.events)

        for m in world.missiles:
            if m.owner != shooter_idx or m.spec.name != weapon:
                continue
            key = id(m)
            tr = traces.setdefault(key, {
                "missile": m, "t": [], "req": [], "avail": [], "lat": [],
                "dist": [], "mach": [], "active": [], "dead_t": None,
                # 弹道图用的世界坐标（x 东 / y 北 / z 上），单位 m
                "mx": [], "my": [], "mz": [], "tx": [], "ty": [], "tz": [],
                "tgt": None,
            })
            if not m.alive:
                if tr["dead_t"] is None:
                    tr["dead_t"] = world.t
                    for t_ev, msg in new_events:
                        if weapon in msg and ("命中" in msg or "近失" in msg):
                            end_events[key] = (t_ev, msg)
                continue
            if not tr["t"] and m.t <= 0.0:
                continue
            tgt = world.acs[m.target]
            dist = math.sqrt(
                (tgt.x - m.x) ** 2 + (tgt.y - m.y) ** 2 + (tgt.z - m.z) ** 2
            )
            tr["t"].append(m.t - m.t_ign)
            tr["req"].append(m.g_req)
            tr["avail"].append(m.g_avail)
            tr["lat"].append(math.sqrt(m.gx ** 2 + m.gy ** 2 + m.gz ** 2) / 9.81)
            tr["dist"].append(dist / 1000.0)
            tr["mach"].append(m.speed / max(air_sound(m.z), 1e-6))
            tr["active"].append(1.0 if m.active else 0.0)
            tr["mx"].append(m.x)
            tr["my"].append(m.y)
            tr["mz"].append(m.z)
            tr["tx"].append(tgt.x)
            tr["ty"].append(tgt.y)
            tr["tz"].append(tgt.z)
            tr["tgt"] = m.target

        # 一旦拿到带引信事件的完整弹道就不必再往下跑（另一枚多半已飞远）；
        # 否则等追踪的这几枚都终止即可收工。
        if world.result is not None or end_events:
            break
        if traces and all(tr["dead_t"] is not None for tr in traces.values()):
            break

    # 优先取「以命中/近失事件收尾」的那一枚，否则取先发射的一枚。
    for key, tr in traces.items():
        if key in end_events and tr["t"]:
            return world, tr, end_events[key], len(traces)
    for key, tr in traces.items():
        if tr["t"]:
            return world, tr, None, len(traces)
    return world, None, None, len(traces)


def summarize(tr: dict, end_ev, world, n_traces: int) -> dict:
    t, req, avail = tr["t"], tr["req"], tr["avail"]
    dt = world.dt
    sat = [r > a for r, a in zip(req, avail)]
    sat_time = sum(sat) * dt
    if end_ev is not None:
        outcome = end_ev[1]
    elif tr["dead_t"] is not None:
        outcome = f"未触发引信，飞至 {t[-1]:.1f} s 后自毁（最近 {min(tr['dist']):.2f} km）"
    else:
        outcome = "弹仍在飞"
    return {
        "weapon": tr["missile"].spec.name,
        "pn_gain": tr["missile"].spec.pn_gain,
        "structural_limit_g": tr["missile"].spec.max_g,
        "traces": n_traces,
        "flight_s": round(t[-1], 3),
        "max_req_g": round(max(req), 2),
        "max_avail_g": round(max(avail), 2),
        "min_avail_g": round(min(avail), 2),
        "saturated_s": round(sat_time, 3),
        "saturated_ratio": round(sat_time / t[-1], 4),
        "outcome": outcome,
    }


def plot(tr: dict, world, info: dict, out: Path) -> None:
    plt.rcParams["font.family"] = ["Noto Sans SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["axes.edgecolor"] = "#d9dfe8"

    m = tr["missile"]
    t, req, avail = tr["t"], tr["req"], tr["avail"]
    lat, dist, mach = tr["lat"], tr["dist"], tr["mach"]
    pitbull = next((t[i] for i, a in enumerate(tr["active"]) if a > 0.5), None)

    bands = bands_of([r > a for r, a in zip(req, avail)], t)
    gaps = bands_of([r <= 1e-9 for r in req], t, min_len=0.05, skip_before=0.1)

    fig = plt.figure(figsize=(13, 8.4), dpi=150, facecolor="white")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.5, 1.0], hspace=0.14)

    ax = fig.add_subplot(gs[0])
    ax.plot(t, req, color=RED, lw=1.7, label="需用过载（PN 原始指令）")
    ax.plot(t, avail, color=BLUE, lw=1.7, label="可用过载（结构/动压上限）")
    ax.plot(t, lat, color=GREY, lw=1.0, ls="--", label="实际建立过载（舵机滞后后）")
    for a0, a1 in bands:
        ax.axvspan(a0, a1, color=RED, alpha=0.12, lw=0)
    for a0, a1 in gaps:
        ax.axvspan(a0, a1, color=AMBER, alpha=0.13, lw=0)

    if pitbull is not None:
        ax.axvline(pitbull, color=GREEN, lw=1.0, ls=":")
        ax.annotate("导引头自主截获", xy=(pitbull, 0.90),
                    xycoords=("data", "axes fraction"), xytext=(4, 0),
                    textcoords="offset points", color=GREEN, fontsize=9, va="center")
    if gaps:
        ax.annotate("视线数据中断\n（盲区/失照，弹按惯导外推）",
                    xy=(gaps[0][0], 0.905), xycoords=("data", "axes fraction"),
                    xytext=(5, 0), textcoords="offset points", color="#c26a00",
                    fontsize=8.5, va="center")

    ax.set_ylabel("过载  g")
    ax.set_ylim(bottom=0)
    ax.set_xlim(t[0], t[-1])
    ax.grid(alpha=0.25, lw=0.6)
    ax.legend(loc="upper right", framealpha=0.9)

    fig.text(
        0.5, 0.912,
        f"飞行时长 {t[-1]:.1f} s   最大需用 {max(req):.1f} g   "
        f"最小可用 {min(avail):.2f} g   需用超过可用 {info['saturated_s']:.2f} s"
        f"（{100 * info['saturated_ratio']:.1f}% 飞行时间）",
        ha="center", fontsize=9, color=INK,
    )

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(t, dist, color=GREEN, lw=1.6, label="弹目距离")
    ax2.set_ylabel("弹目距离  km", color=GREEN)
    ax2.tick_params(axis="y", colors=GREEN)
    ax2.set_xlabel("自点火起的时间  s")
    ax2.set_xlim(t[0], t[-1])
    ax2.set_ylim(bottom=0)
    ax2.grid(alpha=0.25, lw=0.6)

    ax3 = ax2.twinx()
    ax3.plot(t, mach, color=AMBER, lw=1.4, label="马赫数")
    ax3.set_ylabel("马赫数", color=AMBER)
    ax3.tick_params(axis="y", colors=AMBER)
    ax3.set_ylim(bottom=0)

    for a0, a1 in bands:
        for a in (ax2, ax3):
            a.axvspan(a0, a1, color=RED, alpha=0.10, lw=0)
    if pitbull is not None:
        for a in (ax2, ax3):
            a.axvline(pitbull, color=GREEN, lw=0.9, ls=":")

    handles = [ln for ln in ax2.get_lines() + ax3.get_lines()
               if not ln.get_label().startswith("_")]
    ax2.legend(handles, [ln.get_label() for ln in handles],
               loc="center right", framealpha=0.9)

    fig.suptitle(
        f"{m.spec.name} 需用过载 vs 可用过载"
        f"（共发射 {info['traces']} 发，图示为触发引信的一发；"
        f"PN N={m.spec.pn_gain:g}，结构极限 {m.spec.max_g:g} g）\n{info['outcome']}",
        fontsize=11.5,
    )
    fig.subplots_adjust(left=0.075, right=0.925, top=0.855, bottom=0.085)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)


def bands_of(mask, t, min_len: float = 0.0, skip_before: float = 0.0):
    """把布尔序列切成 [起, 止] 时间区间。"""
    out, i0 = [], None
    for i, on in enumerate(mask):
        if on and i0 is None and t[i] > skip_before:
            i0 = i
        elif not on and i0 is not None:
            if t[i] - t[i0] >= min_len:
                out.append((t[i0], t[i]))
            i0 = None
    if i0 is not None and t[-1] - t[i0] >= min_len:
        out.append((t[i0], t[-1]))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default="bvr-1v1", help="场景名")
    ap.add_argument("--seed", type=int, default=12, help="智能体随机种子")
    ap.add_argument("--weapon", default="R-77-1", help="导弹型号")
    ap.add_argument("--shooter", default="SU-57", help="发射方机型")
    ap.add_argument("--out", default=None, help="输出 PNG 路径")
    args = ap.parse_args()

    scenario = resolve_scenario(args.scenario)
    world, tr, end_ev, n_traces = fly(scenario, args.seed, args.weapon, args.shooter)
    if tr is None:
        print(json.dumps(
            {"status": f"本局 {args.shooter} 未发射 {args.weapon}", "seed": args.seed},
            ensure_ascii=False))
        return

    info = summarize(tr, end_ev, world, n_traces)
    out = Path(args.out) if args.out else (
        OUT_DIR / f"{args.weapon.lower()}_overload.png")
    plot(tr, world, info, out)
    print(json.dumps(dict(info, seed=args.seed, png=str(out)),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
