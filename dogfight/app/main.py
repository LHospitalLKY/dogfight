"""M2 入口：6DOF（默认）/3DOF 内核 + 几何态势/机动原语战术层。

用法（dogfight conda 环境）：
    python -m dogfight               # 跑一局并打印过程与结果（6DOF）
    python -m dogfight --baseline    # 双方都用纯追踪基线，做对照
    python -m dogfight.web           # three.js 浏览器观战
"""
from __future__ import annotations

import argparse
import dataclasses
import math
from collections import Counter

from dogfight.config import (
    ConfigBundle,
    DEFAULT_SCENARIO,
    Scenario,
    load_bundle,
)
from dogfight.core.sim import (
    Action, AircraftSpec, World, air_density,
    cd0_at, mach_number,
)
from dogfight.tactics.agents import PurePursuit, TacticalAgent

PHYSICS_DT = 1.0 / 120.0
DECIDE_HZ = 10
STEPS_PER_DECISION = round(1.0 / (DECIDE_HZ * PHYSICS_DT))

_BUNDLE: ConfigBundle | None = None


def get_bundle(config_dir: str | None = None) -> ConfigBundle:
    """懒加载配置包：只有真正跑仿真时才读 YAML。"""
    global _BUNDLE
    if _BUNDLE is None or config_dir is not None:
        _BUNDLE = load_bundle(config_dir)
    return _BUNDLE


def resolve_scenario(name: str | None = None,
                     config_dir: str | None = None) -> Scenario:
    return get_bundle(config_dir).scenario(name or DEFAULT_SCENARIO)


def build_world(
    time_limit: float | None = None,
    dof: int = 6,
    swap: bool = False,
    y_offset: float | None = None,
    awacs: bool | None = None,
    scenario: Scenario | None = None,
) -> World:
    sc = scenario or resolve_scenario()
    specs = [team.aircraft.spec for team in sc.teams]
    if swap:
        specs.reverse()
    init = sc.initial
    return World(
        specs,
        dt=sc.physics_dt,
        arena=init.arena_m,
        time_limit=sc.time_limit if time_limit is None else time_limit,
        dof=dof,
        y_offset=init.y_offset_m if y_offset is None else y_offset,
        x_offset=init.spacing_m,
        z0=init.altitude_m,
        z_floor=init.floor_m,
        z_ceiling=init.ceiling_m,
        awacs=sc.awacs if awacs is None else awacs,
    )


def build_agents(
    baseline: bool = False, swap: bool = False, seed: int | None = None,
    f16_radar: str | None = None, su57_radar: str | None = None,
    scenario: Scenario | None = None,
    radar_policy: dict[str, str] | None = None,
) -> list:
    """seed 只用于实验复现；默认 None 表示每次启动随机占位。

    雷达策略优先级：radar_policy 覆盖 > 旧机型参数 > 场景配置。
    旧的 --f16-radar / --su57-radar 只是兼容别名，新场景请用 team 名覆盖。
    """
    if baseline:
        return [PurePursuit(), PurePursuit()]
    sc = scenario or resolve_scenario()
    overrides = radar_policy or {}
    agents = []
    for index, team in enumerate(sc.teams):
        policy = (
            overrides.get(team.name)
            or overrides.get(team.aircraft.id)
            or (f16_radar if team.aircraft.id == "F-16V" else None)
            or (su57_radar if team.aircraft.id == "SU-57" else None)
            or team.radar_policy
        )
        team_seed = None if seed is None else seed + index
        agents.append(TacticalAgent(
            **dataclasses.asdict(team.aircraft.agent),
            seed=team_seed,
            radar_policy=policy,
        ))
    if swap:
        agents.reverse()
    return agents


def _level_speed(spec: AircraftSpec, z: float, throttle: float = 1.0,
                 ab: float = 0.0) -> float:
    """某高度、某功率设置下的平飞极速（m/s）：解推力 = 零升阻力。"""
    lo, hi = 30.0, 1500.0
    for _ in range(50):
        v = 0.5 * (lo + hi)
        m = mach_number(v, z)
        thrust = spec.thrust_at(throttle, ab, z, m)
        drag = 0.5 * air_density(z) * v * v * cd0_at(m, spec.cd0) * spec.wing_area
        if thrust > drag:
            lo = v
        else:
            hi = v
    return 0.5 * (lo + hi)


def _spec_line(spec: AircraftSpec, z: float = 12000.0) -> str:
    v_top = _level_speed(spec, z, 1.0, 1.0)
    # 不开加力的军用推力平飞极速：Su-57 在这里能维持 Mach 1.3 超巡。
    v_mil = _level_speed(spec, z, 1.0, 0.0)
    m_top = mach_number(v_top, z)
    m_mil = mach_number(v_mil, z)
    return (
        f"{spec.name}: 翼载 {spec.wing_loading:6.0f} N/m2   "
        f"推重比 {spec.thrust_max / spec.weight:4.2f}   "
        f"失速 {spec.v_stall:5.1f} m/s   "
        f"巡航 M{spec.cruise_mach:.2f}   "
        f"12km 满加力 M{m_top:.2f} ({v_top:.0f} m/s)   "
        f"军推 M{m_mil:.2f}   "
        f"过载上限 {spec.n_struct:.1f}g   "
        f"RCS 前/侧/后 {spec.rcs_front or spec.rcs:.2f}/"
        f"{spec.rcs_side or spec.rcs:.2f}/{spec.rcs_back or spec.rcs:.2f} m2   "
        f"雷达 {spec.radar_range / 1000.0:.0f} km @1m2"
    )


def print_specs(scenario: Scenario | None = None) -> None:
    sc = scenario or resolve_scenario()
    print("=" * 76)
    print(f"场景: {sc.id}（{sc.source}）")
    print(
        f"ISA 大气: {air_density(0.0):.3f} kg/m3 @0m   "
        f"{air_density(9000.0):.3f} @9km   "
        f"{air_density(15000.0):.3f} @15km   "
        f"{air_density(20000.0):.3f} @20km"
    )
    for team in sc.teams:
        spec = team.aircraft.spec
        print(f"[{team.name}] 雷达策略 {team.radar_policy}")
        _print_spec(spec)
    print("=" * 76)


def _print_spec(spec: AircraftSpec) -> None:
    print(_spec_line(spec))
    msl = spec.missile
    if msl is not None:
        print(
            f"          近距弹 {spec.missile_count} × {msl.name} "
            f"({msl.mass:.1f} kg)   满载全重 {spec.mass:.0f} kg   "
            f"导引头 ±{math.degrees(msl.seeker_fov):.0f}° / "
            f"{msl.seeker_range / 1000.0:.0f} km   锁定 {msl.lock_time:.1f} s"
        )
    rsl = spec.radar_missile
    if rsl is not None:
        mode = "主动雷达" if rsl.is_arh else "半主动雷达"
        print(
            f"          中距弹 {spec.radar_missile_count} × {rsl.name} "
            f"({rsl.mass:.1f} kg, {mode})   "
            f"射程 {rsl.min_range / 1000.0:.0f}–{rsl.max_range / 1000.0:.0f} km   "
            f"机载雷达 {spec.radar_range / 1000.0:.0f} km / "
            f"弹上截获 {rsl.active_range / 1000.0:.0f} km   锁定 {rsl.lock_time:.1f} s"
        )
        if rsl.is_arh:
            print(
                f"          主动导引头截获 {rsl.active_range / 1000.0:.0f} km（"
                f"对 1 m2 目标），截获后发射后不管"
            )
    print(
        f"          干扰物 箔条 {spec.chaff_count} 发 / 热诱弹 {spec.flare_count} 发"
    )
    print(
        f"          发动机 军推 {spec.mil_thrust() / 1000.0:.1f} kN / "
        f"最大加力 {spec.thrust_max / 1000.0:.1f} kN   燃油 {spec.fuel_capacity:.0f} kg"
    )


def _simulate(
    dof: int,
    seconds: float,
    baseline: bool,
    trace: bool = False,
    swap: bool = False,
    y_offset: float | None = None,
    seed: int | None = None,
    f16_radar: str | None = None,
    su57_radar: str | None = None,
    awacs: bool | None = None,
    scenario: Scenario | None = None,
    radar_policy: dict[str, str] | None = None,
):
    """跑一局，返回 (world, agents, 原语统计)。trace=True 时逐秒打印。"""
    sc = scenario or resolve_scenario()
    world = build_world(
        seconds, dof, swap=swap, y_offset=y_offset, awacs=awacs, scenario=sc,
    )
    agents = build_agents(
        baseline, swap=swap, seed=seed,
        f16_radar=f16_radar, su57_radar=su57_radar,
        scenario=sc, radar_policy=radar_policy,
    )
    steps_per_decision = round(1.0 / (sc.decision_hz * sc.physics_dt))
    steps_per_second = max(1, round(1.0 / sc.physics_dt))
    usage = [Counter() for _ in range(len(sc.teams))]
    actions = [Action() for _ in range(len(sc.teams))]
    steps = 0

    while world.result is None:
        if steps % steps_per_decision == 0:
            actions = [ag.act(world.observe(i)) for i, ag in enumerate(agents)]
            for i, ag in enumerate(agents):
                usage[i][getattr(ag, "primitive", "PURE")] += 1
        world.step(actions)
        steps += 1

        if trace and steps % steps_per_second == 0:
            r, b = world.acs
            dist = math.sqrt((r.x - b.x) ** 2 + (r.y - b.y) ** 2 + (r.z - b.z) ** 2)
            print(
                f"t={world.t:5.1f}s d={dist:5.0f}m  "
                f"{r.spec.name} alt={r.z:5.0f} v={r.v:5.1f} g={r.n:4.1f} hp={r.hp:5.1f} "
                f"msl={r.missiles}+{r.radar_missiles} "
                f"[{getattr(agents[0], 'situation', '-'):9s}/{getattr(agents[0], 'primitive', 'PURE'):9s}]  |  "
                f"{b.spec.name} alt={b.z:5.0f} v={b.v:5.1f} g={b.n:4.1f} hp={b.hp:5.1f} "
                f"msl={b.missiles}+{b.radar_missiles} "
                f"[{getattr(agents[1], 'situation', '-'):9s}/{getattr(agents[1], 'primitive', 'PURE'):9s}]"
            )

    return world, agents, usage


def _print_usage(usage, world) -> None:
    for i, ac in enumerate(world.acs):
        top = ", ".join(f"{k} {v}" for k, v in usage[i].most_common())
        print(f"  {ac.spec.name} 原语: {top}")


def run_headless(
    seconds: float, baseline: bool, dof: int = 6, seed: int | None = None,
    f16_radar: str | None = None, su57_radar: str | None = None,
    awacs: bool | None = None, scenario: Scenario | None = None,
    radar_policy: dict[str, str] | None = None,
) -> None:
    sc = scenario or resolve_scenario()
    print_specs(sc)
    world, _agents, usage = _simulate(
        dof, seconds, baseline, trace=True, seed=seed,
        f16_radar=f16_radar, su57_radar=su57_radar, awacs=awacs,
        scenario=sc, radar_policy=radar_policy,
    )
    print()
    print(world.summary())
    _print_usage(usage, world)


def run_ab(seconds: float, scenario: Scenario | None = None,
           radar_policy: dict[str, str] | None = None) -> None:
    """A/B：同一套战术 Agent，只切换内核自由度。"""
    sc = scenario or resolve_scenario()
    print_specs(sc)
    print("A/B：机体参数、Agent 个性、初始几何完全一致，只切换内核 dof")
    dofs = (3, 6)
    # 固定 seed：两个内核用同一套开局随机占位，差异只来自自由度。
    runs = {
        dof: _simulate(
            dof, seconds, baseline=False, seed=0,
            scenario=sc, radar_policy=radar_policy,
        )
        for dof in dofs
    }

    cols = {}
    for dof in dofs:
        world, _agents, _usage = runs[dof]
        r, b = world.acs
        cols[dof] = {
            "result": world.result or "running",
            "time": f"{world.t:.2f} s",
            "reason": world.reason_code or "-",
            "hp": f"{r.hp:.1f} / {b.hp:.1f}",
            "alt": f"{world.stats['min_z']:.0f} - {world.stats['max_z']:.0f} m",
            "v": f"{world.stats['min_v']:.1f} - {world.stats['max_v']:.1f} m/s",
            "g": f"{world.stats['max_n']:.1f} g",
            "rounds": f"{world.stats['rounds_fired']:.0f}",
            "msl": (
                f"{world.stats['missiles_fired']:.0f} 发 / "
                f"{world.stats['missile_hits']} 中"
            ),
            "energy": f"{0.5 * r.v ** 2:.0f} / {0.5 * b.v ** 2:.0f} J/kg",
        }

    print()
    print(f"{'指标':<12}" + "".join(f"{f'{dof}DOF':<30}" for dof in dofs))
    print("-" * (12 + 30 * len(dofs)))
    labels = (
        ("结局", "result"),
        ("结束时间", "time"),
        ("结束原因", "reason"),
        ("剩余 HP", "hp"),
        ("高度范围", "alt"),
        ("速度范围", "v"),
        ("最大过载", "g"),
        ("发射弹数", "rounds"),
        ("格斗弹", "msl"),
        ("剩余动能 ½v²", "energy"),
    )
    for label, key in labels:
        print((f"{label:<12}" + "".join(f"{cols[dof][key]:<30}" for dof in dofs)).rstrip())

    print()
    for dof in dofs:
        print(f"[{dof}DOF]")
        _print_usage(runs[dof][2], runs[dof][0])


def _winner_frame(world) -> str:
    """把内核的 red/blue 胜负映射回机体名，站位对换后仍可比。"""
    if world.result in (None, "draw"):
        return "draw"
    idx = 0 if world.result == "red" else 1
    return world.acs[idx].spec.name


def run_ab_suite(seconds: float, scenario: Scenario | None = None,
                 radar_policy: dict[str, str] | None = None) -> None:
    """镜像对换 + 多初始偏移：2 内核 × 2 站位 × 3 偏移 = 12 局。

    站位对换时，机体参数和 Agent 个性一起换边，用来区分胜负到底来自
    “机体性能”还是“谁先占住初始位置”。
    """
    sc = scenario or resolve_scenario()
    print_specs(sc)
    print("矩阵实验：dof × 机体站位(左位) × 初始横向偏移")
    dofs = (3, 6)
    rows = []
    for dof in dofs:
        for swap in (False, True):
            for yoff in (-200.0, 0.0, 200.0):
                world, _agents, _usage = _simulate(
                    dof, seconds, baseline=False, swap=swap, y_offset=yoff,
                    seed=0, scenario=sc, radar_policy=radar_policy,
                )
                left, right = world.acs
                rows.append(
                    {
                        "dof": dof,
                        "swap": swap,
                        "yoff": yoff,
                        "left": left.spec.name,
                        "right": right.spec.name,
                        "winner": _winner_frame(world),
                        "time": world.t,
                        "reason": world.reason_code or "-",
                        "hp": (
                            f"{100.0 * left.hp / left.spec.hp_max:.0f}% / "
                            f"{100.0 * right.hp / right.spec.hp_max:.0f}%"
                        ),
                        "alt": (
                            f"{world.stats['min_z']:.0f}-{world.stats['max_z']:.0f}"
                            if dof >= 3 else "-"
                        ),
                        "g": world.stats["max_n"],
                        "msl": (
                            f"{world.stats['missiles_fired']:.0f}/"
                            f"{world.stats['missile_hits']}"
                        ),
                    }
                )

    print()
    print(
        f"{'doF':<4}{'左位机体':<10}{'y偏移':>7}{'胜者':>7}{'用时':>8}"
        f"{'原因':>10}{'HP 左/右':>15}{'高度(m)':>13}{'maxG':>6}{'弹发/中':>9}"
    )
    print("-" * 97)
    for r in rows:
        print(
            f"{r['dof']:<4}{r['left']:<10}{r['yoff']:>+7.0f}{r['winner']:>7}"
            f"{r['time']:>7.2f}s{r['reason']:>10}{r['hp']:>15}"
            f"{r['alt']:>13}{r['g']:>6.1f}{r['msl']:>9}"
        )

    print()
    print("汇总：按内核 / 机体计胜场（每组 6 局）")
    for dof in dofs:
        sub = [r for r in rows if r["dof"] == dof]
        wins = Counter(r["winner"] for r in sub if r["winner"] != "draw")
        draw = sum(1 for r in sub if r["winner"] == "draw")
        codes = Counter(r["reason"] for r in sub)
        parts = " / ".join(f"{name} {count} 胜" for name, count in wins.items())
        print(
            f"  {dof}DOF: {parts} / 平 {draw}"
            f"   结束原因: {', '.join(f'{k} {v}' for k, v in codes.items())}"
        )

    print()
    print("镜像一致性：同一 (dof, y偏移) 下对换站位后胜者是否跟着机体走")
    for dof in dofs:
        for yoff in (-200.0, 0.0, 200.0):
            a = next(
                r for r in rows
                if r["dof"] == dof and r["yoff"] == yoff and not r["swap"]
            )
            b = next(
                r for r in rows
                if r["dof"] == dof and r["yoff"] == yoff and r["swap"]
            )
            tag = "一致" if a["winner"] == b["winner"] else "翻转"
            print(
                f"  {dof}DOF y={yoff:+4.0f}: 左{a['left']}→{a['winner']:<5}"
                f"  左{b['left']}→{b['winner']:<5}  {tag}"
            )


def run_radar_compare(seconds: float, seed: int | None = 3,
                      scenario: Scenario | None = None) -> None:
    """对“被观察方”三种雷达策略各跑一局，对比辐射/静默的战术后果。

    always：雷达常开，能拿火控发中距弹，但一开机就被对方 RWR 发现；
    silent：全程静默，只靠光电吊舱/IRST/目视/预警机，拿不到火控级接触，
            不能发射中距弹，但对方 RWR 全程无告警；
    auto：默认静默，目标进入 40 km 迎头且还有中距弹时短时开机，
          中制导段保持，主动弹末段截获后关机。

    被观察方优先取场景里 F-16V 所在的队伍（旧 CLI 语义），没有则取第二队。
    固定 seed，三次模拟除雷达策略外开局占位完全一致，差异只来自辐射。
    """
    sc = scenario or resolve_scenario()
    focus = next(
        (t for t in sc.teams if t.aircraft.id == "F-16V"), sc.teams[-1],
    )
    other = next(t for t in sc.teams if t is not focus)
    print_specs(sc)
    print(
        f"{focus.aircraft.spec.name} 雷达策略对比"
        f"（{other.aircraft.spec.name} 雷达按场景配置，seed={seed}）"
    )
    rows = []
    for policy in ("always", "silent", "auto"):
        world, _agents, _usage = _simulate(
            6, seconds, baseline=False, seed=seed,
            radar_policy={focus.name: policy}, scenario=sc,
        )
        names = [ac.spec.name for ac in world.acs]
        fi = names.index(focus.aircraft.spec.name)
        si = 1 - fi

        def fmt(value: float | None) -> str:
            return "—" if value is None else f"{value:.1f}s"

        left, right = world.acs
        rows.append({
            "policy": policy,
            "winner": _winner_frame(world),
            "time": world.t,
            "reason": world.reason_code or "-",
            "fc": fmt(world.first_fire_contact_t[fi]),
            "fire": fmt(world.first_fire_t[fi]),
            "su_fc": fmt(world.first_fire_contact_t[si]),
            "su_fire": fmt(world.first_fire_t[si]),
            "msl": f"{world.stats['missiles_fired']:.0f}",
            "hits": f"{world.stats['missile_hits']}",
            "hp": (
                f"{100.0 * left.hp / left.spec.hp_max:.0f}% / "
                f"{100.0 * right.hp / right.spec.hp_max:.0f}%"
            ),
        })

    focus_name = focus.aircraft.spec.name
    other_name = other.aircraft.spec.name
    print()
    print(
        f"{'雷达策略':<12}{'胜者':>7}{'用时':>8}{'原因':>10}"
        f"{focus_name[:6] + '火控':>10}{focus_name[:6] + '开火':>10}"
        f"{other_name[:6] + '火控':>11}{other_name[:6] + '开火':>11}"
        f"{'弹发':>6}{'命中':>6}{'HP 左/右':>14}"
    )
    print("-" * 101)
    for r in rows:
        print(
            f"{r['policy']:<12}{r['winner']:>7}{r['time']:>7.2f}s{r['reason']:>10}"
            f"{r['fc']:>10}{r['fire']:>10}{r['su_fc']:>11}"
            f"{r['su_fire']:>11}{r['msl']:>6}{r['hits']:>6}{r['hp']:>14}"
        )




RADAR_CHOICES = ("always", "silent", "auto")


def _parse_radar_overrides(
    parser: argparse.ArgumentParser, items: list[str] | None,
) -> dict[str, str]:
    """把 `TEAM=POLICY` 列表解析成覆盖表；team 可写队伍名或机体 id。"""
    overrides: dict[str, str] = {}
    for item in items or ():
        team, sep, policy = item.partition("=")
        if not sep or not team.strip():
            parser.error(f"--radar-policy 格式应为 TEAM=POLICY：{item}")
        if policy.strip() not in RADAR_CHOICES:
            parser.error(f"--radar-policy 的策略只能是 {RADAR_CHOICES}：{item}")
        overrides[team.strip()] = policy.strip()
    return overrides



def main() -> None:
    parser = argparse.ArgumentParser(description="Dogfight M2 6DOF + 战术层")
    parser.add_argument(
        "--headless", action="store_true", help="只跑一局并打印结果（默认行为）",
    )
    parser.add_argument("--baseline", action="store_true", help="双方使用纯追踪基线")
    parser.add_argument(
        "--dof", type=int, choices=(2, 3, 6), default=6,
        help="内核自由度：默认 6（刚体）；2/3 仅用于 A/B 对照实验",
    )
    parser.add_argument("--ab", action="store_true", help="同一套 Agent 跑 3DOF/6DOF A/B 对比")
    parser.add_argument("--ab-suite", action="store_true", help="镜像对换 + 多初始偏移的 12 局矩阵实验")
    parser.add_argument(
        "--scenario", default=DEFAULT_SCENARIO,
        help=f"场景名或 YAML 路径（默认 {DEFAULT_SCENARIO}）",
    )
    parser.add_argument(
        "--config-dir", default=None,
        help="配置根目录（默认仓库内 configs/）",
    )
    parser.add_argument(
        "--radar-policy", action="append", metavar="TEAM=POLICY",
        help="按队伍/机体覆盖雷达策略，可重复：red=auto blue=silent",
    )
    parser.add_argument(
        "--seconds", type=float, default=None,
        help="对局时长上限（默认取场景配置的 time_limit_s）",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="固定开局随机占位的随机种子（复现实验用；默认每次启动随机）",
    )
    # 旧机型别名：保留一版，仅作兼容；新场景请用 --radar-policy。
    parser.add_argument(
        "--f16-radar", choices=RADAR_CHOICES, default=None,
        help="[弃用] F-16V 雷达策略，等价于 --radar-policy F-16V=...",
    )
    parser.add_argument(
        "--su57-radar", choices=RADAR_CHOICES, default=None,
        help="[弃用] Su-57 雷达策略，等价于 --radar-policy SU-57=...",
    )
    parser.add_argument(
        "--compare-radar", action="store_true",
        help="场景中被观察方的三种雷达策略各跑一局并对比（默认 seed=3）",
    )
    parser.add_argument(
        "--awacs", action=argparse.BooleanOptionalAction, default=None,
        help="是否启用预警机（默认取场景配置；--no-awacs 关闭）",
    )
    args = parser.parse_args()

    sc = resolve_scenario(args.scenario, args.config_dir)
    seconds = sc.time_limit if args.seconds is None else args.seconds
    radar_policy = _parse_radar_overrides(parser, args.radar_policy)
    if args.f16_radar:
        print("提示：--f16-radar 是旧别名，等价于 --radar-policy F-16V=..."
              "（或使用队伍名）")
        radar_policy["F-16V"] = args.f16_radar
    if args.su57_radar:
        print("提示：--su57-radar 是旧别名，等价于 --radar-policy SU-57=..."
              "（或使用队伍名）")
        radar_policy["SU-57"] = args.su57_radar
    if args.compare_radar:
        run_radar_compare(
            seconds, 3 if args.seed is None else args.seed, scenario=sc,
        )
    elif args.ab_suite:
        run_ab_suite(seconds, scenario=sc, radar_policy=radar_policy)
    elif args.ab:
        run_ab(seconds, scenario=sc, radar_policy=radar_policy)
    else:
        # 唯一可视化入口是 three.js 观战端；无参数时直接在终端跑一局。
        run_headless(
            seconds, args.baseline, args.dof, args.seed,
            None, None, args.awacs, scenario=sc, radar_policy=radar_policy,
        )
        if not args.headless:
            print("\n浏览器观战：python -m dogfight.web  →  http://127.0.0.1:8765/")


# 旧脚本/测试仍按模块常量导入机体与武器；这里按名字回到配置注册表，
# 避免在入口模块复制一份参数。新代码应直接用 build_world/build_agents
# 或 dogfight.config.load_bundle。
_LEGACY_ATTRS = {
    "SU57_SPEC": ("aircraft", "SU-57"),
    "F16V_SPEC": ("aircraft", "F-16V"),
    "SU57_AGENT": ("agent", "SU-57"),
    "F16V_AGENT": ("agent", "F-16V"),
    "R73": ("weapon", "R-73"),
    "AIM9X": ("weapon", "AIM-9X"),
    "R77_1": ("weapon", "R-77-1"),
    "AIM120C": ("weapon", "AIM-120C"),
}


def __getattr__(name: str):
    target = _LEGACY_ATTRS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    kind, key = target
    bundle = get_bundle()
    if kind == "aircraft":
        return bundle.aircraft[key].spec
    if kind == "agent":
        return dataclasses.asdict(bundle.aircraft[key].agent)
    return bundle.weapons[key]


if __name__ == "__main__":
    main()
