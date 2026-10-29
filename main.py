"""M2 入口：6DOF（默认）/3DOF 内核 + 几何态势/机动原语战术层。

用法（dogfight conda 环境）：
    python main.py                # pygame 观战（默认 6DOF）
    python main.py --headless     # 无头跑一局，打印过程与结果
    python main.py --baseline     # 双方都用纯追踪基线，做对照
"""
from __future__ import annotations

import argparse
import math
from collections import Counter

from agents import PurePursuit, TacticalAgent
from sim import (
    Action, AircraftSpec, MissileSpec, World, air_density, rcs_scale,
    cd0_at, mach_number, body_axes, nose_direction,
)

PHYSICS_DT = 1.0 / 120.0
DECIDE_HZ = 10
STEPS_PER_DECISION = round(1.0 / (DECIDE_HZ * PHYSICS_DT))
# 初始间距：双方各距空域中心 24000 m，总间距 48 km。
# 主动弹（R-77-1 / AIM-120C）的载机雷达跟踪距离约 35 km，按隐身缩放后
# 迎头有效射程不超过约 36 km；48 km 的开局让双方都在弹包线外，还留下
# 约 10 km 的余量用于开局抢位（预警机已提供对方初始位置，模拟自带全知观测）。
INITIAL_SPACING = 24000.0
# 高 G 机动标注门限：超过人体可持续耐受（6 g）即视为高 G 机动。
# 高空升力受限时（9–12 km、300 m/s 量级）可用过载本身只有 4–5 g，
# 因此执行高 G 原语且拉到当地可用过载的 75% 也一并标注，否则轨迹图上
# 永远看不到高 G 点。
HIGH_G_MARK = 6.0
HIGH_G_REL = 0.75


# 近距格斗弹公开参数：
# R-73：105 kg，2.9 m / 0.17 m，战斗部 7.4 kg，最大过载约 40 g，
#       导引头离轴 ±45°，最小发射距离 300 m，R-73E 射程 30 km。
# AIM-9X：84.4 kg，3.02 m / 0.13 m，最大过载约 50 g，
#       离轴 ±90°，射程 16 km（Block II 更远）。
# 推进段/阻力按量级取值，命中完全由导引头视场、舵面可用过载和能量决定。
# 锁定距离仍按导引头性能单独设定，射程只限制发射后的最大飞行距离。
# cd0_wave 取 1.2：早期用 2.6 是为了压低峰值马赫，但总阻力系数
# （cd0×(1+wave)）因此达到 2.1–3.3，导弹飞到末段只剩亚声速，目标一个
# 横转就能把它拖掉。真实导弹波阻远小于这个量级，改回 1.2 后 15 km 迎头
# 末段存速从 M2.5 提到 M3.0；R-77-1 的推力同步下调到 26 kN，峰值仍保持 M4.1。
R73 = MissileSpec(
    name="R-73", mass=105.0, length=2.90, diameter=0.17,
    max_g=40.0, seeker_fov=math.radians(45.0),
    gimbal_fov=math.radians(60.0), seeker_range=10000.0,
    lock_time=0.8, min_range=300.0, max_range=30000.0,
    thrust=12000.0, burn_time=5.0, burnout_mass=65.0,
    cd0=0.9, cd0_wave=1.2, cn_max=8.0, warhead_damage=100.0, lethal_radius=9.0,
    life=15.0, cooldown=1.5, pn_gain=4.0, autopilot_tau=0.12,
)

AIM9X = MissileSpec(
    name="AIM-9X", mass=84.4, length=3.02, diameter=0.127,
    max_g=50.0, seeker_fov=math.radians(90.0),
    gimbal_fov=math.radians(90.0), seeker_range=12000.0,
    lock_time=0.5, min_range=300.0, max_range=16000.0,
    thrust=10800.0, burn_time=4.0, burnout_mass=50.0,
    cd0=0.9, cd0_wave=1.2, cn_max=10.0, warhead_damage=110.0, lethal_radius=8.0,
    life=15.0, cooldown=1.5, pn_gain=4.0, autopilot_tau=0.12,
)

# 主动雷达制导中距弹（发射后不管）：中段靠载机雷达/数据链修正，
# 飞到末段由弹上主动雷达导引头自主截获（pitbull），此后载机可以转向脱离。
# R-77-1（изделие 170-1）：Su-57/Su-35 的主力中距弹，190 kg，3.60 m / 0.20 m，
#        战斗部 22 kg，最大过载约 40 g，高空迎头射程约 110 km，主动截获约 20 km。
# AIM-120C-7 AMRAAM：F-16V Block 70 的主力中距弹，152 kg，3.66 m / 0.178 m，
#        战斗部 20 kg，最大过载约 40 g，射程约 105 km，主动截获约 18–20 km。
# seeker_range 在这里表示“载机雷达能稳定跟踪并装订诸元”的距离，
# 会随目标 RCS 按四次方根缩放；active_range 是弹上导引头的截获距离。
R77_1 = MissileSpec(
    name="R-77-1", mass=190.0, length=3.60, diameter=0.20,
    max_g=40.0, seeker_fov=math.radians(60.0),
    gimbal_fov=math.radians(60.0), seeker_range=35000.0,
    lock_time=1.5, min_range=1500.0, max_range=80000.0,
    thrust=26000.0, burn_time=8.0, burnout_mass=100.0,
    cd0=0.55, cd0_wave=1.2, cn_max=8.0, warhead_damage=110.0, lethal_radius=11.0,
    life=50.0, cooldown=3.0, pn_gain=4.0, autopilot_tau=0.18,
    guidance="ARH", loft=True, active_range=20000.0,
)

AIM120C = MissileSpec(
    name="AIM-120C", mass=152.0, length=3.66, diameter=0.178,
    max_g=40.0, seeker_fov=math.radians(60.0),
    gimbal_fov=math.radians(60.0), seeker_range=35000.0,
    lock_time=1.2, min_range=1500.0, max_range=75000.0,
    thrust=21000.0, burn_time=7.0, burnout_mass=80.0,
    cd0=0.55, cd0_wave=1.2, cn_max=9.0, warhead_damage=105.0, lethal_radius=10.0,
    life=45.0, cooldown=3.0, pn_gain=4.2, autopilot_tau=0.18,
    guidance="ARH", loft=True, active_range=18000.0,
)


# 机体参数：公开数据取“半油 + 中距弹”的作战重量。
# 隐身只建模前向 RCS 对雷达探测/锁定距离的影响（红外与机炮不受影响）。
# Su-57：空重 18,500 / 正常起飞 26,700 kg，机翼 78.8 m2，2×142 kN。
# 极速 Mach 2.0（约 2135 km/h），不开加力超巡 Mach 1.3（约 1400 km/h），
# 巡航 Mach 0.95。
# 挂载：机腹弹舱 2 × R-77-1 中距弹 + 4 × R-73 近距弹，整机质量按满载计。
# 前向 RCS 取公开估算的量级（0.1–1 m²），这里用 0.5 m²。
SU57_SPEC = AircraftSpec(
    name="SU-57", mass=24500 + 4 * R73.mass + 2 * R77_1.mass,
    wing_area=78.8, cl_max=1.6, rcs=0.5,
    # 展弦比按真实机翼几何（翼展 14.1 m / 面积 78.8 m²）取 2.52，不再沿用 3.0；
    # 升力线斜率同步抬到 3.85，否则失速迎角处凑不出 CLmax。
    n_struct=9.0, thrust_max=284000.0, cd0=0.040, aspect_ratio=2.52, oswald=0.80,
    cl_alpha=3.85,
    # AL-41F1：单台军用推力 93 kN、最大加力 142 kN，双发合计 186 / 284 kN。
    # 巡航速度取 M0.95（不开加力超巡则可达 M1.3），燃油按半箱 5500 kg 计。
    thrust_mil=186000.0, cruise_mach=0.95, fuel_capacity=5500.0,
    # 推力衰减按公开升限/速度锚点标定：18 km 起推力悬崖，
    # 使 Ps 在 20 km 附近穿过 0，25 km 仍能维持 M1.1 量级。
    lapse_exp=0.65, ram_r0=0.15, ram_r1=0.20,
    derate_z=18000.0, derate_H=6000.0, eta_install=0.95,
    hp_max=110.0, gun_range=1000.0, gun_dps=30.0, ammo_max=150.0,
    rounds_per_sec=25.0, length=20.1,
    # 30 mm 机炮（GSh-30-1 量级）：初速 900 m/s，弹重 390 g。
    muzzle_speed=900.0, bullet_mass=0.390, bullet_diameter=0.030, gun_spread=0.003,
    # AL-41F1 的 3D 矢量喷口：低速大迎角下补充舵面权限，
    # 可维持 55° 过失速迎角，对应眼镜蛇这类机动。
    # 喷流吹过机翼还带来约 0.30 的等效 CL 增量（按油门/加力开度缩放），
    # 使盘旋可用升力上限从 CL≈1.50 抬到 ≈1.80，最小回旋半径明显小于
    # 同参数的常规喷口机型——这是矢量推进在近距格斗里的主要收益。
    tvc=math.radians(15.0), tvc_alpha_max=math.radians(55.0),
    tvc_cl_bonus=0.30,
    missile=R73, missile_count=4,
    radar_missile=R77_1, radar_missile_count=2,
    # Su-57 的干扰物：3 组 UV-50-01 发射器（每组 14 发 50 mm 弹），
    # 共 42 发，按 24 箔条 + 18 热诱弹的混合装载。
    chaff_count=24, flare_count=18,
)

# F-16V Block 70：空重 9,207 / 最大起飞 21,772 kg，机翼 27.87 m2，1×129 kN。
# 极速 Mach 2.0–2.05（约 2120–2175 km/h），巡航 Mach 0.8（11 km 约 849 km/h）。
# 挂载：翼尖导轨 2 × AIM-9X，翼下 2 × AIM-120C 中距弹 + 2 × AIM-9X，共 6 枚。
# F-16V 没有隐身修形，前向 RCS 取 1.2 m²。
F16V_SPEC = AircraftSpec(
    # 标定重量 11,429 kg = 半油 + 6 枚弹的作战重量。
    name="F-16V", mass=10787.4 + 4 * AIM9X.mass + 2 * AIM120C.mass,
    wing_area=27.87, cl_max=1.5, rcs=1.2,
    # cd0 由 0.050 下调到 0.039：旧值在 9 km/M0.9 把推力全吃掉，
    # 既飞不到公开的 M1.6，也和“推力比 0.45–0.55”的锚点矛盾。
    n_struct=9.0, thrust_max=129000.0, cd0=0.039, aspect_ratio=3.56, oswald=0.85,
    # F110-GE-129：军用推力 76.3 kN、最大加力 129 kN；巡航 M0.8，半箱油 1,580 kg。
    thrust_mil=76300.0, cruise_mach=0.80, fuel_capacity=1580.0,
    # 推力衰减标定：13 km 起降额、尺度 9 km，
    # 使 9 km Vmax ≈ M1.56、12 km ≈ M1.73、18 km ≈ M1.12、升限约 16.5 km。
    lapse_exp=0.90, ram_r0=0.12, ram_r1=0.20,
    derate_z=13000.0, derate_H=9000.0, eta_install=0.95,
    hp_max=100.0, gun_range=1000.0, gun_dps=40.0, ammo_max=511.0,
    rounds_per_sec=110.0, length=15.06,
    # M61A1 20 mm：初速 1,030 m/s，弹重 102 g。
    muzzle_speed=1030.0, bullet_mass=0.102, bullet_diameter=0.020, gun_spread=0.003,
    missile=AIM9X, missile_count=4,
    radar_missile=AIM120C, radar_missile_count=2,
    # F-16V Block 70 的 AN/ALE-47 投放器共 90 发双光谱弹，
    # 按 30 箔条 + 60 热诱弹配置。
    chaff_count=30, flare_count=60,
)


# 苏-57 是双发重型，偏能量战；F-16V 单发轻型，偏角度战。
# 镜像实验时这对参数跟着机体走，保证“Airframe + 个性”作为一个整体被对换。
SU57_AGENT = dict(aggression=0.55, energy_pref=0.65)
F16V_AGENT = dict(aggression=0.75, energy_pref=0.40)


def build_world(
    time_limit: float = 300.0,
    dof: int = 6,
    swap: bool = False,
    y_offset: float = 100.0,
) -> World:
    specs = [F16V_SPEC, SU57_SPEC] if swap else [SU57_SPEC, F16V_SPEC]
    return World(
        specs,
        dt=PHYSICS_DT,
        arena=80000.0,
        time_limit=time_limit,
        dof=dof,
        y_offset=y_offset,
        x_offset=INITIAL_SPACING,
        z0=9000.0,
        z_floor=1500.0,
        z_ceiling=30000.0,
    )


def build_agents(
    baseline: bool = False, swap: bool = False, seed: int | None = None,
) -> list:
    """seed 只用于实验复现；默认 None 表示每次启动随机占位。"""
    if baseline:
        return [PurePursuit(), PurePursuit()]
    su57 = TacticalAgent(**SU57_AGENT, seed=seed)
    f16_seed = None if seed is None else seed + 1
    f16v = TacticalAgent(**F16V_AGENT, seed=f16_seed)
    return [f16v, su57] if swap else [su57, f16v]


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
        f"RCS {spec.rcs:.2f} m2 (x{rcs_scale(spec.rcs):.2f} 探测距离)"
    )


def print_specs() -> None:
    print("=" * 76)
    print(
        f"ISA 大气: {air_density(0.0):.3f} kg/m3 @0m   "
        f"{air_density(9000.0):.3f} @9km   "
        f"{air_density(15000.0):.3f} @15km   "
        f"{air_density(20000.0):.3f} @20km"
    )
    for spec in (SU57_SPEC, F16V_SPEC):
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
                f"载机雷达跟踪 {rsl.seeker_range / 1000.0:.0f} km   锁定 {rsl.lock_time:.1f} s"
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
    print("=" * 76)


def _simulate(
    dof: int,
    seconds: float,
    baseline: bool,
    trace: bool = False,
    swap: bool = False,
    y_offset: float = 100.0,
    seed: int | None = None,
):
    """跑一局，返回 (world, agents, 原语统计)。trace=True 时逐秒打印。"""
    world = build_world(seconds, dof, swap=swap, y_offset=y_offset)
    agents = build_agents(baseline, swap=swap, seed=seed)
    usage = [Counter(), Counter()]
    actions = [Action(), Action()]
    steps = 0

    while world.result is None:
        if steps % STEPS_PER_DECISION == 0:
            actions = [ag.act(world.observe(i)) for i, ag in enumerate(agents)]
            for i, ag in enumerate(agents):
                usage[i][getattr(ag, "primitive", "PURE")] += 1
        world.step(actions)
        steps += 1

        if trace and steps % 120 == 0:
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


def _print_usage(usage) -> None:
    for i, spec in enumerate((SU57_SPEC, F16V_SPEC)):
        top = ", ".join(f"{k} {v}" for k, v in usage[i].most_common())
        print(f"  {spec.name} 原语: {top}")


def run_headless(
    seconds: float, baseline: bool, dof: int = 6, seed: int | None = None,
) -> None:
    print_specs()
    world, _agents, usage = _simulate(
        dof, seconds, baseline, trace=True, seed=seed
    )
    print()
    print(world.summary())
    _print_usage(usage)


def run_ab(seconds: float) -> None:
    """A/B：同一套战术 Agent，只切换内核自由度。"""
    print_specs()
    print("A/B：机体参数、Agent 个性、初始几何完全一致，只切换内核 dof")
    dofs = (3, 6)
    # 固定 seed：两个内核用同一套开局随机占位，差异只来自自由度。
    runs = {dof: _simulate(dof, seconds, baseline=False, seed=0) for dof in dofs}

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
        _print_usage(runs[dof][2])


def _winner_frame(world) -> str:
    """把内核的 red/blue 胜负映射回机体（SU57/F16），站位对换后仍可比。"""
    if world.result in (None, "draw"):
        return "draw"
    idx = 0 if world.result == "red" else 1
    return "SU57" if world.acs[idx].spec.name == SU57_SPEC.name else "F16"


def run_ab_suite(seconds: float) -> None:
    """镜像对换 + 多初始偏移：2 内核 × 2 站位 × 3 偏移 = 12 局。

    站位对换时，机体参数和 Agent 个性一起换边，用来区分胜负到底来自
    “机体性能”还是“谁先占住初始位置”。
    """
    print_specs()
    print("矩阵实验：dof × 机体站位(左位) × 初始横向偏移")
    dofs = (3, 6)
    rows = []
    for dof in dofs:
        for swap in (False, True):
            for yoff in (-200.0, 0.0, 200.0):
                world, _agents, _usage = _simulate(
                    dof, seconds, baseline=False, swap=swap, y_offset=yoff,
                    seed=0,
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
        su = sum(1 for r in sub if r["winner"] == "SU57")
        f16 = sum(1 for r in sub if r["winner"] == "F16")
        draw = sum(1 for r in sub if r["winner"] == "draw")
        codes = Counter(r["reason"] for r in sub)
        print(
            f"  {dof}DOF: SU-57 {su} 胜 / F-16V {f16} 胜 / 平 {draw}"
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


def _sample_state(world) -> list[tuple]:
    """记录采样时刻的位置与姿态（机头、机翼方向），供全局轨迹图使用。"""
    rows = []
    for ac in world.acs:
        nx, ny, nz = nose_direction(ac)
        if ac.rigid:
            wx, wy, wz = body_axes(ac.psi, ac.theta, ac.phi)[1]
        else:
            wx, wy, wz = body_axes(ac.psi, ac.gamma, ac.bank)[1]
        rows.append((world.t, ac.x, ac.y, ac.z, nx, ny, nz, wx, wy, wz))
    return rows


def save_trajectory_plot(
    samples,
    path: str,
    interval: float,
    launches=(),
    high_g=(),
) -> None:
    """全局三维轨迹图：整条航迹 + 定时采样的位置/姿态与时间标注。

    launches: (t, x, y, z, owner, name) 导弹发射点；
    high_g:  (t, x, y, z, owner) 高 G 机动点（过载超过人体持续耐受）。
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib import colors
    from matplotlib.lines import Line2D

    fig = plt.figure(figsize=(14.0, 10.0))
    ax = fig.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("viridis")
    tmax = max((s[-1][0] if s else 0.0) for s in samples) or 1.0
    norm = colors.Normalize(0.0, tmax)
    z_ref = 1500.0
    for i, (name, color, marker) in enumerate(
        (("SU-57", "#d23a2a", "o"), ("F-16V", "#2f6fd0", "^"))
    ):
        rows = samples[i]
        if not rows:
            continue
        pts = np.array([(r[1], r[2], r[3]) for r in rows])
        ax.plot(
            pts[:, 0], pts[:, 1], pts[:, 2],
            color=color, lw=1.2, alpha=0.5, label=f"{name} full track",
        )
        for t, x, y, z, nx, ny, nz, wx, wy, wz in rows:
            c = cmap(norm(t))
            # 采样点：位置圆点 + 机头箭头 + 机翼短线，颜色编码时刻。
            ax.scatter(
                [x], [y], [z], color=[c], marker=marker, s=30,
                edgecolors=color, linewidths=0.9, depthshade=False,
            )
            ax.quiver(
                x, y, z, nx, ny, nz, length=700.0, color=[c],
                arrow_length_ratio=0.30, lw=1.2,
            )
            ax.plot(
                [x - 300.0 * wx, x + 300.0 * wx],
                [y - 300.0 * wy, y + 300.0 * wy],
                [z - 300.0 * wz, z + 300.0 * wz],
                color=c, lw=1.2,
            )
            ax.plot([x, x], [y, y], [z_ref, z], color=c, lw=0.5, ls=":", alpha=0.35)
            # 采样点密集时逐点标注会互相遮挡：两机错开 20 s 标注，每 40 s 一次。
            phase = 0.0 if i == 0 else 20.0
            if abs((t - phase) % 40.0) < 2.5 or abs((t - phase) % 40.0 - 40.0) < 2.5:
                ax.text(
                    x, y, z + (320.0 if i == 0 else -380.0), f"t={t:.0f}s",
                    fontsize=7.0, color=color, ha="center", fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.65),
                )

    # 导弹发射点：金色五角星 + 型号标注；高 G 机动点：橙色菱形。
    for t, x, y, z, owner, name in launches:
        ax.scatter(
            [x], [y], [z], marker="*", s=230, color="#ffd166",
            edgecolors=("#d23a2a" if owner == 0 else "#2f6fd0"),
            linewidths=0.9, depthshade=False, zorder=6,
        )
        ax.text(
            x, y, z + 300.0, f"{name} {t:.0f}s", fontsize=6.5, color="#7a5200",
            ha="center", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.10", fc="white", ec="none", alpha=0.55),
        )
    if high_g:
        ax.scatter(
            [p[1] for p in high_g],
            [p[2] for p in high_g],
            [p[3] for p in high_g],
            marker="D", s=52, color="#ff7a45",
            edgecolors="#7a2b00", linewidths=0.8, depthshade=False, zorder=5,
        )

    all_pts = np.array([(r[1], r[2], r[3]) for s in samples for r in s])
    spans = np.ptp(all_pts, axis=0) if len(all_pts) else np.ones(3)
    sx = max(float(spans[0]), 1.0)
    sy = max(float(spans[1]), 1.0)
    sz = max(float(spans[2]), 1.0)
    # 高度方向适度夸张，否则几千米的高差在两万米的水平航程里几乎看不出来。
    ax.set_box_aspect((1.0, max(sy / sx, 0.42), max(3.2 * sz / sx, 0.30)))
    ax.set_zlim(all_pts[:, 2].min() - max(300.0, 0.25 * sz),
                all_pts[:, 2].max() + max(300.0, 0.25 * sz))
    ax.view_init(elev=24.0, azim=-62.0)

    ax.set_xlabel("x east (m)", labelpad=8)
    ax.set_ylabel("y north (m)", labelpad=8)
    ax.set_zlabel("z altitude (m)", labelpad=8)
    ax.set_title(
        f"Global flight trajectory: SU-57 vs F-16V "
        f"(attitude sampled every {interval:.0f} s)",
        pad=16,
    )
    ax.grid(True, alpha=0.22)

    handles = [
        Line2D([0], [0], color="#d23a2a", lw=1.4, label="SU-57 full track"),
        Line2D([0], [0], color="#2f6fd0", lw=1.4, label="F-16V full track"),
        Line2D([0], [0], color="0.35", marker="o", ls="", label="SU-57 sample"),
        Line2D([0], [0], color="0.35", marker="^", ls="", label="F-16V sample"),
        Line2D([0], [0], color="0.35", lw=1.2, label="arrow=nose, bar=wing"),
        Line2D([0], [0], color="0.55", lw=0.8, ls=":", label="altitude projection"),
        Line2D([0], [0], color="#ffd166", marker="*", ls="", ms=12,
               markeredgecolor="#7a5200", label="missile launch"),
        Line2D([0], [0], color="#ff7a45", marker="D", ls="", ms=7,
               markeredgecolor="#7a2b00", label="high-G maneuver"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8)

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6, pad=0.09, label="time (s)")
    ticks = np.linspace(0.0, tmax, 6)
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels(
        [f"{v:.0f}" if tmax >= 10.0 else f"{v:.1f}" for v in ticks]
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_record(
    seconds: float,
    baseline: bool,
    dof: int,
    record_path: str,
    track_path: str,
    follow: int | None = None,
    track_interval: float = 10.0,
    fps: int = 20,
    seed: int | None = None,
) -> None:
    """无窗口渲染并录制视频，同时按固定间隔采样全局轨迹。"""
    import os

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import imageio
    import pygame
    import numpy as np

    from render import Renderer

    world = build_world(seconds, dof)
    agents = build_agents(baseline, seed=seed)
    # 录制要走 SDL dummy 驱动，没有 GL 上下文，直接用 CPU 渲染保证帧可读。
    renderer = Renderer(follow=follow, follow_dist=1200.0, gpu=False)
    steps_per_frame = max(1, round(1.0 / (fps * PHYSICS_DT)))
    actions = [Action(), Action()]
    steps = 0
    samples = [[], []]
    next_sample = 0.0
    # 轨迹事件：导弹发射点与高 G 机动点。按新出现的弹 id 与过载上升沿记录。
    launches = []
    high_g = []
    seen_missiles: set[int] = set()
    high_g_state = [False, False]
    last_high_g = [-1e9, -1e9]

    writer = imageio.get_writer(
        record_path, fps=fps, codec="libx264", quality=7,
        pixelformat="yuv420p", macro_block_size=8,
    )
    try:
        while world.result is None and world.t < seconds:
            for _ in range(steps_per_frame):
                if steps % STEPS_PER_DECISION == 0:
                    actions = [ag.act(world.observe(i)) for i, ag in enumerate(agents)]
                world.step(actions)
                steps += 1
                for m in world.missiles:
                    key = id(m)
                    if m.alive and key not in seen_missiles:
                        seen_missiles.add(key)
                        launches.append(
                            (world.t, m.x, m.y, m.z, m.owner, m.spec.name)
                        )
                for i, ac in enumerate(world.acs):
                    n_avail = (
                        ac.spec.n_available(ac.v, ac.z, ac.mass) if ac.alive else 0.0
                    )
                    hi = ac.alive and (
                        ac.n >= HIGH_G_MARK
                        or (
                            getattr(agents[i], "primitive", "").startswith("HIGH_G")
                            and ac.n >= HIGH_G_REL * n_avail
                        )
                    )
                    if hi and not high_g_state[i] and world.t - last_high_g[i] > 1.5:
                        last_high_g[i] = world.t
                        high_g.append((world.t, ac.x, ac.y, ac.z, i))
                    high_g_state[i] = hi
                if world.result is not None:
                    break
            renderer.draw(world, False, agents)
            frame = pygame.surfarray.array3d(renderer.screen).swapaxes(0, 1)
            writer.append_data(np.ascontiguousarray(frame))
            while world.t >= next_sample:
                for i, row in enumerate(_sample_state(world)):
                    samples[i].append(row)
                next_sample += track_interval
    finally:
        writer.close()
        pygame.quit()

    save_trajectory_plot(samples, track_path, track_interval, launches, high_g)
    print(f"视频已保存: {record_path}")
    print(f"轨迹图已保存: {track_path}")
    print(f"轨迹事件: 导弹发射 {len(launches)} 次，高 G 机动 {len(high_g)} 次")
    print(world.summary())
    for i, spec in enumerate((SU57_SPEC, F16V_SPEC)):
        print(f"  {spec.name} 采样点: {len(samples[i])}")


def run_window(
    seconds: float, baseline: bool, dof: int = 6, seed: int | None = None,
    gpu: bool = True,
) -> None:
    import pygame

    from render import Renderer

    world = build_world(seconds, dof)
    agents = build_agents(baseline, seed=seed)
    renderer = Renderer(gpu=gpu)
    print(f"渲染后端: {'GPU / OpenGL' if renderer.gpu else 'CPU / pygame 软件渲染'}")
    clock = pygame.time.Clock()

    actions = [Action(), Action()]
    accumulator = 0.0
    steps = 0
    paused = False
    running = True

    while running:
        frame_dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    world = build_world(seconds, dof)
                    agents = build_agents(baseline, seed=seed)
                    renderer.reset()
                    actions = [Action(), Action()]
                    accumulator = 0.0
                    steps = 0
                    paused = False

        if not paused and world.result is None:
            accumulator += min(frame_dt, 0.1)
            while accumulator >= PHYSICS_DT:
                if steps % STEPS_PER_DECISION == 0:
                    actions = [ag.act(world.observe(i)) for i, ag in enumerate(agents)]
                world.step(actions)
                accumulator -= PHYSICS_DT
                steps += 1
                if world.result is not None:
                    break

        renderer.draw(world, paused, agents)

    pygame.quit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dogfight M2 6DOF + 战术层")
    parser.add_argument("--headless", action="store_true", help="不打开窗口，直接跑一局")
    parser.add_argument("--baseline", action="store_true", help="双方使用纯追踪基线")
    parser.add_argument(
        "--dof", type=int, choices=(2, 3, 6), default=6,
        help="内核自由度：默认 6（刚体）；2/3 仅用于 A/B 对照实验",
    )
    parser.add_argument("--ab", action="store_true", help="同一套 Agent 跑 3DOF/6DOF A/B 对比")
    parser.add_argument("--ab-suite", action="store_true", help="镜像对换 + 多初始偏移的 12 局矩阵实验")
    parser.add_argument("--seconds", type=float, default=300.0, help="对局时长上限（默认 300 s）")
    parser.add_argument("--record", default=None, metavar="PATH", help="无窗口渲染并录制视频（mp4）")
    parser.add_argument(
        "--track-plot", default="out/trajectory.png",
        help="全局轨迹图输出路径（仅 --record 时生成）",
    )
    parser.add_argument(
        "--follow", type=int, default=None,
        help="录制时跟随的飞机索引（0=SU-57，1=F-16V）；默认不跟随，使用两机中点视角",
    )
    parser.add_argument("--track-interval", type=float, default=10.0, help="轨迹采样间隔（秒）")
    parser.add_argument("--fps", type=int, default=20, help="录制帧率")
    parser.add_argument(
        "--seed", type=int, default=None,
        help="固定开局随机占位的随机种子（复现实验用；默认每次启动随机）",
    )
    parser.add_argument(
        "--no-gpu", action="store_true",
        help="强制使用 CPU 软件渲染（默认尝试 OpenGL，失败自动回退）",
    )
    args = parser.parse_args()

    if args.record:
        run_record(
            args.seconds, args.baseline, args.dof, args.record,
            args.track_plot, args.follow, args.track_interval, args.fps,
            args.seed,
        )
    elif args.ab_suite:
        run_ab_suite(args.seconds)
    elif args.ab:
        run_ab(args.seconds)
    elif args.headless:
        run_headless(args.seconds, args.baseline, args.dof, args.seed)
    else:
        run_window(args.seconds, args.baseline, args.dof, args.seed,
                   gpu=not args.no_gpu)


if __name__ == "__main__":
    main()
