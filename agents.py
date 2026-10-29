"""M2 Agent：四层结构 = 几何判定 → 战术 → 机动 → 控制。

分层
----
1. 几何判定（`_assess`）：把 Observation 压缩成“谁占位”的态势
   （进攻/防御/对头/中立/脱离）。
2. 战术层（`_plan` / `_tactic_*`）：按 恢复 → 生存 → 反击 → 僵持改出 →
   被锁防御 → 交战 的优先级选一个战术族。反击族（`COUNTER_*`）是主动的：
   被咬尾时用机动换角度，或把尾追转成迎头，而不是一路被动规避；改出族
   （`RESET`）用于双方长时间兜圈子、谁都拿不到射击窗口时主动重置几何。
3. 机动层（`MANEUVERS` + `_maneuvers`）：每个机动声明自己的能力
   （高 G / 最低速度 / 高度余量 / 距离门限），由表驱动执行，不再按名字前缀猜。
4. 控制层（`Action`）：滚转角 + 过载 + 油门，由同一套内核执行。

所有机动最终都落到滚转角 + 过载指令上，因此可以被点质量内核执行，
也方便以后交给 RL 直接学习。
"""
from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass

from sim import G, MAX_BANK, Action, Observation, clamp, wrap_pi

# 期望角速度的软上限，避免目标突然跳到正后方时拉出离谱的指令。
MAX_GAMMA_DOT = math.radians(20.0)
# 能量-高度约束：爬升角上限由比剩余功率 Ps 决定，高速时才允许额外 zoom。
ZOOM_ALLOWANCE = math.radians(25.0)
GAMMA_CAP = math.radians(55.0)
GAMMA_FLOOR = math.radians(-60.0)
# 倒扣滚转指令：Action.bank 以 MAX_BANK(80°) 为单位，倒扣需要 180°/80°。
BANK_INVERT = math.pi / MAX_BANK

# ---------- 人体过载耐受模型 ----------
# 飞行员在抗荷服 + 抗荷动作下可以长期承受约 6 g，短时（数秒）极限约 9 g；
# 人体对负过载（-Gz，血液涌向头部）的耐受差得多：持续约 -1.5 g、短时约 -3 g。
# 超过持续耐受后按“过载疲劳”累积，疲劳越大可用过载越接近持续值，
# 累积到极限即 G-LOC（正过载）或红视/意识模糊（负过载），短暂失控。
G_SUSTAIN = 6.0          # 可持续承受的 +Gz
G_PEAK = 9.0             # 短时 +Gz 极限
G_NEG_SUSTAIN = -1.5     # 可持续承受的 -Gz
G_NEG_PEAK = -3.0        # 短时 -Gz 极限
G_LOAD_TAU = 8.0         # 超过持续耐受后累积到失效的时间尺度 s
G_RECOVER_TAU = 20.0     # 低过载（<3 g）下的恢复时间尺度 s
G_NEG_TAU = 4.0          # 负过载累积尺度（比正过载快得多）
G_NEG_RECOVER_TAU = 10.0
GLOC_TIME = 2.5          # 失控后无法有效操纵的时间 s
# 高 G 机动的目标过载：实际还会被结构上限与人体耐受一起压住。
HIGH_G_N = 8.0
# 高 G 原语的准入条件：机体当前高度/速度下能拉出的最大过载低于该值时，
# “高 G 机动”只会变成一次普通盘旋（12 km、200 m/s 时 Su-57 只能拉 3.4 g），
# 既抢不到角度又把能量抽干。此时宁可维持普通追逐，等速度涨起来再拉。
HIGH_G_MIN_AVAIL = 6.5
# 筋斗/殷麦曼的过载上限：全程 7–8 g 会把动能抽干，末段只能挂在高迎角上
# 慢慢掉头。按 6.5 g 左右做完，速度还能留在可用区间，动作也更像标准机动。
VERTICAL_N = 6.5
# 防御转向反转的时间门槛：威胁持续位于反侧超过该时长才换向。
# 否则每次重新进入防御原语都会按瞬时方位翻一次，形成“无收益的反向机动”。
DEF_REVERSE_TIME = 1.5
# 垂直机动：需要连续数秒的滚转/拉杆才能完成，不能被最短驻留时间打断，
# 否则画面里只会看到一次莫名其妙的摆翼，然后标签就被换掉。
VERTICAL = frozenset((
    "SPLIT_S", "HIGH_G_SPLIT_S", "IMMELMANN", "LOOP",
    "BARREL", "HIGH_G_BARREL",
))
# 盘旋类原语：把机头拉到 80° 坡度需要约 1.5 s，最短驻留若只有 0.7 s，
# 刚滚一半就被切走，下一次若转向侧相反又要反向滚 140°，画面里就成了
# “回旋时一直在抖”。这类原语给更长的驻留时间，让每个转弯真正做出来。
TURNING = frozenset((
    "LEAD_TURN", "ONE_CIRCLE", "TWO_CIRCLE", "HIGH_G_TURN",
    "BREAK", "HIGH_G_BREAK", "BEAM",
))
TURN_HOLD = 2.0


@dataclass(frozen=True)
class Maneuver:
    """机动动作（第 2 层）：只描述“怎么做”和它的准入能力。

    tactic 是该机动的归属战术族，仅用于分层说明与日志；真正的选择逻辑
    在战术层，机动层不参与“打不打”的判断。
    """

    name: str
    tactic: str = "OFFENSIVE"
    high_g: bool = False          # 靠透支能量换角度，能量保护要让位
    min_speed_frac: float = 1.0   # 最低速度（相对当地失速）
    low_alt_need: float = 0.0     # 需要的离地余量 m
    high_alt_need: float = 0.0    # 需要的高度余量 m
    ps_positive: bool = False     # 要求 Ps > 0（否则做不完垂直机动）
    max_dist: float = 0.0         # 需要贴脸才能做的动作


# 第 2 层注册表：名字 → 能力声明。准入条件（_defense_ok）与能量保护
# （_limit_energy）都从这里读取，不再用 "HIGH_G" 前缀或名字集合硬编码。
MANEUVERS: dict[str, Maneuver] = {
    "RECOVER": Maneuver("RECOVER", "RECOVER"),
    "UNLOAD": Maneuver("UNLOAD", "SURVIVE"),
    "POSITION": Maneuver("POSITION", "OFFENSIVE"),
    "PURE": Maneuver("PURE", "OFFENSIVE"),
    "LEAD": Maneuver("LEAD", "OFFENSIVE"),
    "LAG": Maneuver("LAG", "OFFENSIVE"),
    "LEAD_TURN": Maneuver("LEAD_TURN", "OFFENSIVE"),
    "ONE_CIRCLE": Maneuver("ONE_CIRCLE", "OFFENSIVE"),
    "TWO_CIRCLE": Maneuver("TWO_CIRCLE", "OFFENSIVE"),
    "HIGH_G_TURN": Maneuver("HIGH_G_TURN", "OFFENSIVE", high_g=True),
    "HIGH_YOYO": Maneuver("HIGH_YOYO", "OFFENSIVE"),
    "LOW_YOYO": Maneuver("LOW_YOYO", "OFFENSIVE"),
    "BEAM": Maneuver("BEAM", "DEFENSIVE"),
    "NOTCH_DESC": Maneuver("NOTCH_DESC", "DEFENSIVE"),
    "CRANK": Maneuver("CRANK", "DEFENSIVE"),
    "F_POLE": Maneuver("F_POLE", "DEFENSIVE"),
    "BREAK": Maneuver("BREAK", "DEFENSIVE"),
    "HIGH_G_BREAK": Maneuver("HIGH_G_BREAK", "DEFENSIVE", high_g=True),
    "SPLIT_S": Maneuver(
        "SPLIT_S", "DEFENSIVE", min_speed_frac=1.35, low_alt_need=2400.0,
    ),
    "HIGH_G_SPLIT_S": Maneuver(
        "HIGH_G_SPLIT_S", "DEFENSIVE", high_g=True, low_alt_need=2200.0,
    ),
    "BARREL": Maneuver("BARREL", "DEFENSIVE", min_speed_frac=1.35),
    "HIGH_G_BARREL": Maneuver(
        "HIGH_G_BARREL", "DEFENSIVE", high_g=True, low_alt_need=2200.0,
    ),
    "IMMELMANN": Maneuver(
        "IMMELMANN", "DEFENSIVE", min_speed_frac=1.70,
        high_alt_need=2500.0, ps_positive=True,
    ),
    "LOOP": Maneuver(
        "LOOP", "DEFENSIVE", min_speed_frac=1.85,
        high_alt_need=3000.0, ps_positive=True,
    ),
    "SCISSORS": Maneuver("SCISSORS", "COUNTER", max_dist=1200.0),
    "COBRA": Maneuver("COBRA", "DEFENSIVE"),
    "EXTEND": Maneuver("EXTEND", "DISENGAGE"),
}


def aim_action(
    obs: Observation,
    az_error: float,
    el_error: float,
    gain: float = 2.2,
    throttle: float = 1.0,
) -> Action:
    """把“想指向的方位/仰角偏差”反解成滚转角与过载指令。

    用 3DOF 方程的逆：先定目标 psi_dot、gamma_dot，再解出 (n, phi)。
    过载超限时由 World 统一裁剪，裁剪后转弯/爬升比例保持不变。
    同时做能量-高度约束：目标航迹角不会超过当前 Ps 允许的爬升角，
    只有富余速度时才把动能短暂换成高度（zoom climb）。
    """
    v = max(obs.own_speed, 40.0)
    cg = max(math.cos(obs.own_gamma), 0.25)
    chi_max = G / (v * cg) * max(obs.own_n_avail, 0.5) * 0.9

    psi_dot = clamp(wrap_pi(az_error) * gain, -chi_max, chi_max)

    ps_climb = math.asin(clamp(obs.own_ps / v, -0.95, 0.95))
    zoom = clamp(obs.own_speed_frac - 1.6, 0.0, 1.0) * ZOOM_ALLOWANCE
    gamma_cap = clamp(ps_climb + zoom, GAMMA_FLOOR, GAMMA_CAP)
    target_gamma = clamp(obs.own_gamma + el_error, GAMMA_FLOOR, gamma_cap)
    gamma_dot = clamp((target_gamma - obs.own_gamma) * gain,
                      -MAX_GAMMA_DOT, MAX_GAMMA_DOT)

    a = max(cg + v * gamma_dot / G, 0.12)   # 垂直方向升力分量
    b = v * cg * psi_dot / G                # 水平方向升力分量
    n = math.hypot(a, b)
    phi = math.atan2(b, a)
    # 坡度最多到 86°，与 level_turn 一致。Action.bank 的 ±1 只是 80°，而
    # 8 g 的协调转弯本来就需要 acos(1/n)≈83°：若把坡度截在 80°，n·cosφ
    # 会比竖直需求多出约 1.4 g，飞机在急转/横转里就会一路爬升，看上去
    # 是“策略一转弯就莫名冲高”，到升限再被强制 UNLOAD。
    phi = clamp(phi, -math.radians(86.0), math.radians(86.0))
    # 先滚转、后拉杆：大坡度转向时若一上来就拉满过载，滚入的一两秒里升力
    # 仍主要在竖直方向，飞机会先被拉起来，等坡度到位再掉回去。追尾、脱离
    # 这类动作里就表现为“莫名其妙抬升高度、同时又掉速”。
    if abs(phi) - abs(obs.own_bank) > math.radians(30.0):
        n = min(n, 1.6)
    return Action(
        bank=clamp(phi / MAX_BANK, -math.radians(86.0) / MAX_BANK,
                   math.radians(86.0) / MAX_BANK),
        n=n,
        throttle=throttle,
        fire=obs.in_gun_cone,
    )


def pursuit_action(
    obs: Observation,
    az: float,
    el: float,
    gain: float = 1.2,
    throttle: float = 1.0,
) -> Action:
    """追踪类原语（纯追踪/提前量/滞后追踪）：带阻尼的指向控制。

    aim_action 用 atan2(b, a) 反解坡度，而 a = cosγ + v·γ̇/G 在高速下对俯仰
    误差非常敏感（300 m/s 时 3° 误差就要 5 g），近距追尾于是变成“拉杆抬头、
    再推杆掉头”的 1 s 周期摆振；反过来 a≈0 时，零点几度的方位误差又能把
    坡度顶到 80° 以上，升力全转到水平，机头迅速下坠再反向压回来。
    这里对俯仰误差限幅、总过载按可维持过载收口，并在需要大推杆时翼平。
    """
    el = clamp(el, -math.radians(4.0), math.radians(4.0))
    act = aim_action(obs, az, el, gain=gain, throttle=throttle)
    if act.n < 1.2 and abs(act.bank) > 0.6:
        # 卸载/推杆时翼平：坡度没有意义，只会把残存升力甩到侧面。
        act.bank = 0.0
    cap = max(min(obs.own_n_avail, max(obs.own_n_sustain * 1.25, 2.0)), 1.0)
    if act.n > cap:
        act.n = cap
    return act


def lead_point(obs: Observation, t_scale: float) -> tuple[float, float]:
    """预测目标的未来位置，返回相对本机的指向偏差 (方位, 仰角)。"""
    ce = math.cos(obs.los_el)
    ux, uy, uz = ce * math.cos(obs.los_az), ce * math.sin(obs.los_az), math.sin(obs.los_el)
    cg = math.cos(obs.foe_gamma)
    fvx = obs.foe_speed * cg * math.cos(obs.foe_psi)
    fvy = obs.foe_speed * cg * math.sin(obs.foe_psi)
    fvz = obs.foe_speed * math.sin(obs.foe_gamma)

    # 前置量按“交汇时间”估计，而不是“距离/本机速度”：咬尾时目标同向逃跑，
    # 后者会把交汇时间高估三四倍，在 1 km 内把机头按 30° 以上的前置角甩到
    # 目标另一侧，表现为回旋追尾时莫名其妙地反向压坡度。
    # 目标在拉开（接近率不足）时几乎不需要前置，退化成纯追踪。
    closing = obs.closure
    if closing > 25.0:
        t_go = min(obs.dist / closing, 1.5)
    else:
        t_go = 0.2
    t_go *= t_scale
    lx = ux * obs.dist + fvx * t_go
    ly = uy * obs.dist + fvy * t_go
    lz = uz * obs.dist + fvz * t_go

    az = math.atan2(ly, lx)
    el = math.atan2(lz, math.hypot(lx, ly))
    # 相对机头（炮口）而非航迹：6DOF 下两者相差一个迎角，用航迹会系统性打偏。
    return wrap_pi(az - obs.own_nose_az), el - obs.own_nose_el


def turn_action(
    obs: Observation,
    sign: float,
    n_target: float,
    el: float = 0.0,
    throttle: float = 1.0,
) -> Action:
    """“向 sign 一侧以 n_target 过载转向”的通用指令。

    aim_action 用一个大方位误差把水平转弯率顶到当前可用上限，
    坡度自然会到 90°；随后把过载缩到 n_target，就能在保持坡度的前提下
    按指定过载（瞬时或持续）稳定盘旋。sign 与相对方位同号。
    """
    act = aim_action(obs, sign * math.radians(150.0), el - obs.own_gamma, gain=6.0)
    act.n = max(1.0, min(n_target, act.n))
    act.throttle = throttle
    act.fire = obs.in_gun_cone
    return act


def heading_turn(
    obs: Observation,
    heading_error: float,
    gain: float = 0.35,
    max_rate_deg: float = 3.0,
    gamma_target: float = 0.0,
    throttle: float = 1.0,
) -> Action:
    """按航向误差做“比例式”转弯：坡度随误差连续变化，过零时不跳变。

    aim_action 用大方位误差把转弯率顶到当前可用上限，适合“尽快把机头转
    过去”的抢位动作；但用于 CRANK / 三九下高 / F-pole 这类“保持某个偏置
    航向”的动作时，它会以 16°/s 的最大角速度冲过目标偏置，误差反号后再
    反向满坡度，物理上要 1.5 s 滚转才能跟上，于是形成周期 2–3 s 的左右
    摆振（观感就是回旋时机体发抖）。

    这里先对期望转弯率限幅（默认 3°/s），再由 tanφ = v·ψ̇/G 反解坡度：
    绕速度矢量的协调转弯满足 n·sinφ = v·ψ̇·cosγ/G、n·cosφ = a，
    因此坡度与转弯率一一对应，误差变小坡度就自动变小，不会过冲。
    """
    v = max(obs.own_speed, 40.0)
    psi_dot = clamp(
        heading_error * gain,
        -math.radians(max_rate_deg), math.radians(max_rate_deg),
    )
    phi = math.atan2(v * psi_dot, G)
    # γ 与目标不一致时，用欠载/过载把航迹角拉过去（下降时允许短暂卸载）。
    gd = clamp((gamma_target - obs.own_gamma) * 1.2,
               -math.radians(6.0), math.radians(6.0))
    a = max(math.cos(obs.own_gamma) + v * gd / G, 0.15)
    n = max(a / max(math.cos(phi), 0.15), 0.2)
    # 先滚转、后拉杆：坡度还没到位时不要拉满过载，否则会先把飞机拽起来。
    if abs(phi) - abs(obs.own_bank) > math.radians(25.0):
        n = min(n, 1.6)
    return Action(
        bank=clamp(phi / MAX_BANK, -1.0, 1.0),
        n=n,
        throttle=throttle,
        fire=obs.in_gun_cone,
    )


def level_turn(
    obs: Observation,
    sign: float,
    n_level: float,
    gamma_target: float = 0.0,
    throttle: float = 1.0,
) -> Action:
    """以指定过载做协调转弯，同时保持目标航迹角。

    aim_action 用 atan2(b, a) 反解坡度，其中 a = cosγ + v·γ̇/G 是“竖直方向
    需要的升力分量”。盘旋时只想保持航迹角，a 会接近 0：坡度被推到 ±90°，
    而且俯仰误差的符号一反转，a 就变号，坡度整档从 +80° 翻到 −80°——
    这正是回旋时飞机左右摇摆、机体抖动的原因，也让“盘旋/急转”看起来
    像在乱滚。这里改按几何关系直接求解坡度：绕速度矢量的协调转弯必须满足

        n·cosφ = cosγ + v·γ̇/G

    由此反解 φ，动作与原语名称就一一对应（急转就是稳定的最大过载盘旋）。
    """
    v = max(obs.own_speed, 40.0)
    n_level = max(n_level, 1.0)
    gd = clamp((gamma_target - obs.own_gamma) * 2.0,
               -MAX_GAMMA_DOT, MAX_GAMMA_DOT)
    vert = (math.cos(obs.own_gamma) + v * gd / G) / n_level
    phi = math.acos(clamp(vert, -1.0, 1.0))
    # 超过 86° 坡度基本等于倒扣，普通盘旋不需要，也避免指令越过倒扣边界。
    phi = min(phi, math.radians(86.0))
    # 坡度还没滚到位时先只给 1.6 g：此时升力主要仍指向上方，直接拉 8 g
    # 会先把飞机整体拽起来，等坡度到位再落回去——观感就是盘旋时莫名爬升、
    # 随后又掉头下压。先滚转、后拉杆，轨迹才是稳定高度上的圆。
    n_cmd = n_level
    bank_err = wrap_pi(-sign * phi - obs.own_bank)
    # 坡度还没滚到位时按误差连续减小过载。原来用 25° 的硬门限整档切换
    # 1.6 g / n_level，坡度贴着门限时每 0.1 s 来回跳，机体就一直在点头。
    roll_in = clamp(
        (math.radians(30.0) - abs(bank_err)) / math.radians(20.0), 0.0, 1.0
    )
    n_cmd = max(1.6, n_level * roll_in)
    return Action(
        bank=clamp(sign * phi / MAX_BANK, -math.radians(86.0) / MAX_BANK,
                   math.radians(86.0) / MAX_BANK),
        n=n_cmd,
        throttle=throttle,
        fire=obs.in_gun_cone,
    )


class PurePursuit:
    """M1 基线保留：机头死盯目标，作为战术层的对照。"""

    def __init__(self, gain: float = 2.0) -> None:
        self.gain = gain
        self.situation = "-"
        self.primitive = "PURE"

    def act(self, obs: Observation) -> Action:
        act = aim_action(obs, obs.off_az, obs.off_el, gain=self.gain)
        act.launch = should_launch(obs)
        act.launch_radar = should_launch_radar(obs)
        # 基线也遵守燃油约束：油尽后只剩军推。
        act.afterburner = 1.0 if obs.own_fuel_frac > 0.02 else 0.0
        return act


def should_launch(obs: Observation) -> bool:
    """有锁定、在发射包线内、且目标已进入近距弹不可逃逸区时打一发。

    只打 NEZ 内的目标：包线边缘发射虽然“能打出去”，但目标一转身就能把弹
    拖到能量耗尽，等于白送一枚弹。对头也允许发射，命中与否交给内核的
    导弹动力学。
    """
    return (
        obs.own_locked
        and obs.own_missiles > 0
        and obs.in_missile_envelope
        and obs.missile_nez
        and obs.own_missiles_in_flight < 1
        and obs.own_speed_frac > 1.35
        and obs.dist > max(obs.missile_min_range, 600.0)
    )


def should_launch_radar(obs: Observation) -> bool:
    """雷达制导中距弹（主动弹）的发射条件：只在不可逃逸区内发射。

    中距弹的价值在超视距，太近时不如留着红外弹；也不能一次把两发都打光，
    先打一发、等命中或丢锁后再补射。主动弹在中制导段仍需要载机雷达
    保持跟踪与数据链，末段自主截获后即可脱离，因此这里同样要求速度余量。
    """
    return (
        obs.own_radar_locked
        and obs.own_radar_missiles > 0
        and obs.in_radar_envelope
        and obs.radar_nez
        and obs.own_radar_in_flight < 1
        and obs.own_speed_frac > 1.35
        and obs.dist > max(obs.radar_min_range, 2500.0)
    )


@dataclass
class _Situation:
    name: str
    angle: float          # 我的机头指向误差
    e_adv: float          # 比能量差
    close: bool
    overshoot: bool
    far: bool


class TacticalAgent:
    """几何判定 + 机动原语。

    aggression 越高越愿意对头开火、贴身剪刀；energy_pref 越高越偏爱
    把速度换成高度的垂直机动。
    """

    MIN_HOLD = 0.7       # 原语最短驻留时间，抑制抖动
    # 垂直机动（高/低悠悠）要拉几秒才能把速度换成高度再压回来，
    # 0.7 s 就被切掉等于白做完一个半程机动，反而白掉能量。
    YOYO_HOLD = 2.0
    SCISSOR_PERIOD = 2.2
    LOW_ENERGY_FRAC = 1.55   # 速度低于 1.55 倍当地失速即进入能量保护
    N_MARGIN = 1.25          # 允许短时超出维持过载的倍数，其余能量透支被禁止
    MISSILE_EVADE_DIST = 9000.0  # 来袭格斗弹进入该距离就开始规避
    SARH_EVADE_DIST = 16000.0    # 中距弹的告警距离更远：超视距阶段就要开始偏置
    MISSILE_BREAK_DIST = 4500.0  # 更远时用能量友好的横转/俯冲拉长 F-pole
    # 先手发射后的偏置距离：自己的中距弹还在飞时，若与目标仍相距该距离以上，
    # 打完立刻转入 CRANK 偏置，而不是继续直冲压近。太近（<8 km）时偏置没有
    # 几何收益，反而浪费补射窗口，交回进攻层。
    POST_LAUNCH_CRANK_DIST = 8000.0
    # 偏置窗口时长：中距弹从 20 km 飞到命中约 15–17 s，全程偏置会把对局
    # 拖成远距僵持（实测发射数 +17%、命中率 38%→22%）。8 s 足够把接近率
    # 压下来、让自己脱离对方当时发射线，又不至于丢掉随后的补射窗口。
    POST_LAUNCH_CRANK_TIME = 8.0
    COBRA_DIST = 1800.0          # 矢量喷口机型的过失速反制距离：来弹贴脸才用
    LOCK_BREAK_DIST = 8000.0     # 被对方锁定且进入该距离后评估是否防御
    OPENING_DIST = 20000.0       # 开局占位阶段：距离大于该值时不直冲对头
    MERGE_START = 5000.0     # 进入该距离开始领先转弯，抢交汇角度
    # 交汇方案在贴身盘旋阶段的保持距离。取 2200 而不是 1800，
    # 是为了让“保持方案”和停用的距离门限错开，避免在门限上每拍翻转转向侧。
    MERGE_HOLD = 2200.0
    MERGE_TIMEOUT = 12.0         # 交汇抢位方案的保持时间 s

    # 防御动作候选池：导弹来袭与被锁定（未射弹）各有一套，具体选哪个
    # 由 _maneuver_prob 按成功率评分决定，不再按固定顺序取第一个可用项。
    # 实弹规避只用水平/下降类动作：殷麦曼、筋斗会把机头指向高空，
    # 在 9 km 外就开始规避时会连续爬升把能量抽干，反而更容易被后续弹命中。
    # 高 G 变体只在速度与过载耐受都允许时才会被 _defense_ok 放行。
    MISSILE_DEFENSE = (
        "BEAM", "HIGH_G_BREAK", "BREAK", "HIGH_G_SPLIT_S",
        "BARREL", "HIGH_G_BARREL", "SPLIT_S",
    )
    LOCK_DEFENSE = (
        "BEAM", "HIGH_G_BREAK", "BREAK", "SCISSORS",
        "HIGH_G_SPLIT_S", "SPLIT_S", "IMMELMANN", "LOOP",
    )
    # 中距弹的进攻性防御也纳入“立即抢占”与能量保护。
    DEFENSE_SET = frozenset(
        MISSILE_DEFENSE + LOCK_DEFENSE
        + ("CRANK", "NOTCH_DESC", "F_POLE", "COBRA")
    )
    # 反击动作是一次性的角度反转：做完（或做砸）要冷却一段时间再评估，
    # 否则被咬尾时会每 0.1 s 重新下发一次破 S，永远停在改出段。
    COUNTER_COOLDOWN = 5.0
    # 反击一旦选定就要执行完整套动作（破 S / 剪刀需要数秒），
    # 不能下一拍就被“被锁防御”抢走，否则只会做一个 0.1 s 的抖动。
    COUNTER_COMMIT = 4.0
    # 僵持改出：双方在近距缠斗里长时间兜圈子、谁都拿不到射击窗口时，
    # 主动放弃当前盘旋、重置几何（拉开重来 / 反向切 / 破 S 掉头）。
    # 门槛取 10 s：实测单次近距缠斗窗口只有 10–14 s，门限若取 20 s，
    # 改出还没触发双方就已经拉开，等于永远不生效；10 s 足够让“谁先转
    # 谁丢角度”的僵持成立，又能在同一个缠斗窗口里做出改出。
    STALEMATE_TIME = 10.0
    RESET_COMMIT = 5.0
    RESET_COOLDOWN = 25.0

    def __init__(
        self,
        aggression: float = 0.5,
        energy_pref: float = 0.5,
        decision_dt: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.aggression = aggression
        self.energy_pref = energy_pref
        self.dt = decision_dt
        # 开局占位不再固定成一个模板：每次启动在机型允许的范围内随机取样。
        # seed 只用于实验复现，正常观战/无头运行时不传，即为真随机。
        self._rng = random.Random(seed)
        self._open_side = self._rng.choice((-1.0, 1.0))
        if energy_pref < 0.5:
            # 角度型（F-16V）：大一点横向间隔，便于交汇前拉出领先转弯。
            self._open_offset = math.radians(self._rng.uniform(9.0, 18.0))
            self._open_climb = math.radians(self._rng.uniform(-5.0, 3.0))
            self._open_throttle = self._rng.uniform(0.95, 1.0)
            # 占位段的净爬升预算：角度型几乎不爬，避免背着高度优势浪费动能。
            self._open_climb_gain = self._rng.uniform(0.0, 300.0)
        else:
            # 能量型（Su-57）：偏置小、爬升存能量，留给交汇后的垂直机动。
            self._open_offset = math.radians(self._rng.uniform(5.0, 13.0))
            self._open_climb = math.radians(self._rng.uniform(0.0, 12.0))
            self._open_throttle = self._rng.uniform(0.85, 1.0)
            # 48 km 的接近段最多只换取 0.5–1.2 km 高度。
            self._open_climb_gain = self._rng.uniform(500.0, 1200.0)

        self.primitive = "PURE"     # 当前机动（第 2 层，保留给日志/渲染）
        self.tactic = "OFFENSIVE"   # 当前战术族（第 1 层）
        self.situation = ""
        # 第 2 层执行表：机动名 → 具体动作函数。选择逻辑不在这一层。
        self._maneuvers = self._build_maneuver_table()
        self._hold = 0.0
        self._phase = 0
        self._phase_t = 0.0
        self._scissor_sign = 1.0
        self._scissor_t = 0.0
        self._def_sign = 1.0
        self._def_reverse_t = 0.0
        self._def_engaged = False
        self._counter_cd = 0.0
        # 已承诺执行的反击 (战术, 机动) 及剩余执行时间。
        self._counter_plan: tuple[str, str] | None = None
        self._counter_commit = 0.0
        # 僵持改出：兜圈子累计时长、当前改出方案与冷却。
        self._reset_plan: tuple[str, str] | None = None
        self._reset_commit = 0.0
        self._reset_cd = 0.0
        self._circle_t = 0.0
        self._opening_done = False
        # 开局爬升目标的当前值：以有限速率逼近 _open_climb，避免 Ps 在门限
        # 附近抖动时把俯仰指令整档切换。
        self._open_climb_cur = 0.0
        self._open_alt0: float | None = None
        self._far = False

        # 垂直机动的分段标记：过顶/俯冲是否已经越过关键点，以及本次机动
        # 是否已经做完（做完才允许被其它原语接管）。
        self._dived = False
        self._maneuver_done = True
        # 高/低 Yo-Yo 做完后的冷却：旧实现每 2.2 s 循环一次且净爬升，
        # 一路把飞机顶到能量升限，之后只能被强制 UNLOAD。
        self._yoyo_cd = 0.0
        # 本次 Yo-Yo 的抬升/俯冲目标与俯冲卸载量：进入时按当时的态势定一次，
        # 之后不再随距离变化重算，否则边做边改目标会把动作做半截。
        self._yoyo_goal: float | None = None
        self._yoyo_t = 1.2      # 高 Yo-Yo 拉起到目标角的时间预算 s
        self._yoyo_n = 0.35     # 低 Yo-Yo 俯冲段的卸载过载

        # 交汇抢位方案：转向侧、单环/双环、以及方案剩余保持时间。
        self._merge_sign = 1.0
        self._merge_plan = ""
        self._plan_ttl = 0.0
        self._turn_sign = 1.0
        self._turn_locked = False

        # 人体过载耐受状态：0=完全恢复，1=到达耐受极限（G-LOC 边界）。
        self.g_load = 0.0
        self.g_load_neg = 0.0
        self._gloc_t = 0.0
        self.n_pos_tol = G_PEAK
        self.n_neg_tol = G_NEG_PEAK

        # 先手发射后的偏置窗口：看到自己的中距弹离架就开计时，
        # 窗口内用 CRANK 保持照射/数据链并拉开几何，窗口结束立刻回到进攻。
        self._post_launch_t = 0.0
        self._prev_radar_in_flight = 0

    # ---------- 主入口 ----------

    def act(self, obs: Observation) -> Action:
        self._update_g_tolerance(obs)
        # 先手偏置窗口：中距弹数量增加说明刚有一发离架。
        if obs.own_radar_in_flight > self._prev_radar_in_flight:
            self._post_launch_t = self.POST_LAUNCH_CRANK_TIME
        self._prev_radar_in_flight = obs.own_radar_in_flight
        self._post_launch_t = max(0.0, self._post_launch_t - self.dt)
        self._counter_cd = max(0.0, self._counter_cd - self.dt)
        if self._counter_commit > 0.0:
            self._counter_commit = max(0.0, self._counter_commit - self.dt)
            if self._counter_commit <= 0.0:
                # 承诺期结束：清空方案并进入冷却，避免立刻重新选同一个反击。
                self._counter_plan = None
                self._counter_cd = self.COUNTER_COOLDOWN
        self._yoyo_cd = max(self._yoyo_cd - self.dt, 0.0)
        self._reset_cd = max(0.0, self._reset_cd - self.dt)
        if self._reset_commit > 0.0:
            self._reset_commit = max(0.0, self._reset_commit - self.dt)
            if self._reset_commit <= 0.0:
                self._reset_plan = None
                self._reset_cd = self.RESET_COOLDOWN
        # 防御转向的连续性：威胁只在持续位于反侧时才换向，避免“一进防御就反转”。
        # 方位接近 ±180°（威胁正在正后方）时两侧完全等价，而 wrap 会让符号
        # 在边界上每拍翻转——若把它也算进计时，防御动作会每 1.5 s 反向滚
        # 140°，看上去就是在抖。这种情况直接保持当前转向侧。
        if obs.incoming_missile_dist < self.SARH_EVADE_DIST:
            az = obs.incoming_missile_az
            if (
                az * self._def_sign < 0.0
                and math.radians(60.0) < abs(az) < math.radians(150.0)
            ):
                self._def_reverse_t += self.dt
            else:
                self._def_reverse_t = 0.0
            if self._def_reverse_t >= DEF_REVERSE_TIME:
                self._def_sign = -self._def_sign
                self._def_reverse_t = 0.0
        else:
            self._def_reverse_t = 0.0
            self._def_engaged = False
        sit = self._assess(obs)
        self.situation = sit.name
        # 兜圈子检测：近距缠斗、双方都没形成锁定、也没有来袭弹，说明谁都
        # 拿不到射击窗口，只是在互相绕圈。累计到 STALEMATE_TIME 就交给
        # 改出族主动改变几何，而不是继续陪对手盘旋到超时。
        if (
            500.0 < obs.dist < 2600.0
            and sit.name in ("NEUTRAL", "DEFENSIVE")
            and obs.incoming_missile_dist > self.MISSILE_EVADE_DIST
        ):
            self._circle_t += self.dt
        else:
            # 不满足时按两倍速率衰减而不是直接归零：真实缠斗里态势会在
            # 锁定/距离门限上短暂跳出窗口，一旦清零就永远攒不到门槛，
            # 改出族形同虚设。衰减式累计能容忍这些抖动，又不会把
            # 两次无关的盘旋拼在一起。
            self._circle_t = max(0.0, self._circle_t - 2.0 * self.dt)
        if obs.dist < self.OPENING_DIST:
            self._opening_done = True

        # 交汇抢位方案在接近到 MERGE_START 内时锁定，随后按方案执行。
        # 只在仍是迎头接近时重算，避免进入盘旋格斗后方案来回翻转。
        # 贴身盘旋期间方案不过期：一旦按瞬时方位重排，转向侧就会跟着
        # 方位角翻转，表现为“回旋追尾时突然反向、且很快又切回来”。
        holding_merge = obs.dist < self.MERGE_HOLD and obs.aspect > math.radians(55.0)
        if self._plan_ttl > 0.0:
            if not holding_merge:
                self._plan_ttl -= self.dt
        elif (
            obs.dist < self.MERGE_START
            and obs.dist > 1500.0
            and obs.aspect > math.radians(120.0)
        ):
            self._make_merge_plan(obs)

        tactic, want = self._plan(sit, obs)
        self.tactic = tactic
        want = self._sticky_defense(want)
        # 垂直机动必须做完：破 S / 殷麦曼 / 筋斗 / 桶滚都依赖连续几秒的
        # 滚转 + 拉杆，被 0.7 s 的最短驻留切走就只剩一次莫名其妙的摆翼。
        # 只有边界/能量这类安全原语可以打断；超时兜底防止卡在某个分段。
        if self.primitive in VERTICAL:
            if self._phase_t >= 10.0:
                self._maneuver_done = True
            if not self._maneuver_done and want not in ("RECOVER", "UNLOAD"):
                want = self.primitive
        # 只有安全类（回场/卸载）与“转入防御”可以立即抢占。
        # 旧实现把防御原语本身也放进可抢占集合，于是“防御→进攻→防御”
        # 每 0.1 s 都能整档切换：破 S 只飞了两个决策周期就被换回平盘，
        # 屏幕上标着破 S、实际却是平飞；盘旋类原语也会被反向滚 140° 打断。
        emergency = ("RECOVER", "UNLOAD")
        bypass = want in emergency or (
            want in self.DEFENSE_SET and self.primitive not in self.DEFENSE_SET
        )
        need = self.YOYO_HOLD if self.primitive in ("HIGH_YOYO", "LOW_YOYO") else self.MIN_HOLD
        if self.primitive in TURNING:
            need = max(need, TURN_HOLD)
        if want != self.primitive and (self._hold >= need or bypass):
            self._enter(want, obs)

        self._hold += self.dt
        self._phase_t += self.dt
        self._scissor_t -= self.dt
        act = self._execute(self.primitive, obs)
        self._set_power(act, obs)
        return act

    def _set_power(self, act: Action, obs: Observation) -> None:
        """发动机功率：进入交战前按巡航功率飞，交战与防御时再点加力。

        加力油耗约为军推的三倍，因此只在“战术层要求全油门”或能量吃紧、
        有弹来袭时使用；燃油低于 2% 后强制回军推，避免长时间空耗。
        24 km 以外的进入段还没进入交战，按维持巡航马赫所需的军推飞，
        避免一开局油门就顶到军用最大、推力条直接接近满格。
        """
        if obs.own_fuel_frac <= 0.02:
            act.afterburner = 0.0
            return
        engaged = (
            obs.dist < 20000.0
            # 收到发射告警（中距弹最远几十公里）就开始加速，为后续的
            # F-pole / 规避攒能量，而不是等弹飞到 16 km 才反应。
            or obs.incoming_missile_dist < 40000.0
            or obs.own_missiles_in_flight > 0
            or obs.own_radar_in_flight > 0
        )
        if not engaged:
            act.throttle = min(
                act.throttle, clamp(obs.own_cruise_throttle + 0.05, 0.30, 0.95)
            )
            act.afterburner = 0.0
            return
        wants = act.throttle >= 0.97
        urgent = (
            obs.own_speed_frac < 1.45
            or obs.incoming_missile_dist < self.SARH_EVADE_DIST
        )
        if obs.own_fuel_frac < 0.12 and not urgent:
            # 油量见底时省着用：只保留军推，除非正在被弹追击。
            act.afterburner = 0.0
            return
        act.afterburner = 1.0 if (wants or urgent) else 0.0

    # ---------- 第一层：几何判定 ----------

    def _assess(self, obs: Observation) -> _Situation:
        ang = obs.angle_off_nose
        aspect = obs.aspect
        foe_off = obs.foe_off_nose

        if obs.dist > 2600.0:
            name = "SEPARATED"
        elif ang < math.radians(30.0) and aspect > math.radians(150.0):
            name = "HEAD_ON"
        elif ang > math.radians(100.0) and foe_off < math.radians(60.0):
            name = "DEFENSIVE"
        elif aspect < math.radians(65.0) and ang < math.radians(80.0):
            name = "OFFENSIVE"
        else:
            name = "NEUTRAL"

        # 阈值附近的态势会在相邻档位间来回跳（ang 100°↔99°），而防御类原语
        # 又能立即抢占，结果就是 PURE/BREAK 每 0.1 s 互换一次。给“正在防御”
        # 加一段粘滞：几何没有明显改善就继续按防御处理。
        if (
            self.situation == "DEFENSIVE"
            and name in ("NEUTRAL", "OFFENSIVE")
            and ang > math.radians(90.0)
            and foe_off < math.radians(75.0)
        ):
            name = "DEFENSIVE"

        # 1800 m 附近的“远/近”会在门限上每 0.1 s 翻转，进而让 PURE/LEAD
        # 反复互换，看起来像无收益的策略抖动；这里加一段滞回。
        far = obs.dist > (1600.0 if self._far else 2000.0)
        self._far = far

        return _Situation(
            name=name,
            angle=ang,
            e_adv=obs.energy_diff,
            close=obs.dist < 600.0,
            overshoot=obs.closure > 60.0 and obs.dist < 600.0 and ang > math.radians(50.0),
            far=far,
        )

    # ---------- 第二层：战术（恢复 / 生存 / 反击 / 交战） ----------

    def _plan(self, sit: _Situation, obs: Observation) -> tuple[str, str]:
        """战术层入口，返回 (战术族, 机动名)。

        优先级：恢复 > 生存 > 反击 > 僵持改出 > 被锁防御 > 交战。生存只处理
        真正的威胁（来袭弹、过载疲劳、能量与弹药底线）；单纯被雷达锁定交给
        反击族优先处理——被咬尾时反咬比横转更有收益。反击不成立时，如果
        双方已经缠斗到兜圈子的程度，由改出族主动改变几何。
        """
        for tactic in (
            self._tactic_recover,
            self._tactic_survive,
            self._tactic_counter,
            self._tactic_reset,
            self._tactic_defend_locked,
        ):
            plan = tactic(sit, obs)
            if plan is not None:
                return plan
        return self._tactic_engage(sit, obs)

    def _tactic_recover(
        self, sit: _Situation, obs: Observation,
    ) -> tuple[str, str] | None:
        """恢复：贴着边界或高度上下限时先回场。"""
        # 转弯半径随速度平方增长，所以边界余量按速度缩放，避免高速时来不及转回。
        edge_need = max(900.0, 8.0 * obs.own_speed)
        low_need = max(1200.0, 3.5 * obs.own_speed)
        if (
            obs.edge_margin < edge_need
            or obs.z_margin_low < low_need
            or obs.z_margin_high < 600.0
        ):
            return "RECOVER", "RECOVER"
        return None

    def _tactic_survive(
        self, sit: _Situation, obs: Observation,
    ) -> tuple[str, str] | None:
        """生存：导弹规避、过载疲劳、能量与弹药底线。

        这一段不追求占位，只保证“先活下来”；全部命中即返回，否则交回
        战术层继续询问反击与交战。单纯被雷达锁定不在这里处理：那是
        反击族的机会窗口，反咬不成立时才由 LOCK_DEFENSE 兜底。
        """
        # 眼镜蛇是瞬态机动：机头拉到极限迎角后必须立刻改出，否则会变成
        # 挂在高迎角上持续爬升，速度被抽干、反而成了靶子。
        if self.primitive == "COBRA" and self._phase_t > 1.5:
            return "SURVIVE", "SPLIT_S" if obs.z_margin_low > 2400.0 else "BREAK"
        # 规避过程中仍要守住能量：机头已经高高抬起、速度又不够时，
        # 继续压坡度拉过载只会变成爬升耗能，先俯冲把航迹角压下来。
        if (
            self.primitive in self.DEFENSE_SET
            and obs.own_gamma > math.radians(25.0)
            and obs.own_speed_frac < 1.70
        ):
            # 破 S 会掉近 2 km 高度，低空时再做就是直接扎地。
            return "SURVIVE", "SPLIT_S" if obs.z_margin_low > 2400.0 else "BREAK"
        # 先手打完立刻转为偏置防御：刚发射的中距弹还需要载机照射/数据链，
        # 这时不再继续直冲压近，而是把目标压到 40° 偏置（CRANK）——一边保持
        # 制导，一边用横向位移拉开交战几何、降低接近率，把自己从对方的
        # 发射包里拉出来。偏置窗口结束后立刻回到进攻，避免远距僵持。
        if (
            self._post_launch_t > 0.0
            and obs.incoming_missile_dist > self.SARH_EVADE_DIST
            and obs.dist > self.POST_LAUNCH_CRANK_DIST
        ):
            return "SURVIVE", "CRANK"
        # 来袭格斗弹优先：远距先做能量友好的横转/俯冲拉长 F-pole，
        # 只有进入近距才用最大过载急转硬甩。
        # 雷达制导中距弹（含主动弹）在超视距阶段就要开始偏置/横转；
        # 红外弹只有进入近距才需要硬甩。
        evade_dist = (
            self.SARH_EVADE_DIST
            if obs.incoming_missile_radar
            else self.MISSILE_EVADE_DIST
        )
        if (
            obs.incoming_missile_dist < evade_dist
            and obs.own_speed_frac > 1.25
        ):
            # 矢量喷口的过失速反制：来弹已经贴脸、自己还有速度时，
            # 拉眼镜蛇把速度矢量“刹住”，让弹从下方冲过去。
            # 只在 6DOF 且装矢量喷口的机型上可用。
            if (
                obs.own_tvc
                and obs.dof >= 6
                and not obs.incoming_missile_radar
                and obs.incoming_missile_dist < self.COBRA_DIST
                and obs.own_speed_frac > 1.90
                and obs.z_margin_low > 1200.0
            ):
                return "SURVIVE", "COBRA"
            # 雷达制导中距弹按 FC3 手册的超视距防御流程处理：
            # 自己的弹先到就 F-pole 偏置，否则三九下高。
            if obs.incoming_missile_radar:
                anti = self._anti_radar(obs)
                if anti is not None:
                    return "SURVIVE", anti
            # 自己的雷达弹还处在需要照射/数据链的中制导段：远距先保持机头
            # 指向把弹导完；主动弹一旦末段截获就不再需要（own_radar_support）。
            keep_illum = (
                obs.own_radar_support
                and obs.radar_tracking
                and obs.incoming_missile_dist > self.MISSILE_BREAK_DIST
            )
            if not keep_illum:
                if obs.incoming_missile_dist > self.MISSILE_BREAK_DIST:
                    return "SURVIVE", "BEAM" if obs.z_margin_low > 1500.0 else "SPLIT_S"
                return "SURVIVE", self._pick_defense(obs, vs_missile=True)
        # 过载疲劳接近失控：没有近距威胁时先卸载恢复，避免 G-LOC 白送人头。
        if (
            (self.g_load > 0.90 or self.g_load_neg > 0.90)
            and obs.incoming_missile_dist > self.MISSILE_BREAK_DIST
        ):
            return "SURVIVE", "UNLOAD"
        # 能量-高度约束：连 1g 平飞都维持不住（Ps<0）说明已超过能量升限，
        # 再拉机动只会把速度抽干，必须先卸载俯冲换回能量。
        if obs.own_ps_level < -10.0 and obs.own_alt > 11000.0:
            return "SURVIVE", "UNLOAD"
        # 打空或重伤就不再纠缠。只有自己明显伤得更重才脱离：双方都被打残时
        # 继续缠斗，否则两架重伤机各自逃跑会把对局拖成“飞出空域”。
        if obs.own_ammo <= 1.0 or (
            obs.own_hp_frac < 0.18
            and obs.foe_hp_frac > max(0.25, obs.own_hp_frac + 0.08)
        ):
            return "DISENGAGE", "EXTEND"
        # 能量纪律：速度接近失速时先卸载俯冲，不允许挂着机头往上爬
        if self.primitive == "UNLOAD":
            # 必须同时满足“速度回来了”和“机头已经压下来”，避免反复抬头
            if obs.own_speed_frac < 1.40 or obs.own_gamma > math.radians(15.0):
                return "SURVIVE", "UNLOAD"
        elif obs.own_speed_frac < 1.20:
            return "SURVIVE", "UNLOAD"
        return None

    def _tactic_counter(
        self, sit: _Situation, obs: Observation,
    ) -> tuple[str, str] | None:
        """反击：被咬尾时用机动换角度，而不是一路被动规避。

        四个反击战术按“最快见效”排序：
        - COUNTER_SNAP：对手已经冲过头，立即剪刀反切；
        - COUNTER_HEADON：能量够时做 180° 垂直反转，把尾追变迎头；
        - COUNTER_REVERSE：同能量或占优且贴脸，用剪刀把机头甩回去；
        - COUNTER_DRAG：能量劣势时先拉开，等对手掉速再反切。

        反击必须让位给生存：有来袭弹、或过载疲劳接近失控时一律不做，
        否则会在弹道末端把机腹反复转向来袭弹。
        """
        if obs.incoming_missile_dist < self.MISSILE_EVADE_DIST:
            return None
        # 承诺期内不重新决策：破 S / 剪刀是数秒量级的动作，中途被下一拍
        # 改判就只剩一个没有收益的抖动。
        if self._counter_commit > 0.0 and self._counter_plan is not None:
            return self._counter_plan
        if obs.own_speed_frac < 1.35 or self.g_load > 0.85 or self._counter_cd > 0.0:
            return None
        # “被咬尾”：目标几乎在正后方，而我的机头背离他。
        tail = (
            obs.foe_off_nose < math.radians(50.0)
            and obs.angle_off_nose > math.radians(110.0)
        )
        if not tail and sit.name != "DEFENSIVE":
            return None
        if obs.dist > 6000.0:
            return None
        # 反切：对手冲过头。
        if sit.overshoot and obs.dist < 900.0 and obs.own_speed_frac > 1.45:
            return self._start_counter("COUNTER_SNAP", "SCISSORS")
        # 迎头化：破 S 的 180° 反转需要离地余量，且不能太慢。
        # energy_diff 的单位是 J/kg（v²/2 + g·z）：-15000 约等于 300 m/s 下
        # 20 m/s 的速度亏欠，再差就说明反转做完会被对手直接压住。
        if (
            obs.dist > 1200.0
            and obs.own_speed_frac > 1.55
            and obs.energy_diff > -15000.0
            and obs.z_margin_low > 1200.0
        ):
            return self._start_counter("COUNTER_HEADON", "SPLIT_S")
        # 反咬：距离很近、速度允许时剪刀是唯一能立刻抢角度的动作。
        # 这里不再要求比能量占优——被咬尾时本来就常年能量吃亏，
        # 要求占优等于永远不反咬。
        if obs.dist < 1500.0 and obs.own_speed_frac > 1.45:
            return self._start_counter("COUNTER_REVERSE", "SCISSORS")
        # 诱敌：能量劣势时先拉开距离，等对手掉速后再转入反切。
        if 2000.0 <= obs.dist <= 6000.0 and obs.energy_diff < -8000.0:
            return self._start_counter("COUNTER_DRAG", "EXTEND")
        return None

    def _start_counter(self, tactic: str, primitive: str) -> tuple[str, str]:
        """锁定一次反击，并设定承诺执行时间（见 COUNTER_COMMIT）。"""
        self._counter_plan = (tactic, primitive)
        self._counter_commit = self.COUNTER_COMMIT
        return self._counter_plan

    def _tactic_reset(
        self, sit: _Situation, obs: Observation,
    ) -> tuple[str, str] | None:
        """僵持改出：缠斗太久、谁都拿不到射击窗口时主动改变几何。

        两个人在同一个水平面上互相盘旋时，角度、能量和距离都会卡在
        一个稳定的循环里：谁先转谁丢角度，于是大家都不转，剩下的只有
        绕圈到超时。改出族承认“当前盘旋没有收益”，用一次明确的动作把
        循环打断：
        - 被对手压着转（DEFENSIVE）：近距反向切、远距破 S 掉头，逼他冲过头；
        - 占位/中立且有能量：主动拉开，把缠斗重置成迎头；
        - 能量不足：急转脱离，先脱离当前盘旋再重新接近。

        方案一旦选定就承诺执行 RESET_COMMIT 秒，避免改出动作被下一拍
        的常规追逐抢回；之后进入 RESET_COOLDOWN，防止连续改出变成逃跑。
        """
        if self._reset_commit > 0.0 and self._reset_plan is not None:
            return self._reset_plan
        if self._reset_cd > 0.0 or self._circle_t < self.STALEMATE_TIME:
            return None
        if obs.dist < 400.0 or obs.dist > 3000.0:
            return None
        if obs.incoming_missile_dist < self.SARH_EVADE_DIST:
            return None
        if sit.name == "DEFENSIVE":
            plan = (
                ("RESET", "SCISSORS") if obs.dist < 1200.0 else ("RESET", "SPLIT_S")
            )
        elif obs.own_speed_frac >= 1.45 and obs.energy_diff >= -6000.0:
            plan = ("RESET", "EXTEND")
        elif obs.z_margin_low > 2400.0:
            # 破 S 不需要速度富余：半滚倒扣本来就是掉高度换机头，只要
            # 留够离地余量就能把水平盘旋改成一次垂直掉头。
            plan = ("RESET", "SPLIT_S")
        else:
            plan = ("RESET", "BREAK")
        self._reset_plan = plan
        self._reset_commit = self.RESET_COMMIT
        self._reset_cd = self.RESET_COOLDOWN
        self._circle_t = 0.0
        return plan

    def _tactic_defend_locked(
        self, sit: _Situation, obs: Observation,
    ) -> tuple[str, str] | None:
        """被锁防御：被雷达/红外锁定但没有立刻可用的反击几何时，转被动规避。

        这一档排在反击之后：同样是“被咬尾”，能反咬就先反咬，反咬不成立
        （距离、速度、能量或冷却不允许）才退回横转/破 S 消耗对手的射击机会。
        """
        if not (obs.foe_locked or obs.foe_radar_locked):
            return None
        if obs.dist >= self.LOCK_BREAK_DIST or obs.own_speed_frac <= 1.30:
            return None
        if not self._should_defend(obs, sit):
            return None
        return "LOCK_DEFENSE", self._pick_defense(obs, vs_missile=False)

    def _tactic_engage(
        self, sit: _Situation, obs: Observation,
    ) -> tuple[str, str]:
        """交战：占位、抢位方案、进攻/防御几何与近距缠斗。"""
        if sit.name == "SEPARATED":
            # 超视距阶段按“先视、先射、先撤”推进。
            # 自己的中距弹还在中制导时必须保持机头指向（数据链/照射），
            # 占位/偏置动作只能在这里之后做；照射一断弹就变无控弹道。
            if obs.own_radar_support and obs.radar_tracking:
                return "SEPARATED", "PURE"
            # 开局先按能量/角度倾向占位。
            if not self._opening_done:
                return "SEPARATED", "POSITION"
            # 进入交汇距离后提前做领先转弯，切进对手转弯圆环的内侧。
            if obs.dist < self.MERGE_START and obs.aspect > math.radians(120.0):
                return "SEPARATED", "LEAD_TURN"
            return "SEPARATED", "PURE"

        if sit.name == "HEAD_ON":
            # 交汇瞬间按抢位方案执行：单环抢先指向，双环持续盘旋。
            # 能量明显占优时改用高 Yo-Yo 把速度换成高度再从上方压下来。
            if self._merge_plan == "ONE_CIRCLE":
                return "HEAD_ON", "ONE_CIRCLE"
            if self._merge_plan == "TWO_CIRCLE":
                return "HEAD_ON", "TWO_CIRCLE"
            cands = ["LEAD", "BREAK"]
            if (
                sit.e_adv > 0.0
                and self.energy_pref >= 0.5
                and self._yoyo_cd <= 0.0
            ):
                cands.append("HIGH_YOYO")
            return "HEAD_ON", self._pick_maneuver(cands, sit, obs)

        # 交汇后的第一圈仍按抢位方案执行：HEAD_ON 只是一个瞬时窗口，
        # 双方真正开始盘旋时已转为 NEUTRAL/DEFENSIVE，若不延续方案，
        # 单环/双环的决策就形同虚设。
        if (
            self._plan_ttl > 0.0
            and self._merge_plan
            and obs.dist < self.MERGE_HOLD
            and obs.aspect > math.radians(55.0)
        ):
            name = "ONE_CIRCLE" if self._merge_plan == "ONE_CIRCLE" else "TWO_CIRCLE"
            return sit.name, name

        if sit.name == "DEFENSIVE":
            # 只有对方真的冲过头（接近率为正、距离很近且机头已偏）才把剪刀
            # 纳入候选；仅仅“贴得近”就反复剪刀，会变成无收益的反向机动。
            cands = ["BREAK"]
            if sit.overshoot:
                cands.append("SCISSORS")
            return "DEFENSIVE", self._pick_maneuver(cands, sit, obs)

        if sit.name == "OFFENSIVE":
            cands = ["PURE", "LEAD"]
            if sit.close:
                cands.append("LAG")
            if (
                not sit.close
                and not sit.far
                and obs.own_speed_frac > 1.55
                and self.n_pos_tol > 6.5
                and obs.own_n_avail >= HIGH_G_MIN_AVAIL
            ):
                # 角度还不够、速度与过载耐受都富余时才把高 G 转弯纳入候选。
                # 机体本身拉不出 6.5 g 时（高空低速）高 G 只是白耗能量。
                cands.append("HIGH_G_TURN")
            return "OFFENSIVE", self._pick_maneuver(cands, sit, obs)

        # NEUTRAL：把当前态势下可用的近距机动都放进候选池，再由成功率
        # 评分挑选；评分相同（例如机头已基本指向时的 PURE/LEAD）随机选。
        cands = ["PURE"]
        if sit.angle < math.radians(25.0) and not sit.far:
            cands.append("LEAD")
        if (
            not sit.far
            and sit.angle > math.radians(40.0)
            and obs.own_speed_frac > 1.60
            and self.n_pos_tol > 6.5
            and obs.own_n_avail >= HIGH_G_MIN_AVAIL
        ):
            cands.append("HIGH_G_TURN")
        # 贴身缠斗时用垂直机动消耗/补充能量。
        if sit.close:
            high_ok = obs.own_ps_level > 5.0 and obs.z_margin_high > 4000.0
            low_ok = obs.z_margin_low > 1200.0
            if (
                obs.own_speed_frac > 1.7
                and self.energy_pref >= 0.5
                and high_ok
                and self._yoyo_cd <= 0.0
            ):
                cands.append("HIGH_YOYO")   # 太快：把多余速度换成高度
            if (
                obs.own_speed_frac < 1.5
                and self.energy_pref < 0.5
                and low_ok
                and self._yoyo_cd <= 0.0
            ):
                cands.append("LOW_YOYO")    # 太慢：俯冲增速再拉起
        return "NEUTRAL", self._pick_maneuver(cands, sit, obs)

    # ---------- 机动成功率评估 ----------

    def _pick_maneuver(
        self,
        candidates: list[str],
        sit: _Situation | None,
        obs: Observation,
        vs_missile: bool = False,
    ) -> str:
        """在候选机动里按“当前态势下的成功率”选一个，并列时随机。

        候选池由战术层给出（已包含距离、能量、过载耐受等硬门限），这里
        只负责排序：对每个候选算一个 0–1 的成功概率，取最大者；若多个
        候选概率完全相同，用 agent 自己的随机源挑一个，避免永远偏向
        列表里靠前的那个动作。
        """
        scored: dict[str, float] = {}
        for name in candidates:
            if not self._defense_ok(name, obs):
                continue
            scored[name] = self._maneuver_prob(name, sit, obs, vs_missile)
        if not scored:
            return candidates[0] if candidates else "PURE"
        best = max(scored.values())
        ties = [n for n, p in scored.items() if abs(p - best) <= 1e-9]
        if len(ties) > 1:
            return self._rng.choice(ties)
        return ties[0]

    def _maneuver_prob(
        self,
        name: str,
        sit: _Situation | None,
        obs: Observation,
        vs_missile: bool = False,
    ) -> float:
        """估计一个机动在当前态势下的成功概率（0–1）。

        评分只表达“这个动作做成并拿到收益的可能性”，不做战术取舍：
        - 指向越准、距离越近，追踪类（PURE/LEAD/LAG）越可靠；
        - 速度与可用过载越富余，高 G 抢角度的成功率越高；
        - 高度余量与能量决定悠悠/破 S 这类垂直机动能不能做完；
        - 防御动作按来袭弹剩余时间和能量余量分档。
        """
        speed = obs.own_speed_frac
        energy = clamp((speed - 1.25) / 0.75, 0.0, 1.0)
        near = clamp((2400.0 - obs.dist) / 1800.0, 0.0, 1.0)
        low = clamp(obs.z_margin_low / 3000.0, 0.0, 1.0)
        high = clamp(obs.z_margin_high / 5000.0, 0.0, 1.0)
        nfrac = clamp(obs.own_n_avail / 9.0, 0.0, 1.0)
        ttg = obs.incoming_missile_ttg
        if not math.isfinite(ttg):
            ttg = 99.0
        aggr = self.aggression

        if name in ("PURE", "LEAD", "LAG", "HIGH_G_TURN",
                    "HIGH_YOYO", "LOW_YOYO"):
            if sit is None:
                return 0.5
            aim = clamp(1.0 - sit.angle / math.radians(90.0), 0.0, 1.0)
            need = 1.0 - aim          # 机头偏离越大，越需要抢角度
            if name == "PURE":
                return 0.52 + 0.18 * aim + 0.14 * near
            if name == "LEAD":
                p = 0.50 + 0.16 * aim + 0.10 * near + 0.06 * aggr
                if sit.name == "HEAD_ON":
                    p += 0.12
                return p
            if name == "LAG":
                p = 0.48 + 0.12 * near
                if sit.overshoot:
                    p += 0.18
                return p
            if name == "HIGH_G_TURN":
                return 0.42 + 0.22 * energy + 0.18 * nfrac + 0.18 * need
            if name == "HIGH_YOYO":
                return (
                    0.40 + 0.22 * energy + 0.14 * high
                    + 0.10 * (1.0 if sit.close else 0.0)
                    + 0.06 * self.energy_pref
                )
            return (      # LOW_YOYO
                0.40 + 0.22 * (1.0 - energy) + 0.14 * low
                + 0.10 * (1.0 if sit.close else 0.0)
                + 0.06 * (1.0 - self.energy_pref)
            )

        if name == "SCISSORS":
            p = 0.44 + 0.16 * near
            if sit is not None and sit.overshoot:
                p += 0.26
            return p
        if name == "BREAK":
            p = 0.55 + 0.10 * energy + 0.05 * low
            if sit is not None and sit.name == "DEFENSIVE":
                p += 0.08
            return p

        # 防御类：按来袭弹剩余时间与能量余量给分。
        if name == "NOTCH_DESC":
            return 0.50 + 0.18 * low + 0.12 * (1.0 if ttg > 6.0 else 0.0)
        if name == "BEAM":
            # 横转是可持续的规避姿态：被锁定（弹还没出）时也应保持，
            # 只有弹进入末段（ttg 很小）才让位给高 G 硬甩。
            return 0.62 + 0.08 * clamp(ttg / 10.0, 0.0, 1.0)
        if name == "HIGH_G_BREAK":
            p = 0.40 + 0.12 * energy + 0.12 * nfrac
            if vs_missile and ttg <= 4.0:
                p += 0.15          # 末段才值得用高 G 硬甩
            return p
        if name == "HIGH_G_BARREL":
            p = 0.40 + 0.12 * energy + 0.10 * nfrac
            if vs_missile and 3.0 < ttg <= 7.0:
                p += 0.10
            return p
        if name == "BARREL":
            return 0.42 + 0.10 * (1.0 - energy)
        if name == "SPLIT_S":
            return 0.46 + 0.12 * low
        if name == "HIGH_G_SPLIT_S":
            p = 0.42 + 0.14 * low + 0.12 * nfrac
            if vs_missile and ttg <= 4.0:
                p += 0.10
            return p
        if name == "IMMELMANN":
            return 0.34 + 0.14 * high
        if name == "LOOP":
            return 0.32 + 0.16 * high
        if name == "COBRA":
            return 0.40 + 0.10 * near
        return 0.45

    # ---------- 人体过载耐受 ----------

    def _update_g_tolerance(self, obs: Observation) -> None:
        """按当前过载累积/恢复 +Gz 与 -Gz 疲劳，并更新可用过载上限。

        正过载超过 6 g 才开始累积，累积速度与超出量成正比；在 3 g 以下恢复。
        负过载的耐受与恢复都远差于正过载，因此用更短的时间尺度。
        任一方向累积到 1 就进入短暂失控（G-LOC / 红视）。
        """
        n = obs.own_n
        if n > G_SUSTAIN:
            self.g_load += (
                (n - G_SUSTAIN) / (G_PEAK - G_SUSTAIN)
                * self.dt / G_LOAD_TAU
            )
        elif n < 3.0:
            self.g_load -= self.dt / G_RECOVER_TAU
        self.g_load = clamp(self.g_load, 0.0, 1.0)

        if n < G_NEG_SUSTAIN:
            self.g_load_neg += (
                (G_NEG_SUSTAIN - n) / (G_NEG_SUSTAIN - G_NEG_PEAK)
                * self.dt / G_NEG_TAU
            )
        elif n > -0.5:
            self.g_load_neg -= self.dt / G_NEG_RECOVER_TAU
        self.g_load_neg = clamp(self.g_load_neg, 0.0, 1.0)

        if (self.g_load >= 1.0 or self.g_load_neg >= 1.0) and self._gloc_t <= 0.0:
            self._gloc_t = GLOC_TIME
        if self._gloc_t > 0.0:
            self._gloc_t = max(0.0, self._gloc_t - self.dt)

        # 疲劳后可用过载从短时极限向持续耐受值回落。
        self.n_pos_tol = G_PEAK - (G_PEAK - G_SUSTAIN) * self.g_load
        self.n_neg_tol = G_NEG_PEAK + (G_NEG_SUSTAIN - G_NEG_PEAK) * self.g_load_neg

    def _limit_g(self, act: Action) -> Action:
        """把人体耐受叠到指令上：超限削减过载，失控时只能松杆。"""
        if self._gloc_t > 0.0:
            return Action(bank=0.0, n=1.0, throttle=act.throttle, fire=False)
        if act.n > self.n_pos_tol:
            act.n = self.n_pos_tol
        elif act.n < self.n_neg_tol:
            act.n = self.n_neg_tol
        return act

    # ---------- 防御原语轮换 ----------

    def _should_defend(self, obs: Observation, sit: _Situation) -> bool:
        """被锁定时评估是否值得转入防御。

        只有在对方确实还有弹、锁定已完成、并且自己不是更好的进攻选择时
        才规避。这样会出现“先开火再防御”和“占优时继续压上”的差异，
        而不是一被锁定就固定逃跑。
        """
        if obs.foe_missiles <= 0 and obs.foe_radar_missiles <= 0:
            return False            # 对方没弹了，锁定没有威胁
        if obs.foe_lock_progress < 0.85:
            return False            # 还没形成稳定发射条件
        if obs.own_speed_frac < 1.35:
            return False            # 能量不够，规避也甩不掉
        # 自己的中距弹正在飞、目标还在照射锥内：先保持照射把它打完。
        # 只有来袭弹已进入近距、确实威胁生命时才中断照射转入规避。
        if (
            obs.own_radar_support
            and obs.radar_tracking
            and obs.incoming_missile_dist > self.MISSILE_BREAK_DIST
        ):
            return False
        if sit.name == "OFFENSIVE" and obs.own_ammo > 1:
            return False            # 自己正占进攻位，继续压上
        if (
            (obs.own_locked or obs.own_radar_locked)
            and obs.own_missiles_in_flight == 0
            and not obs.own_radar_support
            and (obs.in_missile_envelope or obs.in_radar_envelope)
        ):
            return False            # 先把自己的弹打出去，下一拍再评估
        return True

    def _pick_defense(self, obs: Observation, vs_missile: bool) -> str:
        """在防御候选池里按成功率选择规避动作，全程保持同一转向侧。

        远距优先横转消耗弹能量、中距保持横转并做高 G 修正、近距才切硬甩，
        这些战术倾向由 `_maneuver_prob` 的概率项表达，而不是固定的轮换
        顺序；末段刚建立起来的三九/横转姿态也不会被无收益地换掉。
        """
        if obs.own_speed_frac < self.LOW_ENERGY_FRAC:
            return "SPLIT_S" if obs.z_margin_low > 2400.0 else "BREAK"
        pool = self.MISSILE_DEFENSE if vs_missile else self.LOCK_DEFENSE
        cands = [c for c in pool if self._defense_ok(c, obs)]
        if not cands:
            return "BREAK" if vs_missile else "BEAM"
        return self._pick_maneuver(cands, None, obs, vs_missile)

    def _defense_ok(self, primitive: str, obs: Observation) -> bool:
        """防御机动的准入条件，全部取自机动层的能力声明。"""
        m = MANEUVERS.get(primitive)
        if m is None:
            return True
        if m.high_g and (
            obs.own_speed_frac <= 1.45
            or self.n_pos_tol <= 6.0
            or obs.own_n_avail < HIGH_G_MIN_AVAIL
        ):
            return False
        if obs.own_speed_frac < m.min_speed_frac:
            return False
        if m.low_alt_need and obs.z_margin_low <= m.low_alt_need:
            return False
        if m.high_alt_need and obs.z_margin_high <= m.high_alt_need:
            return False
        if m.ps_positive and obs.own_ps_level <= 0.0:
            return False
        if m.max_dist and obs.dist >= m.max_dist:
            return False
        return True

    def _anti_radar(self, obs: Observation) -> str | None:
        """雷达制导中距弹的应对：F-pole 优先，其次三九下高。

        FC3 手册的超视距防御流程：
        - F-pole：自己的中距弹仍在中制导、且预计比对方弹先命中时，保持约 40°
          偏置继续给弹装订，让自己的弹先落地；自己的弹进入末段自主截获
          （主动弹）或结束后再冷转脱离。
        - 三九下高：没有还手弹、或对方弹先到，把威胁放到正侧方并下高，
          既拉长弹道，又用更稠密的空气消耗弹的能量、压出对方制导锥。
        返回 None 表示已进入近距，交回原有的硬甩规避轮换。
        """
        ttg = obs.incoming_missile_ttg
        # 自己的雷达弹仍需要载机支持（中制导）且目标还在跟踪范围内：先 F-pole。
        # 横转 90° 会把自己的中制导断掉，偏置 40° 既保持支持又把对方弹道拉长，
        # 是唯一同时兼顾攻守的动作。
        if (
            obs.own_radar_support
            and obs.radar_tracking
            and ttg > 4.0
        ):
            return "F_POLE"
        if ttg > 14.0:
            return "CRANK"
        if (
            ttg > 4.5
            and obs.z_margin_low > 1500.0
            and obs.own_alt > 2500.0
            and obs.own_speed_frac > 1.35
        ):
            return "NOTCH_DESC"
        return None

    # BEAM 与 NOTCH_DESC 是同一套横转姿态的两档，分界点（威胁剩余时间 4.5 s）
    # 附近会来回跳。保持当前动作，直到出现完全不同的威胁应对需求。
    _NOTCH_FAMILY = frozenset(("BEAM", "NOTCH_DESC"))

    def _sticky_defense(self, want: str) -> str:
        if (
            want != self.primitive
            and self.primitive in self._NOTCH_FAMILY
            and want in self._NOTCH_FAMILY
        ):
            return self.primitive
        return want

    def _enter(self, primitive: str, obs: Observation) -> None:
        self.primitive = primitive
        self._hold = 0.0
        self._phase = 0
        self._phase_t = 0.0
        # 反击战术的转向侧由战术层决定：一律朝目标所在的一侧切，才能把
        # 机头甩回对手身上。防御动作则沿用“威胁在哪侧就往哪侧规避”。
        if self.tactic.startswith("COUNTER") or self.tactic == "RESET":
            sign = 1.0 if obs.off_az >= 0.0 else -1.0
            self._def_sign = sign
            self._scissor_sign = sign
            self._def_engaged = True
            self._def_reverse_t = 0.0
            self._counter_cd = self.COUNTER_COOLDOWN
        # 分段状态每次进入都要重置，否则上一次破 S 的 _dived 会漏进来。
        self._dived = False
        self._maneuver_done = primitive not in VERTICAL
        if primitive == "POSITION":
            self._open_alt0 = None
        # 防御转向只在本次威胁的第一次防御动作时定一次侧：取威胁所在的一侧。
        # 之后由 act() 的计时器统一维护，只在威胁持续位于反侧 1.5 s 才换向。
        # 原来“每次进入防御都按瞬时方位重算”，在盘旋缠斗里方位角常年超过
        # 60°，会导致每进一次防御就反向一次，且很快被下一拍策略替换。
        if primitive in (
            "BREAK", "BEAM", "SPLIT_S", "BARREL", "IMMELMANN", "LOOP",
            "CRANK", "NOTCH_DESC", "F_POLE",
            "HIGH_G_BREAK", "HIGH_G_SPLIT_S", "HIGH_G_BARREL",
        ) and not self._def_engaged:
            self._def_engaged = True
            if obs.incoming_missile_dist < self.SARH_EVADE_DIST:
                self._def_sign = 1.0 if obs.incoming_missile_az >= 0.0 else -1.0
            else:
                # 没有来袭弹时（战斗机动/被锁定的横转），转向侧跟随本次
                # 缠斗的既定转向侧。原来按瞬时方位重算，会和进攻盘旋的
                # 转向侧相反，于是“盘旋一下、急转一下”每次都要反向滚
                # 140°，画面里就是机体一直在左右摇摆。
                self._def_sign = self._turn_sign
            self._def_reverse_t = 0.0
        if primitive in (
            "LEAD_TURN", "ONE_CIRCLE", "TWO_CIRCLE", "HIGH_G_TURN",
        ):
            # 转侧只在第一次进入盘旋时定一次；之后由 act() 的换向计时器
            # 维护（威胁/目标持续位于反侧 1.5 s 才换向）。交汇方案有效时
            # 直接沿用方案侧，避免贴身盘旋里每换一次原语就反向滚一次。
            if self._merge_plan and self._plan_ttl > 0.0:
                self._turn_sign = self._merge_sign
                self._turn_locked = True
            elif not self._turn_locked:
                self._turn_sign = 1.0 if obs.off_az >= 0.0 else -1.0
                self._turn_locked = True
        if primitive == "SCISSORS":
            # 初始剪刀方向跟着威胁走，之后周期性反转。
            self._scissor_sign = 1.0 if obs.off_az >= 0.0 else -1.0
            self._scissor_t = self.SCISSOR_PERIOD
        if primitive in ("HIGH_YOYO", "LOW_YOYO"):
            self._yoyo_goal = None

    # ---------- 第三层：原语执行 ----------

    def _execute(self, primitive: str, obs: Observation) -> Action:
        act = self._dispatch(primitive, obs)
        act = self._limit_energy(act, obs, primitive)
        act = self._limit_g(act)
        pilot_ok = self._gloc_t <= 0.0
        act.launch = pilot_ok and should_launch(obs)
        act.launch_radar = pilot_ok and should_launch_radar(obs)
        return act

    def _limit_energy(
        self, act: Action, obs: Observation, primitive: str = "",
    ) -> Action:
        """能量保护：低速时把过载压到能维持速度的水平，避免持续掉速失速。

        高 G 原语是短时机动，本来就靠透支能量换角度，过载应由人体耐受
        （_limit_g 的 n_pos_tol）收口而不是被“维持速度”的软上限压住，
        否则高 G 机动永远拉不到 6 g 以上、形同普通盘旋。只在速度已经
        掉到接近失速时才重新套用能量保护，避免真的拉进失速。
        """
        m = MANEUVERS.get(primitive)
        if m is not None and m.high_g and obs.own_speed_frac > 1.30:
            return act
        if obs.own_speed_frac < self.LOW_ENERGY_FRAC:
            cap = max(1.0, obs.own_n_sustain * self.N_MARGIN)
            if act.n > cap:
                act.n = cap
        return act

    def _build_maneuver_table(self) -> dict[str, Callable[[Observation], Action]]:
        """第 2 层：机动名 → 执行函数。这里只做动作，不做选择。"""
        return {
            "RECOVER": self._recover,
            "POSITION": self._position,
            "LEAD_TURN": self._lead_turn,
            "ONE_CIRCLE": lambda o: self._circle(o, "ONE_CIRCLE"),
            "TWO_CIRCLE": lambda o: self._circle(o, "TWO_CIRCLE"),
            "HIGH_G_TURN": self._high_g_turn,
            "HIGH_G_BREAK": self._high_g_break,
            "HIGH_G_SPLIT_S": self._high_g_split_s,
            "HIGH_G_BARREL": self._high_g_barrel,
            "BEAM": self._beam,
            "PURE": lambda o: pursuit_action(o, o.off_az, o.off_el),
            "LEAD": lambda o: pursuit_action(o, *lead_point(o, 1.0)),
            "LAG": lambda o: pursuit_action(o, *lead_point(o, -0.6)),
            "BREAK": self._break,
            "SPLIT_S": self._split_s,
            "BARREL": self._barrel,
            "IMMELMANN": self._immelmann,
            "LOOP": self._loop,
            "SCISSORS": self._scissors,
            "HIGH_YOYO": self._high_yoyo,
            "LOW_YOYO": self._low_yoyo,
            "CRANK": self._crank,
            "NOTCH_DESC": self._notch_desc,
            "F_POLE": self._f_pole,
            "COBRA": self._cobra,
            "EXTEND": self._extend,
            "UNLOAD": self._unload,
        }

    def _dispatch(self, primitive: str, obs: Observation) -> Action:
        fn = self._maneuvers.get(primitive)
        if fn is None:
            return aim_action(obs, obs.off_az, obs.off_el)
        return fn(obs)

    def _recover(self, obs: Observation) -> Action:
        """边界/高度安全优先：先转向空域中心，再修正高度。

        低空大俯角时必须先把机头拉起来：只给一个 +12° 的目标航迹角，
        在 −70° 俯冲里等于没救（飞机在 1 s 内就掉 250 m），会直接撞地。
        """
        el_target = 0.0
        pull_out = obs.own_gamma < math.radians(-15.0)
        if obs.z_margin_low < max(1200.0, 3.5 * obs.own_speed):
            el_target = math.radians(25.0) if pull_out else math.radians(12.0)
        elif obs.z_margin_high < 600.0:
            el_target = math.radians(-12.0)
        return aim_action(
            obs,
            wrap_pi(obs.center_az - obs.own_psi),
            el_target - obs.own_gamma,
            gain=2.5,
        )

    def _position(self, obs: Observation) -> Action:
        """开局占位：按“能量战 / 角度战”的机型倾向选择接近方式。

        不再固定侧向展开，而是先决定怎么打：
        - 能量型机体（energy_pref 高，如 Su-57）：小幅偏置保持雷达照射，
          同时爬升存能量，把高度优势留到交汇后做高 Yo-Yo 从上方压下来。
        - 角度型机体（energy_pref 低，如 F-16V）：平飞加速接近角点速度，
          拉开更大的横向间隔，便于交汇前就拉出领先转弯、抢先指向对手。
        偏置方向、幅度、爬升角与油门都在机型允许的范围内随机取样（见
        __init__），因此每次启动的开局几何都不相同，不会固定成一个模板。
        """
        el_target = self._open_climb
        # 占位爬升要有“天花板”：48 km 的接近段若一直按 12° 爬升，会一路爬到
        # 能量升限，之后只能被强制 UNLOAD。这里按剩余距离与剩余爬升预算
        # （最多约 1.2 km）反解目标航迹角，越接近交汇越平，接近段自然改平。
        if self._open_alt0 is None:
            self._open_alt0 = obs.own_alt
        gain_left = self._open_climb_gain - (obs.own_alt - self._open_alt0)
        if gain_left <= 0.0 or obs.own_alt > 13500.0:
            el_target = min(el_target, 0.0)
        else:
            dist_left = max(obs.dist - self.MERGE_START, 500.0)
            el_target = min(el_target, math.atan2(gain_left, dist_left))
        # 能量型只有 Ps 还为正时才爬升，否则先平飞把速度补回来。
        if self.energy_pref >= 0.5 and obs.own_ps_level <= 5.0:
            el_target = min(el_target, math.radians(2.0))
        # 爬升目标按 3°/s 平滑跟随：Ps 在门限附近每拍翻转时，原来会让俯仰
        # 指令在 2° 与 12° 之间整档跳变，飞机会一直点头。
        step = math.radians(3.0) * self.dt
        self._open_climb_cur += clamp(
            el_target - self._open_climb_cur, -step, step
        )
        act = aim_action(
            obs,
            obs.off_az + self._open_side * self._open_offset,
            self._open_climb_cur - obs.own_gamma,
            gain=1.2,
            throttle=self._open_throttle,
        )
        # 占位只是“摆好接近角度”，不需要压大坡度、拉高过载。原来一开局就
        # 8 g 拉起来再掉回去，观感上就是两架飞机不停抖姿态。
        act.bank = clamp(act.bank, -0.35, 0.35)
        # 坡度被限幅后必须同步压低过载：aim_action 是按未限幅的坡度反解的，
        # 直接沿用它的 n，n·cosφ（φ 已被截到 28°）会多出接近 1 g 的竖直分量，
        # 飞机就会在占位段以 2°/s 一路爬升，几十秒后冲到升限被迫 UNLOAD。
        phi_act = abs(act.bank) * MAX_BANK
        n_hold = math.cos(self._open_climb_cur) / max(math.cos(phi_act), 0.2)
        act.n = clamp(min(act.n, n_hold + 0.10), 0.6, 2.2)
        return act

    # ---------- 对头抢位与高 G 机动（任务1/2） ----------

    def _make_merge_plan(self, obs: Observation) -> None:
        """交汇前锁定抢位方案：转向侧 + 单环/双环。

        转向侧取相对方位的符号，即朝对手所在一侧切入；交汇瞬间机头已经
        领先对手的航向，形成“前导追击”的几何。
        单环：瞬时角速度占优 + 偏好角度的机体，用首圈角度优势速战速决。
        双环：持续盘旋更可靠或偏好能量的机体，用可维持的过载慢慢磨角度。
        """
        self._merge_sign = 1.0 if obs.off_az >= 0.0 else -1.0
        instant_edge = obs.own_n_avail / max(obs.own_n_sustain, 0.5)
        # 偏好角度战的机体（如 F-16V）走单环，用首圈角度优势速战速决；
        # 偏好能量战的机体（如 Su-57）走双环，用稳定盘旋磨角度并保留能量。
        # 瞬时转弯能力明显占优的机体即使偏能量，也可以改走单环。
        if self.energy_pref < 0.5 or instant_edge > 1.45:
            self._merge_plan = "ONE_CIRCLE"
        else:
            self._merge_plan = "TWO_CIRCLE"
        self._plan_ttl = self.MERGE_TIMEOUT

    def _lead_turn(self, obs: Observation) -> Action:
        """领先转弯：交汇前提前拉杆，把机头指向对手未来航线的内侧。

        转弯圆环博弈的核心动作——更早转、转得更紧的一方先切进内圈，
        从而先于对手构成射击条件。转向侧在进入时已锁定。
        """
        n = min(obs.own_n_avail, self.n_pos_tol)
        return level_turn(obs, self._turn_sign, n, gamma_target=0.0)

    def _circle(self, obs: Observation, plan: str) -> Action:
        """单环 / 双环的选择执行。

        单环：拉满瞬时过载，双方对头互相切入，靠首圈角度优势先指向对方。
        双环：用能长期维持的过载同向盘旋，逐步把对手甩到机头前方。
        """
        if plan == "ONE_CIRCLE":
            n = min(obs.own_n_avail, self.n_pos_tol)
            return level_turn(obs, self._turn_sign, n, gamma_target=0.0)
        n = min(max(obs.own_n_sustain * 1.10, 2.0), self.n_pos_tol, G_SUSTAIN)
        return level_turn(obs, self._turn_sign, n, gamma_target=math.radians(-2.0))

    def _high_g_turn(self, obs: Observation) -> Action:
        """高 G 转弯（进攻）：角点速度附近拉满瞬时过载抢角度。"""
        n = min(obs.own_n_avail, self.n_pos_tol, HIGH_G_N)
        return level_turn(obs, self._turn_sign, n, gamma_target=0.0)

    def _high_g_break(self, obs: Observation) -> Action:
        """高 G 急转（防御）：最大瞬时过载拉满，略微下压保持速度。"""
        n = min(obs.own_n_avail, self.n_pos_tol, HIGH_G_N)
        return level_turn(obs, self._def_sign, n, gamma_target=math.radians(-4.0))

    def _high_g_split_s(self, obs: Observation) -> Action:
        """高 G 破 S：滚转倒扣后拉满过载俯冲，快速掉高度换速度并转向。"""
        n = min(obs.own_n_avail, self.n_pos_tol, HIGH_G_N)
        return self._split_s_impl(obs, n)

    def _high_g_barrel(self, obs: Observation) -> Action:
        """高 G 载重滚转：拉满过载的同时连续滚转，形成高 G 螺旋下降。"""
        n = min(obs.own_n_avail, self.n_pos_tol, HIGH_G_N)
        return self._barrel_impl(n, period=3.2)

    def _level_cap(self, obs: Observation, frac: float = 1.0) -> float:
        """维持当前航迹角所需的过载上限。

        规避动作取它的一个比例（<1）就能保证不爬升：3DOF/6DOF 里
        坡度偏小时若仍拉正过载，飞机会直接转成爬升，能量掉得比被弹
        打中还快。按实际坡度算上限，可以让坡度回中的瞬间自动卸载。
        """
        return frac * math.cos(obs.own_gamma) / max(math.cos(obs.own_bank), 0.15)

    def _break(self, obs: Observation) -> Action:
        """最大过载急转，方向在进入时锁定，同时略微下压保持速度。"""
        n = min(obs.own_n_avail, self.n_pos_tol)
        return level_turn(obs, self._def_sign, n, gamma_target=math.radians(-4.0))

    def _beam(self, obs: Observation) -> Action:
        """被锁定后的横转：向锁定一侧压坡度，把对方甩到视场边缘。"""
        n = min(obs.own_n_avail * 0.95, self.n_pos_tol)
        return level_turn(obs, self._def_sign, n, gamma_target=math.radians(-2.0))

    def _split_s(self, obs: Observation) -> Action:
        """半滚倒转：先滚成倒扣，再拉杆经垂直向下改出到反向航向。"""
        n = min(obs.own_n_avail * 0.9, self.n_pos_tol, G_PEAK)
        return self._split_s_impl(obs, n)

    def _split_s_impl(self, obs: Observation, n_max: float) -> Action:
        """破 S 标准三段：半滚倒扣 → 副翼中立拉杆过垂直 → 拉平改出。

        旧实现只把坡度命令到 80°（Action.bank=±1 的满刻度），从来没有真正
        倒扣：之后所谓的“拉杆”只是一个大坡度平飞盘旋，看上去就是平飞转向。
        现在第一段直接命令到 180° 倒扣，第二段用 roll_free 让机体在垂直面内
        连续下俯（否则内回路会把滚转角拽回地平线、根本过不了垂直）。
        用 own_gamma（asin 表示，±90°）判段：越过 −70° 后再回到 0 附近，
        说明半圈已经拉完，机头回到水平、航向反转。
        """
        sign = self._def_sign
        if self._phase == 0:
            # 半滚倒扣：滚到机腹朝上，期间不拉杆，避免滚入过程变成爬升。
            if abs(obs.own_bank) >= math.radians(160.0) or self._phase_t >= 4.5:
                self._phase, self._phase_t = 1, 0.0
                self._dived = False
            return Action(bank=sign * BANK_INVERT, n=1.0, throttle=1.0, fire=False)
        if self._phase == 1:
            if obs.own_gamma <= math.radians(-70.0):
                self._dived = True
            if self._dived and abs(obs.own_gamma) < math.radians(18.0):
                self._phase, self._phase_t = 2, 0.0
            return Action(bank=0.0, n=n_max, throttle=1.0, fire=False,
                          roll_free=True)
        # 改出：坡度归零、航迹拉平，确认回到平飞后结束整个机动。
        if abs(obs.own_gamma) < math.radians(8.0) and abs(obs.own_bank) < math.radians(30.0):
            self._maneuver_done = True
        return aim_action(obs, 0.0, -obs.own_gamma, gain=1.6, throttle=0.9)

    def _barrel_impl(self, n_max: float, period: float) -> Action:
        """桶滚：绕速度矢量连续滚转 360°，航迹画成螺旋。

        原来用 sin() 让坡度在小范围内来回摆，实际是“左右摇翅膀”，既不像
        桶滚也躲不开导弹。这里直接给滚转角速度，配合持续正过载，机体在
        滚转中把航迹拉成螺旋，才是标准防御桶滚。
        """
        if self._phase_t >= period:
            self._maneuver_done = True
        return Action(
            bank=0.0,
            n=n_max,
            throttle=1.0,
            fire=False,
            roll_rate=self._def_sign * 2.0 * math.pi / period,
        )

    def _barrel(self, obs: Observation) -> Action:
        """防御性滚筒（标准桶滚）。"""
        n = min(max(obs.own_n_sustain * 1.15, 2.5), self.n_pos_tol)
        return self._barrel_impl(n, period=4.2)

    def _immelmann(self, obs: Observation) -> Action:
        """殷麦曼：拉起半筋斗到倒扣平飞，再半滚改平，航向反转。

        旧实现只爬到 γ≈45° 就平转 150°，既没有过顶也没有真正的半滚，
        在能量上等价于一次大坡度盘旋。标准殷麦曼是“半个筋斗 + 半个滚转”：
        拉起（副翼中立）到机头越过顶点、重新回到平飞但机腹朝上，
        再滚转 180° 翻正，最后小幅度拉平。
        γ 用 asin 表示（±90°），过顶后会从 +90° 侧回到 0 附近。
        """
        n = min(obs.own_n_avail * 0.9, self.n_pos_tol, G_PEAK, VERTICAL_N)
        if self._phase == 0:
            # 拉起：副翼中立，避免内回路在过顶时把滚转角拽回地平线。
            if obs.own_gamma >= math.radians(70.0):
                self._phase, self._phase_t = 1, 0.0
                self._dived = True
            return Action(bank=0.0, n=n, throttle=1.0, fire=False, roll_free=True)
        if self._phase == 1:
            # 越过顶点后继续拉，直到机头重新回到水平（此时为倒扣平飞）。
            if abs(obs.own_gamma) < math.radians(15.0):
                self._phase, self._phase_t = 2, 0.0
            return Action(bank=0.0, n=n, throttle=1.0, fire=False, roll_free=True)
        if self._phase == 2:
            # 半滚翻正：此时机体仍是倒扣（φ≈±180°），要指令回翼平，
            # 不能再指令到倒扣——那样“已经倒扣”会立刻被判完成，等于没翻。
            if abs(obs.own_bank) < math.radians(25.0) or self._phase_t >= 4.5:
                self._phase, self._phase_t = 3, 0.0
            return Action(bank=0.0, n=1.0, throttle=1.0, fire=False)
        if abs(obs.own_gamma) < math.radians(8.0):
            self._maneuver_done = True
        return aim_action(obs, 0.0, -obs.own_gamma, gain=1.6, throttle=0.8)

    def _loop(self, obs: Observation) -> Action:
        """内筋斗：翼平持续拉杆，在垂直平面内完成 360° 回转。

        机身过顶时必须让滚转自由演化（roll_free），否则内回路会一直把
        机体往“地平线翼平”方向拧，筋斗会被拉成螺旋。
        γ（asin 表示）的完整顺序是 +90° → 0（倒扣平飞）→ −90° → 0，
        据此分四段，最后一段拉平。
        """
        if self._phase == 0:
            if obs.own_gamma >= math.radians(75.0) or self._phase_t >= 3.0:
                self._phase, self._phase_t = 1, 0.0
        elif self._phase == 1:
            if abs(obs.own_gamma) <= math.radians(15.0):
                self._phase, self._phase_t = 2, 0.0
        elif self._phase == 2:
            if obs.own_gamma <= math.radians(-70.0):
                self._phase, self._phase_t = 3, 0.0
        if self._phase >= 3:
            if abs(obs.own_gamma) < math.radians(8.0):
                self._maneuver_done = True
            return aim_action(obs, 0.0, -obs.own_gamma, gain=1.6)
        n = max(1.0, min(obs.own_n_avail * 0.9, self.n_pos_tol, G_PEAK, VERTICAL_N))
        return Action(bank=0.0, n=n, throttle=1.0, fire=False, roll_free=True)

    def _scissors(self, obs: Observation) -> Action:
        """滚转剪刀：周期性反向最大过载转弯，逼对手冲过头。"""
        if self._scissor_t <= 0.0:
            self._scissor_sign = -self._scissor_sign
            self._scissor_t = self.SCISSOR_PERIOD
        n = min(obs.own_n_avail * 0.95, self.n_pos_tol)
        return level_turn(obs, self._scissor_sign, n, throttle=0.9)

    def _yoyo_pull_n(
        self, obs: Observation, gamma_goal: float, t_pull: float
    ) -> float:
        """按“要抬多少度、用多久抬到位”反解过载，而不是写死一个 g。

        需要快速改变指向（接近率大、可能冲过头、机头偏得多）时把时间预算
        压短，过载自然升高；还在从容跟随时预算放长，只需要很小的正过载，
        把能量留着换高度。结果仍受可用过载与人体耐受收口。
        """
        dgamma = max(gamma_goal - obs.own_gamma, 0.0)
        v = max(obs.own_speed, 40.0)
        rate = dgamma / max(t_pull, 0.4)
        n = math.cos(obs.own_gamma) + v * rate / G
        return clamp(n, 1.0, min(obs.own_n_avail, self.n_pos_tol))

    def _high_yoyo(self, obs: Observation) -> Action:
        """高 Yo-Yo：接近率过大、快要冲过头时，先把速度换成高度，再从上方
        压回目标的前置线。

        标准高 Yo-Yo 只在一次追击里做一遍：拉起换高度 → 俯冲把机头压回
        前置线 → 结束，之后转回普通追逐。抬升量和过载都由态势决定：
        接近率越大、目标偏角越大，说明越可能冲过头、越需要赶快把机头挪到
        目标前方，抬得越高、拉得越狠；远距慢速跟随时只做一次轻抬。
        原来固定 3.2 g 是两头都不对：慢速跟随时白白耗掉速度，高速快
        冲过头时又抬不起来。
        """
        az, el = lead_point(obs, 1.0)
        if self._phase == 0:
            if self._yoyo_goal is None:
                # 接近率归一化：80 m/s 以上按“快要冲过头”处理。
                closure = clamp(obs.closure / 80.0, 0.0, 1.0)
                # 机头偏差越大，越需要抬得高一点才挪得回前置线。
                point = clamp(abs(az) / math.radians(60.0), 0.0, 1.0)
                ps_climb = math.asin(
                    clamp(obs.own_ps / max(obs.own_speed, 40.0), -0.95, 0.95)
                )
                # 高 Yo-Yo 本来就是拿动能换高度，Ps 允许值之外还要加一段
                # “速度富余可以 zoom”的额度；否则 9 km、320 m/s 时 Ps 只够
                # 5°，抬升会被压成一次几乎平飞的小抖动。
                zoom = (
                    clamp((obs.own_speed_frac - 1.30) / 0.80, 0.0, 1.0)
                    * math.radians(28.0)
                )
                cap = max(ps_climb, math.radians(5.0)) + zoom
                self._yoyo_goal = clamp(
                    math.radians(10.0 + 18.0 * closure + 8.0 * point),
                    math.radians(6.0), min(cap, GAMMA_CAP),
                )
                # 时间预算：贴得越近越拖不起，必须更猛地抬头；远距可以从容拉。
                self._yoyo_t = clamp(obs.dist / 900.0, 0.8, 2.5)
            # 方位仍用坡度跟，幅度限在 32° 内，避免高速下零点几度误差就把
            # 坡度顶到 80°，把高 Yo-Yo 做成一次急转。
            bank = clamp(az / math.radians(45.0), -0.4, 0.4)
            n = self._yoyo_pull_n(obs, self._yoyo_goal, self._yoyo_t)
            if (
                obs.own_gamma >= self._yoyo_goal - math.radians(2.0)
                or self._phase_t >= self._yoyo_t + 1.5
            ):
                self._phase, self._phase_t = 1, 0.0
            return Action(bank=bank, n=n, throttle=1.0, fire=False)
        # 俯冲回收：机头重新压回目标前置线（el≈0）即完成，进入冷却。
        err = clamp(el, -math.radians(8.0), 0.0)
        if abs(el) < math.radians(6.0) or self._phase_t >= 3.5:
            self._maneuver_done = True
            self._yoyo_cd = 6.0
        return pursuit_action(obs, az, err, gain=1.2, throttle=0.7)

    def _low_yoyo(self, obs: Observation) -> Action:
        """低 Yo-Yo：距离拉开、速度不足时，向目标转弯内侧俯冲增速，再把
        机头拉回前置线，缩短咬尾距离。

        俯冲深度同样按态势给：速度越接近失速、距离越远，越需要狠狠推杆
        增速（卸载到 0.25 g）；速度还够、只是距离略远时轻压机头即可
        （0.9 g）。俯冲段同时把机头往目标前置线带，偏角越大坡度越大。
        原来固定 0.35 g 无坡度俯冲，速度富余时等于白掉高度，方位上也没
        跟上目标的转弯。
        """
        az, el = lead_point(obs, 1.0)
        if self._phase == 0:
            if self._yoyo_goal is None:
                # 速度越不足、距离越远，越要往深里扎；两者取较大的需求。
                speed_need = clamp((1.55 - obs.own_speed_frac) / 0.45, 0.0, 1.0)
                range_need = clamp((obs.dist - 900.0) / 1600.0, 0.0, 1.0)
                dive = max(speed_need, range_need)
                self._yoyo_goal = -math.radians(10.0 + 12.0 * dive)
                # 卸载量：需求越大越接近推杆；0.85 g 只是轻压机头。
                self._yoyo_n = clamp(0.85 - 0.70 * dive, 0.10, 0.85)
            bank = clamp(az / math.radians(50.0), -0.45, 0.45)
            if (
                obs.own_gamma <= self._yoyo_goal + math.radians(2.0)
                or self._phase_t >= 5.0
            ):
                self._phase, self._phase_t = 1, 0.0
            return Action(bank=bank, n=self._yoyo_n, throttle=0.75, fire=False)
        err = clamp(el, 0.0, math.radians(8.0))
        if abs(el) < math.radians(6.0) or self._phase_t >= 3.5:
            self._maneuver_done = True
            self._yoyo_cd = 5.0
        return pursuit_action(obs, az, err, gain=1.2)

    def _extend(self, obs: Observation) -> Action:
        """脱离：掉头背离对手，满油门拉开距离。"""
        away = wrap_pi(obs.los_az + math.pi - obs.own_psi)
        # 用比例式航向控制，航迹角只压到略低于水平。原来不做限制，大坡度
        # 掉头时会先拉 7 g 把速度换成高度，再挂着机头慢慢掉回来，看起来
        # 就是“逃跑时不断爬升、同时掉速”。
        return heading_turn(
            obs, away, gain=0.6, max_rate_deg=5.0,
            gamma_target=math.radians(-5.0),
        )

    def _crank(self, obs: Observation) -> Action:
        """偏置（CRANK）：把目标压在机头约 40°，保持照射同时拉开横向距离。

        40° 仍在中距弹 60° 的照射锥内，所以可以在偏置的同时继续制导自己的弹；
        横向位移会让对方弹的视线角速度变大、需要更多过载去追。

        方位误差必须是“当前方位 − 期望方位”。aim_action 的输入是
        “还要往哪边转多少”，写成“期望 − 当前”会变成正反馈：飞机一直
        往同一侧压坡度，方位越转越大，最后把机头甩出 60° 照射锥。
        """
        az = obs.off_az - self._def_sign * math.radians(40.0)
        if abs(az) < math.radians(4.0):
            az = 0.0
        return heading_turn(
            obs, az, gain=0.35, max_rate_deg=3.0,
            gamma_target=math.radians(-3.0),
        )

    def _notch_desc(self, obs: Observation) -> Action:
        """三九下高：把威胁放到正侧方（3/9 线）并下高。

        横转拉长对方的拦截几何、下高进入更稠密的空气消耗弹的能量；
        同时让自己的机头逐渐脱离对方雷达照射锥的跟随。
        """
        az = obs.incoming_missile_az - self._def_sign * math.radians(90.0)
        if abs(az) < math.radians(5.0):
            az = 0.0
        return heading_turn(
            obs, az, gain=0.45, max_rate_deg=4.0,
            gamma_target=math.radians(-6.0),
        )

    def _f_pole(self, obs: Observation) -> Action:
        """F-pole：先偏置保持中制导，把自己的中距弹导完，再冷转脱离。

        第一阶段与 CRANK 相同；自己的弹进入末段自主截获或结束后
        （own_radar_support 为假），立刻掉头拉开距离，让对手剩余的
        在飞弹多飞一段、直到燃料耗尽或丢制导。
        """
        if self._phase == 0:
            if not obs.own_radar_support:
                self._phase, self._phase_t = 1, 0.0
            else:
                return self._crank(obs)
        away = wrap_pi(obs.los_az + math.pi - obs.own_psi)
        return heading_turn(
            obs, away, gain=0.6, max_rate_deg=5.0,
            gamma_target=math.radians(-6.0),
        )

    def _cobra(self, obs: Observation) -> Action:
        """过失速眼镜蛇：把机头拉到失速迎角以上，靠矢量喷口维持姿态。

        近距来弹时用来制造过冲：导弹的导引头还追着原来的速度矢量，
        机头已经抬起、速度快速衰减，弹会从下方冲过去。3DOF 下没有迎角
        状态，这个指令不会生效，所以只对 6DOF + 矢量喷口机型开放。
        """
        return Action(bank=0.0, n=1.0, throttle=0.4, fire=False, post_stall=True)

    def _unload(self, obs: Observation) -> Action:
        """失速边缘的补救：压平机翼、推杆改出，用高度换回速度。"""
        return Action(bank=0.0, n=-1.2, throttle=1.0, fire=False)
