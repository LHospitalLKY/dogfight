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

from sim import G, MAX_BANK, Action, Observation, clamp, predict_path, wrap_pi

# 期望角速度的软上限，避免目标突然跳到正后方时拉出离谱的指令。
MAX_GAMMA_DOT = math.radians(20.0)
# 能量-高度约束：爬升角上限由比剩余功率 Ps 决定，高速时才允许额外 zoom。
ZOOM_ALLOWANCE = math.radians(25.0)
GAMMA_CAP = math.radians(55.0)
GAMMA_FLOOR = math.radians(-60.0)
# 倒扣滚转指令：Action.bank 以 MAX_BANK(80°) 为单位，倒扣需要 180°/80°。
BANK_INVERT = math.pi / MAX_BANK
# 破 S 横偏变体（DIVE_OFFSET）在俯冲段保留的偏转角：比完全倒扣少 40°，
# 升力矢量同时有向下与侧向分量，机体一边下高一边朝对手转弯的内侧偏。
# 这不是 DIVE_BREAK：DIVE_BREAK 全程与标准破 S 共用 _split_s_impl，差别
# 只在改出段——先压一个坡度再拉平（横滚 + 拉杆）来偏出航向。
DIVE_OFFSET_BANK = math.radians(40.0) / MAX_BANK
# 破 S 偏转（DIVE_BREAK）：改出段先压这个坡度，再在坡上拉杆把航迹拉平，
# 升力矢量的侧向分量把机头带向一侧，改出航向在 180° 反向之外再偏出几十度。
DIVE_BREAK_BANK = math.radians(45.0) / MAX_BANK

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
    "BARREL", "HIGH_G_BARREL", "DIVE_BREAK", "DIVE_OFFSET",
    "KVOCHUR_BELL", "KVOCHUR_BELL_BACK", "HERBST",
))
# 钟摆家族：拉起 → 零速尾滑 → 顶点翻落。前倒（过顶）与后倒（退回原航向）
# 共用同一套拉起/尾滑时序，只在翻落段的俯仰指令上分叉。
BELL_PRIMS = ("KVOCHUR_BELL", "KVOCHUR_BELL_BACK")
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
    "LURE": Maneuver("LURE", "OFFENSIVE"),
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
    # 破 S 横偏变体：俯冲段保留 40° 坡度，边下高边横偏（见 _dive_offset）。
    # 比标准破 S 少掉一点高度，离地余量门限可以更低。
    "DIVE_OFFSET": Maneuver(
        "DIVE_OFFSET", "DEFENSIVE", min_speed_frac=1.40, low_alt_need=3000.0,
    ),
    # 破 S + 机头偏转：与标准破 S 共用 _split_s_impl，只在改出段先压 45°
    # 坡度再拉平（横滚 + 拉杆），侧向升力把机头带偏，改出航向不走纯反向。
    "DIVE_BREAK": Maneuver(
        "DIVE_BREAK", "DEFENSIVE", min_speed_frac=1.40, low_alt_need=3000.0,
    ),
    # 克沃丘尔钟摆（真·尾滑/bell）：高 G 拉起至近垂直，速度降到零附近
    # 进入尾滑，机头自然甩过地平线，恢复末段加一次绕纵轴滚转（6DOF+TVC）。
    "KVOCHUR_BELL": Maneuver(
        "KVOCHUR_BELL", "COUNTER", min_speed_frac=1.60,
        low_alt_need=2400.0, high_alt_need=2800.0,
    ),
    # 后倒变体：顶点后反向推杆，机头沿原爬升路径倒回来，尾冲后退后改平，
    # 退出航向基本不变；只用于演示/能量管理，不进战术反击池。
    "KVOCHUR_BELL_BACK": Maneuver(
        "KVOCHUR_BELL_BACK", "DEFENSIVE", min_speed_frac=1.60,
        low_alt_need=2400.0, high_alt_need=2800.0,
    ),
    # Herbst / J 转弯：拉起 55° 后绕速度矢量横滚，反向压机头完成 180° 反转。
    "HERBST": Maneuver(
        "HERBST", "COUNTER", min_speed_frac=1.60,
        low_alt_need=2000.0, high_alt_need=2000.0,
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
    "GUN_LEAD": Maneuver("GUN_LEAD", "GUNS"),
    "GUN_PRESS": Maneuver("GUN_PRESS", "GUNS"),
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

    a = cg + v * gamma_dot / G              # 垂直方向升力分量
    if a < 0.0 and obs.own_gamma > math.radians(30.0):
        # 机头已经高高抬起，而能量只允许更小的爬升角：必须推杆（负过载）
        # 把航迹压下来。旧的 max(a, 0.12) 把推杆指令吃掉了，低速大航迹角
        # 时实际 γ̇ 只剩 +0.3°/s，飞机会“挂”在机头上以 70–80° 缓慢爬升
        # （近失速爬升但不掉速的僵持）。推杆时限幅在人体短时负过载极限。
        return Action(
            bank=0.0,
            n=clamp(a, -2.5, -0.05),
            throttle=throttle,
            fire=obs.in_gun_trigger,
        )
    a = max(a, 0.12)
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
        fire=obs.in_gun_trigger,
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
    """预测目标的未来位置，返回相对本机的指向偏差 (方位, 仰角)。

    目标位置按它当前的航迹角速度 ψ̇/γ̇ 积分（见 sim.predict_path），而不是
    只沿速度切线外推：速度方向永远只是轨迹的切线，目标正在回旋时切线会
    一路甩到圆外，提前量越算越偏。直线飞行时预测退化为匀速外推。
    """
    ce = math.cos(obs.los_el)
    ux, uy, uz = ce * math.cos(obs.los_az), ce * math.sin(obs.los_az), math.sin(obs.los_el)

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
    lx, ly, lz = predict_path(
        (ux * obs.dist, uy * obs.dist, uz * obs.dist),
        obs.foe_speed, obs.foe_track_psi, obs.foe_track_gamma,
        obs.foe_psi_rate, obs.foe_gamma_rate, t_go,
    )

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
    # aim_action 在“机头高抬、能量不允许继续爬”时会给出负过载推杆，
    # 这里不能再按最小 1 g 兜底，否则推杆指令会被拉回平飞、飞机继续挂在机头上。
    act.n = min(n_target, act.n) if act.n > 0.0 else act.n
    act.throttle = throttle
    act.fire = obs.in_gun_trigger
    return act


def heading_turn(
    obs: Observation,
    heading_error: float,
    gain: float = 0.35,
    max_rate_deg: float = 3.0,
    gamma_target: float = 0.0,
    throttle: float = 1.0,
    rate_ff: float = 0.0,
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
    rate_ff 是前馈转弯率：目标视线本身在旋转时（目标正在转弯），
    本机必须以同样的角速度跟着转，否则只剩比例项、永远追不上，
    误差会在偏置角两侧持续振荡。
    """
    v = max(obs.own_speed, 40.0)
    psi_dot = clamp(
        rate_ff + heading_error * gain,
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
        fire=obs.in_gun_trigger,
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
    if abs(sign) < 1e-6:
        # 不需要转弯（sign=0，翼平）：竖直方向升力分量就等于过载本身，
        # 不能再套用“先滚转、后拉杆”的坡度渐入。否则翼平还挂着 n_level
        # （6 g / 3.2 g）等于纯爬升：低油量恢复时航迹角一路从 11° 爬到
        # 27°，随后又因为坡度误差变化把过载从 1.6 g 跳到 6 g，看着就是
        # 莫名其妙的高 G 抖动。
        n_cmd = clamp(math.cos(obs.own_gamma) + v * gd / G, 0.2, n_level)
    return Action(
        bank=clamp(sign * phi / MAX_BANK, -math.radians(86.0) / MAX_BANK,
                   math.radians(86.0) / MAX_BANK),
        n=n_cmd,
        throttle=throttle,
        fire=obs.in_gun_trigger,
    )


def level_out_action(obs: Observation, throttle: float = 0.9) -> Action:
    """垂直机动改出：先把机翼滚回水平，再小增益、限幅地把航迹角拉平。

    破 S / Herbst 这类动作结束时坡度还停在 130°–180°，此时用
    aim_action(gain=1.6) 直接追“航迹角回零”会出事：倒扣状态下正过载
    的升力指向地面，拉杆反而把 γ 往负方向顶；每拍算出的俯仰需求在
    0.1 g 与 9 g 之间来回翻，坡度还没滚回来，机体就先抖成一片。
    这里按两段拆开：
      1) |bank|>30° 时只给“保持当前航迹角”的过载并滚回水平——倒扣时
         该过载自然是负的（推杆保持平飞），不会把飞机往地面拽；
      2) 机翼回中后再用 10°/s 的限幅目标角速度拉平航迹，避免高速下
         一度误差就顶到 9 g 的过冲。
    """
    v = max(obs.own_speed, 40.0)
    # 第一段：坡度还大，先滚平。cos(φ)<0 说明升力矢量朝下，按“维持航迹”
    # 反解出的过载就是负值（推杆），限制在人体短时负过载以内。
    if abs(obs.own_bank) > math.radians(30.0):
        n_hold = math.cos(obs.own_gamma) / max(math.cos(obs.own_bank), 0.25)
        return Action(
            bank=0.0,
            n=clamp(n_hold, G_NEG_SUSTAIN, 2.5),
            throttle=throttle,
            fire=obs.in_gun_trigger,
        )
    # 第二段：机翼已回中，限幅航迹角速度拉平。上限 10°/s 对应 300 m/s
    # 下约 5.7 g，从 −45° 俯冲拉平约 4–5 s；比例增益取 0.9，误差变小过载
    # 自动回落，不会像 gain=1.6 那样冲过平飞再反向推杆。
    gd = clamp(-obs.own_gamma * 0.9,
               -math.radians(10.0), math.radians(10.0))
    a = math.cos(obs.own_gamma) + v * gd / G
    # 从 12° 的爬升角压回平飞需要短暂的负过载（推杆），若把 n 钳在正侧，
    # 低速度下每拍只能把航迹压 2–3°/s，飞机就一直挂在爬升角上出不来。
    n = clamp(a / max(math.cos(obs.own_bank), 0.6), G_NEG_SUSTAIN, G_SUSTAIN)
    return Action(
        bank=0.0,
        n=n,
        throttle=throttle,
        fire=obs.in_gun_trigger,
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
        # 基线不搞双发节奏：一次只打一发，补射交给战术层的围堵规则。
        act.launch_radar = should_launch_radar(obs) and obs.own_radar_in_flight == 0
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
        # 发射纪律：超过 30° 离轴的快照弹在 9 km 高度打不动——弹在 M2 时
        # 转弯半径约 1.6 km，目标就在 1.5 km 时 40° 以上的离轴角物理上转不进去。
        and math.degrees(obs.angle_off_nose) <= 30.0
    )


# 中距弹发射门限（按当前几何估计的命中概率）：不再要求目标已经进入 NEZ。
# 0.22 约对应有效包线的 0.7–0.8，弹离架后要先飞 5–8 s 中制导段才进入末段
# 自主截获，载机因此有真实的引导/照射过程；门限再高就退回 NEZ 内发射，
# 弹一离架就截获，等于没有中制导。更低的门限则是远距乱射，弹飞到末端
# 已经没有能量。
RADAR_LAUNCH_PK = 0.22

# 中距弹发射离轴上限。雷达能在 60° 锥内“锁定”，但从 40° 以上的偏置姿态
# （F-pole/CRANK 的 40° 偏置）打出去的弹，要先花好几秒把速度矢量掰向目标，
# 末段能量全耗在初始转弯上；更常见的是弹还没转过来，目标已经出了导引头
# 视场。20° 要求机头基本压住目标才开火——旧值 30° 仍放行了 28° 的补射，
# 正是“抬头显示器里目标已经贴边、还按发射”的那种浪费弹药的情形。
RADAR_LAUNCH_ANGLE = math.radians(20.0)


def should_launch_radar(obs: Observation) -> bool:
    """雷达制导中距弹的通用发射门限：几何、包线、能量与弹量。

    这里只管“这一发此刻有没有意义”，双发节奏由 TacticalAgent
    「围追堵截」的补射规则决定（见 TacticalAgent._radar_launch_ok）。
    """
    if (
        not obs.own_radar_locked
        or obs.own_radar_missiles <= 0
        or not obs.in_radar_envelope
        or obs.own_speed_frac <= 1.35
        or obs.dist <= max(obs.radar_min_range, 2500.0)
        or obs.own_radar_in_flight >= 2
        # 发射纪律一：机头偏离目标太远（正在偏置/防御）时不开火。
        or obs.angle_off_nose > RADAR_LAUNCH_ANGLE
        # 发射纪律二：只在弹上导引头能很快自主截获的距离内发射。远于
        # active_range 的弹要在载机照射下飞一大段中制导，对手一横转/三九
        # 就打断照射，弹随即转无控——这就是 20–25 km 发射全部落空的原因。
        or (obs.radar_active_range > 0.0 and obs.dist > obs.radar_active_range)
    ):
        return False
    # 已经进 NEZ 当然可以打；没进 NEZ 时按命中概率估计放行，允许打一段
    # 真正的超视距中制导。
    return obs.radar_nez or obs.radar_pk_est >= RADAR_LAUNCH_PK


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
    # ---- 诱导对手先手（LURE）----
    # 对方中距弹还没打光、双方还是迎头，且距离还在 16–24 km 时，不做偏置、
    # 保持迎头进交汇。发射判据是硬阈值（_intercept_estimate 一进 NEZ 就打），
    # 没有欺骗通道能把它提前推远；实测唯一能“诱导”的就是别自己先把几何让出去：
    # - 上界 24 km：再远就还在占位段，没必要提前进入；
    # - 下界 16 km：正好是对方 NEZ 边缘（16.2 km），本段在它开火的同时结束，
    #   立刻交回生存层防御。取 13/18 km 都更差（13 会拖到弹已临空还不敢偏置，
    #   18 则提前把几何让回去，32 局胜场 3:12 与 19:11）。
    # 油门保持 1.0：跑到 n=2.9 倍失速、接近率 524 m/s，收油门在高空压不下
    # 接近率（0.65 只是把回合拖长：131.9 s vs 114.3 s），没有收益。
    LURE_MIN = 16000.0
    LURE_MAX = 24000.0
    LURE_HEAD_ON = math.radians(110.0)
    LURE_THROTTLE = 1.0
    LOW_ENERGY_FRAC = 1.55   # 速度低于 1.55 倍当地失速即进入能量保护
    N_MARGIN = 1.25          # 允许短时超出维持过载的倍数，其余能量透支被禁止
    MISSILE_EVADE_DIST = 9000.0  # 来袭格斗弹进入该距离就开始规避
    # 红外格斗弹不钻多普勒凹口，远距横转/三九对它毫无作用：贴到 2.5 km 内
    # 才投热诱弹并做高 G 急转/桶滚，靠角速度和诱饵把弹甩掉。
    IR_EVADE_DIST = 2500.0
    SARH_EVADE_DIST = 16000.0    # 中距弹进入末段（近距）后的硬甩门限
    # 雷达制导中距弹的告警距离。对方最大发射包线约 36.6 km（Su-57 打
    # F-16V，35 km 标称距离 × RCS 四次方根缩放），取 40 km 保证弹一离架
    # 就进入超视距防御流程。旧值 16 km 时，90° 三九横转要飞十几秒才到位，
    # 弹早就进了末段甚至已经命中，三九/F-pole 在名义上存在、实际不生效。
    BVR_EVADE_DIST = 40000.0
    # 三九凹口的保持门限：|v_r| 低于该值就认为已经进了主瓣杂波凹口
    # （机载雷达 MDV=30 m/s，留约 14 m/s 余量），此时只做小速率修正，避免
    # 大转弯率把机头甩出凹口、又把径向速度顶回门限以上。
    NOTCH_HOLD_VR = 44.0
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
    # 克沃丘尔钟摆的启动距离：对手从尾后压到这个距离内时，用一次
    # “刹停 + 钟摆”诱使对方冲过头（不用于躲导弹——它耗时长、能量代价大）。
    BELL_DIST = 2200.0
    # 一套完整钟摆要近四十秒（拉起 → 尾滑 → 翻落 → 恢复 → 滚转改平），
    # 冷却取同一量级，避免同一场缠斗里连做。
    BELL_COOLDOWN = 30.0
    LOCK_BREAK_DIST = 8000.0     # 被对方锁定且进入该距离后评估是否防御
    OPENING_DIST = 20000.0       # 开局占位阶段：距离大于该值时不直冲对头
    MERGE_START = 5000.0     # 进入该距离开始领先转弯，抢交汇角度
    # 交汇方案在贴身盘旋阶段的保持距离。取 2200 而不是 1800，
    # 是为了让“保持方案”和停用的距离门限错开，避免在门限上每拍翻转转向侧。
    MERGE_HOLD = 2200.0
    MERGE_TIMEOUT = 12.0         # 交汇抢位方案的保持时间 s
    # 中距弹交完后的压近目标距离：中距弹已经交完（或已经进到 16 km 以内）、
    # 还隔着 6 km 以上时就没有继续在超视距对峙的理由了——主动压上去，
    # 进入近距弹/机炮的格斗距离。原来此时会按“被锁防御/僵持改出”处理，
    # 双方在 6–12 km 上兜圈到超时，谁也拿不到近距弹的发射窗口。
    WVR_PRESS_DIST = 6000.0
    # 压近的启动距离：压到对手 NEZ 内沿（16 km）以内后，继续留在超视距
    # 对峙已经没有意义——中距弹的窗口要么已经交完、要么已经被距离关掉。
    # 自己还留着中距弹、但距离已经小于该值时同样压上去，把它留给近距弹
    # 更划算（实测经常出现“留着最后一枚中距弹始终等不到 NEZ”而僵持到超时）。
    BVR_PRESS_DIST = 16000.0
    # 主动减速门限：速度高于角点速度太多时，转弯半径 ∝ v² 会大到抢不到
    # 角度、也会在接近率大的咬尾里直接冲过头。速度超过失速速度的该倍数后
    # 才开始开减速板（现实对应减速板/扰流板/可动鸭翼）。
    BRAKE_SPEED_FRAC = 1.60
    # 角点速度对应的速度比：超过这个倍数才值得用减速板缩小回旋半径。
    BRAKE_CORNER_FRAC = 2.10

    # 防御动作候选池：导弹来袭与被锁定（未射弹）各有一套，具体选哪个
    # 由 _maneuver_prob 按成功率评分决定，不再按固定顺序取第一个可用项。
    # 实弹规避只用水平/下降类动作：殷麦曼、筋斗会把机头指向高空，
    # 在 9 km 外就开始规避时会连续爬升把能量抽干，反而更容易被后续弹命中。
    # 高 G 变体只在速度与过载耐受都允许时才会被 _defense_ok 放行。
    MISSILE_DEFENSE = (
        "BEAM", "HIGH_G_BREAK", "BREAK", "HIGH_G_SPLIT_S",
        "BARREL", "HIGH_G_BARREL", "SPLIT_S", "DIVE_BREAK",
        "DIVE_OFFSET",
    )
    LOCK_DEFENSE = (
        "BEAM", "HIGH_G_BREAK", "BREAK", "SCISSORS",
        "HIGH_G_SPLIT_S", "SPLIT_S", "IMMELMANN", "LOOP",
        "DIVE_BREAK", "DIVE_OFFSET", "HERBST",
    )
    # 中距弹的进攻性防御也纳入“立即抢占”与能量保护。
    DEFENSE_SET = frozenset(
        MISSILE_DEFENSE + LOCK_DEFENSE
        + ("CRANK", "NOTCH_DESC", "F_POLE", "COBRA")
    )
    # 半滚倒转（含高 G 变体）在竖直面里画半个圆，高度净损失 2R；俯冲中
    # 还会加速、半径继续变大。所有“能不能做破 S”的判据统一走 _split_s_room，
    # 不再各自写死一个 2400 m 之类的常数。
    _SPLIT_FAMILY = frozenset(("SPLIT_S", "HIGH_G_SPLIT_S"))
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

    # ---- 航炮攻击几何 ----
    # 航炮门限（900 m / 8°）是机型硬指标，不为了“多打几发”去放宽；能改变
    # 的只有怎么飞到那个炮口解上：尾后压进先用滞后追踪（LAG）保持距离，
    # 机头还偏着时用最大过载压半径（GUN_PRESS），对正后按弹丸飞行时间
    # 打提前量（GUN_LEAD）。
    GUN_ENTRY_DIST = 2600.0                  # 进入尾后压进的距离门限 m
    GUN_EXIT_DIST = 3000.0                   # 已进入后的保持门限（滞环，防边界抖动）
    GUN_TAIL_ANGLE = math.radians(120.0)     # 对方机头背离视线超过该值＝已在尾后
    GUN_TAIL_EXIT = math.radians(105.0)      # 保持尾后几何的退出角（滞环）
    GUN_SPEED_ENTRY = 1.25                   # 进入所需的速度比（相对失速速度）
    GUN_SPEED_EXIT = 1.20                    # 保持门限；低于该值说明能量已不够压炮
    GUN_PRESS_DIST = 1200.0                  # 该距离内改提前量追踪
    # 压进是一次冲刺而不是长期盘旋：8 g 压进每秒要掉 10–15 m/s，超过 3 s
    # 就从“抢半径”变成“掉速被拉开”。到点后转 LAG 保位并重新积累能量。
    GUN_PRESS_TIME = 3.0                    # 单次 GUN_PRESS 最长时间 s
    GUN_PRESS_COOLDOWN = 2.0                # 压进中止后的冷却，防止 LAG↔PRESS 抖
    GUN_OPEN_ABORT = 0.5                    # 接近率连续为负超过该时长即放弃压进
    GUNS_HOLD = 1.0                        # 航炮原语最短驻留（比通用 0.7 s 长）

    # ---- Bingo（燃油硬约束）----
    # 剩余燃油按军推折算的续航秒数低于该值即必须脱离：现实里 Bingo 是
    # “返场 + 备降 + 复飞”的油量，约 5–10 分钟。与超时裁决用同一口径。
    BINGO_ENDURANCE = 300.0

    # ---- 雷达策略（auto）----
    # 目标进入该距离且仍然迎头、自己还有中距弹时，短暂开机建立火控；
    # 中制导段由 own_radar_support 强制保持，末段自主截获后自动关机。
    RADAR_AUTO_ON_DIST = 40000.0
    RADAR_AUTO_BURST = 4.0
    # 红外格斗弹来袭时的规避池：只做急转/桶滚/破 S，不做三九/横转/俯冲。
    # 红外导引头不看多普勒凹口，雷达弹那套规避对它无效。
    IR_DEFENSE = (
        "HIGH_G_BREAK", "BREAK", "HIGH_G_BARREL", "BARREL",
        "HIGH_G_SPLIT_S", "SPLIT_S", "DIVE_BREAK", "DIVE_OFFSET",
    )

    # ---- 中距弹“围追堵截”补射纪律 ----
    # 两发不再同时压出去：首发先飞，把对手逼进转弯/横转的防御航线；等它
    # 已经转出这个角度（或退到空域边缘、逃逸空间被压缩）、首发也进入末段
    # 不再需要中制导时，再从新的方位补一发，堵住它转出去的那条线。转向
    # 角门限取 45°：“被逼转”必须是肉眼可见的大幅规避，25° 的扰动来自
    # 正常瞄准修正，两发几乎同弹道，谈不上夹击。
    RADAR_CUTOFF_TURN = math.radians(45.0)
    # 首发离架后的最小间隔：给对手留出做出反应的时间。实测首发后 5 s 内
    # 对手就开始压坡度，若这时跟着补第二发，两发仍在同一段弹道上。这里
    # 要求间隔 ≥10 s，让首发先把对手的规避姿态逼出来再补。
    RADAR_CUTOFF_MIN_GAP = 10.0
    # 首发进入末段（剩余命中时间小于该值）才补射：首发还在中段时补第二发
    # 只是沿同一条弹道重复；等它压到末段，对手被迫做最后一次规避，第二发
    # 再从新的方位切入，才真正形成“前堵后追”。
    RADAR_CUTOFF_TTG = 18.0
    # 对手离水平边界小于该距离时，视为空间已被压缩，即使没转够角度也补射。
    RADAR_CUTOFF_EDGE = 8000.0

    # ---- 偏置（CRANK / F-pole）的转向侧 ----
    # 偏置方向必须跟着目标的机动方向走：目标向右转时若本机向左偏，目标
    # 相对机头的方位会按“目标转速 + 本机转速”叠加张开，几秒内就掉出 60°
    # 照射锥，载机随即丢火控（也就无法补射）。改为同向偏置后，方位变化
    # 被本机转向抵消，靶标稳定停在 40° 附近。门限带滞回，避免平飞时抖。
    CRANK_TURN_RATE = math.radians(2.0)   # 目标航向变化率门限 rad/s
    # 视线方位角速度死区：低于该值认为目标在平飞，保持上一次的偏置侧，
    # 避免 20 km 外目标轻微摇摆就把 40° 偏置从左侧翻到右侧。
    CRANK_SIDE_DEADBAND = math.radians(0.15)
    # 偏置角附近的“比例项渐变带”：进入该带后比例项按余弦平滑收敛到 0，
    # 与视线角速度前馈无缝衔接（见 _crank）。不能直接置零，否则坡指令跳变。
    CRANK_AZ_BLEND = math.radians(4.0)

    def __init__(
        self,
        aggression: float = 0.5,
        energy_pref: float = 0.5,
        decision_dt: float = 0.1,
        seed: int | None = None,
        lure: bool = False,
        radar_policy: str = "always",
    ) -> None:
        self.aggression = aggression
        self.energy_pref = energy_pref
        self.dt = decision_dt
        # 雷达策略：always 常开 / silent 全程静默 / auto 静默+短时开机。
        # auto 只在需要火控（有中距弹且目标进入 40 km 迎头）时开机，
        # 中制导段（own_radar_support）强制保持，末段自主截获后立即关掉。
        self.radar_policy = radar_policy
        self._radar_on = radar_policy == "always"
        self._radar_burst_t = 0.0
        # 是否执行“诱导对手先手”（见 _tactic_lure）。只有愿意用空间换对方
        # 弹药的一方该打开：它放弃了继续压上，换对方把两发中距弹在自包线
        # 最外沿打掉。
        self.lure = lure
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
        # 航炮压进计时：单次压进时长、压进中止冷却、接近率连续为负的时长。
        self._gun_press_t = 0.0
        self._gun_press_cd = 0.0
        self._gun_open_t = 0.0
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
        # 恢复阶段的“是否还在转向空域中心”：用带死区的状态位而不是
        # |方位误差| >= 5° 的硬门限。门限来回穿越时 bank 在 0° 与 86°、
        # 过载在 1.6 g 与 6 g 之间每 0.1 s 翻一次（低油量高 G 抖动的直接原因）。
        self._rec_turning = False

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
        # 首发射出时目标的航向：补射（围堵）时用它与当前航向做差，判断目标
        # 是否已经被首发逼着转出了角度。
        self._shot_foe_psi = 0.0
        # 首发离架到现在的时间，用作补射的最小间隔。
        self._radar_shot_age = 0.0
        # 目标航向变化率（低通）：只用于破 S 横偏变体的评分；偏置转向侧
        # 由 _crank 直接按视线角速度维护（见 _update_crank_side）。
        self._foe_psi_prev: float | None = None
        self._foe_turn_rate = 0.0
        # 偏置转向侧（+1 = 靶标停在机头左侧 40°，-1 = 右侧）与低通的
        # 视线方位角速度。初值取 +1，表示靶标偏在机头左侧；一旦观察到
        # 目标横向漂移就立刻跟过去。
        self._crank_sign = 1.0
        self._los_rate = 0.0
        # 上一次的视线方位：偏置侧靠它差分出视线角速度；换原语后要清空，
        # 否则中途隔了几秒再算一次差分会被当成“视线突然转了几十度”。
        self._los_az_prev: float | None = None
        # 克沃丘尔钟摆的冷却：这个动作把速度打到接近零、再让机头翻落，
        # 整套要近四十秒，连续做会把能量抽干，所以做完要等一段时间。
        self._bell_cd = 0.0
        # 钟摆机动进入时的机头方位，仅用于诊断（不再做 360° 方位闭环）。
        self._bell_az0: float | None = None
        self._bell_az_prev: float = 0.0
        self._bell_yaw_acc: float = 0.0
        self._bell_az_rate: float = 0.0
        # 分段时间与一次性舵 kick / 末段滚转的计时。
        self._bell_t: float = 0.0
        self._bell_kicked: bool = False
        self._bell_kick_t: float = 0.0
        self._bell_roll_t: float = 0.0

    # ---------- 主入口 ----------

    def act(self, obs: Observation) -> Action:
        self._update_g_tolerance(obs)
        # 先手偏置窗口：中距弹数量增加说明刚有一发离架。
        if obs.own_radar_in_flight > self._prev_radar_in_flight:
            self._post_launch_t = self.POST_LAUNCH_CRANK_TIME
            if self._prev_radar_in_flight == 0:
                # 只记首发的基准航向：第二发是在首发把人逼转之后才补的。
                self._shot_foe_psi = obs.foe_psi
                self._radar_shot_age = 0.0
        self._prev_radar_in_flight = obs.own_radar_in_flight
        self._radar_shot_age += self.dt
        # 目标航向变化率：只用于破 S 横偏变体的评分。偏置侧不再按机头
        # 航向的差分来定——机头朝向和“往哪飞”在 6DOF 下能差出一整个
        # 迎角，且低通阈值会让侧别反复翻转；改为在 _crank 里按视线角速度
        # 直接维护（演示脚本绕过 act() 时也能生效）。
        if obs.contact_quality >= 2:
            if self._foe_psi_prev is not None:
                rate = wrap_pi(obs.foe_psi - self._foe_psi_prev) / self.dt
                self._foe_turn_rate = (
                    0.85 * self._foe_turn_rate + 0.15 * rate
                )
            self._foe_psi_prev = obs.foe_psi
        else:
            self._foe_psi_prev = None
        self._post_launch_t = max(0.0, self._post_launch_t - self.dt)
        self._counter_cd = max(0.0, self._counter_cd - self.dt)
        if self._counter_commit > 0.0:
            self._counter_commit = max(0.0, self._counter_commit - self.dt)
            if self._counter_commit <= 0.0:
                # 承诺期结束：清空方案并进入冷却，避免立刻重新选同一个反击。
                self._counter_plan = None
                self._counter_cd = self.COUNTER_COOLDOWN
        self._yoyo_cd = max(self._yoyo_cd - self.dt, 0.0)
        self._bell_cd = max(0.0, self._bell_cd - self.dt)
        # 航炮计时：按上一拍实际在执行的原语累计，离开 GUN_PRESS 即清零。
        self._gun_press_cd = max(0.0, self._gun_press_cd - self.dt)
        if self.tactic == "GUNS" and self.primitive == "GUN_PRESS":
            self._gun_press_t += self.dt
        else:
            self._gun_press_t = 0.0
        if self.tactic == "GUNS" and obs.closure <= 0.0:
            self._gun_open_t += self.dt
        else:
            self._gun_open_t = 0.0
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
        if obs.incoming_missile_dist < self.BVR_EVADE_DIST:
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
            # 钟摆是“拉起 → 尾滑 → 翻落 → 改出”的完整时序，全程近四十秒；
            # 沿用 14 s 的通用兜底会在恢复段把动作提前切走。
            vertical_limit = 46.0 if self.primitive in BELL_PRIMS else 14.0
            if self._phase_t >= vertical_limit:
                self._maneuver_done = True
            # 已经压住扳机、目标就在炮口窗口里时，不必把这次反转做完：
            # 破 S 是“没射击机会时换角度”的动作，有射击机会就该先打掉。
            gun_now = want in ("PURE", "GUN_PRESS", "GUN_LEAD", "LAG") and obs.in_gun_trigger
            if (
                not self._maneuver_done
                and not gun_now
                and want not in ("RECOVER", "UNLOAD")
            ):
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
        # 航炮三种原语切换要更慢：25° 门限附近 PRESS↔LEAD 每 0.7 s 抖一次，
        # 油门 0.45↔1.0、过载上限同时整档跳，画面就是明显的顿挫。
        if self.tactic == "GUNS" and self.primitive in ("GUN_PRESS", "GUN_LEAD", "LAG"):
            need = max(need, self.GUNS_HOLD)
        if want != self.primitive and (self._hold >= need or bypass):
            self._enter(want, obs)

        self._hold += self.dt
        self._phase_t += self.dt
        self._scissor_t -= self.dt
        act = self._execute(self.primitive, obs)
        self._set_power(act, obs)
        act.radar_on = self._radar_decision(obs)
        return act

    def _radar_decision(self, obs: Observation) -> bool:
        """按策略决定本拍雷达开关（会同步暴露给对手 RWR）。

        always：常开。
        silent：全程关机，只靠 IRST/光电吊舱/目视/预警机；因此拿不到火控，
                不能发射雷达中距弹。
        auto：默认关机，只靠预警机/IRST/侧阵拿到 track；只有当目标已经进
              入自己的中距弹不可逃逸区（或 Pk 足够高）、开机就能打时才短暂
              开机。中制导段（own_radar_support）保持照射，主动弹末段截获
              后立刻关掉，回到静默。
              原先固定“目标进 40 km 就开机”会把双方的开机距离抹平：
              Su-57 的雷达和隐身优势（对 F-16 的 NEZ 约 20 km，而 F-16 对
              它的 NEZ 只有约 14 km）完全体现不出来，还会让隐身机在 40 km
              外先照射、主动暴露。改成按自己的 NEZ 判据后，雷达开机时间
              由机体的探测/武器性能决定，隐身与雷达优势才真正落到战术上。
        """
        if self.radar_policy == "always":
            self._radar_on = True
            return True
        if self.radar_policy == "silent":
            self._radar_on = False
            return False
        need = (
            obs.contact_quality >= 2
            and obs.own_radar_missiles > 0
            # 首发已经在飞时也要允许重新开机：补射（围堵）是先前的弹把对手
            # 逼转之后才打的，不能因为发射架上还有一发就永远不再照射。
            and obs.own_radar_in_flight < 2
            and obs.dist <= self.RADAR_AUTO_ON_DIST
            and obs.aspect > math.radians(90.0)
            # 现在开机就能形成有效射击：已进 NEZ 或 Pk 已足够高。
            and (obs.radar_nez or obs.radar_pk_est >= RADAR_LAUNCH_PK)
        )
        if need:
            self._radar_burst_t = max(self._radar_burst_t, self.RADAR_AUTO_BURST)
        # 中制导段：主动弹还没进入末段自主截获，必须保持照射/数据链。
        if obs.own_radar_support and obs.radar_tracking:
            self._radar_burst_t = max(self._radar_burst_t, self.dt * 2.0)
        if self._radar_burst_t > 0.0:
            self._radar_burst_t -= self.dt
            self._radar_on = True
        else:
            self._radar_on = False
        return self._radar_on

    def _radar_launch_ok(self, obs: Observation) -> bool:
        """中距弹发射纪律：首发按几何打，第二发只在“围堵”时机补。

        不做无脑双发：一进包线就把两发都压出去，两发沿同一条弹道飞，对手
        只要做一个横转/急转就能同时甩掉。改为把第二发留到更好的时机——
        - 首发已经进入末段、不再需要载机中制导（own_radar_support 为假）；
        - 首发已离架至少 RADAR_CUTOFF_MIN_GAP，对手来得及做反应；
        - 对手已经被首发逼着转出 RADAR_CUTOFF_TURN，或者被边界压得没有
          横向逃逸空间；
        这时再从当前几何补一发，弹从另一个方位切入对手转出去的那条航线，
        形成夹击。若对手一直没被逼转、也没被边界压缩，就留着这一发不浪费。
        """
        if not should_launch_radar(obs):
            return False
        n = obs.own_radar_in_flight
        if n == 0:
            return True
        if n >= 2:
            return False
        # 首发还在中制导段：先把它导好，照射中断会直接把它变成无控弹道。
        if obs.own_radar_support:
            return False
        # 首发刚离架：先让对手有时间做反应，避免两发挤在同一条弹道上。
        if self._radar_shot_age < self.RADAR_CUTOFF_MIN_GAP:
            return False
        # 首发还在中段：这时补射只是沿同一条弹道再飞一枚，没有夹击意义。
        # 等到首发压进末段（对手必须做最后一次规避）再补，第二发才从
        # 新的方位切进它的逃逸线。
        if obs.own_radar_ttg > self.RADAR_CUTOFF_TTG:
            return False
        turned = abs(wrap_pi(obs.foe_psi - self._shot_foe_psi)) >= self.RADAR_CUTOFF_TURN
        boxed = obs.foe_edge_margin < self.RADAR_CUTOFF_EDGE
        return turned or boxed

    def _set_power(self, act: Action, obs: Observation) -> None:
        """发动机功率：进入交战前按巡航功率飞，交战与防御时再点加力。

        加力油耗约为军推的三倍，因此只在“战术层要求全油门”或能量吃紧、
        有弹来袭时使用；燃油低于 2% 后强制回军推，避免长时间空耗。
        24 km 以外的进入段还没进入交战，按维持巡航马赫所需的军推飞，
        避免一开局油门就顶到军用最大、推力条直接接近满格。
        """
        # 钟摆的油门是机动时序的一部分（建立上仰后必须收油刹车），
        # 通用功率策略里的“低速/来弹就补加力”会把尾滑直接顶掉。
        if self.primitive in BELL_PRIMS:
            return
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
        if (
            obs.own_speed_frac < 1.50
            and obs.own_gamma > math.radians(40.0)
            and obs.incoming_missile_dist > self.SARH_EVADE_DIST
        ):
            # 大航迹角低速爬升时开加力只会把飞机继续顶在机头上，速度也不会
            # 回来（推力方向几乎与重力同向）。想恢复能量必须先收油门、推杆。
            act.afterburner = 0.0
            act.throttle = min(act.throttle, 0.55)
            return
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
            self._tactic_search,
            self._tactic_press,
            self._tactic_guns,
            self._tactic_lure,
            self._tactic_counter,
            self._tactic_reset,
            self._tactic_defend_locked,
        ):
            plan = tactic(sit, obs)
            if plan is not None:
                return self._floor_guard(plan, obs)
        return self._floor_guard(self._tactic_engage(sit, obs), obs)

    def _floor_guard(
        self, plan: tuple[str, str], obs: Observation,
    ) -> tuple[str, str]:
        """破 S 的离地安全检查：余量不足时换成水平急转。

        反击/改出族一旦锁定动作就按 4–5 s 的承诺期重复返回同一个破 S，
        承诺期里高度可能已经从 5 km 掉到 3 km。这种“承诺”不能覆盖安全：
        低空时统一改成不下降的 BREAK，等承诺期结束再由各战术重新决策。
        """
        if (
            plan[1] in self._SPLIT_FAMILY
            and obs.z_margin_low <= self._split_s_room(obs)
        ):
            return "SURVIVE", "BREAK"
        return plan

    def _turn_radius(self, obs: Observation) -> float:
        """当前速度下竖直面转弯半径 R = v²/(g√(n²−1))。

        过载取与 _split_s 相同的指令值（own_n_avail·0.9，并受耐受与结构
        上限约束）：若按“理论可用过载”估算，半径会偏小两三成，算出来的
        余量正好放行一次实际拉不出来的破 S。
        """
        n = max(min(obs.own_n_avail * 0.9, self.n_pos_tol, G_PEAK), 2.5)
        return obs.own_speed ** 2 / (G * math.sqrt(max(n * n - 1.0, 1.0)))

    def _split_s_room(self, obs: Observation) -> float:
        """做一次破 S 需要的离地余量。

        半滚倒扣后拉半圈，高度净掉 2R（R 见 _turn_radius）；俯冲段还会
        加速 —— 速度涨 15% 半径就涨 32%，所以半径按 1.15·v 算，再乘 1.2
        并加 200 m 安全垫。低空时半径很小、2R 也许不到 2400 m，仍保留
        2400 m 作为机型级的下限。
        """
        n = max(min(obs.own_n_avail * 0.9, self.n_pos_tol, G_PEAK), 2.5)
        r = (1.15 * obs.own_speed) ** 2 / (G * math.sqrt(max(n * n - 1.0, 1.0)))
        return max(2400.0, 2.4 * r + 200.0)

    def _tactic_recover(
        self, sit: _Situation, obs: Observation,
    ) -> tuple[str, str] | None:
        """恢复：贴着边界或高度上下限时先回场。"""
        # 转弯半径随速度平方增长，所以边界余量按速度缩放，避免高速时来不及转回。
        edge_need = max(900.0, 8.0 * obs.own_speed)
        low_need = max(1200.0, 3.5 * obs.own_speed)
        if obs.own_gamma < math.radians(-10.0):
            # 已经在下俯：余量必须盖住“把航迹拉平”要占用的竖直空间，
            # 否则 RECOVER 会在 −70° 俯冲里才启动，剩几百米根本拉不起来。
            low_need = max(
                low_need,
                self._turn_radius(obs)
                * (1.0 - math.cos(obs.own_gamma)) * 1.25 + 300.0,
            )
        high_need = 600.0
        if self.tactic == "RECOVER":
            # 恢复一旦启动就加一段粘滞。低油量脱离（Bingo 的 −5° 下高）正好
            # 落在同一个高度带上：原来掉出 10 m 就切 RECOVER、拉起来 10 m
            # 又切回 EXTEND，两族每 0.5 s 互换一次，指令在 0.2 g 与 8 g 之间
            # 来回跳，看上去就是“低油量时莫名其妙的高 G 抖动”。
            edge_need *= 1.5
            low_need += 600.0
            high_need = 1500.0
        if (
            obs.edge_margin < edge_need
            or obs.z_margin_low < low_need
            or obs.z_margin_high < high_need
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
            return "SURVIVE", (
                "SPLIT_S" if obs.z_margin_low > self._split_s_room(obs) else "BREAK"
            )
        # 规避过程中仍要守住能量：机头已经高高抬起、速度又不够时，
        # 继续压坡度拉过载只会变成爬升耗能，先俯冲把航迹角压下来。
        # 这条不限防御原语：压炮/追逐在低速大航迹角时同样会把飞机顶到
        # 失速附近的垂直爬升里，统一改成卸载推杆。
        if (
            obs.own_gamma > math.radians(45.0)
            and obs.own_speed_frac < 1.55
            and self.primitive not in VERTICAL
            and self.primitive != "COBRA"
        ):
            return "SURVIVE", "UNLOAD"
        if (
            self.primitive in self.DEFENSE_SET
            and obs.own_gamma > math.radians(25.0)
            and obs.own_speed_frac < 1.70
        ):
            # 破 S 会掉 2R 高度（R 随速度平方增长），低空时再做就是直接扎地。
            return "SURVIVE", (
                "SPLIT_S" if obs.z_margin_low > self._split_s_room(obs) else "BREAK"
            )
        # 先手打完立刻转为偏置防御：刚发射的中距弹还需要载机照射/数据链，
        # 这时不再继续直冲压近，而是把目标压到 40° 偏置（CRANK）——一边保持
        # 制导，一边用横向位移拉开交战几何、降低接近率，把自己从对方的
        # 发射包里拉出来。偏置窗口结束后立刻回到进攻，避免远距僵持。
        # 但如果对方的弹会先到（或和我方同时到），偏置就只是把机头
        # 白白钉在对方雷达视线附近，这时必须先转三九生存。
        lost_race = (
            obs.incoming_missile_radar
            and obs.incoming_missile_guided
            and obs.incoming_missile_ttg < obs.own_radar_ttg + 2.0
        )
        if (
            self._post_launch_t > 0.0
            and obs.incoming_missile_dist > self.SARH_EVADE_DIST
            and obs.dist > self.POST_LAUNCH_CRANK_DIST
            # 诱导先手只在“自己还没交弹”时成立：弹已经离架还保持迎头，
            # 等于把机头钉在对方雷达视线上，白送对方第二个发射窗口。
            and (obs.own_radar_in_flight > 0 or not self._lure_hold(obs))
            and not lost_race
        ):
            # 自己的中距弹在飞：这段偏置就是标准的 F-pole——把机头偏出去
            # 40°，既保住弹的照射/数据链，又让对方的雷达和弹看到更大的
            # 距离、更小的接近率。弹打完（命中/脱靶/坠毁）后才退回普通偏置。
            return "SURVIVE", (
                "F_POLE" if obs.own_radar_in_flight > 0 else "CRANK"
            )
        # 来袭格斗弹优先：远距先做能量友好的横转/俯冲拉长 F-pole，
        # 只有进入近距才用最大过载急转硬甩。
        # 雷达制导中距弹（含主动弹）在超视距阶段就要开始偏置/横转；
        # 红外弹没有多普勒凹口可钻，远距横转/三九/俯冲对它完全无效，
        # 贴到 2.5 km 内才投热诱弹并做高 G 急转/桶滚。
        evade_dist = (
            self.BVR_EVADE_DIST
            if obs.incoming_missile_radar
            else self.IR_EVADE_DIST
        )
        if (
            obs.incoming_missile_dist < evade_dist
            and obs.own_speed_frac > 1.25
        ):
            if not obs.incoming_missile_radar:
                # 矢量喷口的过失速反制：来弹已经贴脸、自己还有速度时，
                # 拉眼镜蛇把速度矢量“刹住”，让弹从下方冲过去。
                # 只在 6DOF 且装矢量喷口的机型上可用。
                if (
                    obs.own_tvc
                    and obs.dof >= 6
                    and obs.incoming_missile_dist < self.COBRA_DIST
                    and obs.own_speed_frac > 1.90
                    and obs.z_margin_low > 1200.0
                ):
                    return "SURVIVE", "COBRA"
                # 钟摆不放进导弹规避：整套动作要近四十秒、速度归零，躲不来弹，
                # 它的价值是在近距格斗里诱使对手冲过头（见 _tactic_counter）。
                return "SURVIVE", self._pick_defense(
                    obs, vs_missile=True, ir=True
                )
            # 雷达制导中距弹按 FC3 手册的超视距防御流程处理：
            # 自己的弹先到就 F-pole 偏置，否则三九下高。
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
                    # 远距规避只用横转：破 S 是掉高度的动作，离地 1.5 km
                    # 时再倒扣拉半圈等于自己撞地（原来这里的判断还写反了，
                    # 低空反而去选 SPLIT_S）。
                    return "SURVIVE", "BEAM"
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
        # Bingo：剩余军推续航低于返场+备降油量时必须脱离，不再回缠。
        # 这是硬约束，且排在“弹药打光”之前——没油比没弹更致命。脱离期间
        # 由恢复族兜底，不会为了拉开距离而飞出交战空域。
        if obs.own_mil_endurance < self.BINGO_ENDURANCE:
            return "DISENGAGE", "EXTEND"
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

    def _tactic_press(
        self, sit: _Situation, obs: Observation,
    ) -> tuple[str, str] | None:
        """中距弹交完后的压近：主动进入近距弹的格斗距离。

        自己的中距弹已经全部交完（无剩余、无在飞）、也没有来袭弹时，继续
        在 6 km 外保持横转/偏置只是把对局拖到超时。此时满油门压上去，用
        机头指向把目标带进近距弹的锁定包线（红外导引头仍要求目标落在视场
        内），真正贴脸的威胁再交回生存层处理。
        """
        # 还有弹在飞（雷达弹按 40 km、红外弹按 9 km 告警）时先让生存层优先。
        if (
            obs.incoming_missile_radar
            or obs.incoming_missile_dist < self.MISSILE_EVADE_DIST
        ):
            return None
        # 只按“自己这边”判定，不再要求对方也打空：对方留着最后一枚中距弹
        # 却因 NEZ 门限始终等不到发射窗口时，旧条件会让压近永远不触发，
        # 双方在 6–12 km 上兜圈到超时。对方真发射了，来袭告警会接管。
        if obs.own_radar_in_flight > 0:
            return None
        if obs.own_radar_missiles > 0 and obs.dist > self.BVR_PRESS_DIST:
            return None
        if obs.dist <= self.WVR_PRESS_DIST:
            return None
        return "PRESS", "PURE"

    def _tactic_search(
        self, sit: _Situation, obs: Observation,
    ) -> tuple[str, str] | None:
        """搜索：没有任何稳定跟踪时，朝最后已知位置/空域中心飞。

        这一档排在生存之后、进攻之前：看不见目标就不该做进攻判断。
        observe 已经把 off_az 换成搜索方位，PURE 会自然把机头压过去；
        预警机/IRST/光电吊舱一旦重新建立 track，就交回交战族。
        """
        if obs.contact_quality >= 2:
            return None
        return "SEARCH", "PURE"

    def _tactic_guns(
        self, sit: _Situation, obs: Observation,
    ) -> tuple[str, str] | None:
        """航炮攻击几何：在尾后窗口里把机头压进 900 m / 8° 的炮口解。

        航炮是唯一不需要先锁定再发射的通道，但门限是硬的：既要在 900 m
        内，机头偏差还得小于 8°。所以这一段只在“已经咬住尾后”（对方机头
        明显背离视线）时才进入——正面或侧方压炮口只会变成一次无收益的
        冲撞。压进分三种：

        - 机头还偏着（>25°）：GUN_PRESS，收油门 + 拉满瞬时过载，用更小的
          转弯半径去破解“同速同过载共同盘旋→相对角度恒定 90°”的死锁；
        - 机头基本对正（<25°）且已进 1.2 km：GUN_LEAD，按弹丸飞行时间给
          提前量，纯追踪只会把弹打在自己后面；
        - 其余情况：LAG，滞后追踪保持尾后位置，别一头冲过头。

        门限都带滞环：进入用 2600 m / 120° / 速度比 1.25，已经在 GUNS 里
        则放宽到 3000 m / 105° / 1.20，避免在边界上整族进出。压进本身是
        限时冲刺——接近率转负、连续拉开 0.5 s、或已压满 3 s 就转 LAG 保位，
        不再用 8 g 一直烧到速度见底。
        """
        in_guns = self.tactic == "GUNS"
        dist_lim = self.GUN_EXIT_DIST if in_guns else self.GUN_ENTRY_DIST
        tail_lim = self.GUN_TAIL_EXIT if in_guns else self.GUN_TAIL_ANGLE
        speed_lim = self.GUN_SPEED_EXIT if in_guns else self.GUN_SPEED_ENTRY
        if obs.own_ammo <= 0.0 or obs.dist > dist_lim:
            return None
        if obs.foe_off_nose < tail_lim:
            return None
        if obs.own_speed_frac < speed_lim or obs.z_margin_low < 1500.0:
            return None
        # 进入压进用 25° 门限，已经压进去后放宽到 20° 才退出：否则机头
        # 偏差在门限上每拍来回跨一次，PRESS↔LEAD 连同油门 0.45/1.0 一起抖。
        press_angle = (
            math.radians(20.0) if self.primitive == "GUN_PRESS"
            else math.radians(25.0)
        )
        if obs.angle_off_nose > press_angle:
            # 尾后但机头还偏着：两架同速同过载一起盘旋时相对角度恒定在
            # 90°，靠同样的转弯拿不回来，必须把半径压得比对手小。
            abort = (
                self._gun_press_t >= self.GUN_PRESS_TIME
                or self._gun_open_t >= self.GUN_OPEN_ABORT
            )
            if abort:
                if self.primitive == "GUN_PRESS":
                    self._gun_press_cd = self.GUN_PRESS_COOLDOWN
                return "GUNS", "LAG"
            if self._gun_press_cd > 0.0 or obs.closure <= 0.0:
                # 接近率为负说明目标在拉开：压半径拿不到角度，只会把速度
                # 烧光。转 LAG 开满油门保位，等几何重新收敛再压第二次。
                if self.primitive == "GUN_PRESS":
                    self._gun_press_cd = self.GUN_PRESS_COOLDOWN
                return "GUNS", "LAG"
            return "GUNS", "GUN_PRESS"
        if obs.dist < self.GUN_PRESS_DIST or obs.in_gun_cone:
            # 机头已经基本对正：这时再纯追踪只会把弹丸打在自己后面（弹丸
            # 飞 0.7–1 s，目标在这段时间里会飞 150–250 m），必须按弹丸
            # 飞行时间给提前量。
            return "GUNS", "GUN_LEAD"
        return "GUNS", "LAG"

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
        # 钟摆的战术用法：对手还在尾后压近（0.9–2.2 km）时，用一次
        # “近垂直拉起 → 零速尾滑 → 钟摆翻落”把速度钉住，逼他从上方/侧方
        # 冲过头，再借能量恢复反咬。它不是导弹规避动作。
        if (
            tail
            and 900.0 <= obs.dist < self.BELL_DIST
            and obs.own_n_avail >= HIGH_G_MIN_AVAIL
            and self._bell_cd <= 0.0
            and self._defense_ok("KVOCHUR_BELL", obs)
        ):
            self._bell_cd = self.BELL_COOLDOWN
            return self._start_counter("COUNTER_BELL", "KVOCHUR_BELL")
        # 迎头化：破 S 的 180° 反转需要离地余量，且不能太慢。
        # energy_diff 的单位是 J/kg（v²/2 + g·z）：-15000 约等于 300 m/s 下
        # 20 m/s 的速度亏欠，再差就说明反转做完会被对手直接压住。
        if (
            obs.dist > 1200.0
            and obs.own_speed_frac > 1.55
            and obs.energy_diff > -15000.0
            and obs.z_margin_low > self._split_s_room(obs)
        ):
            # Herbst 比破 S 更快把机头转回来：只拉 55° 就开始绕速度矢量
            # 横滚，不必把整个垂直面画完。6DOF 且高度余量足够时优先用它。
            if (
                obs.dof >= 6
                and obs.own_speed_frac > 1.60
                and obs.z_margin_high > 2000.0
            ):
                return self._start_counter("COUNTER_HEADON", "HERBST")
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
        if obs.incoming_missile_dist < self.BVR_EVADE_DIST:
            return None
        if sit.name == "DEFENSIVE":
            plan = (
                ("RESET", "SCISSORS") if obs.dist < 1200.0 else ("RESET", "SPLIT_S")
            )
        elif obs.own_speed_frac >= 1.45 and obs.energy_diff >= -6000.0:
            plan = ("RESET", "EXTEND")
        elif obs.z_margin_low > self._split_s_room(obs):
            # 破 S 不需要速度富余：半滚倒扣本来就是掉高度换机头，只要
            # 留够离地余量（2R）就能把水平盘旋改成一次垂直掉头。
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

    def _tactic_lure(
        self, sit: _Situation, obs: Observation,
    ) -> tuple[str, str] | None:
        """诱导先手：在对方 NEZ 边缘保持迎头，让对方把中距弹先交出来。

        排在生存之后：真有弹来袭时先活下来；也排在反击/被锁防御之前，
        因为这一段的行为本身就是为了不再给对方“我在压上”的几何。
        """
        if not self._lure_hold(obs):
            return None
        return "SEPARATED", "LURE"

    def _lure_hold(self, obs: Observation) -> bool:
        """当前是否处在“诱导先手”那一段（对方还有中距弹、迎头、13–22 km）。"""
        return (
            self.lure
            and obs.foe_radar_missiles > 0
            and obs.aspect > self.LURE_HEAD_ON
            and self.LURE_MIN <= obs.dist <= self.LURE_MAX
        )

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
            p = 0.62 + 0.08 * clamp(ttg / 10.0, 0.0, 1.0)
            if not vs_missile:
                # 只有雷达锁定、对方弹还没出手时，横转的收益主要是压制对方的
                # 多普勒跟踪，不必像真被弹追时那样牢牢占住第一名；让出一部分
                # 分差，垂直面掉头（殷麦曼）这类能立刻改变交会几何的动作才有
                # 机会被选中。
                p -= 0.12
            return p
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
            # 殷麦曼是“垂直面掉头”：拉起 → 过顶 → 改出后机头反向、净升高。
            # 被锁定但对方弹还没出手时，它能立刻把迎头/尾追关系翻过来，比
            # 横转或平转更快重置几何；代价是几秒内速度掉得厉害，只在高度
            # 余量、能量都富余（本就是它的准入条件）时才值得，并随目标越近
            # 越优先——远距时垂直面掉头只会把自己挂在天上等对方补射。
            close = clamp((6000.0 - obs.dist) / 4000.0, 0.0, 1.0)
            return 0.42 + 0.18 * high + 0.10 * energy + 0.16 * close
        if name == "LOOP":
            return 0.32 + 0.16 * high
        if name == "DIVE_OFFSET":
            # 对手正在横向转弯时，边俯冲边横偏比纯垂直破 S 更早咬回内侧：
            # 这正是它存在的理由，此时评分要明显压过标准破 S / 急转。
            p = 0.44 + 0.10 * low
            if abs(self._foe_turn_rate) >= self.CRANK_TURN_RATE:
                p += 0.16
            return p
        if name == "DIVE_BREAK":
            # 破 S + 机头偏转：比纯破 S 多甩开一段航向，对手转弯时更值。
            p = 0.46 + 0.12 * low
            if abs(self._foe_turn_rate) >= self.CRANK_TURN_RATE:
                p += 0.08
            return p
        if name == "KVOCHUR_BELL":
            return 0.36 + 0.12 * near + 0.12 * high
        if name == "HERBST":
            return 0.38 + 0.14 * high + 0.06 * near
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

    def _pick_defense(
        self, obs: Observation, vs_missile: bool, ir: bool = False,
    ) -> str:
        """在防御候选池里按成功率选择规避动作，全程保持同一转向侧。

        远距优先横转消耗弹能量、中距保持横转并做高 G 修正、近距才切硬甩，
        这些战术倾向由 `_maneuver_prob` 的概率项表达，而不是固定的轮换
        顺序；末段刚建立起来的三九/横转姿态也不会被无收益地换掉。
        红外弹（ir=True）走单独的 IR_DEFENSE 池：导引头不看多普勒凹口，
        BEAM/NOTCH_DESC/F_POLE 对它无效，只能靠大过载急转或桶滚甩掉。
        """
        if obs.own_speed_frac < self.LOW_ENERGY_FRAC:
            return "SPLIT_S" if obs.z_margin_low > self._split_s_room(obs) else "BREAK"
        if ir:
            pool = self.IR_DEFENSE
        else:
            pool = self.MISSILE_DEFENSE if vs_missile else self.LOCK_DEFENSE
        cands = [c for c in pool if self._defense_ok(c, obs)]
        if not cands:
            return "BREAK" if (vs_missile or ir) else "BEAM"
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
        if primitive in self._SPLIT_FAMILY and (
            obs.z_margin_low <= self._split_s_room(obs)
        ):
            return False
        # 钟摆和 Herbst 都靠“绕速度矢量横滚 + 大迎角”改变指向：没有矢量
        # 喷口（或不在 6DOF 内核）时做不出这个姿态，退化成一次普通爬升。
        if primitive in BELL_PRIMS and not (obs.own_tvc and obs.dof >= 6):
            return False
        if primitive == "HERBST" and obs.dof < 6:
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
        """雷达制导中距弹的应对：先保住自己的弹（F-pole），否则三九下高。

        判据直接落在多普勒盲区模型上：PD 雷达/导引头只靠目标自身的径向速度
        把回波从地杂波里挑出来，目标横穿到 |v_r| < MDV 时回波整体落进主瓣
        杂波凹口，载机雷达的照射/数据链与弹上导引头会同时丢目标，弹转为
        无控弹道（sim.World._update_missile 里的 illum / seeker_blind）。
        - F-pole：自己的中距弹还在飞、且比来袭弹先命中时，保持 40° 偏置
          继续给自己的弹装订。对方的雷达弹到命中前都依赖载机照射，先把
          对方载机打掉，对方的弹就失去中制导。
        - 三九下高：其余情况（对方弹先到、自己没弹或弹已自主）把对方放到
          正侧方并下高，用横穿把径向速度压进凹口。
        返回 None 表示高度/能量不允许做三九，交回硬甩规避轮换。
        """
        ttg = obs.incoming_missile_ttg
        # “谁的弹先到”是 F-pole 与三九的分水岭：弹先到就偏置保弹，
        # 弹后到再偏置就是把自己钉在对方雷达视线里挨打。
        # 判据只看“我还有中距弹在飞、且它比来袭弹先到”。旧实现额外要求
        # own_radar_support（弹还没进入主动截获段），但本模拟里双方常在
        # 20 km 上下开火，弹一离架就已进入 18–20 km 的主动截获距离，support
        # 恒为假，F-pole 因此一次都没触发过。主动截获后载机虽然可以不管弹，
        # 但保持 40° 偏置仍能继续拉大与来袭弹的几何、压缩对方弹的能量，
        # 所以只要自己的弹会先落地，就该做 F-pole。
        if (
            obs.own_radar_in_flight > 0
            and obs.own_radar_ttg <= ttg + 1.0
            # 偏置只能把接近率降下来，来弹已经只剩几秒时再偏置就是站着挨打，
            # 这种末段时间窗交给三九/硬甩。
            and ttg > 5.0
            # 弹仍在制导段时需要雷达能照到目标，否则偏置没有意义。
            and (not obs.own_radar_support or obs.radar_tracking)
        ):
            return "F_POLE"
        # 对方弹已经丢了制导、只剩惯性弹道：如果已经进了凹口就守到它飞过
        # （这时改出反而会把机头从凹口甩出来）；还没进凹口就没必要为无控弹
        # 横转 90°。自己的中距弹也打完了才走这一步，否则先按上面的 F-pole
        # 把弹导完。
        if not obs.incoming_missile_guided:
            return "NOTCH_DESC" if self.primitive == "NOTCH_DESC" else None
        # 三九下高要掉高度：按当前-6°的下高角，20 s 的凹口保持只消耗
        # 约 600 m，所以离地 1.2 km、高度 2 km 以上就允许。旧门限 1.5 km
        # 把已经下到 2.8 km 的 F-16V 挡在门外，它只能做硬甩，结果被弹命中。
        if (
            obs.z_margin_low > 1200.0
            and obs.own_alt > 2000.0
            and obs.own_speed_frac > 1.30
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
        # 视线角速度的差分基准也要清空：新原语的第一拍只记录方位、不差分。
        self._los_az_prev = None
        if primitive == "POSITION":
            self._open_alt0 = None
        if primitive in BELL_PRIMS:
            self._bell_az0 = None
            self._bell_yaw_acc = 0.0
            self._bell_az_rate = 0.0
            self._bell_t = 0.0
            self._bell_kicked = False
            self._bell_kick_t = 0.0
            self._bell_roll_t = 0.0
        # 防御转向只在本次威胁的第一次防御动作时定一次侧：取威胁所在的一侧。
        # 之后由 act() 的计时器统一维护，只在威胁持续位于反侧 1.5 s 才换向。
        # 原来“每次进入防御都按瞬时方位重算”，在盘旋缠斗里方位角常年超过
        # 60°，会导致每进一次防御就反向一次，且很快被下一拍策略替换。
        if primitive in (
            "BREAK", "BEAM", "SPLIT_S", "BARREL", "IMMELMANN", "LOOP",
            "CRANK", "NOTCH_DESC", "F_POLE",
            "HIGH_G_BREAK", "HIGH_G_SPLIT_S", "HIGH_G_BARREL",
            "DIVE_BREAK", "DIVE_OFFSET", "KVOCHUR_BELL", "KVOCHUR_BELL_BACK",
            "HERBST",
        ) and not self._def_engaged:
            self._def_engaged = True
            if obs.incoming_missile_dist < self.BVR_EVADE_DIST:
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
        # 钟摆要自己按分段控制减速板（顶点前切开、尾滑段全开），
        # 不能被通用的 _brake_demand 覆盖。
        if primitive not in BELL_PRIMS:
            act.brake = self._brake_demand(obs, primitive)
        pilot_ok = self._gloc_t <= 0.0
        act.launch = pilot_ok and should_launch(obs)
        act.launch_radar = pilot_ok and self._radar_launch_ok(obs)
        return act

    def _brake_demand(self, obs: Observation, primitive: str) -> float:
        """主动减速开度 0..1（减速板/扰流板/可动鸭翼的等效阻力）。

        只在“当前速度明显高于这个几何需要的角点速度”时才打开：
        - 咬尾接近率过大，再不改就冲过头；
        - 近距盘旋/压炮：转弯半径 ∝ v²，减速等于直接缩小回旋半径，
          用几何换角度，比继续拉满过载更有效；
        - 对头交汇擦身而过：迎头速度叠加，不减速会直接飞过对方。
        防御（有来袭弹）与压近阶段不开：那两处都要拿速度换生存/占位。
        """
        if primitive == "PRESS":
            return 0.0
        if (
            obs.incoming_missile_radar
            or obs.incoming_missile_dist < self.SARH_EVADE_DIST
        ):
            return 0.0
        if obs.own_speed_frac < self.BRAKE_SPEED_FRAC:
            return 0.0
        # 1) 接近率过大、马上就要冲过头：用“剩余交汇时间”而不是单看
        # 接近率。3 km 上 90 m/s 的接近率还有 30 多秒，完全不必踩减速板；
        # 只有 8 s 以内会撞上目标时才值得减速。
        if obs.closure > 60.0:
            tgo = obs.dist / obs.closure
            if tgo < 8.0:
                return clamp((8.0 - tgo) / 5.0, 0.3, 1.0)
        # 2) 近距抢角度：盘旋类原语把速度压回角点速度附近。
        # 门限取 2.1 倍失速（约等于这些机型的角点速度）：原来只要超过 1.6
        # 倍就开减速板，两架飞机在近距盘旋里会全程踩着减速板，一起掉到
        # 1.5 倍失速附近，缠斗变成双方都飞不动的慢速绕圈。
        if obs.dist < 3000.0 and obs.own_speed_frac > self.BRAKE_CORNER_FRAC and (
            primitive in TURNING
            or primitive in ("GUN_PRESS", "GUN_LEAD", "SCISSORS",
                             "HIGH_YOYO", "LOW_YOYO")
        ):
            return clamp(
                (obs.own_speed_frac - self.BRAKE_CORNER_FRAC) / 0.6, 0.0, 0.9,
            )
        # 3) 迎头交汇后的近距离高速通过
        if obs.dist < 1000.0 and obs.closure > 180.0:
            return 0.7
        return 0.0

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
            "LURE": self._lure,
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
            "GUN_LEAD": self._gun_lead_action,
            "GUN_PRESS": self._gun_press,
            "BREAK": self._break,
            "SPLIT_S": self._split_s,
            "DIVE_BREAK": self._dive_break,
            "DIVE_OFFSET": self._dive_offset,
            "KVOCHUR_BELL": self._kvochur_bell,
            "KVOCHUR_BELL_BACK": self._kvochur_bell_back,
            "HERBST": self._herbst,
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
        if primitive == "PURE" and self.tactic == "PRESS":
            return self._press_turn(obs)
        fn = self._maneuvers.get(primitive)
        if fn is None:
            return aim_action(obs, obs.off_az, obs.off_el)
        return fn(obs)

    def _press_turn(self, obs: Observation) -> Action:
        """压近转向：用接近瞬时上限的过载把机头拉回目标。

        压近阶段双方常在 7–18 km 上各自画 4 km 半径的大回环（n≈2.8 g、
        约 4°/s），一次 180° 掉头要 45 s，看起来就是“越追越远”。这里把
        过载放开到可维持过载的 1.6 倍（仍受 _limit_g 的人体耐受与
        _limit_energy 的低速保护约束），掉头时间压到 20 s 以内，双方才能
        真正建立起接近率。
        """
        act = aim_action(obs, obs.off_az, obs.off_el, gain=3.0, throttle=1.0)
        cap = min(obs.own_n_avail, max(obs.own_n_sustain * 1.6, 4.0))
        if act.n > cap:
            act.n = cap
        return act

    def _gun_lead(self, obs: Observation) -> tuple[float, float]:
        """航炮提前量：按弹丸飞行时间指向目标的未来位置，返回相对炮线的偏差。

        弹丸出膛后以炮口初速 + 载机速度飞行，900 m 约需 0.7 s；目标在这
        段时间里飞 150–250 m（在 900 m 上就是 10–15°）。纯追踪永远把弹丸
        打在自己后面，所以这里按真实飞行时间 T = 距离 / 弹丸地速 给提前量，
        而不是用 lead_point 里那个按接近率估算、并夹在 1.5 s 的 t_go。
        目标未来位置同样按 ψ̇/γ̇ 积分：0.7 s 里做 6 g 回旋的目标能偏出
        一个转弯半径的量级，沿切线预测会把弹丸打到圆弧外侧。
        """
        v_bullet = max(obs.own_speed + 600.0, 800.0)
        t_go = clamp(obs.dist / v_bullet, 0.0, 1.2)
        ce = math.cos(obs.los_el)
        ux = ce * math.cos(obs.los_az)
        uy = ce * math.sin(obs.los_az)
        uz = math.sin(obs.los_el)
        lx, ly, lz = predict_path(
            (ux * obs.dist, uy * obs.dist, uz * obs.dist),
            obs.foe_speed, obs.foe_track_psi, obs.foe_track_gamma,
            obs.foe_psi_rate, obs.foe_gamma_rate, t_go,
        )
        az = math.atan2(ly, lx)
        el = math.atan2(lz, math.hypot(lx, ly))
        return wrap_pi(az - obs.own_nose_az), el - obs.own_nose_el

    def _gun_lead_action(self, obs: Observation) -> Action:
        """按提前量把机头压到炮口解上（近距、机头已基本对正时使用）。

        这里不用 pursuit_action：它把俯仰误差夹在 ±4° 以抑制高速摆振，
        而炮口的垂直提前量往往有 10° 以上，被夹掉之后弹丸会系统性打低。
        """
        az, el = self._gun_lead(obs)
        act = aim_action(obs, az, el, gain=1.6, throttle=0.85)
        cap = max(min(obs.own_n_avail, max(obs.own_n_sustain * 1.25, 2.0)), 1.0)
        if act.n > cap:
            act.n = cap
        act.fire = obs.in_gun_trigger
        return act

    def _gun_press(self, obs: Observation) -> Action:
        """航炮压进：收油门到角点速度附近，用最大瞬时过载把机头压向目标。

        两架同速、同过载一起盘旋时，相对角度恒定在 90°，谁也别想拿到
        炮口解。破解办法只有一条：让转弯半径比对手小——半径 ∝ v²，
        减速到角点速度附近再拉满瞬时过载，用真实的转弯性能差把角度抢
        回来。这不是放宽炮口门限，而是改变进入炮口解的几何。
        """
        act = pursuit_action(obs, *self._gun_lead(obs), gain=1.6, throttle=0.45)
        if act.n > 0.0:
            # pursuit_action 已经给出推杆（负过载）时不要再按最大过载覆盖，
            # 否则“机头高抬 + 低速”的压炮会继续把航迹往上拽。
            act.n = min(obs.own_n_avail, self.n_pos_tol)
        act.fire = obs.in_gun_trigger
        return act

    def _recover(self, obs: Observation) -> Action:
        """边界/高度安全优先：先转向空域中心，再修正高度。

        低空大俯角时必须先把机头拉起来：只给一个 +12° 的目标航迹角，
        在 −70° 俯冲里等于没救（飞机在 1 s 内就掉 250 m），会直接撞地。
        其余情况只是“余量不够”而不是“马上要撞地”，用中等过载的稳定
        盘旋或浅爬升就够。原来统一走 aim_action（gain=2.5）会把 4° 的
        航迹角误差放大成 10°/s，在 360 m/s 上就是 7 g 的竖直分量，
        恢复动作因此变成 8–9 g 的猛拉——低油量、脱离与恢复互相打断时
        尤为明显，也就是 HUD 上看到的“莫名其妙高 G 抖动”。
        """
        speed = max(obs.own_speed, 60.0)
        pull_out = obs.own_gamma < math.radians(-15.0)
        urgent = obs.z_margin_low < max(1200.0, 3.5 * speed)
        if urgent and pull_out:
            # 低空大俯角：先翼平、按可用过载全力拉起。此时再把升力分给
            # “飞向空域中心”的坡度，竖直分量只剩 n·cosφ，就是压不出
            # 足够的 γ̇，人在 1.5 km 边界上看着飞机以 −70° 扎下去。
            act = aim_action(obs, 0.0, math.radians(25.0) - obs.own_gamma, gain=2.5)
            act.bank = 0.0
            act.n = min(obs.own_n_avail, self.n_pos_tol)
            return act
        if obs.z_margin_high < 600.0:
            return level_turn(obs, 0.0, 2.0, gamma_target=math.radians(-8.0))
        heading = wrap_pi(obs.center_az - obs.own_psi)
        # 是否还在转向空域中心，用 12°/4° 的回差状态位决定：方位误差在
        # 5° 上下每拍穿越一次硬门限时，转向侧、坡度（0°↔86°）和过载
        # （1.6 g↔6 g）都会跟着 0.1 s 翻一次，看上去就是高 G 抖动。
        if abs(heading) > math.radians(12.0):
            self._rec_turning = True
        elif abs(heading) < math.radians(4.0):
            self._rec_turning = False
        sign = 0.0
        if self._rec_turning:
            sign = 1.0 if heading > 0.0 else -1.0
        if obs.edge_margin < max(900.0, 8.0 * speed):
            # 离边界太近：转回空域中心。用 6 g 的协调盘旋——6 g 正好是
            # 持续耐受上限，长时间拉也不会累积疲劳；高度余量同时不足时
            # 顺带 5° 浅爬升。
            climb = math.radians(5.0) if urgent else 0.0
            n = min(obs.own_n_avail, self.n_pos_tol, G_SUSTAIN)
            return level_turn(obs, sign, n, gamma_target=climb)
        # 只是离地余量不足：3 g 左右的浅爬升盘旋就够，不需要压满坡度。
        return level_turn(obs, sign, 3.2, gamma_target=math.radians(2.0))

    def _lure(self, obs: Observation) -> Action:
        """迎头诱导：机头对准目标、不做偏置，把交汇几何留给自己。

        这个原语本身只是“满油门指向目标”，价值全在它不是什么：不是 CRANK、
        不是 POSITION。对方 NEZ 里的航向惩罚项（0.45）让“目标逃逸”成为
        缩小它包线的唯一开关，所以提前偏置等于替对方把发射距离从 16 km
        压到 10 km；保持迎头则让它只能在自己 NEZ 边缘、pk 最低、飞行时间
        最长的那一点开火，而交汇仍是迎头，SU-57 的 Ps 与瞬时过载优势才有
        兑现的入口。
        """
        return pursuit_action(
            obs, obs.off_az, obs.off_el, gain=1.2, throttle=self.LURE_THROTTLE,
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

    def _split_s_impl(self, obs: Observation, n_max: float,
                      deflect: float = 0.0) -> Action:
        """破 S 标准三段：半滚倒扣 → 副翼中立拉杆过垂直 → 拉平改出。

        旧实现只把坡度命令到 80°（Action.bank=±1 的满刻度），从来没有真正
        倒扣：之后所谓的“拉杆”只是一个大坡度平飞盘旋，看上去就是平飞转向。
        现在第一段直接命令到 180° 倒扣，第二段用 roll_free 让机体在垂直面内
        连续下俯（否则内回路会把滚转角拽回地平线、根本过不了垂直）。
        用 own_gamma（asin 表示，±90°）判段：越过 −70° 后再回到 0 附近，
        说明半圈已经拉完，机头回到水平、航向反转。

        deflect（Action.bank 单位，可选）：改出段先滚到这个坡度，再在坡上
        拉杆把航迹从俯冲里拉平——用“横滚 + 拉杆”的侧向升力把机头带偏，
        改出航向就在 180° 反向之外再偏出几十度（DIVE_BREAK 用它）。偏转
        不能靠方向舵直接偏航：倒扣俯冲里蹬舵会诱发螺旋下降；也不能在机头
        接近垂直时压坡度，那时欧拉滚转角是奇异的，滚转指令会把破 S 拧成
        一圈放不下来的下降螺旋。
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
            # 偏转型要早一点进入改出段（45°），给“压坡度 + 拉杆”留出时间；
            # 标准破 S 到 18° 才切，避免拉平过冲。
            if self._dived and abs(obs.own_gamma) < (
                math.radians(45.0) if deflect > 0.0 else math.radians(18.0)
            ):
                self._phase, self._phase_t = 2, 0.0
            return Action(bank=0.0, n=n_max, throttle=1.0, fire=False,
                          roll_free=True)
        if deflect > 0.0:
            # 偏转型改出：航迹还压着就压坡度拉杆，机头在拉平的同时被带向
            # 一侧；拉平后再滚回水平收尾。
            if abs(obs.own_gamma) >= math.radians(10.0):
                return Action(
                    bank=sign * deflect,
                    n=min(obs.own_n_avail, self.n_pos_tol, G_SUSTAIN),
                    throttle=1.0, fire=False,
                )
            if abs(obs.own_bank) < math.radians(30.0):
                self._maneuver_done = True
            return level_out_action(obs)
        # 改出：坡度归零、航迹拉平，确认回到平飞后结束整个机动。
        if abs(obs.own_gamma) < math.radians(8.0) and abs(obs.own_bank) < math.radians(30.0):
            self._maneuver_done = True
        return level_out_action(obs)

    def _dive_break(self, obs: Observation) -> Action:
        """破 S + 机头偏转（dive break）：与标准破 S 共用同一套三段实现。

        区别只在改出段：拉平航迹时先压一个坡度，靠“横滚 + 拉杆”的侧向
        升力把机头带偏，改出航向相对进入方向不只是 180° 反向，还额外偏出
        几十度，脱离线斜着让开对手的追击方向。不再给方向舵偏航指令——
        倒扣俯冲里蹬舵会诱发螺旋下降，拉不回水平。
        """
        n_max = min(obs.own_n_avail * 0.9, self.n_pos_tol, G_PEAK)
        return self._split_s_impl(obs, n_max, deflect=DIVE_BREAK_BANK)

    def _dive_offset(self, obs: Observation) -> Action:
        """破 S 横偏变体：倒下高时保留 40° 偏转，边俯冲边朝内侧横偏。

        标准破 S 是纯垂直面的半滚倒转，航向只反转、不产生侧向位移；对手
        正在横向转弯时，尾部机动照样能甩开，但机头最后指的是一条和对手
        转向相反、需要重新修正的线。这里在俯冲段故意不倒扣到底：滚到
        140°（倒扣少 40°）就拉杆，升力矢量同时有向下和侧向分量，机体在
        掉高度的同时朝对手转弯的内侧偏，出俯冲时航向已经提前压向对手，
        省掉改出后那一次大转弯。最后仍改平回原航向，属于“绕一下再回来”。
        """
        sign = self._def_sign
        n_max = min(obs.own_n_avail * 0.9, self.n_pos_tol, G_PEAK)
        bank_cmd = sign * (BANK_INVERT - DIVE_OFFSET_BANK)
        if self._phase == 0:
            # 直接滚到 140°：旧实现先滚到 180° 再回滚 40°，那一次反向滚
            # （约 1.5 s、坡度从 180° 掉回 140°）正是“拖动发抖”的来源。
            if abs(obs.own_bank) >= math.radians(120.0) or self._phase_t >= 3.5:
                self._phase, self._phase_t = 1, 0.0
                self._dived = False
            return Action(bank=bank_cmd, n=1.0, throttle=1.0, fire=False)
        if self._phase == 1:
            # 俯冲段保持 140° 坡度：cos<0 分量负责下高，sin 分量负责横偏。
            if obs.own_gamma <= math.radians(-45.0):
                self._dived = True
            if (self._dived and self._phase_t >= 2.0) or self._phase_t >= 6.0:
                self._phase, self._phase_t = 2, 0.0
            return Action(
                bank=bank_cmd,
                n=n_max, throttle=1.0, fire=False,
            )
        if (
            abs(obs.own_gamma) < math.radians(12.0)
            and abs(obs.own_bank) < math.radians(30.0)
        ) or self._phase_t >= 6.0:
            self._maneuver_done = True
        return level_out_action(obs)

    def _kvochur_bell(self, obs: Observation, back: bool = False) -> Action:
        """克沃丘尔钟摆（真·尾滑/bell）：近垂直拉起 → 零速尾滑 → 钟摆翻落。

        两种翻落方向共用拉起/尾滑时序：
          - 前倒（back=False）：机头越过顶点继续向前翻落，退出航向约反向；
          - 后倒（back=True）：顶点后反向推杆，机头沿原爬升路径倒回来，
            尾冲后退后改平，退出航向基本不变。

        与旧实现“边前飞边横转 360°”不同，这里把水平速度真正刹到零：
          1) 高 G 拉起：把机头抬到 80–90°；仰角过 60–70° 就收油门、
             全开减速板，不再用 100% 加力顶到顶点；
          2) 空速/VSI 接近零：卸载（n≈1）进入尾滑，机头停在 85° 附近，
             近零速只用小幅方向舵/矢量抑制偏摆，不做副翼闭环；
          3) 速度反向、机头自然甩过地平线；空速回到约 0.6 倍失速、方向舵
             重新有权限时给一次明确的偏航输入，让机头像钟摆一样绕飞机竖轴
             摆出去；
          4) 机头下俯、空速回来后平稳加油门，用渐进 G 拉平；在恢复的最后
             约 1/4 段叠加一次绕纵轴滚转（bell），到位即停、改平。
        """
        sign = self._def_sign
        # _def_sign=+1 表示左转（bank>0 为左滚），而 yaw_rate>0 是机头右偏，
        # 两者符号相反，这里统一成偏航指令方向。
        yaw_sign = -sign
        el = obs.own_nose_el
        gamma = obs.own_gamma
        # 竖直速度分量：负值表示已经在往下掉（尾滑末端）。
        vz = obs.own_speed * math.sin(gamma)
        speed_frac = obs.own_speed_frac
        self._bell_t += self.dt
        if self._bell_az0 is None:
            self._bell_az0 = obs.own_nose_az
            self._bell_az_prev = obs.own_nose_az
        # 机头方位累计只作诊断：不再用 360° 闭环驱动舵面。
        self._bell_yaw_acc += wrap_pi(obs.own_nose_az - self._bell_az_prev)
        self._bell_az_prev = obs.own_nose_az

        if self._phase == 0:
            # 拉起：高 G 把机头抬到近垂直。仰角过 60° 或航迹已经过 55°
            # 就收油门 + 全开减速板，靠阻力/重力减速，而不是继续加力。
            cut = el >= math.radians(60.0) or gamma >= math.radians(55.0)
            if (
                gamma >= math.radians(80.0)
                or el >= math.radians(86.0)
                or (el >= math.radians(70.0) and speed_frac < 0.60)
                or self._bell_t >= 10.0
            ):
                self._phase, self._bell_t = 1, 0.0
            n_pull = clamp(
                min(obs.own_n_avail * 0.95, self.n_pos_tol), 1.0, 7.5
            )
            return Action(
                bank=0.0, n=n_pull,
                throttle=0.0 if cut else 1.0,
                afterburner=0.0 if cut else 1.0,
                fire=False, brake=1.0 if cut else 0.0,
            )

        if self._phase == 1:
            # 顶点/尾滑：接近零速，卸载到 n≈1。不要再硬拉杆，否则会在
            # 失速前缘上停住、而不是自然进入尾滑。
            if (
                (vz < 4.0 and gamma < math.radians(70.0))
                or (el < math.radians(70.0) and speed_frac < 0.55)
                or self._bell_t >= 16.0
            ):
                self._phase, self._bell_t = 2, 0.0
                self._bell_kicked = False
                self._bell_kick_t = 0.0
            pitch = (
                -0.10 if el > math.radians(85.0)
                else 0.08 if el < math.radians(83.0) else 0.0
            )
            return Action(
                bank=0.0, n=1.0, throttle=0.0, afterburner=0.0,
                brake=1.0, fire=False, roll_free=True,
                pitch_rate=pitch, yaw_rate=-0.6 * obs.own_beta,
            )

        if self._phase == 2:
            # 钟摆段：机头已甩过地平线。空速还在零附近，舵面几乎没有权限，
            # 真正的“钟摆侧摆”要等空速回来、方向舵有效后在恢复初段给。
            if back:
                # 后倒：反向推杆，把机头沿原爬升路径压回来。这里不再额外
                # 偏航，保证改出航向基本不变。
                act = Action(
                    bank=0.0, n=1.0, throttle=0.0, afterburner=0.0,
                    brake=1.0, fire=False, roll_free=True,
                    pitch_rate=-0.55,
                    yaw_rate=-0.6 * obs.own_beta,
                )
            else:
                act = Action(
                    bank=0.0, n=1.0, throttle=0.0, afterburner=0.0,
                    brake=1.0, fire=False, roll_free=True,
                    pitch_rate=0.0, yaw_rate=-0.6 * obs.own_beta,
                )
            if el <= math.radians(5.0) or self._bell_t >= 10.0:
                self._phase, self._bell_t = 3, 0.0
                self._bell_kicked = False
                self._bell_kick_t = 0.0
            return act

        if self._phase == 3:
            # 机头已下俯：中立方向舵，平稳加油门，用限幅 10°/s 的渐进 G
            # 把航迹从近垂直俯冲拉回来（不在 q≈0 时猛拉）。
            act = level_out_action(obs, throttle=0.9)
            act.afterburner = 0.0
            # 下坠初段的一次性方向舵 kick：顶点附近空速接近零、方向舵没有
            # 权限，必须先等空速回到约 0.62 倍失速，再给一次明确输入，
            # 机头才会像钟摆一样绕竖轴摆出去；之后交回中立方向舵。
            # 后倒变体不踢舵，保持退出航向基本不变。
            if not back and not self._bell_kicked and speed_frac > 0.62:
                self._bell_kicked = True
                self._bell_kick_t = 2.2
            if not back and self._bell_kicked and self._bell_kick_t > 0.0:
                self._bell_kick_t = max(0.0, self._bell_kick_t - self.dt)
                # 混控里还会叠加 0.6β 的侧滑对齐项，低速恢复段 β 很大，
                # 不扣掉它舵令会被抵消掉一部分；扣掉后才是真正打满 ±0.9。
                act.yaw_rate = clamp(
                    yaw_sign * 0.9 - 0.6 * obs.own_beta, -0.9, 0.9
                )
            # 航迹回到 −20° 附近、空速回到 1.15 倍失速以上，舵面重新有效，
            # 这时才是“恢复最后 1/4 段”的 bell 滚转。
            if (
                gamma > math.radians(-20.0) and speed_frac > 1.15
            ) or self._bell_t >= 12.0:
                self._phase, self._bell_t = 4, 0.0
                # bell 的滚转按**角度**收口而不是按时间：低速恢复段滚转
                # 舵效还在恢复，同一个滚转角速度指令实际只能滚出约一半的
                # 角速度（0.6 rad/s 指令实测约 0.28 rad/s），若按“45°/速率”
                # 算时间，只能滚到 25°。这里给 4 s 兜底，滚到约 45° 即停。
                self._bell_roll_t = 4.0
            return act

        # 第 4 段：bell 的滚转 + 改平。
        if self._bell_roll_t > 0.0:
            self._bell_roll_t = max(0.0, self._bell_roll_t - self.dt)
            if abs(obs.own_bank) < math.radians(46.0):
                return Action(
                    bank=0.0, n=2.4, throttle=0.9, afterburner=0.0,
                    fire=False, roll_rate=sign * 0.60,
                )
            # 到位即停：达到约 45° 后不再继续滚，交给下面的改平段。
            self._bell_roll_t = 0.0
        act = level_out_action(obs, throttle=0.95)
        act.afterburner = 0.4 if speed_frac > 1.10 else 0.0
        if (
            abs(gamma) < math.radians(12.0)
            and abs(obs.own_bank) < math.radians(30.0)
            and speed_frac > 1.05
        ) or self._bell_t >= 8.0:
            self._maneuver_done = True
        return act

    def _kvochur_bell_back(self, obs: Observation) -> Action:
        """钟摆后倒变体：顶点后反向推杆，机头沿原爬升路径倒回来。"""
        return self._kvochur_bell(obs, back=True)

    def _herbst(self, obs: Observation) -> Action:
        """Herbst / J 转弯：拉起 55°、绕速度矢量横滚 160°，再反向压机头。

        和破 S 的差别在于不做完整半滚：只拉到大迎角，机体绕速度矢量转了
        大半圈后，升力矢量的方向已经翻到另一侧，此时继续拉杆，机头会沿
        一个新的方向压下来，航向在很小的空间里反转。用于被咬尾时把
        尾追瞬间变成迎头。
        """
        sign = self._def_sign
        n_pull = min(obs.own_n_avail, self.n_pos_tol, HIGH_G_N)
        # 横滚目标：120° 而不是 180°。完全倒扣时 sinφ≈0，拉杆只会在垂直面内
        # 下坠、航向几乎不变（那就退化成一次破 S）；保留 60° 的“未倒扣余量”
        # 让升力同时有很强的侧向分量，拉杆时航向才会在小空间里真正掉头。
        bank_turn = sign * (math.radians(120.0) / MAX_BANK)
        if self._phase == 0:
            if obs.own_gamma >= math.radians(45.0) or self._phase_t >= 3.5:
                self._phase, self._phase_t = 1, 0.0
            return Action(bank=0.0, n=n_pull, throttle=1.0, fire=False, roll_free=True)
        if self._phase == 1:
            # 绕速度矢量横滚到目标坡度。旧实现按固定 2.0 s 计时结束，
            # 实际转过的角度随速度/滚转速率变化（常只到 130°），下一段
            # “反拉”的方向也就跟着变，看起来时好时坏。改成坡度到位才换段。
            if abs(obs.own_bank) >= math.radians(110.0) or self._phase_t >= 2.5:
                self._phase, self._phase_t = 2, 0.0
            return Action(
                bank=bank_turn,
                n=2.0, throttle=1.0, fire=False,
            )
        if self._phase == 2:
            # 反转段：保持横滚后的坡度，升力矢量既有向下分量又有侧向分量，
            # 拉杆时机头一边俯冲一边横转，航向在小空间里掉头。旧实现用
            # roll_free 让坡度自由演化，实际常在中途反向滚回来，导致航迹
            # 左右摆（观感就是抖）。
            if obs.own_gamma <= math.radians(-20.0) or self._phase_t >= 5.0:
                self._phase, self._phase_t = 3, 0.0
            return Action(
                bank=bank_turn,
                n=n_pull, throttle=1.0, fire=False,
            )
        # 改平：先滚回水平，再限幅拉平航迹，动作完成。
        if (
            abs(obs.own_gamma) < math.radians(12.0)
            and abs(obs.own_bank) < math.radians(30.0)
        ) or self._phase_t >= 5.0:
            self._maneuver_done = True
        return level_out_action(obs)

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
        gamma = math.radians(-5.0)
        if obs.own_mil_endurance < self.BINGO_ENDURANCE:
            # Bingo（低油量脱离）期间保持平飞，不再下高：下高只会把飞机
            # 一路压进恢复门限，切回 RECOVER 拉起几米又切回 EXTEND，
            # 两族在门限上反复互换（低油量时看到的高 G 抖动正是这样来的）。
            gamma = 0.0
        return heading_turn(
            obs, away, gain=0.6, max_rate_deg=5.0,
            gamma_target=gamma,
        )

    def _crank(self, obs: Observation) -> Action:
        """偏置（CRANK）：把目标压在机头约 40°，保持照射同时拉开横向距离。

        40° 仍在中距弹 60° 的照射锥内，所以可以在偏置的同时继续制导自己的弹；
        横向位移会让对方弹的视线角速度变大、需要更多过载去追。

        方位误差必须是“当前方位 − 期望方位”。aim_action 的输入是
        “还要往哪边转多少”，写成“期望 − 当前”会变成正反馈：飞机一直
        往同一侧压坡度，方位越转越大，最后把机头甩出 60° 照射锥。

        偏置侧跟随目标的飞行方向（见 _update_crank_side），并把视线角速度
        作为前馈送给 heading_turn：目标在转弯时视线本身就在转，只有跟得
        上这个转速，靶标才会稳定停在 40° 附近，而不是在偏置角两侧来回摆。
        """
        rate_ff = self._update_crank_side(obs)
        az = obs.off_az - self._crank_sign * math.radians(40.0)
        # 偏置角附近让比例项连续地收敛到 0，而不是在 |az|<4° 时整档置零：
        # 置零会让期望转弯率从“前馈 − 比例”直接跳到“只有前馈”，而坡度是
        # atan2(v·ψ̇/G) 反解出来的，ψ̇ 一个小台阶就能让坡指令从 −70° 翻到
        # +40°，实际表现就是临近偏置角时又反向滚一次。
        if abs(az) < self.CRANK_AZ_BLEND:
            az *= 0.5 * (
                1.0
                - math.cos(math.pi * abs(az) / self.CRANK_AZ_BLEND)
            )
        # 旧实现限制在 3°/s，从迎头转到 40° 偏置要十几秒，往往弹都进末段了
        # 机头还没偏出去，偏置等于没做。6°/s 既能在 7 s 内建立偏置，又不会
        # 像 aim_action 的 16°/s 那样在偏置角上来回摆振。
        return heading_turn(
            obs, az, gain=0.7, max_rate_deg=6.0,
            gamma_target=math.radians(-3.0),
            rate_ff=rate_ff,
        )

    def _update_crank_side(self, obs: Observation) -> float:
        """按视线方位角速度维护偏置侧，并返回前馈转弯率。

        视线方位是**世界系**的（los_az），与机头/航迹的定义无关，对它差分
        得到的角速度同时包含目标横移与本机平移，正是“保持相对方位不变”
        所需的转弯率。ω>0 表示视线在向左转、靶标会往机头左侧漂，此时
        把机头也往左压，靶标就稳定停在左侧 40°；ω<0 反之。

        死区 + 迟滞：平飞时视线轻微摆动不切换偏置侧，避免 40° 偏置整档
        从左侧翻到右侧——那是一次约 128° 的反向滚，观感就是“机体乱摆”。
        """
        if obs.contact_quality < 2 or obs.dist >= 1e6:
            # 失去跟踪：保持上次的侧别，也不做前馈。
            self._los_az_prev = None
            return 0.0
        if self._los_az_prev is not None:
            raw = wrap_pi(obs.los_az - self._los_az_prev) / max(self.dt, 1e-3)
            self._los_rate = 0.85 * self._los_rate + 0.15 * raw
        self._los_az_prev = obs.los_az
        if self._los_rate > self.CRANK_SIDE_DEADBAND:
            self._crank_sign = 1.0
        elif self._los_rate < -self.CRANK_SIDE_DEADBAND:
            self._crank_sign = -1.0
        # 前馈限幅与 heading_turn 的最大转弯率一致，避免近距大角速度把
        # 坡度指令顶到 86°。
        return clamp(self._los_rate, -math.radians(6.0), math.radians(6.0))

    def _notch_desc(self, obs: Observation) -> Action:
        """三九下高：把对方雷达视线放到正侧方（3/9 线）并下高。

        凹口参考必须是**对方载机雷达到本机的视线**，不是来袭弹的相对方位：
        多普勒凹口是对着地杂波开的，本机径向速度是相对载机视线算的；弹在
        十几公里外、弹目视线和载机视线的夹角已经不可忽略。旧实现按弹方位
        转 90°，算出来的是“相对弹横穿”，载机雷达照样能看见，中制导不断。
        """
        sign = self._def_sign
        # 用视线相对**航迹**的方位（los_az − psi）：heading_turn 控制的是
        # 速度矢量的转向率，6DOF 下机头/炮线与航迹差一个迎角，按机头对
        # 90° 会把速度矢量留在凹口外。
        rel = wrap_pi(obs.los_az - obs.own_psi)
        az = rel - sign * math.radians(90.0)
        # 真实径向速度：视线与速度矢量的点积，口径与 sim.doppler_visibility
        # 完全一致（含俯仰项），而不是只看水平分量。
        radial = abs(obs.own_speed * (
            math.sin(obs.los_el) * math.sin(obs.own_gamma)
            + math.cos(obs.los_el) * math.cos(obs.own_gamma) * math.cos(rel)
        ))
        if radial <= self.NOTCH_HOLD_VR:
            # 已经进凹口：只做小速率修正，跟着视线转动保持横穿姿态。
            return heading_turn(
                obs, az, gain=0.6, max_rate_deg=2.0,
                gamma_target=math.radians(-6.0),
            )
        # 还没进凹口：远距用大速率抢进 90°（迎头 90° 要 8–10 s），
        # 接近凹口后收到 4°/s 避免冲过头再反向找。
        near = abs(az) < math.radians(30.0)
        return heading_turn(
            obs, az, gain=0.6 if near else 1.0,
            max_rate_deg=4.0 if near else 10.0,
            gamma_target=math.radians(-6.0),
        )

    def _f_pole(self, obs: Observation) -> Action:
        """F-pole：先偏置把自己那枚中距弹导到命中，再转三九。

        第一阶段与 CRANK 相同，持续到自己那枚弹落地（命中/脱靶/坠毁）或
        已经追不上对方（own_radar_ttg 退回无穷）；此后继续偏置没有收益，
        立刻转三九破对方载机雷达对我的照射/对方弹的中制导，而不是掉头
        “冷转”。旧实现的冷转航向仍沿对方视线（|v_r| 反而最大），对方雷达
        和弹上导引头都看得清清楚楚，只能把弹道稍微拉长一点，挡不住命中。
        """
        if self._phase == 0:
            own_mine = (
                obs.own_radar_in_flight > 0
                and obs.own_radar_ttg <= obs.incoming_missile_ttg + 1.0
            )
            if not own_mine:
                self._phase, self._phase_t = 1, 0.0
            else:
                return self._crank(obs)
        # 自己的弹已经落地/追不上：只要有雷达弹还在朝我来就继续三九。
        # 丢了制导的弹也是威胁（惯性弹道照样能近炸命中），这时改回偏置
        # 会把机头扭回弹道前方；没有来弹时才回到偏置保持姿态。
        if obs.incoming_missile_radar:
            return self._notch_desc(obs)
        return self._crank(obs)

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
