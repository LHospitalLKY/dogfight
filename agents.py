"""M2 Agent：几何态势判定 + 机动原语战术层。

分三层
------
1. 几何判定：把 Observation 压缩成“谁占位”的态势（进攻/防御/对头/中立/脱离）。
2. 原语选择：按态势、能量、距离和个体倾向选一个机动原语。
3. 原语执行：每个决策周期输出一次 Action(bank, n, throttle, fire)。

所有原语最终都落到 3DOF 的滚转角 + 过载指令上，因此可以被同一套
点质量内核执行，也方便以后交给 RL 直接学习。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

from sim import G, MAX_BANK, Action, Observation, clamp, wrap_pi

# 期望角速度的软上限，避免目标突然跳到正后方时拉出离谱的指令。
MAX_GAMMA_DOT = math.radians(20.0)
# 能量-高度约束：爬升角上限由比剩余功率 Ps 决定，高速时才允许额外 zoom。
ZOOM_ALLOWANCE = math.radians(25.0)
GAMMA_CAP = math.radians(55.0)
GAMMA_FLOOR = math.radians(-60.0)

# 不在不可逃逸区时，中距弹的估计命中概率要达到该门限才值得发射。
# 低于门限只意味着“现在打大概率浪费一枚弹”，带着弹继续占位更划算。
RADAR_PK_MIN = 0.80

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
# 防御转向反转的时间门槛：威胁持续位于反侧超过该时长才换向。
# 否则每次重新进入防御原语都会按瞬时方位翻一次，形成“无收益的反向机动”。
DEF_REVERSE_TIME = 1.5


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
    # 先滚转、后拉杆：大坡度转向时若一上来就拉满过载，滚入的一两秒里升力
    # 仍主要在竖直方向，飞机会先被拉起来，等坡度到位再掉回去。追尾、脱离
    # 这类动作里就表现为“莫名其妙抬升高度、同时又掉速”。
    if abs(phi) - abs(obs.own_bank) > math.radians(30.0):
        n = min(n, 1.6)
    return Action(
        bank=clamp(phi / MAX_BANK, -1.0, 1.0),
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
    """有锁定、在发射包线内、且手上没有在飞弹时打一发。

    对头也允许发射：锁定本身已经保证目标在导引头视场内，
    不再额外限制离轴角。对头互射后靠后续的规避动作决出结果。
    命中与否完全交给内核的导弹动力学。
    """
    return (
        obs.own_locked
        and obs.own_missiles > 0
        and obs.in_missile_envelope
        and obs.own_missiles_in_flight < 1
        and obs.own_speed_frac > 1.35
        and obs.dist > max(obs.missile_min_range, 600.0)
    )


def should_launch_radar(obs: Observation) -> bool:
    """雷达制导中距弹（主动弹）的发射条件。

    中距弹的价值在超视距，太近时不如留着红外弹；也不能一次把两发都打光，
    先打一发、等命中或丢锁后再补射。主动弹在中制导段仍需要载机雷达
    保持跟踪与数据链，末段自主截获后即可脱离，因此这里同样要求速度余量。

    打不打还要看值不值：目标进了不可逃逸区就打；否则按弹道前推估计的
    命中概率判断，低于门限就带着弹继续占位，避免远距乱射把对局提前打没。
    """
    return (
        obs.own_radar_locked
        and obs.own_radar_missiles > 0
        and obs.in_radar_envelope
        and obs.own_radar_in_flight < 1
        and obs.own_speed_frac > 1.35
        and obs.dist > max(obs.radar_min_range, 2500.0)
        and (obs.radar_nez or obs.radar_pk_est >= RADAR_PK_MIN)
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
    SCISSOR_PERIOD = 1.4
    LOW_ENERGY_FRAC = 1.55   # 速度低于 1.55 倍当地失速即进入能量保护
    N_MARGIN = 1.25          # 允许短时超出维持过载的倍数，其余能量透支被禁止
    MISSILE_EVADE_DIST = 9000.0  # 来袭格斗弹进入该距离就开始规避
    SARH_EVADE_DIST = 16000.0    # 中距弹的告警距离更远：超视距阶段就要开始偏置
    MISSILE_BREAK_DIST = 4500.0  # 更远时用能量友好的横转/俯冲拉长 F-pole
    COBRA_DIST = 1800.0          # 矢量喷口机型的过失速反制距离：来弹贴脸才用
    LOCK_BREAK_DIST = 8000.0     # 被对方锁定且进入该距离后评估是否防御
    OPENING_DIST = 5000.0        # 开局占位阶段：距离大于该值时不直冲对头
    MERGE_START = 5000.0     # 进入该距离开始领先转弯，抢交汇角度
    # 交汇方案在贴身盘旋阶段的保持距离。取 2200 而不是 1800，
    # 是为了让“保持方案”和停用的距离门限错开，避免在门限上每拍翻转转向侧。
    MERGE_HOLD = 2200.0
    MERGE_TIMEOUT = 12.0         # 交汇抢位方案的保持时间 s

    # 导弹来袭时按顺序轮换的防御动作；被锁定（未射弹）时也有一套。
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
        else:
            # 能量型（Su-57）：偏置小、爬升存能量，留给交汇后的垂直机动。
            self._open_offset = math.radians(self._rng.uniform(5.0, 13.0))
            self._open_climb = math.radians(self._rng.uniform(0.0, 12.0))
            self._open_throttle = self._rng.uniform(0.85, 1.0)

        self.primitive = "PURE"
        self.situation = ""
        self._hold = 0.0
        self._phase = 0
        self._phase_t = 0.0
        self._scissor_sign = 1.0
        self._scissor_t = 0.0
        self._def_sign = 1.0
        self._def_reverse_t = 0.0
        self._def_engaged = False
        self._opening_done = False
        # 开局爬升目标的当前值：以有限速率逼近 _open_climb，避免 Ps 在门限
        # 附近抖动时把俯仰指令整档切换。
        self._open_climb_cur = 0.0
        self._far = False

        # 交汇抢位方案：转向侧、单环/双环、以及方案剩余保持时间。
        self._merge_sign = 1.0
        self._merge_plan = ""
        self._plan_ttl = 0.0
        self._turn_sign = 1.0

        # 人体过载耐受状态：0=完全恢复，1=到达耐受极限（G-LOC 边界）。
        self.g_load = 0.0
        self.g_load_neg = 0.0
        self._gloc_t = 0.0
        self.n_pos_tol = G_PEAK
        self.n_neg_tol = G_NEG_PEAK

    # ---------- 主入口 ----------

    def act(self, obs: Observation) -> Action:
        self._update_g_tolerance(obs)
        # 防御转向的连续性：威胁只在持续位于反侧时才换向，避免“一进防御就反转”。
        if obs.incoming_missile_dist < self.SARH_EVADE_DIST:
            az = obs.incoming_missile_az
            if az * self._def_sign < 0.0 and abs(az) > math.radians(60.0):
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

        want = self._sticky_defense(self._choose(sit, obs))
        # 安全与防御类原语可立即抢占，不受最短驻留时间限制。
        emergency = (
            "RECOVER", "POSITION", "BEAM", "BREAK", "SPLIT_S",
            "BARREL", "SCISSORS", "IMMELMANN", "LOOP",
            "CRANK", "NOTCH_DESC", "F_POLE", "COBRA",
            "LEAD_TURN", "HIGH_G_BREAK", "HIGH_G_SPLIT_S", "HIGH_G_BARREL",
        )
        bypass = want in emergency or self.primitive in emergency
        need = self.YOYO_HOLD if self.primitive in ("HIGH_YOYO", "LOW_YOYO") else self.MIN_HOLD
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

    # ---------- 第二层：原语选择 ----------

    def _choose(self, sit: _Situation, obs: Observation) -> str:
        # 安全优先：贴着边界或高度上下限时先回场。
        # 转弯半径随速度平方增长，所以边界余量按速度缩放，避免高速时来不及转回。
        edge_need = max(900.0, 8.0 * obs.own_speed)
        if obs.edge_margin < edge_need or obs.z_margin_low < 600.0 or obs.z_margin_high < 600.0:
            return "RECOVER"
        # 眼镜蛇是瞬态机动：机头拉到极限迎角后必须立刻改出，否则会变成
        # 挂在高迎角上持续爬升，速度被抽干、反而成了靶子。
        if self.primitive == "COBRA" and self._phase_t > 1.5:
            return "SPLIT_S" if obs.z_margin_low > 1200.0 else "BREAK"
        # 规避过程中仍要守住能量：机头已经高高抬起、速度又不够时，
        # 继续压坡度拉过载只会变成爬升耗能，先俯冲把航迹角压下来。
        if (
            self.primitive in self.DEFENSE_SET
            and obs.own_gamma > math.radians(25.0)
            and obs.own_speed_frac < 1.70
        ):
            return "SPLIT_S"
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
                return "COBRA"
            # 雷达制导中距弹按 FC3 手册的超视距防御流程处理：
            # 自己的弹先到就 F-pole 偏置，否则三九下高。
            if obs.incoming_missile_radar:
                anti = self._anti_radar(obs)
                if anti is not None:
                    return anti
            # 自己的雷达弹还处在需要照射/数据链的中制导段：远距先保持机头
            # 指向把弹导完；主动弹一旦末段截获就不再需要（own_radar_support）。
            keep_illum = (
                obs.own_radar_support
                and obs.radar_tracking
                and obs.incoming_missile_dist > self.MISSILE_BREAK_DIST
            )
            if not keep_illum:
                if obs.incoming_missile_dist > self.MISSILE_BREAK_DIST:
                    return "BEAM" if obs.z_margin_low > 1500.0 else "SPLIT_S"
                return self._pick_defense(obs, vs_missile=True)
        # 过载疲劳接近失控：没有近距威胁时先卸载恢复，避免 G-LOC 白送人头。
        if (
            (self.g_load > 0.90 or self.g_load_neg > 0.90)
            and obs.incoming_missile_dist > self.MISSILE_BREAK_DIST
        ):
            return "UNLOAD"
        # 被对方导引头/雷达锁定：先评估是否值得转入防御。自己也占进攻位、
        # 还没打弹、或对方已经没弹时，继续进攻往往比被动规避收益更高。
        if (
            (obs.foe_locked or obs.foe_radar_locked)
            and obs.dist < self.LOCK_BREAK_DIST
            and obs.own_speed_frac > 1.30
            and self._should_defend(obs, sit)
        ):
            return self._pick_defense(obs, vs_missile=False)
        # 能量-高度约束：连 1g 平飞都维持不住（Ps<0）说明已超过能量升限，
        # 再拉机动只会把速度抽干，必须先卸载俯冲换回能量。
        if obs.own_ps_level < -10.0 and obs.own_alt > 11000.0:
            return "UNLOAD"
        # 打空或重伤就不再纠缠。只有自己明显伤得更重才脱离：双方都被打残时
        # 继续缠斗，否则两架重伤机各自逃跑会把对局拖成“飞出空域”。
        if obs.own_ammo <= 1.0 or (
            obs.own_hp_frac < 0.18
            and obs.foe_hp_frac > max(0.25, obs.own_hp_frac + 0.08)
        ):
            return "EXTEND"
        # 能量纪律：速度接近失速时先卸载俯冲，不允许挂着机头往上爬
        if self.primitive == "UNLOAD":
            # 必须同时满足“速度回来了”和“机头已经压下来”，避免反复抬头
            if obs.own_speed_frac < 1.40 or obs.own_gamma > math.radians(15.0):
                return "UNLOAD"
        elif obs.own_speed_frac < 1.20:
            return "UNLOAD"

        if sit.name == "SEPARATED":
            # 超视距阶段按“先视、先射、先撤”推进。
            # 自己的中距弹还在中制导时必须保持机头指向（数据链/照射），
            # 占位/偏置动作只能在这里之后做；照射一断弹就变无控弹道。
            if obs.own_radar_support and obs.radar_tracking:
                return "PURE"
            # 开局先按能量/角度倾向占位。
            if not self._opening_done:
                return "POSITION"
            # 进入交汇距离后提前做领先转弯，切进对手转弯圆环的内侧。
            if obs.dist < self.MERGE_START and obs.aspect > math.radians(120.0):
                return "LEAD_TURN"
            return "PURE"

        if sit.name == "HEAD_ON":
            # 交汇瞬间按抢位方案执行：单环抢先指向，双环持续盘旋。
            # 能量明显占优时改用高 Yo-Yo 把速度换成高度再从上方压下来。
            if self._merge_plan == "ONE_CIRCLE":
                return "ONE_CIRCLE"
            if self._merge_plan == "TWO_CIRCLE":
                return "TWO_CIRCLE"
            if sit.e_adv > 0.0 and self.energy_pref >= 0.5:
                return "HIGH_YOYO"
            return "LEAD" if self.aggression > 0.6 else "BREAK"

        # 交汇后的第一圈仍按抢位方案执行：HEAD_ON 只是一个瞬时窗口，
        # 双方真正开始盘旋时已转为 NEUTRAL/DEFENSIVE，若不延续方案，
        # 单环/双环的决策就形同虚设。
        if (
            self._plan_ttl > 0.0
            and self._merge_plan
            and obs.dist < self.MERGE_HOLD
            and obs.aspect > math.radians(55.0)
        ):
            return "ONE_CIRCLE" if self._merge_plan == "ONE_CIRCLE" else "TWO_CIRCLE"

        if sit.name == "DEFENSIVE":
            # 只有对方真的冲过头（接近率为正、距离很近且机头已偏）才做剪刀；
            # 仅仅“贴得近”就反复剪刀，会变成每 1.4 s 一次的无收益反向机动。
            if sit.overshoot:
                return "SCISSORS"
            return "BREAK"

        if sit.name == "OFFENSIVE":
            if sit.close:
                return "LAG" if sit.overshoot else "PURE"
            if sit.far:
                return "PURE"
            # 角度还不够、速度与过载耐受都富余时用高 G 转弯抢角度。
            # 机体本身拉不出 6.5 g 时（高空低速）换成普通追逐，等高 G 变得
            # “名副其实”再拉，否则只是白白把能量抽干。
            if (
                obs.own_speed_frac > 1.55
                and self.n_pos_tol > 6.5
                and obs.own_n_avail >= HIGH_G_MIN_AVAIL
            ):
                return "HIGH_G_TURN"
            return "LEAD"

        # 近距缠斗但机头落后：速度与过载耐受都富余时用高 G 转弯抢角度。
        if (
            not sit.far
            and sit.angle > math.radians(40.0)
            and obs.own_speed_frac > 1.60
            and self.n_pos_tol > 6.5
            and obs.own_n_avail >= HIGH_G_MIN_AVAIL
        ):
            return "HIGH_G_TURN"

        # NEUTRAL：贴身缠斗时用垂直机动消耗/补充能量
        if sit.close:
            high_ok = obs.own_ps_level > 5.0 and obs.z_margin_high > 4000.0
            low_ok = obs.z_margin_low > 1200.0
            if obs.own_speed_frac > 1.7 and self.energy_pref >= 0.5 and high_ok:
                return "HIGH_YOYO"      # 太快：把多余速度换成高度
            if obs.own_speed_frac < 1.5 and self.energy_pref < 0.5 and low_ok:
                return "LOW_YOYO"       # 太慢：俯冲增速再拉起
        if sit.angle < math.radians(25.0):
            return "LEAD" if not sit.far else "PURE"
        return "PURE"

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
        """按威胁剩余时间分档选择规避动作，全程保持同一转向侧。

        不再每 3 s 轮换一套动作：末段刚建立起来的三九/横转姿态一旦被换掉，
        导弹只需修正很小的角度就能重新压回前置量，前面拉开的脱靶量全废。
        分档逻辑是远距横转消耗弹能量（同时给箔条创造多普勒条件）→ 中距
        保持横转并做高 G 修正 → 近距才切硬甩。
        """
        if obs.own_speed_frac < self.LOW_ENERGY_FRAC:
            return "SPLIT_S" if obs.z_margin_low > 1200.0 else "BREAK"
        if vs_missile:
            ttg = obs.incoming_missile_ttg
            if ttg > 7.0:
                ladder = ("NOTCH_DESC", "BEAM")
            elif ttg > 3.0:
                ladder = ("BEAM", "HIGH_G_BREAK", "HIGH_G_BARREL")
            else:
                ladder = ("HIGH_G_BREAK", "BREAK", "HIGH_G_SPLIT_S")
        else:
            ladder = ("BEAM", "HIGH_G_BREAK", "BREAK", "SPLIT_S")
        for cand in ladder:
            if self._defense_ok(cand, obs):
                return cand
        return "BREAK" if vs_missile else "BEAM"

    def _defense_ok(self, primitive: str, obs: Observation) -> bool:
        """防御原语的能量/高度/距离/过载耐受准入条件。"""
        if primitive.startswith("HIGH_G"):
            # 高 G 规避要同时有速度和过载余量；疲劳时拉不出高 G，不如老实横转。
            ok = (
                obs.own_speed_frac > 1.45
                and self.n_pos_tol > 6.0
                and obs.own_n_avail >= HIGH_G_MIN_AVAIL
            )
            if primitive in ("HIGH_G_SPLIT_S", "HIGH_G_BARREL"):
                ok = ok and obs.z_margin_low > 600.0
            return ok
        if primitive == "SPLIT_S":
            return obs.z_margin_low > 1200.0 and obs.own_speed_frac > 1.35
        if primitive == "BARREL":
            return obs.own_speed_frac > 1.35
        if primitive == "SCISSORS":
            return obs.dist < 1200.0
        if primitive == "IMMELMANN":
            return (
                obs.own_speed_frac > 1.70
                and obs.z_margin_high > 2500.0
                and obs.own_ps_level > 0.0
            )
        if primitive == "LOOP":
            return (
                obs.own_speed_frac > 1.85
                and obs.z_margin_high > 3000.0
                and obs.own_ps_level > 0.0
            )
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
            az = (
                obs.incoming_missile_az
                if obs.incoming_missile_dist < self.SARH_EVADE_DIST
                else obs.off_az
            )
            self._def_sign = 1.0 if az >= 0.0 else -1.0
            self._def_reverse_t = 0.0
        if primitive in (
            "LEAD_TURN", "ONE_CIRCLE", "TWO_CIRCLE", "HIGH_G_TURN",
        ):
            # 抢位转向在进入时锁定：交汇前方位角很小，直接用当前方位会抖动。
            if abs(obs.off_az) < math.radians(20.0):
                self._turn_sign = self._merge_sign
            else:
                self._turn_sign = 1.0 if obs.off_az >= 0.0 else -1.0
        if primitive == "SCISSORS":
            # 初始剪刀方向跟着威胁走，之后周期性反转。
            self._scissor_sign = 1.0 if obs.off_az >= 0.0 else -1.0
            self._scissor_t = self.SCISSOR_PERIOD

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
        if primitive.startswith("HIGH_G") and obs.own_speed_frac > 1.30:
            return act
        if obs.own_speed_frac < self.LOW_ENERGY_FRAC:
            cap = max(1.0, obs.own_n_sustain * self.N_MARGIN)
            if act.n > cap:
                act.n = cap
        return act

    def _dispatch(self, primitive: str, obs: Observation) -> Action:
        if primitive == "RECOVER":
            return self._recover(obs)
        if primitive == "POSITION":
            return self._position(obs)
        if primitive == "LEAD_TURN":
            return self._lead_turn(obs)
        if primitive in ("ONE_CIRCLE", "TWO_CIRCLE"):
            return self._circle(obs, primitive)
        if primitive == "HIGH_G_TURN":
            return self._high_g_turn(obs)
        if primitive == "HIGH_G_BREAK":
            return self._high_g_break(obs)
        if primitive == "HIGH_G_SPLIT_S":
            return self._high_g_split_s(obs)
        if primitive == "HIGH_G_BARREL":
            return self._high_g_barrel(obs)
        if primitive == "BEAM":
            return self._beam(obs)
        if primitive == "PURE":
            return pursuit_action(obs, obs.off_az, obs.off_el)
        if primitive == "LEAD":
            az, el = lead_point(obs, 1.0)
            return pursuit_action(obs, az, el)
        if primitive == "LAG":
            az, el = lead_point(obs, -0.6)
            return pursuit_action(obs, az, el)
        if primitive == "BREAK":
            return self._break(obs)
        if primitive == "SPLIT_S":
            return self._split_s(obs)
        if primitive == "BARREL":
            return self._barrel(obs)
        if primitive == "IMMELMANN":
            return self._immelmann(obs)
        if primitive == "LOOP":
            return self._loop(obs)
        if primitive == "SCISSORS":
            return self._scissors(obs)
        if primitive == "HIGH_YOYO":
            return self._high_yoyo(obs)
        if primitive == "LOW_YOYO":
            return self._low_yoyo(obs)
        if primitive == "CRANK":
            return self._crank(obs)
        if primitive == "NOTCH_DESC":
            return self._notch_desc(obs)
        if primitive == "F_POLE":
            return self._f_pole(obs)
        if primitive == "COBRA":
            return self._cobra()
        if primitive == "EXTEND":
            return self._extend(obs)
        if primitive == "UNLOAD":
            return self._unload()
        return aim_action(obs, obs.off_az, obs.off_el)

    def _recover(self, obs: Observation) -> Action:
        """边界/高度安全优先：先转向空域中心，再修正高度。"""
        el_target = 0.0
        if obs.z_margin_low < 600.0:
            el_target = math.radians(12.0)
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
        act.n = clamp(act.n, 0.6, 2.2)
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
        return turn_action(obs, self._turn_sign, n, el=0.0)

    def _circle(self, obs: Observation, plan: str) -> Action:
        """单环 / 双环的选择执行。

        单环：拉满瞬时过载，双方对头互相切入，靠首圈角度优势先指向对方。
        双环：用能长期维持的过载同向盘旋，逐步把对手甩到机头前方。
        """
        if plan == "ONE_CIRCLE":
            n = min(obs.own_n_avail, self.n_pos_tol)
            return turn_action(obs, self._turn_sign, n, el=math.radians(3.0))
        n = min(max(obs.own_n_sustain * 1.10, 2.0), self.n_pos_tol, G_SUSTAIN)
        return turn_action(obs, self._turn_sign, n, el=math.radians(-2.0))

    def _high_g_turn(self, obs: Observation) -> Action:
        """高 G 转弯（进攻）：角点速度附近拉满瞬时过载抢角度。"""
        n = min(obs.own_n_avail, self.n_pos_tol, HIGH_G_N)
        act = turn_action(obs, self._turn_sign, n, el=math.radians(2.0))
        # turn_action 的水平转弯率按 0.9×可用过载收敛；高 G 原语要真正拉满，
        # 否则指令 8 g 实际只有 7 g 出头，HUD 上看起来仍是普通盘旋。
        act.n = max(1.0, min(n, obs.own_n_avail))
        return act

    def _high_g_break(self, obs: Observation) -> Action:
        """高 G 急转（防御）：最大瞬时过载拉满，略微下压保持速度。"""
        n = min(obs.own_n_avail, self.n_pos_tol, HIGH_G_N)
        return turn_action(obs, self._def_sign, n, el=math.radians(-5.0))

    def _high_g_split_s(self, obs: Observation) -> Action:
        """高 G 破 S：滚转倒扣后拉满过载俯冲，快速掉高度换速度并转向。"""
        n = min(obs.own_n_avail, self.n_pos_tol, HIGH_G_N)
        if self._phase == 0:
            if self._phase_t >= 0.5:
                self._phase, self._phase_t = 1, 0.0
            return Action(bank=self._def_sign, n=1.2, throttle=1.0, fire=False)
        act = aim_action(
            obs, self._def_sign * math.radians(120.0),
            math.radians(-35.0) - obs.own_gamma, gain=3.0,
        )
        act.n = max(1.0, min(n, act.n))
        return act

    def _high_g_barrel(self, obs: Observation) -> Action:
        """高 G 载重滚转：拉满过载的同时滚转，形成高 G 螺旋下降。"""
        n = min(obs.own_n_avail, self.n_pos_tol, HIGH_G_N)
        bank = self._def_sign * 0.9 * math.sin(self._phase_t * 2.0 * math.pi / 2.0)
        return Action(bank=bank, n=max(1.0, n * 0.9), throttle=1.0, fire=False)

    def _level_cap(self, obs: Observation, frac: float = 1.0) -> float:
        """维持当前航迹角所需的过载上限。

        规避动作取它的一个比例（<1）就能保证不爬升：3DOF/6DOF 里
        坡度偏小时若仍拉正过载，飞机会直接转成爬升，能量掉得比被弹
        打中还快。按实际坡度算上限，可以让坡度回中的瞬间自动卸载。
        """
        return frac * math.cos(obs.own_gamma) / max(math.cos(obs.own_bank), 0.15)

    def _break(self, obs: Observation) -> Action:
        """最大过载急转，方向在进入时锁定，同时略微下压保持速度。"""
        n = max(1.0, min(obs.own_n_avail, self._level_cap(obs, 0.95)))
        return Action(bank=self._def_sign, n=n, throttle=1.0, fire=obs.in_gun_cone)

    def _beam(self, obs: Observation) -> Action:
        """被锁定后的横转：向锁定一侧压坡度，把对方甩到视场边缘。"""
        n = max(1.0, min(obs.own_n_avail * 0.95, self._level_cap(obs, 0.90)))
        return Action(
            bank=self._def_sign,
            n=n,
            throttle=1.0,
            fire=False,
        )

    def _split_s(self, obs: Observation) -> Action:
        """半滚倒转：先滚成倒扣，再拉杆经垂直向下改出到反向航向。

        旧实现只是“压坡度推杆俯冲”，航迹角永远越不过 −40°，本质上是一次
        俯冲逃逸。现在倒扣后仍给正过载：倒飞时升力朝下，n>0 才能让航迹角
        真正转过 −90° 并完成 180° 转向。
        """
        sign = self._def_sign
        if self._phase == 0:
            # 滚转到位（含符号），期间不拉杆以免把坡度带成爬升。
            if abs(obs.own_bank) >= math.radians(150.0) or self._phase_t >= 1.2:
                self._phase, self._phase_t = 1, 0.0
            return Action(bank=sign, n=1.0, throttle=1.0, fire=False)
        n = max(1.0, min(obs.own_n_avail * 0.9, G_PEAK))
        if self._phase == 1:
            if obs.own_gamma <= math.radians(-55.0) or self._phase_t >= 3.0:
                self._phase, self._phase_t = 2, 0.0
            return Action(bank=sign, n=n, throttle=1.0, fire=False)
        # 改出：过了垂直向下就把航迹拉回水平，坡度回到翼平。
        return aim_action(obs, 0.0, -obs.own_gamma, gain=1.6, throttle=0.6)

    def _barrel(self, obs: Observation) -> Action:
        """防御性滚筒：滚转角在小幅范围内摆动，形成螺旋规避航迹。

        幅度和过载都刻意压低：3DOF/6DOF 里当坡度摆回接近翼平而仍在
        拉正过载时，飞机会直接变成爬升，连续滚筒会把能量抽干。
        """
        sign = self._def_sign
        bank = sign * 0.65 * math.sin(self._phase_t * 2.0 * math.pi / 2.4)
        n = min(
            max(obs.own_n_avail * 0.5, 2.0),
            self._level_cap(obs, 0.90),
        )
        return Action(
            bank=bank,
            n=n,
            throttle=1.0,
            fire=False,
        )

    def _immelmann(self, obs: Observation) -> Action:
        """殷麦曼：翼平拉起过顶，倒扣滚转 180°，再拉杆改平并反向。

        旧实现只爬到 γ≈45° 就平转 150°，既没有过顶也没有真正的半滚，
        在能量上等价于一次大坡度盘旋。现在按真实三段执行：拉起 → 顶部滚转
        → 拉杆改出（倒扣时正过载会把机头压回水平）。
        """
        sign = self._def_sign
        if self._phase == 0:
            if obs.own_gamma >= math.radians(70.0) or self._phase_t >= 2.2:
                self._phase, self._phase_t = 1, 0.0
            n = max(1.0, min(obs.own_n_avail * 0.9, G_PEAK))
            return Action(bank=0.0, n=n, throttle=1.0, fire=False)
        if self._phase == 1:
            # 顶部滚转：把机腹翻到上方，为随后的“拉杆改平”做准备。
            if abs(obs.own_bank) >= math.radians(150.0) or self._phase_t >= 1.4:
                self._phase, self._phase_t = 2, 0.0
            return Action(bank=sign, n=1.2, throttle=1.0, fire=False)
        return aim_action(obs, 0.0, -obs.own_gamma, gain=1.8, throttle=0.8)

    def _loop(self, obs: Observation) -> Action:
        """内筋斗：翼平持续拉杆，在垂直平面内完成 360° 回转。

        旧实现分三段追目标航迹角（65° → −40° → 0），最多只画了半个圈就
        退出。现在同向持续拉杆、不再用 _level_cap（它按 cos γ 收过载，正好
        在过顶时把过载压成 0），航迹角依次越过 +90°、−90° 后改平。
        """
        if self._phase == 0:
            if obs.own_gamma >= math.radians(75.0) or self._phase_t >= 2.5:
                self._phase, self._phase_t = 1, 0.0
        elif self._phase == 1:
            if obs.own_gamma <= math.radians(15.0) or self._phase_t >= 3.0:
                self._phase, self._phase_t = 2, 0.0
        elif self._phase == 2:
            if obs.own_gamma <= math.radians(-70.0) or self._phase_t >= 3.0:
                self._phase, self._phase_t = 3, 0.0
        if self._phase == 3:
            return aim_action(obs, 0.0, -obs.own_gamma, gain=1.6)
        n = max(1.0, min(obs.own_n_avail * 0.9, G_PEAK))
        return Action(bank=0.0, n=n, throttle=1.0, fire=False)

    def _scissors(self, obs: Observation) -> Action:
        """滚转剪刀：周期性反向最大过载转弯，逼对手冲过头。"""
        if self._scissor_t <= 0.0:
            self._scissor_sign = -self._scissor_sign
            self._scissor_t = self.SCISSOR_PERIOD
        return Action(
            bank=self._scissor_sign,
            n=obs.own_n_avail * 0.95,
            throttle=0.9,
            fire=obs.in_gun_cone,
        )

    def _high_yoyo(self, obs: Observation) -> Action:
        """高 Yo-Yo：先把速度换成高度（爬升角受 Ps 约束），再俯冲重新指向目标。"""
        if self._phase == 0:
            if self._phase_t >= 1.2 or obs.own_gamma >= math.radians(28.0):
                self._phase, self._phase_t = 1, 0.0
            return aim_action(obs, obs.off_az, math.radians(30.0) - obs.own_gamma)
        if self._phase_t >= 1.0:
            self._phase, self._phase_t = 0, 0.0
        az, el = lead_point(obs, 1.0)
        return aim_action(obs, az, el, gain=1.6, throttle=0.4)

    def _low_yoyo(self, obs: Observation) -> Action:
        """低 Yo-Yo：先俯冲增速，再拉起抢占角度。"""
        if self._phase == 0:
            if self._phase_t >= 0.9 or obs.own_gamma <= math.radians(-22.0):
                self._phase, self._phase_t = 1, 0.0
            return aim_action(
                obs, obs.off_az, math.radians(-25.0) - obs.own_gamma, throttle=0.35
            )
        if self._phase_t >= 1.1:
            self._phase, self._phase_t = 0, 0.0
        return aim_action(obs, obs.off_az, obs.off_el, gain=1.8)

    def _extend(self, obs: Observation) -> Action:
        """脱离：掉头背离对手，满油门拉开距离。"""
        away = wrap_pi(obs.los_az + math.pi - obs.own_psi)
        act = aim_action(obs, away, math.radians(-5.0) - obs.own_gamma, gain=1.3)
        # 脱离用能长期维持的过载，航迹角只压到略低于水平。原来不做限制，
        # 大坡度掉头时会先拉 7 g 把速度换成高度，再挂着机头慢慢掉回来，
        # 看起来就是“逃跑时不断爬升、同时掉速”。
        cap = max(min(obs.own_n_avail, max(obs.own_n_sustain * 1.15, 2.0)), 1.0)
        act.n = min(act.n, cap)
        return act

    def _crank(self, obs: Observation) -> Action:
        """偏置（CRANK）：把目标压在机头约 40°，保持照射同时拉开横向距离。

        40° 仍在中距弹 60° 的照射锥内，所以可以在偏置的同时继续制导自己的弹；
        横向位移会让对方弹的视线角速度变大、需要更多过载去追。

        方位误差必须是“当前方位 − 期望方位”。aim_action 的输入是
        “还要往哪边转多少”，写成“期望 − 当前”会变成正反馈：飞机一直
        往同一侧压坡度，方位越转越大，最后把机头甩出 60° 照射锥。
        """
        az = obs.off_az - self._def_sign * math.radians(40.0)
        return aim_action(obs, az, math.radians(-4.0) - obs.own_gamma, gain=1.9)

    def _notch_desc(self, obs: Observation) -> Action:
        """三九下高：把威胁放到正侧方（3/9 线）并下高。

        横转拉长对方的拦截几何、下高进入更稠密的空气消耗弹的能量；
        同时让自己的机头逐渐脱离对方雷达照射锥的跟随。
        """
        az = obs.incoming_missile_az - self._def_sign * math.radians(90.0)
        return aim_action(obs, az, math.radians(-8.0) - obs.own_gamma, gain=2.0)

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
        return aim_action(obs, away, math.radians(-8.0) - obs.own_gamma, gain=1.7)

    def _cobra(self) -> Action:
        """过失速眼镜蛇：把机头拉到失速迎角以上，靠矢量喷口维持姿态。

        近距来弹时用来制造过冲：导弹的导引头还追着原来的速度矢量，
        机头已经抬起、速度快速衰减，弹会从下方冲过去。3DOF 下没有迎角
        状态，这个指令不会生效，所以只对 6DOF + 矢量喷口机型开放。
        """
        return Action(bank=0.0, n=1.0, throttle=0.4, fire=False, post_stall=True)

    def _unload(self) -> Action:
        """失速边缘的补救：压平机翼、推杆改出，用高度换回速度。"""
        return Action(bank=0.0, n=-1.2, throttle=1.0, fire=False)
