"""M2 内核：2DOF / 3DOF 点质量 + 6DOF 刚体狗斗模型。

坐标约定
--------
x 向东、y 向北、z 向上（单位米），psi 为航向角（从 x 轴逆时针），
gamma 为航迹倾角（抬头为正），v 为真空速（m/s）。

2DOF/3DOF 控制量为过载 n（g）、滚转角 phi（rad）和油门：

    v_dot  = (T - D)/m - g sin(gamma)
    gamma_dot = g/v * (n cos(phi) - cos(gamma))
    psi_dot   = g/(v cos(gamma)) * n sin(phi)

6DOF 在此基础上把战术层的 (bank, n) 交给增稳内回路，翻译成副翼/升降舵/
方向舵偏度，再由刚体六自由度方程（气动力/力矩 + 陀螺耦合 + 欧拉运动学）
积分。机体系 x 前、y 右、z 下（右手系），phi 右滚为正、theta 抬头为正。

比能量 E = g z + v^2/2，所以爬升与加速可以互相兑换（能量机动的基础）。
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass

G = 9.81
MAX_BANK = math.radians(80.0)      # 常规滚转角刻度（Action.bank=1 对应的角度）
# 战术层允许指令到倒扣：Action.bank 以 MAX_BANK 为单位，最大可到 ±180°。
BANK_CMD_MAX = math.pi / MAX_BANK
# 航迹角/俯仰角不再被人为夹断：筋斗与半滚倒转都必须能越过 ±90°。
# 只在 3DOF 的 psi_dot 坐标奇点处给一个角速度上限，避免 cos(gamma)->0 时数值爆掉。
PSI_DOT_LIMIT = math.radians(70.0)
N_MIN = -3.0                       # 战斗机负过载极限约 -3 g
MISSILE_SLEW_RATE = math.radians(90.0)  # 弹体轴对准速度矢量的最大角速度

# ---------- 跨声速/超声速波阻 ----------
# 阻力发散不是阶跃：onset 用 smoothstep 从 0.78 平滑开到 0.95
# （跨度太窄会让 dC_D0/dM 仍接近阶跃），高斯项给出 M1.05 附近的
# 跨声速鼓包，超声速段保留持续波阻。
DRAG_CRIT_MACH = 0.78
DRAG_DD_MACH = 0.95
DRAG_PEAK_MACH = 1.05
DRAG_WIDTH = 0.34
DRAG_BUMP = 0.62
DRAG_SUP_START_MACH = 0.95
DRAG_SUP_END_MACH = 1.35
DRAG_SUPERSONIC = 0.18

# 载机雷达对目标的照射/数据链锥：半主动弹全程需要，主动弹只在末段截获前需要。
SARH_ILLUM_FOV = math.radians(60.0)

# ---------- 发动机功率设置：军用推力 / 加力 ----------
# 涡扇有两个明显不同的功率区间：军用推力（不开加力）与加力推力。
# 单位耗油率按公开量级折算成 kg/(N·s)：军推约 0.67 lb/(lbf·h)，
# 加力约 1.97 lb/(lbf·h)，即加力的油耗约为军推的三倍。
TSFC_MIL = 1.90e-5   # kg/(N·s)，军推
TSFC_AB = 5.60e-5    # kg/(N·s)，加力

# ---------- 雷达隐身 ----------
# 雷达方程里探测距离与目标 RCS 的四次方根成正比。RCS_REF 取 1 m²，
# 所有机载雷达/导引头的标称距离都按这个基准值给出。
RCS_REF = 1.0


def rcs_scale(rcs: float) -> float:
    """目标 RCS 对雷达探测距离的缩放系数（相对 1 m²）。"""
    return (max(rcs, 0.001) / RCS_REF) ** 0.25

# 导弹的终端物理：动压不足时舵面已经压不出升力，导弹只能按弹道坠落。
MISSILE_MIN_Q = 1200.0          # Pa，低于该动压判定失去控制能力
MISSILE_FALL_SPEED = 120.0      # m/s，无控弹减速到该值近似为坠落
# 远程半主动弹的 loft 弹道：助推段先把“虚拟瞄准点”抬到目标上方，末段再压回来，
# 用高度换射程。直接加向上的加速度会被比例导引拉回视线，抬不起来，
# 所以这里改成抬高制导视线的仰角，让弹用自身可用过载飞出拱形弹道。
MISSILE_LOFT_ANGLE = math.radians(18.0)  # 抬升段的最大视线仰角偏置
MISSILE_LOFT_TIME = 10.0        # s，抬升段持续时间（前 60% 满偏置，之后线性收回）
MISSILE_LOFT_MIN_RANGE = 9000.0  # m，只有远距发射才抬升

# ---------- 中距弹发射决策的命中概率估计 ----------
# 不可逃逸区取有效雷达包线的该比例：再远，目标有足够时间掉头把弹拖进尾追，
# 弹的末端能量撑不到近炸半径。
NEZ_FRACTION = 0.55
# 目标已经在逃逸（速度背离本机）时，不可逃逸区按该系数进一步缩水。
NEZ_ASPECT_PENALTY = 0.45
# 概率曲线的过渡带宽度（相对 NEZ）：越小，越接近“进了 NEZ 才算高效”。
PK_TRANSITION = 0.22

# ---------- 机载干扰物（箔条 / 红外热诱弹） ----------
# 箔条是金属丝云，对雷达导引头形成强反射；热诱弹是高温燃烧源，骗红外导引头。
# 两者都由机载投放器自动/手动投放，离开载机后迅速减速、随风飘落。
CHAFF_LIFE = 4.0            # s，箔条云有效持续时间
FLARE_LIFE = 2.5            # s，热诱弹燃烧时间
CHAFF_SEDUCE_RANGE = 2000.0  # m，箔条对雷达导引头的有效诱骗距离
FLARE_SEDUCE_RANGE = 1800.0  # m，热诱弹对红外导引头的有效诱骗距离
CHAFF_SEDUCE_RATE = 0.35     # 1/s，箔条每秒诱骗成功的概率
FLARE_SEDUCE_RATE = 0.45     # 1/s，热诱弹每秒诱骗成功的概率
AUTO_CHAFF_RANGE = 5000.0    # m，雷达弹进入该距离自动投箔条
# 箔条寿命只有几秒、有效诱骗距离约 2 km，撒得太早会在弹飞到之前就飘散失效，
# 所以这个距离必须和 CHAFF_LIFE / CHAFF_SEDUCE_RANGE 匹配，而不是越远越好。
AUTO_FLARE_RANGE = 5000.0    # m，红外弹进入该距离自动投热诱弹
CM_DROP_CD = 1.5             # s，同一种干扰物的投放间隔
CM_DECOY_KEEP = 1.5          # s，诱骗成功后导弹继续追诱饵的时间
# 箔条只有和载机落在同一个多普勒通道里才骗得走雷达导引头：迎头/尾追时
# 载机径向速度很大，箔条一撒出去就和载机分开，所以必须配合横转/三九投放。
CHAFF_NOTCH_MIN = 0.55       # 横转程度（0=迎头/尾追，1=正侧方）低于该值不投放

# ---------- 近炸引信的杀伤概率 ----------
# 破片战斗部不是“进圈即必杀”：中心命中概率约 0.9，越靠边缘越低，
# 交会速度越高引信/战斗部的有效作用窗口越短，还要再打折。
PK_CENTER = 0.95
PK_DECAY = 2.2

# ---------- 标准大气 ISA（0–32 km） ----------
# 密度随高度衰减同时影响升力、阻力和推力，是高空性能与升限的来源。
ISA_T0 = 288.15        # 海平面温度 K
ISA_P0 = 101325.0      # 海平面气压 Pa
ISA_LAPSE = -0.0065    # 对流层温度梯度 K/m
ISA_GAS = 287.053      # 干空气气体常数 J/(kg*K)
ISA_H_TROP = 11000.0   # 对流层顶
ISA_H_STRAT = 20000.0  # 平流层下界
RHO0 = 1.225           # 海平面密度 kg/m3，用于推重比等归一化


def clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def proximity_pk(miss_dist: float, lethal_radius: float) -> float:
    """近炸引信触发后的杀伤概率（0–0.95）。

    以脱靶量相对战斗部半径归一化：正中心约 0.95，等于引信半径时约 0.10。
    破片在这个范围内快速稀疏，所以按平方指数衰减。
    """
    d = clamp(miss_dist / max(lethal_radius, 1e-3), 0.0, 4.0)
    return clamp(PK_CENTER * math.exp(-PK_DECAY * d * d), 0.0, 0.95)


def notch_quality(ac: "Aircraft", mx: float, my: float, mz: float) -> float:
    """载机相对来袭弹的横转程度：0=迎头/尾追，1=正侧方。

    箔条离机后迅速减速，只有载机几乎不带径向速度（三九/横转）时，
    箔条云与载机才处于同一多普勒通道，导引头才会被诱走。
    """
    dx, dy, dz = mx - ac.x, my - ac.y, mz - ac.z
    d = math.sqrt(dx * dx + dy * dy + dz * dz)
    vx, vy, vz = world_velocity(ac)
    radial = abs((vx * dx + vy * dy + vz * dz) / max(d, 1e-6))
    return clamp(1.0 - radial / max(ac.v, 1.0), 0.0, 1.0)


def air_density(z: float) -> float:
    """ISA 大气密度：0–11 km 线性降温，11–20 km 等温，20–32 km 缓慢升温。"""
    h = clamp(z, 0.0, 32000.0)
    exp = -G / (ISA_LAPSE * ISA_GAS)      # 5.2561
    if h <= ISA_H_TROP:
        t = ISA_T0 + ISA_LAPSE * h
        p = ISA_P0 * (t / ISA_T0) ** exp
    elif h <= ISA_H_STRAT:
        t = 216.65
        p_trop = ISA_P0 * (t / ISA_T0) ** exp
        p = p_trop * math.exp(-G * (h - ISA_H_TROP) / (ISA_GAS * t))
    else:
        t = 216.65 + 0.001 * (h - ISA_H_STRAT)
        p_strat = (
            ISA_P0
            * (216.65 / ISA_T0) ** exp
            * math.exp(-G * (ISA_H_STRAT - ISA_H_TROP) / (ISA_GAS * 216.65))
        )
        p = p_strat * (t / 216.65) ** (-G / (0.001 * ISA_GAS))
    return p / (ISA_GAS * t)


def isa_temperature(z: float) -> float:
    """ISA 温度 (K)：对流层线性降温，11 km 以上等温，20 km 以上缓慢升温。"""
    h = clamp(z, 0.0, 32000.0)
    if h <= ISA_H_TROP:
        return ISA_T0 + ISA_LAPSE * h
    if h <= ISA_H_STRAT:
        return 216.65
    return 216.65 + 0.001 * (h - ISA_H_STRAT)


def speed_of_sound(z: float) -> float:
    """当地声速 (m/s)。"""
    return math.sqrt(1.4 * ISA_GAS * isa_temperature(z))


def mach_number(v: float, z: float) -> float:
    return v / speed_of_sound(z)


def thrust_lapse(
    z: float, mach: float = 0.0,
    lapse_exp: float = 0.70, ram_r0: float = 0.30, ram_r1: float = 0.0,
    derate_z: float = 1.0e9, derate_H: float = 1.0, eta_install: float = 1.0,
) -> float:
    """涡扇推力：密度衰减 × 冲压恢复 × 高空降额 × 安装损失。

    旧模型只有 ``(rho/rho0)^0.7 × (1+0.3M)``：指数小于 1 时推力比阻力掉得慢，
    Ps 随高度不降反升，升限虚高到 25 km 以上。现在的三项各有物理含义：

    - ``sigma**lapse_exp``：密度衰减。指数按机型标定，越接近 1 掉得越快；
    - ``ram``：进气道冲压恢复，只在马赫数上增加，且高空（sigma 小）更强，
      并设 1.55 的上限，避免高马赫时推力无限增长；
    - ``derate``：高空推力悬崖。真实发动机在 12–18 km 之后推力迅速塌陷，
      这一项让 Ps 在升限附近自然穿过 0，而不是靠“30 km 解体”兜底；
    - ``eta_install``：进气道/喷口安装损失，典型 0.95–0.98。
    """
    sigma = clamp(air_density(z) / RHO0, 0.0, 1.0)
    ram = min(1.0 + (ram_r0 + ram_r1 * (1.0 - sigma)) * clamp(mach, 0.0, 2.5), 1.55)
    derate = 1.0 if z <= derate_z else math.exp(-(z - derate_z) / max(derate_H, 1.0))
    return sigma ** lapse_exp * ram * derate * eta_install


def drag_divergence(mach: float) -> float:
    """跨声速/超声速波阻增量（相对 cd0 的倍数）。

    真实跨声速阻力是连续但梯度很大的曲线，不是某个马赫数处的阶跃。
    这里用 C1 连续的 smoothstep 控制阻力发散起点，再叠加一个高斯鼓包，
    并在超声速段保留一个持续存在的波阻平台：

    - M < 0.78：几乎没有波阻增量；
    - 0.78–0.95：平滑进入阻力发散段（drag creep）；
    - 1.05 附近：跨声速阻力鼓包峰值；
    - 1.35 以上：保留约 0.18 的超声速波阻，不回落成亚声速水平。

    这样在 M=0.75 附近不会出现人为跳变，也避免“M=2 波阻消失”的问题。
    """
    onset = _smoothstep(DRAG_CRIT_MACH, DRAG_DD_MACH, mach)
    bump = DRAG_BUMP * onset * math.exp(
        -((mach - DRAG_PEAK_MACH) / DRAG_WIDTH) ** 2
    )
    supersonic = DRAG_SUPERSONIC * _smoothstep(
        DRAG_SUP_START_MACH, DRAG_SUP_END_MACH, mach
    )
    return bump + supersonic


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    """标准 C1 smoothstep；在两端函数值与一阶导数都连续。"""
    t = clamp((x - edge0) / max(edge1 - edge0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def cd0_at(mach: float, cd0: float) -> float:
    """计入波阻后的零升阻力系数。"""
    return cd0 * (1.0 + drag_divergence(mach))


def wrap_pi(angle: float) -> float:
    """把角度收敛到 (-pi, pi]。"""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


@dataclass(frozen=True)
class MissileSpec:
    """导弹总体参数（红外格斗弹与雷达中距弹共用）。

    脱靶首先由动力学与几何决定：导引头视场丢失、可用过载不足、能量耗尽
    都会导致脱靶；进入近炸半径后再按脱靶量与交会速度掷一次命中概率，
    避免“进圈即必杀”，也允许打成重伤而不是一击必杀。
    """

    name: str
    mass: float                  # 发射质量 kg
    length: float                # m
    diameter: float              # m
    max_g: float                 # 结构可用过载
    seeker_fov: float            # 发射前锁定半视场角 rad
    gimbal_fov: float | None     # 发射后导引头万向支架极限 rad（缺省=锁定视场）
    seeker_range: float          # 可锁定距离 m
    lock_time: float             # 稳定跟踪所需时间 s
    min_range: float             # 最小发射距离 m
    max_range: float             # 发射包线最大距离 m
    thrust: float                # N
    burn_time: float             # s
    burnout_mass: float          # 推进剂耗尽后质量 kg
    cd0: float                   # 零升阻力系数
    cn_max: float                # 舵面可用法向力系数
    warhead_damage: float        # 近炸中心命中伤害
    lethal_radius: float         # 近炸引信半径 m
    life: float = 15.0           # 最大飞行时间 s
    cooldown: float = 1.5        # 连射间隔 s
    pn_gain: float = 4.0         # 比例导引系数 N
    guidance: str = "IR"         # "IR" 红外 / "SARH" 半主动雷达 / "ARH" 主动雷达
    loft: bool = False           # 远程发射时先抬升弹道再俯冲（loft 弹道）
    # 主动雷达导引头截获距离 m（相对 1 m² 目标）。飞到该距离内即“开眼”（pitbull），
    # 之后不再需要载机照射/数据链，变成发射后不管。
    active_range: float = 0.0
    # 自动驾驶仪/舵机的一阶时间常数 s：横向过载指令不能瞬时建立。
    # 末段目标拉大过载时，这个滞后就是真实的脱靶来源。
    autopilot_tau: float = 0.18
    # 跨声速/超声速波阻增量（相对 cd0），按 smoothstep(0.85,1.15) 打开。
    # 导弹推重比高，峰值马赫几乎完全由这一项决定；旧模型没有它，
    # 中距弹会冲到 M5 以上，比真实弹道快出一大截。
    cd0_wave: float = 0.55

    @property
    def is_sarh(self) -> bool:
        return self.guidance.upper() == "SARH"

    @property
    def is_arh(self) -> bool:
        return self.guidance.upper() == "ARH"

    @property
    def is_radar(self) -> bool:
        return self.guidance.upper() in ("SARH", "ARH")

    @property
    def ref_area(self) -> float:
        return math.pi * (0.5 * self.diameter) ** 2

    @property
    def propellant(self) -> float:
        return max(self.mass - self.burnout_mass, 0.0)

    @property
    def mass_flow(self) -> float:
        return self.propellant / self.burn_time if self.burn_time > 0.0 else 0.0

    @property
    def gimbal(self) -> float:
        """发射后导引头能偏转到的最大离轴角。

        发射前用较窄的 seeker_fov 判定“能否锁定”，发射后用万向支架极限
        判定“是否丢锁”：弹体刚离架时往往还来不及指向目标。
        """
        return self.seeker_fov if self.gimbal_fov is None else self.gimbal_fov


@dataclass(frozen=True)
class AircraftSpec:
    """机体性能参数。改这里就能得到完全不同的最优战术。"""

    name: str
    mass: float                 # kg
    wing_area: float            # m^2
    cl_max: float               # 最大升力系数
    n_struct: float             # 结构过载极限 (g)
    thrust_max: float           # N
    cd0: float                  # 零升阻力系数
    aspect_ratio: float         # 展弦比
    oswald: float = 0.8
    hp_max: float = 100.0
    gun_range: float = 900.0    # m
    gun_cone: float = math.radians(8.0)
    gun_dps: float = 18.0       # 命中时的每秒伤害
    ammo_max: float = 300.0     # 发
    rounds_per_sec: float = 25.0
    turn_tau: float = 0.45      # 操纵响应时间常数 s
    # 前向雷达散射截面 m²（只影响雷达探测/锁定距离，不影响红外与机炮）。
    rcs: float = 1.0

    # ---- 发动机功率设置与燃油 ----
    thrust_mil: float = 0.0     # 军用推力 N；0 表示按 0.62×thrust_max 估算
    cruise_mach: float = 0.80   # 开局巡航马赫数（初始速度 = 当地声速 × 该值）
    fuel_capacity: float = 0.0  # 可消耗燃油 kg；0 表示不建模
    # ---- 推力高度/马赫衰减（由 scripts/calibrate_thrust.py 拟合公开锚点） ----
    lapse_exp: float = 0.70     # 密度衰减指数，越接近 1 高空掉推越快
    ram_r0: float = 0.30        # 海平面冲压恢复斜率
    ram_r1: float = 0.00        # 高空附加冲压（按 1-sigma 增强）
    derate_z: float = 1.0e9     # 推力悬崖起始高度 m
    derate_H: float = 1.0       # 悬崖衰减尺度 m
    eta_install: float = 1.00   # 进气道/喷口安装损失

    # ---- 机炮外弹道（P1-3） ----
    muzzle_speed: float = 900.0      # 弹丸出膛速度 m/s
    bullet_mass: float = 0.102       # 弹丸质量 kg（20 mm 约 0.102，30 mm 约 0.390）
    bullet_diameter: float = 0.020   # 弹丸直径 m
    gun_spread: float = 0.003        # 每发角度散布标准差 rad

    # ---- 6DOF 刚体参数（3DOF/2DOF 不使用） ----
    length: float = 15.0        # 机身长度 m，用于估算俯仰/偏航惯量
    cl_alpha: float = 3.6       # 升力线斜率 1/rad
    cl0: float = 0.12           # 零迎角升力系数
    alpha_stall: float = math.radians(22.0)
    cm_alpha: float = -1.0      # 纵向静稳定（负=稳定）
    cm_de: float = -1.2         # 升降舵效率（负=正舵下压机头）
    cm_q: float = -8.0          # 俯仰阻尼
    cl_p: float = -1.5          # 滚转阻尼
    cl_da: float = 0.12         # 副翼效率
    cl_beta: float = -0.10      # 上反效应
    cl_dr: float = 0.02
    cn_beta: float = 0.10       # 航向静稳定
    cn_r: float = -0.15         # 偏航阻尼
    cn_dr: float = 0.08         # 方向舵效率
    cn_da: float = 0.02
    cy_beta: float = -0.5       # 侧力

    # ---- 推力矢量喷口（仅 6DOF 使用） ----
    tvc: float = 0.0            # 最大偏转角 rad，0 = 无矢量喷口
    tvc_alpha_max: float = 0.0  # 矢量喷口能维持的过失速迎角 rad
    # 矢量喷流吹过机翼/尾翼带来的等效增升（CL 增量，按发动机功率占比缩放）。
    # 无矢量机型为 0；有矢量时失速迎角随功率右移，盘旋可用升力上限提高。
    tvc_cl_bonus: float = 0.0

    # ---- 近距格斗弹挂载（满载计入整机质量） ----
    missile: MissileSpec | None = None
    missile_count: int = 0
    # ---- 雷达制导中距弹挂载 ----
    radar_missile: MissileSpec | None = None
    radar_missile_count: int = 0

    # ---- 机载干扰物：箔条（对雷达）/ 红外热诱弹（对红外） ----
    chaff_count: int = 30
    flare_count: int = 30

    @property
    def weight(self) -> float:
        return self.mass * G

    @property
    def wing_loading(self) -> float:
        return self.weight / self.wing_area

    @property
    def span(self) -> float:
        return math.sqrt(self.aspect_ratio * self.wing_area)

    @property
    def chord(self) -> float:
        return self.wing_area / self.span

    # 转动惯量按质量与特征尺寸估算，量级与同级战斗机公开数据一致。
    # 质量随导弹发射实时变化，所以计算时允许传入当前质量。
    def ixx_of(self, mass: float) -> float:
        return 0.025 * mass * (self.span / 2.0) ** 2

    def iyy_of(self, mass: float) -> float:
        return 0.15 * mass * (self.length / 2.0) ** 2

    def izz_of(self, mass: float) -> float:
        return 0.20 * mass * (self.length / 2.0) ** 2

    @property
    def ixx(self) -> float:
        return self.ixx_of(self.mass)

    @property
    def iyy(self) -> float:
        return self.iyy_of(self.mass)

    @property
    def izz(self) -> float:
        return self.izz_of(self.mass)

    @property
    def v_stall(self) -> float:
        """海平面失速速度，仅用于展示；对局内请用 v_stall_at(z)。"""
        return self.v_stall_at(0.0)

    def v_stall_at(self, z: float, mass: float | None = None) -> float:
        """当地失速速度：密度越低，同样重量需要的速度越高。"""
        m = self.mass if mass is None else mass
        return math.sqrt(
            2.0 * m * G / (air_density(z) * self.wing_area * self.cl_max)
        )

    def lift_scale(self, mach: float) -> float:
        """超声速升力损失：M1.6 以上只保留 70% 的 CLmax。

        真实机翼在超声速段受激波/后掠影响升力曲线整体下移，可用过载因此
        随马赫数下降。跨声速段用 smoothstep 过渡，避免 M1 附近跳变。
        """
        return 1.0 - 0.30 * _smoothstep(0.90, 1.60, mach)

    def cl_max_at(self, mach: float) -> float:
        return self.cl_max * self.lift_scale(mach)

    def n_lift(self, v: float, z: float = 0.0, mass: float | None = None) -> float:
        """升力能提供的过载。低翼载 + 高 CLmax + 高密度在这里体现优势。"""
        if v <= 0.0:
            return 0.0
        m = self.mass if mass is None else mass
        cl = self.cl_max_at(mach_number(v, z))
        return 0.5 * air_density(z) * v * v * self.wing_area * cl / (m * G)

    def n_available(self, v: float, z: float = 0.0,
                    mass: float | None = None) -> float:
        return min(self.n_struct, self.n_lift(v, z, mass))

    def mil_thrust(self) -> float:
        """军用推力（不开加力）：缺省按最大加力的 62% 估算。"""
        return self.thrust_mil if self.thrust_mil > 0.0 else 0.62 * self.thrust_max

    def lapse(self, z: float, mach: float) -> float:
        """该机型的高度/马赫推力衰减系数。所有推力计算必须走这里。"""
        return thrust_lapse(
            z, mach, self.lapse_exp, self.ram_r0, self.ram_r1,
            self.derate_z, self.derate_H, self.eta_install,
        )

    def thrust_at(self, throttle: float, ab: float, z: float,
                  mach: float) -> float:
        """当前可用推力 (N)。

        throttle ∈ [0,1] 只在军推区间内调节；ab ∈ [0,1] 是加力燃烧室开度，
        额外叠加 (最大加力 - 军用推力)。两者都乘上高度/马赫数衰减。
        """
        mil = self.mil_thrust()
        dry = clamp(throttle, 0.0, 1.0) * mil
        wet = clamp(ab, 0.0, 1.0) * (self.thrust_max - mil)
        return (dry + wet) * self.lapse(z, mach)

    def n_sustain(self, v: float, z: float, throttle: float = 1.0,
                  mass: float | None = None, ab: float = 0.0) -> float:
        """推力刚好抵消阻力时的过载：该高度/速度能长期维持的最大 g。"""
        q = 0.5 * air_density(z) * v * v
        if q <= 0.0 or v <= 0.0:
            return 1.0
        m = self.mass if mass is None else mass
        surplus = (
            self.thrust_at(throttle, ab, z, mach_number(v, z))
            - q * cd0_at(mach_number(v, z), self.cd0) * self.wing_area
        )
        k = 1.0 / (math.pi * self.aspect_ratio * self.oswald * self.wing_area)
        n2 = surplus * q / ((m * G) * (m * G) * k)
        return clamp(math.sqrt(max(n2, 0.0)), 1.0, self.n_struct)

    def specific_power(self, v: float, z: float, n: float = 1.0,
                       throttle: float = 1.0,
                       mass: float | None = None,
                       ab: float = 0.0) -> float:
        """比剩余功率 Ps (m/s)：可用于爬升/加速的能量率。Ps<0 表示维持不住。"""
        q = 0.5 * air_density(z) * v * v
        if q <= 0.0 or v <= 0.0:
            return 0.0
        m = self.mass if mass is None else mass
        w = m * G
        mach = mach_number(v, z)
        thrust = self.thrust_at(throttle, ab, z, mach)
        drag0 = q * cd0_at(mach, self.cd0) * self.wing_area
        drag_i = n * n * w * w / (q * math.pi * self.aspect_ratio * self.oswald * self.wing_area)
        return (thrust - drag0 - drag_i) * v / w

    def turn_rate_max(self, v: float, z: float = 0.0,
                      mass: float | None = None) -> float:
        n = self.n_available(v, z, mass)
        if v <= 0.0 or n <= 1.0:
            return 0.0
        return G * math.sqrt(n * n - 1.0) / v


@dataclass
class Action:
    """Agent 的输出，3DOF 的原始控制量。

    bank 是滚转角比例 [-1, 1]，映射到 ±MAX_BANK；
    n 是期望过载 (g)，World 会按结构/升力限制裁剪。
    """

    bank: float = 0.0
    n: float = 1.0
    throttle: float = 1.0
    afterburner: float = 0.0      # 加力开度 0..1，与 throttle 独立
    fire: bool = False
    launch: bool = False          # 本决策周期请求发射一枚格斗弹
    launch_radar: bool = False    # 本决策周期请求发射一枚雷达制导中距弹
    post_stall: bool = False      # 过失速指令：把机头拉到矢量喷口可维持的超大迎角


@dataclass
class Observation:
    """给 Agent 的观测，不含任何未来状态。"""

    # 相对几何
    dist: float
    dist_h: float
    dz: float                    # 对方高度 - 本机高度
    los_az: float                # 视线水平方位（世界系）
    los_el: float                # 视线仰角
    off_az: float                # 视线相对机头的水平偏差
    off_el: float                # 视线相对航迹的俯仰偏差
    angle_off_nose: float        # 三维机头指向误差
    aspect: float                # 对方机尾与视线的夹角：0=我咬尾，180=对头
    foe_off_nose: float          # 对方机头指向我的角度
    closure: float               # 接近率，正为接近
    in_gun_range: bool
    in_gun_cone: bool

    # 本机状态
    own_speed: float
    own_speed_frac: float        # 当前速度 / 失速速度
    own_alt: float
    own_psi: float
    own_gamma: float
    own_n_avail: float
    own_n: float                 # 当前实际过载
    own_n_sustain: float         # 该高度/速度能长期维持的过载
    own_ps: float                # 当前过载下的比剩余功率 (m/s)
    own_ps_level: float          # 1g 平飞时的比剩余功率 (m/s)，<0 即超过能量升限
    own_hp: float
    own_hp_frac: float
    own_ammo: float

    # 对方状态
    foe_speed: float
    foe_alt: float
    foe_psi: float
    foe_gamma: float
    foe_hp: float
    foe_hp_frac: float

    # 能量与空间
    energy_diff: float           # 本方比能量 - 对方比能量
    edge_margin: float           # 到水平空域边界的最小距离
    z_margin_low: float          # 距地面的余量
    z_margin_high: float         # 距高度上限的余量
    center_az: float             # 指向空域中心的水平方位

    # 机头姿态：6DOF 下炮口方向与航迹方向存在迎角/侧滑偏置
    own_nose_az: float = 0.0
    own_nose_el: float = 0.0
    own_bank: float = 0.0            # 当前滚转角 rad，用于限制规避动作的爬升
    dof: int = 3                     # 当前内核自由度，过失速只在 6DOF 生效
    own_tvc: bool = False            # 本机是否带推力矢量喷口
    own_fuel_frac: float = 1.0       # 剩余燃油比例
    own_ab: bool = False             # 当前是否在开加力
    own_thrust_frac: float = 0.0     # 当前推力 / 最大加力推力
    own_cruise_throttle: float = 0.8  # 维持当前速度所需的军推油门（巡航功率）

    # 近距格斗弹：锁定需要目标持续留在导引头视场内累计时间
    own_missiles: int = 0
    own_missiles_in_flight: int = 0   # 仅统计近距红外弹，中距弹另见 own_radar_in_flight
    own_locked: bool = False
    own_lock_progress: float = 0.0   # 0..1，达到 1 即完成锁定
    in_missile_envelope: bool = False
    missile_min_range: float = 0.0
    missile_max_range: float = 0.0
    foe_missiles: int = 0
    foe_locked: bool = False            # 对方是否已经锁定本机
    foe_lock_progress: float = 0.0      # 对方锁定进度 0..1
    incoming_missile_dist: float = 1e9   # 最近一枚向我发射的在飞弹距离
    incoming_missile_az: float = 0.0     # 该弹相对机头的水平方位
    incoming_missile_ttg: float = 1e9    # 该弹预计命中时间 s（按接近率估算）
    incoming_missile_sarh: bool = False  # 该弹是否为半主动雷达制导（需要载机持续照射）
    incoming_missile_radar: bool = False # 该弹是否为雷达制导（含主动弹，告警距离更远）

    # 雷达制导中距弹：发射前需要雷达稳定跟踪；主动弹在中制导段仍需载机
    # 提供目标信息（数据链），末段自主截获后才完全脱离。
    own_radar_missiles: int = 0
    own_radar_in_flight: int = 0
    own_radar_support: bool = False       # 仍有在飞雷达弹需要载机照射/数据链
    own_radar_ttg: float = 1e9            # 我方在飞中距弹预计命中对方的时间 s
    own_radar_locked: bool = False
    own_radar_lock_progress: float = 0.0
    in_radar_envelope: bool = False
    radar_tracking: bool = False          # 目标仍在雷达照射范围内
    radar_min_range: float = 0.0
    radar_max_range: float = 0.0
    foe_radar_missiles: int = 0
    foe_radar_locked: bool = False
    # 中距弹发射决策：现在发射的估计命中概率，以及目标是否已在不可逃逸区。
    # pk 来自“目标按最优逃逸”的弹道前推，NEZ 表示那样也躲不掉。
    radar_pk_est: float = 0.0
    radar_nez: bool = False


@dataclass
class Aircraft:
    spec: AircraftSpec
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    psi: float = 0.0
    gamma: float = 0.0
    v: float = 200.0
    hp: float = 100.0
    ammo: float = 300.0
    alive: bool = True
    n: float = 1.0
    bank: float = 0.0
    throttle: float = 1.0
    ab: float = 0.0               # 当前加力开度 0..1
    fuel: float = 0.0             # 剩余燃油 kg
    thrust: float = 0.0           # 当前总推力 N（供 HUD / 尾焰读数）
    firing: bool = False          # 本物理步是否在开火，供渲染层画炮口特效

    # ---- 近距格斗弹状态 ----
    missiles: int = 0             # 剩余挂弹数
    mass: float = 0.0             # 含剩余挂弹的当前质量；0 表示沿用 spec.mass
    locked: bool = False
    lock_timer: float = 0.0
    launch_cd: float = 0.0

    # ---- 雷达制导中距弹状态 ----
    radar_missiles: int = 0
    radar_locked: bool = False
    radar_lock_timer: float = 0.0
    radar_cd: float = 0.0

    # ---- 干扰物余量与投放冷却 ----
    chaff: int = 0
    flares: int = 0
    chaff_cd: float = 0.0
    flare_cd: float = 0.0

    # ---- 6DOF 刚体状态 ----
    # 机体系：x 前、y 右、z 下（右手系）；phi 右滚为正、theta 抬头为正。
    rigid: bool = False          # 是否由刚体方程驱动
    u: float = 200.0
    vb: float = 0.0
    wb: float = 0.0
    p: float = 0.0
    q: float = 0.0
    r: float = 0.0
    phi: float = 0.0
    theta: float = 0.0
    # 姿态四元数 (w,x,y,z)，机体系→世界系。数值积分只更新四元数，
    # phi/theta/psi 是每步由它反解的派生量（见 euler_from_quat）。
    quat: tuple[float, float, float, float] = (0.0, 1.0, 0.0, 0.0)
    # 本物理步矢量喷口的实际偏转指令（rad，正=抬头方向），供观战与诊断使用。
    tvc_cmd: float = 0.0
    # 机炮待发弹丸的累计余量：射速高于物理步频时，按帧补发而不是每帧固定一发。
    gun_pending: float = 0.0
    # 越过验证高度上限只记录一次事件，不再判定“解体”。
    ceiling_warned: bool = False

    @property
    def energy(self) -> float:
        """比能量 (J/kg)：高度与速度的统一记账单位。"""
        return G * self.z + 0.5 * self.v * self.v

    @property
    def weight(self) -> float:
        """当前全机重量，随格斗弹发射实时下降。"""
        return (self.mass if self.mass > 0.0 else self.spec.mass) * G

    @property
    def velocity(self) -> tuple[float, float, float]:
        return world_velocity(self)


@dataclass
class Missile:
    """在飞格斗弹：点质量 + 比例导引 + 有限舵面过载 + 近炸引信。

    速度矢量直接在世界系积分，机体轴近似取速度方向。
    """

    spec: MissileSpec
    owner: int
    target: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    mass: float
    fuel: float
    t: float = 0.0
    alive: bool = True
    guided: bool = True          # 导引头是否仍咬住目标
    active: bool = False         # 主动雷达弹是否已进入末段自主截获（pitbull）
    # 弹体轴：发射瞬间沿挂架/机头，之后以有限角速度对准速度矢量。
    # 导引头视场是相对弹体轴的，不能直接用速度方向，否则大迎角发射会瞬间丢锁。
    nx: float = 0.0
    ny: float = 0.0
    nz: float = 1.0
    # 被干扰物诱骗：decoy 是诱饵点（箔条云/热诱弹），decoy_t 归零后导弹转无控。
    decoy: tuple[float, float, float] | None = None
    decoy_t: float = 0.0
    # 自动驾驶仪状态：当前实际建立的横向加速度 N，按一阶滞后跟踪制导指令。
    gx: float = 0.0
    gy: float = 0.0
    gz: float = 0.0
    # 近炸命中的随机源：按发射时刻与发射方定种，保证实验可复现。
    roll_seed: int = 0

    @property
    def speed(self) -> float:
        return math.sqrt(self.vx * self.vx + self.vy * self.vy + self.vz * self.vz)

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class Countermeasure:
    """一枚已投放的干扰物（箔条云或热诱弹）。

    离开载机后按指数减速并受重力下落，寿命结束后失效；导弹可以锁定它，
    从而把原本指向载机的制导回路骗到诱饵上。
    """

    kind: str                    # "chaff" 箔条 / "flare" 热诱弹
    owner: int
    x: float
    y: float
    z: float
    vx: float = 0.0
    vy: float = 0.0
    vz: float = 0.0
    t: float = 0.0
    life: float = CHAFF_LIFE
    alive: bool = True
    # 投放时的横转程度（0=迎头/尾追，1=正侧方）：箔条只有落在载机同一
    # 多普勒通道里才骗得走导引头，所以这个值直接缩放诱骗概率。
    quality: float = 1.0

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class Bullet:
    """一发机炮外弹道弹丸。

    旧实现是“炮口在视锥内就按秒伤扣血”，命中与飞行时间无关，也没有
    提前量/下坠。现在弹丸按真实初速（载机速度 + 炮口速度）、高斯散布、
    二次阻力与重力飞行，命中用线段-球判定，单发伤害 = gun_dps/射速。
    """

    owner: int
    x: float
    y: float
    z: float
    vx: float
    vy: float
    vz: float
    mass: float
    drag_k: float        # ρ·C_D·A/(2m)，二次阻力系数 1/m
    damage: float        # 单发命中伤害
    t: float = 0.0
    life: float = 2.5
    alive: bool = True

    @property
    def position(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


def body_axes(psi: float, theta: float, phi: float):
    """机体轴（前/右/下）在世界系（东/北/上）中的方向余弦。"""
    sp, cp = math.sin(psi), math.cos(psi)
    st, ct = math.sin(theta), math.cos(theta)
    sf, cf = math.sin(phi), math.cos(phi)
    xb = (ct * cp, ct * sp, st)
    yb = (
        cf * sp + sf * st * cp,
        -cf * cp + sf * st * sp,
        -sf * ct,
    )
    zb = (
        -sf * sp + cf * st * cp,
        sf * cp + cf * st * sp,
        -cf * ct,
    )
    return xb, yb, zb


# ---------------------------------------------------------------------------
# 四元数姿态（6DOF）
# ---------------------------------------------------------------------------
# 欧拉角在 theta=±90° 处有万向节锁，且旧实现直接把 theta 夹在 ±80°，
# 筋斗/半滚倒转/垂直剪刀因此全部退化为“爬到 80° 再掉回来”。现在姿态用
# 四元数积分，欧拉角只作为派生量输出给观测、武器与渲染，物理上不再夹断。
#
# 本项目的 psi 从 x 轴逆时针计量，机体系 z 轴朝下，因此零姿态对应的旋转
# 是绕机体 x 轴 180°（R = Rx(pi)），欧拉合成顺序为 Rz(psi)·Ry(-theta)·Rx(pi+phi)。
# ---------------------------------------------------------------------------

def _quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def _quat_axis(axis: str, angle: float):
    half = 0.5 * angle
    c, s = math.cos(half), math.sin(half)
    if axis == "x":
        return (c, s, 0.0, 0.0)
    if axis == "y":
        return (c, 0.0, s, 0.0)
    return (c, 0.0, 0.0, s)


def quat_from_euler(phi: float, theta: float, psi: float):
    """按本项目欧拉约定合成四元数（机体系→世界系）。"""
    q = _quat_mul(_quat_axis("z", psi), _quat_axis("y", -theta))
    return _quat_mul(q, _quat_axis("x", math.pi + phi))


def quat_to_matrix(q) -> tuple[tuple[float, float, float], ...]:
    w, x, y, z = q
    return (
        (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
        (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
        (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
    )


def euler_from_quat(q) -> tuple[float, float, float]:
    """从四元数反解 (phi, theta, psi)。

    theta 由 asin 给出，天然落在 [-90°, 90°]：过顶后 psi/phi 会接管方向，
    机体轴仍连续。调用方应把 theta 当作显示量，而不是积分状态。
    """
    m = quat_to_matrix(q)
    theta = math.asin(clamp(m[2][0], -1.0, 1.0))
    ct = math.cos(theta)
    if abs(ct) < 1e-6:
        # 万向节锁：任选 psi，保证 phi 仍能把机头/机腹方向还原。
        psi = math.atan2(m[0][1], m[1][1])
        phi = math.atan2(-m[2][1], -m[2][2]) if abs(m[2][1]) > 1e-9 else 0.0
    else:
        psi = math.atan2(m[1][0], m[0][0])
        phi = math.atan2(-m[2][1], -m[2][2])
    return wrap_pi(phi), theta, wrap_pi(psi)


def world_velocity(ac: Aircraft) -> tuple[float, float, float]:
    """世界系速度。3DOF/2DOF 由 (v, gamma, psi) 给出，6DOF 由机体速度给出。"""
    if ac.rigid:
        xb, yb, zb = body_axes(ac.psi, ac.theta, ac.phi)
        return (
            xb[0] * ac.u + yb[0] * ac.vb + zb[0] * ac.wb,
            xb[1] * ac.u + yb[1] * ac.vb + zb[1] * ac.wb,
            xb[2] * ac.u + yb[2] * ac.vb + zb[2] * ac.wb,
        )
    cg = math.cos(ac.gamma)
    return (
        ac.v * cg * math.cos(ac.psi),
        ac.v * cg * math.sin(ac.psi),
        ac.v * math.sin(ac.gamma),
    )


def nose_direction(ac: Aircraft) -> tuple[float, float, float]:
    """机头指向的单位向量。6DOF 用机体 x 轴（真实炮口方向），点质量用速度方向。"""
    if ac.rigid:
        return body_axes(ac.psi, ac.theta, ac.phi)[0]
    vx, vy, vz = world_velocity(ac)
    n = max(math.sqrt(vx * vx + vy * vy + vz * vz), 1e-6)
    return (vx / n, vy / n, vz / n)


def gun_direction(ac: Aircraft) -> tuple[float, float, float]:
    """炮线方向：弹丸速度 = 载机速度 + 炮口初速沿机头。

    点质量模型没有机体姿态，机头即速度方向，所以结果与 nose_direction 一致。
    初速取机型参数（20 mm 约 1030 m/s、30 mm 约 900 m/s），不再是全局常数。
    """
    nx, ny, nz = nose_direction(ac)
    if not ac.rigid:
        return (nx, ny, nz)
    vx, vy, vz = world_velocity(ac)
    vm = ac.spec.muzzle_speed
    gx = vx + vm * nx
    gy = vy + vm * ny
    gz = vz + vm * nz
    g = max(math.sqrt(gx * gx + gy * gy + gz * gz), 1e-6)
    return (gx / g, gy / g, gz / g)


def _trim_rigid(spec: AircraftSpec, z0: float, v0: float,
                mass: float | None = None) -> tuple[float, float]:
    """按 1g 配平迎角初始化机体速度，避免 6DOF 开局掉高度或抬头。"""
    m = spec.mass if mass is None else mass
    qbar = 0.5 * air_density(z0) * v0 * v0
    cl_need = m * G / max(qbar * spec.wing_area, 1e-6)
    alpha = clamp(
        (cl_need - spec.cl0) / spec.cl_alpha, -spec.alpha_stall, spec.alpha_stall
    )
    return v0 * math.cos(alpha), v0 * math.sin(alpha)


def _cruise_throttle(spec: AircraftSpec, z: float, v: float) -> float:
    """维持巡航速度所需的军用油门：推力刚好抵消该速度下的零升阻力。"""
    mach = mach_number(v, z)
    q = 0.5 * air_density(z) * v * v
    drag = q * cd0_at(mach, spec.cd0) * spec.wing_area
    mil = spec.mil_thrust() * spec.lapse(z, mach)
    return clamp(drag / max(mil, 1e-6), 0.25, 0.95)


def launch_envelope(ac: "Aircraft", mspec: MissileSpec,
                    foe: "Aircraft") -> tuple[float, float]:
    """统一的发射包线 (min_range, max_range)。

    红外导引头只看目标的热信号，与雷达散射截面无关；雷达导引头按雷达方程
    随 RCS 的四次方根缩放。此前观测、锁定、发射三处各写了一遍，出现
    “观测说能打、锁定却打不出”的不一致，这里收成唯一入口。
    """
    reach = mspec.seeker_range
    if mspec.is_radar:
        reach *= rcs_scale(foe.spec.rcs)
    return mspec.min_range, min(mspec.max_range, reach)


class World:
    """固定步长、确定性的三维对局环境。"""

    def __init__(
        self,
        specs: list[AircraftSpec],
        dt: float = 1.0 / 120.0,
        arena: float = 8000.0,
        time_limit: float = 90.0,
        z0: float = 1500.0,
        z_floor: float = 0.0,
        z_ceiling: float = 6000.0,
        dof: int = 6,
        y_offset: float = 100.0,
        x_offset: float = 1500.0,
    ) -> None:
        if dof not in (2, 3, 6):
            raise ValueError("dof 只支持 2、3 或 6")
        self.dof = dof
        self.dt = dt
        self.arena = arena
        self.time_limit = time_limit
        self.z0 = z0
        self.z_floor = z_floor
        self.z_ceiling = z_ceiling
        self.t = 0.0
        self.result: str | None = None   # "red" / "blue" / "draw"
        self.reason: str = ""
        self.reason_code: str = ""       # kill / boundary / timeout / draw
        # 一方被击落后若还有导弹在飞，先把判负结果挂起，等弹落地/命中再结算。
        self._pending_finish: tuple[int, str, str] | None = None
        self.events: list[tuple[float, str]] = []
        self.stats = {
            "min_v": math.inf,
            "max_v": 0.0,
            "max_n": 0.0,
            "min_z": math.inf,
            "max_z": 0.0,
            "max_climb": 0.0,
            "rounds_fired": 0.0,
            "bullets_fired": 0,
            "bullet_hits": 0,
            "damage_dealt": [0.0, 0.0],
            "missiles_fired": 0.0,
            "missile_hits": 0,
            "missiles_launched": [0, 0],
            "missiles_hit": [0, 0],
        }
        # 在飞格斗弹；命中/寿命结束后 alive=False，保留在身上供渲染层读数。
        self.missiles: list[Missile] = []
        # 在飞机炮弹丸：按真实初速/散布/阻力飞行，线段-球判定命中。
        self.bullets: list[Bullet] = []
        # 已投放的干扰物（箔条云 / 热诱弹），失效后保留到清理，供渲染层读数。
        self.counters: list[Countermeasure] = []
        # 干扰诱骗用的随机源固定种子，保证回放与实验可复现。
        self._rng = random.Random(20261027)

        # 蓝机在 y 方向带一点初始偏移，避免完全镜像导致的确定性对称僵局。
        # 2DOF 只是缺少垂直自由度，初始高度仍取 z0。
        # 6DOF 额外按各自的 1g 配平迎角初始化机体速度与俯仰角。
        start_z = z0
        # 开局速度按各自的巡航马赫数换算成当地真空速，不再统一写死 200 m/s。
        sonic = speed_of_sound(start_z)
        v0s = [sonic * spec.cruise_mach for spec in specs]
        rigid = dof == 6
        specs_trim = [
            _trim_rigid(spec, start_z, v0s[i], spec.mass) if rigid else (v0s[i], 0.0)
            for i, spec in enumerate(specs)
        ]
        self.acs = [
            Aircraft(specs[0], x=-x_offset, y=0.0, z=start_z, psi=0.0, v=v0s[0],
                     hp=specs[0].hp_max, ammo=specs[0].ammo_max,
                     mass=specs[0].mass, missiles=specs[0].missile_count,
                     radar_missiles=specs[0].radar_missile_count,
                     chaff=specs[0].chaff_count, flares=specs[0].flare_count,
                     fuel=specs[0].fuel_capacity,
                     throttle=_cruise_throttle(specs[0], start_z, v0s[0]),
                     rigid=rigid, u=specs_trim[0][0], wb=specs_trim[0][1],
                     theta=math.atan2(specs_trim[0][1], specs_trim[0][0])),
            Aircraft(specs[1], x=x_offset, y=y_offset, z=start_z, psi=math.pi, v=v0s[1],
                     hp=specs[1].hp_max, ammo=specs[1].ammo_max,
                     mass=specs[1].mass, missiles=specs[1].missile_count,
                     radar_missiles=specs[1].radar_missile_count,
                     chaff=specs[1].chaff_count, flares=specs[1].flare_count,
                     fuel=specs[1].fuel_capacity,
                     throttle=_cruise_throttle(specs[1], start_z, v0s[1]),
                     rigid=rigid, u=specs_trim[1][0], wb=specs_trim[1][1],
                     theta=math.atan2(specs_trim[1][1], specs_trim[1][0])),
        ]
        self.stats["min_z"] = start_z
        self.stats["max_z"] = start_z
        # 6DOF 的姿态状态只有四元数：用初始欧拉角（含配平迎角）合成一次。
        for ac in self.acs:
            if ac.rigid:
                ac.quat = quat_from_euler(ac.phi, ac.theta, ac.psi)

    # ---------- 对外接口 ----------

    # ---------- 中距弹发射决策：命中概率 / 不可逃逸区估计 ----------

    def _intercept_estimate(self, i: int) -> tuple[float, bool]:
        """估计现在发射中距弹的命中概率，以及目标是否已在不可逃逸区。

        只做决策用的量级估计，不逐拍复刻弹道（理想逃逸下的弹道前推对末段
        十几米的脱靶量过于敏感，反而给不出单调可用的概率）：
        - 有效包线取载机雷达跟踪距离（按目标 RCS 缩放）与弹最大射程的较小值；
        - 不可逃逸区取包线的 NEZ_FRACTION。再远，目标有足够时间掉头把弹拖进
          尾追，弹的末端能量保不住，进不了近炸半径；
        - 目标已经在背离逃逸时，上述距离同步缩水；
        - 命中概率是以 NEZ 为中心的 logistic 曲线：进 NEZ 迅速逼近 1，
          远于 NEZ 时快速衰减。
        """
        me, foe = self.acs[i], self.acs[1 - i]
        spec = me.spec.radar_missile
        if spec is None or me.radar_missiles <= 0 or not foe.alive:
            return 0.0, False
        dx, dy, dz = foe.x - me.x, foe.y - me.y, foe.z - me.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 1e-6:
            return 0.95, True
        losx, losy, losz = dx / dist, dy / dist, dz / dist

        envelope = min(spec.max_range, spec.seeker_range * rcs_scale(foe.spec.rcs))
        # 目标速度沿视线的分量：正数表示正在背离本机逃逸。
        fvx, fvy, fvz = world_velocity(foe)
        fleeing = clamp(
            (fvx * losx + fvy * losy + fvz * losz) / max(foe.v, 1.0), -1.0, 1.0
        )
        aspect = 0.5 * (1.0 + fleeing)     # 0=迎头，1=完全尾追逃逸
        nez = NEZ_FRACTION * envelope * (1.0 - NEZ_ASPECT_PENALTY * aspect)
        nez = max(nez, spec.min_range)

        pk = 1.0 / (1.0 + math.exp((dist - nez) / max(PK_TRANSITION * nez, 1.0)))
        return clamp(pk, 0.0, 0.95), dist <= nez

    def observe(self, i: int) -> Observation:
        me = self.acs[i]
        foe = self.acs[1 - i]
        dx, dy, dz = foe.x - me.x, foe.y - me.y, foe.z - me.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        dist_h = math.hypot(dx, dy)
        if dist < 1e-6:
            dist = 1e-6
        ux, uy, uz = dx / dist, dy / dist, dz / dist

        los_az = math.atan2(dy, dx) if dist_h > 1e-6 else me.psi
        los_el = math.atan2(dz, dist_h)

        mvx, mvy, mvz = me.velocity
        fvx, fvy, fvz = foe.velocity
        gnx, gny, gnz = gun_direction(me)
        fnx, fny, fnz = nose_direction(foe)

        angle_off_nose = math.acos(clamp(ux * gnx + uy * gny + uz * gnz, -1.0, 1.0))
        # 瞄准误差按炮线（弹丸出膛方向）算：6DOF 下它与航迹、机头都不重合。
        nose_az = math.atan2(gny, gnx) if me.rigid else me.psi
        nose_el = math.asin(clamp(gnz, -1.0, 1.0)) if me.rigid else me.gamma
        # 经典 gross aspect：0 = 对方背对我（我咬尾），180 = 对头。
        aspect = math.acos(clamp(ux * fnx + uy * fny + uz * fnz, -1.0, 1.0))
        foe_off = math.acos(clamp(-(ux * fnx + uy * fny + uz * fnz), -1.0, 1.0))

        closure = -((fvx - mvx) * ux + (fvy - mvy) * uy + (fvz - mvz) * uz)
        in_range = dist <= me.spec.gun_range
        in_cone = in_range and angle_off_nose <= me.spec.gun_cone

        # 导引头包线：弹轴（机头）视场 + 锁定距离，与机炮炮线分开计算。
        mspec = me.spec.missile
        msl_max = msl_min = 0.0
        in_msl = False
        lock_progress = 0.0
        if mspec is not None and me.missiles > 0:
            msl_min, msl_max = launch_envelope(me, mspec, foe)
            nnx, nny, nnz = nose_direction(me)
            seeker_ang = math.acos(
                clamp(ux * nnx + uy * nny + uz * nnz, -1.0, 1.0)
            )
            in_msl = (
                msl_min <= dist <= msl_max and seeker_ang <= mspec.seeker_fov
            )
            lock_progress = clamp(me.lock_timer / max(mspec.lock_time, 1e-6), 0.0, 1.0)

        # 红外弹与中距弹是两条独立的发射通道：中距弹在飞时仍可补射近距弹，
        # 因此 in_flight 只统计红外弹，避免一条中距弹把整个发射架锁死。
        in_flight = sum(
            1 for m in self.missiles
            if m.alive and m.owner == i and not m.spec.is_radar
        )
        radar_in_flight = sum(
            1 for m in self.missiles
            if m.alive and m.owner == i and m.spec.is_radar
        )
        # 主动弹打完末段自主截获后，载机不必再保持照射/数据链。
        radar_support = any(
            m.alive and m.owner == i and m.spec.is_radar and not m.active
            for m in self.missiles
        )

        # 雷达包线：中距弹发射前需要雷达稳定跟踪，照射锥比导引头宽得多。
        # 弹已经打完时仍要给出照射状态，供战术层判断“要不要保持机头指向”。
        rspec = me.spec.radar_missile
        radar_max = radar_min = 0.0
        in_radar = False
        radar_progress = 0.0
        radar_tracking = False
        if rspec is not None and (me.radar_missiles > 0 or radar_in_flight > 0):
            # 隐身机（RCS 小）被雷达截获的距离按四次方根缩短。
            radar_min, radar_max = launch_envelope(me, rspec, foe)
            nnx, nny, nnz = nose_direction(me)
            radar_ang = math.acos(
                clamp(ux * nnx + uy * nny + uz * nnz, -1.0, 1.0)
            )
            in_radar = (
                radar_min <= dist <= radar_max and radar_ang <= rspec.seeker_fov
            )
            radar_tracking = (
                dist <= radar_max and radar_ang <= SARH_ILLUM_FOV
            )
            radar_progress = clamp(
                me.radar_lock_timer / max(rspec.lock_time, 1e-6), 0.0, 1.0
            )

        # 现在发射值不值：前推一次弹道，得到估计命中概率与 NEZ 判定。
        radar_pk, radar_nez = 0.0, False
        if rspec is not None and me.radar_missiles > 0 and foe.alive:
            radar_pk, radar_nez = self._intercept_estimate(i)

        # 导弹告警：只统计仍在制导、指向本机的弹。
        incoming_dist = 1e9
        incoming_az = 0.0
        incoming_ttg = 1e9
        incoming_sarh = False
        incoming_radar = False
        mevx, mevy, mevz = me.velocity
        for m in self.missiles:
            if not m.alive or m.target != i or not m.guided:
                continue
            ddx, ddy = m.x - me.x, m.y - me.y
            dd = math.sqrt(ddx * ddx + ddy * ddy + (m.z - me.z) ** 2)
            if dd < incoming_dist:
                incoming_dist = dd
                incoming_az = wrap_pi(math.atan2(ddy, ddx) - nose_az)
                incoming_sarh = m.spec.is_sarh
                incoming_radar = m.spec.is_radar
                # 弹与本机的接近率：用来估算剩余命中时间，供 F-pole 决策。
                vc = (
                    (mevx - m.vx) * ddx + (mevy - m.vy) * ddy
                    + (mevz - m.vz) * (m.z - me.z)
                ) / max(dd, 1e-6)
                incoming_ttg = dd / max(vc, 50.0)

        # 我方在飞中距弹的剩余命中时间：F-pole 的判据是“我的弹先到”。
        own_radar_ttg = 1e9
        fvx, fvy, fvz = foe.velocity
        for m in self.missiles:
            if not m.alive or m.owner != i or not m.spec.is_radar:
                continue
            rx, ry, rz = foe.x - m.x, foe.y - m.y, foe.z - m.z
            dd = math.sqrt(rx * rx + ry * ry + rz * rz)
            # 弹向目标飞：接近率取 (弹速 - 目标速度) 在视线方向的投影。
            vc = (
                (m.vx - fvx) * rx + (m.vy - fvy) * ry + (m.vz - fvz) * rz
            ) / max(dd, 1e-6)
            own_radar_ttg = min(own_radar_ttg, dd / max(vc, 50.0))

        half = self.arena / 2.0
        foe_lock_progress = 0.0
        if foe.spec.missile is not None and foe.missiles > 0:
            foe_lock_progress = clamp(
                foe.lock_timer / max(foe.spec.missile.lock_time, 1e-6), 0.0, 1.0
            )
        foe_radar_locked = bool(
            foe.spec.radar_missile is not None and foe.radar_missiles > 0
            and foe.radar_locked
        )
        if self.dof >= 3:
            z_low = me.z - self.z_floor
            z_high = self.z_ceiling - me.z
        else:
            # 2DOF 没有高度约束，给一个大余量让 Agent 的高度逻辑自然失效
            z_low = z_high = 1e9
        return Observation(
            dist=dist,
            dist_h=dist_h,
            dz=dz,
            los_az=los_az,
            los_el=los_el,
            off_az=wrap_pi(los_az - nose_az),
            off_el=los_el - nose_el,
            angle_off_nose=angle_off_nose,
            aspect=aspect,
            foe_off_nose=foe_off,
            closure=closure,
            in_gun_range=in_range,
            in_gun_cone=in_cone,
            own_speed=me.v,
            own_speed_frac=me.v / me.spec.v_stall_at(me.z, me.mass),
            own_alt=me.z,
            own_psi=me.psi,
            own_gamma=me.gamma,
            own_n_avail=self._n_available_effective(me),
            own_n=me.n,
            own_n_sustain=me.spec.n_sustain(
                me.v, me.z, throttle=me.throttle, mass=me.mass, ab=me.ab
            ),
            own_ps=me.spec.specific_power(
                me.v, me.z, max(me.n, 1.0),
                throttle=me.throttle, mass=me.mass, ab=me.ab,
            ),
            own_ps_level=me.spec.specific_power(
                me.v, me.z, 1.0, throttle=me.throttle, mass=me.mass, ab=me.ab
            ),
            own_fuel_frac=(
                me.fuel / me.spec.fuel_capacity
                if me.spec.fuel_capacity > 0.0 else 1.0
            ),
            own_ab=me.ab > 0.05,
            own_thrust_frac=me.thrust / max(me.spec.thrust_max, 1.0),
            own_cruise_throttle=_cruise_throttle(
                me.spec, me.z, speed_of_sound(me.z) * me.spec.cruise_mach
            ),
            own_hp=me.hp,
            own_hp_frac=me.hp / me.spec.hp_max,
            own_ammo=me.ammo,
            foe_speed=foe.v,
            foe_alt=foe.z,
            foe_psi=foe.psi,
            foe_gamma=foe.gamma,
            foe_hp=foe.hp,
            foe_hp_frac=foe.hp / foe.spec.hp_max,
            energy_diff=me.energy - foe.energy,
            edge_margin=min(half - abs(me.x), half - abs(me.y)),
            z_margin_low=z_low,
            z_margin_high=z_high,
            center_az=math.atan2(-me.y, -me.x),
            own_nose_az=nose_az,
            own_nose_el=nose_el,
            own_bank=me.bank,
            dof=self.dof,
            own_tvc=me.spec.tvc > 0.0,
            own_missiles=me.missiles,
            own_missiles_in_flight=in_flight,
            own_locked=me.locked,
            own_lock_progress=lock_progress,
            in_missile_envelope=in_msl,
            missile_min_range=msl_min,
            missile_max_range=msl_max,
            foe_missiles=foe.missiles,
            foe_locked=foe.locked,
            foe_lock_progress=foe_lock_progress,
            incoming_missile_dist=incoming_dist,
            incoming_missile_az=incoming_az,
            incoming_missile_ttg=incoming_ttg,
            incoming_missile_sarh=incoming_sarh,
            incoming_missile_radar=incoming_radar,
            own_radar_ttg=own_radar_ttg,
            own_radar_missiles=me.radar_missiles,
            own_radar_in_flight=radar_in_flight,
            own_radar_support=radar_support,
            own_radar_locked=me.radar_locked,
            own_radar_lock_progress=radar_progress,
            in_radar_envelope=in_radar,
            radar_tracking=radar_tracking,
            radar_min_range=radar_min,
            radar_max_range=radar_max,
            radar_pk_est=radar_pk,
            radar_nez=radar_nez,
            foe_radar_missiles=foe.radar_missiles,
            foe_radar_locked=foe_radar_locked,
        )

    def step(self, actions: list[Action]) -> None:
        if self.result is not None:
            return
        self._integrate(actions)
        self._update_lock()
        self._weapons(actions)
        self._integrate_bullets()
        self._update_countermeasures()
        self._update_missiles()
        self.t += self.dt
        self._referee()

    # ---------- 内部实现 ----------

    def _integrate(self, actions: list[Action]) -> None:
        for ac, act in zip(self.acs, actions):
            if not ac.alive:
                continue
            if self.dof == 6:
                self._integrate_rigid(ac, act)
            else:
                self._integrate_point(ac, act)

            self.stats["min_v"] = min(self.stats["min_v"], ac.v)
            self.stats["max_v"] = max(self.stats["max_v"], ac.v)
            self.stats["max_n"] = max(self.stats["max_n"], ac.n)
            self.stats["min_z"] = min(self.stats["min_z"], ac.z)
            self.stats["max_z"] = max(self.stats["max_z"], ac.z)
            self.stats["max_climb"] = max(self.stats["max_climb"], abs(math.degrees(ac.gamma)))

    def _apply_power(self, ac: Aircraft, act: Action, mach: float) -> float:
        """设定油门/加力、消耗燃油，返回本步推力 (N)。

        军推随油门在 [0, 军用推力] 内变化；加力是独立开度，叠加到军推之上。
        加力燃烧室需要油，油尽后自动掉回军推；本步油不够时按比例削减推力。
        """
        spec = ac.spec
        ac.throttle = clamp(act.throttle, 0.0, 1.0)
        ab = clamp(act.afterburner, 0.0, 1.0)
        if spec.fuel_capacity > 0.0 and ac.fuel <= 0.0:
            ab = 0.0
        ac.ab = ab
        mil = spec.mil_thrust()
        lapse = spec.lapse(ac.z, mach)
        dry = ac.throttle * mil * lapse
        wet = ab * (spec.thrust_max - mil) * lapse
        thrust = dry + wet
        if spec.fuel_capacity > 0.0:
            burn = (dry * TSFC_MIL + wet * TSFC_AB) * self.dt
            if burn > ac.fuel:
                # 油量不够支撑这一物理步：按剩余燃油等比削减，避免负油量。
                frac = ac.fuel / max(burn, 1e-9)
                dry *= frac
                wet *= frac
                thrust = dry + wet
                ac.ab = ab * frac
                burn = ac.fuel
            ac.fuel = max(ac.fuel - burn, 0.0)
            # 燃油本身计入机体质量：烧掉的油让整机变轻、推重比上升。
            ac.mass = max(ac.mass - burn, 0.35 * spec.mass)
        ac.thrust = thrust
        return thrust

    def _integrate_point(self, ac: Aircraft, act: Action) -> None:
        """2DOF/3DOF：点质量积分。"""
        spec = ac.spec
        m = ac.mass if ac.mass > 0.0 else spec.mass
        # 当地的密度、失速速度与推力衰减，全部按高度实时计算。
        rho = air_density(ac.z)
        v_floor = max(35.0, 0.20 * spec.v_stall_at(ac.z, m))
        v = max(ac.v, v_floor)
        # 不再把 gamma 夹在 ±80°：筋斗/半滚倒转必须能越过 ±90°。
        gamma = wrap_pi(ac.gamma)
        cg = math.cos(gamma)

        n_avail = spec.n_available(v, ac.z, m)
        n_cmd = clamp(act.n, N_MIN, n_avail)
        bank_cmd = clamp(act.bank, -BANK_CMD_MAX, BANK_CMD_MAX) * MAX_BANK

        # 一阶惯性，近似滚转/拉杆的响应延迟。
        alpha = self.dt / (spec.turn_tau + self.dt)
        ac.n = clamp(ac.n + (n_cmd - ac.n) * alpha, N_MIN, n_avail)
        ac.bank = clamp(ac.bank + (bank_cmd - ac.bank) * alpha,
                        -BANK_CMD_MAX * MAX_BANK, BANK_CMD_MAX * MAX_BANK)
        q = 0.5 * rho * v * v
        mach = mach_number(v, ac.z)
        a_thrust = self._apply_power(ac, act, mach) / m
        a_drag0 = q * cd0_at(mach, spec.cd0) * spec.wing_area / m
        a_induced = (
            ac.n * ac.n
            * 2.0 * m * G * G
            / (rho * v * v * math.pi * spec.aspect_ratio * spec.oswald * spec.wing_area)
        )

        # 3DOF 点质量方程。
        if self.dof == 3:
            v_dot = a_thrust - a_drag0 - a_induced - G * math.sin(gamma)
            gamma_dot = (G / v) * (ac.n * math.cos(ac.bank) - math.cos(gamma))
            # psi_dot 在 |cos(gamma)| -> 0 处有坐标奇点（机头垂直向上/向下）。
            # 纯垂直面内（bank≈0 或 π）时航向本就不该变；其余情况给有界极限，
            # 避免 0/0 把航向瞬间甩出去。
            sb = math.sin(ac.bank)
            if abs(cg) > 0.05:
                psi_dot = (G / (v * cg)) * ac.n * sb
            elif abs(sb) < 0.05:
                psi_dot = 0.0
            else:
                psi_dot = PSI_DOT_LIMIT if sb > 0.0 else -PSI_DOT_LIMIT
            psi_dot = clamp(psi_dot, -PSI_DOT_LIMIT, PSI_DOT_LIMIT)
        else:
            # 2DOF：垂直自由度被移除，水平响应与 3DOF 完全一致。
            v_dot = a_thrust - a_drag0 - a_induced
            gamma_dot = 0.0
            psi_dot = (G / v) * ac.n * math.sin(ac.bank)

        ac.v = max(v + v_dot * self.dt, v_floor)
        ac.psi = wrap_pi(ac.psi + psi_dot * self.dt)
        if self.dof == 3:
            ac.gamma = wrap_pi(gamma + gamma_dot * self.dt)
            cg_new = math.cos(ac.gamma)
            ac.x += ac.v * cg_new * math.cos(ac.psi) * self.dt
            ac.y += ac.v * cg_new * math.sin(ac.psi) * self.dt
            ac.z += ac.v * math.sin(ac.gamma) * self.dt
        else:
            ac.gamma = 0.0
            ac.x += ac.v * math.cos(ac.psi) * self.dt
            ac.y += ac.v * math.sin(ac.psi) * self.dt

    def _n_available_effective(self, ac: Aircraft) -> float:
        """当前可用过载：升力上限（含超声速损失与矢量增升）与结构限制取小。"""
        spec = ac.spec
        n_lift = spec.n_lift(ac.v, ac.z, ac.mass)
        boost = self._tvc_cl_boost(ac)
        if boost > 0.0:
            base = max(spec.cl_max_at(mach_number(ac.v, ac.z)), 1e-6)
            n_lift *= (base + boost) / base
        return min(spec.n_struct, n_lift)

    def _tvc_cl_boost(self, ac: Aircraft) -> float:
        """矢量喷流当前的等效增升量（CL 增量）。

        喷流增升取决于发动机功率：加力全开时喷流最强，收油门则迅速减弱，
        因此按油门/加力开度缩放，而不是按随高度衰减的绝对推力缩放——
        否则高空大迎角（矢量喷口真正有价值的包线）反而拿不到增升。
        """
        spec = ac.spec
        if not ac.rigid or spec.tvc <= 0.0 or spec.tvc_cl_bonus <= 0.0:
            return 0.0
        power = clamp(max(ac.throttle, ac.ab), 0.0, 1.0)
        return spec.tvc_cl_bonus * power

    def _cl_max_effective(self, ac: Aircraft) -> float:
        """计入超声速升力损失与矢量增升后的可用升力系数上限。"""
        spec = ac.spec
        base = spec.cl_max_at(mach_number(ac.v, ac.z))
        boost = self._tvc_cl_boost(ac)
        if boost <= 0.0:
            return base
        return spec.cl0 + spec.cl_alpha * spec.alpha_stall + boost

    def _alpha_stall_effective(self, ac: Aircraft) -> float:
        """矢量增升把失速迎角一并右移：CL 增量摊到升力线斜率上。"""
        spec = ac.spec
        boost = self._tvc_cl_boost(ac)
        if boost <= 0.0:
            return spec.alpha_stall
        return spec.alpha_stall + boost / spec.cl_alpha

    def _surface_commands(self, ac: Aircraft, act: Action, V: float,
                          alpha: float, qbar: float) -> tuple[float, float, float, float]:
        """增稳内回路：把战术层的 (bank, n) 翻译成副翼/升降舵/方向舵偏度。

        这是 6DOF 与 3DOF 的关键衔接——战术层仍然只关心“转多快、拉几个 g”，
        由这里负责配平、阻尼和协调转弯，避免把飞控品质问题暴露给战术层。
        """
        spec = ac.spec
        # 三层级联：姿态角误差 → 目标角速度 → 舵面，避免单环比例控制大过冲。
        # 本层约定 bank>0 为左滚，而 phi>0 为右滚。bank 以 MAX_BANK 为单位，
        # 最大允许指令到倒扣 ±180°（破 S / 半滚倒转需要）。
        phi_cmd = -clamp(act.bank, -BANK_CMD_MAX, BANK_CMD_MAX) * MAX_BANK
        p_cmd = clamp(2.5 * wrap_pi(phi_cmd - ac.phi), -1.5, 1.5)
        da = clamp(0.9 * (p_cmd - ac.p), -1.0, 1.0)

        # 俯仰：目标过载 → 目标升力系数 → 目标迎角 → 目标俯仰角速度 → 升降舵
        cl_req = act.n * ac.weight / max(qbar * spec.wing_area, 1e-6)
        cl_req = clamp(cl_req, -self._cl_max_effective(ac), self._cl_max_effective(ac))
        # 矢量喷口能把迎角带到失速以后；无矢量机型仍限制在失速迎角内。
        alpha_lim = spec.alpha_stall + 0.35 * spec.tvc_alpha_max
        # 矢量增升把“线性段可用迎角”一并抬高，否则 cl_req 提高了、
        # 迎角指令却仍被压回原失速迎角，增升拿不到。
        alpha_lim = max(alpha_lim, self._alpha_stall_effective(ac))
        alpha_cmd = clamp(
            (cl_req - spec.cl0) / spec.cl_alpha, -alpha_lim, alpha_lim
        )
        if act.post_stall and spec.tvc_alpha_max > 0.0:
            # 过失速指令：直接把目标迎角拉到矢量喷口能维持的极限。
            alpha_cmd = spec.tvc_alpha_max
        # 前馈抵消静稳定力矩，否则稳态迎角只能达到指令的 ~85%，过载会偏小。
        de_trim = -(spec.cm_alpha * alpha_cmd) / spec.cm_de
        # 航迹旋转前馈：保持目标迎角时需要机头跟着航迹一起转（q ≈ gamma_dot），
        # 否则纯比例环节会在 alpha_cmd - gamma_dot/3 处提前平衡，
        # 稳态实际过载只有指令的 ~80%（高 G 机动拉不出 6 g 的根源）。
        gamma_dot = G * (
            ac.n * math.cos(ac.phi) - math.cos(ac.gamma)
        ) / max(V, 1.0)
        q_cmd = clamp(8.0 * (alpha_cmd - alpha) + gamma_dot, -0.9, 0.9)
        de = clamp(de_trim - 1.5 * (q_cmd - ac.q), -1.0, 1.0)

        # 航向：侧滑角 → 目标偏航角速度 → 方向舵（同时抑制侧滑与偏航振荡）
        beta = math.asin(clamp(ac.vb / V, -1.0, 1.0))
        r_cmd = clamp(2.0 * beta, -0.5, 0.5)
        dr = clamp(3.0 * (r_cmd - ac.r), -1.0, 1.0)
        # 矢量喷口：迎角误差越大喷流偏得越多。低速时舵面几乎没有力矩，
        # 这一路是 Su-57 能维持大迎角、过失速机动的关键。
        tvc_p = 0.0
        if spec.tvc > 0.0:
            tvc_p = clamp(2.0 * (alpha_cmd - alpha), -spec.tvc, spec.tvc) * ac.throttle
        return da, de, dr, tvc_p

    def _integrate_rigid(self, ac: Aircraft, act: Action) -> None:
        """6DOF：刚体六自由度积分（气动力/力矩 + 陀螺耦合 + 欧拉运动学）。"""
        spec = ac.spec
        dt = self.dt
        rho = air_density(ac.z)

        V = max(math.sqrt(ac.u * ac.u + ac.vb * ac.vb + ac.wb * ac.wb), 20.0)
        qbar = 0.5 * rho * V * V
        alpha = math.atan2(ac.wb, max(ac.u, 1.0))
        beta = math.asin(clamp(ac.vb / V, -1.0, 1.0))
        mach = mach_number(V, ac.z)
        thrust = self._apply_power(ac, act, mach)

        da, de, dr, tvc_p = self._surface_commands(ac, act, V, alpha, qbar)
        ac.tvc_cmd = tvc_p

        # 升力曲线：失速后升力快速衰减；矢量喷口机型衰减更慢，
        # 因为喷流仍能把机头压在超大迎角上，机身也持续提供涡升力。
        # 有矢量增升时，线性段一直延伸到新的失速迎角，之后才进入衰减，
        # 相当于喷流把可用升力上限整体抬高。
        a_abs = abs(alpha)
        alpha_stall = self._alpha_stall_effective(ac)
        if a_abs <= alpha_stall:
            # 线性段也要服从超声速升力损失：cl_alpha 按失速迎角处刚好达到
            # cl_max 标定，M1.6 以上再整体乘 lift_scale。
            cl_lin = spec.cl0 + spec.cl_alpha * alpha
            cl_lim = spec.cl_max_at(mach)
            cl = clamp(cl_lin, -cl_lim, cl_lim)
        else:
            cl_stall = spec.cl0 + spec.cl_alpha * alpha_stall
            cl_stall = min(cl_stall, spec.cl_max_at(mach))
            decay_rate = 1.6 if spec.tvc <= 0.0 else 1.0
            floor = 0.35 if spec.tvc <= 0.0 else 0.55
            decay = max(floor, 1.0 - decay_rate * (a_abs - alpha_stall))
            cl = math.copysign(cl_stall * decay, alpha)
        k_ind = 1.0 / (math.pi * spec.aspect_ratio * spec.oswald)
        mach = mach_number(V, ac.z)
        cd = cd0_at(mach, spec.cd0) + k_ind * cl * cl + 0.8 * beta * beta
        cy = spec.cy_beta * beta

        qs = qbar * spec.wing_area
        lift = qs * cl
        drag = qs * cd
        side = qs * cy
        # 推力取 _apply_power 的结果：它已经把军推/加力分开、按燃油削减并
        # 做了高度衰减。这里若再按 throttle×最大加力重算，加力开度会被忽略，
        # 而且巡航油门会凭空多出约 1.5 倍推力，飞机会一路加速、盘旋也停不下来。

        # 机体系合力（标准气动方程）：阻力沿速度反方向，升力垂直于速度。
        # 注意升力在机体系 x 方向的分量是 +L sin(alpha)：迎角越大，升力越前倾。
        fx = thrust - drag * ac.u / V + lift * math.sin(alpha)
        fy = side - drag * ac.vb / V
        fz = -drag * ac.wb / V - lift * math.cos(alpha) + thrust * tvc_p

        # 重力在机体系的分量
        st, ct = math.sin(ac.theta), math.cos(ac.theta)
        sf, cf = math.sin(ac.phi), math.cos(ac.phi)
        m = ac.mass if ac.mass > 0.0 else spec.mass
        du = fx / m - G * st + ac.r * ac.vb - ac.q * ac.wb
        dvb = fy / m + G * sf * ct + ac.p * ac.wb - ac.r * ac.u
        dwb = fz / m + G * cf * ct + ac.q * ac.u - ac.p * ac.vb

        # 气动力矩（无量纲系数 × 动压 × 面积 × 特征长度）
        b, c = spec.span, spec.chord
        p_hat = ac.p * b / (2.0 * V)
        q_hat = ac.q * c / (2.0 * V)
        r_hat = ac.r * b / (2.0 * V)
        roll_m = qs * b * (
            spec.cl_beta * beta + spec.cl_p * p_hat + spec.cl_da * da + spec.cl_dr * dr
        )
        pitch_m = qs * c * (
            spec.cm_alpha * alpha + spec.cm_q * q_hat + spec.cm_de * de
        )
        if tvc_p != 0.0:
            # 矢量喷口的推力方向偏转在尾段产生俯仰力矩，低速时补足舵面权限。
            pitch_m += thrust * tvc_p * 0.35 * spec.length
        yaw_m = qs * b * (
            spec.cn_beta * beta + spec.cn_r * r_hat
            + spec.cn_dr * dr + spec.cn_da * da
        )
        ixx, iyy, izz = spec.ixx_of(m), spec.iyy_of(m), spec.izz_of(m)
        p_dot = (roll_m + (iyy - izz) * ac.q * ac.r) / ixx
        q_dot = (pitch_m + (izz - ixx) * ac.p * ac.r) / iyy
        r_dot = (yaw_m + (ixx - iyy) * ac.p * ac.q) / izz

        ac.u += du * dt
        ac.vb += dvb * dt
        ac.wb += dwb * dt
        ac.p = clamp(ac.p + p_dot * dt, -6.0, 6.0)
        ac.q = clamp(ac.q + q_dot * dt, -4.0, 4.0)
        ac.r = clamp(ac.r + r_dot * dt, -2.0, 2.0)

        # 姿态用四元数积分：q̇ = 0.5·q ⊗ (0, p, q, r)，再归一化。
        # 欧拉角只作为派生量输出，theta 越过 ±90° 时不再被夹断——
        # 筋斗、半滚倒转、垂直剪刀都需要机体连续过顶。
        qw, qx, qy, qz = ac.quat
        dq = _quat_mul((qw, qx, qy, qz), (0.0, ac.p, ac.q, ac.r))
        qn = (
            qw + 0.5 * dq[0] * dt,
            qx + 0.5 * dq[1] * dt,
            qy + 0.5 * dq[2] * dt,
            qz + 0.5 * dq[3] * dt,
        )
        norm = math.sqrt(sum(c * c for c in qn)) or 1.0
        ac.quat = tuple(c / norm for c in qn)
        ac.phi, ac.theta, ac.psi = euler_from_quat(ac.quat)

        # 世界系速度与位置
        rot = quat_to_matrix(ac.quat)
        wx = rot[0][0] * ac.u + rot[0][1] * ac.vb + rot[0][2] * ac.wb
        wy = rot[1][0] * ac.u + rot[1][1] * ac.vb + rot[1][2] * ac.wb
        wz = rot[2][0] * ac.u + rot[2][1] * ac.vb + rot[2][2] * ac.wb
        ac.x += wx * dt
        ac.y += wy * dt
        ac.z += wz * dt

        # 派生出点质量视角的状态，供观测、武器、渲染和裁判复用
        ac.v = max(math.sqrt(ac.u * ac.u + ac.vb * ac.vb + ac.wb * ac.wb), 1.0)
        ac.gamma = math.asin(clamp(wz / ac.v, -1.0, 1.0))
        ac.n = clamp(lift / ac.weight, N_MIN, 12.0)
        ac.bank = ac.phi

    def _update_lock(self) -> None:
        """双通道锁定：红外导引头与载机雷达各自累计稳定跟踪时间。

        目标离开包线立即解锁（不保留记忆），因此高离轴发射前必须先把机头
        指向目标；雷达制导中距弹在发射后还要保持跟踪（主动弹在末段截获前），
        否则导弹会丢失制导。
        """
        for i, ac in enumerate(self.acs):
            ac.launch_cd = max(0.0, ac.launch_cd - self.dt)
            ac.radar_cd = max(0.0, ac.radar_cd - self.dt)
            foe = self.acs[1 - i]
            if not ac.alive or not foe.alive:
                ac.locked = ac.radar_locked = False
                ac.lock_timer = ac.radar_lock_timer = 0.0
                continue

            dx, dy, dz = foe.x - ac.x, foe.y - ac.y, foe.z - ac.z
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            if dist < 1e-6:
                ac.locked = ac.radar_locked = False
                ac.lock_timer = ac.radar_lock_timer = 0.0
                continue
            nx, ny, nz = nose_direction(ac)
            ang = math.acos(clamp(
                (dx * nx + dy * ny + dz * nz) / dist, -1.0, 1.0
            ))

            mspec = ac.spec.missile
            if mspec is None or ac.missiles <= 0:
                ac.locked = False
                ac.lock_timer = 0.0
            else:
                lo, hi = launch_envelope(ac, mspec, foe)
                inside = lo <= dist <= hi and ang <= mspec.seeker_fov
                ac.lock_timer = (
                    min(ac.lock_timer + self.dt, mspec.lock_time) if inside else 0.0
                )
                ac.locked = ac.lock_timer >= mspec.lock_time

            rspec = ac.spec.radar_missile
            if rspec is None or ac.radar_missiles <= 0:
                ac.radar_locked = False
                ac.radar_lock_timer = 0.0
            else:
                # 目标 RCS 越小，雷达截获距离越短（隐身机的核心收益）。
                lo, hi = launch_envelope(ac, rspec, foe)
                inside = lo <= dist <= hi and ang <= rspec.seeker_fov
                ac.radar_lock_timer = (
                    min(ac.radar_lock_timer + self.dt, rspec.lock_time)
                    if inside else 0.0
                )
                ac.radar_locked = ac.radar_lock_timer >= rspec.lock_time

    def _weapons(self, actions: list[Action]) -> None:
        """机炮按射速生成弹丸；命中由后续的外弹道积分判定。

        旧实现是“目标在炮口视锥内就按秒伤扣血”：没有飞行时间、没有提前量、
        没有下坠与散布，6DOF 下炮线偏置也被抹平。现在每次物理步按
        rounds_per_sec·dt 累积发数，弹丸以“载机速度 + 炮口初速”出膛，
        之后只受二次阻力与重力，命中用线段-球判定，单发伤害 = dps/射速。
        """
        for i, (ac, act) in enumerate(zip(self.acs, actions)):
            # 记录扳机状态，渲染层据此画炮口焰与炮线特效。
            ac.firing = bool(ac.alive and act.fire and ac.ammo > 0.0)
            if not ac.firing:
                ac.gun_pending = 0.0
                continue
            spec = ac.spec
            foe = self.acs[1 - i]
            if not foe.alive:
                ac.firing = False
                ac.gun_pending = 0.0
                continue

            ac.gun_pending += spec.rounds_per_sec * self.dt
            n_new = int(ac.gun_pending)
            ac.gun_pending -= n_new
            n_new = min(n_new, 2, int(ac.ammo + 1e-9))
            if n_new <= 0:
                continue
            ac.ammo = max(0.0, ac.ammo - n_new)
            self.stats["rounds_fired"] += n_new
            self.stats["bullets_fired"] += n_new

            nx, ny, nz = nose_direction(ac)
            # 与机头垂直的两个基向量，用来把高斯散布叠加到炮线上。
            rx, ry, rz = (0.0, 0.0, 1.0) if abs(nz) < 0.9 else (1.0, 0.0, 0.0)
            ux = ny * rz - nz * ry
            uy = nz * rx - nx * rz
            uz = nx * ry - ny * rx
            un = max(math.sqrt(ux * ux + uy * uy + uz * uz), 1e-9)
            ux, uy, uz = ux / un, uy / un, uz / un
            wx = ny * uz - nz * uy
            wy = nz * ux - nx * uz
            wz = nx * uy - ny * ux

            vx, vy, vz = world_velocity(ac)
            area = math.pi * (0.5 * spec.bullet_diameter) ** 2
            drag_k = air_density(ac.z) * 0.30 * area / (2.0 * spec.bullet_mass)
            damage = spec.gun_dps / max(spec.rounds_per_sec, 1e-6)
            muzzle = spec.muzzle_speed
            for _ in range(n_new):
                sx = self._rng.gauss(0.0, spec.gun_spread)
                sy = self._rng.gauss(0.0, spec.gun_spread)
                dx = nx + ux * sx + wx * sy
                dy = ny + uy * sx + wy * sy
                dz = nz + uz * sx + wz * sy
                dn = max(math.sqrt(dx * dx + dy * dy + dz * dz), 1e-9)
                dx, dy, dz = dx / dn, dy / dn, dz / dn
                self.bullets.append(Bullet(
                    owner=i,
                    x=ac.x + nx * spec.length * 0.5,
                    y=ac.y + ny * spec.length * 0.5,
                    z=ac.z + nz * spec.length * 0.5,
                    vx=vx + muzzle * dx,
                    vy=vy + muzzle * dy,
                    vz=vz + muzzle * dz,
                    mass=spec.bullet_mass,
                    drag_k=drag_k,
                    damage=damage,
                ))

        self._launch_missiles(actions)

    def _integrate_bullets(self) -> None:
        """弹丸外弹道：二次阻力 + 重力，线段-球命中判定。"""
        dt = self.dt
        for b in self.bullets:
            if not b.alive:
                continue
            b.t += dt
            sp = math.sqrt(b.vx * b.vx + b.vy * b.vy + b.vz * b.vz)
            ax = -b.drag_k * sp * b.vx
            ay = -b.drag_k * sp * b.vy
            az = -b.drag_k * sp * b.vz - G
            x0, y0, z0 = b.x, b.y, b.z
            b.vx += ax * dt
            b.vy += ay * dt
            b.vz += az * dt
            sx, sy, sz = b.vx * dt, b.vy * dt, b.vz * dt
            b.x = x0 + sx
            b.y = y0 + sy
            b.z = z0 + sz

            for j, ac in enumerate(self.acs):
                if j == b.owner or not ac.alive:
                    continue
                r = max(3.0, 0.35 * ac.spec.length)
                l2 = sx * sx + sy * sy + sz * sz
                if l2 <= 1e-12:
                    continue
                t = clamp(((ac.x - x0) * sx + (ac.y - y0) * sy
                           + (ac.z - z0) * sz) / l2, 0.0, 1.0)
                dx = x0 + sx * t - ac.x
                dy = y0 + sy * t - ac.y
                dz = z0 + sz * t - ac.z
                if dx * dx + dy * dy + dz * dz <= r * r:
                    b.alive = False
                    damage = min(b.damage, ac.hp)
                    ac.hp -= damage
                    self.stats["damage_dealt"][b.owner] += damage
                    self.stats["bullet_hits"] += 1
                    break

            if b.alive and (b.t >= b.life or b.z <= self.z_floor):
                b.alive = False

        # 高射速下弹丸列表会快速膨胀，定期清掉已失效的弹丸（渲染只关心在飞的）。
        if len(self.bullets) > 400:
            self.bullets = [b for b in self.bullets if b.alive]

    def _launch_missiles(self, actions: list[Action]) -> None:
        """只有完成锁定、在发射包线内、还有余弹且冷却结束才能发射。"""
        for i, (ac, act) in enumerate(zip(self.acs, actions)):
            if not ac.alive:
                continue
            self._try_launch(i, ac, ac.spec.missile, act.launch)
            self._try_launch(i, ac, ac.spec.radar_missile, act.launch_radar)

    def _try_launch(self, i: int, ac: Aircraft,
                    mspec: MissileSpec | None, requested: bool) -> None:
        if mspec is None or not requested:
            return
        radar = mspec.is_radar
        if (ac.radar_cd if radar else ac.launch_cd) > 0.0:
            return
        count = ac.radar_missiles if radar else ac.missiles
        locked = ac.radar_locked if radar else ac.locked
        if count <= 0 or not locked:
            return
        foe = self.acs[1 - i]
        if not foe.alive:
            return
        dx, dy, dz = foe.x - ac.x, foe.y - ac.y, foe.z - ac.z
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        lo, hi = launch_envelope(ac, mspec, foe)
        if not (lo <= dist <= hi):
            return

        nx, ny, nz = nose_direction(ac)
        vx, vy, vz = world_velocity(ac)
        launch_speed = 45.0
        off = ac.spec.length * 0.5
        self.missiles.append(
            Missile(
                spec=mspec,
                owner=i,
                target=1 - i,
                x=ac.x + nx * off,
                y=ac.y + ny * off,
                z=ac.z + nz * off,
                vx=vx + nx * launch_speed,
                vy=vy + ny * launch_speed,
                vz=vz + nz * launch_speed,
                mass=mspec.mass,
                fuel=mspec.propellant,
                nx=nx, ny=ny, nz=nz,
                roll_seed=int(round(self.t * 1000.0)) * 131 + i * 7919 + 1,
            )
        )
        if radar:
            ac.radar_missiles -= 1
            ac.radar_cd = mspec.cooldown
        else:
            ac.missiles -= 1
            ac.launch_cd = mspec.cooldown
        ac.mass = max(1.0, ac.mass - mspec.mass)
        self.stats["missiles_fired"] += 1
        self.stats["missiles_launched"][i] += 1
        self.events.append((self.t, f"{ac.spec.name} 发射 {mspec.name}"))

    def _update_countermeasures(self) -> None:
        """干扰物投放、飘落与对在飞导弹的诱骗。

        自动投放：雷达制导导弹逼近到 9 km 投箔条，红外弹逼近到 5 km 投热诱弹。
        诱骗判定：诱饵落在导引头视场内且足够近时，按秒概率把导弹的视线
        拉到诱饵上（主动弹需要已进入末段自主截获，中制导段的数据链不受箔条影响）。
        """
        dt = self.dt

        for idx, ac in enumerate(self.acs):
            ac.chaff_cd = max(0.0, ac.chaff_cd - dt)
            ac.flare_cd = max(0.0, ac.flare_cd - dt)
            if not ac.alive:
                continue
            for m in self.missiles:
                if not m.alive or m.target != idx or not m.guided:
                    continue
                dx, dy, dz = ac.x - m.x, ac.y - m.y, ac.z - m.z
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if m.spec.is_radar and ac.chaff > 0 and ac.chaff_cd <= 0.0:
                    if dist <= AUTO_CHAFF_RANGE:
                        # 箔条必须配合横转/三九投放：迎头撒出去只会被多普勒
                        # 滤波直接滤掉，白白浪费干扰弹。
                        q = notch_quality(ac, m.x, m.y, m.z)
                        if q >= CHAFF_NOTCH_MIN:
                            self._drop(idx, ac, "chaff", q)
                            ac.chaff -= 1
                            ac.chaff_cd = CM_DROP_CD
                elif not m.spec.is_radar and ac.flares > 0 and ac.flare_cd <= 0.0:
                    if dist <= AUTO_FLARE_RANGE:
                        self._drop(idx, ac, "flare")
                        ac.flares -= 1
                        ac.flare_cd = CM_DROP_CD

        # 干扰物飘落：强阻尼 + 重力，箔条几乎停在原地，热诱弹掉得更快。
        for cm in self.counters:
            if not cm.alive:
                continue
            cm.t += dt
            if cm.t >= cm.life:
                cm.alive = False
                continue
            decay = math.exp(-1.8 * dt)
            cm.vx *= decay
            cm.vy *= decay
            cm.vz = (cm.vz - G * dt) * decay
            cm.x += cm.vx * dt
            cm.y += cm.vy * dt
            cm.z += cm.vz * dt

        # 诱骗：每枚导弹每个物理步按概率被最近的、在视场内的对应诱饵吸引。
        for m in self.missiles:
            if not m.alive or not m.guided or m.decoy_t > 0.0:
                continue
            # 主动弹只有末段自主截获后才吃箔条；中制导段靠数据链，骗不走。
            if m.spec.is_radar and not m.active:
                continue
            kind = "chaff" if m.spec.is_radar else "flare"
            reach = CHAFF_SEDUCE_RANGE if m.spec.is_radar else FLARE_SEDUCE_RANGE
            rate = CHAFF_SEDUCE_RATE if m.spec.is_radar else FLARE_SEDUCE_RATE
            best = None
            best_d = reach
            for cm in self.counters:
                if not cm.alive or cm.kind != kind or cm.owner == m.owner:
                    continue
                dx, dy, dz = cm.x - m.x, cm.y - m.y, cm.z - m.z
                d = math.sqrt(dx * dx + dy * dy + dz * dz)
                if d >= best_d or d < 1e-6:
                    continue
                # 诱饵必须落在弹体导引头视场内才有意义。
                if (m.nx * dx + m.ny * dy + m.nz * dz) / d < math.cos(m.spec.gimbal):
                    continue
                best, best_d = cm, d
            if best is None:
                continue
            if self._rng.random() < rate * dt * best.quality:
                if m.spec.is_sarh:
                    # 半主动弹没有自主导引头，箔条直接打断载机的照射回路。
                    m.guided = False
                else:
                    m.decoy = best.position
                    m.decoy_t = CM_DECOY_KEEP
                self.events.append((
                    self.t,
                    f"{m.spec.name} 被{'箔条' if kind == 'chaff' else '热诱弹'}诱骗",
                ))

    def _drop(self, idx: int, ac: Aircraft, kind: str,
              quality: float = 1.0) -> None:
        """从机尾投放一枚干扰物：初速取载机速度的 30%，并带一点下坠。

        quality 是投放时的横转程度，箔条只有落在载机同一多普勒通道里
        才有效，这里把它记在诱饵上，供后续诱骗概率使用。
        """
        vx, vy, vz = world_velocity(ac)
        life = CHAFF_LIFE if kind == "chaff" else FLARE_LIFE
        self.counters.append(
            Countermeasure(
                kind=kind,
                owner=idx,
                x=ac.x,
                y=ac.y,
                z=ac.z,
                vx=vx * 0.30,
                vy=vy * 0.30,
                vz=vz * 0.30 - 6.0,
                life=life,
                quality=clamp(quality, 0.0, 1.0),
            )
        )

    def _update_missiles(self) -> None:
        """格斗弹动力学：助推 + 阻力 + 重力 + 比例导引 + 有限舵面过载。

        命中完全由几何与能量决定：导引头一旦看不到目标就转成无控弹道，
        近炸引信用线段最近距离判定，避免高速弹丸在两帧之间穿过目标。
        """
        dt = self.dt
        for m in self.missiles:
            if not m.alive:
                continue
            target = self.acs[m.target]
            m.t += dt
            if m.t >= m.spec.life:
                m.alive = False
                continue
            if not target.alive:
                m.alive = False
                continue

            # 被干扰物诱骗：视线改成追踪诱饵点（诱饵视为静止目标），
            # 诱骗时间结束后导引头彻底丢失目标，转成无控弹道。
            decoy = None
            if m.decoy_t > 0.0:
                m.decoy_t -= dt
                if m.decoy_t <= 0.0:
                    m.decoy = None
                    m.guided = False
                else:
                    decoy = m.decoy

            # 雷达制导中制导段：半主动弹全程靠载机照射；主动弹只在末段
            # 自主截获（pitbull）前靠载机雷达提供目标信息与数据链。
            # 载机被击落、或者机头偏出照射锥且尚未截获，导弹立即转成无控弹道。
            if m.spec.is_radar:
                if m.spec.is_arh and not m.active:
                    # 主动导引头截获距离按目标 RCS 缩放：隐身目标要更近了才“开眼”。
                    mdx, mdy, mdz = target.x - m.x, target.y - m.y, target.z - m.z
                    mdist = math.sqrt(mdx * mdx + mdy * mdy + mdz * mdz)
                    if (
                        m.guided
                        and mdist
                        <= m.spec.active_range * rcs_scale(target.spec.rcs)
                    ):
                        m.active = True
                if not m.active:
                    shooter = self.acs[m.owner]
                    illum = False
                    if shooter.alive:
                        sdx, sdy, sdz = target.x - shooter.x, target.y - shooter.y, target.z - shooter.z
                        sdist = max(math.sqrt(sdx * sdx + sdy * sdy + sdz * sdz), 1e-6)
                        snx, sny, snz = nose_direction(shooter)
                        sang = math.acos(clamp(
                            (sdx * snx + sdy * sny + sdz * snz) / sdist, -1.0, 1.0
                        ))
                        illum = (
                            sdist <= m.spec.max_range and sang <= SARH_ILLUM_FOV
                        )
                    if not illum:
                        m.guided = False

            # 推进剂质量流失，推力只作用到燃料耗尽。
            thrust = 0.0
            if m.fuel > 1e-6 and m.mass > 1e-6:
                burn = min(m.fuel, m.spec.mass_flow * dt)
                m.fuel -= burn
                m.mass -= burn
                thrust = m.spec.thrust
            if m.mass <= 1.0:
                m.alive = False
                continue

            speed = max(m.speed, 1e-6)
            ux, uy, uz = m.vx / speed, m.vy / speed, m.vz / speed

            # 诱骗期间视线指向诱饵，否则指向真目标。
            if decoy is None:
                ax_, ay_, az_ = target.x, target.y, target.z
            else:
                ax_, ay_, az_ = decoy
            rx, ry, rz = ax_ - m.x, ay_ - m.y, az_ - m.z
            dist = max(math.sqrt(rx * rx + ry * ry + rz * rz), 1e-6)
            losx, losy, losz = rx / dist, ry / dist, rz / dist

            rho = air_density(m.z)
            qbar = 0.5 * rho * speed * speed
            qs = max(qbar * m.spec.ref_area, 1e-6)
            a_axis_max = min(m.spec.max_g * G, qs * m.spec.cn_max / m.mass)
            # 动压不足时舵面压不出升力：导弹已经无法修正弹道，
            # 继续按“制导”处理会让它一边掉速一边追着飞机爬，不符合实际。
            if qbar < MISSILE_MIN_Q:
                m.guided = False
            # 更硬的判据：可用过载连 1g 都压不出来时，弹已经跟不上目标，
            # 不能再算“制导中”，否则会出现比飞机还慢的僵尸弹。
            if a_axis_max < G:
                m.guided = False

            # 导引头视场相对弹体轴；发射瞬间沿挂架，之后弹体以可用过载
            # 对应的角速度转向目标视线。目标视线转得比弹体快时才会丢锁。
            cos_seeker = clamp(
                m.nx * losx + m.ny * losy + m.nz * losz, -1.0, 1.0
            )
            ang_seeker = math.acos(cos_seeker)
            # 主动弹在末段截获前由数据链中制导，弹轴还没对准目标也不会丢弹；
            # 截获之后才真正靠自己的导引头，视场越界即丢失目标。
            midcourse = m.spec.is_arh and not m.active
            if ang_seeker > m.spec.gimbal and not midcourse:
                m.guided = False
            else:
                slew = min(
                    MISSILE_SLEW_RATE,
                    a_axis_max / max(speed, 40.0),
                )
                if ang_seeker > 1e-6:
                    frac = min(1.0, slew * dt / ang_seeker)
                    px_, py_, pz_ = (
                        losx - m.nx * cos_seeker,
                        losy - m.ny * cos_seeker,
                        losz - m.nz * cos_seeker,
                    )
                    pn_ = math.sqrt(px_ * px_ + py_ * py_ + pz_ * pz_)
                    if pn_ > 1e-9:
                        px_, py_, pz_ = px_ / pn_, py_ / pn_, pz_ / pn_
                        ca = math.cos(ang_seeker * frac)
                        sa = math.sin(ang_seeker * frac)
                        m.nx = m.nx * ca + px_ * sa
                        m.ny = m.ny * ca + py_ * sa
                        m.nz = m.nz * ca + pz_ * sa
                        nn = math.sqrt(m.nx * m.nx + m.ny * m.ny + m.nz * m.nz)
                        m.nx, m.ny, m.nz = m.nx / nn, m.ny / nn, m.nz / nn

            cmd_x = cmd_y = cmd_z = 0.0     # 本拍制导指令（横向加速度 N）
            if m.guided and dist > 1.0:
                if decoy is not None:
                    tvx = tvy = tvz = 0.0   # 诱饵随风飘落，按静止目标处理
                else:
                    tvx, tvy, tvz = target.velocity
                rvx, rvy, rvz = tvx - m.vx, tvy - m.vy, tvz - m.vz
                vc = -((rvx * rx + rvy * ry + rvz * rz) / dist)
                if vc > 0.0:
                    # LOS 角速度矢量 omega = (r × v_rel) / r²
                    ox = (ry * rvz - rz * rvy) / (dist * dist)
                    oy = (rz * rvx - rx * rvz) / (dist * dist)
                    oz = (rx * rvy - ry * rvx) / (dist * dist)
                    # 真比例导引：a = N * Vc * (omega × LOS)
                    cx = oy * losz - oz * losy
                    cy = oz * losx - ox * losz
                    cz = ox * losy - oy * losx
                    ax = m.spec.pn_gain * vc * cx
                    ay = m.spec.pn_gain * vc * cy
                    az = m.spec.pn_gain * vc * cz
                    # 大离轴发射时纯 PN 在视线角速度为零的瞬间不产生指令，
                    # 叠加一个朝向视线的追踪项，保证弹道能先转进目标方向。
                    # 系数不能太大：2V 在迎角稍偏时就要求十几 g，
                    # 持续最大过载带来的诱导阻力会把弹的能量瞬间抽干。
                    # 远程雷达弹的 loft：助推段把追踪的“虚拟瞄准点”抬高，
                    # 仰角偏置随时间衰减，末段 PN 再把弹压回目标。
                    tlx, tly, tlz = losx, losy, losz
                    if (
                        m.spec.loft
                        and m.t < MISSILE_LOFT_TIME
                        and dist > MISSILE_LOFT_MIN_RANGE
                    ):
                        # 助推前段保持满偏置，后段线性收回到真实视线。
                        # 早期弹速低、可用过载小，若一开始就衰减，弹永远抬不起来。
                        frac = min(1.0, (1.0 - m.t / MISSILE_LOFT_TIME) / 0.4)
                        bias = MISSILE_LOFT_ANGLE * frac
                        hnorm = math.hypot(losx, losy)
                        hx, hy = (losx / hnorm, losy / hnorm) if hnorm > 1e-6 else (1.0, 0.0)
                        tlx = hx * math.cos(bias)
                        tly = hy * math.cos(bias)
                        tlz = math.sin(bias)
                    cos_vlos = clamp(ux * tlx + uy * tly + uz * tlz, -1.0, 1.0)
                    pursuit = 0.6 * speed
                    ax += pursuit * (tlx - ux * cos_vlos)
                    ay += pursuit * (tly - uy * cos_vlos)
                    az += pursuit * (tlz - uz * cos_vlos)
                    a_mag = math.sqrt(ax * ax + ay * ay + az * az)
                    if a_mag > 1e-6:
                        # 可用过载同时受结构极限和舵面升力（动压）限制
                        scale = min(1.0, a_axis_max / a_mag)
                        ax, ay, az = ax * scale, ay * scale, az * scale
                        cmd_x, cmd_y, cmd_z = ax, ay, az

            # 自动驾驶仪/舵机一阶滞后：横向过载指令不能瞬间建立，
            # 末段目标拉高过载时弹的响应会晚半拍，这是真实脱靶的主因。
            alpha = 1.0 - math.exp(-dt / max(m.spec.autopilot_tau, 1e-3))
            m.gx += (cmd_x - m.gx) * alpha
            m.gy += (cmd_y - m.gy) * alpha
            m.gz += (cmd_z - m.gz) * alpha
            gx, gy, gz = m.gx, m.gy, m.gz

            lat = math.sqrt(gx * gx + gy * gy + gz * gz)
            cl_req = lat * m.mass / max(qbar * m.spec.ref_area, 1e-6)
            mach = m.speed / max(speed_of_sound(max(m.z, 0.0)), 1e-6)
            wave = m.spec.cd0_wave * _smoothstep(0.85, 1.15, mach)
            cd = m.spec.cd0 * (1.0 + wave) + 0.15 * cl_req * cl_req
            drag = qbar * m.spec.ref_area * cd
            a_drag = drag / m.mass

            axw = (thrust / m.mass) * m.nx - a_drag * ux + gx
            ayw = (thrust / m.mass) * m.ny - a_drag * uy + gy
            azw = (thrust / m.mass) * m.nz - a_drag * uz + gz - G

            px, py, pz = m.x, m.y, m.z
            m.vx += axw * dt
            m.vy += ayw * dt
            m.vz += azw * dt
            m.x += m.vx * dt
            m.y += m.vy * dt
            m.z += m.vz * dt

            # 近炸引信：用运动线段的最近点判定，避免高速下穿透目标。
            # 被诱骗的弹追的是诱饵，不会误伤真机。
            if decoy is not None:
                miss_dist = 1e9
            else:
                sx, sy, sz = m.x - px, m.y - py, m.z - pz
                seg2 = sx * sx + sy * sy + sz * sz
                if seg2 > 1e-9:
                    tt = clamp(
                        ((target.x - px) * sx + (target.y - py) * sy
                         + (target.z - pz) * sz) / seg2,
                        0.0, 1.0,
                    )
                else:
                    tt = 1.0
                cx_ = px + sx * tt
                cy_ = py + sy * tt
                cz_ = pz + sz * tt
                miss_dist = math.sqrt(
                    (target.x - cx_) ** 2
                    + (target.y - cy_) ** 2
                    + (target.z - cz_) ** 2
                )
            if miss_dist <= m.spec.lethal_radius:
                # 引信触发后再掷一次杀伤概率：脱靶量越大、交会速度越高，
                # 越可能只是近失；即使命中也可能只是打伤而不是击落。
                ratio = miss_dist / m.spec.lethal_radius
                pk = proximity_pk(miss_dist, m.spec.lethal_radius)
                roll = random.Random(m.roll_seed).random()
                m.alive = False
                if roll <= pk:
                    damage = min(
                        m.spec.warhead_damage * (0.45 + 0.55 * (1.0 - ratio)),
                        target.hp,
                    )
                    target.hp -= damage
                    self.stats["damage_dealt"][m.owner] += damage
                    self.stats["missile_hits"] += 1
                    self.stats["missiles_hit"][m.owner] += 1
                    self.events.append((
                        self.t,
                        f"{m.spec.name} 近炸命中 {target.spec.name} "
                        f"({miss_dist:.1f} m, 伤害 {damage:.0f})",
                    ))
                else:
                    self.events.append((
                        self.t,
                        f"{m.spec.name} 近失 {target.spec.name} "
                        f"({miss_dist:.1f} m, PK {pk:.0%})",
                    ))
                continue

            # 无控弹减速到临界速度后按坠落处理，避免出现比飞机还慢的“僵尸弹”。
            if not m.guided and m.speed < MISSILE_FALL_SPEED:
                m.alive = False
                continue

            if m.z <= self.z_floor or abs(m.x) > self.arena / 2 or abs(m.y) > self.arena / 2:
                m.alive = False

    def _referee(self) -> None:
        # 一方被击落时若仍有导弹在飞，先不结算：双方在超视距阶段几乎同时
        # 发射，先被打掉的一方可能已经有一枚弹在飞向对手。等这些弹全部
        # 落地/命中后再判定，避免“先被击落就直接输掉”忽略掉后到的命中。
        if self._pending_finish is not None:
            both_dead = self.acs[0].hp <= 0.0 and self.acs[1].hp <= 0.0
            if both_dead:
                for ac in self.acs:
                    ac.hp = 0.0
                    ac.alive = False
                self._pending_finish = None
                self.result = "draw"
                self.reason = "同归于尽"
                self.reason_code = "kill"
                self.events.append((self.t, self.reason))
            elif not any(m.alive for m in self.missiles):
                winner, reason, code = self._pending_finish
                self._pending_finish = None
                self._finish(winner, reason, code)
            return

        if self.acs[0].hp <= 0.0 and self.acs[1].hp <= 0.0:
            for ac in self.acs:
                ac.hp = 0.0
                ac.alive = False
            self.result = "draw"
            self.reason = "同归于尽"
            self.reason_code = "kill"
            self.events.append((self.t, self.reason))
            return
        for i, ac in enumerate(self.acs):
            if ac.hp <= 0.0:
                ac.hp = 0.0
                ac.alive = False
                reason = f"{ac.spec.name} 被击落"
                if any(m.alive for m in self.missiles):
                    self._pending_finish = (1 - i, reason, "kill")
                    self.events.append((self.t, f"{reason}，等待在飞导弹落地"))
                else:
                    self._finish(1 - i, reason, "kill")
                return
        for i, ac in enumerate(self.acs):
            if self.dof >= 3:
                if ac.z <= self.z_floor:
                    ac.z = self.z_floor
                    ac.alive = False
                    self._finish(1 - i, f"{ac.spec.name} 低于最低高度坠机", "boundary")
                    return
                if ac.z >= self.z_ceiling:
                    # 不再判定“解体”：ISA 只建模到 32 km，推力悬崖已经让飞机
                    # 在升限附近自然失去剩余功率。这里只记录一次越界事件，
                    # 保留诊断价值而不干扰对局结果。
                    if not ac.ceiling_warned:
                        ac.ceiling_warned = True
                        self.events.append((
                            self.t,
                            f"{ac.spec.name} 超过验证高度上限 "
                            f"{self.z_ceiling / 1000.0:.0f} km",
                        ))
            if abs(ac.x) > self.arena / 2 or abs(ac.y) > self.arena / 2:
                ac.alive = False
                self._finish(1 - i, f"{ac.spec.name} 飞出交战空域", "boundary")
                return

        if self.t >= self.time_limit:
            # 按血量比例比较，避免初始 HP 不同导致的重机默认占优。
            r_ratio = self.acs[0].hp / self.acs[0].spec.hp_max
            b_ratio = self.acs[1].hp / self.acs[1].spec.hp_max
            if r_ratio > b_ratio:
                self._finish(0, "超时判定：血量优势", "timeout")
            elif b_ratio > r_ratio:
                self._finish(1, "超时判定：血量优势", "timeout")
            else:
                self.result = "draw"
                self.reason = "超时判定：平局"
                self.reason_code = "draw"
                self.events.append((self.t, self.reason))

    def _finish(self, winner: int, reason: str, code: str) -> None:
        self.result = "red" if winner == 0 else "blue"
        self.reason = reason
        self.reason_code = code
        self.events.append((self.t, reason))

    def summary(self) -> str:
        r, b = self.acs
        return (
            f"t={self.t:6.2f}s  dof={self.dof}  result={self.result or 'running':8s}  {self.reason}\n"
            f"  {r.spec.name:>8s}  alt={r.z:6.0f}m  v={r.v:6.1f} m/s  g={r.n:4.1f}  "
            f"gam={math.degrees(r.gamma):+6.1f}deg  hp={r.hp:5.1f}  ammo={r.ammo:5.0f}  "
            f"msl={r.missiles}+{r.radar_missiles}  mass={r.mass:6.0f}kg  "
            f"ps={r.spec.specific_power(r.v, r.z, max(r.n, 1.0), mass=r.mass):+5.0f} m/s\n"
            f"  {b.spec.name:>8s}  alt={b.z:6.0f}m  v={b.v:6.1f} m/s  g={b.n:4.1f}  "
            f"gam={math.degrees(b.gamma):+6.1f}deg  hp={b.hp:5.1f}  ammo={b.ammo:5.0f}  "
            f"msl={b.missiles}+{b.radar_missiles}  mass={b.mass:6.0f}kg  "
            f"ps={b.spec.specific_power(b.v, b.z, max(b.n, 1.0), mass=b.mass):+5.0f} m/s\n"
            f"  speed [{self.stats['min_v']:.1f}, {self.stats['max_v']:.1f}] m/s  "
            f"alt [{self.stats['min_z']:.0f}, {self.stats['max_z']:.0f}] m  "
            f"max g={self.stats['max_n']:.1f}  max |gam|={self.stats['max_climb']:.0f}deg  "
            f"rounds={self.stats['rounds_fired']:.0f}  "
            f"msl fired={self.stats['missiles_fired']:.0f}  "
            f"hit={self.stats['missile_hits']} (R {self.stats['missiles_hit'][0]} / "
            f"B {self.stats['missiles_hit'][1]})"
        )
