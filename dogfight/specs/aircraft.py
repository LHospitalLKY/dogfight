"""机体性能规格。

数据类只负责“这架飞机是什么”，不负责去哪里读配置、怎么渲染。
挂载通过 Loadout 组合，机体本身不直接持有某一型导弹。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .aero import (
    G,
    _smoothstep,
    air_density,
    cd0_at,
    clamp,
    mach_number,
    thrust_lapse,
)
from .loadout import Loadout
from .weapon import MissileSpec


@dataclass(frozen=True)
class AircraftSpec:
    """机体性能参数。改配置就能得到完全不同的最优战术。"""

    name: str
    mass: float                 # kg，含全部挂载与燃油的作战全重
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
    # 分向 RCS：前/侧/后。0 表示沿用 rcs 的全向值。
    rcs_front: float = 0.0
    rcs_side: float = 0.0
    rcs_back: float = 0.0

    # ---- 传感器 ----（感知层：探测距离、视场与火控等级见 sim 顶部常量）
    radar_range: float = 0.0      # 机头雷达探测距离 @1 m²；0 表示沿用中距弹包线
    side_array: bool = False      # 侧阵（仅 Su-57）：±35°–145°，0.55×，仅 track
    lband: bool = False           # L 波段襟翼雷达（仅 Su-57）：±45°，200 km，仅 detect
    irst_range: float = 0.0       # 机载 IRST 被动跟踪距离 m
    pod_irst_range: float = 0.0   # 光电吊舱被动跟踪距离 m（F-16V Legion）

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

    # ---- 全动鸭翼（J-20） ----
    # 鸭翼是一个可偏转的独立气动面：既提供俯仰操纵/配平，又通过脱体涡
    # 延迟主翼失速。下面的几何/操纵参数缺省为 0，只有 canard=True 且
    # 配置了面积/力臂时才会真正产生力和力矩。
    canard: bool = False
    # 鸭翼总面积与主翼面积之比 S_c/S；两片鸭翼合并计算。
    canard_area_ratio: float = 0.0
    # 鸭翼气动中心到重心的距离，按主翼平均气动弦长归一化（正=在重心之前）。
    canard_arm_ratio: float = 0.0
    canard_chord: float = 1.0                  # 鸭翼平均气动弦长 m
    canard_deflect_max: float = math.radians(25.0)   # 最大偏度 rad
    canard_rate_max: float = math.radians(40.0)      # 最大偏转速率 rad/s
    canard_tau: float = 0.12                   # 作动器一阶时间常数 s
    canard_cl_alpha: float = 3.0               # 鸭翼升力线斜率 1/rad
    canard_cl0: float = 0.0                    # 鸭翼零迎角升力系数
    canard_cl_max: float = 1.5                 # 鸭翼自身 CLmax
    canard_alpha_stall: float = math.radians(18.0)   # 鸭翼自身失速迎角
    # 鸭翼对主翼的下洗：ε = canard_downwash × CL_c（rad）。只取有利的涡增升
    # 而不减下洗，会把鸭翼的收益系统性高估，这里显式扣掉主翼有效迎角。
    canard_downwash: float = 0.10
    # 铰链力矩：H = q·S_c·c_c·coeff·(0.6·α_c + δc)，作动器扭矩不足时偏度被限幅。
    canard_hinge_coeff: float = 0.25
    canard_actuator_torque: float = 35.0e3     # N·m
    # 鸭翼自身型阻/诱导阻力/偏转阻力（均以主翼面积为参考）。
    canard_cd0: float = 0.010
    canard_induced_k: float = 0.08
    canard_deflect_drag: float = 0.05          # 每 rad²
    # 脱体涡的建立/破裂时间常数与破裂速率。
    canard_vortex_tau: float = 0.08            # s
    canard_breakdown: float = 0.6              # 1/rad，超过饱和迎角后涡强衰减
    # 全动鸭翼能维持的过失速目标迎角（与涡饱和角 canard_alpha_max 区分开）。
    canard_post_stall_alpha: float = math.radians(38.0)

    # 保留旧的“涡增升”标定量：现在只表示鸭翼涡对**主翼**的 CLmax 增量，
    # 鸭翼自身的升力由 canard_area_ratio/cl_c 单独计入，避免重复计数。
    canard_cl_bonus: float = 0.0
    canard_alpha_max: float = 0.0             # 涡强饱和的鸭翼当地迎角 rad
    canard_stall_shift: float = 0.0           # 满涡时主翼失速角右移量 rad

    # ---- 挂载：近距弹 / 中距弹 / 干扰物 ----
    loadout: Loadout = Loadout()

    @property
    def missile(self) -> MissileSpec | None:
        """主用近距格斗弹（没有挂载时为 None）。"""
        return self.loadout.ir.weapon if self.loadout.ir is not None else None

    @property
    def missile_count(self) -> int:
        return self.loadout.ir.count if self.loadout.ir is not None else 0

    @property
    def radar_missile(self) -> MissileSpec | None:
        """主用雷达制导中距弹（没有挂载时为 None）。"""
        return self.loadout.radar.weapon if self.loadout.radar is not None else None

    @property
    def radar_missile_count(self) -> int:
        return self.loadout.radar.count if self.loadout.radar is not None else 0

    @property
    def chaff_count(self) -> int:
        return self.loadout.chaff

    @property
    def flare_count(self) -> int:
        return self.loadout.flare

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

    # ---- 全动鸭翼几何与气动 ----
    @property
    def canard_area(self) -> float:
        """两片鸭翼的总面积 m²。"""
        return max(0.0, self.canard_area_ratio) * self.wing_area

    @property
    def canard_arm(self) -> float:
        """鸭翼气动中心到重心的力臂 m（正=在重心之前）。"""
        return max(0.0, self.canard_arm_ratio) * self.chord

    @property
    def canard_span(self) -> float:
        if self.canard_chord <= 1e-6:
            return 0.0
        return self.canard_area / self.canard_chord

    def canard_cl_at(self, alpha_c: float, mach: float) -> float:
        """鸭翼自身升力系数（附着流线性段 + 失速后衰减）。

        alpha_c 是鸭翼当地迎角，等于全机迎角加上鸭翼偏度。
        """
        if not self.canard or self.canard_area <= 0.0:
            return 0.0
        cl_alpha = self.canard_cl_alpha * self.lift_scale(mach)
        a_abs = abs(alpha_c)
        if a_abs <= self.canard_alpha_stall:
            cl = self.canard_cl0 + cl_alpha * alpha_c
            return clamp(cl, -self.canard_cl_max, self.canard_cl_max)
        cl_stall = self.canard_cl0 + cl_alpha * self.canard_alpha_stall
        cl_stall = clamp(cl_stall, -self.canard_cl_max, self.canard_cl_max)
        decay = max(0.35, 1.0 - 1.6 * (a_abs - self.canard_alpha_stall))
        return math.copysign(cl_stall * decay, alpha_c)

    def wing_cl_at(self, alpha: float, mach: float, *,
                   boost: float = 0.0, stall_shift: float = 0.0,
                   decay_rate: float = 1.6, floor: float = 0.35) -> float:
        """主翼升力系数——全模型唯一的升力曲线入口。

        boost 把 CLmax 整体抬高（矢量喷流增升 / 鸭翼涡增升），stall_shift
        把失速迎角右移。两者只在这里各加一次，力模型与控制律都调用本函数，
        避免“控制律按 1.93 算、积分却截到 1.60”的历史不一致。
        """
        alpha_stall = self.alpha_stall + max(0.0, stall_shift)
        cl_lim = self.cl_max_at(mach) + max(0.0, boost)
        a_abs = abs(alpha)
        if a_abs <= alpha_stall:
            cl = self.cl0 + self.cl_alpha * alpha
            return clamp(cl, -cl_lim, cl_lim)
        cl_stall = self.cl0 + self.cl_alpha * alpha_stall
        cl_stall = clamp(cl_stall, -cl_lim, cl_lim)
        decay = max(floor, 1.0 - decay_rate * (a_abs - alpha_stall))
        return math.copysign(cl_stall * decay, alpha)

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
