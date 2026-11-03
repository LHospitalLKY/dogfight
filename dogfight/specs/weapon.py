"""武器规格：红外格斗弹与雷达中距弹共用的导弹总体参数。

这里只描述“弹本身”，不含挂载数量、发射平台或任何配置读取逻辑。
"""
from __future__ import annotations

import math
from dataclasses import dataclass


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
    # 导引头瞬时半视场角（锁定所需的目标-弹轴夹角上限）。
    seeker_fov: float
    # 万向支架极限：导引头能偏转到的最大离轴角，也是离轴发射/跟踪的上限
    # （缺省=锁定视场）。R-73 约 ±60°，AIM-9X 约 ±90°。
    gimbal_fov: float | None
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
    # 近炸杀伤概率的衰减系数（见 proximity_pk）：破片密度随脱靶量按
    # exp(-decay*d²) 衰减。中距弹取 2.2（引信边缘约 0.10）；近距弹
    # 战斗部相对目标更大，边缘仍有 0.35 左右的杀伤概率。
    pk_decay: float = 2.2
    # 抗干扰系数：干扰物诱骗率按它缩放。成像/双色红外导引头（AIM-9X 这类
    # IIR）能按形状把热诱弹从目标里筛掉，诱骗率明显低于早期单色导引头。
    cm_resist: float = 1.0
    # 诱导阻力系数（CDi = k * CN²，CN 以弹体横截面为参考面积）。
    # 大展弦比弹翼（R-73/AIM-9X 这类格斗弹，AR≈2.5、e≈0.85）按翼面折算
    # 到弹体面积约 k≈0.04；AIM-120/R-77 的窄小切尖弹翼 k≈0.15。
    # 旧模型对所有弹统一用 0.15，使格斗弹在大过载转进时被诱导阻力抽干，
    # 3 km 发射都追不上机动目标。
    induced_k: float = 0.15
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
    # 制导启动延迟 s：离架后弹先按惯性沿发射方向飞一小段（脱离机翼气流、
    # 舵面解锁、引信解脱），导引头指令之后才接通。真实近距弹约 0.1–0.25 s，
    # 这段延迟能把“离架瞬间就满过载”的假象消掉。
    guidance_delay: float = 0.0
    # 过载指令建立速率上限 g/s：舵机/自动驾驶仪有速率限制，横向过载是斜坡
    # 建立的。400 g/s 相当于 0.05 s 建立 20 g（约 50 Hz 舵机带宽）：既挡住
    # 一步跳满过载的非物理突变，又不会把末段响应拖慢到打不中机动目标。
    g_slew: float = 0.0
    # 燃气舵（jet vane）直接力：偏转喷流产生、不依赖动压的横向控制力，
    # 按当前推力的比例给出（15° 偏转约 0.26 T）。只有近距弹用复合控制
    # （R-73、AIM-9X）；R-77/AIM-120 这类中距弹是纯气动控制，不用燃气舵。
    tvc_frac: float = 0.0
    # 燃气舵可用窗口 s（助推段及其后一小段，之后喷流不足以提供控制力）。
    tvc_time: float = 0.0
    # 近距弹“先摆姿态、后点火”：离架后先不点主发动机，靠 RCS/燃气发生器把
    # 弹体轴转向目标，姿态对准（或超时）后才点火加速。ignition_timeout<=0
    # 表示不启用预点火段（中距弹直接点火）。
    ignition_delay: float = 0.0     # 发动机保险解脱前的最早点火时刻 s
    ignition_align: float = math.radians(20.0)  # 弹体轴对目标视线的对准门限
    rcs_slew: float = 0.0           # 预点火段 RCS 摆姿态角速度 rad/s
    ignition_timeout: float = 0.0   # 预点火最长时间 s（超时强制点火）
    # 跨声速/超声速波阻增量（相对 cd0），按 smoothstep(0.85,1.15) 打开。
    # 导弹推重比高，峰值马赫几乎完全由这一项决定；旧模型没有它，
    # 中距弹会冲到 M5 以上，比真实弹道快出一大截。
    cd0_wave: float = 0.55
    # 助推-续航两段推力：固体火箭通常先大推力短时助推，再小推力长时续航。
    # boost_time<=0 时退化为原来的“全程恒定推力”。两段总冲保持不变，
    # 因此射程量级不变，但初期加速更快、末段存速分配更接近真实弹道。
    boost_thrust: float = 0.0    # N，助推段推力
    boost_time: float = 0.0      # s，助推段时长
    # 对会机动的目标的有效射程 m（近距弹专用）。导引头/射程包线是“打不动靶”
    # 的上限，末段弹速掉到 2–5 g 时转弯半径 3–5 km，追不上一个 9 g 急转的战斗机。
    # 发射决策只认这个距离内的目标，0 表示不启用（雷达弹按原包线处理）。
    dogfight_reach: float = 0.0

    def thrust_at(self, t: float) -> float:
        """t 时刻的发动机推力（0 表示已经熄火，由燃料量另行判断）。"""
        if 0.0 < self.boost_time and t < self.boost_time:
            return self.boost_thrust
        return self.thrust

    def mass_flow_at(self, t: float) -> float:
        """t 时刻推进剂质量流 kg/s：按恒定比冲折算，与推力成正比。

        两段推力的总冲由 ∫Tdt 给出，质量流取 propellant·T(t)/∫Tdt，
        保证全部推进剂恰好在 burn_time 内烧完，弹重随时间连续下降。
        """
        impulse = (
            self.boost_thrust * self.boost_time
            + self.thrust * max(self.burn_time - self.boost_time, 0.0)
        )
        if impulse <= 0.0:
            return 0.0
        return self.propellant * self.thrust_at(t) / impulse

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
    def gimbal(self) -> float:
        """导引头万向支架能覆盖的最大离轴角。

        发射前锁定、离轴发射扇区与发射后丢锁判定都用这个极限：目标只要
        落在万向支架能指到的范围内就能被截获，弹体刚离架时还来不及指向
        目标，靠支架偏转维持跟踪。
        """
        return self.seeker_fov if self.gimbal_fov is None else self.gimbal_fov
