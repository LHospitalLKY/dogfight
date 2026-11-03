"""机体/武器规格所依赖的气动与大气基础函数。

这一层只做纯数值计算，不引用实体、世界或任何表现层模块，
因此可以被 specs、内核和外部标定脚本共用而不产生循环依赖。
"""
from __future__ import annotations

import math

G = 9.81

# ---------- 标准大气 ISA（0–32 km） ----------
# 密度随高度衰减同时影响升力、阻力和推力，是高空性能与升限的来源。
ISA_T0 = 288.15        # 海平面温度 K
ISA_P0 = 101325.0      # 海平面气压 Pa
ISA_LAPSE = -0.0065    # 对流层温度梯度 K/m
ISA_GAS = 287.053      # 干空气气体常数 J/(kg*K)
ISA_H_TROP = 11000.0   # 对流层顶
ISA_H_STRAT = 20000.0  # 平流层下界
RHO0 = 1.225           # 海平面密度 kg/m3，用于推重比等归一化

# ---------- 跨声速/超声速波阻 ----------
DRAG_CRIT_MACH = 0.78
DRAG_DD_MACH = 0.95
DRAG_PEAK_MACH = 1.05
DRAG_WIDTH = 0.34
DRAG_BUMP = 0.62
DRAG_SUP_START_MACH = 0.95
DRAG_SUP_END_MACH = 1.35
DRAG_SUPERSONIC = 0.18


def clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    """标准 C1 smoothstep；在两端函数值与一阶导数都连续。"""
    t = clamp((x - edge0) / max(edge1 - edge0, 1e-9), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


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


def cd0_at(mach: float, cd0: float) -> float:
    """计入波阻后的零升阻力系数。"""
    return cd0 * (1.0 + drag_divergence(mach))
