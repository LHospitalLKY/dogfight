#!/usr/bin/env python3
"""Headless 物理验证与简化飞行包线测试。

不加载任何 OBJ/贴图，只跑 sim.py 的数学模型和 World 的 3DOF 积分。
输出写到 results/validation/，供仓库审阅。

用法：
    python scripts/validate_models.py
"""
from __future__ import annotations

import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from main import F16V_SPEC, SU57_SPEC, R73, R77_1  # noqa: E402
from sim import (  # noqa: E402
    Action,
    G,
    MAX_BANK,
    Missile,
    TSFC_AB,
    TSFC_MIL,
    World,
    air_density,
    cd0_at,
    clamp,
    drag_divergence,
    isa_temperature,
    mach_number,
    nose_direction,
    speed_of_sound,
    thrust_lapse,
    world_velocity,
    wrap_pi,
)

OUT = ROOT / "results" / "validation"
PLOTS = OUT / "plots"


# ---------------------------------------------------------------------------
# 基础工具
# ---------------------------------------------------------------------------

def total_drag(spec, v: float, z: float, n: float = 1.0) -> float:
    """当前高度/速度/过载下的总阻力（零升 + 诱导）。"""
    q = 0.5 * air_density(z) * v * v
    if q <= 0.0 or v <= 0.0:
        return 0.0
    mach = mach_number(v, z)
    m = spec.mass
    drag0 = q * cd0_at(mach, spec.cd0) * spec.wing_area
    di = n * n * (m * G) ** 2 / (
        q * math.pi * spec.aspect_ratio * spec.oswald * spec.wing_area
    )
    return drag0 + di


def power_action(ac, n: float = 1.0, bank_rad: float = 0.0) -> Action:
    """按当前状态给出刚好配平总阻力的油门/加力（开环配平）。"""
    spec = ac.spec
    v = max(ac.v, 1.0)
    q = 0.5 * air_density(ac.z) * v * v
    mach = mach_number(v, ac.z)
    drag = total_drag(spec, v, ac.z, n)
    mil = spec.mil_thrust() * thrust_lapse(ac.z, mach)
    max_t = spec.thrust_max * thrust_lapse(ac.z, mach)
    if drag <= mil:
        throttle = drag / max(mil, 1e-9)
        ab = 0.0
    else:
        throttle = 1.0
        ab = clamp((drag - mil) / max(max_t - mil, 1e-9), 0.0, 1.0)
    return Action(
        bank=clamp(bank_rad / MAX_BANK, -1.0, 1.0),
        n=n,
        throttle=clamp(throttle, 0.0, 1.0),
        afterburner=ab,
    )


def make_world(specs, z0: float, dof: int = 3, dt: float = 1.0 / 120.0) -> World:
    """建立无边界干扰、无干扰弹的测试世界。"""
    w = World(
        specs,
        dt=dt,
        arena=1.0e9,
        time_limit=1.0e6,
        z0=z0,
        z_floor=0.0,
        z_ceiling=40000.0,
        dof=dof,
        x_offset=0.0,
        y_offset=0.0,
    )
    for ac in w.acs:
        ac.chaff = 0
        ac.flares = 0
    return w


def place(ac, x: float, y: float, z: float, psi: float, v: float) -> None:
    ac.x, ac.y, ac.z = x, y, z
    ac.psi, ac.gamma, ac.v = psi, 0.0, v
    ac.n, ac.bank = 1.0, 0.0
    ac.alive = True


# ---------------------------------------------------------------------------
# 1. 阻力曲线
# ---------------------------------------------------------------------------

def drag_metrics() -> dict:
    machs = np.arange(0.20, 2.5005, 0.001)
    inc = np.array([drag_divergence(float(m)) for m in machs])
    total = 1.0 + inc
    rel_steps = np.abs(np.diff(total) / total[:-1])
    cd_su57 = np.array([cd0_at(float(m), SU57_SPEC.cd0) for m in machs])
    cd_f16 = np.array([cd0_at(float(m), F16V_SPEC.cd0) for m in machs])
    deriv_su57 = np.gradient(cd_su57, machs)
    deriv_f16 = np.gradient(cd_f16, machs)

    peak_i = int(np.argmax(inc))
    return {
        "mach_peak_total": float(machs[peak_i]),
        "peak_relative_increment": float(inc[peak_i]),
        "increment_at_075": float(drag_divergence(0.75)),
        "jump_at_075_relative": float(
            cd0_at(0.75 + 1e-9, SU57_SPEC.cd0)
            / cd0_at(0.75 - 1e-9, SU57_SPEC.cd0)
            - 1.0
        ),
        "max_adjacent_relative_step": float(rel_steps.max()),
        "max_adjacent_step_mach": float(machs[1:][int(rel_steps.argmax())]),
        "max_dcd0_dm_su57": float(np.abs(deriv_su57).max()),
        "max_dcd0_dm_f16": float(np.abs(deriv_f16).max()),
        "increment_at_2": float(drag_divergence(2.0)),
        "samples": {
            str(round(float(m), 3)): float(drag_divergence(float(m)))
            for m in (0.75, 0.78, 0.85, 0.90, 0.95, 1.05, 1.10, 1.20, 1.35, 1.50, 2.00)
        },
    }


def plot_drag(machs, inc, su57, f16) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(machs, 1.0 + inc, lw=2)
    axes[0].axvline(0.75, color="0.6", ls="--", lw=1, label="old switch M=0.75")
    axes[0].axvline(0.78, color="tab:orange", ls=":", lw=1, label="smooth onset")
    axes[0].set_xlabel("Mach")
    axes[0].set_ylabel("C_D0 / C_D0,sub")
    axes[0].set_title("Wave-drag multiplier")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(machs, su57, label="SU-57")
    axes[1].plot(machs, f16, label="F-16V")
    axes[1].set_xlabel("Mach")
    axes[1].set_ylabel("C_D0")
    axes[1].set_title("Zero-lift drag coefficient")
    axes[1].legend()
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS / "drag_curve.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. 飞机包线与定常/能量检查
# ---------------------------------------------------------------------------

def max_ps(spec, z: float) -> tuple[float, float]:
    vmin = spec.v_stall_at(z)
    best_ps, best_v = -1e9, vmin
    for v in np.arange(vmin, 850.0, 2.0):
        ps = spec.specific_power(float(v), z, 1.0, 1.0, spec.mass, 1.0)
        if ps > best_ps:
            best_ps, best_v = float(ps), float(v)
    return best_ps, best_v


def max_level_speed(spec, z: float) -> float | None:
    vmin = spec.v_stall_at(z)
    prev = None
    for v in np.arange(vmin, 900.0, 1.0):
        thrust = spec.thrust_at(1.0, 1.0, z, mach_number(float(v), z))
        f = thrust - total_drag(spec, float(v), z, 1.0)
        if prev is not None and prev > 0.0 >= f:
            return float(v)
        prev = f
    return 900.0 if prev is not None and prev > 0.0 else None


def aircraft_envelope() -> list[dict]:
    rows = []
    for spec in (SU57_SPEC, F16V_SPEC):
        for z in (1000, 3000, 5000, 7000, 9000, 11000, 13000, 15000, 18000):
            vmin = spec.v_stall_at(z)
            vmax = max_level_speed(spec, z)
            ps, ps_v = max_ps(spec, z)
            v08 = 0.8 * speed_of_sound(z)
            ns = spec.n_sustain(v08, z, 1.0, spec.mass, 1.0)
            turn = G * math.sqrt(max(ns * ns - 1.0, 0.0)) / v08
            rows.append({
                "spec": spec.name,
                "altitude_m": z,
                "v_stall_mps": vmin,
                "v_max_mps": vmax,
                "mach_max": (vmax / speed_of_sound(z)) if vmax else None,
                "ps_max_mps": ps,
                "v_at_ps_max": ps_v,
                "n_sustain_m08": ns,
                "turn_rate_m08_deg_s": math.degrees(turn),
                "turn_radius_m08_m": (v08 / turn) if turn > 0 else None,
            })
    return rows


def service_ceiling(spec) -> tuple[float, float]:
    """Ps_max 首次跌破 0 的高度（实用升限）与该高度处的 Ps。

    扫描到 32 km（ISA 上限），避免把循环边界误当成升限；
    100 m 网格 + 线性插值得到过零点。
    """
    prev_z, prev_ps = 0.0, None
    for z in range(0, 32001, 100):
        ps, _ = max_ps(spec, float(z))
        if ps <= 0.0:
            if prev_ps is None:
                return 0.0, ps
            # 在 [prev_z, z] 之间线性插值出 Ps=0 的高度。
            frac = prev_ps / max(prev_ps - ps, 1e-9)
            return prev_z + frac * (z - prev_z), prev_ps
        prev_z, prev_ps = float(z), ps
    return float(prev_z), float(prev_ps)


def level_case(spec, z: float, mach: float, duration: float = 20.0,
               dt: float = 1.0 / 120.0) -> dict:
    v0 = mach * speed_of_sound(z)
    w = make_world([spec, spec], z, dt=dt)
    a, b = w.acs
    place(a, 0.0, 0.0, z, 0.0, v0)
    place(b, 300000.0, 0.0, z, 0.0, v0)
    samples = []
    for k in range(int(duration / w.dt)):
        w.step([power_action(a), Action(n=1.0, throttle=0.0)])
        if k % 60 == 0:
            samples.append((w.t, a.v, a.z, a.throttle, a.ab))
    v_end = a.v
    return {
        "spec": spec.name, "altitude_m": z, "mach_target": mach,
        "v_target_mps": v0, "v_end_mps": v_end,
        "v_drift_mps": v_end - v0,
        "max_v_deviation_mps": max(abs(v - v0) for _, v, *_ in samples),
        "altitude_drift_m": a.z - z,
        "samples": samples,
        "passed": abs(v_end - v0) < 2.0
        and max(abs(v - v0) for _, v, *_ in samples) < 5.0
        and abs(a.z - z) < 1e-6,
    }


def turn_case(spec, z: float, mach: float, bank_deg: float, duration: float = 20.0,
              dof: int = 3, dt: float = 1.0 / 120.0) -> dict:
    """定常盘旋：指令 n=1/cos(bank)，检验转率与高度保持。

    过载与滚转各自有一阶惯性（tau=0.45 s）：两者同时爬升的初期
    `n·cos(bank) > 1`，会先爬升一点；收敛后 γ=0 虽是平衡点，但偏离后
    γ̇ = g/V(n cosφ - cosγ) 会单调放大（点质量模型没有高度保持回路）。
    因此：

    - dof=2：垂直自由度被移除，高度恒定，用来严格检验水平转率公式；
    - dof=3：用于确认垂直耦合不发散，高度按开环工程容差判定，
      瞬态掉高/爬高单独记录，不作为模型误差。
    """
    bank = math.radians(bank_deg)
    n = 1.0 / math.cos(bank)
    v0 = mach * speed_of_sound(z)
    w = make_world([spec, spec], z, dof=dof, dt=dt)
    a, b = w.acs
    place(a, 0.0, 0.0, z, 0.0, v0)
    place(b, 300000.0, 0.0, z, 0.0, v0)
    half_psi = 0.0
    v_sum = 0.0
    count = 0
    z_half = None
    for k in range(int(duration / w.dt)):
        prev = a.psi
        w.step([power_action(a, n, bank), Action(n=1.0, throttle=0.0)])
        dpsi = wrap_pi(a.psi - prev)
        if w.t >= duration / 2.0:
            if z_half is None:
                z_half = a.z
            half_psi += dpsi
        v_sum += a.v
        count += 1
    measured = half_psi / (duration / 2.0)
    v_avg = v_sum / max(count, 1)
    analytic = G * math.sqrt(max(n * n - 1.0, 0.0)) / v_avg
    transient_drift = (z_half - z) if z_half is not None else 0.0
    steady_drift = a.z - (z_half if z_half is not None else z)
    drift_limit = 1e-6 if dof == 2 else 80.0
    return {
        "spec": spec.name, "altitude_m": z, "mach_target": mach, "dof": dof,
        "bank_deg": bank_deg, "n_cmd": n, "v_avg_mps": v_avg,
        "turn_rate_measured_deg_s": math.degrees(measured),
        "turn_rate_analytic_deg_s": math.degrees(analytic),
        "turn_rate_error_pct": abs(measured - analytic) / max(analytic, 1e-9) * 100.0,
        "radius_measured_m": v_avg / max(measured, 1e-9),
        "altitude_drift_m": a.z - z,
        "transient_altitude_drift_m": transient_drift,
        "steady_altitude_drift_m": steady_drift,
        "drift_limit_m": drift_limit,
        "passed": abs(measured - analytic) / max(analytic, 1e-9) < 0.05
        and abs(a.z - z) < drift_limit,
    }


def energy_case(spec, z: float, mach: float, duration: float = 8.0) -> dict:
    v0 = mach * speed_of_sound(z)
    w = make_world([spec, spec], z)
    a, b = w.acs
    place(a, 0.0, 0.0, z, 0.0, v0)
    place(b, 300000.0, 0.0, z, 0.0, v0)
    e0 = 0.5 * v0 * v0 + G * z
    loss = 0.0
    for _ in range(int(duration / w.dt)):
        w.step([Action(n=1.0, throttle=0.0, afterburner=0.0),
                Action(n=1.0, throttle=0.0)])
        loss += total_drag(spec, a.v, a.z, 1.0) * a.v / max(a.mass, 1.0) * w.dt
    e1 = 0.5 * a.v * a.v + G * a.z
    return {
        "spec": spec.name, "altitude_m": z, "mach_target": mach,
        "E0_j_kg": e0, "E1_j_kg": e1,
        "delta_E_j_kg": e0 - e1, "drag_loss_j_kg": loss,
        "relative_error_pct": abs((e0 - e1) - loss) / max(loss, 1e-9) * 100.0,
        "passed": abs((e0 - e1) - loss) / max(loss, 1e-9) < 0.02,
    }


# ---------------------------------------------------------------------------
# 3. 大气、爬升、燃油与数值收敛
# ---------------------------------------------------------------------------

# ISA 标准值（0–20 km，公开标准表）。20 km 以上由同一分段公式外推，
# 不同版本标准表在 1–2% 量级上存在差异，因此只作信息记录、不参与判定。
ISA_REF = [
    (0.0, 1.22500, 288.15),
    (5000.0, 0.73643, 255.65),
    (11000.0, 0.36392, 216.65),
    (15000.0, 0.19367, 216.65),
    (20000.0, 0.08803, 216.65),
]


def isa_check() -> dict:
    rows = []
    for z, rho_ref, t_ref in ISA_REF:
        rho, t = air_density(z), isa_temperature(z)
        rows.append({
            "altitude_m": z,
            "rho_model": rho, "rho_ref": rho_ref,
            "rho_err_pct": abs(rho / rho_ref - 1.0) * 100.0,
            "temp_model": t, "temp_ref": t_ref,
            "temp_err_pct": abs(t / t_ref - 1.0) * 100.0,
        })
    # 11 km / 20 km 两个分段交界处，两侧极限应连续。
    seam = []
    for z in (11000.0, 20000.0):
        a = air_density(z - 1e-6)
        b = air_density(z + 1e-6)
        seam.append({"altitude_m": z, "relative_jump": abs(b / a - 1.0)})
    return {
        "rows": rows,
        "seam": seam,
        "max_rho_err_pct": max(r["rho_err_pct"] for r in rows),
        "max_temp_err_pct": max(r["temp_err_pct"] for r in rows),
        "max_seam_jump": max(s["relative_jump"] for s in seam),
    }


def climb_case(spec, z: float = 9000.0, mach: float = 0.90,
               duration: float = 30.0) -> dict:
    """全加力加速段的能量率一致性（A06/A12）。

    3DOF 点质量模型只接受 (n, bank, throttle)，没有航迹角保持：
    1 g 平飞时 `γ̇=g/V(n cosφ−cosγ)=0`，开环指令无法“爬升”，
    所以这里检验能量率而不是爬升率——全加力平飞加速段应满足
    `dE/dt = V·V̇ = g·Ps`（质量随耗油变化，用区间平均质量）。
    """
    v0 = mach * speed_of_sound(z)
    w = make_world([spec, spec], z)
    a, b = w.acs
    place(a, 0.0, 0.0, z, 0.0, v0)
    place(b, 300000.0, 0.0, z, 0.0, v0)
    e_half = None
    for _ in range(int(duration / w.dt)):
        w.step([
            Action(n=1.0, throttle=1.0, afterburner=1.0),
            Action(n=1.0, throttle=0.0),
        ])
        if e_half is None and w.t >= duration / 2.0:
            e_half = 0.5 * a.v * a.v + G * a.z
            v_half, z_half = a.v, a.z
    half_t = duration / 2.0
    e_end = 0.5 * a.v * a.v + G * a.z
    de_dt = (e_end - e_half) / half_t
    v_dot = (a.v - v_half) / half_t
    v_avg = 0.5 * (a.v + v_half)
    ps = spec.specific_power(v_avg, 0.5 * (a.z + z_half), 1.0, 1.0, a.mass, 1.0)
    expected = G * ps
    return {
        "spec": spec.name, "altitude_m": z, "mach_target": mach,
        "mach_end": mach_number(a.v, a.z),
        "v_dot_mps2": v_dot,
        "dE_dt_measured": de_dt, "g_ps_expected": expected,
        "roc_measured_mps": (a.z - z_half) / half_t,
        "ps_mps": ps, "roc_expected_mps": ps - v_avg * v_dot / G,
        "error_pct": abs(de_dt - expected) / max(abs(expected), 1e-9) * 100.0,
        "passed": abs(de_dt - expected) / max(abs(expected), 1e-9) < 0.10,
    }


def fuel_case(spec, z: float = 9000.0, mach: float = 0.90,
              duration: float = 20.0) -> dict:
    """全加力平飞：质量减少与 TSFC 积分对照（A13）。"""
    v0 = mach * speed_of_sound(z)
    w = make_world([spec, spec], z)
    a, b = w.acs
    place(a, 0.0, 0.0, z, 0.0, v0)
    place(b, 300000.0, 0.0, z, 0.0, v0)
    m0 = a.mass
    predicted = 0.0
    mil = spec.mil_thrust()
    for _ in range(int(duration / w.dt)):
        lapse = thrust_lapse(a.z, mach_number(a.v, a.z))
        dry = mil * lapse
        wet = (spec.thrust_max - mil) * lapse
        predicted += (dry * TSFC_MIL + wet * TSFC_AB) * w.dt
        w.step([
            Action(n=1.0, throttle=1.0, afterburner=1.0),
            Action(n=1.0, throttle=0.0),
        ])
    actual = m0 - a.mass
    return {
        "spec": spec.name, "altitude_m": z, "mach_target": mach,
        "fuel_burn_kg": actual, "predicted_burn_kg": predicted,
        "relative_error_pct": abs(actual - predicted) / max(predicted, 1e-9) * 100.0,
        "fuel_left_kg": a.fuel,
        "passed": abs(actual - predicted) / max(predicted, 1e-9) < 0.01,
    }


def dt_convergence() -> dict:
    """步长收敛（§7.2）：平飞末速与导弹最近距离随 dt 减半的变化。"""
    dts = (1.0 / 60.0, 1.0 / 120.0, 1.0 / 240.0)
    levels = [level_case(F16V_SPEC, 9000.0, 0.80, duration=20.0, dt=d) for d in dts]
    base_v = levels[1]["v_end_mps"]
    level_rows = [{
        "dt": d, "v_end_mps": r["v_end_mps"],
        "delta_vs_120_mps": r["v_end_mps"] - base_v,
        "relative_pct": abs(r["v_end_mps"] - base_v) / base_v * 100.0,
    } for d, r in zip(dts, levels)]
    missiles = [
        missile_case(R77_1, 20000.0, 9000.0, 280.0, 280.0, 1.0, dt=d)
        for d in dts
    ]
    base_d = missiles[1]["min_distance_m"]
    base_t = missiles[1]["flight_time_s"]
    base_end = missiles[1]["ending"]
    missile_rows = [{
        "dt": d, "min_distance_m": r["min_distance_m"],
        "delta_vs_120_m": r["min_distance_m"] - base_d,
        "relative_pct": abs(r["min_distance_m"] - base_d) / max(base_d, 1e-9) * 100.0,
        "flight_time_s": r["flight_time_s"],
        "flight_time_delta_pct": abs(r["flight_time_s"] - base_t) / max(base_t, 1e-9) * 100.0,
        "ending": r["ending"],
        "same_ending": r["ending"] == base_end,
    } for d, r in zip(dts, missiles)]
    level_ok = max(r["relative_pct"] for r in level_rows) < 2.0
    # 弹道本身（弹道时间）应随 dt 收敛；近炸引信是离散触发 + 概率掷点，
    # 在 d_miss≈战斗部半径时命中/脱失会随 dt 翻转，单独记录、不并入收敛判据。
    missile_ok = all(r["flight_time_delta_pct"] < 5.0 for r in missile_rows)
    outcome_stable = all(r["same_ending"] for r in missile_rows)
    return {
        "level": level_rows, "missile": missile_rows,
        "level_ok": level_ok, "missile_ok": missile_ok,
        "outcome_stable": outcome_stable,
        "passed": level_ok and missile_ok,
    }


# ---------------------------------------------------------------------------
# 4. 导弹包线
# ---------------------------------------------------------------------------

def _segment_min_distance(r0, r1) -> float:
    """相对位置从 r0 线性变化到 r1 时，到原点的最小距离（亚步长最近点）。"""
    dx, dy, dz = r1[0] - r0[0], r1[1] - r0[1], r1[2] - r0[2]
    l2 = dx * dx + dy * dy + dz * dz
    if l2 <= 0.0:
        return math.sqrt(r1[0] ** 2 + r1[1] ** 2 + r1[2] ** 2)
    t = clamp(-(r0[0] * dx + r0[1] * dy + r0[2] * dz) / l2, 0.0, 1.0)
    return math.sqrt(
        (r0[0] + t * dx) ** 2 + (r0[1] + t * dy) ** 2 + (r0[2] + t * dz) ** 2
    )


def missile_case(spec, launch_range_m: float, alt: float,
                 shooter_v: float, target_v: float,
                 target_g: float = 1.0, max_time: float | None = None,
                 dt: float = 1.0 / 120.0) -> dict:
    w = make_world([F16V_SPEC, SU57_SPEC], alt, dt=dt)
    shooter, target = w.acs
    place(shooter, 0.0, 0.0, alt, 0.0, shooter_v)
    place(target, launch_range_m, 0.0, alt, math.pi, target_v)
    target.hp = 100000.0
    hp0 = target.hp

    nx, ny, nz = nose_direction(shooter)
    vx, vy, vz = world_velocity(shooter)
    off = shooter.spec.length * 0.5
    m = Missile(
        spec=spec, owner=0, target=1,
        x=shooter.x + nx * off, y=shooter.y + ny * off, z=shooter.z + nz * off,
        vx=vx + nx * 45.0, vy=vy + ny * 45.0, vz=vz + nz * 45.0,
        mass=spec.mass, fuel=spec.propellant,
        nx=nx, ny=ny, nz=nz,
        roll_seed=int(launch_range_m) % 1000003 + 17,
    )
    w.missiles.append(m)

    bank = math.acos(clamp(1.0 / target_g, 0.0, 1.0)) if target_g > 1.0 else 0.0
    min_dist = math.inf
    prev_rel = None
    limit = max_time if max_time is not None else spec.life + 2.0
    steps = int(limit / w.dt)
    for _ in range(steps):
        w.step([
            power_action(shooter, 1.0, 0.0),
            power_action(target, target_g, bank),
        ])
        rel = (target.x - m.x, target.y - m.y, target.z - m.z)
        if prev_rel is None:
            min_dist = min(min_dist, math.sqrt(sum(r * r for r in rel)))
        else:
            min_dist = min(min_dist, _segment_min_distance(prev_rel, rel))
        prev_rel = rel
        if not m.alive:
            break

    hit = target.hp < hp0
    if hit:
        ending = "hit"
    elif not m.alive:
        ending = "miss/fuse"
    else:
        ending = "timeout"
    return {
        "missile": spec.name,
        "launch_range_km": launch_range_m / 1000.0,
        "altitude_m": alt,
        "shooter_v_mps": shooter_v,
        "target_v_mps": target_v,
        "target_g": target_g,
        "hit": hit,
        "ending": ending,
        "min_distance_m": min_dist,
        "flight_time_s": m.t,
        "final_missile_speed_mps": m.speed if m.alive else 0.0,
        "damage": hp0 - target.hp,
    }


def missile_envelope() -> list[dict]:
    cases = []
    for rng in (1000, 2000, 3000, 5000, 8000, 10000, 12000):
        cases.append(missile_case(R73, rng, 5000.0, 250.0, 250.0, 1.0))
    cases.append(missile_case(R73, 5000.0, 5000.0, 250.0, 250.0, 5.0))
    for rng in (5000, 10000, 20000, 30000, 40000, 50000, 60000):
        cases.append(missile_case(R77_1, rng, 9000.0, 280.0, 280.0, 1.0))
    cases.append(missile_case(R77_1, 30000.0, 9000.0, 280.0, 280.0, 5.0))
    return cases


# ---------------------------------------------------------------------------
# 4. 出图、落盘与 headless 快速验证
# ---------------------------------------------------------------------------

def plot_aircraft(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for spec in (SU57_SPEC, F16V_SPEC):
        rs = [r for r in rows if r["spec"] == spec.name]
        h = [r["altitude_m"] / 1000.0 for r in rs]
        axes[0].plot(h, [r["v_stall_mps"] for r in rs], marker="o", label=f"{spec.name} Vmin")
        axes[0].plot(h, [r["v_max_mps"] for r in rs], marker="s", label=f"{spec.name} Vmax")
        axes[1].plot(h, [r["ps_max_mps"] for r in rs], marker="o", label=spec.name)
        axes[2].plot(h, [r["n_sustain_m08"] for r in rs], marker="o", label=spec.name)
    axes[0].set_xlabel("Altitude (km)")
    axes[0].set_ylabel("Speed (m/s)")
    axes[0].set_title("Level-flight envelope")
    axes[1].set_xlabel("Altitude (km)")
    axes[1].set_ylabel("Ps_max (m/s)")
    axes[1].set_title("Specific excess power")
    axes[2].set_xlabel("Altitude (km)")
    axes[2].set_ylabel("n_sustain at M0.8")
    axes[2].set_title("Sustained load factor")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "aircraft_envelope.png", dpi=150)
    plt.close(fig)


def plot_missiles(cases: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, marker in (("R-73", "o"), ("R-77-1", "s")):
        rs = [r for r in cases if r["missile"] == name and r["target_g"] == 1.0]
        if not rs:
            continue
        ax.plot([r["launch_range_km"] for r in rs],
                [r["min_distance_m"] for r in rs],
                marker=marker, label=f"{name}, target 1g")
        for r in rs:
            if r["hit"]:
                ax.scatter(r["launch_range_km"], r["min_distance_m"],
                           marker="x", s=50, color="tab:green")
    ax.set_yscale("log")
    ax.set_xlabel("Launch range (km)")
    ax.set_ylabel("Minimum missile-target distance (m)")
    ax.set_title("Missile fly-out: distance and hit/miss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "missile_envelope.png", dpi=150)
    plt.close(fig)


def run_headless(seconds: float, path: Path) -> dict:
    cmd = [
        sys.executable, str(ROOT / "main.py"),
        "--headless", "--seconds", str(seconds), "--seed", "0",
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, timeout=240,
        )
        log = proc.stdout + proc.stderr
        path.write_text(log, encoding="utf-8")
        return {
            "seconds": seconds,
            "returncode": proc.returncode,
            "has_traceback": "Traceback" in log,
            "mentions_obj": ".obj" in log.lower(),
            "log": str(path.relative_to(ROOT)),
        }
    except subprocess.TimeoutExpired:
        path.write_text("TIMEOUT\n", encoding="utf-8")
        return {"seconds": seconds, "returncode": None, "timeout": True}


def write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    import csv
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fields})


def determinism_check() -> dict:
    """同参数、同 seed 跑两次，输出应逐字节一致（§7.1）。"""
    pa = OUT / "headless_det_a.log"
    pb = OUT / "headless_det_b.log"
    ra = run_headless(5.0, pa)
    rb = run_headless(5.0, pb)
    ta = pa.read_text(encoding="utf-8")
    tb = pb.read_text(encoding="utf-8")
    identical = ra["returncode"] == 0 and rb["returncode"] == 0 and ta == tb
    return {
        "identical": identical,
        "log_bytes": len(ta.encode("utf-8")),
        "passed": identical,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)

    machs = np.arange(0.20, 2.5005, 0.001)
    inc = np.array([drag_divergence(float(m)) for m in machs])
    su57_cd = np.array([cd0_at(float(m), SU57_SPEC.cd0) for m in machs])
    f16_cd = np.array([cd0_at(float(m), F16V_SPEC.cd0) for m in machs])
    plot_drag(machs, inc, su57_cd, f16_cd)

    drag = drag_metrics()
    write_csv(
        [{"mach": float(m), "relative_increment": float(i),
          "cd0_su57": float(a), "cd0_f16": float(b)}
         for m, i, a, b in zip(machs, inc, su57_cd, f16_cd)],
        OUT / "drag_curve.csv",
        ["mach", "relative_increment", "cd0_su57", "cd0_f16"],
    )

    env = aircraft_envelope()
    write_csv(
        env, OUT / "aircraft_envelope.csv",
        ["spec", "altitude_m", "v_stall_mps", "v_max_mps", "mach_max",
         "ps_max_mps", "v_at_ps_max", "n_sustain_m08",
         "turn_rate_m08_deg_s", "turn_radius_m08_m"],
    )
    plot_aircraft(env)

    dynamic = [
        level_case(SU57_SPEC, 9000.0, 0.95),
        level_case(F16V_SPEC, 9000.0, 0.80),
        level_case(F16V_SPEC, 3000.0, 0.65),
        turn_case(F16V_SPEC, 9000.0, 0.80, 45.0),
        turn_case(F16V_SPEC, 9000.0, 0.80, 60.0),
        turn_case(F16V_SPEC, 9000.0, 0.80, 60.0, dof=2),
        energy_case(F16V_SPEC, 9000.0, 0.90),
    ]
    write_csv(
        dynamic, OUT / "dynamic_checks.csv",
        ["spec", "altitude_m", "mach_target", "dof", "v_target_mps", "v_end_mps",
         "v_drift_mps", "max_v_deviation_mps", "altitude_drift_m",
         "bank_deg", "n_cmd", "v_avg_mps", "turn_rate_measured_deg_s",
         "turn_rate_analytic_deg_s", "turn_rate_error_pct",
         "radius_measured_m", "transient_altitude_drift_m",
         "steady_altitude_drift_m", "drift_limit_m", "E0_j_kg", "E1_j_kg",
         "delta_E_j_kg", "drag_loss_j_kg", "relative_error_pct", "passed"],
    )

    missiles = missile_envelope()
    write_csv(
        missiles, OUT / "missile_envelope.csv",
        ["missile", "launch_range_km", "altitude_m", "shooter_v_mps",
         "target_v_mps", "target_g", "hit", "ending", "min_distance_m",
         "flight_time_s", "final_missile_speed_mps", "damage"],
    )
    plot_missiles(missiles)

    isa = isa_check()
    write_csv(
        isa["rows"], OUT / "isa_check.csv",
        ["altitude_m", "rho_model", "rho_ref", "rho_err_pct",
         "temp_model", "temp_ref", "temp_err_pct"],
    )
    climbs = [climb_case(SU57_SPEC), climb_case(F16V_SPEC)]
    write_csv(
        climbs, OUT / "climb_checks.csv",
        ["spec", "altitude_m", "mach_target", "mach_end", "v_dot_mps2",
         "dE_dt_measured", "g_ps_expected", "roc_measured_mps", "ps_mps",
         "roc_expected_mps", "error_pct", "passed"],
    )
    fuels = [fuel_case(SU57_SPEC), fuel_case(F16V_SPEC)]
    write_csv(
        fuels, OUT / "fuel_checks.csv",
        ["spec", "altitude_m", "mach_target", "fuel_burn_kg",
         "predicted_burn_kg", "relative_error_pct", "fuel_left_kg", "passed"],
    )
    conv = dt_convergence()
    write_csv(
        [dict(r, kind="level") for r in conv["level"]]
        + [dict(r, kind="missile") for r in conv["missile"]],
        OUT / "dt_convergence.csv",
        ["kind", "dt", "v_end_mps", "min_distance_m",
         "delta_vs_120_mps", "delta_vs_120_m", "relative_pct",
         "flight_time_s", "flight_time_delta_pct", "ending", "same_ending"],
    )
    det = determinism_check()

    smoke = run_headless(5.0, OUT / "headless_5s.log")
    full = run_headless(60.0, OUT / "headless_60s.log")

    ceilings = {spec.name: service_ceiling(spec) for spec in (SU57_SPEC, F16V_SPEC)}
    metrics = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
            capture_output=True, text=True,
        ).stdout.strip(),
        "drag": drag,
        "service_ceiling_m": {k: v[0] for k, v in ceilings.items()},
        "service_ceiling_ps_mps": {k: v[1] for k, v in ceilings.items()},
        "dynamic": dynamic,
        "missiles": missiles,
        "isa": isa,
        "climb": climbs,
        "fuel": fuels,
        "dt_convergence": conv,
        "determinism": det,
        "headless": {"smoke": smoke, "full": full},
        "pass": {
            "drag_continuous": drag["jump_at_075_relative"] < 1e-9
            and drag["max_adjacent_relative_step"] < 0.005,
            "dynamic_all": all(r["passed"] for r in dynamic),
            "isa_ok": isa["max_rho_err_pct"] < 0.5
            and isa["max_temp_err_pct"] < 0.5
            and isa["max_seam_jump"] < 0.001,
            "climb_ok": all(r["passed"] for r in climbs),
            "fuel_ok": all(r["passed"] for r in fuels),
            "dt_convergence": conv["passed"],
            "deterministic": det["passed"],
            "headless_smoke": smoke["returncode"] == 0
            and not smoke["has_traceback"] and not smoke["mentions_obj"],
            "headless_full": full["returncode"] == 0
            and not full["has_traceback"] and not full["mentions_obj"],
        },
    }
    (OUT / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8",
    )

    lines = []
    lines.append("# 物理模型验证结果")
    lines.append("")
    lines.append(f"- 生成时间（UTC）：`{metrics['generated_at']}`")
    lines.append(f"- Python：`{metrics['python']}`")
    lines.append(f"- 提交：`{metrics['git_commit']}`")
    lines.append("- 测试模式：headless，不加载 OBJ/贴图；只跑 sim.py 与 World 的 2DOF/3DOF 积分。")
    lines.append("")
    lines.append("## 1. 跨声速阻力连续性")
    lines.append("")
    lines.append("| 指标 | 数值 |")
    lines.append("| --- | --- |")
    lines.append(f"| M=0.75 处相对跳变 | {drag['jump_at_075_relative']:.6%} |")
    lines.append(f"| M=0.75 波阻增量 | {drag['increment_at_075']:.6f} |")
    lines.append(f"| 相邻 0.001 M 最大相对变化 | {drag['max_adjacent_relative_step']:.4%} |")
    lines.append(f"| 最大相对变化位置 | M={drag['max_adjacent_step_mach']:.3f} |")
    lines.append(f"| 总增量峰值 | {drag['peak_relative_increment']:.4f} @ M={drag['mach_peak_total']:.3f} |")
    lines.append(f"| M=2.0 持续波阻增量 | {drag['increment_at_2']:.4f} |")
    lines.append(f"| 最大 dCD0/dM（SU-57 / F-16V） | {drag['max_dcd0_dm_su57']:.3f} / {drag['max_dcd0_dm_f16']:.3f} |")
    lines.append("")
    lines.append(f"判定：`{'PASS' if metrics['pass']['drag_continuous'] else 'FAIL'}`（无 M=0.75 阶跃，扫描无 >0.5% 的相邻跳变）")
    lines.append("")
    lines.append("## 2. 飞机动态检查")
    lines.append("")
    lines.append("| 工况 | 结果 | 判据 | 判定 |")
    lines.append("| --- | --- | --- | --- |")
    for r in dynamic:
        if "bank_deg" in r:
            detail = (f"{r['dof']}DOF，{r['bank_deg']:.0f}° 盘旋："
                      f"ω={r['turn_rate_measured_deg_s']:.2f}°/s，"
                      f"解析={r['turn_rate_analytic_deg_s']:.2f}°/s，"
                      f"误差={r['turn_rate_error_pct']:.2f}%，"
                      f"|Δz|={abs(r['altitude_drift_m']):.2f} m"
                      f"（限 {r['drift_limit_m']:.0f} m）")
            criterion = "ω 误差 <5%，高度偏差见括号"
        elif "delta_E_j_kg" in r:
            detail = (f"ΔE={r['delta_E_j_kg']:.0f} J/kg，"
                      f"阻力损失={r['drag_loss_j_kg']:.0f} J/kg，"
                      f"误差={r['relative_error_pct']:.2f}%")
            criterion = "能量收支误差 <2%"
        else:
            detail = (f"v_end={r['v_end_mps']:.2f} m/s，"
                      f"漂移={r['v_drift_mps']:+.2f} m/s，"
                      f"最大偏差={r['max_v_deviation_mps']:.2f} m/s")
            criterion = "速度漂移 <2 m/s，最大偏差 <5 m/s"
        lines.append(f"| {r['spec']} / {r.get('altitude_m', 0)/1000:.1f} km | {detail} | {criterion} | {'PASS' if r['passed'] else 'FAIL'} |")
    lines.append("")
    lines.append(f"判定：`{'PASS' if metrics['pass']['dynamic_all'] else 'FAIL'}`")
    lines.append("")
    lines.append(
        "说明：2DOF 盘旋移除垂直自由度，高度严格恒定，用来检验水平转率公式；"
        "3DOF 盘旋是开环定 `n=1/cosφ` 指令、没有高度保持回路，γ 偏离 0 后 "
        "`γ̇=g/V(n cosφ−cosγ)` 会单调放大，因此高度偏差按 80 m 工程容差判定，"
        "瞬态与稳态数值见 `dynamic_checks.csv`。"
    )
    lines.append("")
    lines.append("## 3. 飞机包线（由当前代码公式计算）")
    lines.append("")
    for name, (z_ceil, ps_ceil) in ceilings.items():
        lines.append(
            f"- {name} 实用升限（Ps_max 过零，100 m 网格 + 线性插值）："
            f"约 {z_ceil/1000:.2f} km（过零点上方最近一格的 Ps={ps_ceil:.2f} m/s）"
        )
    lines.append("- 完整高度-速度表见 `aircraft_envelope.csv`，图见 `plots/aircraft_envelope.png`。")
    lines.append(
        "- 说明：两机升限接近是推力模型的直接结果——`thrust_lapse` 只按密度衰减并叠"
        "冲压恢复，两机推重比又在同一量级（约 1.1），所以高空的 Ps 过零点几乎重合。"
        "这是模型已知的标定缺口，不能当作真实机型升限。"
    )
    lines.append("")
    lines.append("| 机型 | 高度 km | V_min m/s | V_max m/s | M_max | Ps_max m/s | M0.8 维持过载 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in env:
        if r["altitude_m"] in (3000, 9000, 15000):
            vmax = "-" if r["v_max_mps"] is None else f"{r['v_max_mps']:.0f}"
            mmax = "-" if r["mach_max"] is None else f"{r['mach_max']:.2f}"
            lines.append(f"| {r['spec']} | {r['altitude_m']/1000:.0f} | {r['v_stall_mps']:.0f} | {vmax} | {mmax} | {r['ps_max_mps']:.1f} | {r['n_sustain_m08']:.2f} |")
    lines.append("")
    lines.append("## 4. 大气、爬升、燃油与步长收敛")
    lines.append("")
    lines.append("### 4.1 ISA 大气（0–20 km 对照标准表）")
    lines.append("")
    lines.append("| 高度 m | ρ 模型 | ρ 标准 | 误差 | T 模型 | T 标准 | 误差 |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for r in isa["rows"]:
        lines.append(
            f"| {r['altitude_m']:.0f} | {r['rho_model']:.5f} | {r['rho_ref']:.5f} "
            f"| {r['rho_err_pct']:.3f}% | {r['temp_model']:.2f} | {r['temp_ref']:.2f} "
            f"| {r['temp_err_pct']:.3f}% |"
        )
    lines.append("")
    lines.append(
        f"分段交界（11 / 20 km）最大相对跳变：{isa['max_seam_jump']:.2e}。"
        f"判定：`{'PASS' if metrics['pass']['isa_ok'] else 'FAIL'}`"
        "（密度/温度误差 <0.5%，交界连续）"
    )
    lines.append("")
    lines.append(
        "说明：20 km 以上按同一分段公式（20 km 起 0.001 K/m 升温层）外推，"
        "部分公开标准表在该段与本实现有 1–2% 差异，故不参与判定。"
    )
    lines.append("")
    lines.append("### 4.2 能量率与 Ps、燃油与 TSFC")
    lines.append("")
    lines.append("| 检查 | 机型 | 实测 | 解析/预测 | 误差 | 判定 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in climbs:
        lines.append(
            f"| A06/A12 能量率（9 km、M0.9 起全加力加速） | {r['spec']} "
            f"| dE/dt={r['dE_dt_measured']:.1f} J/(kg·s) "
            f"| g·Ps={r['g_ps_expected']:.1f} J/(kg·s)"
            f"（V̇={r['v_dot_mps2']:.2f} m/s²，M→{r['mach_end']:.2f}） "
            f"| {r['error_pct']:.2f}% | {'PASS' if r['passed'] else 'FAIL'} |"
        )
    for r in fuels:
        lines.append(
            f"| A13 耗油（9 km、全加力 20 s） | {r['spec']} "
            f"| {r['fuel_burn_kg']:.2f} kg | {r['predicted_burn_kg']:.2f} kg "
            f"| {r['relative_error_pct']:.2f}% | {'PASS' if r['passed'] else 'FAIL'} |"
        )
    lines.append("")
    lines.append(
        "说明：3DOF 只接受 `(n, bank, throttle)`，没有航迹角保持；1 g 指令下 "
        "`γ̇=g/V(n cosφ−cosγ)=0`，开环无法“匀速爬升”，所以用全加力加速段检验 "
        "`dE/dt=g·Ps`。爬升率与 Ps 的换算是同一恒等式的另一种写法。"
    )
    lines.append("")
    lines.append(
        "### 4.3 步长收敛（dt = 1/60、1/120、1/240）"
    )
    lines.append("")
    lines.append("| 场景 | dt | 指标 | 与 1/120 的差 |")
    lines.append("| --- | --- | --- | --- |")
    for r in conv["level"]:
        lines.append(
            f"| F-16V 9 km M0.8 平飞 20 s | 1/{round(1/r['dt'])} "
            f"| v_end={r['v_end_mps']:.4f} m/s | {r['delta_vs_120_mps']:+.2e} m/s "
            f"（{r['relative_pct']:.4f}%） |"
        )
    for r in conv["missile"]:
        lines.append(
            f"| R-77-1 20 km 迎头 | 1/{round(1/r['dt'])} "
            f"| T={r['flight_time_s']:.3f} s，{r['ending']} "
            f"（d_miss={r['min_distance_m']:.2f} m） "
            f"| ΔT={r['flight_time_delta_pct']:.3f}% |"
        )
    lines.append("")
    lines.append(
        f"判定：`{'PASS' if metrics['pass']['dt_convergence'] else 'FAIL'}`"
        "（平飞末速变化 <2%；导弹弹道时间变化 <5%）"
    )
    lines.append("")
    dt_desc = "、".join(
        f"1/{round(1/r['dt'])}→{r['ending']}" for r in conv["missile"]
    )
    d_desc = " / ".join(f"{r['min_distance_m']:.1f}" for r in conv["missile"])
    t_base = conv["missile"][1]["flight_time_s"]
    lines.append(
        "说明：近炸引信是离散触发 + 概率掷点。R-77-1 20 km 样本的弹道时间在三档 dt 下"
        f"基本一致（{t_base:.3f} s），但最近距离为 {d_desc} m，"
        f"结局{'稳定' if conv['outcome_stable'] else '不稳定'}（{dt_desc}）。"
        "在 `d_miss ≈ 战斗部半径` 的边界上，命中/脱失不是 dt 的连续函数，"
        "因此 Pk 校准必须锁定默认步长 1/120，且单发样本不能解释为命中概率。"
    )
    lines.append("")
    lines.append("## 5. 导弹包线（单发、确定性样本）")
    lines.append("")
    lines.append("| 导弹 | 距离 km | 目标 g | 结果 | 最近距离 m | 飞行时间 s |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in missiles:
        lines.append(f"| {r['missile']} | {r['launch_range_km']:.0f} | {r['target_g']:.0f} | {r['ending']} | {r['min_distance_m']:.1f} | {r['flight_time_s']:.2f} |")
    lines.append("")
    lines.append(
        "说明：这是单发确定性弹道样本，不是命中概率。命中/脱失由概率近炸决定，"
        "所以逐点结果非单调（例如 R-73 在 12 km 命中、5–10 km 脱失，是引信掷点的差别，"
        "不是射程异常）。R-77-1 60 km 属于能量耗尽后失控，最近距离大幅上升。"
        "Pk 与 R_max/NEZ 仍需多样本蒙特卡洛校准。"
    )
    lines.append("")
    lines.append("## 6. Headless 运行与确定性")
    lines.append("")
    lines.append(f"- 5 s：returncode={smoke['returncode']}，Traceback={smoke['has_traceback']}，OBJ={smoke['mentions_obj']}")
    lines.append(f"- 60 s：returncode={full['returncode']}，Traceback={full['has_traceback']}，OBJ={full['mentions_obj']}")
    lines.append(f"- 同 seed 复跑：{'逐字节一致' if det['identical'] else '不一致'}（{det['log_bytes']} B）")
    lines.append("- 日志：`headless_5s.log`、`headless_60s.log`、`headless_det_a.log`、`headless_det_b.log`。")
    lines.append("")
    lines.append("## 7. 覆盖范围与未覆盖项")
    lines.append("")
    lines.append(
        "本轮已覆盖：A01（ISA 与分段连续）、A02（跨声速阻力连续性）、A05/A08（平飞与定常盘旋、"
        "2DOF/3DOF 各一组）、A06（爬升率 vs Ps）、A12（能量收支）、A13（燃油与 TSFC）、"
        "简化 h-V 包线、M01（推进与总冲，见单元测试）、M04/M06（导弹单发弹道）、"
        "§7.1 确定性、§7.2 步长收敛、headless 冒烟与 60 s 运行。"
    )
    lines.append("")
    lines.append(
        "未覆盖：A03/A04（V-n 图与失速边界）、A07（加速性）、A09（瞬时盘旋）、A10/A11/A18（6DOF 滚转、"
        "配平漂移、坐标符号）、A14（TVC/过失速）、A15（G-LOC）、A16（3DOF/6DOF 定量对比）、"
        "A17（保护边界）、M02（导弹可用过载）、M05/M07（R_max/NEZ 与 Pk 校准）、M08–M16"
        "（loft、视场、SARH/ARH、干扰、齐射、统计收敛）、§6 策略因子与消融、P5 独立数据对照。"
    )
    lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"结果已写入 {OUT}")
    print(json.dumps(metrics["pass"], ensure_ascii=False, indent=2))
    return 0 if all(metrics["pass"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
