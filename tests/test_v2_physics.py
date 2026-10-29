"""第二版物理验证：只覆盖 6DOF 与解析式（不再针对 3DOF 建用例）。

对应《现实性评审复核与第二版修改验证方案》的 U / I / E / W 矩阵：

- U：四元数姿态与垂直机动（P0-1）
- I：初始化、平飞、确定性（P0-1 / P0-3）
- E：推力标定后的包线（P1-1 / P1-2）
- W：统一发射包线、机炮外弹道、导弹波阻、高度上限软处理（P0-2 / P1-3 / P1-4）
"""
from __future__ import annotations

import dataclasses
import math

import pytest

from main import AIM120C, AIM9X, F16V_SPEC, R73, R77_1, SU57_SPEC, build_world
from sim import (
    Action,
    G,
    MAX_BANK,
    Missile,
    World,
    air_density,
    cd0_at,
    clamp,
    euler_from_quat,
    launch_envelope,
    mach_number,
    quat_from_euler,
    rcs_scale,
    speed_of_sound,
    wrap_pi,
)


# ---------------------------------------------------------------------------
# 工具
# ---------------------------------------------------------------------------

def world6(specs=(SU57_SPEC, F16V_SPEC), z0: float = 9000.0,
           dt: float = 1.0 / 120.0, z_ceiling: float = 30000.0) -> World:
    return World(
        list(specs), dt=dt, arena=1.0e9, time_limit=1.0e6, z0=z0,
        z_floor=0.0, z_ceiling=z_ceiling, dof=6,
        x_offset=0.0, y_offset=0.0,
    )


def place(ac, x, y, z, psi, v, vx=0.0, vy=0.0, vz=0.0):
    ac.x, ac.y, ac.z = x, y, z
    ac.psi, ac.gamma, ac.v = psi, 0.0, v
    ac.n, ac.bank = 1.0, 0.0
    ac.alive = True
    if ac.rigid:
        ac.quat = quat_from_euler(ac.phi, ac.theta, ac.psi)
        ac.u, ac.vb, ac.wb = v, 0.0, 0.0


def thrust_ratio(spec, z, mach):
    """平飞阻力 / 最大加力推力：E05 的“推力比”锚点。"""
    v = mach * speed_of_sound(z)
    q = 0.5 * air_density(z) * v * v
    drag = q * cd0_at(mach, spec.cd0) * spec.wing_area
    return drag / max(spec.thrust_max * spec.lapse(z, mach), 1e-9)


def ps_max(spec, z, v_step=6.0):
    best = -1e9
    v = spec.v_stall_at(z)
    while v <= 820.0:
        best = max(best, spec.specific_power(v, z, 1.0, 1.0, spec.mass, 1.0))
        v += v_step
    return best


def v_max_mach(spec, z, v_step=4.0):
    """最大平飞马赫数；维持不住平飞时返回 None。

    与 Ps=0 同口径：T = D0 + Di(n=1)。只扣 D0 会在高于升限处解出
    虚假的平飞速度，因为漏掉了低动压下的 1g 诱导阻力。
    """
    vmin = spec.v_stall_at(z)
    k = 1.0 / (math.pi * spec.aspect_ratio * spec.oswald * spec.wing_area)
    prev_v, prev_f = None, None
    last = None
    v = vmin
    while v <= 900.0:
        m = mach_number(v, z)
        q = 0.5 * air_density(z) * v * v
        f = (
            spec.thrust_at(1.0, 1.0, z, m)
            - q * cd0_at(m, spec.cd0) * spec.wing_area
            - (spec.mass * G) ** 2 * k / q
        )
        if prev_f is not None and prev_f > 0.0 >= f:
            last = prev_v + prev_f / max(prev_f - f, 1e-9) * v_step
        prev_v, prev_f = v, f
        v += v_step
    return None if last is None else last / speed_of_sound(z)


def ceiling(spec, z_step=200.0, v_step=8.0):
    """Ps_max 穿过 0 的高度（实用升限）。"""
    prev_z, prev_ps = 0.0, None
    z = 0.0
    while z <= 26000.0:
        ps = ps_max(spec, z, v_step)
        if ps <= 0.0:
            if prev_ps is None:
                return 0.0
            frac = prev_ps / max(prev_ps - ps, 1e-9)
            return prev_z + frac * (z - prev_z)
        prev_z, prev_ps = z, ps
        z += z_step
    return prev_z


# ---------------------------------------------------------------------------
# U：四元数姿态（P0-1）
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("phi,theta,psi", [
    (0.0, 0.0, 0.0),
    (0.4, -0.6, 2.1),
    (-1.2, 0.9, -3.0),
    (2.8, 1.3, 1.0),
])
def test_u01_quat_euler_roundtrip(phi, theta, psi):
    """四元数合成与反解必须一致（theta 在 ±90° 内）。"""
    p2, t2, y2 = euler_from_quat(quat_from_euler(phi, theta, psi))
    assert p2 == pytest.approx(phi, abs=1e-6)
    assert t2 == pytest.approx(theta, abs=1e-6)
    assert y2 == pytest.approx(psi, abs=1e-6)


def test_u02_quat_integration_passes_vertical():
    """6DOF 持续拉杆必须让俯仰越过 70°（旧实现夹断在 ±70°，筋斗做不出来）。"""
    w = world6((SU57_SPEC, F16V_SPEC))
    a = w.acs[0]
    place(a, 0.0, 0.0, 9000.0, 0.0, 320.0)
    a.u, a.wb = 320.0, 0.0
    a.quat = quat_from_euler(0.0, 0.0, 0.0)
    max_theta = 0.0
    max_gamma = 0.0
    for _ in range(int(7.0 / w.dt)):
        w.step([Action(n=9.0, bank=0.0, throttle=1.0, afterburner=1.0),
                Action(n=1.0, bank=0.0, throttle=0.0)])
        max_theta = max(max_theta, abs(a.theta))
        max_gamma = max(max_gamma, abs(a.gamma))
        assert math.isfinite(a.theta) and math.isfinite(a.gamma)
    assert max_theta > math.radians(70.0)
    assert max_gamma > math.radians(60.0)


def test_u03_quat_norm_stays_unit():
    """四元数积分每步归一化，长时间机动后仍为单位长度。"""
    w = world6((F16V_SPEC, SU57_SPEC))
    a = w.acs[0]
    place(a, 0.0, 0.0, 9000.0, 0.0, 350.0)
    a.quat = quat_from_euler(0.0, 0.0, 0.0)
    for _ in range(int(4.0 / w.dt)):
        w.step([Action(n=6.0, bank=0.6, throttle=1.0, afterburner=1.0),
                Action(n=1.0, bank=0.0, throttle=0.0)])
    norm = math.sqrt(sum(c * c for c in a.quat))
    assert norm == pytest.approx(1.0, abs=1e-9)


def test_u04_negative_g_is_reachable():
    """6DOF 下推杆能得到负过载（N_MIN=-3 生效，不再只到 0 g）。"""
    w = world6((F16V_SPEC, SU57_SPEC))
    a = w.acs[0]
    place(a, 0.0, 0.0, 9000.0, 0.0, 350.0)
    a.quat = quat_from_euler(0.0, 0.0, 0.0)
    min_n = 9.0
    for _ in range(int(1.2 / w.dt)):
        w.step([Action(n=-3.0, bank=0.0, throttle=0.6),
                Action(n=1.0, bank=0.0, throttle=0.0)])
        min_n = min(min_n, a.n)
    assert min_n < -0.5


# ---------------------------------------------------------------------------
# I：初始化 / 平飞 / 确定性（P0-1 / P0-3）
# ---------------------------------------------------------------------------

def test_i01_initial_quat_matches_euler():
    w = build_world(time_limit=1.0, dof=6)
    for ac in w.acs:
        phi, theta, psi = euler_from_quat(ac.quat)
        assert phi == pytest.approx(ac.phi, abs=1e-5)
        assert theta == pytest.approx(ac.theta, abs=1e-5)
        # psi 在 ±180° 处等价，按 wrap_pi 比较。
        assert wrap_pi(psi - ac.psi) == pytest.approx(0.0, abs=1e-5)


def test_i02_trimmed_level_flight_holds_altitude():
    """配平后 5 s 平飞高度漂移应很小（迎角/推力初始化正确）。"""
    w = world6((SU57_SPEC, F16V_SPEC))
    a, b = w.acs
    place(a, 0.0, 0.0, 9000.0, 0.0, 288.0)
    a.u, a.wb = 288.0, 0.0
    a.quat = quat_from_euler(0.0, 0.0, 0.0)
    place(b, 300000.0, 0.0, 9000.0, math.pi, 260.0)
    b.quat = quat_from_euler(0.0, 0.0, math.pi)
    throttle = a.spec.lapse(9000.0, mach_number(288.0, 9000.0))
    cruise = clamp(
        (0.5 * air_density(9000.0) * 288.0 ** 2 * cd0_at(0.95, a.spec.cd0)
         * a.spec.wing_area)
        / max(a.spec.mil_thrust() * throttle, 1e-6), 0.2, 1.0)
    for _ in range(int(5.0 / w.dt)):
        w.step([Action(n=1.0, bank=0.0, throttle=cruise),
                Action(n=1.0, bank=0.0, throttle=0.0)])
    assert abs(a.z - 9000.0) < 150.0, f"altitude drift {a.z - 9000.0:.1f} m"
    assert math.isfinite(a.v)


def test_i03_determinism_same_seed_same_state():
    def run():
        w = build_world(time_limit=3.0, dof=6)
        for _ in range(int(3.0 / w.dt)):
            w.step([Action(n=4.0, bank=0.3, throttle=1.0, afterburner=1.0, fire=True),
                    Action(n=2.0, bank=-0.2, throttle=0.8)])
        return [(round(ac.x, 9), round(ac.y, 9), round(ac.z, 9),
                 round(ac.v, 9), ac.quat) for ac in w.acs], len(w.bullets)
    assert run() == run()


def test_i04_rigid_velocity_matches_body_state():
    """世界系速度必须与机体速度/四元数一致（积分链没有断）。"""
    w = world6((SU57_SPEC, F16V_SPEC))
    a = w.acs[0]
    place(a, 0.0, 0.0, 9000.0, 0.7, 300.0)
    a.quat = quat_from_euler(0.0, 0.0, 0.7)
    for _ in range(60):
        w.step([Action(n=5.0, bank=0.4, throttle=1.0, afterburner=1.0),
                Action(n=1.0, bank=0.0, throttle=0.0)])
    vx, vy, vz = a.velocity
    assert math.sqrt(vx * vx + vy * vy + vz * vz) == pytest.approx(a.v, rel=1e-6)
    assert abs(vz / max(a.v, 1e-6) - math.sin(a.gamma)) < 1e-6


# ---------------------------------------------------------------------------
# E：推力标定后的包线（P1-1 / P1-2）
# ---------------------------------------------------------------------------

def test_e01_ps_falls_with_altitude_after_11km():
    """Ps_max 在 11 km 以上单调下降，25 km 两机均为负（旧模型会一路升到 25 km+）。"""
    for spec in (F16V_SPEC, SU57_SPEC):
        ps = {z: ps_max(spec, z) for z in (11000, 14000, 17000, 20000)}
        assert ps[11000] > ps[14000] > ps[17000] > ps[20000], (spec.name, ps)
        assert ps_max(spec, 25000.0) < 0.0, spec.name


def test_e02_service_ceiling_ranges():
    f16 = ceiling(F16V_SPEC)
    su = ceiling(SU57_SPEC)
    assert 16000.0 <= f16 <= 18000.0, f"F-16V ceiling {f16:.0f}"
    assert 19500.0 <= su <= 22000.0, f"Su-57 ceiling {su:.0f}"


def test_e03_vmax_peaks_and_nine_km_speed():
    f16_9 = v_max_mach(F16V_SPEC, 9000.0)
    su_9 = v_max_mach(SU57_SPEC, 9000.0)
    assert 1.45 <= f16_9 <= 1.70, f16_9
    assert 1.45 <= su_9 <= 1.65, su_9
    # F-16V：峰值在 11–13.5 km；18 km 已高于实用升限，维持不住平飞。
    f16_12 = v_max_mach(F16V_SPEC, 12000.0)
    assert 1.55 <= f16_12 <= 1.80, f16_12
    assert v_max_mach(F16V_SPEC, 18000.0) is None
    peak_f16 = max(
        (v if v is not None else -1.0, z)
        for v, z in (
            (v_max_mach(F16V_SPEC, float(z)), z)
            for z in range(9000, 19001, 500)
        )
    )
    assert 11.0 <= peak_f16[1] / 1000.0 <= 13.5, peak_f16
    # Su-57：高空型，Vmax 峰值落在 15–19 km 区间；21 km 不可平飞。
    peak_su = max(
        (v if v is not None else -1.0, z)
        for v, z in (
            (v_max_mach(SU57_SPEC, float(z)), z)
            for z in range(9000, 21001, 500)
        )
    )
    assert 15.0 <= peak_su[1] / 1000.0 <= 19.0, peak_su
    assert v_max_mach(SU57_SPEC, 21000.0) is None


def test_e04_thrust_to_weight():
    assert F16V_SPEC.thrust_max / (F16V_SPEC.mass * G) == pytest.approx(1.15, abs=0.05)
    assert SU57_SPEC.thrust_max / (SU57_SPEC.mass * G) == pytest.approx(1.14, abs=0.05)


def test_e05_military_thrust_ratio_at_9km():
    """9 km/M0.9 平飞阻力约占最大加力推力的 0.40–0.55（锚点量级）。"""
    for spec in (F16V_SPEC, SU57_SPEC):
        ratio = thrust_ratio(spec, 9000.0, 0.9)
        assert 0.38 <= ratio <= 0.55, (spec.name, ratio)


def test_e06_supersonic_lift_loss():
    """超声速升力损失：M2 的可用升力系数比 M0.9 低 25–32%。

    ``n_lift`` 本身随动压 v² 增长，同高度下 M2 的绝对过载反而更大；
    升力损失体现在“同样动压下”的 CLmax，所以这里比较的是系数而非过载。
    """
    for spec in (F16V_SPEC, SU57_SPEC):
        ratio = spec.cl_max_at(2.0) / spec.cl_max_at(0.9)
        assert 0.65 <= ratio <= 0.78, (spec.name, ratio)


def test_e07_sustained_and_instantaneous_turn():
    """9 km/M0.8：稳定盘旋率 8–12°/s；瞬时过载必须显著大于稳定过载。"""
    z, mach = 9000.0, 0.8
    v = mach * speed_of_sound(z)
    for spec in (F16V_SPEC, SU57_SPEC):
        ns = spec.n_sustain(v, z, 1.0, spec.mass, 1.0)
        rate = math.degrees(G * math.sqrt(max(ns * ns - 1.0, 0.0)) / v)
        assert 8.0 <= rate <= 12.0, (spec.name, rate)
        assert spec.n_available(v, z, spec.mass) > ns * 1.2


# ---------------------------------------------------------------------------
# W：武器与边界（P0-2 / P1-3 / P1-4）
# ---------------------------------------------------------------------------

def test_w01_gun_fires_at_rated_rate():
    w = world6((F16V_SPEC, SU57_SPEC))
    a, b = w.acs
    place(a, 0.0, 0.0, 9000.0, 0.0, 300.0)
    a.quat = quat_from_euler(0.0, 0.0, 0.0)
    place(b, 600.0, 0.0, 9000.0, math.pi, 300.0)
    b.quat = quat_from_euler(0.0, 0.0, math.pi)
    ammo0 = a.ammo
    for _ in range(120):
        w.step([Action(n=1.0, throttle=0.6, fire=True),
                Action(n=1.0, throttle=0.0)])
    assert abs(w.stats["bullets_fired"] - F16V_SPEC.rounds_per_sec) <= 3
    assert ammo0 - a.ammo == pytest.approx(w.stats["bullets_fired"])


def test_w02_bullets_feel_gravity():
    w = world6((F16V_SPEC, SU57_SPEC))
    a, b = w.acs
    place(a, 0.0, 0.0, 9000.0, 0.0, 300.0)
    place(b, 50000.0, 0.0, 9000.0, math.pi, 300.0)
    for _ in range(3):
        w.step([Action(n=1.0, throttle=0.6, fire=True),
                Action(n=1.0, throttle=0.0)])
    assert w.bullets, "no bullet spawned"
    bullet = w.bullets[0]
    vz0 = bullet.vz
    for _ in range(30):
        w._integrate_bullets()
    assert bullet.vz < vz0 - 2.0


def test_w03_aligned_gun_shot_hits():
    w = world6((F16V_SPEC, SU57_SPEC))
    a, b = w.acs
    place(a, 0.0, 0.0, 9000.0, 0.0, 300.0)
    a.quat = quat_from_euler(0.0, 0.0, 0.0)
    place(b, 600.0, 0.0, 9000.0, math.pi, 300.0)
    b.quat = quat_from_euler(0.0, 0.0, math.pi)
    b.hp = 100.0
    for _ in range(240):
        w.step([Action(n=1.0, throttle=0.6, fire=True),
                Action(n=1.0, throttle=0.0)])
        if not b.alive:
            break
    assert w.stats["bullet_hits"] > 0
    assert b.hp < 100.0


def test_w04_off_boresight_shot_misses():
    """目标偏离炮线 30°，弹丸直飞不应命中（旧实现只要在视锥内就扣血）。"""
    w = world6((F16V_SPEC, SU57_SPEC))
    a, b = w.acs
    place(a, 0.0, 0.0, 9000.0, 0.0, 300.0)
    a.quat = quat_from_euler(0.0, 0.0, 0.0)
    place(b, 520.0, 300.0, 9000.0, math.pi, 300.0)
    b.quat = quat_from_euler(0.0, 0.0, math.pi)
    for _ in range(240):
        w.step([Action(n=1.0, throttle=0.6, fire=True),
                Action(n=1.0, throttle=0.0)])
    assert w.stats["bullet_hits"] == 0
    assert b.hp == b.spec.hp_max


def peak_mach(mspec, shooter, range_m=6000.0, z=9000.0, v=470.0):
    w = world6((shooter, F16V_SPEC), z0=z)
    a, b = w.acs
    place(a, -range_m / 2, 0.0, z, 0.0, v)
    place(b, range_m / 2, 0.0, z, math.pi, 300.0)
    m = Missile(
        spec=mspec, owner=0, target=1, x=a.x, y=a.y, z=a.z,
        vx=v, vy=0.0, vz=0.0, mass=mspec.mass, fuel=mspec.propellant,
        nx=1.0, ny=0.0, nz=0.0, roll_seed=7,
    )
    w.missiles.append(m)
    best, t = 0.0, 0.0
    while m.alive and t < 40.0:
        w._update_missiles()
        t += w.dt
        best = max(best, mach_number(m.speed, max(m.z, 0.0)))
    return best


@pytest.mark.parametrize("mspec,shooter,lo,hi", [
    (R73, SU57_SPEC, 2.3, 3.0),
    (AIM9X, F16V_SPEC, 2.4, 3.2),
    (R77_1, SU57_SPEC, 4.2, 5.2),
    (AIM120C, F16V_SPEC, 4.1, 4.9),
])
def test_w05_missile_peak_mach(mspec, shooter, lo, hi):
    peak = peak_mach(mspec, shooter)
    assert lo <= peak <= hi, f"{mspec.name} peak M {peak:.2f}"


def test_w06_envelope_rcs_handling():
    """红外弹包线与 RCS 无关；雷达弹按 RCS 的四次方根缩放。"""
    ac = world6((SU57_SPEC, F16V_SPEC)).acs[0]
    stealth = dataclasses.replace(F16V_SPEC, rcs=0.1)
    plain = dataclasses.replace(F16V_SPEC, rcs=1.0)
    ir_lo, ir_hi = launch_envelope(ac, R73, dataclasses.replace(plain, rcs=0.001))
    assert ir_lo == R73.min_range and ir_hi == min(R73.max_range, R73.seeker_range)
    _, radar_hi = launch_envelope(
        ac, R77_1, type("F", (), {"spec": stealth})()
    )
    assert radar_hi == pytest.approx(
        min(R77_1.max_range, R77_1.seeker_range * rcs_scale(stealth.rcs))
    )


def test_w07_high_altitude_no_breakup():
    """超过高度上限只记录一次事件，不再判“解体”。"""
    w = world6((SU57_SPEC, F16V_SPEC), z0=20000.0)
    a = w.acs[0]
    a.z = 30500.0
    a.quat = quat_from_euler(0.0, 0.0, 0.0)
    for _ in range(3):
        w.step([Action(n=1.0, throttle=0.5),
                Action(n=1.0, throttle=0.0)])
    assert a.alive
    warnings = [e for e in w.events if "验证高度上限" in e[1]]
    assert len(warnings) == 1
