"""全动鸭翼模型回归测试。

覆盖：
- 统一升力曲线（力模型与控制律共用同一 CLmax / 失速迎角）；
- α=22°/30°/35° 与 M0.8/1.6 的失速后行为；
- 鸭翼位置/速率限幅与铰链力矩饱和；
- 1g 配平稳定性；
- 差动鸭翼的滚转方向。
"""
import math

import pytest

from dogfight.config import load_bundle
from dogfight.core.sim import Action, World


@pytest.fixture(scope="module")
def specs():
    bundle = load_bundle()
    return bundle.aircraft["J-20A"].spec, bundle.aircraft["F-15EX"].spec


@pytest.fixture()
def world(specs):
    j20, f15 = specs
    return World(
        [j20, f15], dt=1.0 / 120.0, dof=6,
        arena=1e9, time_limit=1e6,
        z0=8000.0, z_ceiling=30000.0,
        x_offset=0.0, y_offset=0.0,
    )


def _alpha_of(ac) -> float:
    return math.degrees(math.atan2(ac.wb, max(ac.u, 1.0)))


def test_unified_clmax_is_single_source(world, specs):
    """控制律用的 CLmax 必须等于力模型在迎角扫描上的峰值。"""
    j20, _ = specs
    ac = world.acs[0]
    for mach in (0.4, 0.8, 1.2, 1.6):
        peak = world._max_total_cl(ac, mach)
        grid = [
            world._total_cl_at(ac, math.radians(a / 4.0), mach)
            for a in range(0, 161)
        ]
        assert peak == pytest.approx(max(grid), rel=0.01, abs=0.01)
        # 6DOF 力模型与控制律调用的是同一入口，不应再出现 1.93 vs 1.60。
        assert peak <= j20.cl_max_at(mach) + j20.canard_cl_bonus + 1e-6


@pytest.mark.parametrize(
    "mach,should_peak_at_30",
    [(0.8, True), (1.6, False)],
)
def test_alpha_sweep_stalls_after_30(world, specs, mach, should_peak_at_30):
    """α=22°/30°/35°：30° 附近是峰值，35° 已进入失速后衰减。"""
    ac = world.acs[0]
    cls = [
        world._total_cl_at(ac, math.radians(a), mach, vortex=1.0, delta=0.0)
        for a in (22.0, 30.0, 35.0)
    ]
    assert cls[2] < cls[1]
    if should_peak_at_30:
        assert cls[1] > cls[0]
    else:
        # 超声速升力整体下降，线性段先被 cl_max 截断，30° 不再高于 22°；
        # 但失速后 35° 仍必须继续衰减。
        assert cls[1] > 0.9 * cls[0]


def test_supersonic_lift_loss(world, specs):
    """M1.6 的可用升力系数必须显著低于 M0.8。"""
    j20, _ = specs
    ac = world.acs[0]
    assert world._max_total_cl(ac, 1.6) < 0.8 * world._max_total_cl(ac, 0.8)


def test_canard_position_and_rate_limits(world, specs):
    """作动器：位置不超过 ±deflect_max，单步变化不超过 rate_max·dt。"""
    j20, _ = specs
    ac = world.acs[0]
    dt = world.dt
    max_step = j20.canard_rate_max * dt
    for _ in range(int(3.0 / dt)):
        prev = ac.canard_deg
        world._update_canard_state(
            ac, j20.canard_deflect_max, Action(canard=1.0),
            math.radians(5.0), 1500.0,
        )
        assert abs(ac.canard_deg - prev) <= max_step + 1e-9
        assert abs(ac.canard_deg) <= j20.canard_deflect_max + 1e-12
    assert ac.canard_deg == pytest.approx(
        j20.canard_deflect_max, abs=math.radians(0.5),
    )


def test_canard_hinge_saturation(world, specs):
    """极高动压 + 大迎角时铰链力矩饱和，偏度被限制在力矩权限内。"""
    j20, _ = specs
    ac = world.acs[0]
    qbar = 4.0e5
    alpha = math.radians(20.0)
    for _ in range(int(3.0 / world.dt)):
        world._update_canard_state(
            ac, j20.canard_deflect_max, Action(canard=1.0),
            alpha, qbar,
        )
    assert ac.canard_hinge <= 1.0
    assert abs(ac.canard_deg) < j20.canard_deflect_max
    # 铰链方程 H = K(0.6α + δ) 必须落在作动器扭矩附近。
    k = qbar * j20.canard_area * j20.canard_chord * j20.canard_hinge_coeff
    h = abs(k * (0.6 * alpha + ac.canard_deg))
    assert h <= j20.canard_actuator_torque * 1.05


def test_canard_trim_is_stable(world, specs):
    """1g 平飞配平：过载收敛到 1，俯仰角速度收敛到 0，鸭翼不抖。"""
    j20, _ = specs
    ac = world.acs[0]
    for _ in range(int(8.0 / world.dt)):
        world.step([
            Action(n=1.0, bank=0.0, throttle=0.6),
            Action(n=1.0, bank=0.0, throttle=0.6),
        ])
    assert ac.n == pytest.approx(1.0, abs=0.05)
    assert abs(ac.q) < 0.01
    assert abs(_alpha_of(ac)) < 5.0
    assert abs(ac.canard_deg) <= j20.canard_deflect_max
    assert ac.canard_hinge < 1.0


def test_differential_canard_rolls_right(world, specs):
    """canard_diff>0（左+右-）应产生正滚转角速度（右滚）。"""
    j20, _ = specs
    ac = world.acs[0]
    for _ in range(int(0.5 / world.dt)):
        world.step([
            Action(n=1.0, bank=0.0, throttle=0.6, canard_diff=1.0),
            Action(n=1.0, bank=0.0, throttle=0.6),
        ])
    assert ac.canard_diff_deg > 0.0
    assert ac.p > 0.0
