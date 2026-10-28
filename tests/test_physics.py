"""物理模型单元回归：跨声速阻力连续性、导弹质量与近炸概率。"""
from __future__ import annotations

import math

import pytest

from main import F16V_SPEC, SU57_SPEC, R73, R77_1
from sim import (
    cd0_at,
    drag_divergence,
    proximity_pk,
)


def test_no_step_at_075():
    """旧模型在 M=0.75 处有约 28.46% 的阶跃；修复后两侧必须重合。"""
    left = cd0_at(0.75 - 1e-9, SU57_SPEC.cd0)
    right = cd0_at(0.75 + 1e-9, SU57_SPEC.cd0)
    assert abs(right - left) < 1e-6


@pytest.mark.parametrize("mach", [0.75, 0.78, 0.90, 0.95, 1.05, 1.35])
def test_drag_c1_at_join_points(mach):
    """smoothstep 拼接点附近左右导数应一致。"""
    h = 1e-6
    left = (drag_divergence(mach) - drag_divergence(mach - h)) / h
    right = (drag_divergence(mach + h) - drag_divergence(mach)) / h
    assert abs(left - right) < 0.01


def test_drag_curve_is_continuous():
    """扫描 M=0.2–2.5，相邻 0.001 马赫的阻力相对变化应很小。"""
    prev = 1.0 + drag_divergence(0.2)
    max_step = 0.0
    for i in range(201, 2501):
        cur = 1.0 + drag_divergence(i / 1000.0)
        max_step = max(max_step, abs(cur - prev) / prev)
        prev = cur
    assert max_step < 0.005, f"max relative step = {max_step:.4%}"


def test_low_mach_has_no_wave_drag():
    assert drag_divergence(0.5) == 0.0
    assert drag_divergence(0.75) == 0.0
    assert drag_divergence(0.78) == 0.0


def test_onset_is_gradual():
    """阻力发散段横跨约 0.17 Ma：M=0.80 处增量应仍很小。"""
    assert 0.0 < drag_divergence(0.80) < 0.02
    assert drag_divergence(0.85) < 0.25


def test_transonic_bump_and_supersonic_floor():
    """M1.05 附近应有明显鼓包；M2 仍保留超声速波阻，不回到亚声速值。"""
    assert drag_divergence(1.10) > 0.60
    assert drag_divergence(2.00) > 0.15
    assert drag_divergence(2.00) < 0.25


def test_cd0_positive_and_finite():
    for mach in (0.0, 0.5, 0.8, 1.0, 1.5, 2.5):
        value = cd0_at(mach, F16V_SPEC.cd0)
        assert math.isfinite(value)
        assert value > 0.0


def test_missile_propellant_and_impulse():
    for spec in (R73, R77_1):
        assert spec.propellant == pytest.approx(spec.mass - spec.burnout_mass)
        assert spec.mass_flow > 0.0
        assert spec.thrust * spec.burn_time > 0.0


def test_proximity_pk_monotonic():
    values = [proximity_pk(d, 10.0) for d in (0.0, 2.0, 5.0, 8.0, 10.0)]
    assert values == sorted(values, reverse=True)
    assert values[0] == pytest.approx(0.95)
    assert 0.0 < values[-1] < 0.2
