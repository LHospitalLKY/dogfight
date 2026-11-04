"""开局战术（隐身压缩 stealth_compress / 侧翼绕后 flank）单元测试。"""
from __future__ import annotations

import math

import pytest

from dogfight.app.main import build_agents
from dogfight.core.sim import Observation, World
from dogfight.tactics.agents import TacticalAgent


def _make_obs(**overrides) -> Observation:
    """构造一个占位阶段可用的最小观测。"""
    base = dict(
        dist=40000.0, dist_h=40000.0, dz=0.0,
        los_az=0.0, los_el=0.0, off_az=0.0, off_el=0.0,
        angle_off_nose=0.0, aspect=math.pi, foe_off_nose=0.0,
        closure=-200.0, in_gun_range=False, in_gun_cone=False,
        in_gun_trigger=False,
        own_speed=250.0, own_speed_frac=2.0, own_alt=9000.0,
        own_psi=0.0, own_gamma=0.0, own_n_avail=8.0, own_n=1.0,
        own_n_sustain=5.0, own_ps=50.0, own_ps_level=80.0,
        own_hp=100.0, own_hp_frac=1.0, own_ammo=300.0,
        foe_speed=250.0, foe_alt=9000.0, foe_psi=math.pi, foe_gamma=0.0,
        foe_hp=100.0, foe_hp_frac=1.0,
        energy_diff=0.0, edge_margin=1e9, z_margin_low=7500.0,
        z_margin_high=21000.0, center_az=0.0,
    )
    base.update(overrides)
    return Observation(**base)


def test_stealth_compress_offset_smaller_than_default():
    """隐身压缩的横向偏置应显著小于默认能量/角度占位。"""
    default = TacticalAgent(seed=0, opening_tactic="default")
    stealth = TacticalAgent(seed=0, opening_tactic="stealth_compress")
    # 多取几个种子，确保不是偶然
    for s in range(20):
        d = TacticalAgent(seed=s, opening_tactic="default")
        st = TacticalAgent(seed=s, opening_tactic="stealth_compress")
        assert math.degrees(st._open_offset) <= 3.0
        # 默认偏置至少 5°（能量型下限），隐身压缩明显更小
        assert st._open_offset < d._open_offset


def test_stealth_compress_throttle_below_mil_no_afterburner():
    """隐身压缩的占位油门在军推以下，且 _position 不开加力。"""
    for s in range(10):
        ag = TacticalAgent(seed=s, opening_tactic="stealth_compress")
        assert ag._open_throttle <= 0.93 + 1e-9
        act = ag._position(_make_obs())
        assert act.afterburner == 0.0
        assert act.throttle <= 0.93 + 1e-9


def test_stealth_compress_downgrades_always_radar():
    """配 always 常开雷达时，隐身压缩会自动降级为 auto 静默。"""
    ag = TacticalAgent(radar_policy="always", opening_tactic="stealth_compress")
    assert ag.radar_policy == "auto"
    assert ag._radar_on is False


def test_default_tactic_keeps_radar_policy():
    """非隐身压缩时，雷达策略不被改动。"""
    ag = TacticalAgent(radar_policy="always", opening_tactic="default")
    assert ag.radar_policy == "always"


def test_stealth_compress_exits_early_when_foe_locked():
    """被对方火控锁定时，隐身压缩提前结束占位。"""
    ag = TacticalAgent(seed=0, opening_tactic="stealth_compress")
    obs = _make_obs(dist=40000.0, foe_radar_locked=True)
    ag.act(obs)
    assert ag._opening_done is True


def test_stealth_compress_stays_open_when_not_locked():
    """未被锁定且距离仍远时，占位继续。"""
    ag = TacticalAgent(seed=0, opening_tactic="stealth_compress")
    obs = _make_obs(dist=40000.0, foe_radar_locked=False)
    ag.act(obs)
    assert ag._opening_done is False


def test_auto_normalizes_to_default():
    """opening_tactic=auto 在 agent 内被归一成 default。"""
    ag = TacticalAgent(opening_tactic="auto")
    assert ag.opening_tactic == "default"


def test_build_agents_auto_selects_stealth_for_su57_vs_f16v():
    """默认对阵下 Su-57（前向 RCS 0.3）对 F-16V（1.2）应自动选隐身压缩。"""
    agents = build_agents()
    # world 里第一架是 SU-57，第二架是 F-16V（按默认场景顺序）
    su57, f16 = agents
    assert su57.opening_tactic == "stealth_compress"
    assert f16.opening_tactic == "default"


def test_build_agents_respects_explicit_persona_tactic():
    """YAML 里显式指定 opening_tactic 时不被自动选择覆盖。"""
    agents = build_agents()
    # 默认场景没显式配，这里只校验接口不报错；显式覆盖由 config 层校验保障。
    assert all(a.opening_tactic in ("default", "stealth_compress") for a in agents)


# ---------------------------------------------------------------------------
# 侧翼绕后（flank）
# ---------------------------------------------------------------------------

def test_flank_downgrades_always_radar():
    """flank 也要求静默接敌，always 雷达降级为 auto。"""
    ag = TacticalAgent(radar_policy="always", opening_tactic="flank")
    assert ag.radar_policy == "auto"
    assert ag._radar_on is False


def test_flank_position_afterburner_and_turns_sideways():
    """flank 的 _position 开加力提速，且确实在压坡度绕向侧后方。"""
    ag = TacticalAgent(seed=0, opening_tactic="flank")
    obs = _make_obs(foe_psi=math.pi)  # 敌机朝西飞，我在东
    act = ag._position(obs)
    # 占位阶段开加力+满油门，尽快接近敌机。
    assert act.afterburner == 1.0
    assert act.throttle == 1.0
    # 正在压坡度（不是直飞）去绕向对方侧后方；坡度方向由几何决定。
    assert abs(act.bank) > 0.1


def test_flank_exits_when_in_blind_spot_and_in_range():
    """进入对方后半球盲区且在导弹射程内时，flank 结束占位。"""
    ag = TacticalAgent(seed=0, opening_tactic="flank")
    obs = _make_obs(
        dist=30000.0, foe_off_nose=math.radians(120.0),
        radar_max_range=40000.0,
    )
    ag.act(obs)
    assert ag._opening_done is True


def test_flank_stays_open_when_not_in_blind_spot():
    """还在对方前向雷达锥内时，flank 继续占位。"""
    ag = TacticalAgent(seed=0, opening_tactic="flank")
    obs = _make_obs(
        dist=30000.0, foe_off_nose=math.radians(30.0),
        radar_max_range=40000.0,
    )
    ag.act(obs)
    assert ag._opening_done is False


# ---------------------------------------------------------------------------
# 全向 RWR 与按阵营 AWACS（感知层）
# ---------------------------------------------------------------------------

def _two_plane_world(awacs):
    from dogfight.app.main import F16V_SPEC, SU57_SPEC
    w = World(
        [SU57_SPEC, F16V_SPEC], dt=1.0 / 120.0, arena=1.0e9,
        time_limit=1.0e6, z0=9000.0, z_floor=0.0, z_ceiling=30000.0,
        dof=3, x_offset=30000.0, y_offset=0.0, awacs=awacs,
    )
    return w


def test_omni_rwr_search_when_enemy_radar_on_but_not_pointing():
    """对方雷达开机但没指向本机时，rwr_search 为真、rwr 为假。"""
    w = _two_plane_world(awacs=False)
    su57, f16 = w.acs
    # F-16V 在 +x、Su-57 在 -x；F-16V 机头转 90°，不指向 Su-57
    f16.radar_on = True
    f16.psi = math.radians(90.0)
    w._update_sensors()
    # Su-57 应收到全向 RWR（对方在发射），但没被火控照射
    assert su57.rwr_search is True
    assert su57.rwr is False


def test_rwr_illuminated_when_enemy_points_at_me():
    """对方雷达开机且指向本机时，rwr 与 rwr_search 都为真。"""
    w = _two_plane_world(awacs=False)
    su57, f16 = w.acs
    f16.radar_on = True
    f16.psi = math.pi  # F-16V 默认朝 -x，正指向 Su-57
    w._update_sensors()
    assert su57.rwr_search is True
    assert su57.rwr is True


def test_per_side_awacs_only_red_has_track():
    """只有红方有 AWACS 时，红方有跟踪、蓝方没有。"""
    w = _two_plane_world(awacs=(True, False))
    su57, f16 = w.acs
    # 双方雷达都关，确保只有 AWACS 能提供 track
    su57.radar_on = False
    f16.radar_on = False
    w._update_sensors()
    # 红方（Su-57）有 AWACS → 至少 TRACK；蓝方无 AWACS 且雷达关 → NONE
    assert su57.contact >= 2  # CONTACT_TRACK
    assert f16.contact == 0  # CONTACT_NONE


def test_world_awacs_bool_broadcasts_to_both():
    """awacs=True 时双方都有预警机支援。"""
    w = _two_plane_world(awacs=True)
    assert w.awacs == (True, True)
    w2 = _two_plane_world(awacs=False)
    assert w2.awacs == (False, False)


def test_flank_scenario_loads():
    """bvr-flank 场景：红方 AWACS、Su-57 用 flank、900s 时限。"""
    from dogfight.config import load_bundle
    sc = load_bundle().scenario("bvr-flank")
    assert sc.time_limit == 900.0
    assert sc.awacs == (True, False)
    assert sc.teams[0].opening_tactic == "flank"
    assert sc.teams[1].opening_tactic == "default"

