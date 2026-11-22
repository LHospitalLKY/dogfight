"""多机数据模型与团队裁判（P1 + 简化 P2）回归测试。"""
from __future__ import annotations

import pytest

from dogfight.app.main import build_agents, build_world, get_bundle
from dogfight.config.scenario import ConfigError, load_scenario
from dogfight.core.sim import Action


def _bundle():
    return get_bundle()


def test_2v2_scenario_loads():
    sc = _bundle().scenario("j16-j15t-2v2")
    assert sc.aircraft_count == 4
    assert not sc.is_duel
    assert [slot.callsign for team in sc.teams for slot in team.slots] == [
        "Flanker1", "Flanker2", "Shark1", "Shark2",
    ]
    assert [slot.role for team in sc.teams for slot in team.slots] == [
        "fighter", "fighter", "interceptor", "interceptor",
    ]
    # 队伍级 aircraft 属性仍返回首架，兼容旧调用方。
    assert sc.teams[0].aircraft.id == "J-16"
    assert sc.teams[1].aircraft.id == "J-15T"


def test_duel_flag_preserved():
    for name in ("j16-j15t", "bvr-1v1"):
        sc = _bundle().scenario(name)
        assert sc.is_duel


def test_world_spawns_two_per_team():
    sc = _bundle().scenario("j16-j15t-2v2")
    world = build_world(scenario=sc, seed=3)
    assert len(world.acs) == 4
    assert world.team_of == [0, 0, 1, 1]
    # 双方分居空域两侧；队内几何由抽到的编队决定，不再固定为横队。
    assert world.acs[0].x < 0 < world.acs[2].x
    assert world.acs[1].x < 0 < world.acs[3].x
    assert len(world.formation_ids) == 2
    for fid, team in zip(world.formation_ids, sc.teams):
        assert fid in team.formation_group
    assert [ac.callsign for ac in world.acs] == [
        "Flanker1", "Flanker2", "Shark1", "Shark2",
    ]


def test_agents_are_per_aircraft():
    sc = _bundle().scenario("j16-j15t-2v2")
    world = build_world(scenario=sc, seed=3)
    agents = build_agents(scenario=sc, seed=3)
    assert len(agents) == len(world.acs) == 4


def test_contacts_and_primary_are_enemy_only():
    sc = _bundle().scenario("j16-j15t-2v2")
    world = build_world(scenario=sc, seed=3)
    agents = build_agents(scenario=sc, seed=3)
    for _ in range(12):
        actions = [ag.act(world.observe(i)) for i, ag in enumerate(agents)]
        world.step(actions)
    for i, ac in enumerate(world.acs):
        assert len(ac.contacts) == 2
        assert all(world.acs[c.index].team != ac.team for c in ac.contacts)
        assert 0 <= ac.primary < len(world.acs)
        assert world.acs[ac.primary].team != ac.team
        obs = world.observe(i)
        assert obs.foe_index == ac.primary
        assert len(obs.contacts) == 2


def test_team_settlement_on_timeout():
    """4 机 1 s 时限：无人开火时按队伍血量结算（应为平局或消解为 draw）。"""
    sc = _bundle().scenario("j16-j15t-2v2")
    world = build_world(time_limit=1.0, scenario=sc, seed=3)
    idle = [Action() for _ in world.acs]
    while world.result is None:
        world.step(idle)
    assert world.result in ("red", "blue", "draw")
    assert world.reason_code in ("timeout", "draw")


def test_aircraft_limit_and_role_validation(tmp_path):
    bundle = _bundle()
    base = """
id: too-many
teams:
  - name: red
    aircraft:
{red}
  - name: blue
    aircraft:
{blue}
"""
    slot = (
        "      - id: J-16\n"
        "        callsign: R{k}\n"
        "        loadout: {{ir: {{weapon: PL-10, count: 1}}}}\n"
    )
    red = "".join(slot.format(k=k) for k in range(5))
    blue = "".join(slot.format(k=k) for k in range(5)).replace("R", "B")
    path = tmp_path / "too_many.yaml"
    path.write_text(base.format(red=red, blue=blue), encoding="utf-8")
    with pytest.raises(ConfigError, match="上限"):
        load_scenario(path, bundle.aircraft, bundle.weapons)

    bad_role = base.format(
        red="      - id: J-16\n        callsign: R0\n        role: bomber\n"
            "        loadout: {ir: {weapon: PL-10, count: 1}}\n",
        blue="      - id: J-15T\n        callsign: B0\n"
             "        loadout: {ir: {weapon: PL-10, count: 1}}\n",
    )
    path2 = tmp_path / "bad_role.yaml"
    path2.write_text(bad_role, encoding="utf-8")
    with pytest.raises(ConfigError, match="role"):
        load_scenario(path2, bundle.aircraft, bundle.weapons)
