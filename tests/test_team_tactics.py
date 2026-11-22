"""P3 团队决策回归测试：军阶分工、协同原语、火力去重、ROE。"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest

from dogfight.app.main import build_agents, build_world, get_bundle
from dogfight.config.scenario import ConfigError, Rules, load_scenario
from dogfight.core.sim import Action, CONTACT_FIRE
from dogfight.tactics.team import TeamBrain


def _scenario():
    return get_bundle().scenario("j16-j15t-2v2")


def _scenario_4v4():
    return get_bundle().scenario("j20a-f15ex-4v4")


def test_rank_defines_leader_not_health():
    sc = _scenario()
    world = build_world(scenario=sc, seed=3)
    brain = TeamBrain(world)
    leader = brain._leader(0)
    assert world.acs[leader].rank == "flight_lead"
    wingman = next(i for i, ac in enumerate(world.acs) if ac.team == 0 and i != leader)
    # 长机血量远低于僚机，军阶不变则长机不变。
    world.acs[leader].hp = 5.0
    world.acs[wingman].hp = world.acs[wingman].spec.hp_max
    assert brain._leader(0) == leader
    brain.plan()
    assert world.acs[leader].rank == "flight_lead"
    assert world.acs[wingman].rank == "wingman"


def test_rank_priority_and_parsing(tmp_path):
    bundle = get_bundle()
    text = """
id: rank-demo
teams:
  - name: red
    aircraft:
      - id: J-16
        callsign: R1
        rank: wingman
        loadout: {ir: {weapon: PL-10, count: 1}}
      - id: J-16
        callsign: R2
        rank: flight_lead
        loadout: {ir: {weapon: PL-10, count: 1}}
  - name: blue
    aircraft:
      - id: J-15T
        callsign: B1
        loadout: {ir: {weapon: PL-10, count: 1}}
"""
    path = tmp_path / "rank.yaml"
    path.write_text(text, encoding="utf-8")
    sc = load_scenario(path, bundle.aircraft, bundle.weapons)
    # 第一架默认 flight_lead；显式写 wingman 则按军阶让位。
    assert sc.teams[0].slots[0].rank == "wingman"
    assert sc.teams[0].slots[1].rank == "flight_lead"
    world = build_world(scenario=sc, seed=1)
    brain = TeamBrain(world)
    assert brain._leader(0) == 1

    bad = text.replace("rank: flight_lead", "rank: commander")
    path2 = tmp_path / "bad_rank.yaml"
    path2.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError, match="rank"):
        load_scenario(path2, bundle.aircraft, bundle.weapons)


def test_bracket_assigns_opposite_arms():
    sc = _scenario()
    world = build_world(scenario=sc, seed=3)
    brain = TeamBrain(world)
    brain.tactic[0] = "bracket"
    brain._plan_team(0)
    members = [i for i, ac in enumerate(world.acs) if ac.team == 0]
    offsets = [world.acs[i].formation_offset_m for i in members]
    assert offsets[0] < 0 < offsets[1]
    targets = [world.acs[i].assigned_target for i in members]
    assert len(set(targets)) == 2


def test_loose_deuce_releases_wingman_on_leader_attacker():
    sc = _scenario()
    world = build_world(scenario=sc, seed=3)
    brain = TeamBrain(world)
    leader = brain._leader(0)
    wingman = next(i for i, ac in enumerate(world.acs) if ac.team == 0 and i != leader)
    attacker = next(i for i, ac in enumerate(world.acs) if ac.team == 1)
    # 把敌机放到长机尾后。
    la = world.acs[leader]
    ea = world.acs[attacker]
    ea.x = la.x - 2000.0
    ea.y = la.y
    ea.z = la.z
    la.psi = 0.0
    ea.psi = 0.0
    brain.tactic[0] = "loose_deuce"
    brain._plan_team(0)
    assert world.acs[wingman].assigned_target == attacker
    assert world.acs[wingman].order_role == "support"


def test_drag_bait_holds_fire_until_enemy_shoots():
    sc = _scenario()
    world = build_world(scenario=sc, seed=3)
    brain = TeamBrain(world)
    leader = brain._leader(0)
    wingman = next(i for i, ac in enumerate(world.acs) if ac.team == 0 and i != leader)
    brain.tactic[0] = "drag_bait"
    brain._plan_team(0)
    assert world.acs[leader].order_role == "bait"
    assert world.acs[leader].weapons_state == "hold"
    assert world.acs[wingman].weapons_state == "hold"
    # 对手开过火之后，僚机解除射击限制。
    world.stats["missiles_launched"][next(
        i for i, ac in enumerate(world.acs) if ac.team == 1
    )] = 1
    brain._plan_team(0)
    assert world.acs[wingman].weapons_state == "free"

    actions = [Action() for _ in world.acs]
    actions[leader].launch_radar = True
    brain.filter_actions(actions)
    assert not actions[leader].launch_radar


def test_role_exchange_when_leader_is_out_of_missiles():
    sc = _scenario()
    world = build_world(scenario=sc, seed=3)
    brain = TeamBrain(world)
    leader = brain._leader(0)
    wingman = next(i for i, ac in enumerate(world.acs) if ac.team == 0 and i != leader)
    world.acs[leader].missiles = 0
    world.acs[leader].radar_missiles = 0
    brain.tactic[0] = "loose_deuce"
    brain._plan_team(0)
    assert world.acs[wingman].order_role == "engaged"
    assert world.acs[leader].order_role == "support"


def test_thach_weave_on_incoming_missile():
    sc = _scenario()
    world = build_world(scenario=sc, seed=3)
    brain = TeamBrain(world)
    leader = brain._leader(0)
    wingman = next(i for i, ac in enumerate(world.acs) if ac.team == 0 and i != leader)
    shooter = next(i for i, ac in enumerate(world.acs) if ac.team == 1)
    world.missiles.append(SimpleNamespace(alive=True, target=leader, owner=shooter))
    brain.tactic[0] = "thach_weave"
    brain._plan_team(0)
    assert world.acs[wingman].assigned_target == shooter
    assert world.acs[leader].weapons_state == "hold"
    assert world.acs[wingman].weapons_state == "free"


def test_fire_dedup_lets_wingman_hold():
    sc = _scenario()
    world = build_world(scenario=sc, seed=3)
    brain = TeamBrain(world)
    leader = brain._leader(0)
    wingman = next(i for i, ac in enumerate(world.acs) if ac.team == 0 and i != leader)
    target = world.acs[leader].assigned_target = 2
    world.acs[wingman].assigned_target = target
    world.acs[leader].weapons_state = "free"
    world.acs[wingman].weapons_state = "free"
    world.acs[leader].radar_locked_target = target
    actions = [Action() for _ in world.acs]
    actions[leader].launch_radar = True
    actions[wingman].launch_radar = True
    brain.filter_actions(actions)
    assert actions[leader].launch_radar
    assert not actions[wingman].launch_radar


@pytest.mark.parametrize("mode", ["weapons_hold"])
def test_roe_weapons_hold_blocks_launch(mode):
    sc = _scenario()
    world = build_world(scenario=dataclasses.replace(
        sc, rules=Rules(roe=mode),
    ), seed=3)
    world.roe = mode
    ac = world.acs[0]
    ac.radar_on = True
    ac.contact = CONTACT_FIRE
    ac.radar_locked = True
    ac.radar_locked_target = 2
    ac.radar_cd = 0.0
    world._try_launch(0, ac, ac.spec.radar_missile, True)
    assert not world.missiles


def test_no_fire_zone_blocks_launch():
    sc = _scenario()
    world = build_world(scenario=sc, seed=3)
    foe = world.acs[2]
    world.no_fire_zones = ((foe.x, foe.y, 1000.0),)
    ac = world.acs[0]
    ac.radar_on = True
    ac.contact = CONTACT_FIRE
    ac.radar_locked = True
    ac.radar_locked_target = 2
    ac.radar_cd = 0.0
    world._try_launch(0, ac, ac.spec.radar_missile, True)
    assert not world.missiles


def test_rules_parsing_and_validation(tmp_path):
    bundle = get_bundle()
    good = """
id: roe-demo
rules:
  roe: weapons_hold
  team_brain: false
  no_fire_zones:
    - {x: 0, y: 0, radius_m: 8000}
teams:
  - name: red
    aircraft:
      - id: J-16
        callsign: R1
        loadout: {ir: {weapon: PL-10, count: 1}}
  - name: blue
    aircraft:
      - id: J-15T
        callsign: B1
        loadout: {ir: {weapon: PL-10, count: 1}}
"""
    path = tmp_path / "roe.yaml"
    path.write_text(good, encoding="utf-8")
    sc = load_scenario(path, bundle.aircraft, bundle.weapons)
    assert sc.rules.roe == "weapons_hold"
    assert sc.rules.team_brain is False
    assert sc.rules.no_fire_zones == ((0.0, 0.0, 8000.0),)

    bad = good.replace("weapons_hold", "free_fire")
    path2 = tmp_path / "bad_roe.yaml"
    path2.write_text(bad, encoding="utf-8")
    with pytest.raises(ConfigError, match="roe"):
        load_scenario(path2, bundle.aircraft, bundle.weapons)


def test_observation_exposes_energy_state():
    sc = _scenario()
    world = build_world(scenario=sc, seed=3)
    obs = world.observe(0)
    assert obs.own_energy == pytest.approx(world.acs[0].energy)
    assert obs.foe_energy == pytest.approx(world.acs[obs.foe_index].energy)
    assert obs.energy_adv == pytest.approx(obs.own_energy - obs.foe_energy)
    assert obs.alt_adv == pytest.approx(world.acs[0].z - world.acs[obs.foe_index].z)


def test_timeout_uses_energy_after_hp_tie():
    sc = _scenario()
    world = build_world(scenario=sc, seed=3)
    for ac in world.acs:
        ac.hp = ac.spec.hp_max * 0.6
    for i, ac in enumerate(world.acs):
        if ac.team == 0:
            ac.z, ac.v = 10000.0, 350.0
        else:
            ac.z, ac.v = 6000.0, 200.0
    world.t = world.time_limit + 1.0
    world._referee()
    assert world.result == "red"
    assert "能量优势" in world.reason


def test_energy_reset_when_low_and_safe():
    sc = _scenario()
    world = build_world(scenario=sc, seed=3)
    brain = TeamBrain(world)
    world.t = 100.0
    for ac in world.acs:
        if ac.team == 0:
            ac.z = 3000.0
        else:
            ac.x = 6000.0
            ac.y = 0.0
            ac.z = 6000.0
    brain.plan()
    assert brain.tactic[0] == "energy_reset"
    leader = brain._leader(0)
    assert world.acs[leader].team_min_alt == 7000.0
    assert world.acs[leader].weapons_state == "hold"
    assert world.acs[leader].team_disengage
    obs = world.observe(leader)
    assert obs.team_min_alt == 7000.0
    assert obs.team_disengage


def test_vertical_bracket_and_high_low_altitudes():
    sc = _scenario()
    world = build_world(scenario=sc, seed=3)
    brain = TeamBrain(world)
    leader = brain._leader(0)
    wingman = next(i for i, ac in enumerate(world.acs) if ac.team == 0 and i != leader)

    brain.tactic[0] = "vertical_bracket"
    brain._plan_team(0)
    assert world.acs[leader].team_min_alt == 2500.0
    assert world.acs[wingman].team_min_alt == 6500.0
    assert world.acs[leader].formation_offset_m < 0
    assert world.acs[wingman].formation_offset_m > 0

    brain.tactic[0] = "high_low"
    brain._plan_team(0)
    assert world.acs[leader].team_min_alt == 3000.0
    assert world.acs[wingman].team_min_alt == 6500.0
    assert world.acs[wingman].order_role == "support"


def test_reset_starts_after_stalemate():
    sc = _scenario()
    world = build_world(scenario=sc, seed=3)
    brain = TeamBrain(world)
    world.t = 100.0
    brain._last_launch_t[0] = 50.0
    red = [ac for ac in world.acs if ac.team == 0]
    blue = [ac for ac in world.acs if ac.team == 1]
    for k, ac in enumerate(blue, start=1):
        ac.x = 6000.0 + 300.0 * k
        ac.y = 400.0 * k
        ac.z = 6000.0
    for ac in red:
        ac.x, ac.y, ac.z = 0.0, 0.0, 5500.0
        ac.v = 250.0
    brain.plan()
    leader = brain._leader(0)
    assert brain.state[0] == "reset"
    assert brain.tactic[0] == "energy_reset"
    assert world.acs[leader].team_disengage
    assert world.acs[leader].team_min_alt == 7000.0
    assert world.acs[leader].weapons_state == "hold"


def test_reset_reengages_after_climb():
    sc = _scenario()
    world = build_world(scenario=sc, seed=3)
    brain = TeamBrain(world)
    brain.state[0] = "reset"
    for ac in world.acs:
        if ac.team == 0:
            ac.z = 6200.0
    brain.plan()
    assert brain.state[0] == "engage"
    assert brain.tactic[0] != "energy_reset"
    assert not any(ac.team_disengage for ac in world.acs if ac.team == 0)


def test_reset_trap_reengages_high_low():
    sc = _scenario()
    world = build_world(scenario=sc, seed=3)
    brain = TeamBrain(world)
    brain.state[0] = "reset"
    for ac in world.acs:
        if ac.team == 0:
            ac.x = 0.0
            ac.y = 0.0
            ac.z = 5500.0
        else:
            ac.z = 3800.0
            ac.x = 4500.0
            ac.y = 0.0
    brain.plan()
    assert brain.state[0] == "engage"
    assert brain.tactic[0] == "high_low"


def test_agent_extends_when_team_disengages():
    sc = _scenario()
    world = build_world(scenario=sc, seed=3)
    agents = build_agents(scenario=sc, seed=3)
    ac = world.acs[0]
    ac.team_disengage = True
    ac.team_min_alt = 7000.0
    ac.z = 4000.0
    act = agents[0].act(world.observe(0))
    assert agents[0].primitive == "EXTEND"
    assert act.throttle > 0.9


def test_2v2_with_brain_runs_and_is_deterministic():
    sc = _scenario()

    def run():
        from dogfight.app.main import _simulate

        world, agents, _ = _simulate(
            6, 600.0, False, scenario=sc, seed=11,
        )
        return world.result, round(world.t, 2), world.stats["missiles_fired"]

    first = run()
    second = run()
    assert first == second
    assert first[0] in ("red", "blue", "draw")
    assert first[2] > 0


def test_1v1_does_not_use_team_brain():
    sc = get_bundle().scenario("j16-j15t")
    world = build_world(scenario=sc, seed=3)
    assert world.team_of == [0, 1]
    assert all(ac.assigned_target == -1 for ac in world.acs)
    agents = build_agents(scenario=sc, seed=3)
    assert len(agents) == 2


def test_four_ship_forms_two_elements():
    """四机：中队长兼第一小队长，第 3 机为第二小队长，各带一架僚机。"""
    sc = _scenario_4v4()
    world = build_world(scenario=sc, seed=3)
    brain = TeamBrain(world)
    for team in (0, 1):
        elements = brain._elements(team)
        assert len(elements) == 2
        assert all(len(element) == 2 for element in elements)
        for lead, wing in elements:
            assert world.acs[lead].rank in ("flight_lead", "element_lead")
            assert world.acs[wing].rank == "wingman"
        # 两名小队长里必有一名中队长，且中队长统管全队。
        assert brain._leader(team) in (elements[0][0], elements[1][0])


def test_commander_splits_targets_when_enemies_outnumber_elements():
    sc = _scenario_4v4()
    world = build_world(scenario=sc, seed=3)
    brain = TeamBrain(world)
    brain.plan()
    assert brain.focus[0] is False
    elements = brain._elements(0)
    first = {world.acs[i].assigned_target for i in elements[0]}
    second = {world.acs[i].assigned_target for i in elements[1]}
    assert len(first) == 1 and len(second) == 1
    assert first != second


def test_commander_focuses_fire_on_remaining_target():
    sc = _scenario_4v4()
    world = build_world(scenario=sc, seed=3)
    brain = TeamBrain(world)
    world.t = 30.0
    survivors = [i for i, ac in enumerate(world.acs) if ac.team == 1][:1]
    for i, ac in enumerate(world.acs):
        if ac.team == 1 and i not in survivors:
            ac.alive = False
    brain.plan()
    assert brain.focus[0] is True
    targets = {world.acs[i].assigned_target for i in range(4)}
    assert targets == set(survivors)


def test_only_one_shooter_per_target():
    sc = _scenario_4v4()
    world = build_world(scenario=sc, seed=3)
    brain = TeamBrain(world)
    target = next(i for i, ac in enumerate(world.acs) if ac.team == 1)
    for i in range(4):
        world.acs[i].assigned_target = target
        world.acs[i].weapons_state = "free"
    world.acs[0].order_role = "engaged"
    world.acs[1].order_role = "support"
    actions = [Action() for _ in world.acs]
    actions[0].launch_radar = actions[1].launch_radar = True
    actions[0].launch = actions[1].launch = True
    brain.filter_actions(actions)
    assert actions[0].launch_radar and actions[0].launch
    assert not actions[1].launch_radar and not actions[1].launch


def test_no_salvo_while_missile_in_flight_on_healthy_target():
    sc = _scenario_4v4()
    world = build_world(scenario=sc, seed=3)
    brain = TeamBrain(world)
    target = next(i for i, ac in enumerate(world.acs) if ac.team == 1)
    for i in range(4):
        world.acs[i].assigned_target = target
        world.acs[i].weapons_state = "free"
    world.missiles.append(SimpleNamespace(alive=True, target=target, owner=0))
    actions = [Action() for _ in world.acs]
    actions[1].launch_radar = True
    brain.filter_actions(actions)
    assert not actions[1].launch_radar
    # 目标重伤后允许另一架补刀。
    world.acs[target].hp = world.acs[target].spec.hp_max * 0.4
    actions[1].launch_radar = True
    brain.filter_actions(actions)
    assert actions[1].launch_radar

