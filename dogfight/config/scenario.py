"""场景配置：把“用哪两架飞机、怎么开局、跑多久”从入口代码里分离出来。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from dogfight.specs.loadout import Loadout
from dogfight.specs.weapon import MissileSpec

from .registry import AircraftEntry, ConfigError, parse_loadout


@dataclass(frozen=True)
class InitialConditions:
    spacing_m: float = 60000.0   # 每架飞机距空域中心的初始距离
    altitude_m: float = 9000.0
    y_offset_m: float = 100.0
    floor_m: float = 1500.0
    ceiling_m: float = 30000.0
    arena_m: float = 200000.0

    @property
    def total_spacing_m(self) -> float:
        return 2.0 * self.spacing_m


@dataclass(frozen=True)
class TeamSpec:
    name: str
    aircraft: AircraftEntry
    # 该队在本次场景里的挂载。武器挂载属于场景配置：同一机体在不同场景
    # 可以挂不同弹，作战全重也随挂载质量变化。
    loadout: Loadout | None = None
    radar_policy: str = "always"
    # 覆盖机体默认的开局占位策略；None 表示沿用 aircraft.agent.opening_tactic。
    opening_tactic: str | None = None
    # 开局航向（度，正北为 0°，顺时针）。None 表示默认对头几何
    # （红方朝东 0°、蓝方朝西 180°）。
    heading_deg: float | None = None


@dataclass(frozen=True)
class Scenario:
    id: str
    teams: tuple[TeamSpec, ...]
    initial: InitialConditions = InitialConditions()
    time_limit: float = 300.0
    physics_dt: float = 1.0 / 120.0
    decision_hz: float = 10.0
    seed: int | None = None
    # True 表示双方都有预警机支援；(bool, bool) 分别指定红/蓝双方。
    awacs: bool | tuple[bool, bool] = True
    source: str = ""


def _number(raw: dict, key: str, default: float, source: str) -> float:
    value = raw.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{source}: {key} 必须是数字") from exc


def load_scenario(
    path: str | Path,
    aircraft: dict[str, AircraftEntry],
    weapons: dict[str, MissileSpec],
) -> Scenario:
    path = Path(path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"{path}: 读取失败（{exc}）") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: 顶层必须是键值表")

    allowed = {
        "id", "teams", "initial", "time_limit_s", "physics_dt_s",
        "decision_hz", "seed", "awacs",
    }
    unknown = set(raw) - allowed
    if unknown:
        raise ConfigError(f"{path}: 未知字段 {', '.join(sorted(unknown))}")

    scenario_id = raw.get("id")
    if not isinstance(scenario_id, str) or not scenario_id:
        raise ConfigError(f"{path}: id 必须是非空字符串")

    teams_raw = raw.get("teams")
    if not isinstance(teams_raw, list) or len(teams_raw) != 2:
        raise ConfigError(f"{path}: teams 必须恰好包含 2 个队伍")
    teams: list[TeamSpec] = []
    for index, item in enumerate(teams_raw):
        where = f"teams[{index}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{path}: {where} 必须是键值表")
        extra = set(item) - {
            "name", "aircraft", "loadout", "radar_policy", "opening_tactic", "heading_deg",
        }
        if extra:
            raise ConfigError(f"{path}: {where} 未知字段 {', '.join(sorted(extra))}")
        aircraft_id = item.get("aircraft")
        if not isinstance(aircraft_id, str) or not aircraft_id:
            raise ConfigError(f"{path}: {where}.aircraft 必须填机体 id")
        if aircraft_id not in aircraft:
            raise ConfigError(
                f"{path}: {where}.aircraft: 未注册的机体 id {aircraft_id!r}"
            )
        if item.get("loadout") is None:
            raise ConfigError(f"{path}: {where}.loadout 必填（武器挂载属于场景配置）")
        loadout = parse_loadout(item["loadout"], weapons, f"{path}: {where}")
        entry = aircraft[aircraft_id]
        # 把场景挂载装到机体上：作战全重 = 空机+燃油质量 + 挂载质量。
        entry = replace(entry, spec=replace(
            entry.spec,
            loadout=loadout,
            mass=entry.spec.base_mass + loadout.weapon_mass,
        ))
        policy = item.get("radar_policy", "always")
        if policy not in ("always", "silent", "auto"):
            raise ConfigError(
                f"{path}: {where}.radar_policy 只支持 always / silent / auto"
            )
        opening_tactic = item.get("opening_tactic")
        if opening_tactic is not None:
            if opening_tactic not in ("auto", "default", "stealth_compress", "flank"):
                raise ConfigError(
                    f"{path}: {where}.opening_tactic 只支持 "
                    f"auto / default / stealth_compress / flank"
                )
        name = item.get("name", f"team{index}")
        if not isinstance(name, str) or not name:
            raise ConfigError(f"{path}: {where}.name 必须是非空字符串")
        heading_deg = item.get("heading_deg")
        if heading_deg is not None:
            try:
                heading_deg = float(heading_deg)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{path}: {where}.heading_deg 必须是数字") from exc
        teams.append(TeamSpec(
            name=name, aircraft=entry, loadout=loadout,
            radar_policy=policy, opening_tactic=opening_tactic,
            heading_deg=heading_deg,
        ))

    initial_raw = raw.get("initial", {})
    if not isinstance(initial_raw, dict):
        raise ConfigError(f"{path}: initial 必须是键值表")
    extra = set(initial_raw) - {
        "spacing_m", "altitude_m", "y_offset_m", "floor_m", "ceiling_m", "arena_m",
    }
    if extra:
        raise ConfigError(f"{path}: initial 未知字段 {', '.join(sorted(extra))}")
    initial = InitialConditions(
        spacing_m=_number(initial_raw, "spacing_m", 60000.0, f"{path}: initial"),
        altitude_m=_number(initial_raw, "altitude_m", 9000.0, f"{path}: initial"),
        y_offset_m=_number(initial_raw, "y_offset_m", 100.0, f"{path}: initial"),
        floor_m=_number(initial_raw, "floor_m", 1500.0, f"{path}: initial"),
        ceiling_m=_number(initial_raw, "ceiling_m", 30000.0, f"{path}: initial"),
        arena_m=_number(initial_raw, "arena_m", 200000.0, f"{path}: initial"),
    )
    if initial.spacing_m <= 0.0 or initial.arena_m <= 0.0:
        raise ConfigError(f"{path}: initial.spacing_m / arena_m 必须为正")
    if initial.floor_m >= initial.ceiling_m:
        raise ConfigError(f"{path}: initial.floor_m 必须小于 ceiling_m")

    seed_raw = raw.get("seed")
    seed = None if seed_raw is None else int(seed_raw)
    awacs_raw = raw.get("awacs", True)
    if isinstance(awacs_raw, bool):
        awacs: bool | tuple[bool, bool] = awacs_raw
    elif isinstance(awacs_raw, (list, tuple)) and len(awacs_raw) == 2:
        awacs = (bool(awacs_raw[0]), bool(awacs_raw[1]))
    else:
        raise ConfigError(f"{path}: awacs 必须是 true/false 或长度为 2 的布尔列表")

    return Scenario(
        id=scenario_id,
        teams=tuple(teams),
        initial=initial,
        time_limit=_number(raw, "time_limit_s", 300.0, str(path)),
        physics_dt=_number(raw, "physics_dt_s", 1.0 / 120.0, str(path)),
        decision_hz=_number(raw, "decision_hz", 10.0, str(path)),
        seed=seed,
        awacs=awacs,
        source=str(path),
    )


def load_scenarios(
    directory: str | Path,
    aircraft: dict[str, AircraftEntry],
    weapons: dict[str, MissileSpec],
) -> dict[str, Scenario]:
    directory = Path(directory)
    if not directory.is_dir():
        raise ConfigError(f"{directory}: 场景配置目录不存在")
    scenarios: dict[str, Scenario] = {}
    for path in sorted(directory.glob("*.yaml")):
        scenario = load_scenario(path, aircraft, weapons)
        if scenario.id in scenarios:
            raise ConfigError(f"{path}: 场景 id 重复 {scenario.id!r}")
        scenarios[scenario.id] = scenario
    return scenarios
