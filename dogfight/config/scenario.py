"""场景配置：把“用哪两架飞机、怎么开局、跑多久”从入口代码里分离出来。

P1 起支持一个队伍挂多架飞机：`aircraft` 既可以写单个机型字符串（1v1 旧写法），
也可以写飞机列表，每项带 `callsign / role / spawn`。总机数上限 8 架。
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import yaml

from dogfight.specs.loadout import Loadout
from dogfight.specs.weapon import MissileSpec

from . import formations
from .registry import AircraftEntry, ConfigError, parse_loadout

MAX_AIRCRAFT = 8
ROLES = ("fighter", "interceptor")
SIDES = ("attacker", "defender")
RANKS = ("flight_lead", "element_lead", "wingman")
MISSIONS = ("intercept", "sweep")
ROE_MODES = ("weapons_free", "weapons_hold")


@dataclass(frozen=True)
class Rules:
    """交战规则：ROE、禁射区与团队决策开关。"""

    roe: str = "weapons_free"
    # (x, y, radius_m)：武器不得射入该圆形区域（按目标位置判定）。
    no_fire_zones: tuple[tuple[float, float, float], ...] = ()
    # 多机场景默认启用 TeamBrain；1v1 会忽略该开关。
    team_brain: bool = True


@dataclass(frozen=True)
class InitialConditions:
    spacing_m: float = 60000.0   # 每架飞机距空域中心的初始距离
    altitude_m: float = 9000.0
    y_offset_m: float = 100.0
    # 同队多机的横向间隔：第 k 架相对队伍中心线的偏移为 (k-(n-1)/2)*lane_m。
    lane_m: float = 3000.0
    floor_m: float = 1500.0
    ceiling_m: float = 30000.0
    arena_m: float = 200000.0

    @property
    def total_spacing_m(self) -> float:
        return 2.0 * self.spacing_m


@dataclass(frozen=True)
class Spawn:
    """出生点覆盖值；None 表示沿用场景默认几何。"""

    x: float | None = None
    y: float | None = None
    z: float | None = None
    psi: float | None = None   # 度，正北 0°，顺时针
    v: float | None = None     # m/s


@dataclass(frozen=True)
class AircraftSlot:
    """队伍里的一架飞机。"""

    entry: AircraftEntry
    callsign: str
    role: str = "fighter"
    # 军阶/职务：长僚机分工由它决定，与血量/战绩无关。
    # flight_lead（四机长机）/ element_lead（双机长机）/ wingman（僚机）。
    rank: str = "wingman"
    spawn: Spawn | None = None
    loadout: Loadout | None = None
    # None 表示继承队伍级配置。
    radar_policy: str | None = None
    opening_tactic: str | None = None


@dataclass(frozen=True)
class TeamSpec:
    name: str
    slots: tuple[AircraftSlot, ...]
    # 场景级挂载：旧 1v1 写法把它写在队伍上，多机列表里作为各槽默认值。
    loadout: Loadout | None = None
    radar_policy: str = "always"
    mission: str = "sweep"
    # 覆盖机体默认的开局占位策略；None 表示沿用 aircraft.agent.opening_tactic。
    opening_tactic: str | None = None
    # 开局航向（度，正北为 0°，顺时针）。None 表示默认对头几何
    # （红方朝东 0°、蓝方朝西 180°）。
    heading_deg: float | None = None
    # 战术角色：进攻方 / 防守方，只影响开场编队池与 HUD 文案。
    side: str = "attacker"
    # 开场编队池；空表示固定横队。开局时按 seed 从中随机抽一种。
    formation_group: tuple[str, ...] = ()

    @property
    def aircraft(self) -> AircraftEntry:
        """兼容旧 1v1 代码：返回该队第一架飞机的机体条目。"""
        return self.slots[0].entry


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
    rules: Rules = Rules()
    source: str = ""
    # 场景背景（HUD 顶部显示的一行战术说明）。
    brief: str = ""

    @property
    def aircraft_count(self) -> int:
        return sum(len(team.slots) for team in self.teams)

    @property
    def is_duel(self) -> bool:
        """两队各一架：继续走 1v1 快路径。"""
        return self.aircraft_count == 2 and all(
            len(team.slots) == 1 for team in self.teams
        )


def _number(raw: dict, key: str, default: float, source: str) -> float:
    value = raw.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{source}: {key} 必须是数字") from exc


def _parse_spawn(raw: object, source: str) -> Spawn | None:
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}.spawn 必须是键值表")
    extra = set(raw) - {"x", "y", "z", "psi", "v"}
    if extra:
        raise ConfigError(f"{source}.spawn 未知字段 {', '.join(sorted(extra))}")
    values: dict[str, float | None] = {}
    for key in ("x", "y", "z", "psi", "v"):
        if raw.get(key) is None:
            values[key] = None
            continue
        try:
            values[key] = float(raw[key])
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{source}.spawn.{key} 必须是数字") from exc
    return Spawn(**values)


def _parse_slot(
    item: object,
    aircraft: dict[str, AircraftEntry],
    weapons: dict[str, MissileSpec],
    default_loadout: Loadout | None,
    default_radar: str,
    source: str,
    default_rank: str = "wingman",
) -> AircraftSlot:
    if isinstance(item, str):
        item = {"id": item}
    if not isinstance(item, dict):
        raise ConfigError(f"{source} 必须是机型 id 或键值表")
    extra = set(item) - {
        "id", "aircraft", "callsign", "role", "rank", "spawn",
        "loadout", "radar_policy", "opening_tactic",
    }
    if extra:
        raise ConfigError(f"{source} 未知字段 {', '.join(sorted(extra))}")
    aircraft_id = item.get("id", item.get("aircraft"))
    if not isinstance(aircraft_id, str) or not aircraft_id:
        raise ConfigError(f"{source}.id 必须填机体 id")
    if aircraft_id not in aircraft:
        raise ConfigError(f"{source}.id: 未注册的机体 id {aircraft_id!r}")

    entry = aircraft[aircraft_id]
    loadout = default_loadout
    if item.get("loadout") is not None:
        loadout = parse_loadout(item["loadout"], weapons, source)
    if loadout is None:
        raise ConfigError(f"{source}.loadout 必填（武器挂载属于场景配置）")
    entry = replace(entry, spec=replace(
        entry.spec,
        loadout=loadout,
        mass=entry.spec.base_mass + loadout.weapon_mass,
    ))

    role = item.get("role", "fighter")
    if role not in ROLES:
        raise ConfigError(f"{source}.role 只支持 {' / '.join(ROLES)}")

    rank = item.get("rank", default_rank)
    if rank not in RANKS:
        raise ConfigError(f"{source}.rank 只支持 {' / '.join(RANKS)}")

    policy = item.get("radar_policy", default_radar)
    if policy not in ("always", "silent", "auto"):
        raise ConfigError(f"{source}.radar_policy 只支持 always / silent / auto")

    opening_tactic = item.get("opening_tactic")
    if opening_tactic is not None and opening_tactic not in (
        "auto", "default", "stealth_compress", "flank",
    ):
        raise ConfigError(
            f"{source}.opening_tactic 只支持 "
            f"auto / default / stealth_compress / flank"
        )

    callsign = item.get("callsign", aircraft_id)
    if not isinstance(callsign, str) or not callsign:
        raise ConfigError(f"{source}.callsign 必须是非空字符串")

    return AircraftSlot(
        entry=entry,
        callsign=callsign,
        role=role,
        rank=rank,
        spawn=_parse_spawn(item.get("spawn"), source),
        loadout=loadout,
        radar_policy=policy,
        opening_tactic=opening_tactic,
    )


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
        "decision_hz", "seed", "awacs", "rules", "brief",
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
    callsigns: set[str] = set()
    total = 0
    for index, item in enumerate(teams_raw):
        where = f"teams[{index}]"
        if not isinstance(item, dict):
            raise ConfigError(f"{path}: {where} 必须是键值表")
        extra = set(item) - {
            "name", "aircraft", "loadout", "radar_policy",
            "opening_tactic", "heading_deg", "mission",
            "side", "formation_group",
        }
        if extra:
            raise ConfigError(f"{path}: {where} 未知字段 {', '.join(sorted(extra))}")

        name = item.get("name", f"team{index}")
        if not isinstance(name, str) or not name:
            raise ConfigError(f"{path}: {where}.name 必须是非空字符串")

        policy = item.get("radar_policy", "always")
        if policy not in ("always", "silent", "auto"):
            raise ConfigError(
                f"{path}: {where}.radar_policy 只支持 always / silent / auto"
            )
        mission = item.get("mission", "sweep")
        if mission not in MISSIONS:
            raise ConfigError(f"{path}: {where}.mission 只支持 {' / '.join(MISSIONS)}")

        team_loadout = None
        if item.get("loadout") is not None:
            team_loadout = parse_loadout(item["loadout"], weapons, f"{path}: {where}")

        opening_tactic = item.get("opening_tactic")
        if opening_tactic is not None and opening_tactic not in (
            "auto", "default", "stealth_compress", "flank",
        ):
            raise ConfigError(
                f"{path}: {where}.opening_tactic 只支持 "
                f"auto / default / stealth_compress / flank"
            )

        heading_deg = item.get("heading_deg")
        if heading_deg is not None:
            try:
                heading_deg = float(heading_deg)
            except (TypeError, ValueError) as exc:
                raise ConfigError(f"{path}: {where}.heading_deg 必须是数字") from exc

        aircraft_raw = item.get("aircraft")
        if isinstance(aircraft_raw, (str, dict)):
            aircraft_items = [aircraft_raw]
        elif isinstance(aircraft_raw, list) and aircraft_raw:
            aircraft_items = aircraft_raw
        else:
            raise ConfigError(
                f"{path}: {where}.aircraft 必须是机型 id、键值表或非空列表"
            )

        slots = tuple(
            _parse_slot(
                entry, aircraft, weapons, team_loadout, policy,
                f"{path}: {where}.aircraft[{k}]",
                default_rank="flight_lead" if k == 0 else "wingman",
            )
            for k, entry in enumerate(aircraft_items)
        )
        for slot in slots:
            if slot.callsign in callsigns:
                raise ConfigError(f"{path}: 呼号重复 {slot.callsign!r}")
            callsigns.add(slot.callsign)
        total += len(slots)

        # 默认第一队为进攻方、第二队为防守方；YAML 可显式覆盖。
        side = item.get("side", SIDES[min(index, 1)])
        if side not in SIDES:
            raise ConfigError(
                f"{path}: {where}.side 只支持 {' / '.join(SIDES)}"
            )
        group_raw = item.get("formation_group", []) or []
        if not isinstance(group_raw, list):
            raise ConfigError(f"{path}: {where}.formation_group 必须是列表")
        group: list[str] = []
        for fid in group_raw:
            if not isinstance(fid, str) or fid not in formations.ALL:
                raise ConfigError(
                    f"{path}: {where}.formation_group 含未知编队 {fid!r}，"
                    f"可选：{', '.join(formations.ALL)}"
                )
            if not formations.supported(fid, len(slots)):
                raise ConfigError(
                    f"{path}: {where}.formation_group 的 {fid!r} "
                    f"不支持 {len(slots)} 机编队"
                )
            group.append(fid)
        teams.append(TeamSpec(
            name=name, slots=slots, loadout=team_loadout,
            radar_policy=policy, mission=mission,
            opening_tactic=opening_tactic, heading_deg=heading_deg,
            side=side, formation_group=tuple(group),
        ))

    if total > MAX_AIRCRAFT:
        raise ConfigError(
            f"{path}: 全场飞机总数 {total} 超过上限 {MAX_AIRCRAFT}"
        )

    initial_raw = raw.get("initial", {})
    if not isinstance(initial_raw, dict):
        raise ConfigError(f"{path}: initial 必须是键值表")
    extra = set(initial_raw) - {
        "spacing_m", "altitude_m", "y_offset_m", "lane_m",
        "floor_m", "ceiling_m", "arena_m",
    }
    if extra:
        raise ConfigError(f"{path}: initial 未知字段 {', '.join(sorted(extra))}")
    initial = InitialConditions(
        spacing_m=_number(initial_raw, "spacing_m", 60000.0, f"{path}: initial"),
        altitude_m=_number(initial_raw, "altitude_m", 9000.0, f"{path}: initial"),
        y_offset_m=_number(initial_raw, "y_offset_m", 100.0, f"{path}: initial"),
        lane_m=_number(initial_raw, "lane_m", 3000.0, f"{path}: initial"),
        floor_m=_number(initial_raw, "floor_m", 1500.0, f"{path}: initial"),
        ceiling_m=_number(initial_raw, "ceiling_m", 30000.0, f"{path}: initial"),
        arena_m=_number(initial_raw, "arena_m", 200000.0, f"{path}: initial"),
    )
    if initial.spacing_m <= 0.0 or initial.arena_m <= 0.0:
        raise ConfigError(f"{path}: initial.spacing_m / arena_m 必须为正")
    if initial.lane_m < 0.0:
        raise ConfigError(f"{path}: initial.lane_m 不能为负")
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

    rules_raw = raw.get("rules", {})
    if not isinstance(rules_raw, dict):
        raise ConfigError(f"{path}: rules 必须是键值表")
    extra = set(rules_raw) - {"roe", "no_fire_zones", "team_brain"}
    if extra:
        raise ConfigError(f"{path}: rules 未知字段 {', '.join(sorted(extra))}")
    roe = rules_raw.get("roe", "weapons_free")
    if roe not in ROE_MODES:
        raise ConfigError(f"{path}: rules.roe 只支持 {' / '.join(ROE_MODES)}")
    zones_raw = rules_raw.get("no_fire_zones", []) or []
    if not isinstance(zones_raw, list):
        raise ConfigError(f"{path}: rules.no_fire_zones 必须是列表")
    zones: list[tuple[float, float, float]] = []
    for k, zone in enumerate(zones_raw):
        if not isinstance(zone, dict):
            raise ConfigError(f"{path}: rules.no_fire_zones[{k}] 必须是键值表")
        z_extra = set(zone) - {"x", "y", "radius_m"}
        if z_extra:
            raise ConfigError(
                f"{path}: rules.no_fire_zones[{k}] 未知字段 "
                f"{', '.join(sorted(z_extra))}"
            )
        try:
            zx = float(zone.get("x", 0.0))
            zy = float(zone.get("y", 0.0))
            zr = float(zone.get("radius_m", 0.0))
        except (TypeError, ValueError) as exc:
            raise ConfigError(
                f"{path}: rules.no_fire_zones[{k}] 参数必须是数字"
            ) from exc
        if zr <= 0.0:
            raise ConfigError(
                f"{path}: rules.no_fire_zones[{k}].radius_m 必须为正"
            )
        zones.append((zx, zy, zr))
    rules = Rules(
        roe=roe,
        no_fire_zones=tuple(zones),
        team_brain=bool(rules_raw.get("team_brain", True)),
    )

    brief_raw = raw.get("brief", "")
    if not isinstance(brief_raw, str):
        raise ConfigError(f"{path}: brief 必须是字符串")
    return Scenario(
        id=scenario_id,
        teams=tuple(teams),
        initial=initial,
        brief=brief_raw.strip(),
        time_limit=_number(raw, "time_limit_s", 300.0, str(path)),
        physics_dt=_number(raw, "physics_dt_s", 1.0 / 120.0, str(path)),
        decision_hz=_number(raw, "decision_hz", 10.0, str(path)),
        seed=seed,
        awacs=awacs,
        rules=rules,
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
