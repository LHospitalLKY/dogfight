"""武器/机体注册表：把 YAML 文件解析成冻结数据类。

寄存器只依赖 specs 包；挂载用武器 id 引用，两阶段解析（先弹后机），
避免机体与武器之间出现循环引用。
"""
from __future__ import annotations

import math
from dataclasses import MISSING, dataclass, fields
from pathlib import Path

import yaml

from dogfight.specs import AircraftSpec, Loadout, MissileSpec, WeaponSlot

from .schema import (
    AIRCRAFT_FRAME_FIELDS,
    AIRCRAFT_SENSOR_FIELDS,
    ConfigError,
    WEAPON_FIELDS,
    apply_fields,
)


@dataclass(frozen=True)
class AgentPersona:
    """随机占位与战术倾向。跟随机体一起配置，便于整体做镜像实验。"""

    aggression: float = 0.5
    energy_pref: float = 0.5
    lure: bool = False
    # auto 沿用 energy_pref 的旧启发式；energy/angle 显式指定开局风格。
    style: str = "auto"


@dataclass(frozen=True)
class AircraftEntry:
    id: str
    spec: AircraftSpec
    agent: AgentPersona


def _read_yaml(path: Path) -> dict:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"{path}: 读取失败（{exc}）") from exc
    if raw is None:
        raise ConfigError(f"{path}: 文件为空")
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: 顶层必须是键值表")
    return raw


def _missing_fields(cls) -> list[str]:
    return [
        f.name for f in fields(cls)
        if f.default is MISSING and f.default_factory is MISSING
    ]


def parse_weapon(raw: dict, source: str) -> MissileSpec:
    unknown = set(raw) - set(WEAPON_FIELDS) - {"id"}
    if unknown:
        raise ConfigError(f"{source}: 未知字段 {', '.join(sorted(unknown))}")
    values = apply_fields(
        raw, WEAPON_FIELDS, source, "weapon", ignore={"id"}
    )
    missing = [name for name in _missing_fields(MissileSpec) if name not in values]
    if missing:
        raise ConfigError(f"{source}: 缺少必填字段 {', '.join(missing)}")
    try:
        spec = MissileSpec(**values)
    except TypeError as exc:
        raise ConfigError(f"{source}: {exc}") from exc

    if spec.gimbal < spec.seeker_fov:
        raise ConfigError(f"{source}: gimbal_fov_deg 不能小于 seeker_fov_deg")
    if spec.min_range > spec.max_range:
        raise ConfigError(f"{source}: min_range_m 不能大于 max_range_km")
    if not 0.0 < spec.burnout_mass < spec.mass:
        raise ConfigError(f"{source}: burnout_mass_kg 必须小于 mass_kg 且大于 0")
    if not 0.0 < spec.seeker_fov < math.pi:
        raise ConfigError(f"{source}: seeker_fov_deg 必须在 (0, 180)")
    if spec.guidance.upper() not in ("IR", "SARH", "ARH"):
        raise ConfigError(f"{source}: guidance 只支持 IR / SARH / ARH")
    if spec.boost_time > 0.0 and spec.boost_thrust <= 0.0:
        raise ConfigError(f"{source}: 配置 boost_time_s 时必须给 boost_thrust_kn")
    return spec


def parse_loadout(raw: object, weapons: dict[str, MissileSpec], source: str) -> Loadout:
    if raw is None:
        return Loadout()
    if not isinstance(raw, dict):
        raise ConfigError(f"{source}: loadout 必须是键值表")
    unknown = set(raw) - {"ir", "radar", "countermeasures"}
    if unknown:
        raise ConfigError(f"{source}: loadout 未知字段 {', '.join(sorted(unknown))}")

    def slot(name: str) -> WeaponSlot | None:
        item = raw.get(name)
        if item is None:
            return None
        if not isinstance(item, dict):
            raise ConfigError(f"{source}: loadout.{name} 必须是键值表")
        extra = set(item) - {"weapon", "count"}
        if extra:
            raise ConfigError(f"{source}: loadout.{name} 未知字段 {', '.join(sorted(extra))}")
        weapon_id = item.get("weapon")
        count = item.get("count", 1)
        if not isinstance(weapon_id, str) or not weapon_id:
            raise ConfigError(f"{source}: loadout.{name}.weapon 必须是武器 id")
        if weapon_id not in weapons:
            raise ConfigError(
                f"{source}: loadout.{name}.weapon: 未注册的武器 id {weapon_id!r}"
            )
        try:
            count_i = int(count)
        except (TypeError, ValueError) as exc:
            raise ConfigError(f"{source}: loadout.{name}.count 非法") from exc
        if count_i < 0:
            raise ConfigError(f"{source}: loadout.{name}.count 不能为负")
        return WeaponSlot(weapons[weapon_id], count_i)

    cm = raw.get("countermeasures", {})
    if cm is None:
        cm = {}
    if not isinstance(cm, dict):
        raise ConfigError(f"{source}: loadout.countermeasures 必须是键值表")
    extra = set(cm) - {"chaff", "flare"}
    if extra:
        raise ConfigError(
            f"{source}: loadout.countermeasures 未知字段 {', '.join(sorted(extra))}"
        )
    try:
        chaff = int(cm.get("chaff", 0))
        flare = int(cm.get("flare", 0))
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{source}: loadout.countermeasures 数量非法") from exc
    if chaff < 0 or flare < 0:
        raise ConfigError(f"{source}: 干扰物数量不能为负")
    return Loadout(ir=slot("ir"), radar=slot("radar"), chaff=chaff, flare=flare)


def parse_aircraft(raw: dict, weapons: dict[str, MissileSpec], source: str) -> AircraftEntry:
    unknown = set(raw) - {"id", "name", "airframe", "sensors", "loadout", "agent"}
    if unknown:
        raise ConfigError(f"{source}: 未知字段 {', '.join(sorted(unknown))}")
    name = raw.get("name")
    if not isinstance(name, str) or not name:
        raise ConfigError(f"{source}: name 必须是非空字符串")
    entry_id = raw.get("id", name)
    if not isinstance(entry_id, str) or not entry_id:
        raise ConfigError(f"{source}: id 必须是非空字符串")

    frame = apply_fields(raw.get("airframe", {}), AIRCRAFT_FRAME_FIELDS, source, "airframe")
    sensors = apply_fields(raw.get("sensors", {}), AIRCRAFT_SENSOR_FIELDS, source, "sensors")
    loadout = parse_loadout(raw.get("loadout"), weapons, source)

    base_mass = frame.pop("base_mass", None)
    if base_mass is None:
        raise ConfigError(f"{source}: airframe.base_mass_kg 必填")
    if base_mass <= 0.0:
        raise ConfigError(f"{source}: airframe.base_mass_kg 必须为正")
    if frame.get("wing_area", 0.0) <= 0.0:
        raise ConfigError(f"{source}: airframe.wing_area_m2 必须为正")
    if frame.get("thrust_max", 0.0) <= 0.0:
        raise ConfigError(f"{source}: airframe.thrust_max_kn 必须为正")
    if not 0.0 < frame.get("cd0", 0.0) < 1.0:
        raise ConfigError(f"{source}: airframe.cd0 必须在 (0, 1) 内")
    if frame.get("aspect_ratio", 0.0) <= 0.0:
        raise ConfigError(f"{source}: airframe.aspect_ratio 必须为正")
    if frame.get("n_struct", 0.0) <= 1.0:
        raise ConfigError(f"{source}: airframe.n_struct_g 必须大于 1")
    frame["mass"] = base_mass + loadout.weapon_mass

    missing = [
        field for field in ("mass", "wing_area", "cl_max", "n_struct", "thrust_max",
                            "cd0", "aspect_ratio")
        if field not in frame
    ]
    if missing:
        raise ConfigError(f"{source}: 缺少必填的机体参数 {', '.join(missing)}")

    try:
        spec = AircraftSpec(name=name, loadout=loadout, **{**frame, **sensors})
    except TypeError as exc:
        raise ConfigError(f"{source}: {exc}") from exc

    agent_raw = raw.get("agent", {})
    if agent_raw is None:
        agent_raw = {}
    if not isinstance(agent_raw, dict):
        raise ConfigError(f"{source}: agent 必须是键值表")
    extra = set(agent_raw) - {"aggression", "energy_pref", "lure", "style"}
    if extra:
        raise ConfigError(f"{source}: agent 未知字段 {', '.join(sorted(extra))}")
    try:
        agent = AgentPersona(
            aggression=float(agent_raw.get("aggression", 0.5)),
            energy_pref=float(agent_raw.get("energy_pref", 0.5)),
            lure=bool(agent_raw.get("lure", False)),
            style=str(agent_raw.get("style", "auto")),
        )
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{source}: agent 参数非法（{exc}）") from exc
    if agent.style not in ("auto", "energy", "angle"):
        raise ConfigError(f"{source}: agent.style 只支持 auto / energy / angle")
    if not 0.0 <= agent.aggression <= 1.0 or not 0.0 <= agent.energy_pref <= 1.0:
        raise ConfigError(f"{source}: agent.aggression / energy_pref 必须在 [0, 1]")
    return AircraftEntry(id=entry_id, spec=spec, agent=agent)


def load_weapons(directory: str | Path) -> dict[str, MissileSpec]:
    directory = Path(directory)
    if not directory.is_dir():
        raise ConfigError(f"{directory}: 武器配置目录不存在")
    weapons: dict[str, MissileSpec] = {}
    for path in sorted(directory.glob("*.yaml")):
        raw = _read_yaml(path)
        spec = parse_weapon(raw, str(path))
        weapon_id = raw.get("id", spec.name)
        if not isinstance(weapon_id, str) or not weapon_id:
            raise ConfigError(f"{path}: id 必须是非空字符串")
        if weapon_id in weapons:
            raise ConfigError(f"{path}: 武器 id 重复 {weapon_id!r}")
        weapons[weapon_id] = spec
    if not weapons:
        raise ConfigError(f"{directory}: 没有找到任何武器配置")
    return weapons


def load_aircraft(
    directory: str | Path, weapons: dict[str, MissileSpec]
) -> dict[str, AircraftEntry]:
    directory = Path(directory)
    if not directory.is_dir():
        raise ConfigError(f"{directory}: 机体配置目录不存在")
    entries: dict[str, AircraftEntry] = {}
    for path in sorted(directory.glob("*.yaml")):
        raw = _read_yaml(path)
        entry = parse_aircraft(raw, weapons, str(path))
        if entry.id in entries:
            raise ConfigError(f"{path}: 机体 id 重复 {entry.id!r}")
        entries[entry.id] = entry
    if not entries:
        raise ConfigError(f"{directory}: 没有找到任何机体配置")
    return entries


def load_aircraft_and_weapons(config_dir: str | Path) -> tuple[dict, dict]:
    """从 configs/ 目录一次加载武器与机体。"""
    config_dir = Path(config_dir)
    weapons = load_weapons(config_dir / "weapons")
    aircraft = load_aircraft(config_dir / "aircraft", weapons)
    return weapons, aircraft
