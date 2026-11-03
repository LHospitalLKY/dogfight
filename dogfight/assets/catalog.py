"""资产注册表：YAML → 冻结数据类，供 three.js 导出与轮廓生成共用。

目录只描述“资源在哪、按多长缩放、怎么换算坐标”，不包含任何机型分支；
渲染与导出代码按条目遍历，新增机体时不需要改代码。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

from dogfight.config import ConfigError

DEFAULT_CATALOG = (
    Path(__file__).resolve().parents[2] / "configs" / "assets" / "models.yaml"
)


@dataclass(frozen=True)
class WebAsset:
    """three.js 导出参数；converter 决定原始 OBJ 的轴向换算方式。"""

    source: Path
    file: str
    converter: str


@dataclass(frozen=True)
class AircraftAsset:
    id: str
    length_m: float
    # 体素抽稀尺度（米）：Web 兜底与俯视轮廓共用同一份原始几何。
    voxel_m: float
    # 仅体素简化时保留的材质；Web 端仍渲染完整原始模型。
    keep_materials: frozenset[str] | None
    web: WebAsset


@dataclass(frozen=True)
class MissileAsset:
    id: str
    source: Path
    length_m: float
    file: str


@dataclass(frozen=True)
class AssetCatalog:
    version: str
    root: Path
    aircraft: dict[str, AircraftAsset]
    missiles: dict[str, MissileAsset]

    def aircraft_asset(self, name: str) -> AircraftAsset | None:
        return self.aircraft.get(name)

    def missile_asset(self, name: str) -> MissileAsset | None:
        return self.missiles.get(name)


def _number(raw: dict, key: str, where: str, *, required: bool = True):
    value = raw.get(key)
    if value is None:
        if required:
            raise ConfigError(f"{where}: 缺少必填字段 {key}")
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where}.{key} 必须是数字") from exc


def _path(base: Path, raw, where: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise ConfigError(f"{where} 必须是非空路径")
    path = Path(raw)
    return path if path.is_absolute() else base / path


def _parse_web(raw: dict, base: Path, where: str) -> WebAsset:
    unknown = set(raw) - {"source", "file", "converter"}
    if unknown:
        raise ConfigError(f"{where}: 未知字段 {', '.join(sorted(unknown))}")
    file = raw.get("file")
    if not isinstance(file, str) or not file:
        raise ConfigError(f"{where}.file 必须是非空文件名")
    converter = raw.get("converter")
    if not isinstance(converter, str) or not converter:
        raise ConfigError(f"{where}.converter 必须是非空字符串")
    return WebAsset(
        source=_path(base, raw.get("source"), f"{where}.source"),
        file=file,
        converter=converter,
    )


def _parse_aircraft(raw: dict, base: Path, where: str) -> AircraftAsset:
    unknown = set(raw) - {"length_m", "voxel_m", "keep_materials", "web"}
    if unknown:
        raise ConfigError(f"{where}: 未知字段 {', '.join(sorted(unknown))}")
    if not isinstance(raw.get("web"), dict):
        raise ConfigError(f"{where}.web 必须是键值表")
    keep = raw.get("keep_materials")
    if keep is not None and not isinstance(keep, list):
        raise ConfigError(f"{where}.keep_materials 必须是列表")
    asset_id = where.rsplit(".", 1)[-1]
    return AircraftAsset(
        id=asset_id,
        length_m=_number(raw, "length_m", where),
        voxel_m=_number(raw, "voxel_m", where),
        keep_materials=frozenset(str(v) for v in keep) if keep else None,
        web=_parse_web(raw["web"], base, f"{where}.web"),
    )


def _parse_missile(raw: dict, base: Path, where: str) -> MissileAsset:
    unknown = set(raw) - {"source", "length_m", "file"}
    if unknown:
        raise ConfigError(f"{where}: 未知字段 {', '.join(sorted(unknown))}")
    file = raw.get("file")
    if not isinstance(file, str) or not file:
        raise ConfigError(f"{where}.file 必须是非空文件名")
    return MissileAsset(
        id=where.rsplit(".", 1)[-1],
        source=_path(base, raw.get("source"), f"{where}.source"),
        length_m=_number(raw, "length_m", where),
        file=file,
    )


def load_catalog(path: str | Path | None = None) -> AssetCatalog:
    path = Path(path) if path is not None else DEFAULT_CATALOG
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"{path}: 读取失败（{exc}）") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: 顶层必须是键值表")
    unknown = set(raw) - {"version", "aircraft", "missiles"}
    if unknown:
        raise ConfigError(f"{path}: 未知字段 {', '.join(sorted(unknown))}")
    version = raw.get("version")
    if not isinstance(version, str) or not version:
        raise ConfigError(f"{path}: version 必须是非空字符串")

    base = path.parents[2]
    aircraft_raw = raw.get("aircraft")
    missiles_raw = raw.get("missiles", {})
    if not isinstance(aircraft_raw, dict) or not aircraft_raw:
        raise ConfigError(f"{path}: aircraft 必须是非空键值表")
    if not isinstance(missiles_raw, dict):
        raise ConfigError(f"{path}: missiles 必须是键值表")

    aircraft = {
        key: _parse_aircraft(value, base, f"aircraft.{key}")
        for key, value in aircraft_raw.items()
    }
    missiles = {
        key: _parse_missile(value, base, f"missiles.{key}")
        for key, value in missiles_raw.items()
    }
    return AssetCatalog(
        version=version, root=base,
        aircraft=aircraft, missiles=missiles,
    )
