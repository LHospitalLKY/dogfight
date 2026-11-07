"""配置入口：一次加载 configs/ 下的武器、机体与场景。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .registry import AircraftEntry, ConfigError, load_aircraft_and_weapons
from .scenario import Scenario, load_scenario, load_scenarios

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
DEFAULT_SCENARIO = "bvr-1v1"


@dataclass(frozen=True)
class ConfigBundle:
    config_dir: Path
    aircraft: dict[str, AircraftEntry]
    weapons: dict
    scenarios: dict[str, Scenario]

    def scenario(self, name_or_path: str | Path) -> Scenario:
        """按 id 取内置场景；也允许直接传一个 YAML 路径。"""
        path = Path(name_or_path)
        if path.is_file():
            return load_scenario(path, self.aircraft, self.weapons)
        key = str(name_or_path)
        if key not in self.scenarios:
            known = ", ".join(sorted(self.scenarios))
            raise ConfigError(f"未知场景 {key!r}，可用：{known}")
        return self.scenarios[key]


def load_bundle(config_dir: str | Path | None = None) -> ConfigBundle:
    config_dir = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    weapons, aircraft = load_aircraft_and_weapons(config_dir)
    scenarios = load_scenarios(config_dir / "scenarios", aircraft, weapons)
    return ConfigBundle(
        config_dir=config_dir,
        aircraft=aircraft,
        weapons=weapons,
        scenarios=scenarios,
    )
