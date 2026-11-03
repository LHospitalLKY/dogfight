"""配置层：YAML → 冻结数据类，并做字段/单位/引用校验。"""
from .loader import ConfigBundle, DEFAULT_CONFIG_DIR, DEFAULT_SCENARIO, load_bundle
from .registry import AgentPersona, AircraftEntry
from .scenario import InitialConditions, Scenario, TeamSpec
from .schema import ConfigError

__all__ = [
    "AgentPersona",
    "AircraftEntry",
    "ConfigBundle",
    "ConfigError",
    "DEFAULT_CONFIG_DIR",
    "DEFAULT_SCENARIO",
    "InitialConditions",
    "Scenario",
    "TeamSpec",
    "load_bundle",
]
