"""资产注册表：模型路径、缩放与导出参数。"""

from .catalog import (
    DEFAULT_CATALOG,
    AircraftAsset,
    AssetCatalog,
    MissileAsset,
    WebAsset,
    load_catalog,
)

__all__ = [
    "DEFAULT_CATALOG", "AircraftAsset", "AssetCatalog", "MissileAsset",
    "WebAsset", "load_catalog",
]
