"""对局的可配置数据模型：武器、机体、挂载与场景。

这个包只包含冻结数据类，不读取文件、不依赖渲染或入口层。
"""
from .aircraft import AircraftSpec
from .loadout import Loadout, WeaponSlot
from .weapon import MissileSpec

__all__ = ["AircraftSpec", "Loadout", "MissileSpec", "WeaponSlot"]
