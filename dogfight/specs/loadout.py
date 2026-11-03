"""机体挂载：把“机体”和“弹”解耦成可配置的组合。

配置文件按武器 id 引用弹种，加载器解析成具体的 MissileSpec；
运行时只需要问 Loadout 要主用近距弹/中距弹与数量。
"""
from __future__ import annotations

from dataclasses import dataclass

from .weapon import MissileSpec


@dataclass(frozen=True)
class WeaponSlot:
    weapon: MissileSpec
    count: int

    @property
    def mass(self) -> float:
        return self.weapon.mass * max(self.count, 0)


@dataclass(frozen=True)
class Loadout:
    ir: WeaponSlot | None = None
    radar: WeaponSlot | None = None
    chaff: int = 0
    flare: int = 0

    @property
    def weapon_mass(self) -> float:
        """全部挂载的发射质量 kg，用于从空机重量折算出作战全重。"""
        total = 0.0
        for slot in (self.ir, self.radar):
            if slot is not None:
                total += slot.mass
        return total
