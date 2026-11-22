"""多机初始编队：给出队内每架飞机相对长机的几何偏移。

偏移以 `lane_m`（同队横向基准间隔）为单位，返回 (前向, 横向, 高度)：
- 前向为正表示朝队首/敌方方向，僚机通常在长机后方，取负值；
- 横向为正表示长机航向的右侧；
- 高度为相对长机的垂直偏移。

队首飞机固定在队伍基准点，其余飞机按编队排在后方，因此双方仍然在
`initial.spacing_m` 处对头进入，只是队形不再是清一色的横排。
"""
from __future__ import annotations

from typing import Callable

# 每种编队对机数的约束；不在约束内的机数视为“该队形不支持”。
_SUPPORT: dict[str, Callable[[int], bool]] = {
    "line_abreast": lambda n: n >= 2,
    "echelon": lambda n: n >= 2,
    "trail": lambda n: n >= 2,
    "spread": lambda n: n >= 2,
    "ladder": lambda n: n >= 2,
    "stack": lambda n: n >= 2,
    "fighting_wing": lambda n: n == 2,
    "finger_four": lambda n: n == 4,
    "fluid_four": lambda n: n == 4,
    "box": lambda n: n == 4,
    "arrowhead": lambda n: n == 4,
    "wedge": lambda n: n in (3, 4),
}

# 中文名用于 HUD / 日志展示。
LABELS = {
    "line_abreast": "横队",
    "echelon": "梯队",
    "trail": "纵队",
    "spread": "疏散",
    "ladder": "阶梯",
    "stack": "叠型",
    "fighting_wing": "战斗僚机",
    "finger_four": "指尖四机",
    "fluid_four": "流动四机",
    "box": "盒形",
    "arrowhead": "箭头",
    "wedge": "楔形",
}

ALL = tuple(_SUPPORT)


def supported(name: str, n: int) -> bool:
    check = _SUPPORT.get(name)
    return check is not None and check(n)


def offsets(name: str, n: int) -> tuple[tuple[float, float, float], ...]:
    """返回 n 架飞机的 (前向, 横向, 高度) 偏移，单位为 lane 的倍数。"""
    if n < 2:
        return tuple((0.0, 0.0, 0.0) for _ in range(n))
    if name == "line_abreast":
        return tuple((0.0, k - (n - 1) / 2.0, 0.0) for k in range(n))
    if name == "echelon":
        return tuple((-0.75 * k, 0.75 * k, 0.06 * k) for k in range(n))
    if name == "trail":
        # 纵队：高一低交替，避免后机被前机尾流/遮挡。
        return tuple((-1.5 * k, 0.0, 0.06 * (k % 2)) for k in range(n))
    if name == "spread":
        return tuple(
            (-0.25 * abs(k - (n - 1) / 2.0), 1.5 * (k - (n - 1) / 2.0), 0.0)
            for k in range(n)
        )
    if name == "ladder":
        return tuple((-1.2 * k, 0.0, 0.15 * k) for k in range(n))
    if name == "stack":
        return tuple((-0.6 * k, 0.0, 0.12 * k) for k in range(n))
    if name == "fighting_wing" and n == 2:
        return ((0.0, 0.0, 0.0), (-0.5, 0.5, 0.0))
    if name == "finger_four" and n == 4:
        return (
            (0.0, 0.0, 0.0),
            (-0.55, 0.55, 0.04),
            (-0.55, -0.55, 0.04),
            (-1.10, 0.0, 0.08),
        )
    if name == "fluid_four" and n == 4:
        # 两个双机拉开：长机在两侧，僚机在其后外侧。
        return (
            (0.0, 0.9, 0.0),
            (-0.35, 1.55, 0.05),
            (0.0, -0.9, 0.02),
            (-0.35, -1.55, 0.07),
        )
    if name == "box" and n == 4:
        return ((0.0, 0.0, 0.0), (0.0, -1.1, 0.0),
                (-1.6, 0.0, 0.05), (-1.6, -1.1, 0.05))
    if name == "arrowhead" and n == 4:
        return ((0.0, 0.0, 0.0), (-0.6, 0.6, 0.0),
                (-0.6, -0.6, 0.0), (-1.4, 0.0, 0.06))
    if name == "wedge" and n in (3, 4):
        out = [(0.0, 0.0, 0.0), (-0.7, 0.7, 0.0), (-0.7, -0.7, 0.0)]
        if n == 4:
            out.append((-1.4, 0.0, 0.06))
        return tuple(out)
    # 未适配的机数/队形：退回横队，保证仍然能正常起飞。
    return tuple((0.0, k - (n - 1) / 2.0, 0.0) for k in range(n))
