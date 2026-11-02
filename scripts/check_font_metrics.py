"""校验 theme.py 的字号、等宽数字宽度与行高是否落在 tokens 范围内。

用法：python scripts/check_font_metrics.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import pygame  # noqa: E402
import pygame.freetype  # noqa: E402

import theme  # noqa: E402
from render import _CachedFont  # noqa: E402

MIN_SIZE = 14
BODY_MIN = 18


def load(name: str, size: int):
    """theme 里的字体既可能是项目内的 .otf 路径，也可能是系统字体名。"""
    pygame.freetype.init()
    path = name if os.path.isfile(name) else pygame.font.match_font(name)
    if path:
        return pygame.freetype.Font(path, size)
    return pygame.freetype.SysFont(name, size)


def main() -> None:
    pygame.init()
    pygame.display.set_mode((320, 240))
    bad = 0

    sizes = {"H1": theme.FS_H1, "BODY": theme.FS_BODY,
             "SMALL": theme.FS_SMALL, "TINY": theme.FS_TINY}
    for label, size in sizes.items():
        ok = size >= MIN_SIZE
        bad += 0 if ok else 1
        print(f"字号 {label:<6} {size:3d} px  {'OK' if ok else 'FAIL'}")
    if theme.FS_BODY < BODY_MIN:
        print(f"正文 {theme.FS_BODY} px < {BODY_MIN} px  FAIL")
        bad += 1

    # 数字走定宽槽位渲染（比例字体本身字宽不一），这里按渲染器实际使用的
    # 步进取值，保证“读数变化时整行不抖动”这条约束仍然成立。
    num = _CachedFont(theme.FONT_NUM, theme.FS_SMALL, tabular=True)
    widths = [num.advance(ch) for ch in "0123456789"]
    spread = max(widths) - min(widths)
    ok = spread == 0
    bad += 0 if ok else 1
    print(f"等宽数字步进 {widths} 极差 {spread}px  {'OK' if ok else 'FAIL'}")

    ui = load(theme.FONT_UI, theme.FS_BODY)
    _, rect = ui.render("Ag", fgcolor=(255, 255, 255))
    line_h = ui.get_sized_height()
    ok = line_h >= theme.FS_BODY * 0.8
    bad += 0 if ok else 1
    print(f"行高 {line_h}px（{rect.height}px 字形）  门限 {theme.FS_BODY * 0.8:.1f}px  "
          f"{'OK' if ok else 'FAIL'}")

    print("PASS" if bad == 0 else f"FAIL {bad} 项")
    raise SystemExit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
