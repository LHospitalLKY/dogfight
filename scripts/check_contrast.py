"""按 WCAG 计算 theme.py 里正文/强调色的对比度。

正文要求 ≥4.5:1，告警与强调要求 ≥3:1。低于门限直接失败。
玻璃面板按“底色叠在场景底色上”的合成色参与计算。

用法：python scripts/check_contrast.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import theme  # noqa: E402


def _lin(c: float) -> float:
    c = c / 255.0
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def luminance(rgb) -> float:
    r, g, b = (v / 255.0 for v in rgb[:3])
    lin = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in (r, g, b)]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def contrast(fg, bg) -> float:
    l1, l2 = luminance(fg), luminance(bg)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def over(fg, bg, alpha: int) -> tuple[int, int, int]:
    """alpha 合成：fg 叠在 bg 上，alpha 取 0–255。"""
    a = alpha / 255.0
    return tuple(int(fg[i] * a + bg[i] * (1.0 - a)) for i in range(3))


def main() -> None:
    glass = over(theme.GLASS_TINT, theme.BG, theme.GLASS_ALPHA_HEAVY)
    checks = [
        ("正文 TEXT / 面板底", theme.TEXT, theme.PANEL_FILL, 4.5),
        ("正文 TEXT / 场景底", theme.TEXT, theme.BG, 4.5),
        ("次级 SUB / 面板底", theme.SUB, theme.PANEL_FILL, 4.5),
        ("HUD 绿 / 场景底", theme.HUD_GREEN, theme.BG, 4.5),
        ("HUD 琥珀 / 面板底", theme.HUD_MARK, theme.PANEL_FILL, 3.0),
        ("告警红 / 面板底", theme.HUD_WARN, theme.PANEL_FILL, 3.0),
        ("红队色 / 面板底", theme.TEAM_COLORS[0], theme.PANEL_FILL, 3.0),
        ("蓝队色 / 面板底", theme.TEAM_COLORS[1], theme.PANEL_FILL, 3.0),
        ("正文 TEXT / 玻璃面板", theme.TEXT, glass, 4.5),
        ("次级 SUB / 玻璃面板", theme.SUB, glass, 4.5),
    ]
    bad = 0
    for name, fg, bg, need in checks:
        ratio = contrast(fg, bg)
        ok = ratio >= need
        bad += 0 if ok else 1
        print(f"{name:<22} {ratio:5.2f}:1  门限 {need:.1f}  {'OK' if ok else 'FAIL'}")
    print("PASS" if bad == 0 else f"FAIL {bad} 项")
    raise SystemExit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
