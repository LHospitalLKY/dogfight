"""从 OBJ 模型生成正交俯视轮廓图（武器显示页用）。

投影方式：正交投影，视线沿机体 -z（从正上方往下看），机头朝上。
坐标变换与 render.py 的 _load_obj_model 保持一致（x 前、y 右、z 下），
因此轮廓与游戏里渲染的机体完全对得上；起落架/吊舱等也按同一套规则过滤。

产物：
    assets/silhouette/su57_top.png     透明底轮廓（白填充 + 深描边）
    assets/silhouette/f16_top.png
    assets/silhouette/top_views.png    两张轮廓的预览图（深色底、带标注）

用法（dogfight 环境）：
    python scripts/build_silhouettes.py
"""
from __future__ import annotations

import os
import sys

# 无显示环境下也要能跑
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import render  # noqa: E402  （依赖项目根目录的 OBJ 路径与模型加载规则）

OUT_DIR = os.path.join(ROOT, "assets", "silhouette")
PX_PER_M = 44.0             # 输出分辨率（像素/米），先按 2 倍超采样再缩回
FILL = (246, 249, 253, 255)
STROKE = (26, 36, 50, 255)


def _rasterize(model: dict, ppm: float) -> pygame.Surface:
    """把所有面片投影到 XY 平面并填充，得到实心轮廓。"""
    xs: list[float] = []
    ys: list[float] = []
    for poly, _kind, _ao in model["faces"]:
        for x, y, _z in poly:
            xs.append(x)
            ys.append(y)
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    margin = 4.0
    w = int(math_ceil((y1 - y0) * ppm)) + int(2 * margin * ppm)
    h = int(math_ceil((x1 - x0) * ppm)) + int(2 * margin * ppm)
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    surf.fill((0, 0, 0, 0))
    ox = margin * ppm - y0 * ppm
    oy = margin * ppm + x1 * ppm

    def to_px(x: float, y: float) -> tuple[float, float]:
        return (ox + y * ppm, oy - x * ppm)

    for poly, _kind, _ao in model["faces"]:
        pts = [to_px(x, y) for x, y, _z in poly]
        if len(pts) < 3:
            continue
        pygame.draw.polygon(surf, FILL, pts)
        # 再描一圈同色边，避免相邻面片之间留出透明细缝。
        pygame.draw.polygon(surf, FILL, pts, 1)
    return surf


def math_ceil(v: float) -> float:
    return float(int(v + 0.999999))


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.copy()
    for _ in range(radius):
        cur = out.copy()
        cur[1:, :] |= out[:-1, :]
        cur[:-1, :] |= out[1:, :]
        cur[:, 1:] |= out[:, :-1]
        cur[:, :-1] |= out[:, 1:]
        out = cur
    return out


def _outline_surface(solid: pygame.Surface, stroke_px: int) -> pygame.Surface:
    """把实心轮廓加上描边：描边画在轮廓外侧，内部保持白填充。"""
    w, h = solid.get_size()
    alpha = pygame.surfarray.array_alpha(solid).transpose(1, 0) > 0
    edge = _dilate(alpha, stroke_px) & ~alpha
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    arr[alpha] = FILL
    arr[edge] = STROKE
    return pygame.image.frombuffer(arr.tobytes(), (w, h), "RGBA").convert_alpha()


def _build(model: dict, name: str) -> pygame.Surface:
    big = _rasterize(model, PX_PER_M * 2.0)      # 2 倍超采样
    big = _outline_surface(big, 3 * 2)
    w = int(round(big.get_width() * 0.5))
    h = int(round(big.get_height() * 0.5))
    small = pygame.transform.smoothscale(big, (w, h))
    path = os.path.join(OUT_DIR, f"{name}_top.png")
    pygame.image.save(small, path)
    print(f"已保存 {path}（{w}×{h}）", flush=True)
    return small


def _preview(images) -> None:
    pygame.font.init()
    font = pygame.font.Font(None, 30)
    pad = 24
    cell_w = max(im.get_width() for _, im, _ in images) + pad * 2
    cell_h = max(im.get_height() for _, im, _ in images) + pad * 2 + 42
    sheet = pygame.Surface((cell_w * len(images), cell_h))
    sheet.fill((7, 18, 15))
    for k, (label, im, color) in enumerate(images):
        x = k * cell_w
        pygame.draw.rect(sheet, (24, 42, 34), (x, 0, cell_w, cell_h), 1)
        sheet.blit(im, (x + (cell_w - im.get_width()) // 2,
                        pad + (cell_h - 42 - im.get_height()) // 2))
        t = font.render(label, True, color)
        sheet.blit(t, (x + (cell_w - t.get_width()) // 2, cell_h - 34))
    path = os.path.join(OUT_DIR, "top_views.png")
    pygame.image.save(sheet, path)
    print(f"已保存 {path}", flush=True)


def main() -> int:
    pygame.init()
    pygame.display.set_mode((1, 1))
    os.makedirs(OUT_DIR, exist_ok=True)
    su57 = _build(render.SU57_MODEL, "su57")
    f16 = _build(render.F16_MODEL, "f16")
    _preview([
        ("SU-57  ORTHO TOP", su57, render.TEAM_COLORS[0]),
        ("F-16V  ORTHO TOP", f16, render.TEAM_COLORS[1]),
    ])
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
