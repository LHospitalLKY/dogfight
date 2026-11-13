"""从 OBJ 模型生成正交俯视轮廓图（武器显示页用）。

投影方式：正交投影，视线沿机体 -z（从正上方往下看），机头朝上。
坐标与 dogfight.view.load_model 一致（x 前、y 右、z 下），因此轮廓与
three.js 里渲染的机体完全对得上；起落架/吊舱等也按同一套规则过滤。

产物：
    assets/silhouettes/su57_top.png     透明底轮廓（白填充 + 深描边）
    assets/silhouettes/f16_top.png
    assets/silhouettes/j20a_top.png     （J-20A / J-20S 共用）
    assets/silhouettes/f15c_top.png
    assets/silhouettes/f15ex_top.png    （F-15E / F-15EX 共用）
    assets/silhouettes/j16_top.png
    assets/silhouettes/j15t_top.png
    assets/silhouettes/top_views.png    上述轮廓的预览图（深色底、带标注）

用法（在仓库根目录）：
    python -m tools.maintenance.build_silhouettes
"""
from __future__ import annotations

import math
import os
import sys

from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from dogfight.view import MODEL_BY_NAME
from dogfight.view.theme import TEAM_COLORS

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(ROOT, "assets", "silhouettes")
FONT_PATH = os.path.join(ROOT, "assets", "fonts", "FiraCode-Regular.ttf")
PX_PER_M = 44.0             # 输出分辨率（像素/米），先按 2 倍超采样再缩回
FILL = (246, 249, 253, 255)
STROKE = (26, 36, 50, 255)


def _rasterize(model: dict, ppm: float) -> Image.Image:
    """把所有面片投影到 XY 平面并填充，得到实心轮廓。"""
    xs: list[float] = []
    ys: list[float] = []
    for poly, _kind in model["faces"]:
        for x, y, _z in poly:
            xs.append(x)
            ys.append(y)
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    margin = 4.0
    w = int(math.ceil((y1 - y0) * ppm)) + int(2 * margin * ppm)
    h = int(math.ceil((x1 - x0) * ppm)) + int(2 * margin * ppm)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    ox = margin * ppm - y0 * ppm
    oy = margin * ppm + x1 * ppm

    def to_px(x: float, y: float) -> tuple[float, float]:
        return (ox + y * ppm, oy - x * ppm)

    for poly, _kind in model["faces"]:
        pts = [to_px(x, y) for x, y, _z in poly]
        if len(pts) < 3:
            continue
        draw.polygon(pts, fill=FILL)
        draw.line(pts + [pts[0]], fill=FILL, width=1)
    return img


def _outline(img: Image.Image, stroke_px: int) -> Image.Image:
    """把实心轮廓加上描边：描边画在轮廓外侧，内部保持白填充。"""
    alpha = img.getchannel("A").point(lambda v: 255 if v > 0 else 0)
    dilated = alpha.filter(ImageFilter.MaxFilter(2 * stroke_px + 1))
    edge = ImageChops.subtract(dilated, alpha)
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(Image.new("RGBA", img.size, FILL), (0, 0), alpha)
    out.paste(Image.new("RGBA", img.size, STROKE), (0, 0), edge)
    return out


def _build(model: dict, name: str) -> Image.Image:
    big = _rasterize(model, PX_PER_M * 2.0)      # 2 倍超采样
    big = _outline(big, 3 * 2)
    small = big.resize(
        (round(big.width * 0.5), round(big.height * 0.5)),
        Image.LANCZOS,
    )
    path = os.path.join(OUT_DIR, f"{name}_top.png")
    small.save(path)
    print(f"已保存 {path}（{small.width}×{small.height}）", flush=True)
    return small


def _font(size: int) -> ImageFont.ImageFont:
    if os.path.exists(FONT_PATH):
        return ImageFont.truetype(FONT_PATH, size)
    return ImageFont.load_default()


def _preview(images, columns: int = 3) -> None:
    font = _font(30)
    pad = 24
    cell_w = max(im.width for _, im, _ in images) + pad * 2
    cell_h = max(im.height for _, im, _ in images) + pad * 2 + 42
    rows = (len(images) + columns - 1) // columns
    cols = min(columns, len(images))
    sheet = Image.new("RGBA", (cell_w * cols, cell_h * rows), (7, 18, 15, 255))
    draw = ImageDraw.Draw(sheet)
    for k, (label, im, color) in enumerate(images):
        gx, gy = k % columns, k // columns
        x = gx * cell_w
        y = gy * cell_h
        draw.rectangle((x, y, x + cell_w - 1, y + cell_h - 1), outline=(24, 42, 34))
        sheet.paste(im, (x + (cell_w - im.width) // 2,
                         y + pad + (cell_h - 42 - im.height) // 2), im)
        tw = draw.textlength(label, font=font)
        draw.text((x + (cell_w - tw) / 2, y + cell_h - 34), label,
                  font=font, fill=color)
    path = os.path.join(OUT_DIR, "top_views.png")
    sheet.convert("RGB").save(path)
    print(f"已保存 {path}", flush=True)


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    builds = (
        ("SU-57", "su57", TEAM_COLORS[0]),
        ("F-16V", "f16", TEAM_COLORS[1]),
        ("J-20A", "j20a", TEAM_COLORS[0]),
        ("F-15C", "f15c", TEAM_COLORS[1]),
        ("F-15EX", "f15ex", TEAM_COLORS[0]),
        ("J-16", "j16", TEAM_COLORS[1]),
        ("J-15T", "j15t", TEAM_COLORS[0]),
    )
    images = []
    for name, stem, color in builds:
        if name not in MODEL_BY_NAME:
            print(f"跳过 {name}：原始 OBJ 未就位", flush=True)
            continue
        images.append((f"{name}  ORTHO TOP", _build(MODEL_BY_NAME[name], stem),
                       color))
    _preview(images)
    return 0


if __name__ == "__main__":
    sys.exit(main())
