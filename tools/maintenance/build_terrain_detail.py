"""从高分辨率原图裁出战斗核心区（100 km）的精细地形贴图。

基础底图 city_ortho.png 是 4096² 覆盖 200 km（约 49 m/px），跟随视角贴近
地面时会被放大成色块。这里直接取 9000² 原图的中心 4500² 区域，缩放到
4096²（约 24 m/px），保留真实的城市纹理；同时生成对应的海岸线与灯光
mask，供前端叠加。

产物（assets/terrain/）：
    city_detail.png        4096×4096  中心 100 km 地形
    city_detail_edge.png   2048×2048  海岸线 mask
    city_detail_lights.png 2048×2048  城市灯光 mask

用法（在仓库根目录）：
    python -m tools.maintenance.build_terrain_detail
"""
from __future__ import annotations

import os

import numpy as np
from PIL import Image, ImageFilter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "assets", "terrain", "city_map.png")
OUT_DIR = os.path.join(ROOT, "assets", "terrain")

SRC_PX = 9000
MAP_M = 200000.0          # 原图覆盖的物理边长
DETAIL_M = 100000.0       # 精细贴图覆盖的物理边长
DETAIL_PX = 4096
MASK_PX = 2048


def main() -> int:
    im = Image.open(SRC).convert("RGB")
    crop_px = int(round(SRC_PX * DETAIL_M / MAP_M))
    x0 = (im.width - crop_px) // 2
    y0 = (im.height - crop_px) // 2
    detail = im.crop((x0, y0, x0 + crop_px, y0 + crop_px))
    detail = detail.resize((DETAIL_PX, DETAIL_PX), Image.LANCZOS)
    detail.save(os.path.join(OUT_DIR, "city_detail.png"))

    mask_src = np.asarray(
        detail.resize((MASK_PX, MASK_PX), Image.LANCZOS)
    ).astype(np.float32)
    r, g, b = mask_src[:, :, 0], mask_src[:, :, 1], mask_src[:, :, 2]

    # 与原图一致的海面判定：青绿色（g、b 明显高于 r）
    sea = ((g > r + 15.0) & (b > r + 15.0)).astype(np.float32)
    sea = np.asarray(
        Image.fromarray((sea * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.6))
    ).astype(np.float32) / 255.0
    gx = (
        np.roll(sea, -1, axis=1) * 2 + np.roll(sea, -1, axis=0)
        - np.roll(sea, 1, axis=1) * 2 - np.roll(sea, 1, axis=0)
    )
    gy = (
        np.roll(sea, -1, axis=0) * 2 + np.roll(sea, -1, axis=1)
        - np.roll(sea, 1, axis=0) * 2 - np.roll(sea, 1, axis=1)
    )
    edge = np.clip(np.sqrt(gx * gx + gy * gy) * 2.4, 0.0, 1.0)
    edge_img = Image.fromarray((edge * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.8))
    edge_img.save(os.path.join(OUT_DIR, "city_detail_edge.png"))

    lights = ((r > 108.0) & (r > g + 16.0) & (r > b + 30.0)).astype(np.float32)
    lights_img = Image.fromarray((lights * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.0))
    lights_np = np.clip(np.asarray(lights_img).astype(np.float32) / 255.0 * 1.5, 0.0, 1.0)
    Image.fromarray((lights_np * 255).astype(np.uint8)).save(
        os.path.join(OUT_DIR, "city_detail_lights.png")
    )

    print(
        f"detail {DETAIL_PX}px / {DETAIL_M/1000:.0f}km "
        f"({DETAIL_M/DETAIL_PX:.1f} m/px), crop {crop_px}px",
        "edge", float((edge > 0.2).mean()),
        "lights", float((lights_np > 0.2).mean()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
