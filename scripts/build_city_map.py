"""把城市地图底图处理成游戏地面贴图（深蓝配色、去文字）。

输入：assets/terrain/city_map.png（9000×9000，CityMapStudio 风格的城市平面图）
处理：
    1. 底部标题/水印文字（"DALIAN"、"XHS © CityMapStudio"）整段抹掉；
    2. 按海面/陆地/绿地/建成区四类重新着色，压成与主界面一致的深蓝冷调；
    3. 缩放到 4096×4096（对应 200 km 空域，约 49 m/px）。
产物：assets/terrain/city_ortho.png

用法（dogfight 环境）：
    python scripts/build_city_map.py
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np
import pygame

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "assets", "terrain", "city_map.png")
OUT_DIR = os.path.join(ROOT, "assets", "terrain")
OUT = os.path.join(OUT_DIR, "city_ortho.png")
OUT_PX = 4096

# 原图（9000 px）坐标系下的文字区域：底部标题条，整段填回纯海面。
TEXT_RECT = (3600, 8380, 5400, 8930)     # x0, y0, x1, y1
SRC_PX = 9000.0

# 深蓝配色（与 render.py 的 BG/GRID/HUD 冷色调一致）。
COL_SEA = (14, 28, 46)
COL_SEA_LIGHT = (26, 44, 64)            # 近岸/浅水与地图下缘的浅色渐变
COL_LAND = (46, 58, 72)
COL_GREEN = (34, 62, 50)
COL_URBAN = (176, 96, 56)               # 建成区（城市街区）
COL_URBAN_DENSE = (200, 112, 62)        # 高密度街区，提亮一档
COL_ROAD = (132, 96, 68)                # 道路/浅橙色线条
CHUNK = 512


def _strip_text(arr: np.ndarray) -> None:
    """把底部文字整段抹平：逐行取左侧干净海面的中位色填回。"""
    h = arr.shape[0]
    sx = h / SRC_PX
    x0, y0, x1, y1 = TEXT_RECT
    x0, x1 = int(x0 * sx), int(x1 * sx)
    y0, y1 = int(y0 * sx), int(y1 * sx)
    ref = np.median(arr[y0:y1, : max(8, int(x0 * 0.8))], axis=1)
    arr[y0:y1, x0:x1] = ref[:, None, :]


def _recolor(arr: np.ndarray) -> np.ndarray:
    """按地图配色分类重映射到深蓝调，分块处理避免一次性吃满内存。"""
    h = arr.shape[0]
    out = np.empty_like(arr)
    # 地图最下缘是一段「渐隐到纸白」的样式：颜色在接近白色后失去青绿偏色，
    # 会被误判成陆地并留下一条浅色横带。该行以下实测已无任何陆地色块，
    # 因此整片直接压成纯海面色。
    fade_row = int(7900.0 / SRC_PX * h)
    for y0 in range(0, h, CHUNK):
        y1 = min(h, y0 + CHUNK)
        blk = arr[y0:y1].astype(np.float32)
        r, g, b = blk[:, :, 0], blk[:, :, 1], blk[:, :, 2]
        # 海面：青绿色（g、b 都明显高于 r）
        sea = (g > r + 15.0) & (b > r + 15.0)
        # 建成区：偏橙红
        urban = (r > g + 15.0) & (r > b + 15.0)
        # 绿地：偏黄绿，蓝通道低于红/绿
        green = (b < r) & (b < g) & (g > 110.0) & ~sea & ~urban
        chunk_out = np.empty_like(blk, dtype=np.uint8)
        chunk_out[:] = COL_LAND
        chunk_out[green] = COL_GREEN
        chunk_out[urban] = COL_URBAN
        chunk_out[urban & ((r - g) > 90.0)] = COL_URBAN_DENSE
        # 浅橙色细线是道路：介于建成区与陆地之间的弱橙色
        road = urban & ((r - g) < 60.0)
        chunk_out[road] = COL_ROAD
        chunk_out[sea] = COL_SEA
        light = sea & ((r + g + b) / 3.0 > 190.0)
        chunk_out[light] = COL_SEA_LIGHT
        fade_rows = np.arange(y0, y1) >= fade_row
        if fade_rows.any():
            chunk_out[fade_rows] = COL_SEA
        out[y0:y1] = chunk_out
    return out


def main() -> int:
    pygame.init()
    pygame.display.set_mode((1, 1))
    if not os.path.exists(SRC):
        print(f"缺少源图：{SRC}", file=sys.stderr)
        return 1

    surf = pygame.image.load(SRC)
    # 先缩到目标分辨率再做分类，既省内存，也顺带把文字边缘抹匀。
    surf = pygame.transform.smoothscale(surf, (OUT_PX, OUT_PX))
    arr = pygame.surfarray.array3d(surf).transpose(1, 0, 2).copy()

    _strip_text(arr)
    rgb = _recolor(arr)

    out_surf = pygame.image.frombuffer(rgb.tobytes(), (OUT_PX, OUT_PX), "RGB")
    os.makedirs(OUT_DIR, exist_ok=True)
    pygame.image.save(out_surf, OUT)
    print(f"已保存 {OUT}（{OUT_PX}×{OUT_PX}）", flush=True)
    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
