"""固定 seed / 时刻 / 分辨率的金标截图，并支持分区像素 diff。

用法：
    python scripts/visual_golden.py                          # 生成到 docs/images/golden
    python scripts/visual_golden.py --compare /tmp/golden_old  # 与旧基线对比
    python scripts/visual_golden.py --size 1600x1000           # 只跑一档分辨率
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from main import PHYSICS_DT, STEPS_PER_DECISION, build_agents, build_world  # noqa: E402
from render import Renderer  # noqa: E402
from sim import Action  # noqa: E402

# 金标时刻：覆盖开局、超视距、防御、近距格斗与稳定巡航。
TIMES = (0.0, 2.0, 6.0, 12.0, 25.0, 45.0, 70.0)
SIZES = ((1600, 1000), (1920, 1080), (1920, 720))
# 分区：顶部信息区、场景、底部座舱面板。diff 阈值按“变化像素占比”给定。
REGIONS = (("hud", 0.0, 0.26), ("scene", 0.26, 0.58), ("panels", 0.58, 1.0))
DIFF_RATIO_LIMIT = 0.001  # 0.1% 的像素发生变化即报警


def render_frames(size, out_dir: Path, seed: int = 3) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    world = build_world(300.0, 6, awacs=True)
    agents = build_agents(False, seed=seed)
    renderer = Renderer(gpu=False, size=size)
    steps = 0
    actions = [Action(), Action()]
    paths: list[Path] = []

    def snap(index: int) -> None:
        renderer.draw(world, False, agents)
        path = out_dir / f"{size[0]}x{size[1]}_f{index}_{world.t:06.1f}.png"
        pygame.image.save(renderer.screen, str(path))
        paths.append(path)

    snap(0)
    index = 1
    while index < len(TIMES) and world.result is None:
        if steps % STEPS_PER_DECISION == 0:
            actions = [ag.act(world.observe(i)) for i, ag in enumerate(agents)]
        world.step(actions)
        steps += 1
        if world.t >= TIMES[index]:
            snap(index)
            index += 1
    pygame.quit()
    return paths


def region_diff(a_path: Path, b_path: Path) -> list[tuple[str, float, int]]:
    a = pygame.surfarray.array3d(pygame.image.load(str(a_path))).astype(np.int16)
    b = pygame.surfarray.array3d(pygame.image.load(str(b_path))).astype(np.int16)
    if a.shape != b.shape:
        raise SystemExit(f"尺寸不一致: {a_path.name} {a.shape} vs {b.shape}")
    h = a.shape[1]
    rows = (a != b).any(axis=2)  # (宽, 高)
    out = []
    for name, lo, hi in REGIONS:
        band = rows[:, int(h * lo):int(h * hi)]
        ratio = float(band.mean())
        peak = int(np.abs(a[:, int(h * lo):int(h * hi)] - b[:, int(h * lo):int(h * hi)]).max())
        out.append((name, ratio, peak))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="金标截图与像素 diff")
    parser.add_argument("--out", default=str(ROOT / "docs" / "images" / "golden"))
    parser.add_argument("--compare", default=None,
                        help="旧基线目录；逐帧做分区像素 diff")
    parser.add_argument("--size", default=None, help="只跑一档，如 1600x1000")
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()

    sizes = SIZES
    if args.size:
        w, h = (int(v) for v in args.size.lower().split("x"))
        sizes = ((w, h),)

    out_dir = Path(args.out)
    made: list[Path] = []
    for size in sizes:
        made += render_frames(size, out_dir, args.seed)
    print(f"金标截图: {len(made)} 张 -> {out_dir}")

    if not args.compare:
        return
    bad = 0
    old_dir = Path(args.compare)
    for path in made:
        old = old_dir / path.name
        if not old.exists():
            print(f"缺少基线: {path.name}")
            bad += 1
            continue
        parts = region_diff(old, path)
        line = "  ".join(f"{n}={r * 100:.3f}% (max {p})" for n, r, p in parts)
        print(f"{path.name}: {line}")
        if any(r > DIFF_RATIO_LIMIT for _n, r, _p in parts):
            bad += 1
    print("PASS" if bad == 0 else f"FAIL {bad} 帧超出阈值")
    raise SystemExit(0 if bad == 0 else 1)


if __name__ == "__main__":
    main()
