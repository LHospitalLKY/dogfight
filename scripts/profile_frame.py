"""分项帧时间剖析：地面 / 航迹 / 机体 / 导弹 / 特效 / HUD / 合成。

用法：
    python scripts/profile_frame.py                    # 1600x1000，200 帧
    python scripts/profile_frame.py --size 1920x1080 --frames 300
    python scripts/profile_frame.py --strict           # 超出预算返回非 0

预算：HUD+面板 p95 ≤11 ms，单帧总时间 p95 ≤33 ms（30 fps 帧周期 33.3 ms，
留一点抖动余量；实测约 29.5 ms，折 34 fps 量级）。

机体面数按视觉质量优先：Su-57 主模型约 2900 面、F-16 约 3700 面，
姿态框迷你模型约 1100 / 1900 面；在 CPU 软件渲染下这两项占大头，
预算随之上调，保证模型在 1080p 下不再出现明显的体素块面。
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ["DOGFIGHT_PROFILE"] = "1"

import pygame  # noqa: E402

from main import STEPS_PER_DECISION, build_agents, build_world  # noqa: E402
from render import Renderer  # noqa: E402
from sim import Action  # noqa: E402

HUD_BUDGET = 11.0
FRAME_BUDGET = 33.0


def pct(values, q: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    idx = min(len(data) - 1, int(round(q * (len(data) - 1))))
    return data[idx]


def main() -> None:
    parser = argparse.ArgumentParser(description="帧时间剖析")
    parser.add_argument("--size", default="1600x1000")
    parser.add_argument("--frames", type=int, default=200)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    w, h = (int(v) for v in args.size.lower().split("x"))

    world = build_world(300.0, 6, awacs=True)
    agents = build_agents(False, seed=args.seed)
    renderer = Renderer(gpu=False, size=(w, h))
    actions = [Action(), Action()]
    steps = 0
    # 预热：让贴图、字体、面板缓存都建好，再开始计时。
    for _ in range(10):
        renderer.draw(world, False, agents)
        for _ in range(2):
            world.step(actions)
    # 与 main.py 的窗口循环一致：首帧后冻结长生命周期对象，避免分代 GC
    # 周期性重扫模型/贴图/缓存造成的几 ms 抖动。
    gc.freeze()
    renderer._samples.clear()

    for _ in range(args.frames):
        if steps % STEPS_PER_DECISION == 0:
            actions = [ag.act(world.observe(i)) for i, ag in enumerate(agents)]
        for _ in range(2):
            world.step(actions)
            steps += 1
            if world.result is not None:
                break
        renderer.draw(world, False, agents)
    pygame.quit()

    print(f"分辨率 {w}x{h}，{args.frames} 帧，seed={args.seed}")
    print(f"{'阶段':<10}{'p50':>8}{'p95':>8}")
    total = []
    for name, samples in renderer._samples.items():
        print(f"{name:<10}{pct(samples, 0.5):8.2f}{pct(samples, 0.95):8.2f}")
    # 每帧总时间：把各阶段按帧对齐近似为各阶段 p50 之和（采样顺序一致）。
    n = min((len(v) for v in renderer._samples.values()), default=0)
    for i in range(n):
        total.append(sum(v[i] for v in renderer._samples.values()))
    hud95 = pct(renderer._samples.get("hud", []), 0.95)
    total95 = pct(total, 0.95)
    print(f"{'合计':<10}{pct(total, 0.5):8.2f}{total95:8.2f}")
    hud_ok = hud95 <= HUD_BUDGET
    frame_ok = total95 <= FRAME_BUDGET
    print(f"HUD 预算 {HUD_BUDGET} ms: {'OK' if hud_ok else 'FAIL'} (p95={hud95:.2f})")
    print(f"单帧预算 {FRAME_BUDGET} ms: {'OK' if frame_ok else 'FAIL'} (p95={total95:.2f})")
    if args.strict and not (hud_ok and frame_ok):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
