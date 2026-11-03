"""冻结/比对对局基线：固定 seed 跑完整一局，输出可复现的快照。

用法（在仓库根目录）：
    python -m tools.maintenance.golden_snapshot          # 生成/覆盖 tests/golden/seed3.json
    python -m tools.maintenance.golden_snapshot --check  # 与基线逐项比对
    python -m tools.maintenance.golden_snapshot --seconds 120   # 缩短时长（仅调试用）

快照包含两类内容：
1. 全部机体/武器参数（dataclasses.asdict），用于确认配置化前后参数完全一致；
2. 固定 seed 的每 10 s 状态采样与最终结果，用于确认重构没有改变行为。
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

from dogfight.app.main import STEPS_PER_DECISION, build_agents, build_world
from dogfight.core.sim import Action

ROOT = Path(__file__).resolve().parents[2]
GOLDEN = ROOT / "tests" / "golden" / "seed3.json"

# 机体参数里挂载相关的字段在配置化后会换成 Loadout 组合，
# 比对时统一展开成“主近距弹 / 主中距弹 + 数量”的语义口径。
_SKIP = {"missile", "radar_missile", "missile_count", "radar_missile_count", "loadout"}


def _spec_dump(spec) -> dict:
    d = {
        f.name: getattr(spec, f.name)
        for f in dataclasses.fields(spec)
        if f.name not in _SKIP
    }
    d["ir_missile"] = (
        dataclasses.asdict(spec.missile) if spec.missile is not None else None
    )
    d["ir_count"] = spec.missile_count
    d["radar_missile"] = (
        dataclasses.asdict(spec.radar_missile)
        if spec.radar_missile is not None else None
    )
    d["radar_count"] = spec.radar_missile_count
    d["chaff_count"] = spec.chaff_count
    d["flare_count"] = spec.flare_count
    return d


def _ac_state(ac) -> dict:
    return {
        "name": ac.spec.name,
        "x": round(ac.x, 6),
        "y": round(ac.y, 6),
        "z": round(ac.z, 6),
        "v": round(ac.v, 6),
        "gamma": round(ac.gamma, 6),
        "psi": round(ac.psi, 6),
        "hp": round(ac.hp, 6),
        "ammo": round(ac.ammo, 6),
        "missiles": ac.missiles,
        "radar_missiles": ac.radar_missiles,
        "fuel": round(ac.fuel, 6),
        "mass": round(ac.mass, 6),
        "alive": bool(ac.alive),
    }


def run(seconds: float, seed: int) -> dict:
    world = build_world(seconds, dof=6)
    agents = build_agents(baseline=False, seed=seed)
    actions = [Action(), Action()]
    steps = 0
    samples = []
    next_sample = 0.0
    while world.result is None and world.t < seconds:
        if steps % STEPS_PER_DECISION == 0:
            actions = [ag.act(world.observe(i)) for i, ag in enumerate(agents)]
        world.step(actions)
        steps += 1
        while world.t >= next_sample:
            samples.append({
                "t": round(world.t, 6),
                "acs": [_ac_state(ac) for ac in world.acs],
                "primitives": [getattr(ag, "primitive", "") for ag in agents],
            })
            next_sample += 10.0

    return {
        "seconds": seconds,
        "seed": seed,
        "result": world.result,
        "reason": world.reason_code,
        "end_time": round(world.t, 6),
        "stats": {k: (round(v, 6) if isinstance(v, float) else v)
                  for k, v in world.stats.items()},
        "acs": [_ac_state(ac) for ac in world.acs],
        "samples": samples,
    }


def _diff(path: str, expected, actual, out: list[str]) -> None:
    if isinstance(expected, dict) and isinstance(actual, dict):
        for key in expected.keys() | actual.keys():
            if key not in expected:
                out.append(f"{path}.{key}: 新增 {actual[key]!r}")
            elif key not in actual:
                out.append(f"{path}.{key}: 缺失")
            else:
                _diff(f"{path}.{key}", expected[key], actual[key], out)
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            out.append(f"{path}: 长度 {len(expected)} -> {len(actual)}")
            return
        for i, (a, b) in enumerate(zip(expected, actual)):
            _diff(f"{path}[{i}]", a, b, out)
        return
    if isinstance(expected, float) and isinstance(actual, float):
        if not math.isclose(expected, actual, rel_tol=1e-9, abs_tol=1e-9):
            out.append(f"{path}: {expected!r} -> {actual!r}")
        return
    if expected != actual:
        out.append(f"{path}: {expected!r} -> {actual!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="与基线比对而不是覆盖")
    parser.add_argument("--seconds", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--path", default=str(GOLDEN))
    args = parser.parse_args()

    world = build_world(args.seconds, dof=6)
    snapshot = {
        "aircraft": {ac.spec.name: _spec_dump(ac.spec) for ac in world.acs},
        "run": run(args.seconds, args.seed),
    }
    path = Path(args.path)
    if args.check:
        expected = json.loads(path.read_text(encoding="utf-8"))
        diffs: list[str] = []
        _diff("snapshot", expected, snapshot, diffs)
        if diffs:
            print(f"基线不一致：{len(diffs)} 项")
            for line in diffs[:40]:
                print("  " + line)
            raise SystemExit(1)
        print("基线一致")
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"已写入 {path}")


if __name__ == "__main__":
    main()
