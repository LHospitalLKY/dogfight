"""`python -m dogfight` 的命令入口。

子命令：
    validate [config_dir]   静态校验配置（CI/提交前用）
    dump <scenario>         打印解析后的 SI 参数（人工核对量纲）
不带子命令时转交给原来的 `main.py`，保持旧用法不变。
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

from dogfight.config import ConfigError, DEFAULT_CONFIG_DIR, load_bundle


def _validate(config_dir: str | Path) -> int:
    bundle = load_bundle(config_dir)
    print(f"配置目录：{bundle.config_dir}")
    print(f"武器 {len(bundle.weapons)}：{', '.join(sorted(bundle.weapons))}")
    print(f"机体 {len(bundle.aircraft)}：{', '.join(sorted(bundle.aircraft))}")
    for scenario in bundle.scenarios.values():
        names = " vs ".join(team.aircraft.spec.name for team in scenario.teams)
        print(f"场景 {scenario.id}: {names}，{scenario.time_limit:.0f} s")
    print("配置校验通过")
    return 0


def _dump(scenario_ref: str) -> int:
    bundle = load_bundle()
    scenario = bundle.scenario(scenario_ref)
    payload = {
        "scenario": {
            "id": scenario.id,
            "time_limit": scenario.time_limit,
            "physics_dt": scenario.physics_dt,
            "decision_hz": scenario.decision_hz,
            "initial": dataclasses.asdict(scenario.initial),
            "seed": scenario.seed,
        },
        "teams": [
            {
                "name": team.name,
                "radar_policy": team.radar_policy,
                "agent": dataclasses.asdict(team.aircraft.agent),
                "aircraft": dataclasses.asdict(team.aircraft.spec),
            }
            for team in scenario.teams
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        if argv and argv[0] == "validate":
            return _validate(argv[1] if len(argv) > 1 else DEFAULT_CONFIG_DIR)
        if argv and argv[0] == "dump":
            if len(argv) != 2:
                print("用法：python -m dogfight dump <scenario-id|yaml路径>",
                      file=sys.stderr)
                return 2
            return _dump(argv[1])
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 1

    from dogfight.app.main import main as run_main
    run_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
