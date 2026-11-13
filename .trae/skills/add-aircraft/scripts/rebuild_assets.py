"""新增机体 / 更新模型后的一键刷新。

用法（仓库任意目录下）：

    python .trae/skills/add-aircraft/scripts/rebuild_assets.py

依次做三件事：

1. 按 configs/assets/models.yaml 重导 Web 模型——只有 version 变了才会真正重导，
   没变时直接复用 static/models 下的文件；
2. 重新生成 assets/silhouettes/*.png 俯视轮廓；
3. 检查「configs/aircraft 机型名 → models.yaml 登记 → app.js 的 CAMO_SPECS /
   SIL_IMAGES → 轮廓图文件」是否闭环。

任一项缺失时以退出码 1 结束，并把缺的条目打印出来。
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
APP_JS = ROOT / "dogfight" / "web" / "static" / "app.js"
SIL_DIR = ROOT / "assets" / "silhouettes"


def _block(text: str, start: str, end: str) -> str:
    """截取 app.js 里从 start 到 end 的片段，找不到时返回空串。"""
    i = text.find(start)
    if i < 0:
        return ""
    j = text.find(end, i + len(start))
    return text[i:j if j > 0 else len(text)]


def export_models() -> bool:
    from dogfight.web.server import MODEL_DIR, MODEL_TAG, export_models as run_export

    try:
        models = run_export()
    except Exception as exc:  # 缺原始模型、YAML 写错等都在这里报出来
        print(f"[导出] 失败：{exc}")
        return False
    print(f"[导出] 模型版本 {MODEL_TAG}", flush=True)
    for name, meta in models["aircraft"].items():
        path = MODEL_DIR / Path(meta["url"]).name
        try:
            with path.open(encoding="utf-8", errors="ignore") as fh:
                faces = sum(1 for line in fh if line.startswith("f "))
            size = path.stat().st_size / 1e6
            print(f"  {name:<10}{faces:>8} 面  {size:5.2f} MB")
        except OSError:
            print(f"  {name:<10} 导出文件缺失：{path}")
            return False
    return True


def build_silhouettes() -> bool:
    print("[轮廓] 重建俯视轮廓", flush=True)
    done = subprocess.run(
        [sys.executable, "-m", "tools.maintenance.build_silhouettes"], cwd=ROOT
    )
    return done.returncode == 0


def check_links() -> list[str]:
    import yaml

    names = []
    for path in sorted((ROOT / "configs" / "aircraft").glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if raw.get("name"):
            names.append((str(raw["name"]), path.name))
    catalog = yaml.safe_load(
        (ROOT / "configs" / "assets" / "models.yaml").read_text(encoding="utf-8")
    ) or {}
    registered = set(catalog.get("aircraft") or {})
    src = APP_JS.read_text(encoding="utf-8")
    camo = _block(src, "const CAMO_SPECS = {", "function camoTexture")
    sil = _block(src, "const SIL_IMAGES = {};", "function drawSilhouette")

    missing: list[str] = []
    for name, source in names:
        if name not in registered:
            missing.append(f"{name}（{source}）：未在 configs/assets/models.yaml 的 aircraft 下登记")
        if f'"{name}":' not in camo:
            missing.append(f"{name}（{source}）：app.js 的 CAMO_SPECS 缺条目（会让前端整页黑屏）")
        match = re.search(rf'"{re.escape(name)}":\s*"([^"]+\.png)"', sil)
        if not match:
            missing.append(f"{name}（{source}）：app.js 的 SIL_IMAGES 缺条目")
        elif not (SIL_DIR / match.group(1)).exists():
            missing.append(f"{name}（{source}）：轮廓图 assets/silhouettes/{match.group(1)} 不存在")
    return missing


def main() -> int:
    parser = argparse.ArgumentParser(description="新增机体 / 更新模型后的一键刷新")
    parser.add_argument("--no-export", action="store_true", help="跳过 Web 模型重导")
    parser.add_argument("--no-silhouettes", action="store_true", help="跳过俯视轮廓重建")
    args = parser.parse_args()

    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    ok = True
    if not args.no_export:
        ok = export_models() and ok
    if not args.no_silhouettes:
        ok = build_silhouettes() and ok

    missing = check_links()
    if missing:
        print("[检查] 未闭环：")
        for item in missing:
            print(f"  - {item}")
        return 1
    print("[检查] 机型 → 模型登记 → 涂装 → 俯视轮廓 全部就位")
    if not ok:
        print("[检查] 但导出/轮廓步骤有失败项，见上面的输出")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
