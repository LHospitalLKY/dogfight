---
name: add-aircraft
description: Add a new aircraft or update a 3D model in the dogfight project, covering model export, silhouette, paint config, flight config, docs and tests. Use when adding an aircraft, replacing an OBJ, or changing decimation. Do not use for sim-only tuning or HUD work.
---

# 新增机体 / 更新模型

模型换新或新机型入库时按顺序执行。第 4 步的脚本把能自动化的部分一次跑完，缺项会以非零退出码报出来。

## 1. 模型与导出

1. 原始模型放 `assets/aircraft/<file>.obj`（有 `.mtl` 与贴图目录一并放好）。
2. `configs/assets/models.yaml` 的 `aircraft` 下登记 `length_m`、`voxel_m`、`web.source / file / converter / decimate_m`。
3. `version` 递增（`web-obj-vN`）——这会触发全部模型重导，`dogfight/web/static/models/*.obj` 和 `meta.json` 要一起提交。
4. `dogfight/view/__init__.py` 里实现 converter 并注册进 `CONVERTERS`：
   - 机头方向用座舱座椅/内饰位置确认，不要按文件名猜；
   - 机体材质取面数最多的那个，其余归 `stores`；
   - 座舱盖：材质名含 `glass` + 机头 12%–28% + 高度 ≥ 50% + 中线 ±5%；
   - 喷口：机尾发动机带内按 0.5 m 窗口密度找喷口环取质心，不要用“机尾最外点 ± 翼展比例”（侧卫尾梁会把尾焰顶到舱外）；
   - `decimate_m` 要小于薄翼面厚度（侧卫内外蒙皮约 7 cm，取 4 cm），抽稀用 `cluster(..., quadric=True)`。
5. 模型自带 DDS 涂装：`web` 下加 `textures: true`。导出会取机体材质的 `map_Kd` 与 `xxx_n.dds`，缩到 2048 转 WebP，并把角点 UV 写进 OBJ（`f v/vt`）；细节见 `docs/机型参数与武器挂载.md` §4.5。

## 2. 仿真配置

1. 新建 `configs/aircraft/<id>.yaml`：几何/质量、推力、`cd0`/展弦比/`cl_max`、结构过载、燃油、机炮、操纵面、RCS、雷达/IRST、agent 参数、挂载。
2. 特殊气动面改 `dogfight/core/sim.py` 并用 CL–迎角曲线验证。固定鸭翼要 `canard_deflect_max: 0`，且 `a_can` 分支加同条件，否则平尾被旁路。
3. 需要新对阵再加 `configs/scenarios/*.yaml`，跑 seeds 1–3 确认能结束。

## 3. Web 观战端

1. `tools/maintenance/build_silhouettes.py` 的 `builds` 加一条，生成 `assets/silhouettes/<id>_top.png`。
2. `app.js` 的 `SIL_IMAGES` 注册该图，`CAMO_SPECS` 补涂装条目（渐变、军徽 `mark`、`markCenter / markSize / markLateral`）。缺 `CAMO_SPECS` 会让页面抛 `Cannot read properties of undefined (reading 'image')` 直接黑屏。
3. 尾焰位置取自 `meta.json` 的 `engines`，导出后确认在喷口中心。

## 4. 一键刷新

```bash
python .trae/skills/add-aircraft/scripts/rebuild_assets.py
```

脚本依次重导 Web 模型、重建俯视轮廓、检查「机型名 → models.yaml → CAMO_SPECS / SIL_IMAGES → 轮廓图」闭环，任何缺项都会列出来并以退出码 1 结束。只改前端时用 `--no-export`，只改模型时用 `--no-silhouettes`。

## 5. 文档与验证

1. 更新 `docs/机型参数与武器挂载.md`：1.1–1.4 总表、2.1 挂载表、3.3 配置要点、4.1 模型清单、4.4 转换器说明。
2. `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=. python -m pytest -q`（应为 68 passed）。
3. 真实浏览器里核对外形、涂装/军徽、武器面板俯视轮廓、尾焰位置。不要用 `?capture=` 判断对战画面，那是单独的模型陈列渲染。
