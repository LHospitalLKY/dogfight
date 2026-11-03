# HMI 精细化改造方案（车机级一致性与 60 fps）

> **历史文档（1.1.1 起）**：本文记录的是 pygame 软件渲染阶段的 HMI 规范，`render.py` / `theme.py` 及其 HUD 绘制已在“Web 观战端”重构中移除，当前可视化只有 three.js 浏览器端（`dogfight/web/`）。
> 文中 `theme.py` 现为 `dogfight/view/theme.py`，`scripts/` 下保留的脚本已迁到 `tools/maintenance/`；路径与行号仅作当时记录。

本方案把“车机 HMI 那种高 DPI、干净、统一、稳定 60 帧”的标准落到当前的 pygame + CPU 软件渲染实现上。
前提：本项目不启用 GPU 路径（`moderngl` 上下文在 WSLg 下不可用，`render.py` 的 `_init_gpu()` 会自动回退到 CPU），
因此所有精细度手段必须是 CPU 可承受的：**不做逐帧全屏 2× 超采样，改为静态件 2× 预渲染 + 字形缓存**。

---

## 1. 现状快照

| 项目 | 现状 | 位置 |
| --- | --- | --- |
| 渲染规模 | `render.py` 3054 行，27 个 `_draw_*` 方法 | `render.py` |
| 颜色 | 模块级常量散落：`BG/GRID/BORDER/TEXT/SUB/TEAM_COLORS/HUD_*`，另有 `CANOPY_COLOR/INTAKE_COLOR/NOZZLE_COLOR/TRAIL_LIGHT/LIGHT` | `render.py` L36–L48 等 |
| 字体 | 只有 3 个固定字号 `Font(None, 22/18/64)`，无等宽数字，`render()` 每帧即时生成 Surface | `render.py` L415–L417 |
| 帧率 | 窗口 `clock.tick(60)`；录制 1280×800 / 20 fps | `main.py` L894–L905、L792–L828 |
| 面板 | `_draw_aircraft_panel` / `_draw_station` / `_draw_scope` / `_draw_fwd_hud` / `_draw_weapon_rows` 各自一套边框与内边距 | `render.py` |
| 地面 | 城市贴图透视投影 + mip 链 + 冷雾 | `render.py` `_draw_ground` |

---

## 2. 目标指标

| 维度 | 目标 | 验收方式 |
| --- | --- | --- |
| 逻辑分辨率 | 1920×1080（默认）、1600×1000、1280×720、1920×720、2560×1440 均成立，窗口可拖拽缩放 | 逐档截图无溢出/重叠 |
| 帧率 | 窗口 30 fps 量级（33.3 ms 帧周期）；机体面数提上来后按 33 ms 帧周期验收 | 帧时间剖析 p95 |
| 分项预算 | 地面 ≤3 ms、模型/尾焰 ≤17 ms、飘带 ≤2 ms、HUD+面板 ≤11 ms、合成 ≤1.5 ms | 剖析脚本 |
| 字号 | 正文 ≥18 px@1×，关键读数 22 px，面板标题 16 px，最小 14 px | 度量脚本 |
| 对比度 | 正文 vs 面板底 ≥4.5:1；告警 ≥3:1 | 对比度脚本 |
| 间距/圆角 | 4/8 px 栅格；面板圆角 4 px、小件 2 px | 代码审查 + 截图 |
| 线宽 | 静态件逻辑 0.5 px（2× 下 1 px），降采样后为真 1 px | 截图放大核对 |
| 动效 | 面板 200 ms、数值 150 ms、告警闪烁 500 ms | 统一 easing 常量 |
| 状态刷新 | 告警/锁定即时，不做插值 | 行为核对 |

---

## 3. 设计系统层：`theme.py`

新增 `theme.py`，`render.py` 只从这里取常量，删除散落硬编码。

- **颜色三层**：`BG`（背景）、容器（`PANEL_FILL/PANEL_EDGE/GLASS_ALPHA` 三档）、强调（`TEAM_COLORS` 两队 + `WARN/MARK` 两色）。其余一律降饱和。
- **字体**：`FONT_UI`（无衬线）、`FONT_NUM`（等宽数字）；字号集合 `FS_H1/FS_BODY/FS_SMALL/FS_TINY`。
- **间距**：`SPACE_1=4 / SPACE_2=8 / SPACE_3=12`，圆角 `RADIUS_PANEL=4 / RADIUS_CHIP=2`。
- **动效**：`DUR_PANEL=200 / DUR_NUM=150 / DUR_BLINK=500`（ms），`EASE_OUT_CUBIC`。
- **图元网格**：`ICON_BOX=24`、`ICON_STROKE=2`。

迁移原则：P0 阶段只做“原值搬迁”，像素 diff 必须为 0。

---

## 4. 渲染实现层

1. **DPI 解耦**：引入 `ui_scale`（物理像素 ÷ 逻辑像素），布局按逻辑坐标计算，绘制时乘 scale；`Renderer` 接收窗口尺寸，内部按逻辑尺寸排布。
2. **字体**：`pygame.font.Font(None, …)` → `pygame.freetype`（带 hinting，可设 px 高度）；加字形缓存 `(face, size, text, color)`；所有读数用等宽字体并按固定槽位右对齐。
3. **静态层缓存**：面板底/边框/标题/刻度/图标预渲染为带 alpha 的 `Surface`（2× 大小 + `smoothscale`）；缓存键 = `(面板类型, 逻辑尺寸, 状态档位)`，失效于弹药/燃油档位变化。
4. **发丝线**：静态层使用 0.5 逻辑 px。
5. **预乘 Alpha**：半透明面板统一预乘色，消除边缘暗边；玻璃透明度只保留 3 档。
6. **图标矢量化**：弹体 logo、雷达光点、来袭楔形、导弹星标统一 24 px 画板 / 2 px 描边；机体轮廓沿用 `assets/silhouette/*_top.png`，缩放前 `smoothscale`。
7. **色温统一**：3D 场景与 HMI 共用白点/对比度；面板偏冷白，场景不偏黄。

**不做**：MSDF（需要 shader，CPU 路径不可用）；逐帧全屏 2× SSAA（CPU 预算不够）。录制与金标截图允许离线 2× 渲染。

---

## 5. 动效与稳定性

1. 统一时长与 easing，全部来自 `theme.py`。
2. 数值插值：速度/高度/距离一阶插值（τ≈120 ms）；**告警、锁定、命中即时刷新**。
3. 三层解耦：仿真 120 Hz / 决策 10 Hz / 渲染 60 Hz，渲染在两次物理步之间做位置与姿态插值。
4. 禁止布局抖动：等宽数字 + 固定槽位 + 标签位置写死栅格。

---

## 6. 验收工具（`scripts/`，不进运行时）

| 脚本 | 作用 |
| --- | --- |
| `scripts/visual_golden.py` | 固定 seed/t/分辨率生成金标截图；分区（HUD/面板/场景）像素 diff |
| `scripts/check_contrast.py` | 读 `theme.py`，按 WCAG 计算正文/告警对比度，<4.5:1 失败 |
| `scripts/check_font_metrics.py` | 校验字号、数字宽度、行高落在 tokens 范围 |
| `scripts/profile_frame.py` | 分项计时（地面/模型/飘带/HUD/合成），输出 p50/p95 |

金标图存 `docs/images/golden/`。

---

## 7. 阶段计划与验收

| 阶段 | 内容 | 验收 |
| --- | --- | --- |
| P0 | `theme.py` 建立（原值搬迁）+ 帧剖析插桩 + 现状基线截图 | 像素 diff = 0；基线帧时间有记录 |
| P1 | freetype 字体 + 等宽数字槽位 | 度量脚本通过；文本不跳动 |
| P2 | 统一面板 chrome + 静态层缓存 | HUD+面板 ≤4 ms；仅允许圆角/内边距差异 |
| P3 | 图标 24 px 网格、发丝线、预乘 alpha | 1 px 线成立；半透明边缘无暗边 |
| P4 | 动效与数值插值 | 告警即时、数值平滑 |
| P5 | 验收脚本 + 三档分辨率 + 录制 2× 基准图 | 脚本全绿；`pytest` 42 项与 `--headless --seed 3` 正常 |

---

## 8. 风险与回滚

- **配图一致性**：面板布局改动后 `docs/images` 6 张配图需整体重生成。
- **字形宽度变化**：freetype 与 `Font(None,…)` 宽度不同，所有硬编码坐标需复核。
- **缓存失效**：静态层缓存键若不区分燃料/弹药档位，会显示过期数值。
- **低矮分辨率**：1920×720 更扁，座舱三面板由 `_station_metrics()` 按“可用宽度 ≥520 px”决定并排还是竖排，竖排时雷达占满簇宽。
- **回滚**：每个阶段独立可回退；P0 之前先打基线截图，任何阶段 diff 超阈值即回退该阶段。

---

## 9. 实施记录

### 9.1 阶段完成情况

| 阶段 | 落地内容 | 验收结果 |
| --- | --- | --- |
| P0 | 新增 `theme.py`（颜色三层、字号、4/8 栅格、圆角、动效、图元网格）；`render.py` 全部改为从 tokens 取值 | 金标分区像素 diff `0.000%`，基线帧时间留档 |
| P1 | `pygame.freetype` 渲染 + 字形缓存 `_CachedFont`；字体统一为 `assets/font/FiraCode-Regular.ttf`（Fira Code），数字走定宽槽位（18 px 步进 11.0 px，与旧等宽字体同宽），另设 14 px `FS_TINY` 供雷达屏角标使用 | `check_font_metrics.py` PASS：极差 0 px、行高 27 px（20 px 字形） |
| P2 | `_panel_bg` 统一面板 chrome（填充 + 圆角 + 强调条，2× 预渲染再 `smoothscale`）+ `_chrome_cache`；座舱区全部改用它 | HUD+面板 p95 3.83 ms ≤ 4 ms |
| P3 | 图标 24 px 画板 / 2 px 描边；静态件 0.5 逻辑 px 降采样成真 1 px；`_alpha_rect` 按尺寸/颜色/透明度缓存 | 截图放大核对 1 px 线成立，半透明边缘无暗边 |
| P4 | 数值一阶插值（τ=120 ms，`_smooth`），告警/锁定/命中即时刷新 | 读数平滑、状态不延迟 |
| P5 | `scripts/` 四个验收脚本 + 三档分辨率金标图 | 脚本全绿（见 9.3） |

### 9.2 帧时间：优化前 / 优化后

`scripts/profile_frame.py --frames 240 --strict`，1600×1000、seed=3（单位 ms，p50 / p95）：

| 阶段 | 优化前 | 优化后 |
| --- | --- | --- |
| ground | 13.4 / — | 4.60 / 5.00 |
| aircraft（含机炮包线） | 26.4 / — | 3.49 / 3.80 |
| trails（含地面网格） | — | 0.61 / 0.96 |
| missiles / effects | — | 0.05 / 0.07、0.05 / 0.07 |
| hud | 25.3 / — | 3.45 / 3.83 |
| **合计** | **p95 64.03** | **12.47 / 13.13** |

中间版本（做完体素降面与矩阵化模型、但尚未处理 GC 与整屏叠加层）实测合计 p50 15.89–16.25、p95 16.61–17.14，
HUD p95 4.05–4.16（临界超预算）——这也是补上最后两项优化的原因。

### 9.3 验收结果（最终一轮）

| 项 | 命令 | 结果 |
| --- | --- | --- |
| 编译 | `python -m py_compile main.py render.py theme.py scripts/*.py` | 通过 |
| 单测 | `python -m pytest -q` | 42 passed |
| 无头对局 | `python main.py --headless --seed 3` | blue 胜，254.60 s，与原版一致 |
| 对比度 | `python scripts/check_contrast.py` | PASS（正文 15.16:1，告警 5.68:1） |
| 字体度量 | `python scripts/check_font_metrics.py` | PASS（步积极差 0.0 px） |
| 金标 diff | `python scripts/visual_golden.py --compare docs/images/golden` | 21 帧 × 三区全为 `0.000%`，PASS（字体换 Fira Code、并删去左右状态卡与右侧高度尺后重拍过基线） |
| 帧剖析 | `python scripts/profile_frame.py --frames 240 --strict` | HUD p95 3.77 OK；合计 p95 14.08 OK |

注：本机装有 ROS，若 shell 里带了 `PYTHONPATH`，`pytest` 会先加载 `/opt/ros/...` 的 `launch` 包并报 `No module named 'yaml'`；
用 `env -u PYTHONPATH python -m pytest -q` 即可（与代码无关）。

### 9.4 关键优化与回滚点

| 改动 | 做法 | 收益 / 回滚方式 |
| --- | --- | --- |
| 模型旋转矩阵化 | 机体旋转 + 相机旋转 + 缩放折成单个 3×3，逐顶点只做 9 次乘加；光照方向转到相机系后做点积；背面剔除改在相机系判断 | `_draw_model` 与 `_draw_attitude_model` 合计降约 60% |
| 体素降面 | 主视图 0.10 m / 0.60 m，姿态框 `mini_faces` 0.22 m / 1.30 m；同一 OBJ 只解析一次、多次聚类 | Su-57 1525→517 面（mini 52），F-16 1865→766 面（mini 155）；改回原步长即恢复原画质。**后续调整**：为消除姿态框/主视图的体素块面，现取主模型 0.03 / 0.28、迷你 0.06 / 0.40（约 2900/3700 与 1100/1900 面），帧预算相应放宽到 33 ms 帧周期（见场景渲染风格改造方案 §5） |
| 地面贴图 | `GROUND_STEP` 6→8；`opaque=True` 时先在低分辨率把边缘羽化与 `BG` 预合成，走不透明 blit | ground 12.04→5.59；把 `opaque` 传回 False 即回到逐像素 alpha |
| 静态件缓存 | `_chrome_cache`（面板底/边框）+ `_alpha_cache`（纯色半透明底） | HUD 4.59→3.81，单帧 p95 17.53→15.61 |
| 分代 GC 冻结 | `main.py` 首帧后 `gc.freeze()`，把模型/贴图/字形/面板缓存推进永久代；`profile_frame.py` 同样处理 | 每帧抖动少 2–4 ms，p95 20.44→17.1→（叠加层优化后）13.13；去掉该调用即恢复旧行为 |
| 叠加层脏矩形 | 机炮包线/导弹/开火特效各用一层 SRCALPHA 面，只清并只贴上一次画过的区域 | 机炮包线 1.40→0.25 ms，开火特效 1.22→0.00 ms（原来不发射时也整屏清+贴） |

### 9.5 已知波动

- WSL2 上同一版本连续跑 `profile_frame.py`，合计 p95 有 ±0.5 ms 抖动、HUD p95 有 ±0.2 ms 抖动；
  最终版留了约 3 ms 余量（13.13 vs 16.6、3.83 vs 4.0），抖动不会把预算翻红。
- `visual_golden.py` 在沙箱里结束时会因字体缓存目录权限报一次 exit=1（截图已全部写出），
  对比逻辑本身正常；如需干净退出，可先设 `XDG_CACHE_HOME=/tmp`。

### 9.6 本轮座舱版面与信息归并（2026-11）

| 项 | 做法 | 结果 |
| --- | --- | --- |
| 三块面板拆开 | `_station_metrics()` 只算尺寸、`_station_panels()` 只算矩形、`_draw_station()` 只画内容；雷达贴屏幕角、HUD 在其内侧、武器栏在 HUD 下方 | 1920×1080：雷达 280×304、HUD 352×188、武器 352×108；1600×1000 并排；1280×800 竖排 |
| 雷达放大 | 雷达不再被“竖排大框”压扁，取整块簇宽减 HUD | 1600×1000 圆半径 85（原 ≈50），1920×1080 112 |
| 面板底衬统一 | 新增 `STATION_FILL=(5,15,11)` / `STATION_ALPHA=214`，三块共用，只用队伍色边线区分 | 读数不再被背后的 3D 场景干扰 |
| HP 归位 | 从姿态框机名行移出，进 HUD 顶部 24 px 状态带（左 `HP` + 血量条，<25% 转红） | 全界面只出现一次 |
| G 归位 | 从速度带下方（会压刻度）移到状态带右侧 | 不再与速度表叠字 |
| 目标方位 | `_bearing_marker` 改 `ACCENT_RED` 红色小箭头，落在航向带刻度上方；删除 `TGT BRG` 文字与深色横幅 | 少一行文字，方位仍可读 |
| 距离读数 | 菱形上方的距离 chip 若会压到航向带，自动改放菱形下方，并把右边界收窄到 `rect.right-118` | 5 位距离（如 `109.0`）不再压航向刻度 |
| 读数基线对齐 | `_CachedFont` 逐字形栅格化改 `origin=True`；矮字形（`.` `-` `:`）不再浮到字顶 | `G 0.9` 的小数点落回基线，负号与数字同高 |
| 航向刻度让开读数框 | 主刻度数字在中央 52 px 读数框左右 `26 px + 半字宽 + 4 px` 内跳过（原先只跳过刻度线） | `352` 与 `000/030` 不再叠字 |
| 配图与金标 | `docs/images/golden` 21 帧、`docs/images` 5 张 README 配图整体重生成（1280×800、seed=3） | 金标基线重拍；红色方位箭头用 `CONTACT_DETECT` 场景核对 |
