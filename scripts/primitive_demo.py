"""逐原语录制“标准动作”演示视频（双方交互版）。

每个原语配一个对手脚本：
- 需要双方配合才能看懂的（单环/双环、剪刀、Yo-Yo、高 G 抢位）让对手一起
  做对应动作，几何关系才成立；
- 单机动作（筋斗、殷麦曼、破 S、桶滚）只让对手平飞或追击当参照；
- 眼镜蛇这类瞬态动作按“拉起 → 保持 → 改出”分段飞完，不再只录一个抬头。

镜头固定在两机连线的中点（Renderer 默认的 follow=None 模式），双方始终
同框，便于看清“谁在追谁、谁先掉头”。

用法：
    python scripts/primitive_demo.py                 # 全部原语
    python scripts/primitive_demo.py --only SPLIT_S LOOP
    python scripts/primitive_demo.py --dof 3         # 3DOF 内核
    python scripts/primitive_demo.py --dry --trace   # 只跑物理、打印姿态
"""
from __future__ import annotations

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import TacticalAgent  # noqa: E402
from main import F16V_SPEC, PHYSICS_DT, STEPS_PER_DECISION, SU57_SPEC  # noqa: E402
from sim import Action, World, quat_from_euler  # noqa: E402

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "primitives"
)

# 场景：两侧的初始 (x, y, z, 航向)。航向 0 = +x；π = 迎头。
#   merge     —— 迎头交汇 3.2 km，双方水平同高
#   chase     —— 咬尾 1.4 km，目标同向
#   chase_far —— 咬尾 2.4 km，目标同向（低 Yo-Yo / 距离不足的场景）
#   pursued   —— 被咬尾 1.1 km，对手在本机后下方（眼镜蛇让对手冲过头）
#   far       —— 超视距 26 km 迎头
#   edge/low  —— 贴近空域边界 / 最低高度
SCENES = {
    "merge": ((0.0, 0.0, 9000.0, 0.0), (3200.0, 0.0, 9000.0, math.pi)),
    # 咬尾场景给目标一点横向/高度偏移，双方盘旋对穿时才不会贴到几十米内穿模。
    "chase": ((0.0, 0.0, 9000.0, 0.0), (1400.0, 350.0, 9040.0, 0.0)),
    "chase_far": ((0.0, 0.0, 9000.0, 0.0), (2400.0, 420.0, 9060.0, 0.0)),
    # 剪刀从“被咬尾”开始：追尾机在后上方带点横向偏置，贴身对穿时不穿模。
    "scissors": ((0.0, 0.0, 9000.0, 0.0), (-620.0, 1050.0, 8800.0, 0.0)),
    # 偏置迎头：对穿时不落在同一条直线上，给 CRANK / EXTEND 留出安全间距。
    "merge_off": ((0.0, 0.0, 9000.0, 0.0), (3200.0, 1200.0, 9090.0, math.pi)),
    "pursued": ((0.0, 0.0, 9000.0, 0.0), (-500.0, 150.0, 8760.0, 0.0)),
    "far": ((0.0, 0.0, 9000.0, 0.0), (26000.0, 0.0, 9000.0, math.pi)),
    "edge": ((28000.0, 0.0, 9000.0, 0.0), (20000.0, 0.0, 9000.0, math.pi)),
    "low": ((28000.0, 0.0, 1900.0, 0.0), (20000.0, 0.0, 1900.0, math.pi)),
    "offset": ((0.0, 0.0, 9000.0, 0.0), (12000.0, -7000.0, 9000.0, math.pi)),
}

# 对手脚本（第二架飞机固定执行的机动）。
#   level       —— 平飞当参照
#   pure        —— 咬尾追击（对着本机拉前置）
#   break       —— 最大过载急转躲避（被追尾方的回旋）
#   high_g_break—— 高 G 急转躲避
#   scissors    —— 反向剪刀（自带周期换向）
#   one/two     —— 单环 / 双环的配合转弯
#   extend      —— 掉头脱离
FOE_PRIMITIVE = {
    "level": None,
    "pure": "PURE",
    "break": "BREAK",
    "high_g_break": "HIGH_G_BREAK",
    "scissors": "SCISSORS",
    "one_circle": "ONE_CIRCLE",
    "two_circle": "TWO_CIRCLE",
    "lead": "LEAD",
    "lag": "LAG",
    "extend": "EXTEND",
}

# 初始速度默认 280 m/s；垂直机动（筋斗/殷麦曼）要靠动能换高度，低速会把
# 飞机挂在迎角上，而 COBRA 反过来必须在低速才做得出来（高速下 55° 迎角
# 会产生 9 g 升力，那是急转而不是过失速）。
DEFAULT_SPEED = 280.0


def demo(prim, scene, secs, *,
         speed=DEFAULT_SPEED, foe="level", foe_sign=1.0, foe_kind=None,
         foe_speed=DEFAULT_SPEED, seq=None, sign=1.0):
    """组装一条演示配置。

    sign / foe_sign 是双方的转向侧：同号 = 双环（同向盘旋），
    反号 = 单环（对头互切）。seq 用于需要分段的动作（如眼镜蛇改出）。
    """
    return dict(
        prim=prim, scene=scene, secs=secs, speed=speed,
        foe=foe_kind or foe, foe_sign=foe_sign, foe_speed=foe_speed,
        seq=seq, sign=sign,
    )


DEMOS = [
    # ---- 单机/安全类：对手平飞当参照 ----
    demo("RECOVER", "low", 8.0),
    demo("POSITION", "far", 12.0),
    # ---- 交汇抢位：双方一起切内圈（反号 = 单环，对头互切） ----
    demo("LEAD_TURN", "merge", 8.0, foe="lead", foe_sign=-1.0),
    demo("ONE_CIRCLE", "merge", 12.0, foe="one_circle", foe_sign=-1.0),
    demo("TWO_CIRCLE", "merge", 14.0, foe="two_circle", foe_sign=1.0),
    # ---- 进攻/防御：目标回旋，本机抢角度或急转 ----
    demo("HIGH_G_TURN", "chase", 8.0, foe="break"),
    demo("HIGH_G_BREAK", "chase", 8.0, foe="pure", sign=-1.0),
    demo("HIGH_G_SPLIT_S", "chase", 16.0),
    demo("HIGH_G_BARREL", "chase", 10.0, foe="pure"),
    demo("BEAM", "chase", 8.0, foe="pure", sign=-1.0),
    demo("PURE", "chase", 10.0, foe="break"),
    demo("LEAD", "chase", 10.0, foe="break"),
    demo("LAG", "chase", 10.0, foe="break"),
    demo("BREAK", "chase", 8.0, foe="pure", sign=-1.0),
    # ---- 垂直机动：对手平飞 ----
    demo("SPLIT_S", "chase", 16.0),
    demo("BARREL", "chase", 10.0, foe="pure"),
    demo("IMMELMANN", "chase", 20.0),
    demo("LOOP", "chase", 34.0),
    # ---- 剪刀：双方同向开转、各自反向换向，才看得出“谁冲过头” ----
    # 剪刀是防御动作：本机被咬尾，靠反复反向急转逼追尾机冲过头。
    demo("SCISSORS", "scissors", 14.0, foe="pure", sign=-1.0, speed=165.0,
         foe_speed=240.0),
    # ---- Yo-Yo：被追尾方回旋，进攻方用垂直面调整 ----
    demo("HIGH_YOYO", "chase", 10.0, foe="break", speed=320.0, foe_speed=230.0),
    # 低 Yo-Yo 要有能量富余才做得出来：本机比目标快，俯冲换速度去缩短距离。
    demo("LOW_YOYO", "chase_far", 10.0, foe="break", speed=250.0, foe_speed=225.0),
    # ---- 超视距类 ----
    demo("CRANK", "merge_off", 8.0),
    demo("NOTCH_DESC", "chase", 8.0, foe="pure", sign=-1.0),
    # F-pole 的完整形态是“偏置照射 → 自己的弹主动后再冷转”，无弹演示里
    # 用 CRANK 接 EXTEND 表示这两段。
    demo("F_POLE", "far", 12.0, seq=[("CRANK", 5.0), ("EXTEND", 7.0)]),
    # ---- 过失速：拉起（保持 2 s）→ 推杆改出 → 恢复追逐 ----
    # 追尾机保持平飞冲刺：本机急减速后它来不及减速会直接冲到前面去，
    # 这才能看出眼镜蛇的“让对手冲过头”的作用。
    demo("COBRA", "pursued", 8.0, speed=130.0, foe="level", foe_speed=300.0,
         seq=[("COBRA", 2.0), ("UNLOAD", 2.0), ("PURE", 4.0)]),
    demo("EXTEND", "merge_off", 9.0),
    demo("UNLOAD", "merge", 6.0),
]


class PassiveTarget:
    """对手：翼平、1 g、固定油门，保持平飞当空间参照。"""

    primitive = "-"
    situation = "-"

    def act(self, _obs) -> Action:
        return Action(bank=0.0, n=1.0, throttle=0.65, fire=False)


class Pilot:
    """按脚本原语飞行的演示机，支持多段序列。"""

    def __init__(self, primitive: str, sign: float = 1.0, seq=None,
                 period: float | None = None) -> None:
        self.core = TacticalAgent(energy_pref=0.55, seed=7)
        if period is not None:
            # 剪刀的换向节拍错开，双方才不会同时反向（像集体摆翼）。
            self.core.SCISSOR_PERIOD = period
        self.primitive = primitive
        self.situation = "-"
        self.sign = sign
        self.seq = seq or [(primitive, None)]
        self.dt = self.core.dt
        self._seg = 0
        self._seg_t = 0.0
        self._started = False

    def start(self, obs) -> None:
        self._seg = 0
        self._seg_t = 0.0
        self._enter(self.seq[0][0], obs)
        self._started = True

    def _enter(self, primitive: str, obs) -> None:
        self.primitive = primitive
        self.core._enter(primitive, obs)
        self.core.primitive = primitive
        # 演示里转向侧/防御侧是脚本指定死的，不能让 _enter 按瞬时方位重算。
        self.core._turn_sign = self.sign
        self.core._turn_locked = True
        self.core._def_sign = self.sign
        self.core._scissor_sign = self.sign
        # 只执行指定原语：跳过态势判定与原语选择，保留限能量/限过载。
        self.core._hold = 0.0

    def act(self, obs) -> Action:
        if not self._started:
            self.start(obs)
        self._seg_t += self.dt
        while (
            self._seg + 1 < len(self.seq)
            and self.seq[self._seg][1] is not None
            and self._seg_t >= self.seq[self._seg][1]
        ):
            self._seg += 1
            self._seg_t = 0.0
            self._enter(self.seq[self._seg][0], obs)
        self.core.primitive = self.primitive
        self.core._hold += self.dt
        self.core._phase_t += self.dt
        self.core._scissor_t -= self.dt
        act = self.core._execute(self.primitive, obs)
        act.afterburner = 1.0 if act.throttle > 0.9 else 0.0
        return act


def _reposition(world: World, subject, target, speed: float,
                foe_speed: float = DEFAULT_SPEED) -> None:
    """把两架飞机摆到场景位置，并给一个干净的水平初态。"""
    ac0, ac1 = world.acs
    sx, sy, sz, spsi = subject
    tx, ty, tz, tpsi = target

    ac0.x, ac0.y, ac0.z, ac0.psi = sx, sy, sz, spsi
    ac0.gamma = 0.0
    ac0.v = speed
    ac0.u, ac0.vb, ac0.wb = ac0.v, 0.0, 0.0
    ac0.theta = ac0.phi = 0.0
    ac0.p = ac0.q = ac0.r = 0.0
    ac0.n, ac0.bank, ac0.throttle, ac0.ab = 1.0, 0.0, 1.0, 0.0
    ac0.fuel = ac0.spec.fuel_capacity
    if ac0.rigid:
        # 6DOF 的姿态积分在四元数上，只改 psi 会在第一步被反解回去。
        ac0.quat = quat_from_euler(ac0.phi, ac0.theta, ac0.psi)

    ac1.x, ac1.y, ac1.z, ac1.psi = tx, ty, tz, tpsi
    ac1.gamma = 0.0
    ac1.v = foe_speed
    ac1.u, ac1.vb, ac1.wb = ac1.v, 0.0, 0.0
    ac1.theta = ac1.phi = 0.0
    ac1.p = ac1.q = ac1.r = 0.0
    ac1.n, ac1.bank, ac1.throttle, ac1.ab = 1.0, 0.0, 1.0, 0.0
    ac1.fuel = ac1.spec.fuel_capacity
    if ac1.rigid:
        ac1.quat = quat_from_euler(ac1.phi, ac1.theta, ac1.psi)

    # 演示不涉及武器与伤害：清空挂弹、炮弹、干扰物，血量拉满。
    for ac in (ac0, ac1):
        ac.missiles = 0
        ac.radar_missiles = 0
        ac.ammo = 0.0
        ac.chaff = 0
        ac.flares = 0
        ac.hp = 1.0e9


def _make_foe(kind: str, sign: float) -> object:
    primitive = FOE_PRIMITIVE.get(kind)
    if primitive is None:
        return PassiveTarget()
    period = 2.8 if primitive == "SCISSORS" else None
    return Pilot(primitive, sign=sign, period=period)


def record(name: str, cfg: dict, dof: int,
           dry: bool = False, trace: bool = False) -> dict:
    """录一段演示（dry=True 时只跑物理、不渲染），返回动作统计。"""
    import imageio
    import numpy as np
    import pygame
    from render import Renderer

    seconds = cfg["secs"]
    world = World(
        [SU57_SPEC, F16V_SPEC],
        dt=PHYSICS_DT,
        arena=80000.0,
        time_limit=seconds + 5.0,
        z0=9000.0,
        z_floor=1500.0,
        z_ceiling=30000.0,
        dof=dof,
        y_offset=0.0,
        x_offset=3400.0,
    )
    subject, target = SCENES[cfg["scene"]]
    _reposition(world, subject, target, cfg["speed"], cfg["foe_speed"])

    core = Pilot(
        cfg["prim"], sign=cfg["sign"], seq=cfg["seq"],
        period=2.2 if cfg["prim"] == "SCISSORS" else None,
    )
    core.start(world.observe(0))
    foe = _make_foe(cfg["foe"], cfg["foe_sign"])

    fs = max(1, round(1.0 / (20.0 * PHYSICS_DT)))
    actions = [Action(), Action()]
    steps = 0
    fps = 20
    path = os.path.join(OUT_DIR, f"{name}.mp4")
    writer = None
    if not dry:
        writer = imageio.get_writer(
            path, fps=fps, codec="libx264", quality=7,
            pixelformat="yuv420p", macro_block_size=8,
        )

    z0 = world.acs[0].z
    psi_unwrapped = 0.0
    prev_psi = world.acs[0].psi
    banks: list[float] = []
    gammas: list[float] = []
    ns: list[float] = []
    vs: list[float] = []
    seps: list[float] = []

    # 中点视角：follow=None 时相机取两机几何中心，自动按间距取景。
    renderer = None if dry else Renderer(
        follow=None, gpu=False, span_factor=1.55,
    )
    next_trace = 0.0
    try:
        while world.t < seconds and world.result is None:
            for _ in range(fs):
                if steps % STEPS_PER_DECISION == 0:
                    act = core.act(world.observe(0))
                    actions = [act, foe.act(world.observe(1))]
                    ac0 = world.acs[0]
                    psi_now = ac0.psi
                    # 过顶时欧拉角会在 theta=±90° 处整跳 180°，那不是真实转向；
                    # 只累计接近水平飞行时的航向变化，避免把表示奇点算成掉头。
                    if abs(ac0.gamma) < math.radians(60.0):
                        psi_unwrapped += math.degrees(
                            (psi_now - prev_psi + math.pi) % (2.0 * math.pi) - math.pi
                        )
                    prev_psi = psi_now
                    banks.append(math.degrees(ac0.phi))
                    gammas.append(math.degrees(ac0.gamma))
                    ns.append(ac0.n)
                    vs.append(ac0.v)
                    a, b = world.acs
                    seps.append(
                        math.dist((a.x, a.y, a.z), (b.x, b.y, b.z))
                    )
                    if trace and world.t >= next_trace:
                        print(
                            f"    t={world.t:5.1f} ph{core.core._phase} "
                            f"psi={math.degrees(ac0.psi):7.1f} "
                            f"gam={math.degrees(ac0.gamma):6.1f} "
                            f"phi={math.degrees(ac0.phi):7.1f} "
                            f"bank_cmd={act.bank:5.2f} n_cmd={act.n:4.1f} "
                            f"z={ac0.z:6.0f} v={ac0.v:5.1f} "
                            # 6DOF 下 theta-gamma 近似迎角（翼平做过失速时最直观）
                            f"a={math.degrees(ac0.theta - ac0.gamma):5.1f} "
                            f"sep={seps[-1]:6.0f} foe={foe.primitive}"
                        )
                        next_trace += 1.0
                world.step(actions)
                steps += 1
                if world.result is not None:
                    break
            if dry:
                continue
            renderer.draw(world, False, [core, foe])
            frame = pygame.surfarray.array3d(renderer.screen).swapaxes(0, 1)
            writer.append_data(np.ascontiguousarray(frame))
    finally:
        if writer is not None:
            writer.close()
        if not dry:
            pygame.quit()

    return {
        "primitive": cfg["prim"],
        "scene": cfg["scene"],
        "foe": cfg["foe"],
        "heading_deg": psi_unwrapped,
        "dz_m": world.acs[0].z - z0,
        "bank_min": min(banks) if banks else 0.0,
        "bank_max": max(banks) if banks else 0.0,
        "gamma_min": min(gammas) if gammas else 0.0,
        "gamma_max": max(gammas) if gammas else 0.0,
        "n_max": max(ns) if ns else 0.0,
        "v_min": min(vs) if vs else 0.0,
        "v_max": max(vs) if vs else 0.0,
        "sep_min": min(seps) if seps else 0.0,
        "sep_max": max(seps) if seps else 0.0,
        "done": core.core._maneuver_done,
        "t_end": world.t,
        "path": path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None, help="只录这些原语")
    parser.add_argument("--dof", type=int, default=6)
    parser.add_argument("--start", type=int, default=0, help="跳过前 N 个")
    parser.add_argument("--dry", action="store_true", help="只跑物理、不写视频")
    parser.add_argument("--trace", action="store_true", help="逐秒打印姿态")
    args = parser.parse_args()

    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.makedirs(OUT_DIR, exist_ok=True)

    demos = DEMOS
    if args.only:
        wanted = {s.upper() for s in args.only}
        demos = [d for d in DEMOS if d["prim"] in wanted]

    print(f"{'原语':16s} {'场景':10s} {'对手':11s} {'航向°':>7s} {'Δ高m':>7s} "
          f"{'坡度范围°':>13s} {'γ范围°':>13s} {'n峰':>5s} {'间距m':>13s} 完成")
    rows = []
    for idx, cfg in enumerate(demos[args.start:], start=args.start):
        name = f"{idx:02d}_{cfg['prim']}"
        r = record(name, cfg, args.dof, dry=args.dry, trace=args.trace)
        rows.append(r)
        print(
            f"{r['primitive']:16s} {r['scene']:10s} {r['foe']:11s} "
            f"{r['heading_deg']:7.1f} {r['dz_m']:7.0f} "
            f"{r['bank_min']:6.1f}..{r['bank_max']:6.1f} "
            f"{r['gamma_min']:6.1f}..{r['gamma_max']:6.1f} "
            f"{r['n_max']:5.1f} "
            f"{r['sep_min']:6.0f}..{r['sep_max']:6.0f} "
            f"{'√' if r['done'] else '×'}"
        )

    print(f"\n视频目录: {OUT_DIR}")
    if any(not r["done"] for r in rows):
        print("注意：标记 × 的原语在录制时长内没有走完，可能需要加长或检查分段判据。")
    if any(r["sep_min"] < 120.0 for r in rows):
        print("注意：有机动的双方最近间距小于 120 m，画面里会穿模，可调整场景偏移。")


if __name__ == "__main__":
    main()
