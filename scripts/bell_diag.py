"""克沃丘钟摆单点诊断：机头方位/俯仰角、角速度、迎角、机头水平投影方位。"""
import math, os, sys
sys.path.insert(0, "/home/lky/MyProgramm/dogfight")

from agents import TacticalAgent
from main import F16V_SPEC, PHYSICS_DT, STEPS_PER_DECISION, SU57_SPEC
from sim import Action, World, quat_from_euler, nose_direction

SPEED = float(os.environ.get("SPEED", "320"))
DOF = int(os.environ.get("DOF", "6"))
SECS = float(os.environ.get("SECS", "16"))
YAW1 = float(os.environ.get("YAW1", "nan"))
YAW2 = float(os.environ.get("YAW2", "nan"))
YAW3 = float(os.environ.get("YAW3", "nan"))

world = World(
    [SU57_SPEC, F16V_SPEC], dt=PHYSICS_DT, arena=80000.0,
    time_limit=SECS + 5.0, z0=9000.0, z_floor=1500.0, z_ceiling=30000.0,
    dof=DOF, y_offset=0.0, x_offset=3400.0,
)
ac0, ac1 = world.acs
ac0.x, ac0.y, ac0.z, ac0.psi = 0.0, 0.0, 9000.0, 0.0
ac0.gamma = 0.0
ac0.v = SPEED
ac0.u, ac0.vb, ac0.wb = ac0.v, 0.0, 0.0
ac0.theta = ac0.phi = 0.0
ac0.p = ac0.q = ac0.r = 0.0
ac0.n, ac0.bank, ac0.throttle, ac0.ab = 1.0, 0.0, 1.0, 0.0
ac0.fuel = ac0.spec.fuel_capacity
ac0.quat = quat_from_euler(ac0.phi, ac0.theta, ac0.psi)
for ac in (ac0, ac1):
    ac.missiles = 0; ac.radar_missiles = 0; ac.ammo = 0.0
    ac.chaff = 0; ac.flares = 0; ac.hp = 1.0e9

core = TacticalAgent(energy_pref=0.55, seed=7)
core._enter("KVOCHUR_BELL", world.observe(0))
core.primitive = "KVOCHUR_BELL"
core._turn_sign = -1.0
core._turn_locked = True
core._def_sign = -1.0
core._scissor_sign = -1.0
core._hold = 0.0

_orig = core._kvochur_bell


def _patched(obs):
    a = _orig(obs)
    ph = core._phase
    for k, v in ((1, YAW1), (2, YAW2), (3, YAW3)):
        if ph == k and not math.isnan(v):
            a.yaw_rate = v
    return a


core._kvochur_bell = _patched


class Passive:
    primitive = "-"
    situation = "-"
    def act(self, _obs):
        return Action(bank=0.0, n=1.0, throttle=0.65, fire=False)


foe = Passive()
actions = [Action(), Action()]
steps = 0
spin = 0.0
prev_pp = None
cross = {}
marks = {}
while world.t < SECS and world.result is None:
    if steps % STEPS_PER_DECISION == 0:
        obs = world.observe(0)
        act = core._execute("KVOCHUR_BELL", obs)
        core._phase_t += core.dt
        act.afterburner = 1.0 if act.throttle > 0.9 else 0.0
        actions = [act, foe.act(world.observe(1))]
        az0 = core._bell_az0 or 0.0
        el = math.degrees(obs.own_nose_el)
        azr = math.degrees((obs.own_nose_az - az0 + math.pi) % (2*math.pi) - math.pi)
        fwd = math.cos(math.radians(el)) * math.cos(math.radians(azr))
        pp = math.degrees(math.atan2(math.sin(math.radians(el)), fwd))
        if prev_pp is not None:
            spin += (pp - prev_pp + 180.0) % 360.0 - 180.0
        prev_pp = pp
        prev_el = marks.get("_prev_el", 0.0)
        # 记录“机头朝上 45°”和“机头朝下 45°”两个时刻的水平投影方位，
        # 它们的差值就是用户要求的“朝上—朝下之间的偏航量”。
        if "up" not in marks and prev_el < 45.0 <= el:
            marks["up"] = math.degrees(obs.own_nose_az)
            marks["up_acc"] = math.degrees(core._bell_yaw_acc)
        if "top" not in marks and "up" in marks and prev_el > 45.0 >= el:
            marks["top"] = math.degrees(obs.own_nose_az)
            marks["top_acc"] = math.degrees(core._bell_yaw_acc)
        if "down" not in marks and prev_el > -45.0 >= el:
            marks["down"] = math.degrees(obs.own_nose_az)
            marks["down_acc"] = math.degrees(core._bell_yaw_acc)
        marks["_prev_el"] = el
        nx, ny, nz = nose_direction(ac0)
        tilt = math.sqrt(nx*nx + ny*ny)
        taz = math.degrees(math.atan2(ny, nx))
        if tilt > 0.25:
            cross[f"spin{int(abs(spin)//30*30)}"] = round(taz, 1)
        alpha = math.degrees(math.atan2(ac0.wb, max(ac0.u, 1.0)))
        print(
            f"t={world.t:5.2f} ph{core._phase} spin={spin:7.1f} pp={pp:7.1f} "
            f"el={el:6.1f} azr={azr:7.1f} acc={math.degrees(core._bell_yaw_acc):7.1f} "
            f"tilt={tilt:4.2f} taz={taz:7.1f} "
            f"th={math.degrees(ac0.theta):6.1f} gam={math.degrees(ac0.gamma):6.1f} "
            f"phi={math.degrees(ac0.phi):7.1f} "
            f"p={ac0.p:5.2f} q={ac0.q:5.2f} r={ac0.r:5.2f} "
            f"a={alpha:6.1f} v={ac0.v:5.1f} yr={act.yaw_rate:4.2f}"
        )
    world.step(actions)
    steps += 1

up = marks.get("up")
down = marks.get("down")
yaw_span = None
if up is not None and down is not None:
    yaw_span = ((down - up + 180.0) % 360.0) - 180.0
print("done", core._maneuver_done, "z", round(ac0.z),
      "psi", round(math.degrees(ac0.psi), 1))
print("marks up/down az:", up, down,
      "yaw_span(deg):", None if yaw_span is None else round(yaw_span, 1),
      "acc:", round(math.degrees(core._bell_yaw_acc), 1))
print("up/top/down:", marks.get("up"), marks.get("top"), marks.get("down"),
      "swept:", None if "up_acc" not in marks else
      round(marks["top_acc"] - marks["up_acc"], 1))
print("taz samples:", cross)
