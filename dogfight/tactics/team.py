"""团队决策层（P3）：按军阶/职务定长僚机，执行双机编队协同原语。

长僚机分工只看 `Aircraft.rank`（flight_lead / element_lead / wingman）与
入队顺序，不看血量、战绩或剩余弹药；长机战损后由同队下一个军阶最高者接替。

协同原语参考外军双机编队条令（element / engaged vs supporting fighter /
bracket / drag / loose deuce / double attack 等）：

- bracket        钳形：双机分左右两臂进入，目标夹在中间，通常 25 km 外、
                 目标航向与视线夹角接近对头时使用；
- double_attack  双击：长机接战，僚机留在外侧高处保留能量，长机失去优势
                 或僚机进入武器包线后再进入；
- loose_deuce    松散双机：双机拉开间距，谁被目标咬住，另一机就反制；
- fix_and_kill   牵制+击杀：一架压住目标（fix），另一架外绕抢角度；
- drag_bait      拖曳诱敌：一架拖曳并保持无线电静默/不开火，另一机埋伏；
- thach_weave    防御交叉：长机被尾追时双机对穿，由僚机反制追击者；
- disengage      脱离：弹药/能量见底或局势不利时，全员收火并掩护脱离。

约束：本模块只写 `Aircraft` 上的团队指令（目标、角色、火力状态、队形偏移），
物理与单机机动仍由 `World` / `TacticalAgent` 负责；1v1 不启用。
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from dogfight.core.sim import Action, World

RANK_PRIORITY = {"flight_lead": 3, "element_lead": 2, "wingman": 1}

# 队形偏移（米）与高度偏移（米）
BRACKET_OFFSET = 7000.0
BRACKET_ALT = 600.0
DOUBLE_ATTACK_OFFSET = 4500.0
DOUBLE_ATTACK_ALT = 1500.0
LOOSE_DEUCE_OFFSET = 3000.0
FIX_OFFSET = 6500.0

# 脱离-再进入循环
RESET_ALT = 7000.0        # 重整目标高度
REENGAGE_ALT = 6000.0     # 爬到此高度即可重新进入
RESET_TRIGGER_ALT = 4500.0
STALEMATE_S = 35.0        # 无发射/无进展超过该时间即考虑脱离
RESET_COOLDOWN_S = 20.0   # 完成一次重整后的冷却，避免连续触发

# 中队长目标决策：集中火力 / 分头接战的驻留时间，避免目标数抖动时来回切。
FOCUS_LOCK_S = 12.0


@dataclass
class Order:
    target: int
    role: str            # engaged / support / bait / cover / egress
    weapons: str         # free / tight / hold
    offset_m: float = 0.0
    alt_m: float = 0.0
    min_alt: float = 0.0


class TeamBrain:
    """按队伍维护战术、长僚机分工与协同行为。"""

    def __init__(self, world: World) -> None:
        self.world = world
        self.tactic: dict[int, str] = {0: "default", 1: "default"}
        self.orders: dict[int, Order] = {}
        self._planned = False
        # 战术切换的驻留时间：避免在距离门限附近每拍来回翻转。
        self._tactic_t: dict[int, float] = {0: 0.0, 1: 0.0}
        self._tactic_lock_s = 6.0
        # 脱离-再进入状态机
        self.state: dict[int, str] = {0: "engage", 1: "engage"}
        self._launch_seen: dict[int, int] = {0: 0, 1: 0}
        self._last_launch_t: dict[int, float] = {0: 0.0, 1: 0.0}
        self._reset_ready: dict[int, float] = {0: -1e9, 1: -1e9}
        # 中队长决策：True = 两个双机集中打一个目标；False = 各双机分头接战。
        self.focus: dict[int, bool] = {0: True, 1: True}
        self._focus_t: dict[int, float] = {0: 0.0, 1: 0.0}

    # ---------------- 基础查询 ----------------

    def _members(self, team: int) -> list[int]:
        w = self.world
        return [
            i for i, ac in enumerate(w.acs)
            if ac.team == team and ac.alive and ac.hp > 0.0
        ]

    def _enemies(self, team: int) -> list[int]:
        w = self.world
        return [
            i for i, ac in enumerate(w.acs)
            if ac.team != team and ac.alive and ac.hp > 0.0
        ]

    def _leader(self, team: int) -> int:
        """按军阶选长机；军阶相同按入队顺序，绝不按血量。"""
        members = self._members(team)
        if not members:
            return -1
        members.sort(
            key=lambda i: (
                -RANK_PRIORITY.get(self.world.acs[i].rank, 0),
                self.world.acs[i].slot_index,
            )
        )
        return members[0]

    def _elements(self, team: int) -> list[list[int]]:
        """把队伍拆成双机编队：每名小队长（含中队长）带一架僚机。

        四机典型结构为 flight_lead / wingman / element_lead / wingman，
        按入队顺序配对后得到 [中队长, 僚机] 与 [第二小队长, 僚机]。
        """
        members = self._members(team)
        if not members:
            return []
        ordered = sorted(members, key=lambda i: self.world.acs[i].slot_index)
        leads = [
            i for i in ordered
            if RANK_PRIORITY.get(self.world.acs[i].rank, 0) >= 2
        ]
        wings = [i for i in ordered if i not in leads]
        elements: list[list[int]] = []
        for lead in leads:
            element = [lead]
            if wings:
                element.append(wings.pop(0))
            elements.append(element)
        # 小队长全部战损时，剩余僚机各自成组，仍按单机执行。
        while wings:
            elements.append([wings.pop(0)])
        return elements

    def _choose_focus(self, team: int, threats: list[int]) -> bool:
        """中队长决策：两个双机集火一个目标，还是分头各打一个。

        目标数多于双机数时分头压制；目标收拢到双机数以内后改为集火，
        若其中已有重伤目标则优先集火补刀。单双机队伍始终由本机战术决定。
        """
        elements = self._elements(team)
        if len(elements) <= 1 or len(threats) <= 1:
            return True
        if len(threats) > len(elements):
            return False
        return any(
            self.world.acs[j].hp <= 0.5 * self.world.acs[j].spec.hp_max
            for j in threats
        )

    def _apply_focus_targets(self, team: int, threats: list[int]) -> None:
        """按中队长的决策重写每架飞机的主目标；单双机队伍保持原战术分配。"""
        elements = self._elements(team)
        if len(elements) <= 1 or not threats:
            return
        for k, element in enumerate(elements):
            target = (
                threats[0] if self.focus[team]
                else threats[k % len(threats)]
            )
            for i in element:
                self.world.acs[i].assigned_target = target
                order = self.orders.get(i)
                if order is not None:
                    order.target = target

    def _shooter_priority(self, i: int) -> tuple:
        """同一目标的主射手顺序：接战角色优先，其次军阶，最后入队顺序。"""
        ac = self.world.acs[i]
        return (
            0 if ac.order_role == "engaged" else 1,
            -RANK_PRIORITY.get(ac.rank, 0),
            ac.slot_index,
        )

    def _launch_target(self, i: int, radar: bool) -> int:
        """实际发射目标：优先火控锁定对象，其次团队分配目标。

        发射门控用的是 `radar_locked_target` / `locked_target`，若只按
        `assigned_target` 去重，僚机锁定别的敌机后仍可能跟长机齐射同一目标。
        """
        w = self.world
        ac = w.acs[i]
        j = ac.radar_locked_target if radar else ac.locked_target
        if 0 <= j < len(w.acs) and w.acs[j].alive and w.acs[j].team != ac.team:
            return j
        return w._primary(i)

    def _dist(self, i: int, j: int) -> float:
        a, b = self.world.acs[i], self.world.acs[j]
        return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)

    def _threat(self, own: int, foe_index: int) -> float:
        """威胁度：近、残血、带弹、机头指向我方，分数越高越优先。"""
        w = self.world
        me, foe = w.acs[own], w.acs[foe_index]
        dist = self._dist(own, foe_index)
        score = -dist / 10000.0
        score += 1.5 * (1.0 - foe.hp / foe.spec.hp_max)
        if foe.radar_missiles > 0:
            score += 0.8
        if foe.missiles > 0:
            score += 0.7
        if dist > 1e-6:
            nx, ny, nz = _nose(foe)
            cosang = (
                (me.x - foe.x) * nx + (me.y - foe.y) * ny + (me.z - foe.z) * nz
            ) / dist
            if cosang > math.cos(math.radians(30.0)):
                score += 0.6
        return score

    def _incoming_shooter(self, own: int, team: int) -> int:
        w = self.world
        for m in w.missiles:
            if m.alive and m.target == own and w.team_of[m.owner] != team:
                return m.owner
        return -1

    def _enemy_behind(self, own: int, max_dist: float = 6000.0) -> int:
        """咬在本机尾后的敌机（机头指向本机、且在本机后方）。"""
        w = self.world
        me = self.world.acs[own]
        best, best_d = -1, math.inf
        for j in self._enemies(me.team):
            d = self._dist(own, j)
            if d > max_dist:
                continue
            nx, ny, nz = _nose(me)
            ux = (w.acs[j].x - me.x) / max(d, 1e-6)
            uy = (w.acs[j].y - me.y) / max(d, 1e-6)
            uz = (w.acs[j].z - me.z) / max(d, 1e-6)
            if ux * nx + uy * ny + uz * nz < -0.3 and d < best_d:
                best, best_d = j, d
        return best

    def _missiles_left(self, team: int) -> int:
        return sum(
            ac.missiles + ac.radar_missiles
            for ac in self.world.acs
            if ac.team == team and ac.alive
        )

    def _team_alt(self, team: int) -> float:
        members = [self.world.acs[i] for i in self._members(team)]
        if not members:
            return 0.0
        return sum(ac.z for ac in members) / len(members)

    def _team_energy(self, team: int) -> float:
        members = [self.world.acs[i] for i in self._members(team)]
        if not members:
            return 0.0
        return sum(ac.energy for ac in members) / len(members)

    def _alt_adv(self, team: int) -> float:
        return self._team_alt(team) - self._team_alt(1 - team)

    def _energy_adv(self, team: int) -> float:
        return self._team_energy(team) - self._team_energy(1 - team)

    def _nearest_enemy(self, own: int) -> int:
        enemies = self._enemies(self.world.acs[own].team)
        if not enemies:
            return -1
        return min(enemies, key=lambda j: self._dist(own, j))

    def _enemy_has_fired(self, team: int) -> bool:
        w = self.world
        return any(
            w.stats["missiles_launched"][i] > 0
            for i, ac in enumerate(w.acs) if ac.team != team
        )

    # ---------------- 战术选择 ----------------

    def _head_on(self, team: int) -> bool:
        """双方接近段是否仍接近对头（bracket 的前置条件）。"""
        leader = self._leader(team)
        if leader < 0:
            return False
        target = self._nearest_enemy(leader)
        if target < 0:
            return False
        me, foe = self.world.acs[leader], self.world.acs[target]
        d = max(self._dist(leader, target), 1.0)
        nx, ny, nz = _nose(foe)
        # 目标机头指向我方 → 对头。
        return (
            (me.x - foe.x) * nx + (me.y - foe.y) * ny + (me.z - foe.z) * nz
        ) / d > math.cos(math.radians(35.0))

    def _update_launch_clock(self) -> None:
        """记录两队最近一次发射时刻，用于识别“打了很久没有进展”。"""
        w = self.world
        for team in (0, 1):
            total = sum(
                w.stats["missiles_launched"][i]
                for i, ac in enumerate(w.acs) if ac.team == team
            )
            if total > self._launch_seen[team]:
                self._launch_seen[team] = total
                self._last_launch_t[team] = w.t

    def _reset_safe(
        self, team: int, members: list[int], nearest: float
    ) -> bool:
        if self._missiles_left(team) == 0:
            return False
        if len(members) < 1 or nearest < 3000.0:
            return False
        return all(
            self._incoming_shooter(i, team) < 0 for i in members
        )

    def _update_reset_state(
        self,
        team: int,
        members: list[int],
        enemies: list[int],
        nearest: float,
    ) -> tuple[str, str | None]:
        """脱离-再进入状态机。

        engage → reset：低空、或 35 s 内没有发射且已进入 8 km 缠斗。
        reset → engage：爬回 6 km；或被对手追入 10 km 且已有 1.5 km 高度
        优势——此时直接转 high_low 高位反打，而不是继续逃。
        """
        w = self.world
        t = w.t
        callsign = w.acs[members[0]].callsign
        if self.state[team] == "reset":
            alt = self._team_alt(team)
            if alt >= REENGAGE_ALT:
                self.state[team] = "engage"
                self._reset_ready[team] = t + RESET_COOLDOWN_S
                w.events.append(
                    (t, f"{callsign} 重整完成（{alt / 1000.0:.1f} km），重新进入")
                )
                return "engage", None
            if (
                alt >= 5200.0
                and self._alt_adv(team) >= 1500.0
                and nearest <= 10000.0
            ):
                self.state[team] = "engage"
                self._reset_ready[team] = t + RESET_COOLDOWN_S
                w.events.append(
                    (t, f"{callsign} 对手追入，高位反打（{alt / 1000.0:.1f} km）")
                )
                return "engage", "high_low"
            return "reset", "energy_reset"

        if t < 60.0 or t < self._reset_ready[team]:
            return "engage", None
        if not self._reset_safe(team, members, nearest):
            return "engage", None
        low = self._team_alt(team) < RESET_TRIGGER_ALT and nearest > 3500.0
        # 无进展只在低空才值得脱离重整：已经在 6 km 以上、能量充足时，
        # 脱离爬升没有收益，应继续施压而不是每 35 s 空转一次。
        stale = (
            t - self._last_launch_t[team] > STALEMATE_S
            and 3000.0 <= nearest <= 8000.0
            and self._team_alt(team) < REENGAGE_ALT
        )
        if low or stale:
            self.state[team] = "reset"
            reason = "低空" if low else f"无进展 {t - self._last_launch_t[team]:.0f}s"
            w.events.append((t, f"{callsign} 脱离重整（{reason}）"))
            return "reset", "energy_reset"
        return "engage", None

    def _choose(self, team: int) -> str:
        w = self.world
        members = self._members(team)
        enemies = self._enemies(team)
        if not members or not enemies:
            return "default"
        leader = self._leader(team)
        if self._missiles_left(team) == 0:
            return "disengage"
        nearest = min(self._dist(leader, j) for j in enemies)
        state, forced = self._update_reset_state(team, members, enemies, nearest)
        # 长机被尾追/被弹：先看能否由僚机反制。超视距的来袭弹交给单机
        # 的防御/notch 处理，只有进入目视距离才切交叉支援。防御优先于重整。
        if len(members) > 1:
            if (
                self._incoming_shooter(leader, team) >= 0
                and nearest <= 15000.0
            ):
                return "thach_weave"
            if self._enemy_behind(leader) >= 0:
                return "drag_bait"
        if forced is not None:
            return forced
        # 钳形一旦展开就保持到 18 km 内再收拢，避免在 22 km 门限附近
        # 因对头角短暂抖动而在 bracket / double_attack 之间来回切。
        if self.tactic.get(team) == "bracket" and nearest > 18000.0:
            return "bracket"
        if len(members) > len(enemies):
            return "fix_and_kill"
        alt_adv = self._alt_adv(team)
        if nearest > 22000.0:
            if alt_adv >= 500.0:
                return "vertical_bracket"
            return "bracket" if self._head_on(team) else "double_attack"
        if nearest > 14000.0:
            return "high_low" if alt_adv >= 800.0 else "double_attack"
        if alt_adv >= 1000.0 and len(members) > 1:
            return "high_low"
        return "loose_deuce"

    def plan(self) -> None:
        """每个决策拍调用：选战术、定长僚机、写协同指令。"""
        w = self.world
        self._update_launch_clock()
        if not self._planned:
            self._planned = True
            for team in (0, 1):
                self.tactic[team] = self._choose(team)
                self._tactic_t[team] = w.t
                w.events.append((
                    w.t, f"{w.team_names[team]} 选择 {self.tactic[team]}",
                ))
        else:
            for team in (0, 1):
                tactic = self._choose(team)
                if tactic == self.tactic[team]:
                    continue
                # 防御性切换（被弹/被尾追/脱离）立即生效；其余至少驻留 6 s。
                urgent = tactic in (
                    "thach_weave", "drag_bait", "disengage", "energy_reset",
                )
                if not urgent and w.t - self._tactic_t[team] < self._tactic_lock_s:
                    continue
                self.tactic[team] = tactic
                self._tactic_t[team] = w.t
                w.events.append((
                    w.t, f"{w.team_names[team]} 转 {tactic}",
                ))

        for team in (0, 1):
            self._update_focus(team)
            self._plan_team(team)

    def _update_focus(self, team: int) -> None:
        """中队长按当前威胁排序决定集火还是分头，并带驻留时间避免抖动。"""
        members = self._members(team)
        enemies = self._enemies(team)
        if not members or not enemies:
            return
        leader = self._leader(team)
        threats = sorted(enemies, key=lambda j: self._threat(leader, j), reverse=True)
        want = self._choose_focus(team, threats)
        if want == self.focus[team]:
            return
        # 开局第一拍允许直接定下方案，之后切换需满足驻留时间。
        if self.world.t > 0.0 and self.world.t - self._focus_t[team] < FOCUS_LOCK_S:
            return
        self.focus[team] = want
        self._focus_t[team] = self.world.t
        mode = "集中火力" if want else "分头接战"
        self.world.events.append((
            self.world.t, f"{self.world.team_names[team]} 中队长决定{mode}",
        ))

    def _plan_team(self, team: int) -> None:
        w = self.world
        members = self._members(team)
        enemies = self._enemies(team)
        if not members or not enemies:
            return
        leader = self._leader(team)
        wingmen = [i for i in members if i != leader]
        threats = sorted(enemies, key=lambda j: self._threat(leader, j), reverse=True)
        tactic = self.tactic[team]

        # 角色交换：长机弹尽而僚机还有弹，由僚机接替接战（EF/SF 交换）。
        if (
            wingmen
            and w.acs[leader].missiles + w.acs[leader].radar_missiles == 0
            and w.acs[wingmen[0]].missiles + w.acs[wingmen[0]].radar_missiles > 0
        ):
            leader, wingmen = wingmen[0], [leader] + wingmen[1:]

        if tactic == "bracket":
            self._plan_bracket(leader, wingmen, threats)
        elif tactic == "vertical_bracket":
            self._plan_vertical_bracket(leader, wingmen, threats)
        elif tactic == "high_low":
            self._plan_high_low(leader, wingmen, threats)
        elif tactic == "energy_reset":
            self._plan_energy_reset(members, threats)
        elif tactic == "double_attack":
            self._plan_double_attack(leader, wingmen, threats)
        elif tactic == "fix_and_kill":
            self._plan_fix_and_kill(leader, wingmen, threats)
        elif tactic == "drag_bait":
            self._plan_drag(team, leader, wingmen, threats)
        elif tactic == "thach_weave":
            self._plan_thach(team, leader, wingmen, threats)
        elif tactic == "disengage":
            self._plan_disengage(team, members, threats)
        else:
            self._plan_loose_deuce(leader, wingmen, threats)

        # 防御/脱离类战术有明确的反制或撤离目标，不再按中队长决策覆盖。
        if tactic not in ("drag_bait", "thach_weave", "energy_reset", "disengage"):
            self._apply_focus_targets(team, threats)

    # ---------------- 各原语的指令分配 ----------------

    def _plan_bracket(self, leader: int, wingmen: list[int], threats: list[int]) -> None:
        self._assign(leader, threats[0], "engaged", "tight", -BRACKET_OFFSET, BRACKET_ALT)
        for k, wing in enumerate(wingmen):
            target = threats[(k + 1) % len(threats)]
            self._assign(wing, target, "support", "tight", BRACKET_OFFSET, BRACKET_ALT)

    def _plan_vertical_bracket(
        self, leader: int, wingmen: list[int], threats: list[int]
    ) -> None:
        """高低钳形：长机走低臂，僚机走高臂（6.5 km 以上），形成垂直夹击。"""
        self._assign(
            leader, threats[0], "engaged", "tight",
            -BRACKET_OFFSET, 0.0, min_alt=2500.0,
        )
        for k, wing in enumerate(wingmen):
            target = threats[(k + 1) % len(threats)]
            self._assign(
                wing, target, "support", "tight",
                BRACKET_OFFSET, DOUBLE_ATTACK_ALT, min_alt=6500.0,
            )

    def _plan_high_low(
        self, leader: int, wingmen: list[int], threats: list[int]
    ) -> None:
        """高位掩护：长机低空接战，僚机保持在 6.5 km 以上待机/反制。"""
        self._assign(
            leader, threats[0], "engaged", "free", 0.0, 0.0, min_alt=3000.0,
        )
        for k, wing in enumerate(wingmen):
            attacker = self._enemy_behind(leader, 9000.0)
            target = attacker if attacker >= 0 else threats[0]
            self._assign(
                wing, target, "support", "tight",
                DOUBLE_ATTACK_OFFSET, DOUBLE_ATTACK_ALT,
                side=self._side_of(wing, leader), min_alt=6500.0,
            )

    def _plan_energy_reset(
        self, members: list[int], threats: list[int]
    ) -> None:
        """能量重整：全队收火爬回 7 km，高度够了再重新进入。"""
        for i in members:
            self._assign(
                i, threats[0], "egress", "hold",
                0.0, 0.0, min_alt=RESET_ALT, disengage=True,
            )

    def _plan_double_attack(
        self, leader: int, wingmen: list[int], threats: list[int]
    ) -> None:
        self._assign(leader, threats[0], "engaged", "free", 0.0, 0.0)
        for wing in wingmen:
            self._assign(
                wing, threats[0], "support", "tight",
                DOUBLE_ATTACK_OFFSET, DOUBLE_ATTACK_ALT, min_alt=6000.0,
            )

    def _plan_loose_deuce(
        self, leader: int, wingmen: list[int], threats: list[int]
    ) -> None:
        self._assign(leader, threats[0], "engaged", "free", 0.0, 0.0, min_alt=2500.0)
        for k, wing in enumerate(wingmen):
            # 谁在咬长机，僚机就反制谁；否则打次号威胁。
            attacker = self._enemy_behind(leader, 9000.0)
            target = attacker if attacker >= 0 else threats[(k + 1) % len(threats)]
            self._assign(
                wing, target, "support", "tight",
                LOOSE_DEUCE_OFFSET, 300.0, side=self._side_of(wing, leader),
                min_alt=5000.0,
            )

    def _plan_fix_and_kill(
        self, leader: int, wingmen: list[int], threats: list[int]
    ) -> None:
        # 长机压住目标（fix），僚机外绕抢角度（kill）。
        self._assign(leader, threats[0], "engaged", "tight", 0.0, 0.0)
        for k, wing in enumerate(wingmen):
            target = threats[(k + 1) % len(threats)]
            self._assign(
                wing, target, "support", "free",
                FIX_OFFSET, 1200.0, side=self._side_of(wing, leader),
                min_alt=6500.0,
            )

    def _plan_drag(self, team: int, leader: int, wingmen: list[int], threats: list[int]) -> None:
        # 长机拖曳并收火；僚机保持外侧待机，等对手先开火或进入包线。
        self._assign(leader, threats[0], "bait", "hold", 0.0, 0.0)
        for k, wing in enumerate(wingmen):
            weapons = "free" if self._enemy_has_fired(team) else "hold"
            self._assign(
                wing, threats[0], "support", weapons,
                DOUBLE_ATTACK_OFFSET, DOUBLE_ATTACK_ALT,
                side=self._side_of(wing, leader), min_alt=6500.0,
            )

    def _plan_thach(self, team: int, leader: int, wingmen: list[int], threats: list[int]) -> None:
        shooter = self._incoming_shooter(leader, team)
        attacker = shooter if shooter >= 0 else self._enemy_behind(leader, 9000.0)
        if attacker < 0:
            attacker = threats[0]
        # 长机收火并向僚机方向交叉；僚机自由开火反制追击者。
        self._assign(
            leader, attacker, "egress", "hold",
            2500.0, 0.0, side=self._side_of(leader, wingmen[0]) if wingmen else 1.0,
        )
        for wing in wingmen:
            self._assign(wing, attacker, "support", "free", 0.0, 300.0)

    def _plan_disengage(self, team: int, members: list[int], threats: list[int]) -> None:
        w = self.world
        for i in members:
            ac = w.acs[i]
            home = 1.0 if ac.x < 0.0 else -1.0
            self._assign(
                i, threats[0], "egress", "hold", home * 6000.0, 0.0,
                disengage=True,
            )

    def _assign(
        self,
        own: int,
        target: int,
        role: str,
        weapons: str,
        offset_m: float,
        alt_m: float,
        side: float = 1.0,
        min_alt: float = 0.0,
        disengage: bool = False,
    ) -> None:
        ac = self.world.acs[own]
        ac.assigned_target = target
        ac.order_role = role
        ac.weapons_state = weapons
        ac.formation_offset_m = side * offset_m
        ac.formation_alt_m = alt_m
        ac.team_min_alt = min_alt
        ac.team_disengage = disengage
        self.orders[own] = Order(
            target, role, weapons, ac.formation_offset_m, alt_m, min_alt,
        )

    def _side_of(self, own: int, other: int) -> float:
        """相对僚机/长机的横向侧：保证双机向相反方向展开。"""
        w = self.world
        a, b = w.acs[own], w.acs[other]
        return 1.0 if a.y >= b.y else -1.0

    # ---------------- 火力去重 / 射击纪律 ----------------

    def filter_actions(self, actions: list[Action]) -> None:
        w = self.world
        for team in (0, 1):
            members = self._members(team)
            if not members:
                continue

            # 射击纪律：hold 全禁；tight 限制中距弹在包线外发射。
            for i in members:
                if i >= len(actions):
                    continue
                state = w.acs[i].weapons_state
                if state == "hold":
                    actions[i].launch = False
                    actions[i].launch_radar = False
                    actions[i].fire = False
                elif state == "tight":
                    target = self._launch_target(i, True)
                    if target >= 0:
                        spec = w.acs[i].spec.radar_missile
                        if spec is not None:
                            reach = spec.max_range * 0.7
                            if self._dist(i, target) > reach:
                                actions[i].launch_radar = False

            # 克制发射：同一目标只留一个主射手，其余飞机掩护，伺机补刀。
            # 主射手按接战角色 → 军阶 → 入队顺序选出；已有一枚本队导弹在飞
            # 且目标尚未重伤时，其他人不再补射，避免一轮齐射浪费多枚弹。
            for want_radar in (True, False):
                by_target: dict[int, list[int]] = {}
                for i in members:
                    if i >= len(actions):
                        continue
                    if (actions[i].launch_radar if want_radar else actions[i].launch):
                        target = self._launch_target(i, want_radar)
                        if target >= 0:
                            by_target.setdefault(target, []).append(i)
                for target, shooters in by_target.items():
                    shooters.sort(key=self._shooter_priority)
                    if want_radar:
                        in_flight = any(
                            m.alive and m.target == target
                            and w.acs[m.owner].team == team
                            for m in w.missiles
                        )
                        hurt = (
                            w.acs[target].hp
                            <= 0.5 * w.acs[target].spec.hp_max
                        )
                        if in_flight and not hurt:
                            for i in shooters:
                                actions[i].launch_radar = False
                            continue
                    # 只保留优先级最高的那架；目标重伤时下一拍由它补刀。
                    for i in shooters[1:]:
                        if want_radar:
                            actions[i].launch_radar = False
                        else:
                            actions[i].launch = False


def _nose(ac) -> tuple[float, float, float]:
    """机头方向单位向量（6DOF 取机体 x 轴，点质量取速度方向）。"""
    from dogfight.core.sim import nose_direction
    return nose_direction(ac)
