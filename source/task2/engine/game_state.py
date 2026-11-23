from typing import Optional
from .actions import ACTION_TO_DIRECTION, Action, Direction
from .entities import Base, Bullet, Tank
from .game_map import GameMap


class GameState:
    __slots__ = (
        "width",
        "height",
        "walls",
        "tanks",
        "bases",
        "bullets",
        "cycle",
        "max_cycles",
        "max_bullets_per_tank",
    )

    TANK_INITIAL_HP = 6
    BASE_INITIAL_HP = 9
    WALL_INITIAL_HP = 3
    BULLET_SPEED = 3
    DAMAGE_PER_HIT = 1
    WIN_UTILITY = 1000.0

    def __init__(
        self,
        width: int,
        height: int,
        walls: dict[tuple[int, int], int],
        tanks: tuple[Tank, Tank],
        bases: tuple[Base, Base],
        bullets: list[Bullet],
        cycle: int = 0,
        max_cycles: int = 200,
        max_bullets_per_tank: Optional[int] = None,
    ) -> None:
        self.width = width
        self.height = height
        self.walls = walls
        self.tanks = tanks
        self.bases = bases
        self.bullets = bullets
        self.cycle = cycle
        self.max_cycles = max_cycles
        self.max_bullets_per_tank = max_bullets_per_tank

    @classmethod
    def from_map(
        cls,
        game_map: GameMap,
        max_cycles: int = 200,
        max_bullets_per_tank: Optional[int] = None,
    ) -> "GameState":
        walls = {pos: cls.WALL_INITIAL_HP for pos in game_map.walls}
        t0_x, t0_y = game_map.tank_spawns[0]
        t1_x, t1_y = game_map.tank_spawns[1]
        b0_x, b0_y = game_map.base_positions[0]
        b1_x, b1_y = game_map.base_positions[1]
        tanks = (
            Tank(0, t0_x, t0_y, Direction.UP, cls.TANK_INITIAL_HP),
            Tank(1, t1_x, t1_y, Direction.DOWN, cls.TANK_INITIAL_HP),
        )
        bases = (
            Base(0, b0_x, b0_y, cls.BASE_INITIAL_HP),
            Base(1, b1_x, b1_y, cls.BASE_INITIAL_HP),
        )
        return cls(
            width=game_map.width,
            height=game_map.height,
            walls=walls,
            tanks=tanks,
            bases=bases,
            bullets=[],
            cycle=0,
            max_cycles=max_cycles,
            max_bullets_per_tank=max_bullets_per_tank,
        )

    def clone(self) -> "GameState":
        return GameState(
            width=self.width,
            height=self.height,
            walls=self.walls.copy(),
            tanks=(self.tanks[0].clone(), self.tanks[1].clone()),
            bases=(self.bases[0].clone(), self.bases[1].clone()),
            bullets=[b.clone() for b in self.bullets],
            cycle=self.cycle,
            max_cycles=self.max_cycles,
            max_bullets_per_tank=self.max_bullets_per_tank,
        )

    def walls_key(self) -> tuple[tuple[tuple[int, int], int], ...]:
        return tuple(sorted(self.walls.items()))

    def state_key(self) -> tuple:
        t0, t1 = self.tanks
        b0, b1 = self.bases
        return (
            (t0.x, t0.y, t0.facing, t0.hp),
            (t1.x, t1.y, t1.facing, t1.hp),
            (b0.hp, b1.hp),
            self.walls_key(),
            tuple((b.x, b.y, b.direction, b.owner) for b in self.bullets),
        )

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def is_blocked(self, x: int, y: int, ignore_tank_id: Optional[int] = None) -> bool:
        if not self.in_bounds(x, y):
            return True
        if (x, y) in self.walls:
            return True
        for base in self.bases:
            if base.hp > 0 and base.x == x and base.y == y:
                return True
        for tank in self.tanks:
            if tank.player_id != ignore_tank_id and tank.hp > 0 and tank.x == x and tank.y == y:
                return True
        return False

    def legal_actions(self, player_id: int) -> list[Action]:
        tank = self.tanks[player_id]
        if tank.hp <= 0:
            return [Action.STAY]
        actions: list[Action] = []
        for action, direction in ACTION_TO_DIRECTION.items():
            nx = tank.x + direction.dx
            ny = tank.y + direction.dy
            if not self.is_blocked(nx, ny, ignore_tank_id=player_id):
                actions.append(action)
        if self.max_bullets_per_tank is None:
            actions.append(Action.FIRE)
        else:
            active_count = sum(1 for b in self.bullets if b.owner == player_id)
            if active_count < self.max_bullets_per_tank:
                actions.append(Action.FIRE)
        actions.append(Action.STAY)
        return actions

    def step_cycle(self, action_p0: Action, action_p1: Action) -> "GameState":
        nxt = self.clone()
        actions = (action_p0, action_p1)
        intended_moves: dict[int, tuple[int, int, Direction]] = {}
        for pid in (0, 1):
            tank = nxt.tanks[pid]
            if tank.hp <= 0:
                continue
            action = actions[pid]
            direction = ACTION_TO_DIRECTION.get(action)
            if direction is not None:
                nx = tank.x + direction.dx
                ny = tank.y + direction.dy
                if not nxt.is_blocked(nx, ny, ignore_tank_id=pid):
                    intended_moves[pid] = (nx, ny, direction)
        same_target = (
            0 in intended_moves
            and 1 in intended_moves
            and intended_moves[0][:2] == intended_moves[1][:2]
        )
        for pid, (nx, ny, direction) in intended_moves.items():
            tank = nxt.tanks[pid]
            tank.facing = direction
            if not same_target:
                tank.x = nx
                tank.y = ny
        surviving_bullets: list[Bullet] = []
        for bullet in nxt.bullets:
            hit = False
            for tank in nxt.tanks:
                if (
                    tank.hp > 0
                    and tank.player_id != bullet.owner
                    and tank.x == bullet.x
                    and tank.y == bullet.y
                ):
                    tank.hp -= self.DAMAGE_PER_HIT
                    hit = True
                    break
            if not hit:
                surviving_bullets.append(bullet)
        nxt.bullets = surviving_bullets
        for pid in (0, 1):
            tank = nxt.tanks[pid]
            if tank.hp > 0 and actions[pid] == Action.FIRE:
                if self.max_bullets_per_tank is None or (
                    sum(1 for b in nxt.bullets if b.owner == pid) < self.max_bullets_per_tank
                ):
                    nxt.bullets.append(Bullet(tank.x, tank.y, tank.facing, pid))
        for _ in range(self.BULLET_SPEED):
            if not nxt.bullets:
                break
            next_bullets: list[Bullet] = []
            for bullet in nxt.bullets:
                nx = bullet.x + bullet.direction.dx
                ny = bullet.y + bullet.direction.dy
                if not nxt.in_bounds(nx, ny):
                    continue
                pos = (nx, ny)
                if pos in nxt.walls:
                    nxt.walls[pos] -= self.DAMAGE_PER_HIT
                    if nxt.walls[pos] <= 0:
                        del nxt.walls[pos]
                    continue
                hit_entity = False
                for tank in nxt.tanks:
                    if (
                        tank.hp > 0
                        and tank.player_id != bullet.owner
                        and tank.x == nx
                        and tank.y == ny
                    ):
                        tank.hp -= self.DAMAGE_PER_HIT
                        hit_entity = True
                        break
                if hit_entity:
                    continue
                for base in nxt.bases:
                    if base.hp > 0 and base.x == nx and base.y == ny:
                        base.hp -= self.DAMAGE_PER_HIT
                        hit_entity = True
                        break
                if hit_entity:
                    continue
                bullet.x = nx
                bullet.y = ny
                next_bullets.append(bullet)
            nxt.bullets = next_bullets
        nxt.cycle += 1
        return nxt

    def is_player_defeated(self, player_id: int) -> bool:
        return self.tanks[player_id].hp <= 0 or self.bases[player_id].hp <= 0

    def is_terminal(self) -> bool:
        return (
            self.is_player_defeated(0)
            or self.is_player_defeated(1)
            or self.cycle >= self.max_cycles
        )

    def winner(self) -> Optional[int]:
        p0_dead = self.is_player_defeated(0)
        p1_dead = self.is_player_defeated(1)
        if p0_dead and p1_dead:
            return None
        if p1_dead:
            return 0
        if p0_dead:
            return 1
        return None

    def utility(self, maximizing_player: int) -> float:
        my_dead = self.is_player_defeated(maximizing_player)
        opp_dead = self.is_player_defeated(1 - maximizing_player)
        if my_dead and opp_dead:
            return 0.0
        remaining = float(max(0, self.max_cycles - self.cycle))
        if opp_dead:
            return self.WIN_UTILITY + remaining
        if my_dead:
            return -self.WIN_UTILITY - remaining
        return 0.0
