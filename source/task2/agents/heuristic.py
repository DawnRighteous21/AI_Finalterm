from collections import OrderedDict
import heapq
from typing import Optional
from ..engine.actions import ACTION_TO_DIRECTION, Action, Direction
from ..engine.game_state import GameState


class HeuristicEvaluator:
    def __init__(
        self,
        w_tank_hp: float = 45.0,
        w_base_hp: float = 35.0,
        w_base_distance: float = 6.0,
        w_chase_distance: float = 1.5,
        w_alignment: float = 14.0,
        w_bullet_threat: float = 25.0,
        max_cache_entries: int = 512,
    ) -> None:
        self.w_tank_hp = w_tank_hp
        self.w_base_hp = w_base_hp
        self.w_base_distance = w_base_distance
        self.w_chase_distance = w_chase_distance
        self.w_alignment = w_alignment
        self.w_bullet_threat = w_bullet_threat
        self._max_cache_entries = max_cache_entries
        self._dist_cache: OrderedDict[tuple, dict[tuple[int, int], int]] = OrderedDict()

    def clear_cache(self) -> None:
        self._dist_cache.clear()

    def evaluate(self, state: GameState, maximizing_player: int) -> float:
        if state.is_terminal():
            return state.utility(maximizing_player)
        opp = 1 - maximizing_player
        my_tank = state.tanks[maximizing_player]
        opp_tank = state.tanks[opp]
        my_base = state.bases[maximizing_player]
        opp_base = state.bases[opp]

        f_tank_hp = float(my_tank.hp - opp_tank.hp)
        f_base_hp = float(my_base.hp - opp_base.hp)

        my_base_dist = self._weighted_bfs_distance(
            state, (my_tank.x, my_tank.y), (opp_base.x, opp_base.y)
        )
        opp_base_dist = self._weighted_bfs_distance(
            state, (opp_tank.x, opp_tank.y), (my_base.x, my_base.y)
        )
        f_base_distance = float(opp_base_dist - my_base_dist)

        my_tank_dist = self._weighted_bfs_distance(
            state, (my_tank.x, my_tank.y), (opp_tank.x, opp_tank.y)
        )
        f_chase_distance = -float(my_tank_dist)

        f_alignment = (
            self._alignment_score(state, maximizing_player)
            - self._alignment_score(state, opp)
        )

        f_threat = (
            self._bullet_threat(state, maximizing_player)
            - self._bullet_threat(state, opp)
        )

        return (
            self.w_tank_hp * f_tank_hp
            + self.w_base_hp * f_base_hp
            + self.w_base_distance * f_base_distance
            + self.w_chase_distance * f_chase_distance
            + self.w_alignment * f_alignment
            - self.w_bullet_threat * f_threat
        )

    def order_actions(
        self,
        state: GameState,
        player_id: int,
        actions: list[Action],
        preferred_action: Optional[Action] = None,
    ) -> list[Action]:
        tank = state.tanks[player_id]
        opp = 1 - player_id
        opp_tank = state.tanks[opp]
        opp_base = state.bases[opp]
        in_danger = self._cell_in_enemy_bullet_path(state, tank.x, tank.y, player_id)

        def priority(action: Action) -> tuple[int, float]:
            if preferred_action is not None and action == preferred_action:
                return (0, 0.0)
            if action == Action.FIRE:
                if self._has_clear_shot(state, tank.x, tank.y, tank.facing, opp_tank.x, opp_tank.y):
                    return (1, 0.0)
                if self._has_clear_shot(state, tank.x, tank.y, tank.facing, opp_base.x, opp_base.y):
                    return (1, 0.5)
                if self._faces_wall_toward_target(
                    state, tank.x, tank.y, tank.facing, opp_base.x, opp_base.y
                ):
                    return (2, 0.5)
                if self._faces_adjacent_wall(state, tank.x, tank.y, tank.facing):
                    return (3, 1.0)
                return (4, 0.0)
            direction = ACTION_TO_DIRECTION.get(action)
            if direction is not None:
                nx = tank.x + direction.dx
                ny = tank.y + direction.dy
                still_danger = self._cell_in_enemy_bullet_path(state, nx, ny, player_id)
                dist = min(
                    abs(nx - opp_tank.x) + abs(ny - opp_tank.y),
                    abs(nx - opp_base.x) + abs(ny - opp_base.y),
                )
                if in_danger and not still_danger:
                    return (1, float(dist))
                if not in_danger and still_danger:
                    return (5, float(dist))
                return (2, float(dist))
            return (6 if in_danger else 5, 0.0)

        return sorted(actions, key=priority)

    def _weighted_bfs_distance(
        self,
        state: GameState,
        source: tuple[int, int],
        target: tuple[int, int],
    ) -> int:
        if source == target:
            return 0
        key = (state.width, state.height, state.walls_key(), target)
        dist_map = self._dist_cache.get(key)
        if dist_map is not None:
            self._dist_cache.move_to_end(key)
            return dist_map.get(source, state.width + state.height)

        dist_map = {target: 0}
        heap: list[tuple[int, int, int]] = [(0, target[0], target[1])]
        while heap:
            cost, cx, cy = heapq.heappop(heap)
            if cost > dist_map[(cx, cy)]:
                continue
            for direction in Direction:
                nx = cx + direction.dx
                ny = cy + direction.dy
                if not state.in_bounds(nx, ny):
                    continue
                wall_hp = state.walls.get((nx, ny), 0)
                step_cost = 1 if wall_hp == 0 else (wall_hp + 2)
                new_cost = cost + step_cost
                pos = (nx, ny)
                if new_cost < dist_map.get(pos, 1_000_000):
                    dist_map[pos] = new_cost
                    heapq.heappush(heap, (new_cost, nx, ny))

        self._dist_cache[key] = dist_map
        if len(self._dist_cache) > self._max_cache_entries:
            self._dist_cache.popitem(last=False)
        return dist_map.get(source, state.width + state.height)

    def _alignment_score(self, state: GameState, player_id: int) -> float:
        tank = state.tanks[player_id]
        opp = 1 - player_id
        opp_tank = state.tanks[opp]
        opp_base = state.bases[opp]
        score = 0.0
        for tx, ty, weight in (
            (opp_tank.x, opp_tank.y, 1.0),
            (opp_base.x, opp_base.y, 0.85),
        ):
            if self._has_clear_shot(state, tank.x, tank.y, tank.facing, tx, ty):
                score += weight
            elif self._line_unobstructed(state, tank.x, tank.y, tx, ty):
                score += 0.45 * weight
        return score

    def _bullet_threat(self, state: GameState, player_id: int) -> float:
        tank = state.tanks[player_id]
        base = state.bases[player_id]
        threat = 0.0
        for bullet in state.bullets:
            if bullet.owner == player_id:
                continue
            if self._has_clear_shot(state, bullet.x, bullet.y, bullet.direction, tank.x, tank.y):
                dist = abs(bullet.x - tank.x) + abs(bullet.y - tank.y)
                threat += 1.5 if dist <= GameState.BULLET_SPEED else 0.75
            if self._has_clear_shot(state, bullet.x, bullet.y, bullet.direction, base.x, base.y):
                dist = abs(bullet.x - base.x) + abs(bullet.y - base.y)
                threat += 1.0 if dist <= GameState.BULLET_SPEED else 0.4
        return threat

    def _cell_in_enemy_bullet_path(
        self,
        state: GameState,
        x: int,
        y: int,
        player_id: int,
    ) -> bool:
        for bullet in state.bullets:
            if bullet.owner == player_id:
                continue
            if self._has_clear_shot(state, bullet.x, bullet.y, bullet.direction, x, y):
                dist = abs(bullet.x - x) + abs(bullet.y - y)
                if dist <= GameState.BULLET_SPEED * 2:
                    return True
        return False

    def _has_clear_shot(
        self,
        state: GameState,
        sx: int,
        sy: int,
        direction: Direction,
        tx: int,
        ty: int,
    ) -> bool:
        if sx == tx and sy == ty:
            return False
        dx, dy = direction.dx, direction.dy
        if dx == 0:
            if sx != tx:
                return False
            if (ty - sy) * dy <= 0:
                return False
        else:
            if sy != ty:
                return False
            if (tx - sx) * dx <= 0:
                return False
        cx, cy = sx + dx, sy + dy
        while (cx, cy) != (tx, ty):
            if (cx, cy) in state.walls:
                return False
            cx += dx
            cy += dy
        return True

    def _line_unobstructed(
        self,
        state: GameState,
        sx: int,
        sy: int,
        tx: int,
        ty: int,
    ) -> bool:
        if sx == tx and sy == ty:
            return False
        if sx == tx:
            step = 1 if ty > sy else -1
            for y in range(sy + step, ty, step):
                if (sx, y) in state.walls:
                    return False
            return True
        if sy == ty:
            step = 1 if tx > sx else -1
            for x in range(sx + step, tx, step):
                if (x, sy) in state.walls:
                    return False
            return True
        return False

    def _faces_adjacent_wall(
        self,
        state: GameState,
        x: int,
        y: int,
        direction: Direction,
    ) -> bool:
        return (x + direction.dx, y + direction.dy) in state.walls

    def _faces_wall_toward_target(
        self,
        state: GameState,
        x: int,
        y: int,
        direction: Direction,
        tx: int,
        ty: int,
    ) -> bool:
        nx = x + direction.dx
        ny = y + direction.dy
        if (nx, ny) not in state.walls:
            return False
        return (abs(nx - tx) + abs(ny - ty)) < (abs(x - tx) + abs(y - ty))
