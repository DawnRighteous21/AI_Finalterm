from dataclasses import dataclass
import math
import time
from typing import Optional
from ..engine.actions import Action
from ..engine.game_state import GameState
from .base_agent import TankAgent
from .heuristic import HeuristicEvaluator
from .search_metrics import SearchMetrics


class _SearchTimeout(Exception):
    pass


@dataclass(slots=True)
class _TTEntry:
    depth: int
    value: float
    flag: str
    best_action: Optional[Action]


class HeuristicAlphaBetaAgent(TankAgent):
    EXACT = "EXACT"
    LOWER = "LOWER"
    UPPER = "UPPER"

    def __init__(
        self,
        max_depth: int = 3,
        time_limit_s: Optional[float] = None,
        use_transposition_table: bool = True,
        evaluator: Optional[HeuristicEvaluator] = None,
        max_tt_size: int = 100_000,
    ) -> None:
        mode = (
            f"ID-{int(time_limit_s * 1000)}ms"
            if time_limit_s is not None
            else f"d={max_depth}"
        )
        super().__init__(name=f"HeuristicAlphaBeta({mode})")
        self.max_depth = max_depth
        self.time_limit_s = time_limit_s
        self.use_transposition_table = use_transposition_table
        self.evaluator = evaluator or HeuristicEvaluator()
        self._max_tt_size = max_tt_size
        self._tt: dict[tuple, _TTEntry] = {}

    def clear_transposition_table(self) -> None:
        self._tt.clear()

    def choose_action(self, state: GameState, player_id: int) -> Action:
        start = time.perf_counter()
        deadline = (
            start + self.time_limit_s if self.time_limit_s is not None else None
        )
        metrics = SearchMetrics()
        legal = state.legal_actions(player_id)
        best_action = legal[0]
        best_value = -math.inf

        if self.use_transposition_table and len(self._tt) > self._max_tt_size:
            self._tt.clear()

        start_depth = 1 if self.time_limit_s is not None else self.max_depth
        for depth in range(start_depth, self.max_depth + 1):
            try:
                action, value = self._search_root(
                    state,
                    depth=depth,
                    player_id=player_id,
                    preferred_action=best_action,
                    deadline=deadline,
                    metrics=metrics,
                )
                best_action = action
                best_value = value
                metrics.depth_reached = depth
                if abs(best_value) >= GameState.WIN_UTILITY:
                    break
            except _SearchTimeout:
                break

        metrics.elapsed_ms = (time.perf_counter() - start) * 1000.0
        metrics.best_value = best_value
        self.last_metrics = metrics
        return best_action

    def _check_deadline(self, deadline: Optional[float], nodes: int) -> None:
        if deadline is not None and (nodes & 255) == 0:
            if time.perf_counter() >= deadline:
                raise _SearchTimeout()

    def _search_root(
        self,
        state: GameState,
        depth: int,
        player_id: int,
        preferred_action: Optional[Action],
        deadline: Optional[float],
        metrics: SearchMetrics,
    ) -> tuple[Action, float]:
        actions = self.evaluator.order_actions(
            state,
            player_id,
            state.legal_actions(player_id),
            preferred_action=preferred_action,
        )
        best_action = actions[0]
        alpha = -math.inf
        beta = math.inf

        for action in actions:
            value = self._min_value(
                state,
                depth=depth,
                alpha=alpha,
                beta=beta,
                maximizing_player=player_id,
                max_action=action,
                deadline=deadline,
                metrics=metrics,
            )
            if value > alpha:
                alpha = value
                best_action = action
        return best_action, alpha

    def _max_value(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        maximizing_player: int,
        deadline: Optional[float],
        metrics: SearchMetrics,
    ) -> float:
        metrics.nodes_explored += 1
        self._check_deadline(deadline, metrics.nodes_explored)
        if state.is_terminal() or depth == 0:
            return self.evaluator.evaluate(state, maximizing_player)

        alpha_orig = alpha
        preferred_action: Optional[Action] = None
        tt_key = None
        if self.use_transposition_table:
            tt_key = (state.state_key(), state.cycle, maximizing_player)
            entry = self._tt.get(tt_key)
            if entry is not None:
                preferred_action = entry.best_action
                if entry.depth >= depth:
                    metrics.tt_hits += 1
                    if entry.flag == self.EXACT:
                        return entry.value
                    if entry.flag == self.LOWER and entry.value > alpha:
                        alpha = entry.value
                    elif entry.flag == self.UPPER and entry.value < beta:
                        beta = entry.value
                    if alpha >= beta:
                        metrics.cutoffs += 1
                        return entry.value

        best = -math.inf
        best_action: Optional[Action] = None
        actions = self.evaluator.order_actions(
            state,
            maximizing_player,
            state.legal_actions(maximizing_player),
            preferred_action=preferred_action,
        )
        for action in actions:
            val = self._min_value(
                state,
                depth,
                alpha,
                beta,
                maximizing_player,
                action,
                deadline,
                metrics,
            )
            if val > best:
                best = val
                best_action = action
            if best >= beta:
                metrics.cutoffs += 1
                break
            if best > alpha:
                alpha = best

        if self.use_transposition_table and tt_key is not None:
            if best <= alpha_orig:
                flag = self.UPPER
            elif best >= beta:
                flag = self.LOWER
            else:
                flag = self.EXACT
            self._tt[tt_key] = _TTEntry(depth, best, flag, best_action)
        return best

    def _min_value(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        maximizing_player: int,
        max_action: Action,
        deadline: Optional[float],
        metrics: SearchMetrics,
    ) -> float:
        metrics.nodes_explored += 1
        self._check_deadline(deadline, metrics.nodes_explored)
        opp = 1 - maximizing_player
        best = math.inf
        actions = self.evaluator.order_actions(
            state, opp, state.legal_actions(opp)
        )
        for opp_action in actions:
            next_state = (
                state.step_cycle(max_action, opp_action)
                if maximizing_player == 0
                else state.step_cycle(opp_action, max_action)
            )
            val = self._max_value(
                next_state,
                depth - 1,
                alpha,
                beta,
                maximizing_player,
                deadline,
                metrics,
            )
            if val < best:
                best = val
            if best <= alpha:
                metrics.cutoffs += 1
                return best
            if best < beta:
                beta = best
        return best
