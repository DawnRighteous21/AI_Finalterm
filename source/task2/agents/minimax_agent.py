import math
import time
from ..engine.actions import Action
from ..engine.game_state import GameState
from .base_agent import TankAgent
from .search_metrics import SearchMetrics


class MinimaxAgent(TankAgent):
    def __init__(self, max_depth: int = 2) -> None:
        super().__init__(name=f"Minimax(d={max_depth})")
        self.max_depth = max_depth

    def choose_action(self, state: GameState, player_id: int) -> Action:
        start = time.perf_counter()
        metrics = SearchMetrics(depth_reached=self.max_depth)
        legal = state.legal_actions(player_id)
        best_action = legal[0]
        best_value = -math.inf

        for action in legal:
            value = self._min_value(
                state,
                depth=self.max_depth,
                maximizing_player=player_id,
                max_action=action,
                metrics=metrics,
            )
            if value > best_value:
                best_value = value
                best_action = action

        metrics.elapsed_ms = (time.perf_counter() - start) * 1000.0
        metrics.best_value = best_value
        self.last_metrics = metrics
        return best_action

    def _max_value(
        self,
        state: GameState,
        depth: int,
        maximizing_player: int,
        metrics: SearchMetrics,
    ) -> float:
        metrics.nodes_explored += 1
        if state.is_terminal() or depth == 0:
            return state.utility(maximizing_player)

        best = -math.inf
        for action in state.legal_actions(maximizing_player):
            val = self._min_value(state, depth, maximizing_player, action, metrics)
            if val > best:
                best = val
        return best

    def _min_value(
        self,
        state: GameState,
        depth: int,
        maximizing_player: int,
        max_action: Action,
        metrics: SearchMetrics,
    ) -> float:
        metrics.nodes_explored += 1
        opp = 1 - maximizing_player
        best = math.inf
        for opp_action in state.legal_actions(opp):
            next_state = (
                state.step_cycle(max_action, opp_action)
                if maximizing_player == 0
                else state.step_cycle(opp_action, max_action)
            )
            val = self._max_value(next_state, depth - 1, maximizing_player, metrics)
            if val < best:
                best = val
        return best
