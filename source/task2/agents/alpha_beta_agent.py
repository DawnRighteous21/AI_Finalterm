import math
import time
from typing import Optional
from ..engine.actions import Action
from ..engine.game_state import GameState
from .base_agent import TankAgent
from .heuristic import HeuristicEvaluator
from .search_metrics import SearchMetrics


class AlphaBetaAgent(TankAgent):
    def __init__(
        self,
        max_depth: int = 2,
        move_orderer: Optional[HeuristicEvaluator] = None,
    ) -> None:
        super().__init__(name=f"AlphaBeta(d={max_depth})")
        self.max_depth = max_depth
        self.move_orderer = move_orderer or HeuristicEvaluator()

    def choose_action(self, state: GameState, player_id: int) -> Action:
        start = time.perf_counter()
        metrics = SearchMetrics(depth_reached=self.max_depth)
        legal = self.move_orderer.order_actions(
            state, player_id, state.legal_actions(player_id)
        )
        best_action = legal[0]
        alpha = -math.inf
        beta = math.inf

        for action in legal:
            value = self._min_value(
                state,
                depth=self.max_depth,
                alpha=alpha,
                beta=beta,
                maximizing_player=player_id,
                max_action=action,
                metrics=metrics,
            )
            if value > alpha:
                alpha = value
                best_action = action

        metrics.elapsed_ms = (time.perf_counter() - start) * 1000.0
        metrics.best_value = alpha
        self.last_metrics = metrics
        return best_action

    def _max_value(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        maximizing_player: int,
        metrics: SearchMetrics,
    ) -> float:
        metrics.nodes_explored += 1
        if state.is_terminal() or depth == 0:
            return state.utility(maximizing_player)

        best = -math.inf
        actions = self.move_orderer.order_actions(
            state, maximizing_player, state.legal_actions(maximizing_player)
        )
        for action in actions:
            val = self._min_value(
                state, depth, alpha, beta, maximizing_player, action, metrics
            )
            if val > best:
                best = val
            if best >= beta:
                metrics.cutoffs += 1
                return best
            if best > alpha:
                alpha = best
        return best

    def _min_value(
        self,
        state: GameState,
        depth: int,
        alpha: float,
        beta: float,
        maximizing_player: int,
        max_action: Action,
        metrics: SearchMetrics,
    ) -> float:
        metrics.nodes_explored += 1
        opp = 1 - maximizing_player
        best = math.inf
        actions = self.move_orderer.order_actions(
            state, opp, state.legal_actions(opp)
        )
        for opp_action in actions:
            next_state = (
                state.step_cycle(max_action, opp_action)
                if maximizing_player == 0
                else state.step_cycle(opp_action, max_action)
            )
            val = self._max_value(
                next_state, depth - 1, alpha, beta, maximizing_player, metrics
            )
            if val < best:
                best = val
            if best <= alpha:
                metrics.cutoffs += 1
                return best
            if best < beta:
                beta = best
        return best
