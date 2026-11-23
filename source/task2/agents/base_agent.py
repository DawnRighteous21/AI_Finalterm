from abc import ABC, abstractmethod
from ..engine.actions import Action
from ..engine.game_state import GameState
from .search_metrics import SearchMetrics


class TankAgent(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.last_metrics = SearchMetrics()

    @abstractmethod
    def choose_action(self, state: GameState, player_id: int) -> Action:
        raise NotImplementedError
