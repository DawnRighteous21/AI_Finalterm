from dataclasses import dataclass
from .actions import Direction


@dataclass(slots=True)
class Tank:
    player_id: int
    x: int
    y: int
    facing: Direction
    hp: int = 6

    def clone(self) -> "Tank":
        return Tank(self.player_id, self.x, self.y, self.facing, self.hp)


@dataclass(slots=True)
class Base:
    player_id: int
    x: int
    y: int
    hp: int = 9

    def clone(self) -> "Base":
        return Base(self.player_id, self.x, self.y, self.hp)


@dataclass(slots=True)
class Bullet:
    x: int
    y: int
    direction: Direction
    owner: int

    def clone(self) -> "Bullet":
        return Bullet(self.x, self.y, self.direction, self.owner)
