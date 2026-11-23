from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GameMap:
    width: int
    height: int
    walls: tuple[tuple[int, int], ...]
    tank_spawns: tuple[tuple[int, int], tuple[int, int]]
    base_positions: tuple[tuple[int, int], tuple[int, int]]

    @classmethod
    def from_file(cls, path: str | Path) -> "GameMap":
        lines = [
            line.strip()
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return cls.from_lines(lines)

    @classmethod
    def from_lines(cls, lines: list[str]) -> "GameMap":
        if not lines:
            raise ValueError("Map cannot be empty.")
        height = len(lines)
        width = len(lines[0])
        walls: list[tuple[int, int]] = []
        tanks: dict[int, tuple[int, int]] = {}
        bases: dict[int, tuple[int, int]] = {}
        for y, row in enumerate(lines):
            if len(row) != width:
                raise ValueError(f"Inconsistent row length at row {y}.")
            for x, ch in enumerate(row):
                if ch == "#":
                    walls.append((x, y))
                elif ch == "1":
                    tanks[0] = (x, y)
                elif ch == "2":
                    tanks[1] = (x, y)
                elif ch == "A":
                    bases[0] = (x, y)
                elif ch == "B":
                    bases[1] = (x, y)
                elif ch != ".":
                    raise ValueError(f"Unknown map symbol '{ch}' at ({x}, {y}).")
        if 0 not in tanks or 1 not in tanks:
            raise ValueError("Map must define both tank spawns '1' and '2'.")
        if 0 not in bases or 1 not in bases:
            raise ValueError("Map must define both bases 'A' and 'B'.")
        return cls(
            width=width,
            height=height,
            walls=tuple(walls),
            tank_spawns=(tanks[0], tanks[1]),
            base_positions=(bases[0], bases[1]),
        )
