from dataclasses import dataclass


@dataclass
class SearchMetrics:
    nodes_explored: int = 0
    cutoffs: int = 0
    tt_hits: int = 0
    depth_reached: int = 0
    elapsed_ms: float = 0.0
    best_value: float = 0.0
