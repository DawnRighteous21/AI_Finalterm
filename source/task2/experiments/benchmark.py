from pathlib import Path
from typing import Callable
from ..agents import (
    AlphaBetaAgent,
    HeuristicAlphaBetaAgent,
    HeuristicEvaluator,
    HeuristicMinimaxAgent,
    MinimaxAgent,
    TankAgent,
)
from ..engine import Base, Direction, GameMap, GameState, Tank

MAPS_DIR = Path(__file__).resolve().parent.parent / "maps"


def build_endgame_scenario() -> GameState:
    walls = {(0, 2): 2, (4, 2): 2}
    tanks = (
        Tank(0, 2, 4, Direction.UP, hp=2),
        Tank(1, 2, 1, Direction.DOWN, hp=1),
    )
    bases = (
        Base(0, 0, 4, hp=3),
        Base(1, 4, 0, hp=3),
    )
    return GameState(5, 5, walls, tanks, bases, bullets=[], cycle=0, max_cycles=50)


def build_skirmish_scenario() -> GameState:
    state = GameState.from_map(
        GameMap.from_file(MAPS_DIR / "skirmish_7x7.txt"),
        max_cycles=120,
    )
    state.tanks[0].x, state.tanks[0].y, state.tanks[0].facing = 2, 4, Direction.UP
    state.tanks[1].x, state.tanks[1].y, state.tanks[1].facing = 4, 2, Direction.DOWN
    return state


def build_arena_scenario() -> GameState:
    state = GameState.from_map(
        GameMap.from_file(MAPS_DIR / "battle_city_13x13.txt"),
        max_cycles=200,
    )
    state.tanks[0].x, state.tanks[0].y, state.tanks[0].facing = 6, 8, Direction.UP
    state.tanks[1].x, state.tanks[1].y, state.tanks[1].facing = 6, 4, Direction.DOWN
    return state


def run_benchmark(runs: int = 3) -> None:
    scenarios: list[tuple[str, GameState, int]] = [
        ("Endgame 5x5", build_endgame_scenario(), 2),
        ("Skirmish 7x7", build_skirmish_scenario(), 2),
        ("Arena 13x13", build_arena_scenario(), 2),
        ("Arena 13x13", build_arena_scenario(), 3),
    ]

    print(
        f"{'Scenario':<14} {'Depth':>5} {'Configuration':<28} "
        f"{'Nodes':>10} {'Cutoffs':>9} {'Time(ms)':>10} {'Action':<11} {'Value':>8}"
    )
    print("-" * 102)

    latex_rows: list[str] = []
    for scenario_name, state, depth in scenarios:
        factories: list[tuple[str, Callable[[], TankAgent]]] = [
            ("Minimax", lambda d=depth: MinimaxAgent(max_depth=d)),
            ("Alpha-Beta", lambda d=depth: AlphaBetaAgent(max_depth=d)),
            ("Heuristic Minimax", lambda d=depth: HeuristicMinimaxAgent(max_depth=d)),
            (
                "Heuristic AB (No TT)",
                lambda d=depth: HeuristicAlphaBetaAgent(
                    max_depth=d, use_transposition_table=False
                ),
            ),
            (
                "Heuristic AB + TT",
                lambda d=depth: HeuristicAlphaBetaAgent(
                    max_depth=d, use_transposition_table=True
                ),
            ),
        ]

        values: dict[str, float] = {}
        for label, factory in factories:
            total_ms = 0.0
            last_action = None
            last_metrics = None
            for _ in range(runs):
                agent = factory()
                last_action = agent.choose_action(state, player_id=0)
                last_metrics = agent.last_metrics
                total_ms += last_metrics.elapsed_ms
            avg_ms = total_ms / runs
            values[label] = last_metrics.best_value
            print(
                f"{scenario_name:<14} {depth:>5} {label:<28} "
                f"{last_metrics.nodes_explored:>10} {last_metrics.cutoffs:>9} "
                f"{avg_ms:>10.2f} {last_action.value:<11} {last_metrics.best_value:>8.1f}"
            )
            latex_rows.append(
                f"{scenario_name} & {depth} & {label} & "
                f"{last_metrics.nodes_explored} & {avg_ms:.2f} \\\\"
            )

        assert abs(values["Minimax"] - values["Alpha-Beta"]) < 1e-6
        assert abs(values["Heuristic Minimax"] - values["Heuristic AB (No TT)"]) < 1e-6
        print(f"  [Verified] Exact value equivalence passed for {scenario_name} (d={depth}).")
        print("-" * 102)

    print("\nReal-Time Iterative Deepening Test (Arena 13x13, 100 ms budget):")
    arena_state = build_arena_scenario()
    rt_agent = HeuristicAlphaBetaAgent(
        max_depth=6,
        time_limit_s=0.10,
        use_transposition_table=True,
        evaluator=HeuristicEvaluator(),
    )
    rt_action = rt_agent.choose_action(arena_state, player_id=0)
    m = rt_agent.last_metrics
    print(
        f"  Depth reached: {m.depth_reached} cycles ({m.depth_reached * 2} plies) | "
        f"Nodes: {m.nodes_explored} | TT hits: {m.tt_hits} | "
        f"Time: {m.elapsed_ms:.2f} ms | Action: {rt_action.value}"
    )

    print("\nLaTeX Table Rows:")
    for row in latex_rows:
        print(row)


def run_match_demo(max_cycles: int = 60) -> None:
    state = GameState.from_map(
        GameMap.from_file(MAPS_DIR / "battle_city_13x13.txt"),
        max_cycles=max_cycles,
    )
    p0 = HeuristicAlphaBetaAgent(max_depth=5, time_limit_s=0.10, use_transposition_table=True)
    p1 = AlphaBetaAgent(max_depth=2)
    while not state.is_terminal():
        a0 = p0.choose_action(state, player_id=0)
        a1 = p1.choose_action(state, player_id=1)
        state = state.step_cycle(a0, a1)
    print(
        f"\nDemo Match ({p0.name} vs {p1.name}) ended at cycle {state.cycle}: "
        f"Winner={state.winner()} | "
        f"P0(T={state.tanks[0].hp}, B={state.bases[0].hp}) vs "
        f"P1(T={state.tanks[1].hp}, B={state.bases[1].hp})"
    )


if __name__ == "__main__":
    run_benchmark(runs=3)
    run_match_demo()
