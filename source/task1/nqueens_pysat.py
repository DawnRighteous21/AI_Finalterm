import time
from pysat.solvers import Glucose3


class NQueensEncoder:
    """Encodes the N-Queens problem as a CNF clause set (DIMACS variables)."""

    def __init__(self, n: int):
        if n < 4:
            raise ValueError("N must be >= 4 (N = 2, 3 have no solution; N = 1 is trivial).")
        self.n = n

    def var_id(self, i: int, j: int) -> int:
        return i * self.n + j + 1

    def build_cnf(self) -> list[list[int]]:
        """C1: one queen per row. C2: one binary clause per attacking pair.

        For each square, only the squares "after" it are visited, along its row,
        column and the two downward diagonals. Each attacking pair is generated
        exactly once, in O(N^3) time overall instead of testing all O(N^4) pairs.
        """
        n = self.n
        clauses = [[self.var_id(i, j) for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                a = self.var_id(i, j)
                for l in range(j + 1, n):                    # same row
                    clauses.append([-a, -self.var_id(i, l)])
                for k in range(i + 1, n):                    # same column
                    clauses.append([-a, -self.var_id(k, j)])
                for d in range(1, n - i):                    # two downward diagonals
                    if j + d < n:
                        clauses.append([-a, -self.var_id(i + d, j + d)])
                    if j - d >= 0:
                        clauses.append([-a, -self.var_id(i + d, j - d)])
        return clauses


class Glucose3Solver:
    """Solves the CNF produced by an NQueensEncoder with Glucose3."""

    def __init__(self, encoder: NQueensEncoder):
        self.encoder = encoder
        self.n = encoder.n
        self.clauses = encoder.build_cnf()

    def decode(self, model: list[int]) -> list[tuple[int, int]]:
        return [((v - 1) // self.n, (v - 1) % self.n) for v in model if v > 0]

    def is_valid(self, queens: list[tuple[int, int]]) -> bool:
        if len(queens) != self.n:
            return False
        for a in range(len(queens)):
            i, j = queens[a]
            for b in range(a + 1, len(queens)):
                k, l = queens[b]
                if i == k or j == l or abs(i - k) == abs(j - l):
                    return False
        return True

    def solve(self) -> tuple[list[tuple[int, int]], float]:
        """Returns (queen positions, solve() time in ms). Empty list if UNSAT."""
        solver = Glucose3()
        solver.append_formula(self.clauses)
        t0 = time.perf_counter()
        sat = solver.solve()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        queens = self.decode(solver.get_model()) if sat else []
        solver.delete()
        return queens, elapsed_ms

    def print_board(self, queens: list[tuple[int, int]]) -> None:
        placed = set(queens)
        for i in range(self.n):
            print(" ".join("Q" if (i, j) in placed else "." for j in range(self.n)))


if __name__ == "__main__":
    sizes = [4, 8, 10, 12, 16, 20, 24, 30]
    repeats = 10

    print(f"{'N':>4} | {'Vars':>6} | {'CNF Clauses':>11} | {'Avg Time (ms)':>13} | {'Valid':>5}")
    print("-" * 52)
    first_solutions = {}
    for n in sizes:
        solver = Glucose3Solver(NQueensEncoder(n))
        times = []
        for _ in range(repeats):
            queens, ms = solver.solve()
            assert solver.is_valid(queens), f"Invalid solution at N={n}"
            times.append(ms)
        first_solutions[n] = queens
        avg_ms = sum(times) / repeats
        print(f"{n:>4} | {n * n:>6} | {len(solver.clauses):>11} | {avg_ms:>13.2f} | {'True':>5}")

    for n in (4, 8):
        print(f"\nSolution for N = {n}: {sorted(first_solutions[n])}")
        Glucose3Solver(NQueensEncoder(n)).print_board(first_solutions[n])
