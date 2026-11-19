from itertools import combinations


class Run:
    def __init__(self, run_id, direction, clue_pos, target, cells):
        self.run_id = run_id
        self.direction = direction
        self.clue_pos = clue_pos
        self.target = target
        self.cells = tuple(cells)

    @property
    def length(self):
        return len(self.cells)

    def __repr__(self):
        return f"Run({self.run_id}, target={self.target}, cells={self.cells})"


class KakuroBoard:
    _combo_cache = {}

    def __init__(self, grid):
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0]) if grid else 0
        self.variables = []
        self.runs = []
        self.cell_runs = {}
        self.neighbors = {}
        self.arcs = []
        self._parse_grid()

    @classmethod
    def get_combinations(cls, length, target):
        key = (length, target)
        if key not in cls._combo_cache:
            cls._combo_cache[key] = tuple(
                combo
                for combo in combinations(range(1, 10), length)
                if sum(combo) == target
            )
        return cls._combo_cache[key]

    @classmethod
    def get_candidate_digits(cls, length, target):
        digits = set()
        for combo in cls.get_combinations(length, target):
            digits.update(combo)
        return digits

    def _parse_grid(self):
        for r in range(self.rows):
            for c in range(self.cols):
                if self.grid[r][c] == 0:
                    pos = (r, c)
                    self.variables.append(pos)
                    self.cell_runs[pos] = {}
                    self.neighbors[pos] = set()

        h_count = 0
        v_count = 0
        for r in range(self.rows):
            for c in range(self.cols):
                cell = self.grid[r][c]
                if not isinstance(cell, tuple):
                    continue
                down_clue, right_clue = cell

                if right_clue is not None:
                    cells = []
                    col = c + 1
                    while col < self.cols and self.grid[r][col] == 0:
                        cells.append((r, col))
                        col += 1
                    if cells:
                        h_count += 1
                        run = Run(f"H{h_count}", "H", (r, c), right_clue, cells)
                        self._register_run(run)

                if down_clue is not None:
                    cells = []
                    row = r + 1
                    while row < self.rows and self.grid[row][c] == 0:
                        cells.append((row, c))
                        row += 1
                    if cells:
                        v_count += 1
                        run = Run(f"V{v_count}", "V", (r, c), down_clue, cells)
                        self._register_run(run)

        for var, nbrs in self.neighbors.items():
            for nbr in sorted(nbrs):
                self.arcs.append((var, nbr))

    def _register_run(self, run):
        self.runs.append(run)
        for pos in run.cells:
            self.cell_runs[pos][run.direction] = run
        for a, b in combinations(run.cells, 2):
            self.neighbors[a].add(b)
            self.neighbors[b].add(a)

    def initial_domains(self, preprocess=True):
        if not preprocess:
            return {var: set(range(1, 10)) for var in self.variables}

        domains = {}
        for var in self.variables:
            h_run = self.cell_runs[var]["H"]
            v_run = self.cell_runs[var]["V"]
            h_digits = self.get_candidate_digits(h_run.length, h_run.target)
            v_digits = self.get_candidate_digits(v_run.length, v_run.target)
            domains[var] = h_digits & v_digits
        return domains

    def is_valid_solution(self, assignment):
        if len(assignment) != len(self.variables):
            return False
        for var in self.variables:
            val = assignment.get(var)
            if not isinstance(val, int) or not (1 <= val <= 9):
                return False
        for run in self.runs:
            values = [assignment[pos] for pos in run.cells]
            if len(set(values)) != run.length or sum(values) != run.target:
                return False
        return True

    def format_solution(self, assignment):
        lines = []
        for r in range(self.rows):
            row_cells = []
            for c in range(self.cols):
                cell = self.grid[r][c]
                if cell == 0:
                    val = assignment.get((r, c), ".")
                    row_cells.append(f"{str(val):^5}")
                elif isinstance(cell, tuple):
                    d = "-" if cell[0] is None else str(cell[0])
                    rt = "-" if cell[1] is None else str(cell[1])
                    row_cells.append(f"{d}\\{rt}".center(5))
                else:
                    row_cells.append("  #  ")
            lines.append(" ".join(row_cells))
        return "\n".join(lines)


GRID_5X5 = [
    [None, None, (21, None), (14, None), None],
    [None, (14, 12), 0, 0, (11, None)],
    [(None, 21), 0, 0, 0, 0],
    [(None, 11), 0, 0, 0, 0],
    [None, (None, 16), 0, 0, None],
]

GRID_9X9 = [
    [None, None, (44, None), (16, None), (9, None), None, (15, None), (44, None), None],
    [None, (None, 22), 0, 0, 0, (11, 3), 0, 0, (14, None)],
    [None, (17, 28), 0, 0, 0, 0, 0, 0, 0],
    [(None, 14), 0, 0, None, (12, 20), 0, 0, 0, 0],
    [(None, 17), 0, 0, (29, 22), 0, 0, 0, 0, (7, None)],
    [None, (12, 28), 0, 0, 0, 0, (None, 12), 0, 0],
    [(None, 14), 0, 0, 0, 0, (7, None), (12, 6), 0, 0],
    [(None, 39), 0, 0, 0, 0, 0, 0, 0, None],
    [None, (None, 16), 0, 0, (None, 8), 0, 0, 0, None],
]


def create_5x5_board():
    return KakuroBoard(GRID_5X5)


def create_9x9_board():
    return KakuroBoard(GRID_9X9)


def main():
    for name, board in [("5x5", create_5x5_board()), ("9x9", create_9x9_board())]:
        raw_domains = board.initial_domains(preprocess=False)
        pre_domains = board.initial_domains(preprocess=True)
        raw_total = sum(len(d) for d in raw_domains.values())
        pre_total = sum(len(d) for d in pre_domains.values())
        singletons = {
            (r + 1, c + 1): sorted(d)
            for (r, c), d in pre_domains.items()
            if len(d) == 1
        }
        print(
            f"{name}: vars={len(board.variables)}, runs={len(board.runs)}, "
            f"binary_constraints={len(board.arcs) // 2}, "
            f"domain_sum={raw_total}->{pre_total}, singletons(1-based)={singletons}"
        )


if __name__ == "__main__":
    main()
