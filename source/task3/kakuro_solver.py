import time
from collections import deque

from kakuro_board import create_5x5_board, create_9x9_board


class KakuroSolver:
    def __init__(
        self,
        board,
        use_preprocess=True,
        inference="mac",
        use_heuristics=True,
        timeout=5.0,
    ):
        self.board = board
        self.use_preprocess = use_preprocess
        self.inference = inference
        self.use_heuristics = use_heuristics
        self.timeout = timeout

        self.nodes_explored = 0
        self.backtracks = 0
        self.propagations = 0
        self.elapsed_ms = 0.0
        self.timed_out = False
        self.trace = []

    def _reset_stats(self):
        self.nodes_explored = 0
        self.backtracks = 0
        self.propagations = 0
        self.elapsed_ms = 0.0
        self.timed_out = False
        self.trace = []

    def _record(self, action, cell, value, assignment, domains, detail):
        self.trace.append(
            {
                "action": action,
                "cell": cell,
                "value": value,
                "assignment": dict(assignment),
                "domains": {pos: sorted(vals) for pos, vals in domains.items()},
                "detail": detail,
            }
        )

    def solve(self, record_trace=False):
        self._reset_stats()
        start = time.perf_counter()

        raw_domains = self.board.initial_domains(preprocess=False)
        if record_trace:
            self._record("init", None, None, {}, raw_domains, "Initial domains {1..9}")

        domains = self.board.initial_domains(preprocess=self.use_preprocess)
        if record_trace and self.use_preprocess:
            self._record(
                "preprocess",
                None,
                None,
                {},
                domains,
                "Narrowed domains by Dig(H) & Dig(V)",
            )

        if any(len(d) == 0 for d in domains.values()):
            self.elapsed_ms = (time.perf_counter() - start) * 1000.0
            return None

        solution = self._backtrack({}, domains, start, record_trace)
        self.elapsed_ms = (time.perf_counter() - start) * 1000.0

        if solution is not None and record_trace:
            self._record("solved", None, None, solution, domains, "Solution found")

        return solution

    def _backtrack(self, assignment, domains, start_time, record_trace):
        if (
            self.timeout is not None
            and (time.perf_counter() - start_time) > self.timeout
        ):
            self.timed_out = True
            return None

        if len(assignment) == len(self.board.variables):
            return dict(assignment)

        var = self._select_variable(assignment, domains)
        values = self._order_values(var, assignment, domains)

        for val in values:
            if not self._is_locally_consistent(var, val, assignment):
                continue

            self.nodes_explored += 1
            assignment[var] = val
            saved_domains = {pos: set(vals) for pos, vals in domains.items()}
            domains[var] = {val}

            if record_trace:
                r, c = var
                self._record(
                    "assign",
                    var,
                    val,
                    assignment,
                    domains,
                    f"Assign ({r + 1},{c + 1}) = {val}",
                )

            ok = True
            if self.inference == "mac":
                ok = self._run_mac(var, assignment, domains, record_trace)
            elif self.inference == "fc":
                ok = self._run_forward_checking(
                    var, val, assignment, domains, record_trace
                )

            if ok:
                result = self._backtrack(assignment, domains, start_time, record_trace)
                if result is not None:
                    return result
                if self.timed_out:
                    del assignment[var]
                    domains.clear()
                    domains.update(saved_domains)
                    return None

            del assignment[var]
            domains.clear()
            domains.update(saved_domains)
            self.backtracks += 1

            if record_trace:
                r, c = var
                self._record(
                    "backtrack",
                    var,
                    val,
                    assignment,
                    domains,
                    f"Backtrack ({r + 1},{c + 1}) != {val}",
                )

        return None

    def _select_variable(self, assignment, domains):
        unassigned = [v for v in self.board.variables if v not in assignment]
        if not self.use_heuristics:
            return unassigned[0]

        def mrv_degree_key(var):
            dom_size = len(domains[var])
            degree = sum(
                1 for nbr in self.board.neighbors[var] if nbr not in assignment
            )
            return (dom_size, -degree, var)

        return min(unassigned, key=mrv_degree_key)

    def _order_values(self, var, assignment, domains):
        values = sorted(domains[var])
        if not self.use_heuristics or len(values) <= 1:
            return values

        unassigned_nbrs = [
            nbr for nbr in self.board.neighbors[var] if nbr not in assignment
        ]

        def lcv_cost(val):
            return sum(1 for nbr in unassigned_nbrs if val in domains[nbr])

        return sorted(values, key=lambda val: (lcv_cost(val), val))

    def _is_locally_consistent(self, var, val, assignment):
        for run in self.board.cell_runs[var].values():
            assigned_sum = val
            used = {val}
            unassigned_count = 0

            for cell in run.cells:
                if cell == var:
                    continue
                if cell in assignment:
                    v = assignment[cell]
                    if v in used:
                        return False
                    used.add(v)
                    assigned_sum += v
                else:
                    unassigned_count += 1

            rem = run.target - assigned_sum
            if unassigned_count == 0:
                if rem != 0:
                    return False
            else:
                min_rem = unassigned_count * (unassigned_count + 1) // 2
                max_rem = unassigned_count * (19 - unassigned_count) // 2
                if rem < min_rem or rem > max_rem:
                    return False
                combos = self.board.get_combinations(unassigned_count, rem)
                if not any(used.isdisjoint(combo) for combo in combos):
                    return False
        return True

    def _run_forward_checking(self, var, val, assignment, domains, record_trace):
        for nbr in self.board.neighbors[var]:
            if nbr in assignment:
                continue
            if val in domains[nbr]:
                domains[nbr].remove(val)
                self.propagations += 1
                if not domains[nbr]:
                    return False
                if record_trace:
                    r, c = nbr
                    self._record(
                        "propagate",
                        nbr,
                        val,
                        assignment,
                        domains,
                        f"FC removed {val} from ({r + 1},{c + 1})",
                    )

        for run in self.board.cell_runs[var].values():
            ok, changed = self._revise_run(run, assignment, domains)
            if not ok:
                return False
            if changed:
                self.propagations += len(changed)
                if record_trace:
                    for cell, pruned in changed:
                        r, c = cell
                        self._record(
                            "propagate",
                            cell,
                            None,
                            assignment,
                            domains,
                            f"FC {run.run_id} pruned {pruned} from ({r + 1},{c + 1})",
                        )
        return True

    def _run_mac(self, var, assignment, domains, record_trace):
        arc_queue = deque()
        arc_set = set()
        run_queue = deque()
        run_set = set()

        for nbr in self.board.neighbors[var]:
            if nbr not in assignment:
                arc = (nbr, var)
                arc_queue.append(arc)
                arc_set.add(arc)

        for run in self.board.cell_runs[var].values():
            run_queue.append(run)
            run_set.add(run)

        while arc_queue or run_queue:
            while arc_queue:
                xi, xj = arc_queue.popleft()
                arc_set.discard((xi, xj))

                if xi in assignment:
                    continue

                removed = self._revise_arc(xi, xj, domains)
                if removed:
                    self.propagations += 1
                    if not domains[xi]:
                        return False
                    if record_trace:
                        r, c = xi
                        self._record(
                            "propagate",
                            xi,
                            None,
                            assignment,
                            domains,
                            f"AC-3 pruned {removed} from ({r + 1},{c + 1})",
                        )
                    self._enqueue_dependents(
                        xi, xj, assignment, arc_queue, arc_set, run_queue, run_set
                    )

            if run_queue:
                run = run_queue.popleft()
                run_set.discard(run)

                ok, changed = self._revise_run(run, assignment, domains)
                if not ok:
                    return False

                for cell, pruned in changed:
                    self.propagations += 1
                    if record_trace:
                        r, c = cell
                        self._record(
                            "propagate",
                            cell,
                            None,
                            assignment,
                            domains,
                            f"REVISE-RUN {run.run_id} pruned {pruned} from ({r + 1},{c + 1})",
                        )
                    self._enqueue_dependents(
                        cell, None, assignment, arc_queue, arc_set, run_queue, run_set
                    )

        return True

    def _enqueue_dependents(
        self, cell, exclude_nbr, assignment, arc_queue, arc_set, run_queue, run_set
    ):
        for nbr in self.board.neighbors[cell]:
            if nbr != exclude_nbr and nbr not in assignment:
                arc = (nbr, cell)
                if arc not in arc_set:
                    arc_queue.append(arc)
                    arc_set.add(arc)

        for run in self.board.cell_runs[cell].values():
            if run not in run_set:
                run_queue.append(run)
                run_set.add(run)

    @staticmethod
    def _revise_arc(xi, xj, domains):
        if len(domains[xj]) != 1:
            return []
        only_val = next(iter(domains[xj]))
        if only_val in domains[xi]:
            domains[xi].remove(only_val)
            return [only_val]
        return []

    def _revise_run(self, run, assignment, domains):
        assigned_vals = []
        unassigned = []
        for cell in run.cells:
            if cell in assignment:
                assigned_vals.append(assignment[cell])
            else:
                unassigned.append(cell)

        used = set(assigned_vals)
        if len(used) != len(assigned_vals):
            return False, []

        rem = run.target - sum(assigned_vals)
        m = len(unassigned)

        if m == 0:
            return (rem == 0), []

        if m == 1:
            cell = unassigned[0]
            if rem in domains[cell] and rem not in used:
                if len(domains[cell]) > 1:
                    pruned = sorted(domains[cell] - {rem})
                    domains[cell] = {rem}
                    return True, [(cell, pruned)]
                return True, []
            return False, []

        min_rem = m * (m + 1) // 2
        max_rem = m * (19 - m) // 2
        if rem < min_rem or rem > max_rem:
            return False, []

        cell_domains = [domains[c] for c in unassigned]
        union_dom = set().union(*cell_domains)
        combos = self.board.get_combinations(m, rem)

        supported = [set() for _ in range(m)]
        any_valid = False

        for combo in combos:
            if not used.isdisjoint(combo):
                continue
            combo_set = set(combo)
            if not combo_set.issubset(union_dom):
                continue

            base_match = self._find_matching(cell_domains, combo)
            if base_match is None:
                continue

            any_valid = True
            for digit, cell_idx in base_match.items():
                supported[cell_idx].add(digit)

            for i in range(m):
                candidates = (cell_domains[i] & combo_set) - supported[i]
                for val in candidates:
                    witness = self._find_matching(cell_domains, combo, i, val)
                    if witness is not None:
                        supported[i].add(val)
                        for digit, cell_idx in witness.items():
                            supported[cell_idx].add(digit)

        if not any_valid:
            return False, []

        changed = []
        for i, cell in enumerate(unassigned):
            new_dom = domains[cell] & supported[i]
            if not new_dom:
                return False, []
            if len(new_dom) < len(domains[cell]):
                pruned = sorted(domains[cell] - new_dom)
                domains[cell] = new_dom
                changed.append((cell, pruned))

        return True, changed

    @staticmethod
    def _find_matching(cell_domains, digits, fixed_cell=-1, fixed_digit=-1):
        match_digit = {}

        def augment(u, visited):
            for d in digits:
                if d == fixed_digit or d in visited or d not in cell_domains[u]:
                    continue
                visited.add(d)
                if d not in match_digit or augment(match_digit[d], visited):
                    match_digit[d] = u
                    return True
            return False

        for u in range(len(cell_domains)):
            if u == fixed_cell:
                continue
            if not augment(u, set()):
                return None

        return match_digit


def run_benchmarks(name, board):
    configs = [
        ("Plain BT (Fixed)", False, "none", False),
        ("Preproc + FC (Fixed)", True, "fc", False),
        ("Preproc + FC + Heuristics", True, "fc", True),
        ("Preproc + MAC (Fixed)", True, "mac", False),
        ("Full (Preproc + MAC + Heuristics)", True, "mac", True),
    ]

    print(f"\n=== Benchmark: {name} ===")
    print(
        f"{'Configuration':<35} | {'Status':<7} | {'Nodes':>7} | {'Backtracks':>10} | {'Time (ms)':>9}"
    )
    print("-" * 80)

    full_solution = None
    for label, pre, inf, heur in configs:
        solver = KakuroSolver(
            board, use_preprocess=pre, inference=inf, use_heuristics=heur, timeout=5.0
        )
        sol = solver.solve(record_trace=False)
        valid = sol is not None and board.is_valid_solution(sol)
        status = "TIMEOUT" if solver.timed_out else ("VALID" if valid else "FAIL")
        print(
            f"{label:<35} | {status:<7} | {solver.nodes_explored:>7} | "
            f"{solver.backtracks:>10} | {solver.elapsed_ms:>9.3f}"
        )
        if label.startswith("Full"):
            full_solution = sol

    return full_solution


def run_stability_test(name, board, runs=20):
    times = []
    solutions = []
    for _ in range(runs):
        solver = KakuroSolver(
            board, use_preprocess=True, inference="mac", use_heuristics=True
        )
        sol = solver.solve(record_trace=False)
        assert sol is not None and board.is_valid_solution(sol)
        times.append(solver.elapsed_ms)
        solutions.append(tuple(sorted(sol.items())))

    consistent = len(set(solutions)) == 1
    avg_ms = sum(times) / len(times)
    print(
        f"Stability ({name}, {runs} runs): identical_solution={consistent}, "
        f"avg={avg_ms:.3f} ms, min={min(times):.3f} ms, max={max(times):.3f} ms"
    )


def main():
    for name, board in [
        ("5x5 Example", create_5x5_board()),
        ("9x9 Complex", create_9x9_board()),
    ]:
        solution = run_benchmarks(name, board)
        run_stability_test(name, board, runs=20)
        if solution is not None:
            print(f"\nSolved {name} Board:")
            print(board.format_solution(solution))

        trace_solver = KakuroSolver(
            board, use_preprocess=True, inference="mac", use_heuristics=True
        )
        trace_solver.solve(record_trace=True)
        print(f"UI Trace steps recorded ({name}): {len(trace_solver.trace)}")


if __name__ == "__main__":
    main()
