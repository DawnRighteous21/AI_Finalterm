from itertools import combinations


class NQueensCNFEncoder:
    def __init__(self, n):
        if n < 1:
            raise ValueError("Board size must be at least 1")
        self.n = n
        self.clauses = []

    def var(self, row, col):
        """Return the integer id of the variable of square (row, col)."""
        return row * self.n + col + 1

    def square_of(self, var_id):
        """Inverse of var(): return (row, col) of a variable id."""
        return divmod(var_id - 1, self.n)

    def num_vars(self):
        return self.n * self.n

    @staticmethod
    def is_attacking(a, b):
        """True if two distinct squares share a row, column or diagonal."""
        (r1, c1), (r2, c2) = a, b
        return r1 == r2 or c1 == c2 or abs(r1 - r2) == abs(c1 - c2)

    def squares(self):
        return [(i, j) for i in range(self.n) for j in range(self.n)]

    def row_clauses(self):
        """C1: at least one queen in every row."""
        return [[self.var(i, j) for j in range(self.n)] for i in range(self.n)]

    def attack_clauses(self):
        """C2: for every attacking pair, not (x_a and x_b) = (-x_a or -x_b)."""
        result = []
        for a, b in combinations(self.squares(), 2):
            if self.is_attacking(a, b):
                result.append([-self.var(*a), -self.var(*b)])
        return result

    def encode(self):
        """Generate and store all clauses, then return them."""
        self.clauses = self.row_clauses() + self.attack_clauses()
        return self.clauses

    def num_clauses(self):
        return len(self.clauses)

    def decode(self, model):
        """Convert a SAT model (list of signed ints) to queen positions."""
        queens = []
        for lit in model:
            if lit > 0:
                queens.append(self.square_of(lit))
        return sorted(queens)

    def is_valid_solution(self, queens):
        """Check N queens and no attacking pair."""
        if len(queens) != self.n:
            return False
        return not any(self.is_attacking(a, b) for a, b in combinations(queens, 2))

    @staticmethod
    def expected_clause_count(n):
        """Closed form: N + 2N*C(N,2) + 4*C(N,3) + 2*C(N,2)."""
        c2 = n * (n - 1) // 2
        c3 = n * (n - 1) * (n - 2) // 6
        return n + 2 * n * c2 + 4 * c3 + 2 * c2


def main():
    for n in (4, 8):
        encoder = NQueensCNFEncoder(n)
        encoder.encode()
        print(
            "N =",
            n,
            "| variables =",
            encoder.num_vars(),
            "| clauses =",
            encoder.num_clauses(),
            "| formula =",
            NQueensCNFEncoder.expected_clause_count(n),
        )

    try:
        from pysat.solvers import Glucose3
    except ImportError:
        print("python-sat is not installed, skipping the solver check")
        return

    encoder = NQueensCNFEncoder(8)
    encoder.encode()
    with Glucose3(bootstrap_with=encoder.clauses) as solver:
        if solver.solve():
            queens = encoder.decode(solver.get_model())
            print("Queens:", queens)
            print("Valid:", encoder.is_valid_solution(queens))
        else:
            print("UNSAT")


if __name__ == "__main__":
    main()
