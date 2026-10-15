from collections import defaultdict
from math import exp


def preference(wins, losses, ties):
    counts = (wins, losses, ties)
    if any(x < 0 or int(x) != x for x in counts):
        raise ValueError("invalid nonnegative integer counts")
    total = sum(counts)
    if total == 0:
        raise ValueError("empty comparison")
    return (wins + ties / 2) / total


def kappa(matrix):
    size = len(matrix)
    if size < 2 or any(len(row) != size for row in matrix):
        raise ValueError("square matrix required")
    if any(x < 0 for row in matrix for x in row):
        raise ValueError("negative count")
    total = sum(map(sum, matrix))
    if total == 0:
        raise ValueError("empty matrix")
    rows = [sum(row) / total for row in matrix]
    cols = [sum(matrix[i][j] for i in range(size)) / total
            for j in range(size)]
    observed = sum(matrix[i][i] for i in range(size)) / total
    expected = sum(a * b for a, b in zip(rows, cols))
    coefficient = None if expected == 1 else (
        (observed - expected) / (1 - expected)
    )
    return {"agreement": observed, "expected": expected,
            "kappa": coefficient, "total": total}


def elo_expected(a, b):
    return 1 / (1 + 10 ** ((b - a) / 400))


def elo_update(a, b, score_a, factor=16):
    if score_a not in {0, .5, 1}:
        raise ValueError("invalid score")
    change = factor * (score_a - elo_expected(a, b))
    return a + change, b - change


def connected(players, comparisons):
    graph = defaultdict(set)
    for a, b, _ in comparisons:
        graph[a].add(b)
        graph[b].add(a)
    if not players:
        return False
    visited = {players[0]}
    pending = [players[0]]
    while pending:
        current = pending.pop()
        for neighbour in graph[current] - visited:
            visited.add(neighbour)
            pending.append(neighbour)
    return visited == set(players)


def logistic(value):
    if value >= 0:
        return 1 / (1 + exp(-value))
    numerator = exp(value)
    return numerator / (1 + numerator)


def fit_bradley_terry(comparisons, penalty=.02,
                     iterations=20000, tolerance=1e-9):
    if penalty <= 0:
        raise ValueError("positive regularization required")
    players = sorted({p for a, b, _ in comparisons
                      for p in (a, b)})
    if len(players) < 2:
        raise ValueError("need at least two players")
    if not connected(players, comparisons):
        raise ValueError("comparison graph is disconnected")
    if any(a == b or y not in {0, 1}
           for a, b, y in comparisons):
        raise ValueError("binary outcomes required")
    theta = {p: 0.0 for p in players}
    total = len(comparisons)
    for step in range(iterations):
        gradient = {p: -penalty * theta[p] for p in players}
        for a, b, result in comparisons:
            fitted = logistic(theta[a] - theta[b])
            residual = (result - fitted) / total
            gradient[a] += residual
            gradient[b] -= residual
        rate = .8
        updated = {p: theta[p] + rate * gradient[p]
                   for p in players}
        centre = sum(updated.values()) / len(players)
        updated = {p: value - centre
                   for p, value in updated.items()}
        change = max(abs(updated[p] - theta[p])
                     for p in players)
        theta = updated
        if change < tolerance:
            return {"parameters": theta,
                    "iterations": step + 1,
                    "penalty": penalty, "converged": True}
    return {"parameters": theta, "iterations": iterations,
            "penalty": penalty, "converged": False}


def reviewer_matrix(records, first, second, categories):
    by_task = defaultdict(dict)
    for row in records:
        task, reviewer = row["task_id"], row["reviewer_id"]
        if reviewer in by_task[task]:
            raise ValueError("duplicate task and reviewer")
        if row["category"] not in categories:
            raise ValueError("unknown category")
        by_task[task][reviewer] = row["category"]
    index = {label: i for i, label in enumerate(categories)}
    matrix = [[0] * len(categories) for _ in categories]
    missing = 0
    for ratings in by_task.values():
        if first not in ratings or second not in ratings:
            missing += 1
            continue
        a, b = ratings[first], ratings[second]
        matrix[index[a]][index[b]] += 1
    return {"matrix": matrix, "missing_pairs": missing}


def demo():
    balanced = kappa([[40, 10], [10, 40]])
    rare = kappa([[90, 5], [5, 0]])
    assert abs(balanced["kappa"] - .6) < 1e-12
    assert abs(rare["kappa"] + .052631578947) < 1e-10
    assert kappa([[100, 0], [0, 0]])["kappa"] is None
    assert preference(90, 60, 50) == .575
    a, b = elo_update(1400, 1600, 1)
    assert abs(a + b - 3000) < 1e-12
    observations = [("A", "B", 1)] * 7
    observations += [("A", "B", 0)] * 3
    fitted = fit_bradley_terry(observations)
    assert fitted["converged"]
    assert fitted["parameters"]["A"] > 0
    return {"balanced": balanced, "rare": rare,
            "fit": fitted}


if __name__ == "__main__":
    print(demo())
