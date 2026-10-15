from collections import defaultdict
from math import ceil, log, sqrt
from random import Random
from statistics import NormalDist, mean


def check_count(successes, total):
    if not isinstance(total, int) or total <= 0:
        raise ValueError("positive integer total required")
    if not isinstance(successes, int):
        raise ValueError("integer successes required")
    if not 0 <= successes <= total:
        raise ValueError("count outside bounds")


def wilson(successes, total, confidence=.95,
           one_sided=False):
    check_count(successes, total)
    if not 0 < confidence < 1:
        raise ValueError("invalid confidence")
    probability = confidence if one_sided else (
        (1 + confidence) / 2
    )
    z = NormalDist().inv_cdf(probability)
    p = successes / total
    scale = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / scale
    width = z / scale * sqrt(
        p * (1 - p) / total + z * z / (4 * total**2)
    )
    return centre - width, centre + width


def zero_event_upper(total, alpha=.05):
    check_count(0, total)
    if not 0 < alpha < 1:
        raise ValueError("invalid alpha")
    return 1 - alpha ** (1 / total)


def zero_event_size(threshold, alpha=.05):
    if not 0 < threshold < 1 or not 0 < alpha < 1:
        raise ValueError("invalid probability")
    return ceil(log(alpha) / log(1 - threshold))


def holm_adjust(p_values):
    if any(not 0 <= p <= 1 for p in p_values):
        raise ValueError("invalid p-value")
    order = sorted(range(len(p_values)),
                   key=lambda i: p_values[i])
    adjusted = [0.0] * len(p_values)
    previous = 0.0
    size = len(order)
    for rank, index in enumerate(order):
        current = min(1.0, (size - rank) * p_values[index])
        previous = max(previous, current)
        adjusted[index] = previous
    return adjusted


def quantile(values, probability):
    if not values or not 0 <= probability <= 1:
        raise ValueError("invalid quantile input")
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    left = int(position)
    right = min(left + 1, len(ordered) - 1)
    fraction = position - left
    return (ordered[left] * (1 - fraction)
            + ordered[right] * fraction)


def group_differences(records):
    grouped = defaultdict(list)
    seen = set()
    for row in records:
        key = row["task_id"]
        if key in seen:
            raise ValueError("duplicate task")
        seen.add(key)
        a, b = row["score_a"], row["score_b"]
        if not 0 <= a <= 1 or not 0 <= b <= 1:
            raise ValueError("invalid score")
        grouped[row["group_id"]].append(b - a)
    if len(grouped) < 2:
        raise ValueError("need at least two groups")
    return [mean(values) for _, values in
            sorted(grouped.items())]


def paired_cluster_bootstrap(records, repeats=10000,
                             confidence=.95, seed=42):
    if repeats < 100:
        raise ValueError("too few Monte Carlo repeats")
    if not 0 < confidence < 1:
        raise ValueError("invalid confidence")
    differences = group_differences(records)
    size = len(differences)
    generator = Random(seed)
    draws = []
    for _ in range(repeats):
        selected = [differences[generator.randrange(size)]
                    for _ in range(size)]
        draws.append(mean(selected))
    alpha = 1 - confidence
    return {
        "effect": mean(differences),
        "lower": quantile(draws, alpha / 2),
        "upper": quantile(draws, 1 - alpha / 2),
        "groups": size,
        "repeats": repeats,
        "seed": seed,
    }


def acceptance_status(lower, upper, threshold):
    if not 0 <= lower <= upper <= 1:
        raise ValueError("invalid bounds")
    if lower >= threshold:
        return "confirmed"
    if upper < threshold:
        return "not_met"
    return "inconclusive"


def demo():
    lower, upper = wilson(90, 100)
    assert abs(lower - .8256343385) < 1e-9
    assert abs(upper - .9447708629) < 1e-9
    assert zero_event_size(.001) == 2995
    assert zero_event_size(.01) == 299
    assert zero_event_upper(200) > .01
    adjusted = holm_adjust([.004, .018, .041, .20])
    expected = [.016, .054, .082, .20]
    assert all(abs(a - b) < 1e-12
               for a, b in zip(adjusted, expected))
    records = [
        {"task_id": "q1", "group_id": "d1",
         "score_a": 0, "score_b": 1},
        {"task_id": "q2", "group_id": "d1",
         "score_a": 1, "score_b": 1},
        {"task_id": "q3", "group_id": "d2",
         "score_a": 1, "score_b": 0},
    ]
    result = paired_cluster_bootstrap(records, repeats=1000)
    reversed_records = [dict(row, score_a=row["score_b"],
                             score_b=row["score_a"])
                        for row in records]
    reverse = paired_cluster_bootstrap(
        reversed_records, repeats=1000,
    )
    assert abs(result["effect"] + reverse["effect"]) < 1e-12
    assert abs(result["lower"] + reverse["upper"]) < 1e-12
    return {"wilson": (lower, upper), "bootstrap": result}


if __name__ == "__main__":
    print(demo())
