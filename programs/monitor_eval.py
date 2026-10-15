from hashlib import sha256
from math import erfc, sqrt
from statistics import NormalDist


def assign(user_id, experiment_id, salt):
    if not all([user_id, experiment_id, salt]):
        raise ValueError("assignment keys required")
    text = "|".join([salt, experiment_id, user_id])
    number = int.from_bytes(
        sha256(text.encode("utf-8")).digest()[:8], "big"
    )
    return "A" if number < 2**63 else "B"


def srm(count_a, count_b, probability_a=.5):
    if min(count_a, count_b) < 0:
        raise ValueError("negative assignments")
    n = count_a + count_b
    if n == 0 or not 0 < probability_a < 1:
        raise ValueError("invalid SRM input")
    expected_a = n * probability_a
    expected_b = n - expected_a
    statistic = ((count_a - expected_a)**2 / expected_a
                 + (count_b - expected_b)**2 / expected_b)
    return {"chi_square": statistic,
            "p_value_approx": erfc(sqrt(statistic / 2))}


def total_variation(reference, current):
    if set(reference) != set(current):
        raise ValueError("categories must match")
    for values in [reference, current]:
        if any(v < 0 for v in values.values()):
            raise ValueError("negative share")
        if abs(sum(values.values()) - 1) > 1e-9:
            raise ValueError("shares must sum to one")
    return sum(abs(reference[k] - current[k])
               for k in reference) / 2


def difference_interval(k_a, n_a, k_b, n_b,
                        confidence=.95):
    if min(n_a, n_b) < 1 or not 0 < confidence < 1:
        raise ValueError("invalid denominators")
    if not 0 <= k_a <= n_a or not 0 <= k_b <= n_b:
        raise ValueError("invalid successes")
    a, b = k_a / n_a, k_b / n_b
    standard_error = sqrt(a * (1 - a) / n_a
                          + b * (1 - b) / n_b)
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    return {"effect": b - a,
            "lower": b - a - z * standard_error,
            "upper": b - a + z * standard_error,
            "method": "independent Wald approximation"}


def audit_events(assignments, events):
    assigned = {}
    for row in assignments:
        user = row["user_id"]
        if user in assigned:
            raise ValueError("duplicate assignment")
        if row["variant"] not in {"A", "B"}:
            raise ValueError("unknown variant")
        assigned[user] = row["variant"]
    seen = set()
    complete = set()
    for event in events:
        key = event["event_id"]
        if key in seen:
            raise ValueError("duplicate event")
        seen.add(key)
        user = event["user_id"]
        if user not in assigned:
            raise ValueError("unassigned outcome")
        if assigned[user] != event["variant"]:
            raise ValueError("assignment changed")
        if event["type"] == "final_outcome":
            if user in complete:
                raise ValueError("duplicate final outcome")
            complete.add(user)
    counts = {v: sum(x == v for x in assigned.values())
              for v in ["A", "B"]}
    return {"assignments": counts,
            "outcomes": len(complete),
            "missing_outcomes": sorted(set(assigned)
                                       - complete),
            "srm": srm(counts["A"], counts["B"])}


def demo():
    assert assign("u1", "exp1", "demo") == assign(
        "u1", "exp1", "demo"
    )
    assert srm(200, 200)["p_value_approx"] == 1
    assert srm(300, 100)["p_value_approx"] < 1e-10
    tv = total_variation({"easy": .8, "hard": .2},
                         {"easy": .6, "hard": .4})
    assert abs(tv - .2) < 1e-12
    assignments = [{"user_id": "u1", "variant": "A"},
                   {"user_id": "u2", "variant": "B"}]
    events = [{"event_id": "e1", "user_id": "u1",
               "variant": "A", "type": "final_outcome"}]
    result = audit_events(assignments, events)
    assert result["missing_outcomes"] == ["u2"]
    try:
        audit_events(assignments, events + events)
    except ValueError:
        pass
    else:
        raise AssertionError("duplicate not detected")
    return {"audit": result,
            "comparison": difference_interval(
                160, 200, 176, 200
            ), "total_variation": tv}


if __name__ == "__main__":
    import json
    print(json.dumps(demo(), indent=2))
