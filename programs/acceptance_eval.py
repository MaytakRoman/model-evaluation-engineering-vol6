from math import exp, lgamma, log, log1p


def check_counts(k, n):
    if type(n) is not int or n < 1:
        raise ValueError("positive integer n required")
    if type(k) is not int or not 0 <= k <= n:
        raise ValueError("invalid integer count")


def binomial_cdf(k, n, p):
    if not 0 <= p <= 1:
        raise ValueError("invalid probability")
    if k < 0:
        return 0.0
    if k >= n or p == 0:
        return 1.0
    if p == 1:
        return 0.0
    terms = [
        lgamma(n + 1) - lgamma(j + 1)
        - lgamma(n - j + 1)
        + j * log(p) + (n - j) * log1p(-p)
        for j in range(k + 1)
    ]
    largest = max(terms)
    value = exp(largest) * sum(
        exp(t - largest) for t in terms
    )
    return min(1.0, value)


def solve_cdf(k, n, target):
    left, right = 0.0, 1.0
    for _ in range(65):
        middle = (left + right) / 2
        if binomial_cdf(k, n, middle) > target:
            left = middle
        else:
            right = middle
    return (left + right) / 2


def exact_bounds(k, n, alpha=.05):
    check_counts(k, n)
    if not 0 < alpha < 1:
        raise ValueError("invalid alpha")
    lower = 0.0 if k == 0 else solve_cdf(
        k - 1, n, 1 - alpha
    )
    upper = 1.0 if k == n else solve_cdf(
        k, n, alpha
    )
    return lower, upper


def evaluate(records, config):
    expected = config["expected_tasks"]
    if type(expected) is not int or expected < 1:
        raise ValueError("invalid planned denominator")
    if len(records) != expected:
        raise ValueError("planned denominator mismatch")
    seen = set()
    groups = set()
    for row in records:
        if row["task_id"] in seen:
            raise ValueError("duplicate task")
        seen.add(row["task_id"])
        if row["group_id"] in groups:
            raise ValueError("exact method needs iid tasks")
        groups.add(row["group_id"])
        if row["configuration_id"] != config["id"]:
            raise ValueError("configuration mismatch")
        for key in ["quality", "critical", "within_limit",
                    "trace_complete", "authority_ok"]:
            if type(row[key]) is not bool:
                raise ValueError("Boolean outcome required")
        if row["cost"] < 0:
            raise ValueError("negative cost")
    n = len(records)
    alpha = config["family_alpha"] / 3
    quality = sum(r["quality"] for r in records)
    critical = sum(r["critical"] for r in records)
    timely = sum(r["within_limit"] for r in records)
    lower_q = exact_bounds(quality, n, alpha)[0]
    upper_r = exact_bounds(critical, n, alpha)[1]
    lower_t = exact_bounds(timely, n, alpha)[0]
    checks = {
        "Q1": lower_q >= config["quality_min"],
        "S1": upper_r <= config["risk_max"],
        "T1": lower_t >= config["timely_min"],
        "A1": all(r["authority_ok"] for r in records),
        "R1": all(r["trace_complete"] for r in records),
    }
    cost = sum(r["cost"] for r in records)
    return {
        "tasks": n, "quality_successes": quality,
        "critical_events": critical,
        "timely_successes": timely,
        "alpha_per_bound": alpha,
        "quality_lower": lower_q,
        "risk_upper": upper_r,
        "timely_lower": lower_t,
        "cost_total": cost,
        "cost_per_accepted": cost / quality
        if quality else None,
        "checks": checks,
        "statistical_and_technical_pass": all(
            checks.values()
        ),
        "release_decision": "separate authorized review",
    }


def sample_config():
    return {
        "id": "educational-v1", "expected_tasks": 600,
        "family_alpha": .05, "quality_min": .85,
        "risk_max": .01, "timely_min": .94,
    }


def sample_records():
    return [{
        "task_id": f"task_{i:04d}",
        "group_id": f"independent_{i:04d}",
        "configuration_id": "educational-v1",
        "quality": i < 552,
        "critical": False, "within_limit": i < 582,
        "trace_complete": True, "authority_ok": True,
        "cost": 1.2, "data_kind": "synthetic_fixture",
    } for i in range(600)]


def demo():
    result = evaluate(sample_records(), sample_config())
    assert result["statistical_and_technical_pass"]
    assert abs(result["risk_upper"] - (
        1 - (.05 / 3) ** (1 / 600)
    )) < 1e-11
    altered = sample_records()
    altered[0]["authority_ok"] = False
    rejected = evaluate(altered, sample_config())
    assert not rejected["checks"]["A1"]
    assert not rejected["statistical_and_technical_pass"]
    return result


if __name__ == "__main__":
    import json
    print(json.dumps(demo(), indent=2))
