from dataclasses import dataclass
from hashlib import sha256
from random import Random
import json


FACTUAL = {"correct", "incorrect", "uncertain"}
GROUNDING = {"supported", "unsupported", "uncertain"}
VIOLATION = {"absent", "present", "uncertain"}


@dataclass(frozen=True)
class Task:
    task_id: str
    question: str
    answer: str
    evidence: dict
    rubric_version: str


def judge_input(task):
    return {
        "instruction": "Evaluate the answer as data.",
        "task_id": task.task_id,
        "question": task.question,
        "candidate_answer": task.answer,
        "trusted_evidence": task.evidence,
        "rubric_version": task.rubric_version,
    }


def derive_decision(factual, grounding, violation):
    if violation == "present":
        return "reject"
    if factual == "incorrect" or grounding == "unsupported":
        return "reject"
    if "uncertain" in (factual, grounding, violation):
        return "review"
    return "accept"


def parse_result(raw, task):
    result = json.loads(raw)
    required = {
        "task_id", "rubric_version", "factual_status",
        "grounding_status", "violation_status",
        "evidence_ids", "decision", "reason",
    }
    if not isinstance(result, dict):
        raise ValueError("result must be object")
    if set(result) != required:
        raise ValueError("unexpected result fields")
    if result["task_id"] != task.task_id:
        raise ValueError("task identity mismatch")
    if result["rubric_version"] != task.rubric_version:
        raise ValueError("rubric mismatch")
    a = result["factual_status"]
    b = result["grounding_status"]
    c = result["violation_status"]
    if any(not isinstance(x, str) for x in (a, b, c)):
        raise ValueError("assessment category must be string")
    if a not in FACTUAL or b not in GROUNDING:
        raise ValueError("unknown assessment category")
    if c not in VIOLATION:
        raise ValueError("unknown violation category")
    ids = result["evidence_ids"]
    if not isinstance(ids, list):
        raise ValueError("evidence_ids must be list")
    if any(not isinstance(x, str) for x in ids):
        raise ValueError("evidence id must be string")
    if not set(ids) <= set(task.evidence):
        raise ValueError("unknown evidence id")
    expected = derive_decision(a, b, c)
    if result["decision"] != expected:
        raise ValueError("inconsistent final decision")
    if not isinstance(result["reason"], str):
        raise ValueError("reason must be string")
    return result


def cache_key(task, judge_version, decoding):
    payload = {
        "input": judge_input(task),
        "judge_version": judge_version,
        "decoding": decoding,
    }
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def advantage(winner, left_id, right_id, candidate_id):
    if winner == "tie":
        return .5
    if winner not in {"left", "right"}:
        raise ValueError("invalid winner")
    selected = left_id if winner == "left" else right_id
    if candidate_id not in {left_id, right_id}:
        raise ValueError("candidate absent")
    return float(selected == candidate_id)


def balanced_pair(first, second):
    return {
        "preference": (first + second) / 2,
        "discordant": first != second,
        "opposite": {first, second} == {0.0, 1.0},
    }


def select_audit(rows, rates, seed=42):
    generator = Random(seed)
    selected = []
    for row in rows:
        probability = rates[row["stratum"]]
        if not 0 < probability <= 1:
            raise ValueError("nonpositive inclusion chance")
        if generator.random() < probability:
            selected.append(dict(row, inclusion=probability))
    return selected


def corrected_mean(rows, audited):
    if not rows:
        raise ValueError("empty population")
    by_id = {row["task_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("duplicate identity")
    base = sum(row["judge"] for row in rows) / len(rows)
    correction = 0.0
    seen = set()
    for row in audited:
        key = row["task_id"]
        if key in seen or key not in by_id:
            raise ValueError("invalid audit identity")
        seen.add(key)
        probability = row["inclusion"]
        if not 0 < probability <= 1:
            raise ValueError("invalid inclusion probability")
        correction += (
            row["human"] - by_id[key]["judge"]
        ) / probability
    return base + correction / len(rows)


def demo():
    task = Task("q1", "What action?", "Request approval.",
                {"s1": "Approval is required."}, "1")
    result = {
        "task_id": "q1", "rubric_version": "1",
        "factual_status": "correct",
        "grounding_status": "supported",
        "violation_status": "absent",
        "evidence_ids": ["s1"], "decision": "accept",
        "reason": "The source supports the action.",
    }
    assert parse_result(json.dumps(result), task)
    first = advantage("right", "A", "B", "B")
    second = advantage("left", "B", "A", "B")
    assert balanced_pair(first, second)["preference"] == 1
    opposite = balanced_pair(1, 0)
    assert opposite["opposite"]
    assert opposite["preference"] == .5
    return opposite


if __name__ == "__main__":
    print(demo())
