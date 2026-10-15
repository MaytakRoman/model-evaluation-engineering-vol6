from collections import defaultdict
from hashlib import sha256
from pathlib import Path
import json


ALLOWED_SPLITS = {"development", "acceptance"}
INPUT_FIELDS = {"question", "context"}


def canonical(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )


def read_jsonl(path):
    records = []
    for number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError as exc:
            message = f"line {number}: bad JSON"
            raise ValueError(message) from exc
        if not isinstance(item, dict):
            message = f"line {number}: object required"
            raise ValueError(message)
        records.append(item)
    return records


def validate_record(record):
    errors = []
    for key in (
        "task_id", "group_id", "split", "task_type",
        "source_version", "rubric_version",
    ):
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            errors.append("missing string: " + key)
    if record.get("split") not in ALLOWED_SPLITS:
        errors.append("unknown split")
    payload = record.get("input")
    if not isinstance(payload, dict):
        errors.append("input must be object")
    else:
        extra = set(payload) - INPUT_FIELDS
        if extra:
            errors.append("unexpected input fields")
        if not payload.get("question"):
            errors.append("empty question")
        if "reference" in payload:
            errors.append("reference leaked to input")
    reference = record.get("reference")
    if not isinstance(reference, dict):
        errors.append("reference must be object")
    elif not reference.get("required_facts"):
        errors.append("reference has no required facts")
    return errors


def validate_dataset(records):
    errors = []
    seen = set()
    group_splits = defaultdict(set)
    text_splits = defaultdict(set)
    for number, record in enumerate(records, start=1):
        record_errors = validate_record(record)
        for error in record_errors:
            errors.append((number, error))
        if record_errors:
            continue
        task_id = record.get("task_id")
        if task_id in seen:
            errors.append((number, "duplicate task_id"))
        seen.add(task_id)
        group_id = record.get("group_id")
        split = record.get("split")
        group_splits[group_id].add(split)
        question = record.get("input", {}).get("question", "")
        normalized = " ".join(question.casefold().split())
        text_splits[normalized].add(split)
    for group, splits in group_splits.items():
        if len(splits) > 1:
            errors.append((group, "group crosses splits"))
    for question, splits in text_splits.items():
        if question and len(splits) > 1:
            message = "exact question leakage"
            errors.append((question, message))
    return errors


def candidate_view(record):
    errors = validate_record(record)
    if errors:
        raise ValueError(errors)
    source = record["input"]
    return {key: source[key] for key in INPUT_FIELDS
            if key in source}


def make_manifest(records, protocol_version):
    errors = validate_dataset(records)
    if errors:
        raise ValueError(errors)
    ordered = sorted(records, key=lambda x: x["task_id"])
    encoded = canonical(ordered).encode("utf-8")
    counts = defaultdict(int)
    groups = defaultdict(set)
    for record in ordered:
        split = record["split"]
        counts[split] += 1
        groups[split].add(record["group_id"])
    return {
        "protocol_version": protocol_version,
        "sha256": sha256(encoded).hexdigest(),
        "tasks": dict(counts),
        "groups": {k: len(v) for k, v in groups.items()},
    }


def sample_record(task_id, group_id, split, question):
    return {
        "task_id": task_id,
        "group_id": group_id,
        "split": split,
        "task_type": "source_grounding",
        "source_version": "documents_1",
        "rubric_version": "rubric_1",
        "input": {"question": question, "context": "rule"},
        "reference": {"required_facts": ["request approval"]},
    }


def demo():
    records = [
        sample_record("q1", "d1", "development", "First?"),
        sample_record("q2", "d1", "acceptance", "Second?"),
    ]
    assert any("group crosses" in str(e)
               for e in validate_dataset(records))
    records[1]["group_id"] = "d2"
    assert not validate_dataset(records)
    first = make_manifest(records, "1")
    second = make_manifest(list(reversed(records)), "1")
    assert first["sha256"] == second["sha256"]
    view = candidate_view(records[0])
    assert "reference" not in view
    assert "split" not in view
    return first


if __name__ == "__main__":
    print(canonical(demo()))
