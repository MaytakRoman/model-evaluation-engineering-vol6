from dataclasses import asdict, dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
import json
import math


class Status(str, Enum):
    COMPLETED = "completed"
    TIMEOUT = "timeout"
    SYSTEM_ERROR = "system_error"
    INVALID_ENV = "invalid_environment"


def canonical_json(value):
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False,
    )


def digest_text(value):
    return sha256(value.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class Configuration:
    schema_version: str
    model_id: str
    prompt_hash: str
    source_version: str
    rubric_version: str
    temperature: float
    max_attempts: int

    def validate(self):
        for key in (
            "schema_version", "model_id", "source_version",
            "rubric_version",
        ):
            if not getattr(self, key).strip():
                raise ValueError("empty field: " + key)
        if len(self.prompt_hash) != 64:
            raise ValueError("invalid prompt hash")
        int(self.prompt_hash, 16)
        if not math.isfinite(self.temperature):
            raise ValueError("nonfinite temperature")
        if self.temperature < 0:
            raise ValueError("negative temperature")
        if self.max_attempts < 1:
            raise ValueError("invalid attempt limit")

    def fingerprint(self):
        self.validate()
        return digest_text(canonical_json(asdict(self)))


@dataclass(frozen=True)
class Measurement:
    measurement_id: str
    population: str
    independent_unit: str
    score_definition: str
    timeout_seconds: float
    quality_threshold: float
    risk_threshold: float

    def validate(self):
        for key in (
            "measurement_id", "population",
            "independent_unit", "score_definition",
        ):
            if not getattr(self, key).strip():
                raise ValueError("empty field: " + key)
        if self.timeout_seconds <= 0:
            raise ValueError("nonpositive time limit")
        for threshold in (
            self.quality_threshold, self.risk_threshold,
        ):
            if not 0 <= threshold <= 1:
                raise ValueError("invalid threshold")


@dataclass(frozen=True)
class Observation:
    task_id: str
    group_id: str
    configuration_id: str
    status: Status
    score: float | None
    critical: bool | None
    elapsed_seconds: float
    cost: float

    def validate(self):
        if not all((self.task_id, self.group_id,
                    self.configuration_id)):
            raise ValueError("missing identity")
        for value in (self.elapsed_seconds, self.cost):
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid resource value")
        if self.status == Status.INVALID_ENV:
            if self.score is not None:
                raise ValueError("invalid trial has score")
        elif self.score is None:
            raise ValueError("valid trial needs score")
        elif not 0 <= self.score <= 1:
            raise ValueError("score outside unit interval")


def validate_trial(config, measure, observation):
    config.validate()
    measure.validate()
    observation.validate()
    if observation.configuration_id != config.fingerprint():
        raise ValueError("configuration mismatch")
    if observation.status == Status.TIMEOUT:
        if observation.score != 0:
            raise ValueError("timeout cannot be success")
    if observation.status == Status.COMPLETED:
        if observation.elapsed_seconds > (
            measure.timeout_seconds
        ) and observation.score != 0:
            raise ValueError("success exceeds time budget")


def save_manifest(path, config, measure):
    config.validate()
    measure.validate()
    payload = {
        "configuration": asdict(config),
        "configuration_id": config.fingerprint(),
        "measurement": asdict(measure),
    }
    Path(path).write_text(
        canonical_json(payload) + "\n", encoding="utf-8",
    )
    return payload


def demo(output_dir=None):
    config = Configuration(
        "1", "local_demo", digest_text("fixed prompt"),
        "documents_1", "rubric_1", 0.0, 2,
    )
    measure = Measurement(
        "acceptance_1", "specified requests",
        "document", "correct within budget", 30, .85, .01,
    )
    record = Observation(
        "q1", "doc1", config.fingerprint(),
        Status.COMPLETED, 1.0, False, 4.5, .02,
    )
    validate_trial(config, measure, record)
    timeout = Observation(
        "q2", "doc2", config.fingerprint(),
        Status.TIMEOUT, 0.0, False, 30.1, .03,
    )
    validate_trial(config, measure, timeout)
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        save_manifest(output / "manifest.json", config, measure)
        rows = [asdict(record), asdict(timeout)]
        (output / "observations.jsonl").write_text(
            "".join(canonical_json(row) + "\n" for row in rows),
            encoding="utf-8",
        )
    assert config.fingerprint() == config.fingerprint()
    return config.fingerprint()


if __name__ == "__main__":
    print(demo())
