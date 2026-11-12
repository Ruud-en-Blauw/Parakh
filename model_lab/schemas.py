"""Typed, JSON-serializable contracts for the ModelLab.

The lab intentionally uses standard-library dataclasses so the offline workflow
has no runtime dependency or network requirement. Constructors validate all
external values before records enter the rest of the system.
"""

from __future__ import annotations

from dataclasses import fields, dataclass, field, is_dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, Mapping

from .errors import ValidationError

ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
DIFFICULTIES = {1, 2, 3, 4}
SPLITS = {"train", "calibration", "holdout"}
DOMAINS = {
    "communication", "calendar", "task_planning", "expenses", "shopping",
    "travel", "document_extraction", "tabular_analysis", "support",
    "source_reasoning", "privacy", "conversation_memory", "multilingual",
    "automation", "ambiguity",
}
EVALUATION_METHODS = {"exact_json", "exact_text", "rubric", "schema", "imported"}


class AttemptStatus(str, Enum):
    SUCCESS = "success"
    REFUSAL = "refusal"
    PROVIDER_FAILURE = "provider_failure"
    TIMEOUT = "timeout"
    INVALID_PROVIDER_RESPONSE = "invalid_provider_response"
    MALFORMED_OUTPUT = "malformed_output"
    CANCELLED = "cancelled"
    PARTIAL = "partial"
    IMPORTED_INVALID = "imported_invalid"
    GRADING_FAILURE = "grading_failure"
    INFRASTRUCTURE_FAILURE = "infrastructure_failure"


class MetricKind(str, Enum):
    MEASURED = "measured"
    PROVIDER_REPORTED = "provider_reported"
    IMPORTED = "imported"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def require_id(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        raise ValidationError(f"{field_name} must be a stable identifier")
    return value


def require_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValidationError(f"{field_name} must be non-empty text")
    return value


def stable_json(value: Any) -> str:
    return json.dumps(_plain(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(stable_json(value).encode("utf-8")).hexdigest()


def _plain(value: Any) -> Any:
    # Fast paths for the JSON-native types that dominate evidence records (same results as below).
    kind = type(value)
    if kind is str or kind is int or kind is float or kind is bool or value is None:
        return value
    if kind is dict:
        return {str(k): _plain(v) for k, v in value.items()}
    if kind is list or kind is tuple:
        return [_plain(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {item.name: _plain(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


class FrozenDict(dict[Any, Any]):
    """Detached JSON-compatible immutable record, not an arbitrary-code sandbox."""
    def _immutable(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("record mappings are immutable")

    def __setitem__(self, key: Any, value: Any) -> None:
        raise TypeError("record mappings are immutable")

    def __delitem__(self, key: Any) -> None:
        raise TypeError("record mappings are immutable")

    def clear(self) -> None:
        raise TypeError("record mappings are immutable")

    def pop(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("record mappings are immutable")

    def popitem(self) -> Any:
        raise TypeError("record mappings are immutable")

    def setdefault(self, *args: Any, **kwargs: Any) -> Any:
        raise TypeError("record mappings are immutable")

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise TypeError("record mappings are immutable")

    def __ior__(self, other: Any) -> Any:
        raise TypeError("record mappings are immutable")

    def __copy__(self) -> FrozenDict:
        return self

    def __deepcopy__(self, memo: Any) -> FrozenDict:
        return self


def freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise ValidationError("JSON keys must be strings")
        return FrozenDict((key, freeze_json(item)) for key, item in value.items())
    if isinstance(value, (tuple, list)):
        return tuple(freeze_json(item) for item in value)
    if value is None or isinstance(value, (str, bool, int, Enum)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise ValidationError("record values must be finite JSON-compatible values")


def _freeze_fields(record: Any, *names: str) -> None:
    for name in names:
        object.__setattr__(record, name, freeze_json(getattr(record, name)))


def _finite(value: Any) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and (isinstance(value, int) or math.isfinite(value))


@dataclass(frozen=True)
class RubricCriterion:
    criterion: str
    weight: float = 1.0

    def __post_init__(self) -> None:
        require_text(self.criterion, "rubric criterion")
        if not _finite(self.weight) or self.weight <= 0:
            raise ValidationError("rubric weight must be positive")


@dataclass(frozen=True)
class Evaluation:
    method: str
    reference_answer: Any
    rubric: tuple[RubricCriterion, ...]
    critical_failures: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.method not in EVALUATION_METHODS:
            raise ValidationError(f"unsupported evaluation method: {self.method}")
        if not self.rubric:
            raise ValidationError("evaluation rubric cannot be empty")
        if any(not isinstance(x, RubricCriterion) for x in self.rubric):
            raise ValidationError("rubric must contain criteria")
        if any(not isinstance(x, str) or not x.strip() for x in self.critical_failures):
            raise ValidationError("critical failures must be non-empty strings")
        if self.method == "exact_json" and not isinstance(self.reference_answer, (Mapping, list, tuple, str, int, float, bool, type(None))):
            raise ValidationError("exact_json reference must be JSON-compatible")
        object.__setattr__(self, "rubric", tuple(self.rubric))
        _freeze_fields(self, "reference_answer", "critical_failures")


@dataclass(frozen=True)
class Limits:
    max_output_tokens: int
    max_tool_calls: int

    def __post_init__(self) -> None:
        if type(self.max_output_tokens) is not int or self.max_output_tokens <= 0:
            raise ValidationError("max_output_tokens must be a positive integer")
        if type(self.max_tool_calls) is not int or self.max_tool_calls < 0:
            raise ValidationError("max_tool_calls must be a non-negative integer")


@dataclass(frozen=True)
class Provenance:
    origin: str
    real_personal_data: bool
    source_uri: str | None = None
    source_hash: str | None = None

    def __post_init__(self) -> None:
        require_text(self.origin, "provenance origin")
        if not isinstance(self.real_personal_data, bool):
            raise ValidationError("real_personal_data must be boolean")


@dataclass(frozen=True)
class Case:
    case_id: str
    family_id: str
    split: str
    complexity_level: int
    domain: str
    language: str
    tags: tuple[str, ...]
    messages: tuple[dict[str, str], ...]
    evaluation: Evaluation
    fixture_policy: str
    suite_version: str
    provenance: Provenance
    limits: Limits

    def __post_init__(self) -> None:
        require_id(self.case_id, "case_id")
        require_id(self.family_id, "family_id")
        require_text(self.suite_version, "suite_version")
        if self.split not in SPLITS:
            raise ValidationError(f"invalid split: {self.split}")
        if type(self.complexity_level) is not int or self.complexity_level not in DIFFICULTIES:
            raise ValidationError("complexity_level must be 1, 2, 3, or 4")
        if self.domain not in DOMAINS:
            raise ValidationError(f"invalid domain: {self.domain}")
        if not self.language or not isinstance(self.language, str):
            raise ValidationError("language is required")
        if not self.messages:
            raise ValidationError("messages cannot be empty")
        for message in self.messages:
            if not isinstance(message, Mapping) or set(message) != {"role", "content"} or message.get("role") not in {"system", "user", "assistant", "tool"}:
                raise ValidationError("messages must contain valid role/content records")
            require_text(message.get("content", ""), "message content")
        require_text(self.fixture_policy, "fixture_policy")
        _freeze_fields(self, "messages", "tags")

    @property
    def prompt_hash(self) -> str:
        return stable_hash(self.messages)

    def candidate_payload(self) -> dict[str, Any]:
        """Return the only structure allowed to cross into candidate execution."""
        return {
            "case_id": self.case_id,
            "messages": [dict(message) for message in self.messages],
            "limits": {"max_output_tokens": self.limits.max_output_tokens, "max_tool_calls": self.limits.max_tool_calls},
            "prompt_hash": self.prompt_hash,
        }


@dataclass(frozen=True)
class Suite:
    suite_version: str
    cases: tuple[Case, ...]
    source_path: str | None = None
    source_hash: str | None = None

    def __post_init__(self) -> None:
        require_text(self.suite_version, "suite_version")
        if not self.cases:
            raise ValidationError("suite must contain at least one case")
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValidationError("duplicate case IDs")
        families = {case.family_id for case in self.cases}
        if not families:
            raise ValidationError("suite must contain family IDs")
        if any(case.suite_version != self.suite_version for case in self.cases):
            raise ValidationError("case/suite version mismatch")
        splits: dict[str, str] = {}
        for case in self.cases:
            if splits.setdefault(case.family_id, case.split) != case.split:
                raise ValidationError("all cases in a family must share one split")
        object.__setattr__(self, "cases", tuple(self.cases))

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(case.case_id for case in self.cases)

    def get(self, case_id: str) -> Case:
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise ValidationError(f"unknown case ID: {case_id}")

    def candidate_records(self) -> tuple[dict[str, Any], ...]:
        return tuple(case.candidate_payload() for case in self.cases)


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model: str
    revision: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    context_condition: str = "short"

    def __post_init__(self) -> None:
        require_id(self.provider, "provider")
        # Provider model identifiers commonly include a namespace/path (for
        # example ``openai/gpt-4.1``); they are still required to be a bounded,
        # non-whitespace string but are not local record IDs.
        if not isinstance(self.model, str) or not self.model.strip() or len(self.model) > 256 or any(char.isspace() for char in self.model):
            raise ValidationError("model must be a bounded non-whitespace identifier")
        if not isinstance(self.parameters, Mapping):
            raise ValidationError("parameters must be an object")
        require_text(self.context_condition, "context_condition")
        _freeze_fields(self, "parameters")


@dataclass(frozen=True)
class Budget:
    max_cases: int | None = None
    max_requests: int | None = None
    max_runtime_seconds: float | None = None
    max_cost_minor: int | None = None
    currency: str | None = None

    def __post_init__(self) -> None:
        for name in ("max_cases", "max_requests", "max_cost_minor"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValidationError(f"{name} must be a non-negative integer or null")
        if self.max_runtime_seconds is not None and (not _finite(self.max_runtime_seconds) or self.max_runtime_seconds < 0):
            raise ValidationError("max_runtime_seconds must be non-negative or null")
        if self.max_cost_minor is not None and not self.currency:
            raise ValidationError("currency is required when max_cost_minor is set")


@dataclass(frozen=True)
class Run:
    run_id: str
    suite_version: str
    case_ids: tuple[str, ...]
    model_config: ModelConfig
    seed: int | None
    budget: Budget
    started_at: str
    status: str = "created"
    environment: dict[str, Any] = field(default_factory=dict)
    prompt_hashes: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_id(self.run_id, "run_id")
        require_text(self.suite_version, "suite_version")
        if not self.case_ids:
            raise ValidationError("run must contain cases")
        if len(set(self.case_ids)) != len(self.case_ids):
            raise ValidationError("run case IDs must be unique")
        for case_id in self.case_ids:
            require_id(case_id, "case_id")
        if self.prompt_hashes and set(self.prompt_hashes) != set(self.case_ids):
            raise ValidationError("run prompt hashes must cover the exact case set")
        if any(not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value) for value in self.prompt_hashes.values()):
            raise ValidationError("run prompt hashes must be SHA-256 digests")
        if self.status not in {"created", "running", "completed", "partial", "failed", "cancelled", "incomplete"}:
            raise ValidationError(f"invalid run status: {self.status}")
        _freeze_fields(self, "case_ids", "environment", "prompt_hashes")


@dataclass(frozen=True)
class Attempt:
    attempt_id: str
    run_id: str
    logical_request_id: str
    case_id: str
    model_config: ModelConfig
    prompt_hash: str
    response_text: str | None
    status: AttemptStatus
    started_at: str | None
    completed_at: str | None = None
    finish_reason: str | None = None
    error: str | None = None
    first_token_latency_ms: float | None = None
    completion_latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_minor: int | None = None
    currency: str | None = None
    metric_kinds: dict[str, MetricKind] = field(default_factory=dict)
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("attempt_id", "run_id", "logical_request_id", "case_id", "prompt_hash"):
            require_id(getattr(self, name), name)
        if self.response_text is not None and not isinstance(self.response_text, str):
            raise ValidationError("response_text must be text or null")
        for name in ("first_token_latency_ms", "completion_latency_ms"):
            value = getattr(self, name)
            if value is not None and (not _finite(value) or value < 0):
                raise ValidationError(f"{name} must be non-negative or null")
        for name in ("input_tokens", "output_tokens", "cost_minor"):
            value = getattr(self, name)
            if value is not None and (type(value) is not int or value < 0):
                raise ValidationError(f"{name} must be a non-negative integer or null")
        if self.cost_minor is not None and not self.currency:
            raise ValidationError("currency is required with cost_minor")
        if not isinstance(self.raw_metadata, dict):
            raise ValidationError("raw_metadata must be an object")
        if not isinstance(self.status, AttemptStatus):
            raise ValidationError("status must be an AttemptStatus")
        _freeze_fields(self, "raw_metadata", "metric_kinds")


@dataclass(frozen=True)
class Grade:
    grade_id: str
    attempt_id: str
    grader_id: str
    grader_version: str
    method: str
    passed: bool | None
    score: float | None
    failure_reason: str | None
    evidence: dict[str, Any]
    created_at: str

    def __post_init__(self) -> None:
        for name in ("grade_id", "attempt_id", "grader_id", "grader_version"):
            require_id(getattr(self, name), name)
        if self.method not in EVALUATION_METHODS | {"human", "model_judge"}:
            raise ValidationError(f"unsupported grade method: {self.method}")
        if self.score is not None and (not _finite(self.score) or not 0 <= self.score <= 1):
            raise ValidationError("grade score must be between 0 and 1")
        if self.passed is None and self.score is None:
            if not self.failure_reason:
                raise ValidationError("abstained grade needs a reason")
        if not isinstance(self.evidence, dict):
            raise ValidationError("grade evidence must be an object")
        if self.passed is not None and not isinstance(self.passed, bool):
            raise ValidationError("grade passed must be boolean or null")
        _freeze_fields(self, "evidence")


@dataclass(frozen=True)
class HumanReview:
    review_id: str
    run_id: str
    case_id: str
    blind_label: str
    reviewer_pseudonym: str
    decision: str
    score: float | None
    notes: str | None
    confidence: float | None
    created_at: str
    attempt_id: str | None = None

    def __post_init__(self) -> None:
        for name in ("review_id", "run_id", "case_id", "blind_label", "reviewer_pseudonym"):
            require_id(getattr(self, name), name)
        if self.decision not in {"accept", "reject", "tie", "abstain"}:
            raise ValidationError("invalid human review decision")
        for name in ("score", "confidence"):
            value = getattr(self, name)
            if value is not None and (not _finite(value) or not 0 <= value <= 1):
                raise ValidationError(f"{name} must be between 0 and 1")
        if self.attempt_id is not None:
            require_id(self.attempt_id, "attempt_id")


@dataclass(frozen=True)
class ProvenanceRecord:
    record_id: str
    source_type: str
    source_label: str
    source_hash: str
    imported_at: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("record_id", "source_type", "source_label", "source_hash"):
            require_id(getattr(self, name), name)
        _freeze_fields(self, "metadata")


EVENT_TYPE_RE = re.compile(r"^[a-z][a-z_]{1,63}$")


@dataclass(frozen=True)
class RunEvent:
    """One append-only operational event (run progress, grading, actions).

    Events describe what the lab did; they never carry oracle/reference data or
    credentials. ``event_id`` is assigned by storage and is None before insert.
    """

    timestamp: str
    event_type: str
    message: str
    run_id: str | None = None
    status: str | None = None
    case_id: str | None = None
    model: str | None = None
    provider: str | None = None
    duration_ms: float | None = None
    cost_minor: int | None = None
    currency: str | None = None
    error: str | None = None
    artifact: str | None = None
    event_id: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.event_type, str) or not EVENT_TYPE_RE.fullmatch(self.event_type):
            raise ValidationError("event_type must be a lowercase identifier")
        require_text(self.timestamp, "timestamp")
        if not isinstance(self.message, str):
            raise ValidationError("event message must be text")
        for name in ("run_id", "case_id"):
            value = getattr(self, name)
            if value is not None:
                require_id(value, name)
        if self.duration_ms is not None and (not _finite(self.duration_ms) or self.duration_ms < 0):
            raise ValidationError("event duration_ms must be a non-negative number")
        if self.cost_minor is not None and (isinstance(self.cost_minor, bool) or not isinstance(self.cost_minor, int) or self.cost_minor < 0):
            raise ValidationError("event cost_minor must be a non-negative integer")


def to_dict(value: Any) -> dict[str, Any] | list[Any] | Any:
    return _plain(value)


def canonical_record(value: Any) -> str:
    return stable_json(_plain(value))
