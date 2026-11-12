"""Read model over a ModelLab workspace for operators and the TUI.

Everything here is derived from durable state (suite files, the SQLite store,
environment variables for provider configuration). Nothing is estimated:
missing measurements stay ``None`` and are rendered as unknown by callers.
Oracle/reference answers never leave this module; case views are built from
candidate-safe metadata only.
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from model_lab.benchmark import load_suite, validate_suite
from model_lab.errors import ModelLabError
from model_lab.schemas import Attempt, AttemptStatus, Case, Grade, Run, RunEvent, Suite
from model_lab.storage import SQLiteStore

DB_NAME = "model_lab.sqlite3"
PROVENANCE_BY_MODE = {"offline_synthetic": "SIMULATED", "imported": "IMPORTED", "live_pilot": "MEASURED"}


def provenance_label(run: Run) -> str:
    """SIMULATED (fake provider), IMPORTED, MEASURED (live provider) or UNKNOWN."""
    mode = str(run.environment.get("mode", ""))
    if mode in PROVENANCE_BY_MODE:
        return PROVENANCE_BY_MODE[mode]
    if run.model_config.provider == "fake":
        return "SIMULATED"
    return "UNKNOWN"


@dataclass(frozen=True)
class CostSummary:
    total_minor: int | None
    currency: str | None
    known: int
    unknown: int
    note: str


def summarize_cost(attempts: Iterable[Attempt]) -> CostSummary:
    """Sum only when every attempt reports a cost in exactly one currency."""
    costs = [(a.cost_minor, a.currency) for a in attempts]
    known = [(c, cur) for c, cur in costs if c is not None]
    unknown = len(costs) - len(known)
    if not costs:
        return CostSummary(None, None, 0, 0, "no attempts")
    if not known:
        return CostSummary(None, None, 0, unknown, "not measured")
    currencies = {cur for _, cur in known}
    if unknown:
        return CostSummary(None, None, len(known), unknown, f"incomplete: {unknown} attempts without cost")
    if len(currencies) != 1 or None in currencies:
        return CostSummary(None, None, len(known), 0, "mixed or missing currency; not summed")
    return CostSummary(sum(c for c, _ in known), next(iter(currencies)), len(known), 0, "provider reported")


@dataclass(frozen=True)
class RunSummary:
    run_id: str
    model: str
    provider: str
    suite_version: str
    mode: str
    provenance: str
    status: str
    started_at: str
    finished_at: str | None
    duration_seconds: float | None
    case_count: int
    cases_attempted: int
    cases_failed: int
    attempts: int
    retries: int
    graded: int
    passed: int
    failed_grades: int
    abstained: int
    cost: CostSummary
    latency_ms_mean: float | None
    latency_known: int
    seed: int | None
    parameters: dict[str, Any]
    context_condition: str | None
    budget: dict[str, Any]
    environment: dict[str, Any]

    @property
    def pass_rate(self) -> float | None:
        decided = self.passed + self.failed_grades
        return self.passed / decided if decided else None

    @property
    def progress(self) -> float | None:
        return self.cases_attempted / self.case_count if self.case_count else None


def _latest_per_case(attempts: list[Attempt]) -> dict[str, Attempt]:
    latest: dict[str, Attempt] = {}
    for attempt in attempts:
        latest[attempt.case_id] = attempt  # storage returns insertion order
    return latest


def _seconds_between(start: str | None, end: str | None) -> float | None:
    if not start or not end:
        return None
    try:
        return max(0.0, (datetime.fromisoformat(end) - datetime.fromisoformat(start)).total_seconds())
    except ValueError:
        return None


def summarize_run(store: SQLiteStore, run_id: str) -> RunSummary:
    run = store.get_run(run_id)
    attempts = store.list_attempts(run_id)
    latest = _latest_per_case(attempts)
    final_ids = {a.attempt_id for a in latest.values()}
    grades = [g for g in store.list_grades(run_id) if g.attempt_id in final_ids]  # one grade per case: its final attempt
    finish_events = [e for e in store.list_events(run_id=run_id, limit=50) if e.event_type == "run_finished"]
    finished_at = finish_events[-1].timestamp if finish_events else (max((a.completed_at for a in attempts if a.completed_at), default=None) if run.status not in {"running", "created"} else None)
    latencies = [a.completion_latency_ms for a in attempts if a.completion_latency_ms is not None]
    return RunSummary(
        run_id=run.run_id, model=run.model_config.model, provider=run.model_config.provider,
        suite_version=run.suite_version, mode=str(run.environment.get("mode", "unknown")), provenance=provenance_label(run),
        status=run.status, started_at=run.started_at, finished_at=finished_at,
        duration_seconds=_seconds_between(run.started_at, finished_at),
        case_count=len(run.case_ids), cases_attempted=len(latest),
        cases_failed=sum(a.status is not AttemptStatus.SUCCESS for a in latest.values()),
        attempts=len(attempts), retries=len(attempts) - len(latest),
        graded=len(grades), passed=sum(g.passed is True for g in grades), failed_grades=sum(g.passed is False for g in grades),
        abstained=sum(g.passed is None for g in grades), cost=summarize_cost(attempts),
        latency_ms_mean=mean(latencies) if latencies else None, latency_known=len(latencies), seed=run.seed,
        parameters=dict(run.model_config.parameters), context_condition=run.model_config.context_condition,
        budget={k: v for k, v in run.budget.__dict__.items()}, environment=dict(run.environment),
    )


@dataclass(frozen=True)
class SuiteSummary:
    path: str
    version: str | None
    source_hash: str | None
    cases: int
    families: int
    domains: dict[str, int]
    complexity: dict[str, int]
    splits: dict[str, int]
    error: str | None = None


def discover_suites(root: str | Path = "benchmarks") -> list[Path]:
    base = Path(root)
    return sorted(base.glob("*.jsonl")) if base.is_dir() else []


def summarize_suite(path: str | Path) -> SuiteSummary:
    try:
        suite = load_suite(path)
    except (ModelLabError, OSError, ValueError) as exc:
        return SuiteSummary(str(path), None, None, 0, 0, {}, {}, {}, error=str(exc))
    return summarize_loaded_suite(suite, path)


def summarize_loaded_suite(suite: Suite, path: str | Path) -> SuiteSummary:
    counts = validate_suite(suite)
    return SuiteSummary(str(path), suite.suite_version, suite.source_hash, counts["cases"], counts["families"],
                        dict(sorted(Counter(c.domain for c in suite.cases).items())), counts["complexity"], counts["splits"])


@dataclass(frozen=True)
class CaseView:
    """Operator-facing case metadata. Deliberately excludes evaluation/oracle data."""

    case_id: str
    family_id: str
    domain: str
    complexity_level: int
    split: str
    language: str
    tags: tuple[str, ...]
    evaluation_method: str
    message_count: int
    prompt_preview: str
    latest_status: str
    latest_grade: str


def case_views(suite: Suite, attempts_by_case: dict[str, Attempt] | None = None, grades_by_attempt: dict[str, Grade] | None = None) -> list[CaseView]:
    attempts_by_case = attempts_by_case or {}
    grades_by_attempt = grades_by_attempt or {}
    views = []
    for case in suite.cases:
        payload = case.candidate_payload()
        messages = payload.get("messages", [])
        first_user = next((m.get("content", "") for m in messages if isinstance(m, dict) and m.get("role") == "user"), "")
        attempt = attempts_by_case.get(case.case_id)
        grade = grades_by_attempt.get(attempt.attempt_id) if attempt else None
        views.append(CaseView(
            case_id=case.case_id, family_id=case.family_id, domain=case.domain, complexity_level=case.complexity_level,
            split=case.split, language=case.language, tags=tuple(case.tags), evaluation_method=case.evaluation.method,
            message_count=len(messages), prompt_preview=str(first_user),
            latest_status=attempt.status.value if attempt else "not run",
            latest_grade=_grade_label(grade) if attempt else "not run",
        ))
    return views


def _grade_label(grade: Grade | None) -> str:
    if grade is None:
        return "ungraded"
    if grade.passed is None:
        return "needs review"
    return "passed" if grade.passed else "failed"


@dataclass(frozen=True)
class BreakdownRow:
    group: str
    graded: int
    passed: int
    failed: int
    abstained: int
    attempt_failures: int

    @property
    def pass_rate(self) -> float | None:
        decided = self.passed + self.failed
        return self.passed / decided if decided else None


@dataclass(frozen=True)
class RunResults:
    by_domain: list[BreakdownRow]
    by_complexity: list[BreakdownRow]
    by_split: list[BreakdownRow]
    attempt_status_counts: dict[str, int]
    refusals: int
    critical_unassessed: int
    review_items: list[ReviewItem]
    ungraded_attempts: int


@dataclass(frozen=True)
class ReviewItem:
    run_id: str
    case_id: str
    model: str
    attempt_id: str
    response_status: str
    grading_status: str
    reason: str
    critical: str
    review_status: str
    provenance: str
    response_excerpt: str
    error: str | None


def run_results(store: SQLiteStore, suite: Suite | None, run_id: str) -> RunResults:
    run = store.get_run(run_id)
    attempts = store.list_attempts(run_id)
    latest = _latest_per_case(attempts)
    grades = {g.attempt_id: g for g in store.list_grades(run_id)}
    reviewed = {r.case_id: r.decision for r in store.list_reviews(run_id)}
    cases: dict[str, Case] = {c.case_id: c for c in suite.cases} if suite else {}
    groups: dict[str, dict[str, list[int]]] = {"domain": {}, "complexity": {}, "split": {}}
    review: list[ReviewItem] = []
    critical_unassessed = 0
    for case_id, attempt in latest.items():
        grade = grades.get(attempt.attempt_id)
        case = cases.get(case_id)
        keys = {"domain": case.domain if case else "unknown (case not in suite)",
                "complexity": f"level {case.complexity_level}" if case else "unknown",
                "split": case.split if case else "unknown"}
        outcome = [1, int(grade is not None and grade.passed is True), int(grade is not None and grade.passed is False),
                   int(grade is not None and grade.passed is None), int(attempt.status is not AttemptStatus.SUCCESS)]
        if grade is None:
            outcome[0] = 0
        for dimension, key in keys.items():
            bucket = groups[dimension].setdefault(key, [0, 0, 0, 0, 0])
            for i, value in enumerate(outcome):
                bucket[i] += value
        critical = str(grade.evidence.get("critical_assessment", "unassessed")) if grade else "unassessed"
        if critical == "unassessed":
            critical_unassessed += 1
        needs_review = attempt.status is not AttemptStatus.SUCCESS or grade is None or grade.passed is not True
        if needs_review:
            review.append(ReviewItem(
                run_id=run_id, case_id=case_id, model=run.model_config.model, attempt_id=attempt.attempt_id,
                response_status=attempt.status.value, grading_status=_grade_label(grade),
                reason=(grade.failure_reason if grade and grade.failure_reason else (attempt.error or "")) or "",
                critical=critical, review_status=reviewed.get(case_id, "not reviewed"), provenance=provenance_label(run),
                response_excerpt=(attempt.response_text or "")[:400], error=attempt.error,
            ))

    def rows(dimension: str) -> list[BreakdownRow]:
        return [BreakdownRow(key, *values) for key, values in sorted(groups[dimension].items())]

    return RunResults(rows("domain"), rows("complexity"), rows("split"),
                      dict(Counter(a.status.value for a in latest.values())),
                      sum(a.status is AttemptStatus.REFUSAL for a in latest.values()), critical_unassessed, review,
                      sum(a.attempt_id not in grades for a in latest.values()))


@dataclass(frozen=True)
class ProviderStatus:
    provider: str
    model: str
    availability: str
    configuration: str
    capabilities: str
    eligibility: str
    measured_runs: int


def provider_statuses(runs: list[RunSummary]) -> list[ProviderStatus]:
    """Configuration state only; live endpoints are never probed from here."""
    seen = Counter((r.provider, r.model) for r in runs)
    rows = [ProviderStatus("fake", "synthetic-*", "available (offline, deterministic)", "built in",
                           "no streaming, no tools, pricing unknown", "offline/demo runs only; never real-model evidence",
                           sum(n for (p, _), n in seen.items() if p == "fake"))]
    ollama = bool(os.environ.get("MODELLAB_OLLAMA_ENDPOINT"))
    openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
    live_rule = "paid/live dispatch only via CLI `pilot run --allow-paid` with a verified immutable plan"
    rows.append(ProviderStatus("ollama", "per plan", "configured, not probed" if ollama else "unavailable",
                               "MODELLAB_OLLAMA_ENDPOINT set" if ollama else "MODELLAB_OLLAMA_ENDPOINT missing",
                               "unknown until measured", live_rule, sum(n for (p, _), n in seen.items() if p == "ollama")))
    rows.append(ProviderStatus("openrouter", "per plan", "configured, not probed" if openrouter else "unavailable",
                               "OPENROUTER_API_KEY present (value hidden)" if openrouter else "OPENROUTER_API_KEY missing",
                               "unknown until measured", live_rule, sum(n for (p, _), n in seen.items() if p == "openrouter")))
    for (provider, model), count in sorted(seen.items()):
        if provider not in {"fake", "ollama", "openrouter"}:
            rows.append(ProviderStatus(provider, model, "unknown", "seen in stored runs only", "unknown", "unknown", count))
    for (provider, model), count in sorted(seen.items()):
        if provider == "fake":
            rows.append(ProviderStatus(provider, model, "available (offline)", "synthetic model label", "deterministic fixture",
                                       "SIMULATED only", count))
    return rows


def environment_checks(suite_path: str | Path) -> dict[str, Any]:
    """The `model-lab doctor` checks (offline readiness)."""
    path = Path(suite_path)
    checks: dict[str, Any] = {"python_3_11_plus": sys.version_info >= (3, 11), "benchmark_present": path.is_file()}
    if checks["benchmark_present"]:
        checks["benchmark_cases"] = validate_suite(load_suite(path))["cases"]
    checks["offline_ready"] = all(value is True for key, value in checks.items() if key != "benchmark_cases") and checks.get("benchmark_cases") == 60
    return checks


@dataclass(frozen=True)
class ActiveRun:
    run_id: str
    model: str
    provider: str
    completed: int
    total: int
    failures: int
    retries: int
    elapsed_seconds: float | None
    current_operation: str
    current_case: str | None
    cost: CostSummary


@dataclass(frozen=True)
class Snapshot:
    data_dir: str
    db_path: str
    db_exists: bool
    suite: SuiteSummary | None
    suites: list[str]
    runs: list[RunSummary]
    providers: list[ProviderStatus]
    health: dict[str, Any]
    events: list[RunEvent]
    active: list[ActiveRun] = field(default_factory=list)
    error: str | None = None

    @property
    def completed_runs(self) -> int:
        return sum(r.status == "completed" for r in self.runs)

    @property
    def failed_runs(self) -> int:
        return sum(r.status in {"failed", "partial", "cancelled", "incomplete"} for r in self.runs)

    @property
    def review_pending(self) -> int:
        return sum(r.abstained + r.failed_grades + (r.cases_attempted - r.graded) for r in self.runs)


def _active(summary: RunSummary, events: list[RunEvent], now: datetime) -> ActiveRun:
    last = events[-1] if events else None
    started = next((e.timestamp for e in events if e.event_type == "run_started"), summary.started_at)
    try:
        elapsed = (now - datetime.fromisoformat(started)).total_seconds()
    except ValueError:
        elapsed = None
    return ActiveRun(summary.run_id, summary.model, summary.provider, summary.cases_attempted, summary.case_count,
                     summary.cases_failed, summary.retries, elapsed,
                     last.message if last else "waiting for first event", last.case_id if last else None, summary.cost)


def safe_environment_checks(suite_path: str | Path) -> dict[str, Any]:
    try:
        return environment_checks(suite_path)
    except ModelLabError as exc:
        return {"offline_ready": False, "error": str(exc)}


def _cached_summary(store: SQLiteStore, run_id: str, cache: dict[str, tuple[tuple[Any, ...], RunSummary]] | None) -> RunSummary:
    if cache is None:
        return summarize_run(store, run_id)
    last_event = store.list_events(run_id=run_id, limit=1)
    key = (store.get_run(run_id).status, store.count("attempts", run_id), store.count("grades", run_id),
           last_event[-1].event_id if last_event else 0)
    cached = cache.get(run_id)
    if cached is not None and cached[0] == key:
        return cached[1]
    summary = summarize_run(store, run_id)
    cache[run_id] = (key, summary)
    return summary


def open_store(data_dir: str | Path) -> SQLiteStore | None:
    """Open the workspace DB if it exists; never creates one on a read."""
    path = Path(data_dir) / DB_NAME
    return SQLiteStore(path) if path.is_file() else None


def load_snapshot(data_dir: str | Path, suite_path: str | Path, *, event_limit: int = 200, suite: SuiteSummary | None = None,
                  health: dict[str, Any] | None = None, summary_cache: dict[str, tuple[tuple[Any, ...], RunSummary]] | None = None) -> Snapshot:
    """Assemble the workspace view. Pollers pass ``suite``/``health`` (computed once)
    and a ``summary_cache`` so unchanged runs are not re-summarized on every tick."""
    data = Path(data_dir)
    db_path = data / DB_NAME
    if suite is None:
        suite = summarize_suite(suite_path) if Path(suite_path).is_file() else None
    suites = [str(p) for p in discover_suites(Path(suite_path).parent if Path(suite_path).parent.is_dir() else "benchmarks")]
    if health is None:
        health = safe_environment_checks(suite_path)
    store = open_store(data)
    if store is None:
        return Snapshot(str(data), str(db_path), False, suite, suites, [], provider_statuses([]), health, [])
    try:
        events = store.list_events(limit=event_limit)  # read before runs: run state is never older than the events shown
        runs = [_cached_summary(store, run_id, summary_cache) for run_id in store.list_run_ids()]
        now = datetime.now().astimezone()
        active = [_active(r, store.list_events(run_id=r.run_id, limit=20), now) for r in runs if r.status == "running"]
        return Snapshot(str(data), str(db_path), True, suite, suites, runs, provider_statuses(runs), health, events, active)
    except Exception as exc:  # corrupted/foreign DB: show it instead of crashing the console
        return Snapshot(str(data), str(db_path), True, suite, suites, [], provider_statuses([]), health, [], error=f"{type(exc).__name__}: {exc}")
    finally:
        store.close()
