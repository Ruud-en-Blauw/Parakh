"""SQLite persistence for ModelLab runs and immutable evidence."""

from __future__ import annotations

import json
import sqlite3
from functools import wraps
from pathlib import Path
from threading import RLock
from typing import Any

from model_lab.errors import IntegrityError, NotFoundError, ValidationError
from model_lab.schemas import Attempt, Budget, Grade, HumanReview, ProvenanceRecord, Run, RunEvent, canonical_record


def _json(value: Any) -> str:
    return canonical_record(value)


def synchronized(method):
    @wraps(method)
    def call(self, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)
    return call


class SQLiteStore:
    """Small transactional store; raw attempts are append-only by design."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA busy_timeout = 5000")
        if self.path != ":memory:":
            # WAL lets a reader (e.g. the TUI) poll while a run is writing without blocking it.
            self.connection.execute("PRAGMA journal_mode = WAL")
        self._lock = RLock()
        self._create_schema()

    @synchronized
    def close(self) -> None:
        self.connection.close()

    def _create_schema(self) -> None:
        self.connection.executescript("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY, suite_version TEXT NOT NULL, case_ids_json TEXT NOT NULL,
            model_config_json TEXT NOT NULL, seed INTEGER, budget_json TEXT NOT NULL,
            started_at TEXT NOT NULL, status TEXT NOT NULL, environment_json TEXT NOT NULL,
            created_record_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS attempts (
            attempt_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
            logical_request_id TEXT NOT NULL, case_id TEXT NOT NULL, prompt_hash TEXT NOT NULL,
            status TEXT NOT NULL, response_text TEXT, started_at TEXT NOT NULL, completed_at TEXT,
            model_config_json TEXT NOT NULL, evidence_json TEXT NOT NULL, raw_record_json TEXT NOT NULL
        );
        CREATE UNIQUE INDEX IF NOT EXISTS ux_attempt_logical ON attempts(run_id, logical_request_id, attempt_id);
        CREATE INDEX IF NOT EXISTS ix_attempt_run_case ON attempts(run_id, case_id);
        CREATE TABLE IF NOT EXISTS grades (
            grade_id TEXT PRIMARY KEY, attempt_id TEXT NOT NULL REFERENCES attempts(attempt_id),
            record_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_grade_attempt ON grades(attempt_id);
        CREATE TABLE IF NOT EXISTS reviews (
            review_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id), record_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_review_run_case ON reviews(run_id, json_extract(record_json, '$.case_id'));
        CREATE TABLE IF NOT EXISTS provenance (
            record_id TEXT PRIMARY KEY, record_json TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS budget_reservations (
            reservation_id TEXT PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(run_id),
            logical_request_id TEXT NOT NULL, request_count INTEGER NOT NULL, reserved_cost_minor INTEGER,
            actual_cost_minor INTEGER, currency TEXT, state TEXT NOT NULL, created_at TEXT NOT NULL,
            UNIQUE(run_id, logical_request_id)
        );
        CREATE INDEX IF NOT EXISTS ix_budget_run_state ON budget_reservations(run_id, state);
        CREATE TABLE IF NOT EXISTS quarantines (
            quarantine_id INTEGER PRIMARY KEY AUTOINCREMENT, source_label TEXT NOT NULL,
            reason TEXT NOT NULL, raw_record_json TEXT NOT NULL, created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS run_events (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, timestamp TEXT NOT NULL,
            event_type TEXT NOT NULL, record_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS ix_run_events_run ON run_events(run_id, event_id);
        """)
        for table in ("attempts", "grades", "reviews", "provenance", "quarantines", "run_events"):
            for operation in ("UPDATE", "DELETE"):
                self.connection.execute(f"CREATE TRIGGER IF NOT EXISTS immutable_{table}_{operation.lower()} BEFORE {operation} ON {table} BEGIN SELECT RAISE(ABORT, 'immutable evidence'); END")
        self.connection.execute("CREATE TRIGGER IF NOT EXISTS immutable_run_record BEFORE UPDATE OF created_record_json, case_ids_json, model_config_json, budget_json, environment_json ON runs BEGIN SELECT RAISE(ABORT, 'immutable run record'); END")

    def _begin(self) -> None:
        self.connection.execute("BEGIN IMMEDIATE")

    def create_run(self, run: Run) -> None:
        with self._lock:
            try:
                self.connection.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                    run.run_id, run.suite_version, _json(run.case_ids), _json(run.model_config), run.seed,
                    _json(run.budget), run.started_at, run.status, _json(run.environment), _json(run),
                ))
            except sqlite3.IntegrityError as exc:
                raise IntegrityError(f"run already exists: {run.run_id}") from exc

    @synchronized
    def get_run(self, run_id: str) -> Run:
        row = self.connection.execute("SELECT created_record_json, status FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"unknown run: {run_id}")
        from model_lab.schemas import ModelConfig
        value = json.loads(row["created_record_json"])
        return Run(run_id=value["run_id"], suite_version=value["suite_version"], case_ids=tuple(value["case_ids"]),
                   model_config=ModelConfig(**value["model_config"]), seed=value["seed"], budget=Budget(**value["budget"]),
                   started_at=value["started_at"], status=row["status"], environment=value["environment"], prompt_hashes=value.get("prompt_hashes", {}))

    @synchronized
    def update_run_status(self, run_id: str, status: str) -> None:
        if status not in {"created", "running", "completed", "partial", "failed", "cancelled", "incomplete"}:
            raise ValidationError("invalid run status")
        if self.connection.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is None:
            raise NotFoundError(f"unknown run: {run_id}")
        self.connection.execute("UPDATE runs SET status = ? WHERE run_id = ?", (status, run_id))

    @synchronized
    def claim_run(self, run: Run) -> None:
        if run.status != "created" or canonical_record(self.get_run(run.run_id)) != canonical_record(run):
            raise IntegrityError("run configuration changed or already executed; use a new run ID")
        if self.connection.execute("UPDATE runs SET status = 'running' WHERE run_id = ? AND status = 'created'", (run.run_id,)).rowcount != 1:
            raise IntegrityError("run is already claimed")

    def add_attempt(self, attempt: Attempt, *, raw_record: Any | None = None) -> None:
        with self._lock:
            run = self.get_run(attempt.run_id)
            if attempt.case_id not in run.case_ids or attempt.model_config != run.model_config:
                raise ValidationError("attempt case/model does not match immutable run")
            if run.prompt_hashes and run.prompt_hashes.get(attempt.case_id) != attempt.prompt_hash:
                raise ValidationError("attempt prompt hash does not match immutable run")
            raw = attempt if raw_record is None else raw_record
            try:
                self.connection.execute("INSERT INTO attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                    attempt.attempt_id, attempt.run_id, attempt.logical_request_id, attempt.case_id, attempt.prompt_hash,
                    attempt.status.value, attempt.response_text, attempt.started_at or "unknown", attempt.completed_at,
                    _json(attempt.model_config), _json(attempt), _json(raw),
                ))
            except sqlite3.IntegrityError as exc:
                raise IntegrityError(f"duplicate immutable attempt: {attempt.attempt_id}") from exc

    @synchronized
    def get_attempt(self, attempt_id: str) -> Attempt:
        row = self.connection.execute("SELECT evidence_json FROM attempts WHERE attempt_id = ?", (attempt_id,)).fetchone()
        if row is None:
            raise NotFoundError(f"unknown attempt: {attempt_id}")
        return self._attempt_from_json(json.loads(row["evidence_json"]))

    @synchronized
    def list_attempts(self, run_id: str) -> list[Attempt]:
        rows = self.connection.execute("SELECT evidence_json FROM attempts WHERE run_id = ? ORDER BY started_at, attempt_id", (run_id,)).fetchall()
        return [self._attempt_from_json(json.loads(row["evidence_json"])) for row in rows]

    @staticmethod
    def _attempt_from_json(value: dict[str, Any]) -> Attempt:
        from model_lab.schemas import MetricKind, ModelConfig, AttemptStatus
        value["model_config"] = ModelConfig(**value["model_config"])
        value["status"] = AttemptStatus(value["status"])
        value["metric_kinds"] = {k: MetricKind(v) for k, v in value.get("metric_kinds", {}).items()}
        return Attempt(**value)

    @synchronized
    def add_grade(self, grade: Grade) -> None:
        try:
            self.connection.execute("INSERT INTO grades VALUES (?, ?, ?)", (grade.grade_id, grade.attempt_id, _json(grade)))
        except sqlite3.IntegrityError as exc:
            raise IntegrityError(f"duplicate grade or unknown attempt: {grade.grade_id}") from exc

    @synchronized
    def add_review(self, review: HumanReview) -> None:
        if not review.attempt_id:
            raise ValidationError("review must bind to an exported attempt")
        attempt = self.get_attempt(review.attempt_id)
        if (review.run_id, review.case_id) != (attempt.run_id, attempt.case_id):
            raise ValidationError("review run/case does not match attempt")
        try:
            self.connection.execute("INSERT INTO reviews VALUES (?, ?, ?)", (review.review_id, review.run_id, _json(review)))
        except sqlite3.IntegrityError as exc:
            raise IntegrityError(f"duplicate review or unknown run: {review.review_id}") from exc

    @synchronized
    def list_reviews(self, run_id: str) -> list[HumanReview]:
        return [HumanReview(**json.loads(row[0])) for row in self.connection.execute("SELECT record_json FROM reviews WHERE run_id = ? ORDER BY review_id", (run_id,))]

    @synchronized
    def add_provenance(self, record: ProvenanceRecord) -> None:
        try:
            self.connection.execute("INSERT INTO provenance VALUES (?, ?)", (record.record_id, _json(record)))
        except sqlite3.IntegrityError as exc:
            raise IntegrityError(f"duplicate provenance record: {record.record_id}") from exc

    @synchronized
    def quarantine(self, source_label: str, reason: str, raw_record: Any, created_at: str) -> None:
        try:
            _json(raw_record)
        except (ValueError, TypeError):
            raw_record = {"unparseable_record_repr": repr(raw_record)}
        self.connection.execute("INSERT INTO quarantines(source_label, reason, raw_record_json, created_at) VALUES (?, ?, ?, ?)", (source_label, reason, _json(raw_record), created_at))

    @synchronized
    def quarantined(self) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT * FROM quarantines ORDER BY quarantine_id").fetchall()
        return [dict(row) for row in rows]

    @synchronized
    def add_event(self, event: RunEvent) -> int:
        record = {**json.loads(_json(event)), "event_id": None}
        cursor = self.connection.execute(
            "INSERT INTO run_events(run_id, timestamp, event_type, record_json) VALUES (?, ?, ?, ?)",
            (event.run_id, event.timestamp, event.event_type, json.dumps(record, sort_keys=True)),
        )
        assert cursor.lastrowid is not None
        return int(cursor.lastrowid)

    @synchronized
    def list_events(self, *, run_id: str | None = None, after_id: int = 0, limit: int = 500) -> list[RunEvent]:
        """Newest ``limit`` events with id > ``after_id``, returned oldest first."""
        where = "event_id > ?"
        params: list[Any] = [after_id]
        if run_id is not None:
            where += " AND run_id = ?"
            params.append(run_id)
        rows = self.connection.execute(
            f"SELECT event_id, record_json FROM run_events WHERE {where} ORDER BY event_id DESC LIMIT ?", (*params, limit),
        ).fetchall()
        return [RunEvent(**{**json.loads(row["record_json"]), "event_id": row["event_id"]}) for row in reversed(rows)]

    @synchronized
    def list_run_ids(self) -> list[str]:
        """All run IDs, most recently created first."""
        return [row[0] for row in self.connection.execute("SELECT run_id FROM runs ORDER BY rowid DESC")]

    @synchronized
    def list_grades(self, run_id: str) -> list[Grade]:
        rows = self.connection.execute(
            "SELECT grades.record_json FROM grades JOIN attempts ON attempts.attempt_id = grades.attempt_id WHERE attempts.run_id = ? ORDER BY grades.grade_id",
            (run_id,),
        )
        return [Grade(**json.loads(row[0])) for row in rows]

    @synchronized
    def count(self, table: str, run_id: str | None = None) -> int:
        if table not in {"attempts", "grades", "reviews", "budget_reservations"}:
            raise ValidationError("unsupported count table")
        if run_id is None:
            return int(self.connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
        if table == "grades":
            return int(self.connection.execute("SELECT COUNT(*) FROM grades JOIN attempts ON attempts.attempt_id = grades.attempt_id WHERE attempts.run_id = ?", (run_id,)).fetchone()[0])
        return int(self.connection.execute(f"SELECT COUNT(*) FROM {table} WHERE run_id = ?", (run_id,)).fetchone()[0])

    def reserve_budget(self, reservation_id: str, run_id: str, logical_request_id: str, max_requests: int | None, max_cost_minor: int | None, estimated_cost_minor: int | None, currency: str | None, created_at: str, max_cases: int | None = None) -> None:
        Budget(max_requests=max_requests, max_cost_minor=max_cost_minor, currency=currency, max_cases=max_cases)
        if estimated_cost_minor is not None and (type(estimated_cost_minor) is not int or estimated_cost_minor < 0):
            raise ValidationError("estimated cost must be a non-negative integer")
        with self._lock:
            self._begin()
            try:
                if self.connection.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is None:
                    raise NotFoundError(f"unknown run: {run_id}")
                persisted = self.get_run(run_id).budget
                def tighter(a, b):
                    return min(x for x in (a, b) if x is not None) if a is not None or b is not None else None
                max_requests, max_cases = tighter(max_requests, persisted.max_requests), tighter(max_cases, persisted.max_cases)
                max_cost_minor = tighter(max_cost_minor, persisted.max_cost_minor)
                if persisted.currency and currency != persisted.currency:
                    raise ValidationError("reservation currency differs from run")
                if self.connection.execute("SELECT 1 FROM budget_reservations WHERE run_id = ? AND logical_request_id = ?", (run_id, logical_request_id)).fetchone():
                    raise IntegrityError("budget reservation already exists for logical request")
                active = self.connection.execute("SELECT COUNT(*), COALESCE(SUM(reserved_cost_minor), 0) FROM budget_reservations WHERE run_id = ? AND state != 'released'", (run_id,)).fetchone()
                request_limit = min(x for x in (max_requests, max_cases) if x is not None) if any(x is not None for x in (max_requests, max_cases)) else None
                if request_limit is not None and active[0] >= request_limit:
                    raise ValidationError("request budget exhausted")
                if max_cost_minor is not None and estimated_cost_minor is None:
                    raise ValidationError("finite cost budget requires a known reservation")
                reserved_cost = self.connection.execute("""
                    SELECT COALESCE(SUM(CASE
                        WHEN state = 'released' THEN 0
                        WHEN state = 'settled' THEN COALESCE(actual_cost_minor, reserved_cost_minor)
                        ELSE reserved_cost_minor
                    END), 0)
                    FROM budget_reservations WHERE run_id = ?
                """, (run_id,)).fetchone()[0]
                if max_cost_minor is not None and reserved_cost + estimated_cost_minor > max_cost_minor:
                    raise ValidationError("cost budget exhausted")
                self.connection.execute("INSERT INTO budget_reservations VALUES (?, ?, ?, 1, ?, NULL, ?, 'reserved', ?)", (reservation_id, run_id, logical_request_id, estimated_cost_minor, currency, created_at))
                self.connection.execute("COMMIT")
            except Exception:
                self.connection.execute("ROLLBACK")
                raise

    def settle_budget(self, reservation_id: str, actual_cost_minor: int | None, state: str = "settled") -> None:
        if actual_cost_minor is not None and (type(actual_cost_minor) is not int or actual_cost_minor < 0):
            raise ValidationError("actual cost must be a non-negative integer")
        if (state == "settled") != (actual_cost_minor is not None):
            raise ValidationError("settled cost must be known; unknown/released cost must be null")
        with self._lock:
            if state not in {"settled", "released", "unknown"}:
                raise ValidationError("invalid budget settlement state")
            updated = self.connection.execute("UPDATE budget_reservations SET actual_cost_minor = ?, state = ? WHERE reservation_id = ? AND state = 'reserved'", (actual_cost_minor, state, reservation_id)).rowcount
            if updated != 1:
                raise NotFoundError(f"unknown or already settled reservation: {reservation_id}")

    def close_without_delete(self) -> None:
        """Explicitly named hook for callers that must not destroy evidence."""
        self.close()
