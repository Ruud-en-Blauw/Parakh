"""Bounded candidate execution with durable attempts and recovery state."""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Iterable

from model_lab.benchmark import select_cases
from model_lab.domain.budget import BudgetLedger
from model_lab.errors import BudgetExceededError, ValidationError
from model_lab.application.events import record_event
from model_lab.domain.isolation import candidate_input
from model_lab.providers.base import Provider, ProviderRequest, ProviderRuntimeError
from model_lab.schemas import Attempt, AttemptStatus, Run, Suite, utc_now
from model_lab.storage.sqlite import SQLiteStore


@dataclass(frozen=True)
class ExecutionResult:
    run_id: str
    status: str
    attempts: tuple[Attempt, ...]
    failures: int


def _id(prefix: str, *parts: str) -> str:
    return f"{prefix}-" + hashlib.sha256(":".join(parts).encode()).hexdigest()[:32]


class ExecutionEngine:
    def __init__(self, store: SQLiteStore, provider: Provider, *, max_retries: int = 0,
                 retry_delay_seconds: float = 0.0, estimated_cost_minor_per_logical_request: int | None = None,
                 provider_timeout_seconds: float | None = None, cap_checker: Callable[[], bool] | None = None) -> None:
        if max_retries < 0:
            raise ValidationError("max_retries must be non-negative")
        if estimated_cost_minor_per_logical_request is not None and estimated_cost_minor_per_logical_request < 0:
            raise ValidationError("estimated_cost_minor_per_logical_request must be non-negative")
        if provider_timeout_seconds is not None and provider_timeout_seconds <= 0:
            raise ValidationError("provider_timeout_seconds must be positive")
        self.store, self.provider = store, provider
        self.max_retries, self.retry_delay_seconds = max_retries, retry_delay_seconds
        self.estimated_cost_minor_per_logical_request = estimated_cost_minor_per_logical_request
        self.provider_timeout_seconds = provider_timeout_seconds
        self.cap_checker = cap_checker
        self.budgets = BudgetLedger(store)

    def _is_cap_reached(self) -> bool:
        if self.cap_checker is not None and self.cap_checker():
            return True
        # Capped adapters (providers/openrouter.py) expose max_requests/request_count; the
        # Provider protocol does not, so read them defensively.
        max_requests = getattr(self.provider, "max_requests", None)
        return isinstance(max_requests, int) and getattr(self.provider, "request_count", 0) >= max_requests

    def execute(self, suite: Suite, run: Run, *, case_ids: Iterable[str] | None = None, splits: Iterable[str] | None = None, domains: Iterable[str] | None = None, workflows: Iterable[str] | None = None, complexity_levels: Iterable[int] | None = None, cancel_event: threading.Event | None = None) -> ExecutionResult:
        selected = select_cases(suite, case_ids=case_ids or run.case_ids, splits=splits, domains=domains, workflows=workflows, complexity_levels=complexity_levels)
        selected_ids = {case.case_id for case in selected}
        if not selected_ids.issubset(set(run.case_ids)):
            raise ValidationError("selection contains cases outside the run")
        try:
            self.store.get_run(run.run_id)
        except Exception:
            self.store.create_run(run)
        self.store.update_run_status(run.run_id, "running")
        ctx = {"run_id": run.run_id, "model": run.model_config.model, "provider": run.model_config.provider}
        total = len(selected)
        record_event(self.store, "run_started", f"run started: {total} cases", status="running", **ctx)
        failures = 0
        produced: list[Attempt] = []
        run_started = time.monotonic()
        for index, case in enumerate(selected, 1):
            progress = f"case {index}/{total}"
            if cancel_event and cancel_event.is_set():
                self.store.update_run_status(run.run_id, "cancelled")
                record_event(self.store, "run_cancelled", f"cancelled before {progress}", status="cancelled", **ctx)
                break
            if run.budget.max_runtime_seconds is not None and time.monotonic() - run_started >= run.budget.max_runtime_seconds:
                self.store.update_run_status(run.run_id, "partial")
                record_event(self.store, "runtime_limit_reached", f"max_runtime_seconds reached before {progress}", status="partial", **ctx)
                break
            if self._is_cap_reached():
                self.store.update_run_status(run.run_id, "incomplete")
                record_event(self.store, "cap_reached", f"request cap reached before {progress}", status="incomplete", **ctx)
                break
            logical = _id("req", run.run_id, case.case_id)
            record_event(self.store, "case_started", f"{progress} started", case_id=case.case_id, **ctx)
            try:
                reservation = self.budgets.reserve(
                    run.run_id,
                    logical,
                    run.budget,
                    estimated_cost_minor=self.estimated_cost_minor_per_logical_request if run.budget.max_cost_minor is not None else None,
                )
            except BudgetExceededError as exc:
                failures += 1
                self.store.update_run_status(run.run_id, "partial")
                record_event(self.store, "budget_exhausted", f"budget exhausted at {progress}", status="partial", case_id=case.case_id, error=str(exc), **ctx)
                break
            logical_attempts: list[Attempt] = []
            for retry in range(self.max_retries + 1):
                attempt_id = _id("att", run.run_id, case.case_id, str(retry))
                started = utc_now()
                status = AttemptStatus.SUCCESS
                response_text = None
                error = None
                finish_reason = None
                metadata: dict[str, object] = {}
                input_tokens = output_tokens = cost_minor = None
                currency = first_latency = completion_latency = None
                record_event(self.store, "provider_request_started", f"{progress} provider request (attempt {retry + 1})", case_id=case.case_id, **ctx)
                call_started = time.monotonic()
                try:
                    response = self.provider.generate(ProviderRequest(candidate_input(case), run.model_config.model, run.model_config.parameters, self.provider_timeout_seconds or run.budget.max_runtime_seconds, run.seed))
                    response_text, finish_reason = response.text, response.finish_reason
                    input_tokens, output_tokens, cost_minor, currency = response.input_tokens, response.output_tokens, response.cost_minor, response.currency
                    first_latency, completion_latency, metadata = response.first_token_latency_ms, response.completion_latency_ms, dict(response.metadata)
                    if response_text is None:
                        status = AttemptStatus.INVALID_PROVIDER_RESPONSE
                        error = "provider returned no response text"
                    elif response_text == "":
                        status = AttemptStatus.MALFORMED_OUTPUT
                        error = "provider returned empty output"
                except TimeoutError as exc:
                    status, error = AttemptStatus.TIMEOUT, str(exc)
                except ProviderRuntimeError as exc:
                    status, error = AttemptStatus.PROVIDER_FAILURE, str(exc)
                except Exception as exc:
                    status, error = AttemptStatus.INFRASTRUCTURE_FAILURE, str(exc)
                completed = utc_now()
                attempt = Attempt(attempt_id=attempt_id, run_id=run.run_id, logical_request_id=logical, case_id=case.case_id, model_config=run.model_config, prompt_hash=case.prompt_hash, response_text=response_text, status=status, started_at=started, completed_at=completed, finish_reason=finish_reason, error=error, first_token_latency_ms=first_latency, completion_latency_ms=completion_latency, input_tokens=input_tokens, output_tokens=output_tokens, cost_minor=cost_minor, currency=currency, raw_metadata=metadata)
                self.store.add_attempt(attempt)
                record_event(self.store, "response_received", f"{progress} response: {status.value}", status=status.value, case_id=case.case_id,
                             duration_ms=round((time.monotonic() - call_started) * 1000, 3), cost_minor=cost_minor, currency=currency, error=error, artifact=attempt_id, **ctx)
                produced.append(attempt)
                logical_attempts.append(attempt)
                if status is AttemptStatus.SUCCESS or status not in {AttemptStatus.PROVIDER_FAILURE, AttemptStatus.TIMEOUT} or retry == self.max_retries:
                    if status is not AttemptStatus.SUCCESS:
                        failures += 1
                    record_event(self.store, "case_completed" if status is AttemptStatus.SUCCESS else "case_failed", f"{progress} {status.value}", status=status.value, case_id=case.case_id, **ctx)
                    break
                record_event(self.store, "retry_scheduled", f"{progress} retry {retry + 2}/{self.max_retries + 1} after {status.value}", status=status.value, case_id=case.case_id, **ctx)
                if self.retry_delay_seconds:
                    time.sleep(self.retry_delay_seconds)
            # A timeout/failure can still be billable. Never settle using just
            # the final retry: all retry costs must be known before an exact
            # settlement can replace the conservative reservation.
            retry_costs = [attempt.cost_minor for attempt in logical_attempts]
            actual_cost = (
                sum(cost for cost in retry_costs if cost is not None)
                if retry_costs and all(cost is not None for cost in retry_costs)
                else None
            )
            self.budgets.settle(reservation, actual_cost_minor=actual_cost)
            last_attempt = logical_attempts[-1] if logical_attempts else None
            if last_attempt and last_attempt.status is AttemptStatus.PROVIDER_FAILURE and last_attempt.error:
                err_text = last_attempt.error
                if any(m in err_text for m in ("hard request cap", "ABORTED", "non-zero price")):
                    self.store.update_run_status(run.run_id, "incomplete")
                    record_event(self.store, "run_stopped", f"run stopped cleanly on cap/abort: {err_text}", status="incomplete", **ctx)
                    break
        current = self.store.get_run(run.run_id)
        if current.status == "running":
            self.store.update_run_status(run.run_id, "partial" if failures else "completed")
        final = self.store.get_run(run.run_id).status
        record_event(self.store, "run_finished", f"run {final}: {len(produced)} attempts, {failures} failed", status=final,
                     duration_ms=round((time.monotonic() - run_started) * 1000, 3), **ctx)
        return ExecutionResult(run.run_id, final, tuple(produced), failures)
