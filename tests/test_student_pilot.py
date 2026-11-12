"""Offline proofs for the capped, resumable student-set pilot.

Only fake providers and a fake transport are used. Nothing here reads a real
``.env*`` file or opens a socket; the OpenRouter key is a synthetic value.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from model_lab import pilot
from model_lab.benchmark import load_suite
from model_lab.cli import main
from model_lab.constraints import file_sha256
from model_lab.errors import PilotBlockedError
from model_lab.operator_input import OperatorInputError, build_immutable_plan, validate_operator_input
from model_lab.providers.base import ProviderCapabilities, ProviderConfigurationError, ProviderResponse
from model_lab.storage import SQLiteStore
from tests.test_operator_input import complete_input


ROOT = Path(__file__).parents[1]
STUDENT = ROOT / "benchmarks_student" / "cases.jsonl"
CONSTRAINTS = ROOT / "docs/live_pilots/router-constraint-map-v0.2.0.json"
MANIFEST = ROOT / "docs/product_sources/SOURCE_MANIFEST.json"
FREE_A = "openrouter:cohere/north-mini-code:free"
FREE_B = "openrouter:liquid/lfm-2.5-2.6b:free"
SYNTHETIC_KEY = "sk-or-SYNTHETIC-TEST-KEY-do-not-use-0000"


def _train_ids(count: int) -> list[str]:
    return [case.case_id for case in load_suite(STUDENT).cases if case.split == "train"][:count]


def student_input(*, case_ids=None, split="train", repeats=1, retry_limit=0, identifiers=(FREE_A, FREE_B)) -> dict:
    value = complete_input()
    value["suite_kind"] = "student"
    value["candidates"] = []
    zero = {"input_per_million": 0, "output_per_million": 0, "cache_per_million": 0, "reasoning_per_million": 0, "tool_per_call": 0}
    value["pricing_snapshot"]["rates"] = {}
    value["budget"]["per_model_max_spend"] = {}
    for identifier in identifiers:
        provider, model, *revision = identifier.split(":")
        value["candidates"].append({"identifier": identifier, "provider": provider, "model": model, "display_label": identifier, "revision": revision[0] if revision else ""})
        value["pricing_snapshot"]["rates"][identifier] = dict(zero)
        value["budget"]["per_model_max_spend"][identifier] = 5
    value["execution"].update({"repeats": repeats, "retry_limit": retry_limit})
    value["case_selection"] = {"split": split, "case_ids": list(case_ids or [])}
    return value


def build_student_plan(value: dict, **overrides) -> dict:
    return build_immutable_plan(
        value, suite_path=STUDENT, source_hash=file_sha256(MANIFEST), constraint_hash=file_sha256(CONSTRAINTS),
        source_manifest_path=str(MANIFEST), constraint_map_path=str(CONSTRAINTS), created_at="2026-11-01T10:00:00+00:00", **overrides,
    )


def run_pilot(plan: dict, out: Path, **kwargs) -> dict:
    return pilot.run_authorized_immutable_pilot(
        plan, suite_path=STUDENT, output_dir=out, source_manifest_path=str(MANIFEST), constraint_map_path=str(CONSTRAINTS), **kwargs,
    )


class FakeProvider:
    """Synthetic provider: records every case it is asked about, returns a fixed answer."""

    name = "openrouter"

    def __init__(self) -> None:
        self.capabilities = ProviderCapabilities("openrouter", "fake")
        self.calls: list[str] = []

    def generate(self, request) -> ProviderResponse:
        self.calls.append(request.candidate.case_id)
        return ProviderResponse(text="synthetic answer", finish_reason="stop", input_tokens=1, output_tokens=1, cost_minor=0, currency="USD", metadata={})


@pytest.fixture
def fakes(monkeypatch) -> dict[str, FakeProvider]:
    providers = {FREE_A: FakeProvider(), FREE_B: FakeProvider()}
    monkeypatch.setattr(pilot, "_provider_for", lambda candidate, **_: providers[candidate["identifier"]])
    return providers


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def read(self, limit: int = -1) -> bytes:
        return self._body if limit < 0 else self._body[:limit]


class FakeTransport:
    """Stands in for urllib's opener: records requests and never touches the network."""

    def __init__(self) -> None:
        self.calls = 0
        self.authorization_headers: list[str | None] = []

    def open(self, req, timeout=None):
        self.calls += 1
        self.authorization_headers.append(req.get_header("Authorization"))
        body = {"id": "gen-test", "model": "synthetic", "choices": [{"message": {"content": "synthetic answer"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "cost": 0}}
        return _FakeResponse(json.dumps(body).encode("utf-8"))


ZERO_CATALOG = [
    {"id": "cohere/north-mini-code:free", "pricing": {"prompt": "0", "completion": "0"}},
    {"id": "liquid/lfm-2.5-2.6b:free", "pricing": {"prompt": "0", "completion": "0"}},
]


# --- explicit suite_kind and the core 36-case gate -------------------------------------------

def test_student_path_requires_explicit_suite_kind_and_core_gate_stays_strict():
    # Three candidates so the core path reaches its 36-case gate instead of the candidate-count check.
    no_kind = student_input(case_ids=_train_ids(4), identifiers=(FREE_A, FREE_B, "openrouter:vendor/model-c:free"))
    del no_kind["suite_kind"]
    with pytest.raises(OperatorInputError, match="36"):
        validate_operator_input(no_kind, suite_path=STUDENT)

    core = complete_input()
    core["case_selection"]["case_ids"] = _train_ids(4)
    with pytest.raises(OperatorInputError, match="36"):
        build_immutable_plan(core, suite_path=STUDENT)

    with pytest.raises(OperatorInputError, match="mismatch"):
        validate_operator_input(student_input(case_ids=_train_ids(4)), suite_path=STUDENT, suite_kind="core")


def test_student_selection_is_train_and_calibration_never_holdout():
    plan = build_student_plan(student_input(split="train+calibration"))
    assert plan["suite_kind"] == "student"
    assert len(plan["case_ids"]) == 40
    splits = {case.split for case in load_suite(STUDENT).cases if case.case_id in set(plan["case_ids"])}
    assert splits == {"train", "calibration"}

    holdout_id = next(case.case_id for case in load_suite(STUDENT).cases if case.split == "holdout")
    with pytest.raises(OperatorInputError, match="holdout"):
        validate_operator_input(student_input(case_ids=[holdout_id]), suite_path=STUDENT)
    with pytest.raises(OperatorInputError, match="2 to 5"):
        validate_operator_input(student_input(identifiers=(FREE_A,)), suite_path=STUDENT)


def test_core_plan_shape_with_student_cases_is_blocked_at_dispatch():
    plan = build_student_plan(student_input(split="train+calibration"))
    plan["suite_kind"] = "core"
    with pytest.raises(PilotBlockedError, match="36"):
        pilot.require_dispatch_authorization(plan, allow_paid=True, source_manifest_path=MANIFEST, constraint_map_path=CONSTRAINTS)


# --- request cap, clean stop, resume ----------------------------------------------------------

def test_request_cap_stops_cleanly_as_incomplete_then_resumes_without_duplicate_calls(tmp_path: Path, fakes):
    plan = build_student_plan(student_input(case_ids=_train_ids(4), retry_limit=1))

    first = run_pilot(plan, tmp_path, max_requests_per_day=8)
    # Candidate A finishes (4 calls). Candidate B stops before its 4th case: the next case
    # would need its full retry budget (2) on top of 7 dispatches, which exceeds the cap of 8.
    assert first["dispatch_requests"] == 7
    by_candidate = {row["candidate"]: row for row in first["results"]}
    assert by_candidate[FREE_A]["status"] == "completed"
    assert by_candidate[FREE_B]["status"] == "incomplete"
    assert by_candidate[FREE_B]["attempts"] == 3

    store = SQLiteStore(tmp_path / "model_lab.sqlite3")
    try:
        incomplete_run = by_candidate[FREE_B]["run_id"]
        assert any(event.event_type == "cap_reached" for event in store.list_events(run_id=incomplete_run))
    finally:
        store.close()

    second = run_pilot(plan, tmp_path)
    resumed = {row["candidate"]: row for row in second["results"]}
    assert resumed[FREE_A]["reused"] is True and resumed[FREE_A]["attempts"] == 0
    assert resumed[FREE_B]["reused"] is True and resumed[FREE_B]["pending_cases"] == 1
    assert resumed[FREE_B]["status"] == "completed"
    assert second["dispatch_requests"] == 1

    assert sorted(fakes[FREE_A].calls) == sorted(_train_ids(4))
    assert sorted(fakes[FREE_B].calls) == sorted(_train_ids(4))
    assert len(fakes[FREE_A].calls) == len(set(fakes[FREE_A].calls)) == 4
    assert len(fakes[FREE_B].calls) == len(set(fakes[FREE_B].calls)) == 4


def test_start_is_refused_with_math_when_pending_requests_exceed_the_cap(tmp_path: Path, fakes):
    plan = build_student_plan(student_input(case_ids=_train_ids(4)))
    with pytest.raises(PilotBlockedError, match=r"4 \+ 4 = 8 exceed max_requests_per_day=7"):
        run_pilot(plan, tmp_path, max_requests_per_day=7)
    assert fakes[FREE_A].calls == [] and fakes[FREE_B].calls == []


def test_completed_run_is_reused_without_any_provider_call(tmp_path: Path, fakes):
    plan = build_student_plan(student_input(case_ids=_train_ids(4)))
    run_pilot(plan, tmp_path)
    calls_before = (len(fakes[FREE_A].calls), len(fakes[FREE_B].calls))

    again = run_pilot(plan, tmp_path)
    assert all(row["reused"] is True and row["status"] == "completed" for row in again["results"])
    assert again["dispatch_requests"] == 0 and again["dispatch_performed"] is False
    assert (len(fakes[FREE_A].calls), len(fakes[FREE_B].calls)) == calls_before


def test_plan_hash_mismatch_on_resume_is_refused(tmp_path: Path, fakes):
    original = build_student_plan(student_input(case_ids=_train_ids(4)))
    run_pilot(original, tmp_path)
    calls_before = (len(fakes[FREE_A].calls), len(fakes[FREE_B].calls))

    changed_input = student_input(case_ids=_train_ids(4))
    changed_input["prompt_template_revision"] = "candidate-v2"  # different frozen plan, same runs on disk
    changed = build_student_plan(changed_input)
    assert changed["plan_hash"] != original["plan_hash"]
    with pytest.raises(PilotBlockedError, match="plan hash mismatch on resume"):
        run_pilot(changed, tmp_path)
    assert (len(fakes[FREE_A].calls), len(fakes[FREE_B].calls)) == calls_before


def test_core_gate_is_enforced_by_the_runner_for_non_student_plans(tmp_path: Path):
    plan = build_student_plan(student_input(case_ids=_train_ids(4)))
    plan["suite_kind"] = "core"
    with pytest.raises(PilotBlockedError):
        run_pilot(plan, tmp_path)
    assert not (tmp_path / "model_lab.sqlite3").exists()


# --- zero-price check and API key handling, through the real OpenRouter adapter -----------------

def test_nonzero_catalog_price_aborts_before_any_execution(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", SYNTHETIC_KEY)
    transport = FakeTransport()
    catalog = [
        {"id": "cohere/north-mini-code:free", "pricing": {"prompt": "0.000001", "completion": "0"}},
        {"id": "liquid/lfm-2.5-2.6b:free", "pricing": {"prompt": "0", "completion": "0"}},
    ]
    plan = build_student_plan(student_input(case_ids=_train_ids(4)))
    with pytest.raises(PilotBlockedError, match="non-zero"):
        run_pilot(plan, tmp_path, catalog_models=catalog, openrouter_opener=transport)
    assert transport.calls == 0
    assert not (tmp_path / "model_lab.sqlite3").exists()


def test_snapshot_style_catalog_price_is_checked_too(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", SYNTHETIC_KEY)
    catalog = [{"id": "cohere/north-mini-code:free", "prompt_price_usd": "0.5", "completion_price_usd": "0"}]
    plan = build_student_plan(student_input(case_ids=_train_ids(4)))
    with pytest.raises(PilotBlockedError, match="non-zero"):
        run_pilot(plan, tmp_path, catalog_models=catalog, openrouter_opener=FakeTransport())


def test_api_key_never_appears_in_outputs_artifacts_or_repr(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", SYNTHETIC_KEY)
    transport = FakeTransport()
    plan = build_student_plan(student_input(case_ids=_train_ids(2)))

    result = run_pilot(plan, tmp_path, catalog_models=ZERO_CATALOG, openrouter_opener=transport)

    assert transport.calls == 4  # 2 cases x 2 free candidates, through the real adapter
    assert set(transport.authorization_headers) == {f"Bearer {SYNTHETIC_KEY}"}  # key is used...
    assert SYNTHETIC_KEY not in json.dumps(result)  # ...but never echoed
    for artifact in tmp_path.rglob("*"):
        if artifact.is_file():
            assert SYNTHETIC_KEY.encode() not in artifact.read_bytes(), artifact.name

    adapter = pilot._provider_for(plan["candidates"][0], catalog_models=ZERO_CATALOG, openrouter_opener=transport)
    assert SYNTHETIC_KEY not in repr(adapter)
    assert SYNTHETIC_KEY not in str(adapter)


def test_missing_key_blocks_before_execution(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def no_file_key(*_args, **_kwargs):
        raise ProviderConfigurationError("OpenRouter key not found. Key must be configured strictly in .env.openrouter")

    monkeypatch.setattr("model_lab.providers.openrouter.load_openrouter_api_key", no_file_key)
    transport = FakeTransport()
    plan = build_student_plan(student_input(case_ids=_train_ids(2)))
    with pytest.raises(PilotBlockedError, match="key not found"):
        run_pilot(plan, tmp_path, catalog_models=ZERO_CATALOG, openrouter_opener=transport)
    assert transport.calls == 0
    assert not (tmp_path / "model_lab.sqlite3").exists()


def test_paid_openrouter_models_are_refused_by_the_provider_factory(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", SYNTHETIC_KEY)
    transport = FakeTransport()
    with pytest.raises(PilotBlockedError, match="not a :free model"):
        pilot._provider_for({"provider": "openrouter", "model": "openai/gpt-4.1", "revision": ""}, openrouter_opener=transport)
    assert transport.calls == 0


# --- CLI wiring -----------------------------------------------------------------------------------

def test_cli_suite_kind_and_day_cap_flags_are_wired(tmp_path: Path, monkeypatch):
    operator = tmp_path / "student-input.json"
    operator.write_text(json.dumps(student_input(split="train+calibration")), encoding="utf-8")
    assert main(["pilot", "validate-input", "--input", str(operator), "--suite", str(STUDENT)]) == 0
    assert main(["pilot", "validate-input", "--input", str(operator), "--suite", str(STUDENT), "--suite-kind", "student"]) == 0
    assert main(["pilot", "validate-input", "--input", str(operator), "--suite", str(STUDENT), "--suite-kind", "core"]) != 0

    plan_file = tmp_path / "plan.json"
    assert main(["pilot", "plan", "--operator-input", str(operator), "--suite", str(STUDENT),
                 "--constraint-map", str(CONSTRAINTS), "--out", str(plan_file)]) == 0
    assert json.loads(plan_file.read_text(encoding="utf-8"))["suite_kind"] == "student"

    def forbidden(*_args, **_kwargs):
        raise AssertionError("no provider may be built without --allow-paid")

    monkeypatch.setattr(pilot, "_provider_for", forbidden)
    assert main(["pilot", "run", "--plan", str(plan_file), "--suite", str(STUDENT), "--out", str(tmp_path / "out"),
                 "--max-requests-per-day", "5", "--source-manifest", str(MANIFEST), "--constraint-map", str(CONSTRAINTS)]) != 0
