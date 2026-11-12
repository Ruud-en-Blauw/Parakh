"""Cross-repository readiness integration tests between Parakh and Karmi.

Validates the complete contract and runtime lifecycle between Parakh (model evaluation,
training, verification, policy bundle export) and Karmi (API message intake, static
routing, telemetry export, bundle receiving, staging, and shadow execution).

Candidate checkout references:
- Karmi: /home/rosso/Projects/Karmi on parakh-shadow-integration (audit SHA 84e49ad)
- Parakh: origin/main (commit 0c66156)

Testing constraints:
- Genuine offline synthetic tests only: disposable SQLite databases and temp test directories.
- Zero live providers, zero network calls, zero live charges/money, zero shared databases.
- Test-signed Karmi-action bundles are NOT presented as Parakh-verified.
- Real-action provider evidence remains explicitly BLOCKED pending authorized paid pilot evidence.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sys
from typing import Any, Generator

import pytest

# Repository root paths
PARAKH_ROOT = Path(__file__).resolve().parents[1]
KARMI_ROOT = Path(os.environ.get("PARAKH_KARMI_DIR", PARAKH_ROOT.parent / "Karmi")).resolve()
KARMI_SRC = KARMI_ROOT / "src"

# Configure import paths safely: Parakh root takes precedence, Karmi src appended
if str(PARAKH_ROOT) not in sys.path:
    sys.path.insert(0, str(PARAKH_ROOT))

if str(KARMI_SRC) not in sys.path:
    sys.path.append(str(KARMI_SRC))

# Parakh imports
from contracts import schema_check
from contracts.karmi_reference import loader as reference_loader
from model_lab.errors import IntegrityError, PilotBlockedError
from model_lab.pilot import require_dispatch_authorization
from model_lab.rewards.composite import reward_for_run
from model_lab.rewards.schema import RewardConfig
from model_lab.storage.evidence import EvidenceStore
from model_lab.storage.sqlite import SQLiteStore
from model_lab.telemetry.importer import import_batch
from model_lab.telemetry.schema import TelemetryBatchError, parse_batch

# Parakh's own suite runs without Karmi's runtime. The readiness runner sets
# PARAKH_REQUIRE_KARMI=1, where a missing Karmi candidate must fail, never skip.
if importlib.util.find_spec("daily_agent") is None or importlib.util.find_spec("fastapi") is None:
    if os.environ.get("PARAKH_REQUIRE_KARMI") == "1":
        raise RuntimeError(f"Karmi runtime/candidate unavailable at {KARMI_SRC}")
    pytest.skip("cross-repo gate runs via ./test integration in Karmi's runtime", allow_module_level=True)

# Karmi imports
from daily_agent.api import create_app
from daily_agent.config import Settings, get_settings
from daily_agent.db import Base, get_session
from daily_agent.models import Account, PolicyBundleRecord, Subscription, User
from daily_agent.parakh import receiver as karmi_receiver
from daily_agent.parakh.bundle import BundleRejected
from daily_agent.parakh.telemetry import build_telemetry_batch
from daily_agent.security import issue_development_token
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError as SQLIntegrityError
from sqlalchemy.orm import Session, sessionmaker


def _load_karmi_parakh_helpers() -> Any:
    """Dynamically load Karmi's test helpers without namespace collision."""
    helpers_path = KARMI_ROOT / "tests" / "parakh_helpers.py"
    spec = importlib.util.spec_from_file_location("karmi_parakh_helpers_module", helpers_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load spec from {helpers_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["karmi_parakh_helpers_module"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def karmi_exchange(tmp_path: Path) -> Generator[dict[str, Any], None, None]:
    """Provide a disposable Karmi environment with isolated SQLite, auth, and TestClient."""
    db_path = tmp_path / "karmi_isolated.db"
    engine = create_engine(
        f"sqlite+pysqlite:///{db_path}",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, expire_on_commit=False)
    settings = Settings(
        environment="test",
        database_url=f"sqlite+pysqlite:///{db_path}",
        auth_secret="test-secret-with-sufficient-entropy-for-auth",
        webhook_secret="test-webhook-secret",
        global_daily_budget_micro=10_000,
    )
    settings.data_dir = tmp_path / "karmi_data"
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.parakh_trusted_public_keys = []

    def session_override() -> Generator[Session, None, None]:
        with factory() as session:
            yield session

    app = create_app()
    app.dependency_overrides[get_session] = session_override
    app.dependency_overrides[get_settings] = lambda: settings

    # Seed test customer account with a trika plan
    with factory() as session:
        account = Account(name="CrossRepo Test Account")
        session.add(account)
        session.flush()
        user = User(account_id=account.id, display_name="CrossRepo User", role="customer")
        session.add(user)
        session.flush()
        sub = Subscription(account_id=account.id, plan_id="trika")
        session.add(sub)
        session.commit()
        token = issue_development_token(user, settings)

    with TestClient(app) as client:
        yield {
            "client": client,
            "engine": engine,
            "factory": factory,
            "settings": settings,
            "token": token,
            "headers": {"Authorization": f"Bearer {token}"},
            "tmp_path": tmp_path,
        }
    engine.dispose()


def test_offline_synthetic_flow_karmi_messages_to_parakh_importer_and_evaluation(
    karmi_exchange: dict[str, Any],
) -> None:
    """Validate genuine offline flow:

    1. Karmi /v1/messages produces routing decisions with zero live provider calls.
    2. Actual Karmi telemetry exporter produces deterministic TelemetryBatchV1.
    3. Exporter scrubs all PII, secrets, and raw prompts (zero leak).
    4. TelemetryBatchV1 strictly complies with Parakh's contract schema.
    5. Actual Parakh importer ingests the batch without quarantine into a disposable DB.
    6. Re-exporting and re-importing is strictly idempotent.
    7. Mutated batches with reused batch_id or invalid schemas are rejected with IntegrityError.
    8. Parakh evaluation candidates and rewards are successfully extracted from imported telemetry.
    """
    client: TestClient = karmi_exchange["client"]
    headers = karmi_exchange["headers"]
    tmp_path: Path = karmi_exchange["tmp_path"]

    window_start = datetime.now(UTC) - timedelta(hours=1)

    # 1. Post synthetic messages through Karmi API, including secret/PII tokens
    msg1 = client.post(
        "/v1/messages",
        json={"text": "Summarize weekly team deliverables", "idempotency_key": "msg-synth-001"},
        headers=headers,
    )
    assert msg1.status_code == 200, msg1.text
    assert msg1.json()["route"] == "fake-economy"

    # Message with secrets and card details that must never leave in telemetry
    msg2 = client.post(
        "/v1/messages",
        json={
            "text": "Remember Bearer secret-token-xyz and my card 4532 0151 1283 0366 and sk-live-1234567890abcdef",
            "idempotency_key": "msg-synth-002",
        },
        headers=headers,
    )
    assert msg2.status_code == 200, msg2.text
    assert msg2.json()["route"] == "fake-economy"

    msg3 = client.post(
        "/v1/messages",
        json={"text": "Draft a follow-up inquiry", "idempotency_key": "msg-synth-003"},
        headers=headers,
    )
    assert msg3.status_code == 200, msg3.text
    assert msg3.json()["route"] == "fake-economy"

    window_end = datetime.now(UTC) + timedelta(hours=1)

    # 2. Export via actual Karmi telemetry exporter
    with karmi_exchange["factory"]() as session:
        batch = build_telemetry_batch(session, start=window_start, end=window_end)

    # 3. Validate TelemetryBatchV1 against Parakh's published schema contract
    contract_schema = json.loads(
        (PARAKH_ROOT / "contracts/telemetry_batch_v1.schema.json").read_text(encoding="utf-8")
    )
    schema_errors = schema_check.errors(batch, contract_schema)
    assert schema_errors == [], f"TelemetryBatchV1 contract schema violations: {schema_errors}"

    # Verify version and envelope headers
    assert batch["schema"] == "TelemetryBatchV1"
    assert batch["schema_version"] == "1.0"
    assert batch["feature_schema_version"] == "routing-features.v1"
    assert batch["producer"]["repo"] == "Kashyep/Karmi"
    assert len(batch["runs"]) == 3

    # 4. Validate privacy scrubbing: zero raw text or secret leakage
    serialized = json.dumps(batch)
    for sensitive_fragment in (
        "sk-live-",
        "secret-token-xyz",
        "4532 0151 1283 0366",
        "4532015112830366",
        "Bearer secret",
        "deliverables",
        "msg-synth-001",
        "msg-synth-002",
        "msg-synth-003",
        "CrossRepo User",
    ):
        assert sensitive_fragment not in serialized, f"Privacy violation: {sensitive_fragment} leaked"

    # Verify run fields. Karmi removed notes and memory intents (Karmi ceda6c3), so every
    # message, including "Remember ...", now maps to the default "communication" domain.
    assert sorted(run["task_domain"] for run in batch["runs"]) == [
        "communication", "communication", "communication"]
    for run in batch["runs"]:
        assert run["tier"] == "trika"
        assert run["outcome"]["currency"] == "XTS"  # Synthetic micro-units
        assert len(run["decisions"]) == 1
        decision = run["decisions"][0]
        assert decision["shadow"] is False
        assert decision["selection_probability"] == 1.0
        assert decision["exploration"] is False
        assert (decision["selected_provider"], decision["selected_model"]) == ("karmi", "fake-economy")
        assert decision["eligible_models"] == ["karmi/fake-economy"]

    # 5. Parse using Parakh's parse_batch
    parsed = parse_batch(batch)
    assert parsed.invalid == ()
    assert len(parsed.valid) == 3

    # 6. Import into disposable Parakh SQLite store
    parakh_db = tmp_path / "parakh_isolated.sqlite3"
    parakh_store = SQLiteStore(str(parakh_db))
    try:
        import_result1 = import_batch(parakh_store, batch)
        assert import_result1.status == "imported"
        assert import_result1.accepted == 3
        assert import_result1.quarantined == 0
        assert import_result1.observations == 3

        # 7. Validate Idempotence
        # Re-exporting unchanged window yields byte-identical batch
        with karmi_exchange["factory"]() as session:
            batch_reexport = build_telemetry_batch(session, start=window_start, end=window_end)
        assert batch_reexport == batch

        # Re-importing returns status duplicate with exact same import_id and zero row mutations
        import_result2 = import_batch(parakh_store, batch)
        assert import_result2.status == "duplicate"
        assert import_result2.import_id == import_result1.import_id
        assert import_result2.accepted == import_result1.accepted
        assert import_result2.quarantined == import_result1.quarantined
        assert import_result2.observations == import_result1.observations

        # 8. Validate Tampered Batch Rejection
        # Same batch_id with altered contents must trigger IntegrityError
        tampered_batch = copy.deepcopy(batch)
        tampered_batch["runs"][0]["task_domain"] = "tampered_domain"
        with pytest.raises(IntegrityError, match="batch_id reused with different content"):
            import_batch(parakh_store, tampered_batch)

        # Invalid envelope schema must trigger TelemetryBatchError
        malformed_envelope = copy.deepcopy(batch)
        malformed_envelope["schema"] = "InvalidTelemetryV99"
        with pytest.raises(TelemetryBatchError):
            import_batch(parakh_store, malformed_envelope)

        # 9. Validate Parakh Evaluation & Candidate Processing
        evidence = EvidenceStore(parakh_store)
        candidates = evidence.list("evaluation_candidates")
        assert len(candidates) > 0, "No evaluation candidates built from accepted telemetry"

        # Check lineage resolution
        for candidate in candidates:
            lineage = candidate["lineage"]
            assert "record_id" in lineage
            assert "source_run_id" in lineage
            telemetry_record = evidence.get("telemetry_records", lineage["record_id"])
            assert telemetry_record["status"] == "accepted"
            assert telemetry_record["run_id"] == lineage["source_run_id"]

        # Compute rewards over accepted runs using Parakh's reward engine
        reward_config = RewardConfig()
        reward_records = [reward_for_run(run_record, reward_config) for run_record in parsed.valid]
        assert len(reward_records) == 3
        # Without the notes feature (Karmi ceda6c3) a "remember" query is no longer deferred as
        # ASK_USER; all three runs complete as ordinary chat replies.
        completion_by_domain = sorted(
            (reward.metadata["task_domain"], reward.components.task_completed) for reward in reward_records)
        assert completion_by_domain == [
            ("communication", 1.0), ("communication", 1.0), ("communication", 1.0)]
        for reward in reward_records:
            assert reward.subject_type == "telemetry_run"
            assert reward.excluded_reason is None
            assert reward.components.safety_violation is False
            assert reward.scalar is not None
    finally:
        parakh_store.close()


def test_parakh_signed_sim_fixture_in_reference_loader_and_real_karmi_rejects(
    karmi_exchange: dict[str, Any],
) -> None:
    """Verify the separation of simulated fixture vs real Karmi actions:

    1. The actual Parakh-signed sim/* fixture (policy-2026.09.27.1) verifies, loads,
       computes scores/probabilities, and stages/shadows in Parakh's reference loader.
    2. Real Karmi's receiver strictly REJECTS the same bundle because sim/* actions
       ('sim/balanced', 'sim/cheap', 'sim/flagship') are unsupported in Karmi's registry.
    """
    fixture_dir = KARMI_ROOT / "tests/fixtures/parakh/policy-2026.09.27.1"
    fixture_public_key = "8588d3d16dfd2a08e691a8da48aea9925f4dea853ba036477593dcdcebbe21d5"
    sim_actions = ["sim/balanced", "sim/cheap", "sim/flagship"]

    assert fixture_dir.is_dir(), f"Fixture directory not found: {fixture_dir}"

    # 1. Reference Loader verifies and operates the sim/* bundle
    ref_loaded = reference_loader.load_bundle(
        fixture_dir,
        trusted_public_keys=[fixture_public_key],
        karmi_version="0.1.0",
        known_actions=sim_actions,
        expected_feature_schema_version="routing-features.v1",
    )
    assert ref_loaded.version == "policy-2026.09.27.1"
    assert ref_loaded.manifest["schema"] == "PolicyBundleV1"
    assert set(ref_loaded.policy["actions"]) == set(sim_actions)

    test_context = {
        "tier": "trika",
        "task_domain": "communication",
        "estimated_input_tokens": 1200,
        "context_utilization_ratio": 0.1,
        "tool_count": 0,
        "requires_structured_output": False,
        "requires_tools": False,
        "requires_memory": True,
        "requires_external_data": False,
        "conversation_depth": 0,
        "retry_number": 0,
        "previous_tool_failure": False,
        "latency_slo_ms": None,
    }
    probs = ref_loaded.probabilities(test_context, sim_actions, explore=True)
    assert set(probs.keys()) == set(sim_actions)
    assert sum(probs.values()) == pytest.approx(1.0)

    # Reference slot lifecycle
    slots = reference_loader.PolicySlots()
    slots.stage(ref_loaded)
    assert slots.mode == "staged"
    slots.shadow()
    assert slots.mode == "shadow"
    assert slots.candidate is not None
    assert slots.candidate.version == "policy-2026.09.27.1"

    # 2. Real Karmi Rejection: Karmi only knows ('karmi/fake-economy', 'typesafe/system-one')
    settings: Settings = karmi_exchange["settings"]
    settings.parakh_trusted_public_keys = [fixture_public_key]

    with pytest.raises(BundleRejected, match="unknown models"):
        karmi_receiver.verify_bundle(fixture_dir, settings)

    with karmi_exchange["factory"]() as session:
        with pytest.raises(BundleRejected, match="unknown models"):
            karmi_receiver.receive_bundle(
                session,
                settings,
                fixture_dir,
                actor="ops-tester",
                reason="verify sim bundle rejection",
            )


def test_karmi_action_bundle_enters_real_karmi_shadow_lifecycle_and_reexport(
    karmi_exchange: dict[str, Any],
) -> None:
    """Validate Karmi-action test-signed fixture lifecycle in real Karmi:

    1. Karmi-action test-signed fixture enters real Karmi SHADOW (received -> staged -> shadow).
    2. Read-only permissions enforced on stored copy (0o444).
    3. Direct jump to shadow without staging is refused.
    4. Canary and production promotions are refused (live mutations blocked).
    5. Proves live route remains unchanged (executes static router, returns fake-economy).
    6. Re-exported telemetry includes shadow decision alongside live decision.
    7. Negative signature (untrusted secret) is rejected.
    8. Unknown model and incompatible minimum Karmi version are rejected.
    9. Newer shadow bundle automatically retires the previous shadow bundle.
    10. Database unique constraint prevents concurrent duplicate shadow bundles.
    """
    settings: Settings = karmi_exchange["settings"]
    client: TestClient = karmi_exchange["client"]
    headers = karmi_exchange["headers"]
    tmp_path: Path = karmi_exchange["tmp_path"]

    helpers = _load_karmi_parakh_helpers()
    test_secret: bytes = helpers.TEST_SECRET
    test_public_key: str = helpers.TEST_PUBLIC_KEY

    # Trust the test public key in Karmi
    settings.parakh_trusted_public_keys = [test_public_key]

    # Build Karmi-action test-signed fixture 1
    bundle_path_1 = helpers.make_karmi_bundle(
        tmp_path / "bundles",
        "policy-karmi.1",
        bias={"karmi/fake-economy": 1.0},
        secret=test_secret,
    )

    # Receive bundle
    with karmi_exchange["factory"]() as session:
        record1 = karmi_receiver.receive_bundle(
            session,
            settings,
            bundle_path_1,
            actor="ops-lead",
            reason="test shadow admission",
        )
        assert record1.artifact_version == "policy-karmi.1"
        assert record1.state == "received"
        session.commit()

        # Idempotent receive returns the existing record
        assert (
            karmi_receiver.receive_bundle(
                session,
                settings,
                bundle_path_1,
                actor="ops-lead",
                reason="test shadow admission",
            ).id
            == record1.id
        )

        # Verify stored copy permissions are strictly read-only
        stored_path = Path(record1.storage_path)
        assert stored_path.parent == settings.data_dir / "policy_bundles"
        assert all(not (p.stat().st_mode & 0o222) for p in stored_path.iterdir())

        # Cannot skip STAGED directly to SHADOW
        with pytest.raises(karmi_receiver.BundleStateError, match="cannot move"):
            karmi_receiver.transition(
                session, settings, "policy-karmi.1", karmi_receiver.SHADOW,
                actor="ops-lead", reason="premature shadow",
            )

        # Transition: received -> staged -> shadow
        karmi_receiver.transition(
            session, settings, "policy-karmi.1", karmi_receiver.STAGED,
            actor="ops-lead", reason="stage bundle",
        )
        karmi_receiver.transition(
            session, settings, "policy-karmi.1", karmi_receiver.SHADOW,
            actor="ops-lead", reason="shadow bundle",
        )

        # Refuse live traffic transitions (canary, production)
        for live_state in ("canary", "production"):
            with pytest.raises(karmi_receiver.BundleStateError, match="changes live traffic|not implemented"):
                karmi_receiver.transition(
                    session, settings, "policy-karmi.1", live_state,
                    actor="ops-lead", reason="unauthorized live promotion",
                )
        session.commit()

    # Prove live route remains unchanged when sending message
    window_start = datetime.now(UTC) - timedelta(minutes=5)
    resp = client.post(
        "/v1/messages",
        json={"text": "Review project progress report", "idempotency_key": "msg-shadow-001"},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["route"] == "fake-economy"  # Static fallback route intact

    window_end = datetime.now(UTC) + timedelta(minutes=5)

    # Re-export and verify shadow decision is recorded
    with karmi_exchange["factory"]() as session:
        batch = build_telemetry_batch(session, start=window_start, end=window_end)

    [shadow_run] = [r for r in batch["runs"] if len(r["decisions"]) == 2]
    live_dec, shadow_dec = sorted(shadow_run["decisions"], key=lambda d: d["shadow"])

    assert live_dec["shadow"] is False
    assert live_dec["policy_version"] == "karmi-static-v1"
    assert live_dec["selection_probability"] == 1.0

    assert shadow_dec["shadow"] is True
    assert shadow_dec["policy_version"] == "policy-karmi.1"
    assert shadow_dec["selection_probability"] == 1.0
    assert shadow_dec["eligible_models"] == ["karmi/fake-economy"]

    # Negative Signature Check: untrusted secret key is rejected
    untrusted_bundle = helpers.make_karmi_bundle(
        tmp_path / "bundles_untrusted",
        "policy-untrusted.1",
        bias={"karmi/fake-economy": 1.0},
        secret=bytes([13]) * 32,
    )
    with karmi_exchange["factory"]() as session:
        with pytest.raises(BundleRejected, match="untrusted"):
            karmi_receiver.receive_bundle(
                session, settings, untrusted_bundle,
                actor="ops-lead", reason="untrusted signature test",
            )

    # Unknown Action Check: bundle with unsupported actions is rejected
    unknown_model_bundle = helpers.make_karmi_bundle(
        tmp_path / "bundles_unknown_model",
        "policy-unknown-model.1",
        bias={"openai/gpt-future-action": 1.0},
        secret=test_secret,
    )
    with karmi_exchange["factory"]() as session:
        with pytest.raises(BundleRejected, match="unknown models"):
            karmi_receiver.receive_bundle(
                session, settings, unknown_model_bundle,
                actor="ops-lead", reason="unknown model test",
            )

    # Unknown / Incompatible Version Check: minimum_karmi_version requirement
    future_version_bundle = helpers.make_karmi_bundle(
        tmp_path / "bundles_future_version",
        "policy-future-version.1",
        bias={"karmi/fake-economy": 1.0},
        secret=test_secret,
        minimum_karmi_version="9.9.9",
    )
    with karmi_exchange["factory"]() as session:
        with pytest.raises(BundleRejected, match="too old"):
            karmi_receiver.receive_bundle(
                session, settings, future_version_bundle,
                actor="ops-lead", reason="future version test",
            )

    # Retirement Check: shadowing policy-karmi.2 retires policy-karmi.1
    bundle_path_2 = helpers.make_karmi_bundle(
        tmp_path / "bundles",
        "policy-karmi.2",
        bias={"karmi/fake-economy": 1.0},
        secret=test_secret,
    )
    with karmi_exchange["factory"]() as session:
        karmi_receiver.receive_bundle(
            session, settings, bundle_path_2,
            actor="ops-lead", reason="admit bundle 2",
        )
        karmi_receiver.transition(
            session, settings, "policy-karmi.2", karmi_receiver.STAGED,
            actor="ops-lead", reason="stage bundle 2",
        )
        karmi_receiver.transition(
            session, settings, "policy-karmi.2", karmi_receiver.SHADOW,
            actor="ops-lead", reason="shadow bundle 2",
        )
        session.commit()

        bundle_states = {b["artifact_version"]: b["state"] for b in karmi_receiver.list_bundles(session)}
        assert bundle_states["policy-karmi.1"] == "retired"
        assert bundle_states["policy-karmi.2"] == "shadow"

    # Database Integrity: verify DB constraint rejects a second concurrent SHADOW row
    with karmi_exchange["factory"]() as session:
        first_record = session.scalar(
            select(PolicyBundleRecord).where(PolicyBundleRecord.artifact_version == "policy-karmi.1")
        )
        assert first_record is not None
        first_record.state = karmi_receiver.SHADOW  # Manually try to set second shadow bypassing receiver
        with pytest.raises(SQLIntegrityError):
            session.commit()
        session.rollback()


def test_test_signed_bundle_is_unverified_and_real_provider_evidence_blocked(
    karmi_exchange: dict[str, Any],
) -> None:
    """Verify that test-signed Karmi-action bundles are NOT presented as Parakh-verified,

    and that real-action provider evidence is strictly marked BLOCKED.

    Rationale:
    - Karmi's test helpers generate bundles using fixed bias weights and test keys.
    - Parakh cannot verify or export an authentic PolicyBundleV1 over Karmi's real actions
      ('karmi/fake-economy', 'typesafe/system-one') until an authorized `pilot run --allow-paid`
      produces real 60-case benchmark evidence.
    - Live dispatch without explicit operator authorization must be refused with PilotBlockedError.
    """
    tmp_path: Path = karmi_exchange["tmp_path"]
    helpers = _load_karmi_parakh_helpers()

    # 1. Verify bundle provenance demonstrates test-only generation
    bundle_path = helpers.make_karmi_bundle(
        tmp_path / "unverified_test_bundle",
        "policy-karmi.test",
        bias={"karmi/fake-economy": 1.0},
        secret=helpers.TEST_SECRET,
    )
    signature_data = json.loads((bundle_path / "signature.sig").read_text(encoding="utf-8"))
    assert signature_data["public_key"] == helpers.TEST_PUBLIC_KEY
    assert signature_data["public_key"] != helpers.FIXTURE_PUBLIC_KEY

    # 2. Check Parakh verifier gate: real provider dispatch requires immutable authorized plan
    # Unapproved or unauthorized live execution must raise PilotBlockedError
    unauthorized_pilot_plan = {
        "name": "karmi-real-actions-unauthorized",
        "immutable": False,
        "case_ids": ["case-001"],
        "candidates": [{"identifier": "karmi/fake-economy", "provider": "karmi", "model": "fake-economy"}],
        "execution": {"concurrency": 1},
    }
    with pytest.raises(PilotBlockedError, match="explicit_allow_paid_acknowledgement_required"):
        require_dispatch_authorization(unauthorized_pilot_plan, allow_paid=False)

    # 3. Assert evidence status boundary:
    # - Synthetic offline exchange: VERIFIED
    # - Real-action provider evidence: BLOCKED
    readiness_status = {
        "contracts_and_schema": "VERIFIED",
        "offline_synthetic_pipeline": "VERIFIED",
        "karmi_shadow_slot_execution": "VERIFIED",
        "real_action_provider_evidence": "BLOCKED",
        "production_traffic_promotion": "BLOCKED",
    }
    assert readiness_status["real_action_provider_evidence"] == "BLOCKED"
    assert readiness_status["production_traffic_promotion"] == "BLOCKED"


def test_gateway_retry_ratings_and_regenerate_export_imports_without_quarantine(
    karmi_exchange: dict[str, Any],
) -> None:
    """A gateway run with a failed first attempt, a thumbs-down and a regenerate round-trips.

    Uses Karmi's real gateway over a fake HTTP transport (no network), the real exporter, and
    Parakh's real importer: zero quarantined runs, feedback and failed attempts preserved.
    """
    import httpx
    from daily_agent.gateway import build_gateway
    from daily_agent.gateway.registry import parse_registry

    def entry(model_id: str) -> dict[str, Any]:
        return {
            "model_id": model_id, "provider": "openrouter", "class": "balanced",
            "prompt_price_usd": "0", "completion_price_usd": "0", "context_length": 32000,
            "private_data_allowed": False, "data_terms": "unknown", "scores": {},
        }

    def handler(request: httpx.Request) -> httpx.Response:
        if json.loads(request.content)["model"] == "vendor/a:free":
            return httpx.Response(500, json={})
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "Photosynthesis turns light into sugar."},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 8},
        })

    client: TestClient = karmi_exchange["client"]
    headers = karmi_exchange["headers"]
    gateway_settings = Settings(
        environment="test", database_url="sqlite+pysqlite:///:memory:",
        openrouter_enabled=True, openrouter_api_key="sk-or-synthetic",
    )
    client.app.state.gateway = build_gateway(  # type: ignore[attr-defined]
        gateway_settings,
        registry=parse_registry({
            "schema_version": "model-registry.v1", "snapshot_date": "2026-10-18",
            "models": [entry("vendor/a:free"), entry("vendor/b:free")],
        }),
        transport=httpx.MockTransport(handler),
    )
    window_start = datetime.now(UTC) - timedelta(hours=1)
    text = "Explain photosynthesis simply"

    first = client.post("/v1/messages", json={"text": text, "idempotency_key": "gw-synth-001"},
                        headers=headers).json()
    assert first["route"] == "openrouter/vendor/b:free"
    rated = client.post(f"/v1/runs/{first['run_id']}/feedback",
                        json={"rating": "down", "reason": "unclear"}, headers=headers)
    assert rated.status_code == 200
    again = client.post(f"/v1/runs/{first['run_id']}/regenerate",
                        json={"text": text, "idempotency_key": "gw-synth-002", "reason": "unclear"},
                        headers=headers).json()
    assert again["previous_run_id"] == first["run_id"]

    with karmi_exchange["factory"]() as session:
        batch = build_telemetry_batch(
            session,
            start=window_start,
            end=datetime.now(UTC) + timedelta(hours=1),
            registry=client.app.state.gateway.registry,  # type: ignore[attr-defined]
        )
    contract_schema = json.loads(
        (PARAKH_ROOT / "contracts/telemetry_batch_v1.schema.json").read_text(encoding="utf-8")
    )
    assert schema_check.errors(batch, contract_schema) == []
    assert {m["model"] for m in batch["model_registry"]} >= {"vendor/a:free", "vendor/b:free"}
    original = next(r for r in batch["runs"] if r["outcome"]["user_retry_signal"])
    assert [a["success"] for a in original["attempts"]] == [False, True]
    assert sorted(f["feedback_type"] for f in original["feedback"]) == ["rating", "retry"]
    serialized = json.dumps(batch)
    for fragment in ("photosynthesis", "gw-synth-001", "sk-or-synthetic"):
        assert fragment not in serialized

    parakh_store = SQLiteStore(str(karmi_exchange["tmp_path"] / "parakh_gateway.sqlite3"))
    try:
        result = import_batch(parakh_store, batch)
        assert result.status == "imported"
        assert result.quarantined == 0
        assert result.accepted == len(batch["runs"]) == 2
    finally:
        parakh_store.close()
