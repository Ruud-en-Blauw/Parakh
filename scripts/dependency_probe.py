#!/usr/bin/env python3
"""Karmi against disposable PostgreSQL/Redis: native integration suite and real outage/recovery.

Exit 0 = all assertions held, 1 = application failure, 2 = environment unavailable (BLOCKED, with
evidence written). Only synthetic identities and disposable containers are used.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

import disposable_services as ds
from performance_probe import serve
from resilience_probe import stop

LEAK_MARKERS = ("daily_agent_test:", "postgresql", "psycopg", "Traceback", "password")


def request(method: str, url: str, *, body: dict[str, object] | None = None,
            token: str | None = None, timeout: float = 5) -> tuple[int, str]:
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"} if data else {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers, method=method),
                                    timeout=timeout) as response:
            return response.status, response.read(4096).decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        with error:
            return error.code, error.read(4096).decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, ConnectionError):
        return 0, ""


def wait_status(url: str, want: int, timeout: float) -> int:
    deadline, status = time.monotonic() + timeout, 0
    while time.monotonic() < deadline:
        status, _ = request("GET", url, timeout=3)
        if status == want:
            return status
        time.sleep(0.25)
    return status


def integration(karmi: Path) -> dict[str, object]:
    env = {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "TMPDIR") if key in os.environ}
    env.update(PYTHONPATH=f"{karmi / 'src'}{os.pathsep}{karmi}", DAILY_AGENT_ENVIRONMENT="test",
               DAILY_AGENT_INTEGRATION_DATABASE_URL=ds.DATABASE_URL,
               DAILY_AGENT_INTEGRATION_REDIS_URL=ds.REDIS_URL, DAILY_AGENT_LIVE_MODELS_ENABLED="false")
    # Karmi's addopts already include -q; adding another would suppress the summary line.
    completed = subprocess.run([str(karmi / ".venv/bin/python"), "-m", "pytest", "-p", "no:cacheprovider",
                                "-m", "integration", "tests/integration"],
                               cwd=karmi, env=env, capture_output=True, text=True, timeout=900, check=False)
    print(completed.stdout[-4000:], completed.stderr[-2000:], sep="\n")
    summary = (completed.stdout.strip().splitlines() or [""])[-1]
    return {"passed": completed.returncode == 0, "exit_code": completed.returncode, "summary": summary}


def outage(karmi: Path, services: ds.Services) -> dict[str, object]:
    with socket.socket() as address:
        address.bind(("127.0.0.1", 0))
        port = address.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    checks: dict[str, bool] = {}
    # Karmi creates its schema automatically only on SQLite; a fresh Postgres needs the migrations.
    env = {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "TMPDIR") if key in os.environ}
    env.update(PYTHONPATH=str(karmi / "src"), DAILY_AGENT_ENVIRONMENT="test",
               DAILY_AGENT_DATABASE_URL=ds.DATABASE_URL, DAILY_AGENT_LIVE_MODELS_ENABLED="false")
    checks["schema_migrated_to_head"] = subprocess.run(
        [str(karmi / ".venv/bin/python"), "-m", "alembic", "upgrade", "head"], cwd=karmi, env=env,
        capture_output=True, timeout=300, check=False).returncode == 0
    with tempfile.TemporaryDirectory(prefix="karmi-outage-") as temporary:
        server = serve(karmi, Path(temporary) / "unused.db", port,
                       database_url=ds.DATABASE_URL, redis_url=ds.REDIS_URL)
        try:
            checks["ready_on_postgres"] = wait_status(base + "/ready", 200, 30) == 200
            status, raw = request("POST", base + "/dev/token?role=customer")
            token = json.loads(raw)["token"] if status == 200 else None
            checks["synthetic_identity"] = token is not None
            first = request("POST", base + "/v1/messages", token=token,
                            body={"text": "Draft a short status note", "idempotency_key": "outage-before"})
            checks["message_before_outage"] = first[0] == 200

            services.container("stop", services.postgres)
            down_ready = request("GET", base + "/ready", timeout=10)
            down_message = request("POST", base + "/v1/messages", token=token, timeout=10,
                                   body={"text": "Draft during outage", "idempotency_key": "outage-during"})
            checks["ready_reports_failure"] = 500 <= down_ready[0] < 600
            checks["message_not_accepted_during_outage"] = down_message[0] != 200
            checks["no_internal_detail_leaked"] = not any(
                marker in body for marker in LEAK_MARKERS for body in (down_ready[1], down_message[1]))
            checks["process_survived_outage"] = server.poll() is None

            services.container("start", services.postgres)
            recovered = services.wait_postgres() and wait_status(base + "/ready", 200, 30) == 200
            checks["ready_recovers"] = recovered
            retried = request("POST", base + "/v1/messages", token=token,
                              body={"text": "Draft during outage", "idempotency_key": "outage-during"})
            replay = request("POST", base + "/v1/messages", token=token,
                             body={"text": "Draft a short status note", "idempotency_key": "outage-before"})
            checks["retry_after_recovery_succeeds"] = retried[0] == 200
            checks["pre_outage_request_deduplicated"] = (
                replay[0] == 200 and first[0] == 200
                and json.loads(replay[1]).get("run_id") == json.loads(first[1]).get("run_id"))

            services.container("stop", services.redis)
            checks["redis_outage_does_not_break_ready"] = wait_status(base + "/ready", 200, 10) == 200
        finally:
            stop(server)
    return {"checks": checks, "passed": all(checks.values()),
            "redis_note": "Karmi's app runtime does not use Redis (config only); Redis is exercised by "
                          "tests/integration. A Redis outage therefore must not affect /ready."}

def sql_count(services: ds.Services, table: str) -> int:
    result = subprocess.run([services.engine, "exec", services.postgres, "psql", "-U", "daily_agent_test",
                             "-d", "daily_agent_test", "-tAc", f"SELECT count(*) FROM {table}"],
                            capture_output=True, text=True, timeout=30, check=False)
    return int(result.stdout.strip()) if result.returncode == 0 and result.stdout.strip().isdigit() else -1


def rollback(karmi: Path, services: ds.Services) -> dict[str, object]:
    """Schema rollback rehearsal on disposable PostgreSQL: full chain round trip, then a one-release
    rollback with live data. The previous release's binary is not exercised (schema level only)."""
    env = {key: os.environ[key] for key in ("PATH", "HOME", "LANG", "TMPDIR") if key in os.environ}
    env.update(PYTHONPATH=str(karmi / "src"), DAILY_AGENT_ENVIRONMENT="test",
               DAILY_AGENT_DATABASE_URL=ds.DATABASE_URL, DAILY_AGENT_LIVE_MODELS_ENABLED="false")
    python = str(karmi / ".venv/bin/python")
    walked = subprocess.run([python, str(Path(__file__).with_name("migration_rollback.py"))], cwd=karmi, env=env,
                            capture_output=True, text=True, timeout=600, check=False)
    try:
        chain = json.loads(walked.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError):
        return {"checks": {"migration_chain_round_trip": False}, "passed": False,
                "error": walked.stderr.strip().splitlines()[-1:] or ["no output"]}
    checks: dict[str, bool] = dict(chain["checks"])

    def alembic(*args: str) -> bool:
        return subprocess.run([python, "-m", "alembic", *args], cwd=karmi, env=env, capture_output=True,
                              timeout=300, check=False).returncode == 0

    with socket.socket() as address:
        address.bind(("127.0.0.1", 0))
        port = address.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    body = {"text": "Draft a short status note", "idempotency_key": "rollback-before"}
    with tempfile.TemporaryDirectory(prefix="karmi-rollback-") as temporary:
        server = serve(karmi, Path(temporary) / "unused.db", port, database_url=ds.DATABASE_URL)
        try:
            checks["ready_at_head"] = wait_status(base + "/ready", 200, 30) == 200
            status, raw = request("POST", base + "/dev/token?role=customer")
            token = json.loads(raw)["token"] if status == 200 else None
            first = request("POST", base + "/v1/messages", token=token, body=body)
            checks["data_written_at_head"] = first[0] == 200
        finally:
            stop(server)
        before = {table: sql_count(services, table) for table in ("users", "accounts", "runs")}
        checks["downgrade_one_release"] = alembic("downgrade", "-1")
        after = {table: sql_count(services, table) for table in before}
        checks["core_rows_preserved_by_downgrade"] = before == after and min(before.values()) > 0
        checks["upgrade_after_rollback"] = alembic("upgrade", "head")
        server = serve(karmi, Path(temporary) / "unused.db", port, database_url=ds.DATABASE_URL)
        try:
            checks["ready_after_reupgrade"] = wait_status(base + "/ready", 200, 30) == 200
            replay = request("POST", base + "/v1/messages", token=token, body=body)
            checks["idempotent_replay_survives_rollback"] = (
                replay[0] == 200 and first[0] == 200
                and json.loads(replay[1]).get("run_id") == json.loads(first[1]).get("run_id"))
            fresh = request("POST", base + "/v1/messages", token=token,
                            body={"text": "Draft after rollback", "idempotency_key": "rollback-after"})
            checks["new_request_after_reupgrade"] = fresh[0] == 200
        finally:
            stop(server)
    return {"checks": checks, "passed": all(checks.values()), "revisions": chain["revisions"],
            "head": chain["head"], "row_counts": before,
            "data_discarded_by_one_release_rollback": chain["tables_dropped_by_head_rollback"],
            "limitation": "previous-release application binary not run against the downgraded schema"}



def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("integration", "outage", "rollback"))
    parser.add_argument("--karmi", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    karmi = args.karmi.resolve()
    evidence = args.output / f"dependency-{args.mode}.json"
    if not (karmi / ".venv/bin/python").is_file():
        result: dict[str, object] = {"status": "BLOCKED", "reason": "Karmi .venv/bin/python missing"}
    else:
        try:
            with ds.disposable() as services:
                body = (integration(karmi) if args.mode == "integration" else outage(karmi, services)
                        if args.mode == "outage" else rollback(karmi, services))
            result = {"status": "PASS" if body["passed"] else "FAIL", "engine": services.engine, **body}
        except ds.Unavailable as error:
            result = {"status": "BLOCKED", "reason": str(error)}
    evidence.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"{args.mode}: {result['status']}", json.dumps(result.get("checks", result.get("reason", ""))))
    return {"PASS": 0, "FAIL": 1, "BLOCKED": 2}[str(result["status"])]


if __name__ == "__main__":
    sys.exit(main())
