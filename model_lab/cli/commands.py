"""Dependency-free command line interface for the ModelLab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

from model_lab.application.execution import ExecutionEngine
from model_lab.application.reporting import write_report_bundle
from model_lab.benchmark import export_candidate_jsonl, load_suite, validate_suite
from model_lab.analysis import compare_grades
from model_lab.application.observability import environment_checks
from model_lab.constraints import evaluate_constraints, file_sha256, load_constraint_map
from model_lab.errors import IntegrityError, ModelLabError, ValidationError
from model_lab.grading import grade_attempt
from model_lab.ingestion import ingest_file
from model_lab.operator_input import build_immutable_plan, load_operator_input, validate_operator_input, write_immutable_plan
from model_lab.pipeline import run_offline_demo
from model_lab.pilot import (
    build_pilot_plan,
    dispatch_preview,
    parse_candidate_spec,
    require_dispatch_authorization,
    run_authorized_immutable_pilot,
    validate_pilot_plan,
    write_plan,
)
from model_lab.promptfoo import export_promptfoo_manifest, import_promptfoo_fixture
from model_lab.providers.fake import FakeProvider
from model_lab.review import export_blind_review, import_blind_reviews, review_bindings
from model_lab.schemas import Attempt, Budget, Grade, ModelConfig, Run, Suite, utc_now
from model_lab.storage import SQLiteStore
from model_lab.cli import optimization


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="model-lab", description="Offline-first reproducible model evaluation laboratory")
    sub = parser.add_subparsers(dest="command", required=True)

    suite = sub.add_parser("suite", help="validate, inspect, or export a benchmark suite")
    suite_sub = suite.add_subparsers(dest="suite_command", required=True)
    validate_cmd = suite_sub.add_parser("validate")
    validate_cmd.add_argument("path", default="benchmarks/seed_cases.jsonl", nargs="?", help="path to benchmark suite file (default: benchmarks/seed_cases.jsonl)")
    inspect_cmd = suite_sub.add_parser("inspect")
    inspect_cmd.add_argument("path", default="benchmarks/seed_cases.jsonl", nargs="?", help="path to benchmark suite file (default: benchmarks/seed_cases.jsonl)")
    inspect_cmd.add_argument("--group-by", help="comma-separated grouping fields (e.g. domain,complexity_level)")
    export = suite_sub.add_parser("export")
    export.add_argument("path", default="benchmarks/seed_cases.jsonl", nargs="?", help="path to benchmark suite file (default: benchmarks/seed_cases.jsonl)")
    export.add_argument("--out", required=True)
    export.add_argument("--candidate-only", action="store_true", default=False)

    demo = sub.add_parser("demo", help="run the complete 60-case offline fake-provider workflow")
    demo.add_argument("--suite", default="benchmarks/seed_cases.jsonl", help="path to benchmark suite file (default: benchmarks/seed_cases.jsonl)")
    demo.add_argument("--out", default="lab-data/demo", help="output directory for demo artifacts (default: lab-data/demo)")
    demo.add_argument("--seed", type=int, default=7, help="random seed for demo evaluation (default: 7)")

    doctor = sub.add_parser("doctor", help="check local offline prerequisites")
    doctor.add_argument("--suite", default="benchmarks/seed_cases.jsonl", help="path to benchmark suite file (default: benchmarks/seed_cases.jsonl)")

    tui = sub.add_parser("tui", help="interactive terminal console (requires the 'tui' extra: pip install -e '.[tui]')")
    tui.add_argument("--data", default="lab-data/tui", help="workspace directory holding model_lab.sqlite3 (default: lab-data/tui)")
    tui.add_argument("--suite", default="benchmarks/seed_cases.jsonl", help="benchmark suite file (default: benchmarks/seed_cases.jsonl)")
    tui.add_argument("--demo", action="store_true", help="if the workspace has no runs, start the offline SIMULATED demo on launch")
    tui.add_argument("--plan", help="optional pilot plan JSON to preflight (read-only; paid dispatch stays CLI-only)")

    init = sub.add_parser("init", help="create a local ModelLab workspace")
    init.add_argument("path", default="lab-data", nargs="?", help="workspace directory path (default: lab-data)")

    plan = sub.add_parser("plan", help="write a bounded experiment plan without executing it")
    plan.add_argument("--suite", default="benchmarks/seed_cases.jsonl", help="path to benchmark suite file (default: benchmarks/seed_cases.jsonl)")
    plan.add_argument("--models", required=True, help="comma-separated provider/model labels")
    plan.add_argument("--repeats", type=int, default=1, help="number of repeats per case (default: 1)")
    plan.add_argument("--out", required=True, help="destination file path for experiment plan JSON")
    plan.add_argument("--budget-usd", type=float, help="optional budget cap in USD")

    run = sub.add_parser("run", help="run one offline fake-provider experiment")
    run.add_argument("--suite", default="benchmarks/seed_cases.jsonl", help="path to benchmark suite file (default: benchmarks/seed_cases.jsonl)")
    run.add_argument("--out", default="lab-data/run", help="output directory for run artifacts (default: lab-data/run)")
    run.add_argument("--seed", type=int, default=7, help="random seed (default: 7)")
    run.add_argument("--model", default="synthetic-v1", help="model label (default: synthetic-v1)")
    run.add_argument("--max-cases", type=int, help="optional limit on number of suite cases to evaluate")

    imported = sub.add_parser("import", help="import structured attempts into an existing SQLite run")
    imported.add_argument("path", help="path to file containing attempt records")
    imported.add_argument("--db", required=True, help="path to SQLite database file")
    imported.add_argument("--run", required=True, dest="run_id", help="target run ID in the database")
    imported.add_argument("--format", choices=("jsonl", "json", "csv", "aggregate-report"), help="input file format (jsonl, json, csv, or aggregate-report)")
    imported.add_argument("--source-label", default="cli-import", help="label for import source (default: cli-import)")

    for name in ("grade", "compare", "report", "audit"):
        command = sub.add_parser(name, help=f"{name} an existing offline experiment")
        command.add_argument("--summary", required=False, help="path to experiment summary JSON file")
        command.add_argument("--run", required=False, dest="run_id", help="target run ID")
        if name in {"grade", "compare", "report", "audit"}:
            command.add_argument("--db", help="path to SQLite database file")
            command.add_argument("--suite", help="path to benchmark suite file")
        if name == "compare":
            command.add_argument("--left-run", help="baseline run ID for comparison")
            command.add_argument("--right-run", help="candidate run ID for comparison")
            command.add_argument("--out", help="output path for comparison result JSON")
        if name == "report":
            command.add_argument("--out", help="output directory for report bundle")
        if name == "audit":
            command.add_argument("--out", help="output file path for audit JSON")

    promptfoo = sub.add_parser("promptfoo", help="export/import versioned Promptfoo-shaped fixtures")
    promptfoo_sub = promptfoo.add_subparsers(dest="promptfoo_command", required=True)
    promptfoo_export = promptfoo_sub.add_parser("export")
    promptfoo_export.add_argument("--suite", default="benchmarks/seed_cases.jsonl")
    promptfoo_export.add_argument("--out", required=True)
    promptfoo_export.add_argument("--run-id", default="promptfoo-run")
    promptfoo_import = promptfoo_sub.add_parser("import")
    promptfoo_import.add_argument("--manifest", required=True)
    promptfoo_import.add_argument("--artifact", required=True)
    promptfoo_import.add_argument("--quarantine-dir")
    promptfoo_run = promptfoo_sub.add_parser("run")
    promptfoo_run.add_argument("--manifest", required=True)
    promptfoo_run.add_argument("--allow-paid", action="store_true", help="require explicit operator acknowledgement for paid dispatch")

    pilot = sub.add_parser("pilot", help="prepare and inspect a controlled development-only live pilot")
    pilot_sub = pilot.add_subparsers(dest="pilot_command", required=True)
    pilot_validate = pilot_sub.add_parser("validate-input", help="validate operator input without dispatch")
    pilot_validate.add_argument("--input", required=True, dest="operator_input")
    pilot_validate.add_argument("--suite", default="benchmarks/seed_cases.jsonl")
    pilot_validate.add_argument("--suite-kind", choices=("core", "student"), default=None,
                                help="core (default: 36 train cases) or student (train/calibration, never holdout); must match the input's suite_kind if set")
    pilot_plan = pilot_sub.add_parser("plan")
    pilot_plan.add_argument("--suite", default="benchmarks/seed_cases.jsonl")
    pilot_plan.add_argument("--out", required=True)
    pilot_plan.add_argument("--operator-input", help="complete operator YAML; converts to an immutable plan")
    pilot_plan.add_argument("--suite-kind", choices=("core", "student"), default=None,
                            help="with --operator-input only: core (default) or student; use --suite benchmarks_student/cases.jsonl for student")
    pilot_plan.add_argument("--constraint-map", default="docs/live_pilots/router-constraint-map-v0.2.0.json")
    pilot_plan.add_argument("--source-manifest", default="docs/product_sources/SOURCE_MANIFEST.json")
    pilot_plan.add_argument("--candidate", action="append", default=[], help="provider:model[:revision], repeatable")
    pilot_plan.add_argument("--repeats", type=int, default=2)
    pilot_plan.add_argument("--temperature", type=float, default=0.0)
    pilot_plan.add_argument("--seed", type=int, default=17)
    pilot_plan.add_argument("--max-output-tokens", type=int, default=1200)
    pilot_plan.add_argument("--timeout-seconds", type=float, default=60.0)
    pilot_plan.add_argument("--concurrency", type=int, default=1)
    pilot_plan.add_argument("--max-retries", type=int, default=0)
    pilot_plan.add_argument("--max-spend-minor", type=int)
    pilot_plan.add_argument("--currency", default="USD")
    pilot_plan.add_argument("--pricing-snapshot", help="JSON file or source label for the frozen pricing snapshot")
    pilot_plan.add_argument("--prompt-template-revision", default="candidate-v1")
    pilot_show = pilot_sub.add_parser("show")
    pilot_show.add_argument("path")
    pilot_run = pilot_sub.add_parser("run", help="fail-closed dispatch gate; no paid calls in this preparation pass")
    pilot_run.add_argument("--plan", required=True)
    pilot_run.add_argument("--allow-paid", action="store_true")
    pilot_run.add_argument("--suite", default="benchmarks/seed_cases.jsonl")
    pilot_run.add_argument("--out", default="lab-data/live-pilots")
    pilot_run.add_argument("--source-manifest", default="docs/product_sources/SOURCE_MANIFEST.json")
    pilot_run.add_argument("--constraint-map", default="docs/live_pilots/router-constraint-map-v0.2.0.json")
    pilot_run.add_argument("--max-requests-per-day", type=int, default=None,
                           help="refuse to start if pending requests exceed this; a run reaching it stops cleanly as incomplete")
    pilot_run.add_argument("--catalog", help="OpenRouter catalog JSON snapshot; openrouter candidates are zero-price checked before any dispatch")

    review = sub.add_parser("review", help="export or import blind human review records")
    review_sub = review.add_subparsers(dest="review_command", required=True)
    review_export = review_sub.add_parser("export")
    review_export.add_argument("--db", required=True)
    review_export.add_argument("--run", required=True, dest="run_id")
    review_export.add_argument("--out", required=True)
    review_export.add_argument("--include-prompt", action="store_true")
    review_export.add_argument("--suite", help="path to benchmark suite file")
    review_import = review_sub.add_parser("import")
    review_import.add_argument("--db", required=True)
    review_import.add_argument("--run", required=True, dest="run_id")
    review_import.add_argument("--in", required=True, dest="input_path")

    router = sub.add_parser("router", help="draft routing evidence and contextual-bandit router optimization")
    router_sub = router.add_subparsers(dest="router_command", required=True)
    recommend = router_sub.add_parser("recommend")
    recommend.add_argument("--summary", default="lab-data/demo/summary.json")
    recommend.add_argument("--out")
    recommend.add_argument("--draft", action="store_true", default=False)
    recommend.add_argument("--constraint-map", default="docs/live_pilots/router-constraint-map-v0.2.0.json")
    optimization.add_parsers(sub, router_sub)
    catalog = sub.add_parser("catalog", help="inspect and snapshot provider catalogs")
    catalog_sub = catalog.add_subparsers(dest="catalog_command", required=True)
    catalog_snap = catalog_sub.add_parser("snapshot", help="snapshot zero-price free models from openrouter")
    catalog_snap.add_argument("--out", default="docs/live_pilots/catalog", help="output directory or file for snapshot JSON")
    catalog_snap.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    catalog_snap.add_argument("--date", help="snapshot date YYYY-MM-DD")
    return parser


def _print_json(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2))


def _run_rows(suite: Suite, store: SQLiteStore, run_id: str) -> tuple[list[Attempt], list[Grade], list[dict[str, Any]]]:
    attempts = store.list_attempts(run_id)
    cases = {case.case_id: case for case in suite.cases}
    grades = [grade_attempt(cases[attempt.case_id], attempt) for attempt in attempts]
    grades_by_attempt = {grade.attempt_id: grade for grade in grades}
    rows: list[dict[str, Any]] = []
    for attempt in attempts:
        case = cases[attempt.case_id]
        grade = grades_by_attempt[attempt.attempt_id]
        rows.append({
            "attempt_id": attempt.attempt_id, "case_id": attempt.case_id,
            "provider": attempt.model_config.provider, "model": attempt.model_config.model,
            "domain": case.domain, "workflow": case.family_id,
            "complexity_level": case.complexity_level, "split": case.split,
            "passed": grade.passed, "score": grade.score,
            "latency_ms": attempt.completion_latency_ms, "cost_minor": attempt.cost_minor,
            "response_text": attempt.response_text, "status": attempt.status.value,
            "prompt_hash": attempt.prompt_hash, "synthetic": attempt.raw_metadata.get("synthetic", False),
        })
    return attempts, grades, rows


def _write_json(path: str | Path, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if optimization.handles(args):
            return optimization.run(args)
        if args.command == "suite":
            suite = load_suite(args.path)
            if args.suite_command in {"validate", "inspect"}:
                _print_json(validate_suite(suite))
            elif args.suite_command == "export":
                if not args.candidate_only:
                    raise ModelLabError("only --candidate-only export is supported for protected seed suites")
                count = export_candidate_jsonl(suite, args.out)
                _print_json({"exported": count, "path": args.out, "candidate_only": True})
            return 0
        if args.command == "demo":
            _print_json(run_offline_demo(args.suite, args.out, seed=args.seed))
            return 0
        if args.command == "doctor":
            checks = environment_checks(args.suite)
            _print_json(checks)
            return 0 if checks["offline_ready"] else 2
        if args.command == "tui":
            try:
                from model_lab.tui import run_tui
            except ImportError as exc:
                raise ModelLabError(f"the TUI needs Textual: pip install -e '.[tui]' ({exc})") from exc
            return run_tui(data_dir=args.data, suite_path=args.suite, demo=args.demo, plan_path=args.plan)
        if args.command == "init":
            destination = Path(args.path)
            destination.mkdir(parents=True, exist_ok=True)
            config = {"version": 1, "data_dir": str(destination), "provider_mode": "offline_fake_default", "secrets_from_environment": True}
            _write_json(destination / "model-lab.json", config)
            _print_json({"initialized": True, "path": str(destination), "config": str(destination / "model-lab.json")})
            return 0
        if args.command == "plan":
            suite = load_suite(args.suite)
            if args.repeats < 1:
                raise ModelLabError("--repeats must be positive")
            models = tuple(model.strip() for model in args.models.split(",") if model.strip())
            if not models:
                raise ModelLabError("--models must contain at least one model")
            cases = len(suite.cases)
            plan = {"plan_version": 1, "suite_version": suite.suite_version, "suite_hash": suite.source_hash, "models": list(models), "cases": cases, "repeats": args.repeats, "maximum_requests": cases * args.repeats * len(models), "budget_usd": args.budget_usd, "paid_execution_allowed": False, "synthetic_default": True}
            _write_json(args.out, plan)
            _print_json(plan)
            return 0
        if args.command == "run":
            suite = load_suite(args.suite)
            store = SQLiteStore(Path(args.out) / "model_lab.sqlite3")
            try:
                case_ids = suite.case_ids[:args.max_cases] if args.max_cases else suite.case_ids
                run = Run(run_id="run-cli", suite_version=suite.suite_version, case_ids=case_ids, model_config=ModelConfig(provider="fake", model=args.model, parameters={"temperature": 0, "synthetic": True}), seed=args.seed, budget=Budget(max_cases=len(case_ids), max_requests=len(case_ids)), started_at=utc_now(), environment={"mode": "offline_synthetic"})
                result = ExecutionEngine(store, FakeProvider(model=args.model)).execute(suite, run, case_ids=case_ids)
                _print_json({"run_id": result.run_id, "status": result.status, "attempts": len(result.attempts), "failures": result.failures, "synthetic": True})
            finally:
                store.close()
            return 0
        if args.command == "import":
            store = SQLiteStore(args.db)
            try:
                result = ingest_file(store, args.run_id, args.path, fmt=args.format, source_label=args.source_label)
                _print_json({"imported": result.imported, "quarantined": result.quarantined, "duplicates": result.duplicates})
            finally:
                store.close()
            return 0
        if args.command == "promptfoo":
            if args.promptfoo_command == "export":
                suite = load_suite(args.suite)
                manifest = export_promptfoo_manifest([case.candidate_payload() for case in suite.cases], run_id=args.run_id)
                _write_json(args.out, manifest)
                _print_json({"exported": len(suite.cases), "path": args.out, "candidate_only": True, "manifest_hash": manifest["manifest_hash"]})
            elif args.promptfoo_command == "run":
                if not getattr(args, "allow_paid", False):
                    raise ModelLabError("live Promptfoo invocation blocked: explicit --allow-paid acknowledgement required")
                raise ModelLabError("live Promptfoo invocation blocked: approved live runner required")
            else:
                manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
                result = import_promptfoo_fixture(args.artifact, manifest, quarantine_dir=args.quarantine_dir)
                _print_json({"accepted": result.accepted, "attempts": len(result.attempts), "errors": list(result.errors), "source_hash": result.source_hash, "quarantine_path": str(result.quarantine_path) if result.quarantine_path else None})
                return 0 if result.accepted else 2
            return 0
        if args.command == "pilot":
            if args.pilot_command == "validate-input":
                value = load_operator_input(args.operator_input)
                normalized = validate_operator_input(value, suite_path=args.suite, suite_kind=args.suite_kind)
                _print_json({"valid": True, "suite_version": normalized["suite_version"], "case_count": len(normalized["case_ids"]), "candidate_count": len(normalized["candidates"]), "bounds": normalized["bounds"], "dispatch_performed": False})
                return 0
            if args.pilot_command == "plan":
                if args.operator_input:
                    value = load_operator_input(args.operator_input)
                    source_hash = file_sha256(args.source_manifest)
                    constraint_hash = file_sha256(args.constraint_map)
                    plan = build_immutable_plan(
                        value, suite_path=args.suite, source_hash=source_hash, constraint_hash=constraint_hash,
                        source_manifest_path=args.source_manifest, constraint_map_path=args.constraint_map,
                        suite_kind=args.suite_kind,
                    )
                    write_immutable_plan(plan, args.out)
                    _print_json(plan)
                    return 0
                if args.suite_kind not in (None, "core"):
                    raise ValidationError("--suite-kind student requires --operator-input")
                pricing_snapshot = None
                if args.pricing_snapshot:
                    pricing_path = Path(args.pricing_snapshot)
                    if pricing_path.is_file():
                        pricing_snapshot = json.loads(pricing_path.read_text(encoding="utf-8"))
                    else:
                        pricing_snapshot = {"source": args.pricing_snapshot, "status": "operator_supplied_unverified"}
                plan = build_pilot_plan(
                    args.suite,
                    candidates=[parse_candidate_spec(value) for value in args.candidate],
                    repeats=args.repeats,
                    temperature=args.temperature,
                    seed=args.seed,
                    max_output_tokens=args.max_output_tokens,
                    timeout_seconds=args.timeout_seconds,
                    concurrency=args.concurrency,
                    max_retries=args.max_retries,
                    max_spend_minor=args.max_spend_minor,
                    currency=args.currency,
                    pricing_snapshot=pricing_snapshot,
                    prompt_template_revision=args.prompt_template_revision,
                )
                write_plan(plan, args.out)
                _print_json(plan)
                return 0
            plan = json.loads(Path(args.path if args.pilot_command == "show" else args.plan).read_text(encoding="utf-8"))
            if args.pilot_command == "show":
                display = dict(plan)
                display["current_blockers"] = validate_pilot_plan(plan)
                _print_json(display)
                return 0 if not display["current_blockers"] else 2
            _print_json(dispatch_preview(plan))
            require_dispatch_authorization(
                plan, allow_paid=args.allow_paid, source_manifest_path=args.source_manifest,
                constraint_map_path=args.constraint_map,
            )
            catalog_models = None
            if args.catalog:
                catalog_value = json.loads(Path(args.catalog).read_text(encoding="utf-8"))
                catalog_models = catalog_value.get("data", catalog_value) if isinstance(catalog_value, dict) else catalog_value
            _print_json(run_authorized_immutable_pilot(
                plan, suite_path=args.suite, output_dir=args.out,
                source_manifest_path=args.source_manifest, constraint_map_path=args.constraint_map,
                max_requests_per_day=args.max_requests_per_day, catalog_models=catalog_models,
            ))
            return 0
        if args.command == "review":
            store = SQLiteStore(args.db)
            try:
                if args.review_command == "export":
                    cases = ()
                    if args.include_prompt:
                        if not args.suite:
                            raise ValidationError("review export with --include-prompt requires --suite")
                        cases = load_suite(args.suite).cases
                    records = export_blind_review(store.list_attempts(args.run_id), include_prompt=args.include_prompt, cases=cases)
                    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                    Path(args.out).write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records), encoding="utf-8")
                    _print_json({"exported": len(records), "path": args.out, "blind": True})
                else:
                    records = [json.loads(line) for line in Path(args.input_path).read_text(encoding="utf-8").splitlines() if line.strip()]
                    bindings = review_bindings(store.list_attempts(args.run_id))
                    reviews = import_blind_reviews(records, expected_labels=bindings)
                    for review in reviews:
                        store.add_review(review)
                    _print_json({"imported": len(reviews), "run_id": args.run_id, "blind": True})
            finally:
                store.close()
            return 0
        if args.command == "router" and args.router_command == "recommend":
            if not args.draft:
                raise ModelLabError("routing recommendations require explicit --draft acknowledgement")
            source = Path(args.summary)
            value = json.loads(source.read_text(encoding="utf-8"))
            recommendations = value.get("routing_recommendations", [])
            constraint_map = load_constraint_map(args.constraint_map)
            constraint_evaluation = evaluate_constraints(constraint_map)
            constraint_evaluation["source_manifest_sha256"] = constraint_map.get("source_manifest_sha256")
            constraint_evaluation["constraint_map_sha256"] = file_sha256(args.constraint_map)
            activation_ready = bool(constraint_evaluation["activation_ready"]) and not any(item.get("synthetic") for item in recommendations)
            result = {"draft": True, "activation_ready": activation_ready, "recommendations": recommendations,
                      "constraint_evaluation": constraint_evaluation, "production_config_changed": False,
                      "limitations": ["Draft evidence cannot activate production routing."]}
            if args.out:
                _write_json(args.out, result)
            _print_json(result)
            return 0 if activation_ready else 2
        if args.command == "catalog":
            if args.catalog_command == "snapshot":
                from model_lab.providers.openrouter import snapshot_catalog
                out_path = snapshot_catalog(
                    out_dir=args.out,
                    base_url=args.base_url,
                    snapshot_date=args.date,
                )
                _print_json({"status": "ok", "path": str(out_path)})
                return 0
        if args.command in {"grade", "compare", "report", "audit"}:
            if args.command == "grade" and args.db and args.suite and args.run_id:
                suite = load_suite(args.suite)
                store = SQLiteStore(args.db)
                added = abstained = 0
                try:
                    for attempt in store.list_attempts(args.run_id):
                        grade = grade_attempt(suite.get(attempt.case_id), attempt)
                        try:
                            store.add_grade(grade)
                            added += 1
                        except IntegrityError:
                            abstained += 1
                finally:
                    store.close()
                _print_json({"graded": added, "already_present": abstained, "run_id": args.run_id})
                return 0
            if args.db and ((args.command == "compare" and args.left_run and args.right_run) or (args.command in {"report", "audit"} and args.run_id)):
                store = SQLiteStore(args.db)
                try:
                    if args.command == "audit":
                        run = store.get_run(args.run_id)
                        audit = {"run_id": args.run_id, "status": run.status, "attempts": store.count("attempts", args.run_id), "grades": store.count("grades", args.run_id), "reviews": store.count("reviews", args.run_id), "quarantined": len(store.quarantined()), "synthetic": run.environment.get("mode") == "offline_synthetic"}
                        if args.out:
                            _write_json(args.out, audit)
                        _print_json(audit)
                        return 0
                    if not args.suite:
                        raise ModelLabError(f"{args.command} with --db requires --suite")
                    suite = load_suite(args.suite)
                    if args.command == "report":
                        _, _, rows = _run_rows(suite, store, args.run_id)
                        output = args.out or "reports"
                        artifacts = write_report_bundle(rows, output, metadata={"run_id": args.run_id, "suite_version": suite.suite_version, "suite_hash": suite.source_hash})
                        _print_json({"run_id": args.run_id, "artifacts": {key: str(value) for key, value in artifacts.items()}})
                        return 0
                    if not args.left_run or not args.right_run:
                        raise ModelLabError("compare with --db requires --left-run and --right-run")
                    _, left_grades, _ = _run_rows(suite, store, args.left_run)
                    _, right_grades, _ = _run_rows(suite, store, args.right_run)
                    comparison = compare_grades(left_grades, right_grades, cases=suite.cases)
                    value = comparison.to_dict()
                    if args.out:
                        _write_json(args.out, value)
                    _print_json(value)
                    return 0
                finally:
                    store.close()
            if not args.summary:
                raise ModelLabError(f"{args.command} requires --summary for the offline workflow")
            value = json.loads(Path(args.summary).read_text(encoding="utf-8"))
            _print_json({"command": args.command, "summary": value, "synthetic": True})
            return 0
    except (ModelLabError, OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"model-lab: error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
