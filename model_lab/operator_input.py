"""Strict, offline validation for development-only live-pilot operator input.

This module deliberately does not dispatch providers.  It turns a complete,
human-edited YAML input into a hashable plan after validating the case split,
candidate identifiers, pricing, authorization, and conservative call/cost
bounds.  The parser accepts the small YAML subset used by the template and
JSON, which is valid YAML, without adding a runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, ROUND_CEILING, InvalidOperation
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, NoReturn

from .benchmark import load_suite, select_cases
from .errors import ValidationError
from .schemas import stable_hash, stable_json


SUPPORTED_PROVIDERS = frozenset({"ollama", "openrouter"})
REQUIRED_TOP_LEVEL = frozenset({
    "schema_version", "development_only", "exclusions", "case_selection",
    "candidates", "prompt_template_revision", "pricing_snapshot", "execution",
    "budget", "authorization",
})
TOP_LEVEL = REQUIRED_TOP_LEVEL | {"suite_kind"}
REQUIRED_EXCLUSIONS = frozenset({
    "customer_data", "calibration_holdout", "production_router_changes", "external_actions",
})
REQUIRED_RATES = frozenset({
    "input_per_million", "output_per_million", "cache_per_million",
    "reasoning_per_million", "tool_per_call",
})
REQUIRED_RATE_UNITS = frozenset({
    "input_per_million", "output_per_million", "cache_per_million",
    "reasoning_per_million", "tool_per_call",
})
REQUIRED_EXECUTION = frozenset({
    "max_input_tokens", "max_output_tokens", "timeout_seconds", "retry_limit",
    "concurrency", "repeats", "temperature", "seed", "judge_enabled", "red_team_enabled",
})


class OperatorInputError(ValidationError):
    """An operator input cannot safely become a frozen pilot plan."""


@dataclass(frozen=True)
class CandidateInput:
    identifier: str
    provider: str
    model: str
    display_label: str
    revision: str | None


def _fail(message: str) -> NoReturn:
    raise OperatorInputError(message)


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{name} must be a mapping")
    return value


def _exact_keys(value: Mapping[str, Any], allowed: set[str] | frozenset[str], name: str) -> None:
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        _fail(f"{name} contains unsupported fields: {', '.join(unknown)}")


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        _fail(f"{name} must be non-empty text")
    return value.strip()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _fail(f"{name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(f"{name} must be a non-negative integer")
    return value


def _number(value: Any, name: str, *, positive: bool = False) -> Decimal:
    if isinstance(value, bool) or value is None or isinstance(value, (dict, list)):
        _fail(f"{name} must be a number")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError):
        _fail(f"{name} must be a number")
    if not parsed.is_finite() or (parsed <= 0 if positive else parsed < 0):
        _fail(f"{name} must be {'positive' if positive else 'non-negative'}")
    return parsed


def _currency(value: Any, name: str) -> str:
    result = _nonempty_string(value, name).upper()
    if not re.fullmatch(r"[A-Z]{3}", result):
        _fail(f"{name} must be a three-letter currency code")
    return result


def _date(value: Any, name: str) -> str:
    result = _nonempty_string(value, name)
    try:
        date.fromisoformat(result)
    except ValueError:
        _fail(f"{name} must be an ISO date (YYYY-MM-DD)")
    return result


def parse_candidate_identifier(value: Any) -> tuple[str, str, str, str | None]:
    identifier = _nonempty_string(value, "candidate.identifier")
    parts = identifier.split(":", 2)
    if len(parts) not in (2, 3) or not parts[0] or not parts[1] or (len(parts) == 3 and not parts[2]):
        _fail("candidate identifier must use provider:model[:revision]")
    provider, model = parts[0], parts[1]
    if provider not in SUPPORTED_PROVIDERS:
        _fail(f"unsupported provider in candidate identifier: {provider}")
    revision = parts[2] if len(parts) == 3 else None
    if any(char.isspace() for char in identifier):
        _fail("candidate identifier cannot contain whitespace")
    return identifier, provider, model, revision


def _candidate(value: Any, index: int) -> CandidateInput:
    record = _mapping(value, f"candidates[{index}]")
    _exact_keys(record, {"identifier", "provider", "model", "display_label", "revision"}, f"candidates[{index}]")
    identifier, parsed_provider, parsed_model, parsed_revision = parse_candidate_identifier(record.get("identifier"))
    provider = _nonempty_string(record.get("provider"), f"candidates[{index}].provider")
    model = _nonempty_string(record.get("model"), f"candidates[{index}].model")
    revision_value = record.get("revision")
    revision = None if revision_value in (None, "") else _nonempty_string(revision_value, f"candidates[{index}].revision")
    if (provider, model, revision) != (parsed_provider, parsed_model, parsed_revision):
        _fail(f"candidates[{index}] provider/model/revision do not match identifier")
    return CandidateInput(identifier, provider, model, _nonempty_string(record.get("display_label"), f"candidates[{index}].display_label"), revision)


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return None
    if value.startswith(("'", '"')) and value[-1:] == value[0]:
        return value[1:-1]
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "Null", "~"}:
        return None
    if value.startswith(("[", "{")):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            _fail("inline YAML collections must use JSON syntax")
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _yaml_subset(text: str) -> Any:
    """Parse mappings/lists/scalars with indentation, sufficient for the template."""
    lines: list[tuple[int, str]] = []
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        if "\t" in raw:
            _fail("tabs are not permitted in operator YAML")
        content = raw.strip()
        if " #" in content:
            content = content.split(" #", 1)[0].rstrip()
        lines.append((len(raw) - len(raw.lstrip(" ")), content))

    def block(position: int, indent: int) -> tuple[Any, int]:
        if position >= len(lines) or lines[position][0] < indent:
            return {}, position
        is_list = lines[position][1] == "-" or lines[position][1].startswith("- ")
        result: Any = [] if is_list else {}
        while position < len(lines) and lines[position][0] == indent:
            content = lines[position][1]
            if is_list:
                if not content.startswith("-"):
                    break
                tail = content[1:].strip()
                position += 1
                if tail and ":" in tail and not tail.startswith(("'", '"')):
                    key, raw_value = tail.split(":", 1)
                    item: dict[str, Any] = {key.strip(): _parse_scalar(raw_value)} if raw_value.strip() else {key.strip(): None}
                    if position < len(lines) and lines[position][0] > indent:
                        nested, position = block(position, lines[position][0])
                        if raw_value.strip():
                            if not isinstance(nested, dict):
                                _fail("list mapping continuation must be a mapping")
                            item.update(nested)
                        elif isinstance(nested, dict):
                            item[key.strip()] = nested
                    result.append(item)
                elif tail:
                    result.append(_parse_scalar(tail))
                elif position < len(lines) and lines[position][0] > indent:
                    nested, position = block(position, lines[position][0])
                    result.append(nested)
                else:
                    result.append(None)
            else:
                if content.startswith("-") or ":" not in content:
                    _fail("expected a mapping entry")
                key, raw_value = content.split(":", 1)
                key = key.strip()
                if not key:
                    _fail("mapping key cannot be empty")
                position += 1
                if raw_value.strip():
                    result[key] = _parse_scalar(raw_value)
                elif position < len(lines) and lines[position][0] > indent:
                    result[key], position = block(position, lines[position][0])
                else:
                    result[key] = None
        return result, position

    parsed, position = block(0, lines[0][0] if lines else 0)
    if position != len(lines):
        _fail("invalid YAML indentation")
    return parsed


def load_operator_input(source: str | Path) -> dict[str, Any]:
    path = Path(source)
    text = path.read_text(encoding="utf-8") if path.exists() else str(source)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = _yaml_subset(text)
    if not isinstance(value, dict):
        _fail("operator input root must be a mapping")
    return value


def validate_operator_input(value: Mapping[str, Any], *, suite_path: str | Path, suite_kind: str | None = None) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail("operator input root must be a mapping")
    _exact_keys(value, TOP_LEVEL, "operator input")
    missing = sorted(REQUIRED_TOP_LEVEL - set(value))
    if missing:
        _fail("missing required fields: " + ", ".join(missing))
    if value["schema_version"] != "0.1.0":
        _fail("unsupported operator input schema_version")
    if value["development_only"] is not True:
        _fail("development_only must be true")

    declared_kind = value.get("suite_kind")
    if suite_kind is not None and declared_kind is not None and suite_kind != declared_kind:
        _fail(f"suite_kind mismatch: input declares {declared_kind!r} but caller passed {suite_kind!r}")
    effective_suite_kind = suite_kind or declared_kind or "core"
    if effective_suite_kind not in {"core", "student"}:
        _fail(f"unsupported suite_kind: {effective_suite_kind}")

    exclusions = _mapping(value["exclusions"], "exclusions")
    _exact_keys(exclusions, REQUIRED_EXCLUSIONS, "exclusions")
    if any(exclusions.get(key) is not True for key in REQUIRED_EXCLUSIONS):
        _fail("all development-only exclusions must be true")

    selection = _mapping(value["case_selection"], "case_selection")
    _exact_keys(selection, {"split", "case_ids"}, "case_selection")
    student_splits: tuple[str, ...] = ()
    if effective_suite_kind == "student":
        split_choice = selection.get("split")
        if split_choice in (None, "student", "train+calibration"):
            student_splits = ("train", "calibration")
        elif split_choice in ("train", "calibration"):
            student_splits = (split_choice,)
        else:
            _fail("student pilot case selection must use train, calibration, or student split")
    else:
        if selection.get("split") != "train":
            _fail("live pilot case selection must use the train split only")
    if selection.get("case_ids") not in (None, []):
        ids = selection["case_ids"]
        if not isinstance(ids, list) or any(not isinstance(item, str) for item in ids):
            _fail("case_selection.case_ids must be a list of identifiers")
        if any(item.startswith(("L1-04", "L1-05", "L2-04", "L2-05", "L3-04", "L3-05", "L4-04", "L4-05")) for item in ids):
            _fail("calibration/holdout cases cannot be selected for the live pilot")

    raw_candidates = value["candidates"]
    min_candidates = 2 if effective_suite_kind == "student" else 3
    if not isinstance(raw_candidates, list) or not min_candidates <= len(raw_candidates) <= 5:
        _fail(f"candidates must contain {min_candidates} to 5 candidate models")
    candidates = [_candidate(item, index) for index, item in enumerate(raw_candidates)]
    identifiers = [item.identifier for item in candidates]
    if len(identifiers) != len(set(identifiers)):
        _fail("candidate identifiers must be unique")

    prompt_template_revision = _nonempty_string(value["prompt_template_revision"], "prompt_template_revision")

    pricing = _mapping(value["pricing_snapshot"], "pricing_snapshot")
    _exact_keys(pricing, {"as_of", "currency", "source", "rate_units", "rates"}, "pricing_snapshot")
    snapshot_date = _date(pricing.get("as_of"), "pricing_snapshot.as_of")
    currency = _currency(pricing.get("currency"), "pricing_snapshot.currency")
    _nonempty_string(pricing.get("source"), "pricing_snapshot.source")
    rate_units = _mapping(pricing.get("rate_units"), "pricing_snapshot.rate_units")
    if set(rate_units) != set(REQUIRED_RATE_UNITS):
        _fail("pricing_snapshot.rate_units must define every price unit")
    expected_units = {
        "input_per_million": "currency_per_million_tokens",
        "output_per_million": "currency_per_million_tokens",
        "cache_per_million": "currency_per_million_tokens",
        "reasoning_per_million": "currency_per_million_tokens",
        "tool_per_call": "currency_per_call",
    }
    if any(rate_units[name] != expected_units[name] for name in expected_units):
        _fail("pricing_snapshot.rate_units contains unsupported or missing price units")
    rates = _mapping(pricing.get("rates"), "pricing_snapshot.rates")
    if set(rates) != set(identifiers):
        _fail("pricing_snapshot.rates must contain exactly one entry per candidate identifier")
    normalized_rates: dict[str, dict[str, Decimal]] = {}
    for identifier in identifiers:
        rate_record = _mapping(rates[identifier], f"pricing_snapshot.rates[{identifier}]")
        _exact_keys(rate_record, REQUIRED_RATES, f"pricing_snapshot.rates[{identifier}]")
        normalized_rates[identifier] = {key: _number(rate_record[key], f"{identifier}.{key}") for key in REQUIRED_RATES}

    execution = _mapping(value["execution"], "execution")
    _exact_keys(execution, REQUIRED_EXECUTION, "execution")
    normalized_execution = {
        "max_input_tokens": _positive_int(execution.get("max_input_tokens"), "execution.max_input_tokens"),
        "max_output_tokens": _positive_int(execution.get("max_output_tokens"), "execution.max_output_tokens"),
        "timeout_seconds": float(_number(execution.get("timeout_seconds"), "execution.timeout_seconds", positive=True)),
        "retry_limit": _nonnegative_int(execution.get("retry_limit"), "execution.retry_limit"),
        "concurrency": _positive_int(execution.get("concurrency"), "execution.concurrency"),
        "repeats": _positive_int(execution.get("repeats"), "execution.repeats"),
        "temperature": execution.get("temperature"),
        "seed": execution.get("seed"),
        "judge_enabled": execution.get("judge_enabled"),
        "red_team_enabled": execution.get("red_team_enabled"),
    }
    if normalized_execution["temperature"] is not None:
        _number(normalized_execution["temperature"], "execution.temperature")
    if normalized_execution["seed"] is not None:
        _nonnegative_int(normalized_execution["seed"], "execution.seed")

    if not isinstance(normalized_execution["judge_enabled"], bool) or not isinstance(normalized_execution["red_team_enabled"], bool):
        _fail("execution judge/red-team options must be boolean")
    if normalized_execution["judge_enabled"] or normalized_execution["red_team_enabled"]:
        _fail("paid judges and red-team calls are unsupported for the first development pilot")

    budget = _mapping(value["budget"], "budget")
    _exact_keys(budget, {"currency", "total_max_spend", "per_model_max_spend"}, "budget")
    budget_currency = _currency(budget.get("currency"), "budget.currency")
    if budget_currency != currency:
        _fail("pricing and budget currencies must match")
    total_max = _number(budget.get("total_max_spend"), "budget.total_max_spend", positive=True)
    per_model_raw = _mapping(budget.get("per_model_max_spend"), "budget.per_model_max_spend")
    if set(per_model_raw) != set(identifiers):
        _fail("budget.per_model_max_spend must contain exactly one entry per candidate")
    per_model = {identifier: _number(per_model_raw[identifier], f"per_model_max_spend[{identifier}]", positive=True) for identifier in identifiers}

    authorization = _mapping(value["authorization"], "authorization")
    _exact_keys(authorization, {"operator_identity", "authorized_at", "approved", "maximum_spend_authorized"}, "authorization")
    if authorization.get("approved") is not True:
        _fail("explicit operator authorization approved=true is required")
    _nonempty_string(authorization.get("operator_identity"), "authorization.operator_identity")
    authorized_at = _nonempty_string(authorization.get("authorized_at"), "authorization.authorized_at")
    try:
        datetime.fromisoformat(authorized_at.replace("Z", "+00:00"))
    except ValueError:
        _fail("authorization.authorized_at must be an ISO timestamp")
    authorized_max = _number(authorization.get("maximum_spend_authorized"), "authorization.maximum_spend_authorized", positive=True)
    if authorized_max > total_max:
        _fail("authorized maximum spend cannot exceed total_max_spend")

    suite = load_suite(suite_path)
    if effective_suite_kind == "student":
        if normalized_execution["concurrency"] != 1:
            _fail("student path requires concurrency=1")
        if selection.get("case_ids"):
            requested = tuple(selection["case_ids"])
            selected = select_cases(suite, case_ids=requested)
        else:
            selected = select_cases(suite, splits=list(student_splits))
        if any(case.split not in student_splits for case in selected):
            _fail("holdout or unselected-split cases cannot be selected for the live pilot")
    else:
        selected = select_cases(suite, splits=["train"])
        if selection.get("case_ids"):
            requested = tuple(selection["case_ids"])
            selected = select_cases(suite, case_ids=requested)
        if len(selected) != 36 or any(case.split != "train" for case in selected):
            _fail("pilot must select exactly the 36 train cases")

    base_calls = len(selected) * len(candidates) * normalized_execution["repeats"]
    maximum_calls = base_calls * (normalized_execution["retry_limit"] + 1)
    per_call_minor: dict[str, int] = {}
    for candidate in candidates:
        rate = normalized_rates[candidate.identifier]
        # This intentionally reserves the maximum input once as uncached and
        # once as cached, plus maximum visible/reasoning output. It is a
        # conservative admission bound, not an assertion about actual usage.
        token_cost = (
            Decimal(normalized_execution["max_input_tokens"]) * (rate["input_per_million"] + rate["cache_per_million"])
            + Decimal(normalized_execution["max_output_tokens"]) * (rate["output_per_million"] + rate["reasoning_per_million"])
        ) / Decimal(1_000_000)
        tool_cost = rate["tool_per_call"]
        per_call_minor[candidate.identifier] = int((token_cost + tool_cost).quantize(Decimal("0.01"), rounding=ROUND_CEILING) * 100)
    per_candidate_bound = {identifier: (per_call_minor[identifier] * len(selected) * normalized_execution["repeats"] * (normalized_execution["retry_limit"] + 1)) for identifier in identifiers}
    total_bound_minor = sum(per_candidate_bound.values())
    total_max_minor = int((total_max * 100).quantize(Decimal("1"), rounding=ROUND_CEILING))
    per_model_max_minor = {identifier: int((per_model[identifier] * 100).quantize(Decimal("1"), rounding=ROUND_CEILING)) for identifier in identifiers}
    if total_bound_minor > total_max_minor:
        _fail("conservative spend bound exceeds total_max_spend")
    authorized_max_minor = int((authorized_max * 100).quantize(Decimal("1"), rounding=ROUND_CEILING))
    if total_bound_minor > authorized_max_minor:
        _fail("conservative spend bound exceeds the authorized maximum spend")
    for identifier, bound in per_candidate_bound.items():
        if bound > per_model_max_minor[identifier]:
            _fail(f"conservative spend bound exceeds per-model ceiling for {identifier}")

    return {
        "schema_version": "0.1.0",
        "development_only": True,
        "suite_kind": effective_suite_kind,
        "exclusions": dict(exclusions),
        "suite_version": suite.suite_version,
        "suite_hash": suite.source_hash,
        "case_ids": [case.case_id for case in selected],
        "candidates": [{"identifier": c.identifier, "provider": c.provider, "model": c.model, "display_label": c.display_label, "revision": c.revision} for c in candidates],
        "prompt_template_revision": prompt_template_revision,
        "pricing_snapshot": {"as_of": snapshot_date, "currency": currency, "source": pricing["source"], "rate_units": dict(rate_units), "rates": {identifier: {key: str(val) for key, val in normalized_rates[identifier].items()} for identifier in identifiers}},
        "execution": normalized_execution,
        "budget": {"currency": currency, "total_max_spend": str(total_max), "per_model_max_spend": {key: str(value) for key, value in per_model.items()}},
        "authorization": {"operator_identity": authorization["operator_identity"], "authorized_at": authorized_at, "approved": True, "maximum_spend_authorized": str(authorized_max)},
        "bounds": {"base_calls": base_calls, "maximum_calls": maximum_calls, "per_call_bound_minor": per_call_minor, "per_candidate_bound_minor": per_candidate_bound, "total_bound_minor": total_bound_minor, "currency": currency},
    }


def build_immutable_plan(value: Mapping[str, Any], *, suite_path: str | Path, source_hash: str | None = None,
                         constraint_hash: str | None = None, source_manifest_path: str | None = None,
                         constraint_map_path: str | None = None, created_at: str | None = None,
                         suite_kind: str | None = None) -> dict[str, Any]:
    validated = validate_operator_input(value, suite_path=suite_path, suite_kind=suite_kind)
    plan = dict(validated)
    plan["plan_version"] = "0.1.0"
    plan["frozen_at"] = created_at or datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    plan["operator_input_hash"] = stable_hash(value)
    plan["source_hash"] = source_hash
    plan["constraint_hash"] = constraint_hash
    plan["source_manifest_path"] = source_manifest_path
    plan["constraint_map_path"] = constraint_map_path
    plan["case_set_hash"] = stable_hash(plan["case_ids"])
    plan["model_configuration_hash"] = stable_hash({"candidates": plan["candidates"], "execution": plan["execution"], "prompt_template_revision": plan["prompt_template_revision"]})
    plan["pricing_snapshot_hash"] = stable_hash(plan["pricing_snapshot"])
    plan["immutable"] = True
    plan["plan_hash"] = hashlib.sha256(stable_json(plan).encode("utf-8")).hexdigest()
    return plan


def write_immutable_plan(plan: Mapping[str, Any], destination: str | Path) -> None:
    if plan.get("immutable") is not True or not plan.get("plan_hash"):
        _fail("only a hashed immutable plan can be written")
    path = Path(destination)
    serialized = json.dumps(plan, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            _fail("immutable plan already exists with different content")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialized, encoding="utf-8")
