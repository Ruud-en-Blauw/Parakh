#!/usr/bin/env python3
"""Capped capability probe of every free OpenRouter chat model on a few student cases.

NOT the full pilot: OpenRouter's free tier allows only 50 requests/day, so this spends at most
``--cap`` dispatches (every HTTP attempt counts, including a retry after a 429) on train cases.
Results are provisional. The key is read only through ``load_openrouter_api_key``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from model_lab.benchmark import load_suite, select_cases  # noqa: E402
from model_lab.domain.isolation import CandidateInput  # noqa: E402
from model_lab.providers.base import ProviderRequest  # noqa: E402
from model_lab.providers.openrouter import OpenRouterAdapter, load_openrouter_api_key  # noqa: E402

STUDENT = ROOT / "benchmarks_student"
# Not single chat models: a random-model router, music generators, a safety classifier.
EXCLUDED = {
    "openrouter/free",
    "google/lyria-3-pro-preview",
    "google/lyria-3-clip-preview",
    "nvidia/nemotron-3.5-content-safety:free",
    # HTTP 403 "only available on agentic harnesses": not callable through the chat API.
    "thinkingmachines/inkling:free",
    "thinkingmachines/inkling-small:free",
}
PROBE_CASES = ("STU-MATH-01", "STU-CODE-01", "STU-HIN-01")
UNSAFE_CODE = re.compile(r"\b(os|sys|subprocess|socket|shutil|pathlib|requests|urllib)\b|open\(|__import__|eval\(|exec\(")


def check(ref: dict, answer: str) -> bool:
    kind = ref["check_type"]
    if kind == "exact_numeric":
        tail = answer[-300:]
        numbers = [float(x) for x in re.findall(r"-?\d+(?:\.\d+)?", tail.replace(",", ""))]
        return any(abs(n - float(ref["expected"])) < 1e-5 for n in numbers)
    if kind == "devanagari":
        return bool(re.search(r"[\u0900-\u097F]", answer)) and sum(
            kw in answer for kw in ref.get("keywords", [])
        ) >= max(1, len(ref.get("keywords", [])) // 2)
    if kind == "python_exec":
        match = re.search(r"```(?:python)?\n(.*?)```", answer, re.S)
        code = match.group(1) if match else answer
        if UNSAFE_CODE.search(code):
            return False
        with tempfile.TemporaryDirectory() as tmp:
            try:
                done = subprocess.run(  # noqa: S603
                    [sys.executable, "-I", "-c", code + "\n" + ref["check_code"]],
                    cwd=tmp, capture_output=True, timeout=5, check=False,
                )
            except subprocess.TimeoutExpired:
                return False
        return done.returncode == 0
    raise ValueError(kind)


def write(out, args, dispatched, halt, aborted, rows, probe_cases) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "status": "provisional_capped_probe", "cap": args.cap, "dispatched": dispatched, "halt": halt,
        "aborted_models": aborted, "cases": list(probe_cases), "rows": rows,
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--cap", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--spacing", type=float, default=3.5, help="seconds between dispatches (20 req/min)")
    parser.add_argument("--max-tokens", type=int, default=800)
    parser.add_argument("--cases", default=",".join(PROBE_CASES))
    args = parser.parse_args()

    key = load_openrouter_api_key(args.env_file)
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    models = [m for m in catalog if m["id"] not in EXCLUDED]
    refs = json.loads((STUDENT / "reference_evaluations.json").read_text(encoding="utf-8"))
    suite = load_suite(STUDENT / "cases.jsonl")
    probe_cases = tuple(args.cases.split(","))
    cases = {c.case_id: c for c in select_cases(suite, case_ids=probe_cases)}
    dispatched = 0
    rows: list[dict] = []
    if args.out.exists():  # resume: keep earlier rows, never repeat a (model, case) pair
        previous = json.loads(args.out.read_text(encoding="utf-8"))
        rows = previous["rows"]
        dispatched = previous["dispatched"]
    aborted: dict[str, str] = {}
    halt = None

    for model in models:
        for case_id in probe_cases:
            if any(r["model_id"] == model["id"] and r["case_id"] == case_id for r in rows):
                continue
            if model["id"] in aborted:
                break
            if dispatched >= args.cap:
                halt = "request cap reached"
                break
            adapter = OpenRouterAdapter(
                model["id"], api_key=key, max_requests=1, max_retries=0,
                catalog_models=[{"id": model["id"], "pricing": {"prompt": "0", "completion": "0"}}],
            )
            request = ProviderRequest(
                candidate=CandidateInput(cases[case_id]), model=model["id"],
                parameters={"temperature": 0, "max_tokens": args.max_tokens}, timeout_seconds=90,
            )
            row = {"model_id": model["id"], "case_id": case_id, "category": refs[case_id]["category"]}
            for attempt in (1, 2):
                if dispatched >= args.cap:
                    row["error"] = "cap reached before retry"
                    break
                dispatched += 1
                time.sleep(args.spacing)
                adapter = OpenRouterAdapter(
                    model["id"], api_key=key, max_requests=1, max_retries=0,
                    catalog_models=[{"id": model["id"], "pricing": {"prompt": "0", "completion": "0"}}],
                )
                try:
                    response = adapter.generate(request)
                    text = response.text or ""
                    row.update(ok=True, passed=check(refs[case_id], text), finish_reason=response.finish_reason,
                               input_tokens=response.input_tokens, output_tokens=response.output_tokens,
                               latency_ms=round(response.completion_latency_ms or 0, 1),
                               returned_price_usd=response.metadata.get("returned_price_usd"),
                               empty=not text.strip())
                    row.pop("error", None)
                    break
                except Exception as err:  # noqa: BLE001 - timeouts etc. must not lose collected rows
                    message = f"{type(err).__name__}: {err}"
                    row.update(ok=False, passed=False, error=message[:200])
                    if "non-zero price" in message:
                        aborted[model["id"]] = message[:120]
                        break
                    if "HTTP 429" in message and attempt == 1:
                        time.sleep(10)  # bounded single resume; both dispatches count toward the cap
                        continue
                    break
            rows.append(row)
            write(args.out, args, dispatched, halt, aborted, rows, probe_cases)
            if "per-day" in row.get("error", "") or "free-models-per-day" in row.get("error", ""):
                halt = "daily free-model limit reported by provider"
                break
        if halt:
            break

    write(args.out, args, dispatched, halt, aborted, rows, probe_cases)
    print(f"dispatched={dispatched} cap={args.cap} halt={halt} rows={len(rows)} models={len(models)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
