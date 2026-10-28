---
name: verify
description: Run ModelLab's full verification sequence (doctor, unit, integration, e2e, smoke, demo) with the correct interpreter. Use before claiming a change is done, or when asked to verify/check/validate the repo.
---

Run these in order and report the real output of each — never summarize a
command you didn't run.

1. `.venv/bin/python -m pytest -q` — full unit suite. System `python3` does
   not have pytest installed; using it will fail with `No module named
   pytest`, which is an environment mismatch, not proof of a real failure.
2. `make PYTHON=.venv/bin/python test-release` — runs `lint` (ruff), `doctor`,
   `test-unit`, `test-integration`, `test-e2e`, `test-smoke` in sequence
   (see `Makefile`). This is the same gate CI runs (`.github/workflows/ci.yml`),
   modulo the interpreter path.
3. `make PYTHON=.venv/bin/python demo` — offline synthetic demo run against
   `benchmarks/seed_cases.jsonl`, writes to a disposable `lab-data/` dir.
   Confirms the CLI's end-to-end path still produces a report.

If you only changed one module, run the single matching test file first for
a fast loop (`tests/test_<area>.py`), then still run the full suite before
calling the task done — a change can pass its own test and break another
file's assumptions (shared shim modules, shared schemas).

Do not report "tests pass" without pasting the actual `N passed`/`M failed`
line.
