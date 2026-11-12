# OpenRouter Free Pilot Operator Execution Guide

> **Reality check (updated; still not run live):**
> * **Wired and tested offline** (`tests/test_student_pilot.py`, fake provider and fake transport only):
>   * `suite_kind`: `core` (default, the strict 36-train-case gate, unchanged) or `student`. The student path
>     requires an explicit `suite_kind: student` (or `--suite-kind student`). It selects train and/or
>     calibration cases, never holdout, and needs 2 to 5 candidates. Use `--suite benchmarks_student/cases.jsonl`.
>   * OpenRouter candidates use `providers/openrouter.py::OpenRouterAdapter`. Only `:free` models are
>     accepted. The key comes from `OPENROUTER_API_KEY`, then `.env.openrouter`, and is never printed or stored.
>     `--catalog <snapshot-or-/models.json>` runs a zero-price check before any dispatch. The adapter also
>     aborts on any non-zero returned cost.
>   * `pilot run --max-requests-per-day N` refuses to start when pending requests exceed N, and shows the math.
>     A cap reached mid-run stops cleanly and marks the run `incomplete`.
>   * Resume: a `completed` run is reused with no provider call. Other runs dispatch only cases with no
>     attempt yet. A stored run from a different plan hash is refused.
> * **Still true caveats:**
>   * Nothing here has been run live. No network call was made while building this. The owner's live run is
>     not verified until it is executed and read back.
>   * The start guard blocks any invocation whose pending work exceeds the cap. A multi-day schedule (the 12-day
>     and 4-day math in `karmi-free-models-plan.md`) is therefore not executable yet. Choosing between day-sliced
>     plans and a relaxed guard is an owner decision (`karmi-free-models-plan.md` §6).
>   * Retries count against the cap. A case is held back when its full retry budget would not fit under the cap.
>   * Crash safety is partial. A crash between reserving a case and recording its attempt leaves a reservation
>     that the resume will hit as a duplicate. This is not exercised by a test and needs manual handling.
>   * Judges and red-team are disabled. Concurrency is 1.
>   * Some commands below (e.g. `grade` arguments) were written without being run. Confirm each against
>     `model-lab <cmd> --help` before use.


## Status and Safety Boundary
This document defines the strict, reproducible procedure for running a live evaluation pilot using OpenRouter's free-tier models once credentials and verified request caps are established.

**CRITICAL RULES:**
1. **NEVER run anything live during development or testing.** Live calls require explicit, out-of-band operator authorization and `--allow-paid` confirmation.
2. **FREE MODELS ONLY:** Only models ending in `:free` with verified `$0.00` catalog prompt and completion pricing are permitted. Any non-zero cost returned or declared immediately aborts execution.
3. **NO KEY IN LOGS:** The OpenRouter API key is read strictly from an untracked local file `.env.openrouter` (variable `OPENROUTER_API_KEY`). It must NEVER be checked into git, logged, printed to stdout/stderr, or included in report metadata.

---

## 1. Rate Limits and Request Cap Proposal Formula

OpenRouter enforces two tiers of rate limits on free-tier models (`vendor/model:free`):

| Account Tier | Credit Requirement | Requests Per Minute (RPM) | Requests Per Day (RPD) |
|---|---|---|---|
| Free / Unfunded | Less than 10 credits purchased | 20 RPM | 50 RPD |
| Funded Free-Tier | At least 10 credits purchased (>= $10) | 20 RPM | 1,000 RPD |

Before executing any pilot plan, query the key limits endpoint to inspect available quota:

```bash
curl -s -X GET "https://openrouter.ai/api/v1/key" \
  -H "Authorization: Bearer $(grep OPENROUTER_API_KEY .env.openrouter | cut -d= -f2)"
```

The response includes:
```json
{
  "data": {
    "is_free_tier": true,
    "free_model_daily_requests": {
      "limit": 50,
      "used": 0,
      "remaining": 50
    }
  }
}
```

### Request Cap Proposal Formula

Let:
- $M$: Number of candidate models evaluated (e.g. 3 models).
- $C_{\text{train}}$: Number of train cases (e.g. 31 cases in `benchmarks_student/train.jsonl`).
- $C_{\text{calib}}$: Number of calibration cases (e.g. 9 cases in `benchmarks_student/calibration.jsonl`).
- $R$: Number of repeats per case (fixed to 3 repeats for statistical stability).
- $Q_{\text{rem}}$: Remaining daily requests from `GET /key` (`free_model_daily_requests.remaining`).
- $\alpha$: Safety headroom multiplier for retries and transient failures ($\alpha = 0.80$).

The planned dispatch requests $N_{\text{dispatch}}$ across train + calibration is:
$$N_{\text{dispatch}} = M \times (C_{\text{train}} + C_{\text{calib}}) \times R$$

Example with 3 models, 40 cases (31 train + 9 calibration), and 3 repeats:
$$N_{\text{dispatch}} = 3 \times (31 + 9) \times 3 = 360 \text{ requests}$$

The proposed hard request cap $K_{\text{cap}}$ enforced by `OpenRouterAdapter` must satisfy:
$$K_{\text{cap}} = \min(\lfloor Q_{\text{rem}} \times \alpha \rfloor, N_{\text{dispatch}} + \text{retry\_allowance})$$

**Constraint:**
- If $N_{\text{dispatch}} > Q_{\text{rem}}$, the pilot MUST NOT be dispatched in a single day. The operator must either fund the account with >= $10 credits to raise the daily limit to 1,000 RPD, or split the evaluation across multiple UTC days.

---

## 2. Step-by-Step Execution Workflow

### Step 1: Catalog Snapshot
Fetch the latest zero-price catalog of verified free models:

```bash
~/.venvs/parakh/bin/python -m model_lab catalog snapshot \
  --out docs/live_pilots/catalog \
  --date $(date +%Y-%m-%d)
```
Verify that candidate models appear in `docs/live_pilots/catalog/openrouter-free-YYYY-MM-DD.json` with `prompt_price_usd: "0"` and `completion_price_usd: "0"`.

### Step 2: Prepare Operator Input YAML
Copy the operator template to a secure location outside version control:

```bash
mkdir -p /secure/pilot
cp docs/live_pilots/operator-input.template.yaml /secure/pilot/operator-input.yaml
```

Configure `/secure/pilot/operator-input.yaml`:
- Candidate models: e.g. `openrouter/meta-llama/llama-3.3-70b-instruct:free`, `openrouter/google/gemini-2.0-flash-lite-preview:free`, `openrouter/qwen/qwen-2.5-72b-instruct:free`.
- Repeats: 3.
- Split: train + calibration only. Holdout cases remain strictly unreferenced.
- Ceilings: Currency USD, spend ceiling $0.00 (free tier).
- Request cap: Derived from the proposal formula above.

Validate the operator input without executing calls:
```bash
~/.venvs/parakh/bin/python -m model_lab pilot validate-input \
  --input /secure/pilot/operator-input.yaml \
  --suite benchmarks_student/cases.jsonl
```

### Step 3: Freeze Immutable Pilot Plan
Generate the cryptographically hashed, immutable execution plan:

```bash
~/.venvs/parakh/bin/python -m model_lab pilot plan \
  --operator-input /secure/pilot/operator-input.yaml \
  --suite benchmarks_student/cases.jsonl \
  --out /secure/pilot/plan-frozen.json

~/.venvs/parakh/bin/python -m model_lab pilot show /secure/pilot/plan-frozen.json
```
Verify the plan displays exact candidate model IDs, case counts, repeat counts, seed, zero spend bound, and input hashes.

### Step 4: Dispatch Train + Calibration (3 Repeats)
Execute the authorized pilot with `--allow-paid` gate confirmation:

```bash
~/.venvs/parakh/bin/python -m model_lab pilot run \
  --plan /secure/pilot/plan-frozen.json \
  --allow-paid \
  --suite benchmarks_student/cases.jsonl \
  --out lab-data/live-pilots/run-01
```

During execution:
- Concurrency is bounded (`concurrency: 1` recommended for free-tier rate limits).
- Every attempt enforces 20 RPM backoff and Retry-After headers if 429 occurs.
- If request count reaches $K_{\text{cap}}$, dispatch halts immediately.
- If any returned token cost > 0, dispatch aborts.

### Step 5: Score and Build Provisional Registry
Score the resulting attempts using the automated evaluation checks:

```bash
~/.venvs/parakh/bin/python -m model_lab grade \
  --db lab-data/live-pilots/run-01/store.sqlite \
  --suite benchmarks_student/cases.jsonl \
  --run-id run-01
```

Build the `model-registry.v1` provisional JSON using `model_lab.providers.registry`:
```python
from model_lab.providers.registry import build_model_registry, build_shortlist_per_class

# model_evaluations populated from run grades
registry = build_model_registry(model_evaluations, provisional=True)
shortlist = build_shortlist_per_class(registry, top_n=1)
# Writes provisional model_registry.v1
```

### Step 6: Unlock Sealed Holdout for Shortlist Only
The 9 sealed holdout cases in `benchmarks_student/holdout.jsonl` (hash-verified against `benchmarks_student/holdout.sha256`) are ONLY evaluated on the top shortlist model per class (`economy`, `balanced`, `advanced`).
Non-shortlisted candidates never touch the holdout split.

Re-verify the holdout hash before evaluation:
```bash
sha256sum -c benchmarks_student/holdout.sha256
```
Once holdout evaluation is verified, promote the winning model configuration in the registry from `provisional: true` to `provisional: false` (`score_status: "verified"`).
