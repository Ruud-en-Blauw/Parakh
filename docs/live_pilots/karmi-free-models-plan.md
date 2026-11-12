# Karmi free-model student pilot: plan (not executed)

**Status: planning only.** Nothing in this document has been run live. No provider call, catalog
snapshot, or key read has happened for this plan. Every `[OWNER: ...]` field is a decision the
owner must make before any run. Read `OPENROUTER_PILOT.md` first; its safety rules apply here.

## 1. Scope

| Item | Value |
| --- | --- |
| Suite | `benchmarks_student/cases.jsonl` (49 synthetic cases) |
| Train split | 31 cases |
| Calibration split | 9 cases (development total: **40** = 31 + 9) |
| Holdout split | 9 cases, **untouched**; no plan in this document selects them |
| Candidates | Five OpenRouter free models from the Karmi registry (below) |

Candidate identifiers. The identifier's revision slot carries the OpenRouter `:free` variant, so
the model is `vendor/model` and the revision is `free`:

```
openrouter:cohere/north-mini-code:free
openrouter:liquid/lfm-2.5-2.6b:free
openrouter:nvidia/nemotron-3-ultra-550b-a55b:free
openrouter:nvidia/nemotron-3.5-lightning:free
openrouter:poolside/laguna-xs-2.1:free
```

The registry file `docs/live_pilots/registry/model_registry-2026-10-19.json` lists seven models.
The two not used here are `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free` and
`nvidia/nemotron-3-super-120b-a12b:free`. Add them only with an explicit owner decision.

## 2. Request math

Assumes `retry_limit = 0`. Each retry adds up to one request per case.

| Scenario | Models | Cases | Repeats | Requests | At 50 requests/day |
| --- | --- | --- | --- | --- | --- |
| Full study, 3 repeats | 5 | 40 | 3 | 5 × 40 × 3 = **600** | 600 / 50 = **12 days** |
| Full study, 1 repeat | 5 | 40 | 1 | 5 × 40 × 1 = **200** | 200 / 50 = **4 days** |
| Smallest useful first run | 2 | 9 (calibration) | 1 | 2 × 9 × 1 = **18** | 18 / 50 = under 1 day |

The 50/day figure is the unfunded free-tier daily limit quoted in `OPENROUTER_PILOT.md` §1. It is a
planning number only; the owner sets the real cap (section 5).

### What the runner allows with these numbers

These day counts are planning arithmetic. The current runner does **not** execute them as
multi-day runs.

- `pilot run --max-requests-per-day N` refuses to start when the pending requests exceed `N`. The
  refusal message shows the math, for example `pending requests 4 + 4 = 8 exceed max_requests_per_day=7`.
- Under a cap of 50, the 18-request first run can start. The 200- and 600-request studies are refused
  in one invocation.
- Within a single invocation, dispatch stops cleanly at the cap. The run is marked `incomplete` and
  its evidence is kept. Re-invoking resumes only the cases without an attempt, but the start guard
  still applies on the next invocation. A later day's invocation therefore runs only if its
  remaining pending work fits under the cap.

**Consequence:** the 12-day and 4-day schedules need an owner decision on multi-day execution
(section 6) before they can run.

## 3. Key source: read first

- **Never use Karmi's beta OpenRouter key or account for this pilot.** The beta account is the
  hosted product's credential and must not carry research traffic.
- The owner chooses the key source. A separate OpenRouter account dedicated to this pilot is the
  expected option. `[OWNER: key source and account]`
- Put the key in the untracked local file `.env.openrouter` (`OPENROUTER_API_KEY=...`) or in the
  `OPENROUTER_API_KEY` environment variable. The environment variable is checked first. The file is
  searched in the current directory, its parents, and the repository root.
- Never paste the key into chat, a plan, a YAML input, a log, or a commit. Plans and reports never
  contain it, and the adapter's `repr` does not show it.
- Free models only: every candidate must end in `:free`. The runner refuses other models before any
  dispatch. The adapter also aborts if a response reports a non-zero cost.

## 4. Commands the owner runs

Check each flag with `model-lab pilot <cmd> --help` before use. These commands were written without
being run.

```
# 0. Catalog snapshot (makes a network call; owner-run only)
model-lab catalog snapshot --date YYYY-MM-DD --out docs/live_pilots/catalog

# 1. Operator input: copy the template, set candidates, pricing, budget, authorization
#    and "suite_kind: student". Then validate without dispatch.
model-lab pilot validate-input --input <owner-input.yaml> --suite benchmarks_student/cases.jsonl --suite-kind student

# 2. Freeze the plan (hashes the suite, sources, constraints, pricing and candidates)
model-lab pilot plan --operator-input <owner-input.yaml> --suite benchmarks_student/cases.jsonl \
  --suite-kind student --out lab-data/live-pilots/<plan>.json

# 3. Dispatch. --allow-paid is the gate's name; it is still required for free models.
model-lab pilot run --plan lab-data/live-pilots/<plan>.json --suite benchmarks_student/cases.jsonl \
  --allow-paid --catalog docs/live_pilots/catalog/openrouter-free-<date>.json \
  --max-requests-per-day <OWNER CAP> --out lab-data/live-pilots/<dir>
```

The catalog snapshot keeps only zero-priced models, so it cannot show a model that became paid. The
runner's check at dispatch is the adapter's returned-cost check. For a stronger catalog check, pass
the raw `/models` response (`{"data": [...]}`) to `--catalog`. `pilot run` accepts that shape too.

### Smallest useful first run

2 models × 9 calibration cases × 1 repeat = **18 requests**. Set `case_selection.split: calibration`,
keep `repeats: 1` and `retry_limit: 0`, and pick any two of the five candidates. Use a cap of at least
18. This is the only run this plan supports in one invocation under the current guard.

## 5. Owner blanks

| Decision | Value |
| --- | --- |
| Daily request cap (`--max-requests-per-day`) | `[OWNER: ____ requests/day]` |
| Candidate selection (2 to 5 of the five) | `[OWNER: ____]` |
| Repeats (1 or 3) | `[OWNER: ____]` |
| Approved maximum spend | `[OWNER: ____ USD]`. The schema requires a positive ceiling even though free-model bounds compute to 0. |
| Operator identity and `authorized_at` | `[OWNER: ____]` |
| Key source and account (section 3) | `[OWNER: ____]` |
| Catalog snapshot date | `[OWNER: ____]` |

## 6. Open decision: multi-day execution

The start guard, as built, blocks any invocation whose pending work exceeds the cap. That makes
the 12-day schedule impossible. Two options:

- **A. Keep the guard; split the study into day-sized frozen plans.** Each day gets its own plan
  hash and its own evidence set. Nothing resumes across days, and a merge step is needed later.
- **B. Relax the start guard** so one invocation starts, dispatches up to the cap, stops cleanly as
  `incomplete`, and resumes the pending cases on a later day under the same plan hash. This changes
  a safety gate and needs owner sign-off. It is the option that matches the 12-day math.

Recommendation: **B**, after owner approval. It keeps one plan and one evidence set. This plan does
not implement either option.

## 7. Evidence and reporting

- Report VERIFIED and NOT VERIFIED separately. Offline proofs live in `tests/test_student_pilot.py`.
- Live results are NOT VERIFIED until the owner runs step 3 and the results are read back.
- Results go into `docs/live_pilots/runs/` or `lab-data/` (gitignored). Raw attempts are
  append-only. Do not edit them.
