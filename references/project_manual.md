# Autonumerics Project Manual

## Directory Structure

```
workspace/{problem_slug}/
├── problem.md                        # NL problem description (read-only after creation)
├── problem_spec.json                 # structured specification — SDE or PDE (formulator writes, shared read-only)
├── STATE.md                          # YAML frontmatter: phase / problem_type / plans
├── plans/
│   ├── {id}-{plan_slug}/             # one directory per numerical scheme or discretization
│   │   ├── SOLUTION.md               # scheme strategy + results + <review> block
│   │   ├── solver.py                 # numerical solver (solver writes, evaluator reads/imports)
│   │   └── evaluate.py              # accuracy evaluator (evaluator writes and runs)
│   └── ...
└── REPORT.md                         # conductor final summary
```

## Pipeline

```
problem.md
    |
    v
formulator          reads problem.md → classifies as SDE or PDE → writes problem_spec.json
    |
    v
plan-creator-sde    (if SDE) reads problem_spec.json → creates EM/Milstein plans
plan-creator-pde    (if PDE) reads problem_spec.json → creates FD/spectral/implicit plans
    |
    +-- plan_1: solver-{sde|pde} <-> evaluator-{sde|pde}
    +-- plan_2: solver-{sde|pde} <-> evaluator-{sde|pde}
    +-- ...
```

Phases:
- **init**:     problem.md exists, waiting for formulator
- **running**:  solver ↔ evaluator iterating toward score 10
- **done**:     best plan identified, REPORT.md written
- **blocked**:  terminal — the formulator's requirements ledger did not carry every constraint in problem.md across into problem_spec.json, so the conductor stopped before creating plans (see *Requirements Ledger*)

## Never read `benchmark/`

Nothing in this pipeline may read anything under `benchmark/` — not `problems.py`, not `verify.py`, not `manifest.json`, not `results/`. That directory holds the independently-authored ground truth this pipeline is graded against, and the whole value of that grade is that it was produced without seeing the answer. An agent that reads the reference it is about to be scored on turns an independent check into a self-check, silently and irreversibly.

This is enforced by permission deny rules during benchmark runs, but treat it as a rule you follow rather than a fence you test. If you find yourself lacking information — a scoring criterion, a reference value — the answer is in `problem.md` or `problem_spec.json`, or it is genuinely missing and you should say so. It is never in `benchmark/`.

## File Handoff Protocol

| File | conductor | formulator | plan-creator | solver | evaluator |
|---|---|---|---|---|---|
| `problem.md` | — | R | R | R | R |
| `problem_spec.json` | R (eq_type only) | W | R | R | R |
| `plans/{id}/SOLUTION.md` | R | — | W | W | W (overwrite `<review>` block) |
| `plans/{id}/solver.py` | — | — | — | WX | R (import) |
| `plans/{id}/evaluate.py` | — | — | — | — | WX |
| `STATE.md` | W | — | — | — | — |
| `REPORT.md` | W | — | — | — | — |

- **R** = read-only, **W** = writable, **X** = Bash execute
- `<review>` block: keep only one copy at the end of SOLUTION.md; evaluator overwrites it on every pass

## Score Convention

Score 10 = terminal condition. Both SDE and PDE evaluators use the same 1–10 scale.

**SDE** (`evaluator-sde`):
- **10**: `variance_rel_err < 10%` AND (`|exact_mean| < 0.01` OR `mean_rel_err < 5%`)
- **7–9**: variance passes, mean slightly above threshold
- **4–6**: variance fails, code ran cleanly
- **1–3**: non-finite outputs or crash

**PDE** (`evaluator-pde`):
- **10**: `relative_L2_error < 1%`
- **8**: relative_L2_error < 5%
- **6**: relative_L2_error < 20%
- **4**: relative_L2_error < 50%
- **2**: ran but error ≥ 50%
- **1**: crash
- **7**: no analytic solution, but code ran correctly

## Evaluation Thresholds

**SDE** (stored in `problem_spec.json → evaluation_thresholds`):
```
variance_relative_error  < 0.10  (10%)
mean_relative_error      < 0.05   (5%)   — skipped when |exact_mean| < 0.01
```

**PDE** (stored in `problem_spec.json → evaluation_thresholds`):
```
rel_l2_err_max = 0.01  (1%)
```

**SDE stability problems** (no closed-form moments — blow-up / domain tests). The moment tolerances do not apply; the criterion is `evaluation_thresholds.stability_check`:
```
{"type": "finite"}                      every terminal state finite (no Inf/NaN)
{"type": "domain", "abs_bound": 1.0}    additionally |X(T)| < abs_bound for every path
null                                    not a stability problem
```
A spec with `has_analytic_solution: false` and `stability_check: null` has no scoreable criterion at all and is a formulator error.

## Requirements Ledger

`problem_spec.json` carries a top-level `requirements` array: one entry per constraint in `problem.md`, each with a verbatim `quote`, a `status` (`mapped` / `ambiguous` / `dropped`) and the `spec_path` that carries it.

It exists because nothing downstream of the formulator ever reads `problem.md`. A parameter, boundary condition or scoring rule that fails to cross into the spec does not become an unrecorded detail — it stops existing, and the pipeline solves an easier problem than the one it was given, scoring itself 10/10 for doing so. The ledger turns that from a silent omission into a visible one.

The conductor **halts the run at `phase: blocked`** if `requirements` is missing or empty, if any entry is `dropped`, or if a `mapped` entry has no `spec_path`. `ambiguous` entries do not halt: they record where `problem.md` was underspecified and the formulator made a documented standard choice, and they are surfaced in REPORT.md so the reader can see where the solved problem may differ from the one they described.

## Python Environment

Use `uv` with the project's `.venv` (Python 3.13 or newer). Installed packages: `numpy`, `scipy`.

To run code: `uv run python workspace/{problem_slug}/plans/{id}-{plan_slug}/solver.py`
