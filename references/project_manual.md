# Autonumerics Project Manual

## Directory Structure

```
workspace/{problem_slug}/
├── problem.md                        # NL problem description (read-only after creation)
├── problem_spec.json                 # structured specification — SDE or PDE (formulator writes, shared read-only)
├── STATE.md                          # YAML frontmatter: phase / problem_type / plans
├── plans/
│   ├── {id}-{plan_slug}/             # one directory per numerical scheme or discretization
│   │   ├── SOLUTION.md               # scheme strategy + results + <metrics> + <review> block
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
formulator          reads problem.md → classifies as SDE or PDE → derives ground truth OR a
                    verification plan (surrogate / manufactured solution / invariants)
                    → writes problem_spec.json
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
| `plans/{id}/SOLUTION.md` | R | — | W | W | W (overwrite `<metrics>` + `<review>` blocks) |
| `plans/{id}/solver.py` | — | — | — | WX | R (import) |
| `plans/{id}/evaluate.py` | — | — | — | — | WX |
| `plans/*/solver.py` (siblings) | — | — | — | — | R (import, cross-plan consensus only) |
| `STATE.md` | W | — | — | — | — |
| `REPORT.md` | W | — | — | — | — |

- **R** = read-only, **W** = writable, **X** = Bash execute
- `<review>` block: keep only one copy at the end of SOLUTION.md; evaluator overwrites it on every pass
- `<metrics>` block: sits immediately before `<review>`; evaluator overwrites it on every pass
- An evaluator may **import and run** a sibling plan's `solver.py` for cross-plan consensus
  (§11/§22 of `verification_manual.md`), but may never modify it

## Score Convention

Score 10 = terminal condition. Both SDE and PDE evaluators use the same 1–10 scale.

Which rubric applies depends on the **provenance** of the reference (see below). The rubrics here
are the *analytic* path — when there is no closed form, the evaluator uses the corresponding rubric
in `verification_manual.md` (§12 for PDE, §23 for SDE).

**SDE, analytic path** (`evaluator-sde`, `analytic_moments.has_analytic_solution == true`):
- **10**: `variance_rel_err < 10%` AND (`|exact_mean| < 0.01` OR `mean_rel_err < 5%`), **and the
  estimates are MC-resolved** (§14)
- **7–9**: variance passes, mean slightly above threshold
- **6**: MC-inconclusive — error bars too wide to decide; raise `num_paths`
- **4–5**: variance fails, code ran cleanly
- **1–3**: non-finite outputs or crash

**PDE, analytic path** (`evaluator-pde`, `analytic_solution != null`):
- **10**: `relative_L2_error < 1%` AND observed order ≥ floor AND no structural gate violated
- **8**: relative_L2_error < 5%
- **7**: error < 1% but observed order below the floor (accurate on one grid, not converging)
- **6**: relative_L2_error < 20%
- **4**: relative_L2_error < 50%
- **3**: a hard structural gate violated (div ≠ 0, negative density) even if accurate
- **2**: ran but error ≥ 50%
- **1**: crash

## Provenance

Ground truth is not binary. Every score carries a `provenance` tag naming the evidence it rests on:

| Tag | Meaning |
|---|---|
| `analytic` | Closed-form solution / exact moments |
| `surrogate` | Deterministic reference derived by the evaluator (SDE moment ODE, Kolmogorov solve, stationary density) |
| `manufactured` | Manufactured solution or degenerate-limit check passed at design order |
| `self_convergence` | Richardson/GCI estimate only — no independent reference |
| `none` | No test could be run (caps the score at 2) |

A 10 earned against a closed form and a 10 earned by self-convergence are **different claims**. The
tag must survive into `<metrics>`, `SOLUTION.md` and `REPORT.md`. See `verification_manual.md` for
the full ladder of evidence and every test available at each tier.

## The `<metrics>` Block

The evaluator writes this immediately before the `<review>` block on every pass. The conductor
parses it to rank plans when scores tie.

```
<metrics>
provenance: analytic | surrogate | manufactured | self_convergence | none
estimated_rel_error: 3.10e-03
error_is_estimate: false
observed_order: 1.98
order_floor: 1.50
invariants_ok: true
constraints_ok: true
mc_se_rel: 0.0041          # SDE only
resolved: true             # SDE only
wall_time_s: 12.4
</metrics>
```

`estimated_rel_error` is always populated — it is the primary tiebreaker, so an evaluator with no
true reference must still supply the Richardson estimate. Omit fields that do not apply.

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

## Solver Contracts

Both contracts take optional arguments that exist **only** so the evaluator can run verification
tests. Defaults preserve the original behaviour, so a solver that ignores them still runs — it just
cannot be certified above Tier C (`self_convergence`).

```python
def solve_pde(N: int, override: dict | None = None) -> dict:
def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
```

- `override` — `ic` / `source` / `bc` / `params` / `dt_factor`; powers manufactured solutions,
  degenerate limits, translation-invariance and temporal isolation (§7 of `verification_manual.md`)
- `dW` — pre-generated Brownian increments, so refinement levels share one Brownian path; without
  this, Monte Carlo noise swamps the discretization bias and no convergence order is measurable
  (§18)
- `observables` — accumulates `∫₀ᵀ φ(X_s) ds` per path, for the Dynkin identity check (§20)

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
