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

Both SDE and PDE evaluators use the same 1–10 scale, and it answers **two** questions:

- **`certified`** — did every gate this problem's evidence path requires come back clean? This is the
  terminal condition: what the winner early-exit stops on and what `benchmark/report.py` reads for
  `VERIFIED_PASS`.
- **`score`** — that, *and* how far inside the tolerance the run landed. The evidence tier opens a
  **band** (`verifylib/kernel/score.py:BANDS`) and the quality index places the plan within it:

  | Evidence | Band |
  |---|---|
  | Tier A / A⁻ quoted / A′ / Tier B with two circularity breaks | **9–10** |
  | One circularity break (`manufactured_partial`, A⁻ unavailable) | **8–9** |
  | Self-convergence only | **6–8** |
  | A gate violated / nothing ran / crash | **3 / 2 / 1** — findings have no quality |

  The top of a width-1 band needs **a third of the achievable accuracy headroom**. On a PDE that is
  two decades below the tolerance (`estimated_rel_error <= rel_l2_err_max / 100`). It is a *fraction*
  rather than a fixed number of decades because what is achievable differs by kind: a Monte Carlo
  estimate's accuracy is bounded by `1/√M`, not by its scheme, so at 50 000 paths against a 10%
  variance tolerance the ceiling is **0.90 decades** and no SDE solver can reach two.

  Measured on `pde_heat_1d`: FTCS lands 2.45 decades inside the tolerance, Crank–Nicolson 2.59 and
  4th-order RK4 6.67 — all three score 10. Measured on `pde_heat_2d`: all three second-order plans
  land at 1.99–2.00 decades and score 9, and the 4th-order plan that would have cleared it never ran
  — which is the finding, not a defect in the scale.

Three rules bound placement:

- It can only ever **lower** a score, and it never moves `certified`. So the loop's cost and the
  benchmark's pass rate are exactly what they were before the score began resolving quality.
- It applies only where the **rubric put the run at the top of its band**. The rows below that are
  already accuracy-graded — §12's 7 is "converging, not yet accurate enough" — so placing them again
  would demote one shortfall twice.
- On a **chaotic** problem it does not read the full-`T` error. `converged` still gates on it, so a
  solver wrong at `T` does not certify; but §28 waives pointwise accuracy there because trajectories
  separate past the Lyapunov time, and grading quality on it re-imposes the penalty the waiver exists
  to remove.

**The score is computed, not assigned.** `verifylib/kernel/score.py` applies the rubrics below and
emits `score`, `certified` and `provenance` in the metrics; the evaluator transcribes them into the review and
`review.py` requires the `Score: N/10` headline to match `metrics.score` **exactly**. The evaluator's
one authority is asymmetric — it may record `agent_cap: {score, reason}` to move the score *down*,
never up. That escape hatch exists for what no check measures: a `SOLUTION.md` describing a different
scheme than `solver.py` implements, an answer smuggled in as a lookup table, a plan that "converges"
by returning its input. It is the same shape `benchmark/verify.py`'s `ledger_audit` already uses
(`score_cap: 3`).

Which rubric applies depends on the **provenance** of the reference (see below). The rubrics here
are the *analytic* path — when there is no closed form, the kernel uses the corresponding rubric
in `verification_manual.md` (§12 for PDE, §23 for SDE), under the certification ceiling in §26.

**SDE, analytic path** (`evaluator-sde`, `analytic_moments.has_analytic_solution == true`):

Rows in the order they are applied. The order decides what a sign-flipped drift scores: such a
solver explodes the variance, widens the CI, and would otherwise land on the MC-inconclusive row
with a Dynkin `z` of ~1500 and feedback telling the author to raise `num_paths`.

- **1**: the solver crashed, or returned non-finite output
- **4**: a `gate: true` structural constraint is violated (negative paths, support breach)
- **3**: Dynkin fails — the paths do not satisfy the generator built from the spec's own drift and
  diffusion. Exact terminal moments cannot separate this from a correct solve, because two
  processes can share two moments, so the usual cause is the wrong equation integrated correctly
- **2**: no moment could be measured against the closed form — the spec claims an analytic
  solution and neither a declared expression nor a `*_value_at_T` was evaluable for either moment
- **8**: the moments match, the MC is resolved and the constraints hold, but the observed **strong
  order** does not clear its floor. Same score and same row as the no-closed-form rubric
  (`verification_manual.md` §23) gives the identical state — scoring it 7 here and 8 there would
  open a cross-path divergence. The reason branches on the measured order: near zero is a solver
  that does not consume the supplied Brownian increments (§18), below the floor is a scheme
  converging more slowly than it claims
- **5**: as above, but a declared non-gate **invariant** drifts materially — the terminal
  distribution is right and a structural property the spec states of the *process* is not
- **6**: MC-inconclusive — error bars too wide to decide; raise `num_paths`. Not a solver bug
- **9**: all of the 10 row's conditions hold **and `dynkin_ok` never reached a verdict** — the
  budget refused the ladder, the generator could not be built, the observables run crashed, or the
  extrapolation abstained. Everything measurable passed and nothing establishes that the paths came
  from this SDE; the 3 row above says why exact moments cannot stand in. Uncertified, not suspect
- **10**: `variance_rel_err < 10%` AND (`|exact_mean| < 0.01` OR `mean_rel_err < 5%`), **and the
  estimates are MC-resolved** (§14), **and `dynkin_ok` is `True`**. An order study that
  *abstained* — `orders_ok: "unresolved"`, because its own error bar admits more than the band — is
  reported in the reason and is **not** counted against the plan: an unmeasurable strong order on
  heavy-tailed multiplicative noise is a property of the estimator, not of the solver. The two
  abstentions are priced differently on purpose, and `verification_manual.md` §23 gives the
  argument: the order study measures the *scheme*, and something else can measure that; Dynkin
  measures the *equation*, and nothing else can
- **8**: the variance passes, the mean is slightly above threshold
- **4**: the variance fails, the code ran cleanly

Until 2026-10-15 this rubric read only the two moment thresholds and the MC resolution, and **not**
`dynkin_ok`, `orders_ok` or `invariants_ok`. Since 21 of the 23 SDE problems are on this path, that
meant a solver which drew from the exact terminal law and never integrated the SDE at all scored
**10, certified, provenance `analytic`**. The rows above are what closed it; see
`docs/plan-sde-verification-gaps.md` §3 Phase 1.

The **9** row is the second half of that fix, added later the same day. The first pass added a cap
for a Dynkin that *fails* and left a Dynkin that never ran reading as nothing at all, so the 10 row
still fell back to moment evidence alone wherever the check could not produce a verdict — which is
the original hole reached through a different door. See `docs/plan-sde-verification-gaps.md` §13a.

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
| `manufactured` | Manufactured solution or degenerate-limit check passed at design order, **and** the D1 operator residual clean against a validated operator — two independent circularity breaks |
| `manufactured_partial` | Exactly **one** circularity break: Tier B alone, or a clean D1 alone. Caps at 9 |
| `self_convergence` | Richardson/GCI estimate only — no independent reference. Caps at 8 |
| `none` | No test could be run (caps the score at 2) |

**A 10 without a closed form requires two independent circularity breaks.** MMS checks the
*discretization* against a known solution of a *modified* problem; D1 checks the *produced field*
against the *stated* operator on the real problem, where the shock, layer or stiffness the
manufactured problem lacks actually lives. They are independent only given a correct operator — a
misread equation fools both — which is why D1 counts as a break only when the operator has been
independently validated (`verification_manual.md` §26).

A 10 earned against a closed form and a 10 earned by self-convergence are **different claims**. The
tag must survive into `<metrics>`, `SOLUTION.md` and `REPORT.md`. See `verification_manual.md` for
the full ladder of evidence and every test available at each tier.

`analytic` splits by whether the closed form was **checked**, because an unvalidated closed form is
weaker evidence than a validated deterministic surrogate:

| Tag | Meaning |
|---|---|
| `analytic` | Closed form, **validated** against the spec's own operator / IC / BC |
| `analytic_unvalidated` | Closed form accepted on provenance alone — quoted verbatim, or the check could not run. The gap is recorded |

Ordering, most to least: `analytic` > `analytic_unvalidated` > `surrogate` > `manufactured` >
`manufactured_partial` > `self_convergence` > `none`.

## Claimed Solutions Must Be Checkable

> **An `analytic_solution` (or `analytic_moments`) enters Tier A when it is validated against the
> problem's own declared structure. Where validation cannot run, a verbatim quote from `problem.md`
> is accepted in its place and the gap is recorded. Otherwise the field is `null`.**

A wrong reference is the one input that fails common-mode: every plan for a problem descends from
one spec, so all plans are wrong identically, cross-plan consensus confirms it, the solver spends
its iterations chasing feedback derived from it, and the report cites it. Nothing downstream can
detect it, which is why it is checked at the spec.

The axis is **checkability, not origin**. A formula you were given and a formula you recalled are
equally dangerous if neither is verified. Quoting establishes blame, not correctness — so a quoted
formula that *fails* validation is still `null`.

`verifylib` reports one of five outcomes: `validated`, `validated_off_singularity`, `quoted`,
`unavailable`, `failed`. Only `failed` nulls the field, and it does so unconditionally.

## Requirements Are Re-Audited After the Run

The ledger proves the *spec* carries every constraint in `problem.md`. A second pass at
certification asks whether the *solver obeyed it*, and the two kinds of finding are treated
differently on purpose:

| Finding | Source | Consequence |
|---|---|---|
| Grid, domain bounds, `t_final`, Dirichlet edge values, SDE `num_paths` / `seed` / `T` | The run's own output — facts | **Hard gate.** Score capped at 3; the review names the ledger entry by `id` and quote |
| A spec parameter bound to a different constant in `solver.py` | A source scan | **Warning** (`ledger_warnings`). Never an automatic cap |

The split is not timidity. A source scan misses `eps = 1e-3 * 10` and false-positives on a
nondimensionalised solver where the parameter never appears literally. A false hard-fail on a
correct solver is worse than a missed relaxation, because it is invisible as a false positive — it
looks exactly like a caught bug.

## The `<metrics>` Block

The evaluator writes this immediately before the `<review>` block on every pass. The conductor
parses it to rank plans when scores tie.

```
<metrics>
score: 9
certified: true            # every gate this evidence path requires came back clean
provenance: analytic | analytic_unvalidated | surrogate | manufactured | manufactured_partial | self_convergence | none
estimated_rel_error: 3.10e-03
error_is_estimate: false
observed_order: 1.98
order_floor: 1.50
design_order: 2            # the scheme's own declared rate, from SOLUTION.md frontmatter
scheme_family: fd
invariants_ok: true
constraints_ok: true
invariant_max_drift: 4.2e-12
d1_outcome: clean | slow | stalled | unresolved | unavailable
d1_slope: 1.98
operator_validated: true
reference_outcome: validated       # what the reference check said about the closed form
order_check: true | waived_spec | waived_chaotic
asymptotic_source: ladder | probe
mc_se_rel: 0.0041          # SDE only
resolved: true             # SDE only
dynkin_z: 1.4              # SDE only -- extrapolated generator residual, in its own s.e.
resolution_evidence: not_used_as_accuracy_evidence
agent_cap: {"score": 3, "reason": "..."}   # only when the evaluator capped downward
solve_time_s: 0.83         # the solver, at the graded grid
wall_time_s: 12.4          # the whole evaluation
</metrics>
```

The kernel emits this block; the evaluator copies it verbatim rather than retyping it. Any value
that is a **skip** rather than a verdict uses the closed vocabulary `not_run`, `unavailable`,
`unresolved`, `not_reported`, `waived_chaotic`, `faulty_probe` — none of which is ever truthy and
none of which may be rendered as a pass.

`estimated_rel_error` is always populated — an evaluator with no true reference must still supply
the Richardson estimate. Omit fields that do not apply.

**`certified` and `score` answer different questions, and only the first is a pass.** `certified` is
"did every gate this problem's evidence path require come back clean" — the question the winner
early-exit and `benchmark/report.py`'s `VERIFIED_PASS` have always been asking. `score` answers that
*and* how far inside the tolerance the run landed, so a plan clearing every gate at 1.1× inside
`rel_l2_err_max` certifies exactly as one three decades inside it does, and does not score the same.
Rank on `score`; exit and gate on `certified`.

**`solve_time_s`, not `wall_time_s`, is the cost.** `wall_time_s` is the whole evaluation — the
ladder, MMS, D1, the temporal probe — which `verification_manual.md` §24 prices between 1.0× and 35×
depending on evidence path and which Stage-2 checks ran. Two plans on one problem can differ in it by
an order of magnitude for reasons that have nothing to do with their solvers.

**`design_order`, not `order_floor`, is what an observed order should be read against.**
`order_floor` is a pass bar — 1.0, and 0.0 on `pde_kuramoto_sivashinsky` — so
`observed_order − order_floor` is not a quality margin, and on an algebraic scheme a large positive
one is evidence the ladder is *not* asymptotic (`richardson.guards`' `p_sane`). On the SDE side
`order_floor` carries the expected **strong** order, which is a target with a two-sided band, not a
floor: the quantity to compare is `|observed_order − order_floor|`, and smaller is better.

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
Where `stability_check` is non-null it **replaces** moment scoring rather than adding to it, which
is how `benchmark/verify.py`'s `verify_sde_stability` grades it. A non-`null` value that is not an
object of this shape is a **schema error** (`schema.check_stability_check`): prose there reads as a
declared criterion and measures nothing, which is worse than leaving it `null`.

*Corrected 2026-10-15.* This paragraph used to read: "A spec with `has_analytic_solution: false`
and `stability_check: null` has no scoreable criterion at all and is a formulator error." That is
no longer true and enforcing it would hard-fail 2 of the 7 live SDE specs.
`verification_manual.md` §15–17's surrogate tier gives a Path-B problem a real reference — a moment
ODE, a Kolmogorov solve or a stationary density — so a spec can be `has_analytic_solution: false`,
`stability_check: null`, and still be scored against ground truth. `sde_ginzburg_landau_s6` is
deliberately in that state and its own `implementation_notes` say why: setting `stability_check`
non-null would tell the evaluator to drop the moment tolerances `problem.md` actually scores on, so
finiteness is carried as a gated `verification.constraints` entry instead.
`sde_double_well_langevin` is the same.

## Requirements Ledger

`problem_spec.json` carries a top-level `requirements` array: one entry per constraint in `problem.md`, each with a verbatim `quote`, a `status` (`mapped` / `ambiguous` / `dropped`) and the `spec_path` that carries it.

It exists because nothing downstream of the formulator ever reads `problem.md`. A parameter, boundary condition or scoring rule that fails to cross into the spec does not become an unrecorded detail — it stops existing, and the pipeline solves an easier problem than the one it was given, scoring itself 10/10 for doing so. The ledger turns that from a silent omission into a visible one.

The conductor **halts the run at `phase: blocked`** if `requirements` is missing or empty, if any entry is `dropped`, or if a `mapped` entry has no `spec_path`. `ambiguous` entries do not halt: they record where `problem.md` was underspecified and the formulator made a documented standard choice, and they are surfaced in REPORT.md so the reader can see where the solved problem may differ from the one they described.

## Python Environment

Use `uv` with the project's `.venv` (Python 3.13 or newer). Installed packages: `numpy`, `scipy`.

To run code: `uv run python workspace/{problem_slug}/plans/{id}-{plan_slug}/solver.py`
