# benchmark

The **independent grader**. The pipeline scores itself — the formulator writes down the analytic
solution, the evaluator grades against it — and that is circular: a formulator that transcribes the
wrong solution can still be scored 10/10. This directory closes the loop with a problem set whose
answer keys are authored here, validated here, and never visible to an agent.

Three suites that run separately:

- **The main benchmark — 49 problems** whose solution has a closed form (or machine-precision
  moments). Runs by default.
- **The no-closed-form suite — 7 problems** (`--suite no-closed-form`) with *no formula to
  transcribe*, which is where a pipeline that scores itself against a closed form has nothing to
  lean on.
- **The externally anchored suite — 10 problems** (`--suite benchop`), all SDE, whose answer keys
  were published by a third party in 2018. Seven are graded against those published values; the
  other three are the same problems with an initial condition moved, graded against values this
  repo computes, so that a solver which recalls rather than solves scores well on the first set
  and badly on the second.

Each suite carries its own C0/C1/C2 baseline, measured over its own denominator, so they are kept
apart by default and never folded together silently; `--all-problems` runs all three.

**Every problem is independently checkable** — by exact ground truth, a discretization-free
Monte-Carlo reference, a harness-owned high-resolution reference field, a scalar quantity of
interest, or a stability (no-blow-up) criterion — so the benchmark can catch the pipeline
believing it succeeded when it did not.

## Two graders, one solver

This is the idea the verdict table is built on, and it is worth getting straight before reading
any of the code. The same `solver.py` is graded twice, by two layers that are kept ignorant of
each other:

| | What it grades against | What it never sees |
|---|---|---|
| **The kernel** — [`verifylib/`](../verifylib/) | the plan's own `problem_spec.json`: ladders, invariants, residuals, surrogates | the benchmark's ground truth |
| **The harness** — `verify.py` (here) | ground truth authored in `problems.py` and the reference artifacts | the pipeline's self-score, until the verdict is computed |

The kernel is what the pipeline uses to pick its winner, so it has to work without an answer key.
The harness has the answer key and is never in the loop. A **verdict** is the cross-check between
them, which is why `OVERCLAIM` is the finding this directory exists to produce: it means the
kernel certified something that is wrong. [`tests/test_cross_layer.py`](tests/test_cross_layer.py)
pins that asymmetry — a seeded defect the kernel *catches* must come out `FAIL` or `UNDERCLAIM`,
never `OVERCLAIM`.

`run.py` records the kernel's score and provenance beside the harness verdict for every problem
(`--skip-kernel` to skip; it costs one extra sandboxed solve), and grades **every** plan
directory rather than only the winner (`--skip-plans` to skip), which is what produces the
`wrong_plan_won` statistic — invisible in a winner-only record.

## Layout

```
benchmark/
├── problems.py               the 66 problems + independently-authored ground truth
├── setup.py                  write workspace/{slug}/problem.md (+ the solver contract) and manifest.json
├── run.py                    PIPELINE runner: setup → conductor → verify → results → report
├── verify.py                 independent verification (SDE + PDE); --sandbox for untrusted solvers
├── runner.py                 sandboxed subprocess execution (timeout, mem cap, import policy)
├── validate_ground_truth.py  checks the ANSWER KEYS, by routes independent of problems.py
├── make_references.py        OFFLINE: build the harness-owned reference fields
├── benchop.py                OFFLINE: re-derive the 24 published BENCHOP-SLV reference values,
│                            and --calibrate each gate against a correct and a wrong scheme
├── oneshot.py                BASELINE runner: C0 (one-shot) / C1 (single agent + tools)
├── baseline_select.py        pick a matched baseline's answer with the conductor's own rule
├── compare.py                head-to-head: pipeline vs baselines (accuracy + cost)
├── report.py                 REPORT.md generator (verdicts, discrepancies, tiers)
├── manifest.json             generated: the authoritative per-problem record (suite included)
├── pipeline-settings.json    the deny rules that keep this directory out of the pipeline's reach
├── design/                   the suite's own design record — plans, candidate routes, lit reviews
├── references/               generated: {slug}.npz reference fields + benchop_slv.json (never read by the pipeline)
├── results/                  generated run artifacts + the committed REPORT.md
└── tests/                    464 tests on the harness itself; no model calls, no workspace
```

`benchmark/` is a directory of scripts, not a package: each module puts its own directory on
`sys.path` and imports `problems` bare. `tests/conftest.py` does the same thing once.

### design/

Five documents that record why the suite looks like this. They are design records, not
specifications — [`docs/`](../docs/) holds the same kind of thing for the pipeline.

| Document | What it is |
|---|---|
| [design/BASELINE_COMPARISON_PLAN.md](design/BASELINE_COMPARISON_PLAN.md) <br> *674 lines* | The C0/C1/C2 ladder: what each rung isolates, how fairness is enforced, and the log of what the real sweeps measured — including the void C0@Opus run and why it was void |
| [design/NO_CLOSED_FORM_CANDIDATES.md](design/NO_CLOSED_FORM_CANDIDATES.md) <br> *309 lines* | The catalogue of routes to truth without a closed form, and the candidate problems scored against them. The authoring checklist for a new no-closed-form problem lives here |
| [design/SYSTEMS_3D_EXTENSION_PLAN.md](design/SYSTEMS_3D_EXTENSION_PLAN.md) <br> *253 lines* | Extending the harness to multi-field and 3-D problems: general-`d` grids, the `fields` contract, structural diagnostics, masked domains |
| [design/HARDNUMERICS_IMPORT_LIST.md](design/HARDNUMERICS_IMPORT_LIST.md) <br> *161 lines* | The 13 problems imported from the HardNumerics survey, with what each one stresses |
| [design/hard_sde_candidates_from_lit.md](design/hard_sde_candidates_from_lit.md) <br> *157 lines* | The numerical-SDE literature review the tier-3 SDE weighting comes from |

## The problem set (66)

**Main benchmark: 49** (27 PDE, 22 SDE). **No-closed-form suite: 7** (6 PDE, 1 SDE).
**Externally anchored suite: 10** (0 PDE, 10 SDE).

| | Tier 1 | Tier 2 | Tier 3 | Total |
|---|---|---|---|---|
| **PDE** | 5 | 5 | 23 | **33** |
| **SDE** | 5 | 5 | 13 | **23** |

Tier 1 is textbook material the pipeline's manuals cover (expected to pass). Tier 2 needs genuine
care — boundary layers, indefinite operators, anisotropy, multi-D, positivity/log guards. Tier 3
pushes *beyond* the manuals (shocks, a time-fractional operator, stiffness, heavy-tailed Monte
Carlo, chaos, multi-field systems in 3-D), which is where a benchmark earns its keep by exposing
limits. `manifest.json` is the authoritative listing; `run.py --list` prints the current selection.

**The 21-problem discriminator subset** (`problems.discriminator_problems()`, flagged
`"discriminator": True`) is where the thesis has to show up: 13 tier-3 PDEs from
`stefan_1d_similarity` through `maxwell_3d`, and 8 tier-3 SDEs. Run it at higher `--reps`.

**The no-closed-form suite** — each independently checked without a formula:

| Slug | Route | What makes it hard |
|---|---|---|
| `pde_kuramoto_sivashinsky` | reference field | chaotic, stiff, fourth-order. Two spectral integrators agree to **1.245e-09** at t=50 — six orders below the 1% gate, which is why chaos does not prevent a reference |
| `pde_cahn_hilliard_2d_coarsening` | reference field | the matched pair to the MMS Cahn–Hilliard, source deleted. A first-order energy-stable scheme is stable, smooth, mass-conserving — and wrong. Agreement **9.303e-08** |
| `pde_kdv_zabusky_kruskal` | reference field | a cosine fissions into solitons whose *phases* a second-order scheme gets wrong at any affordable grid. Three exact invariants. Agreement **5.583e-09** |
| `pde_gross_pitaevskii_1d` | reference field | trapped cubic Schrödinger as a real/imaginary two-field system; the error is a phase error. Agreement **1.375e-11** |
| `pde_burgers_viscous_1d` | semi-analytic exact | an internal layer of width ~2·ν at Re=100. The truth is a Cole–Hopf quadrature (converged to 5.6e-16), not a formula |
| `pde_schrodinger_eigen_2d` | semi-analytic exact + functional | an eigenproblem: a number *and* a mode. The ground state is negative, so the lowest eigenvalue is not the smallest in magnitude |
| `sde_double_well_langevin` | semi-analytic exact | no closed form for the transient or its moments; the truth is a Kolmogorov-equation solve, declared `exact` because it evaluates as an ordinary `moments` callable. The mean at T is the fraction of paths that have not crossed the barrier, and a first-order weak scheme misprices the escape rate by O(k·dt) |

**The externally anchored suite** — ten SDE problems, seven of whose answer keys were published
by a third party in 2018 and re-derived here before being trusted
([`benchop.py`](benchop.py) → `references/benchop_slv.json`), and three *controls* at a shifted
initial condition whose keys this repo computes:

| Slug | Key | What makes it hard |
|---|---|---|
| `sde_sabr_cev_call_i` | published | CEV exponent ½, so the origin is attainable and **absorbing**; a scheme that reflects or goes negative is solving something else |
| `sde_sabr_cev_call_ii` | published | the same over ten years at vol-of-vol 0.8, where absorption is common and the volatility spans orders of magnitude. The suite's loosest gate (1e-2), because `0.04/T` forces 2500 steps and so caps the paths |
| `sde_heston_call_feller` | published | 2κη is a fifth of σ²: the variance sits on zero. Three values spanning 31×, each gated at what Monte Carlo resolves for *it* — 1.5e-2, 4e-3, 2e-3 |
| `sde_heston_dnt` | published | a **continuously monitored** two-sided barrier. Per-step monitoring is biased high at O(√dt) — at the shipped 400k × 500 it reaches 3.6× the 6e-3 gate where a Brownian bridge reaches 0.35× |
| `sde_qlsv_dnt` | published | the same corridor with a local volatility quadratic in S, where the Lamperti chart is −100/S rather than the logarithm and the bridge is only approximate. Per-step monitoring misses the gate on one of the three spots here against two on the Heston corridor |
| `sde_hhw_call_corr` | published ±5e-3 | three correlated factors, a stochastic discount factor and a Feller ratio of 0.04. The only value in the suite this repo cannot confirm to the precision its source claims |
| `sde_hhw_call_zerocorr` | published | the same with the rate decorrelated, which removes none of the difficulty and admits an independent check |
| `sde_heston_call_hivar` · `sde_heston_dnt_hivar` · `sde_sabr_cev_call_ii_lowvol` | **harness** | the controls. Identical model, payoff, horizon and Feller ratio; only the initial variance or volatility moves, by enough to shift every answer 4.8–35× its own gate. A solver that recalled a published constant scores well on the sibling and badly here |

Nothing in those statements names the benchmark, quotes a reference value, or says which three are
the controls — a model warned off the memorized value would make the measurement come back clean
for the wrong reason.

**Every gate in this suite is calibrated both ways.** The repo's rule is that a correct scheme
must clear a gate at the resolution the problem ships with and a plausible wrong one must fail it,
and `benchop.py --calibrate` measures both for all ten — at the path count the entry declares and
the step count `sde_verify_dt` leaves it at, with the two schemes differing by exactly one
decision, so what separates them is that decision. The decisions are the CEV boundary at S = 0
(absorb or reflect), the variance below zero (truncate or reflect) and barrier monitoring
(a Brownian bridge or once per step). Results live in `references/benchop_slv.json` under
`calibration` and are summarised in [plan-benchop-slv.md](../docs/plan-benchop-slv.md) step 8;
`validate_ground_truth.py` checks that each stored calibration still refers to the gate its entry
declares today. The narrowest separation is SABR set I, at 0.29× the gate against 1.97×. One
qualification is recorded rather than smoothed over: on SABR set II a plain absorbing Euler in S
lands at 0.90–0.99× of the gate, so that gate separates a wrong scheme decisively and a merely
under-resolved one not at all.

**SDE tier 3 — hard-literature additions:**

| Slug | Route | What makes it hard |
|---|---|---|
| `log_heston_feller_violated` | exact | coupled 2-D affine (log-price + CIR variance), 2a=0.2 < σ²=1.0 ⇒ variance hits zero; naive √Y-Euler goes negative. Exact moments from a closed affine ODE system |
| `multichannel_stiff_m13` | exact | 13 non-commuting multiplicative noise channels on a spiral drift; exact moments via matrix exponential / Lyapunov ODE |
| `quintic_random_ic` | exact | superlinear `dX=-X⁵dt` with a Gaussian IC; Gauss–Hermite variance. Untamed Euler overshoots on the tail |
| `ginzburg_landau_s4`, `ginzburg_landau_s6` | reference | superlinear cubic drift with multiplicative noise; untamed Euler biases E[X²] low (σ=6 also NaNs). Reference = MC of the exact solution |
| `quintic_drift_noise` | stability | `dX=-X⁵dt + X dW`; plain Euler diverges (HJK), tamed stays finite |
| `fene_blowup` | stability | singular spring `dX=-X/(1-X²)dt + dW` confined to (-1,1); a fixed step overshoots the boundary and escapes to NaN |

## Ground truth, and where it comes from

Every reference is authored in [`problems.py`](problems.py) and cross-checked by
[`validate_ground_truth.py`](validate_ground_truth.py) — never the formulator's. Three declared
routes, over the 66 problems: **exact 58 · reference 6 · stability 2**.

- **exact** (48) — closed-form or machine-precision moments. SDE moments are checked against
  high-accuracy integration of the moment ODEs derived from the SDE by Itô (independent of the
  closed forms), against matrix-exponential / affine systems, and against direct simulation;
  exp-OU, whose moment ODEs do not close, is checked with an exact log-OU sampler. PDE solutions
  are checked against their defining identities — IC, BC, PDE residual, reference values. A typo
  in an answer key is caught here rather than silently corrupting a verdict.
- **reference** (6) — two different meanings, same discipline. For superlinear SDEs with no
  elementary moment formula (`ginzburg_landau_s4/s6`), a **discretization-free Monte-Carlo** of
  the known exact solution — zero scheme bias — compared within a standard-error-aware tolerance
  and anchored to a literature value. For PDEs with no closed form, a **harness-owned
  high-resolution field** computed offline by **two independent method families** whose
  disagreement is measured and stored as the problem's `reference_error`
  ([`make_references.py`](make_references.py) → `references/{slug}.npz`).
- **stability** (2) — for blow-up and domain stress tests with no ground truth at all
  (`has_ground_truth=False`), the independent check is only that a correct scheme stays finite and
  in-domain, which is exactly what a naive scheme gets wrong.
- **published** — the answer key was computed by somebody else, before this project existed.
  The 24 BENCHOP-SLV reference values are the only ones of this kind, and they close the one gap
  the other three routes cannot: `problems.py` holds a key the pipeline cannot read, but that key
  and the pipeline share an author and a repo. A published number does not.

**A published number is still derived before it is trusted**, because a transcribed constant is
only as good as the reading of the problem it came from. [`benchop.py`](benchop.py) re-derives
every one of the 24 from the published parameters by at least one independent route, and by two
different method families — a quadrature and a grid, or a grid and a sampler — wherever no closed
form exists, so that a shared mistake has to be made twice in two different languages. What it
writes to `references/benchop_slv.json` is three separate numbers per value: the published figure,
the uncertainty *that figure itself* carries, and how closely this repo reproduced it. Only the
second belongs in `reference_abs_err`; conflating it with the third would widen a gate because of
our own effort rather than because of the answer key's precision.

Two rules make the reference route honest. The pass tolerance is widened to
`REFERENCE_TOL_FACTOR = 3.0` times the reference's own error, so the answer key's error can never
fail a correct solver — the same rule the SDE side applies to Monte-Carlo standard error. And
`verify.py` **refuses to score against a reference that does not carry a measured error**: a
reference whose accuracy nobody established is an answer key, not ground truth.

A problem may also declare `functional_truth`, where the answer is a scalar quantity of interest
(an eigenvalue, an option price) rather than a field, and `diagnostics` with `gate=True` for
structural checks — mass conservation, a divergence-free constraint, non-triviality. A structural
gate is never the only check, because a wrong-but-conservative scheme passes it.

**A functional is gated relatively, floored per value.** `functional_tol` is a relative tolerance,
because relative error is the more uniform measure across a problem whose values span 30× — but a
reference *number* is usually known to an **absolute** precision instead: a published figure
rounded to a digit, a quadrature bound, a Monte-Carlo standard error. A problem declares that as
`reference_abs_err` (one number, or `{name: abs_err}`), and each value's gate becomes
`max(functional_tol, 3 × abs_err / |ref|)`. The factor is `REFERENCE_TOL_FACTOR` again, and the
floor is taken per value because one absolute uncertainty is a different relative precision at
every value: ±0.005 is 1.3e-4 relative at 37.52 and 0.13 at 0.0393. It only ever loosens. The
record carries the relative error, the absolute error and `functional_tol_used` — the gate each
value was actually held to — so a near-threshold result can be read without re-deriving the rule.

**On the SDE side a functional is also checked at two path counts.** The route runs on either
problem type — an SDE problem may carry `functional_truth` beside its terminal moments, as
independent gates, or instead of them, since `moments` is no longer required. What the SDE side
needs and the PDE side does not is a defence against a solver that simply returns the published
number: the 2-grid order check already fails a constant, and a price graded from a single run
would not notice one. So `solve_sde` is called a second time at `functional_level_paths`
(default `num_paths // 4`), and the two reported values must **differ** — two Monte-Carlo
estimates at different sample sizes do — and must agree within
`tol_abs × (1 + √(n_graded/n_coarse))`, which is 3 × `tol_abs` at the default quarter because a
standard error falls as `1/√n`. The band is symmetric: nothing requires the finer run to be the
more accurate one, because these curves were measured non-monotone at this suite's path counts.
Only *constant* and *inconsistent* fail; a crashed coarse solve is recorded as `not_run` and
leaves the verdict alone, so a flaky sandbox cannot fail a correct solver.

That coarse solve is also **the SDE side's free work-precision point** (step 9). It is a real
(error, time) pair at a quarter of the work, in the same sandbox, and the harness was already
timing it — so a functional-scored SDE problem emits a two-point curve for no extra compute,
the way a PDE problem gets its two points from the order-check ladder. Measured on
`sde_heston_dnt` with a correct solver: 100 000 paths, 15.1 s, 2.57e-3, against 400 000 paths,
52.2 s, 2.37e-3 — four times the work for 8% of the error, because at these counts the error
is discretization bias and not sampling noise. Where a problem carries moment gates *as well*,
the point is recorded under `skipped` instead: the graded error is then
max(mean, variance, functional) and the coarse level scores only the functional.

## Verdicts

After the pipeline finishes a problem, the harness re-imports the winning `solver.py`, re-runs it,
and compares the raw numerical output to the independent ground truth.

| Verdict | Meaning |
|---|---|
| `VERIFIED_PASS` | pipeline scored 10/10 **and** the independent check (exact or reference) confirms it |
| `OVERCLAIM` | pipeline scored 10/10 but the independent check **fails** — the key finding |
| `STABLE_PASS` | stability problem: a correct scheme stayed finite / in-domain |
| `BLOWUP` | stability problem: the solution diverged or escaped its domain — a stability overclaim |
| `CONSTRAINT_VIOLATION` | accurate-looking, but a hard structural gate broke (mass, `div u = 0`, `div B = 0`) |
| `UNDERCLAIM` | pipeline gave up below 10 but the solution actually passes |
| `FAIL` | neither the pipeline nor the independent check passes |
| `UNVERIFIED` | independent check could not run (error / inconclusive) |
| `SELF_ONLY` | no ground truth exists; only the pipeline self-score is available |
| `SPEC_BLOCKED` | the conductor's requirements gate halted the run before any plan was created — the formulator's ledger did not carry every constraint in `problem.md` into `problem_spec.json` |
| `RUN_ERROR` | the pipeline run itself failed or timed out |

## Pass criteria

Identical to the pipeline's own thresholds, applied independently:

- **PDE** — the solver is `solve_pde(N)`, run at `N` and `2N`. A pass requires **both** relative
  L2 error `< 1%` (`PDE_L2_TOL`) at the fine grid **and** an observed spatial convergence order
  `p = log2(err_N / err_2N)` at least the problem's floor (`min_order`, default 1.0) — so a scheme
  accurate on one grid but not converging fails. Below `ORDER_SUPERCONV_FLOOR = 1e-9` the error is
  at noise and the order check passes. A shock problem (fractional L2 order) can waive it with
  `pde_order_check=False`. Legacy no-argument `solve_pde()` solvers are still verified single-grid.
- **SDE (exact)** — variance relative error `< 10%` **and** (mean relative error `< 5%` or the
  exact mean is within `0.01` of zero). Moments are recomputed at the fixed `(num_paths, dt, seed)`
  recorded in `manifest.json`.
- **SDE (reference)** — the same thresholds, each widened to at least `3×` the reference
  Monte-Carlo standard error, so a correct solution inside reference noise is not failed.
- **SDE (stability)** — every terminal state is finite, and for a domain problem all within
  `|X| < bound`. Any Inf/NaN or domain escape is a `BLOWUP`.

The independent re-run scales the step to the horizon, `dt = min(problem_dt, 0.04/T)`
(`verify.sde_verify_dt`), so a *correct* scheme's discretization error cannot masquerade as a
correctness failure. That function is the single source of truth for the evaluation step:
`setup.py` states it in `problem.md`, so the solver is told the step it will be judged at rather
than having the verifier silently override its choice.

## Isolation is enforced, not assumed

"Never the formulator's" is a claim about what the pipeline can *see*, so it is enforced.
[`pipeline-settings.json`](pipeline-settings.json) denies every pipeline invocation read access to
`benchmark/`, and `run.py` and `oneshot.py` pass it with `claude --settings`. Deny rules are
honored even under `--dangerously-skip-permissions`, and passing them per-invocation rather than
putting them in the repo's `.claude/settings.json` leaves ordinary development sessions able to
work on `benchmark/` normally. On top of that base file, `run.isolated_settings` writes a per-run
settings file (kept beside the transcript in `results/logs/`) that also denies every *other*
problem's `workspace/<slug>/` and every `workspace_*` archive at the repo root, so a run sees
exactly one workspace — its own, live copy — and cannot crib a sibling's or a previous run's
`problem_spec.json`.

This closes the direct paths — `Read`, `Grep`/`Glob`, and the Bash commands whose file arguments
the harness can see (`cat`, `head`, `ls`, `sed`, …) — plus the inline-interpreter route
(`python -c`, `uv run python -c`), denied outright because a solver or evaluator only ever needs
to run *script files*. What it cannot catch is a file opened from inside a script the agent wrote
itself; no pattern rule can. The remaining defense there is
[`project_manual.md`](../references/project_manual.md#never-read-benchmark), which every agent
reads and which states the rule and the reason.

The leak this exists to prevent is real and was observed before the rules were added, three
times. A `pde_heston_2d` spec cited `benchmark/problems.py, _heston_call` and told the evaluator
to transcribe it verbatim. A `sde_quintic_drift_noise` evaluator cited
`benchmark/verify.py::verify_sde_stability` for its pass criterion. And a Cahn–Hilliard formulator
`cat`-ed its manufactured-solution twin's `problem_spec.json` as a template, in two separate
attempts — a cross-*workspace* leak rather than a `benchmark/` read, which is why the per-run
isolated settings file exists alongside the base deny rules. Each one turned an independent check
into a self-check, silently.

The first two are preserved byte-for-byte as
[`verifylib/tests/fixtures/`](../verifylib/tests/fixtures/)`leak_heston_spec.json` and
`leak_quintic_evaluate.py`, which the leakage tests assert on directly; `pyproject.toml` excludes
that directory from ruff so a reformat cannot quietly destroy the evidence. The Cahn–Hilliard
leak has no tracked fixture — its artifact is the workspace snapshot under `results/`, which is
gitignored.

## Running it

All commands run from the repo root with `uv`.

```bash
# 1. stage the inputs (writes workspace/{slug}/problem.md for every problem, plus manifest.json)
uv run python benchmark/setup.py

# 2a. the full main benchmark — sequential, and EXPENSIVE: each problem drives the
#     multi-agent pipeline (Opus + Sonnet, up to 3 plans x 5 iterations). Budget hours.
uv run python benchmark/run.py

# 2b. or a cheap subset / single smoke test / the hard suite
uv run python benchmark/run.py --only sde_gbm,pde_heat_1d
uv run python benchmark/run.py --only pde_heat_1d --timeout 1800

# anything outside the main suite must name its own results file, and is refused
# without one -- see "Naming conventions" in results/README.md for the accident
uv run python benchmark/run.py --suite no-closed-form \
    --results-json benchmark/results/pipeline/no_closed_form.json
uv run python benchmark/run.py --suite benchop \
    --results-json benchmark/results/pipeline/benchop.json

# resume after an interruption (skips problems already completed in results.json)
uv run python benchmark/run.py --resume

# re-verify + re-report from existing workspaces, WITHOUT calling the pipeline (free)
uv run python benchmark/run.py --skip-run
```

Useful `run.py` flags: `--timeout <s>` (per-problem wall clock, default 2400), `--fresh` (clear a
workspace's prior pipeline outputs first), `--type` / `--tier` (select a slice), `--model`,
`--results-json <path>` (write somewhere other than `results/results.json`; **required** for any
selection that reaches outside the main suite, which both `run.py` and `oneshot.py` refuse to
merge into the standing baseline), `--list` (preview the selection), `--skip-kernel` / `--skip-plans` (drop the per-problem kernel record and the
every-plan grading), and `--no-skip-permissions --permission-mode acceptEdits` if you prefer not
to bypass permission prompts — the default passes `--dangerously-skip-permissions` for unattended
batch runs.

Check the answer keys, a single solved problem, or the report, at any time:

```bash
uv run python benchmark/validate_ground_truth.py          # the answer keys themselves
uv run python benchmark/verify.py --slug sde_gbm          # one solved problem, independently
uv run python benchmark/verify.py --slug pde_heat_1d --plan-dir workspace/pde_heat_1d/plans/2-crank-nicolson
uv run python benchmark/report.py --results benchmark/results/<file>.json --out <out>.md
```

> `report.py` defaults to `results/results.json`, which currently holds the **August 28-problem
> main-suite run**. The committed [`results/REPORT.md`](results/REPORT.md) was generated from
> `results/pipeline/no_closed_form_v2.json` (the 3-problem second wave, 2026-09-25), so a bare
> `uv run python benchmark/report.py` overwrites it with the older run. Pass `--results` and
> `--out` explicitly.

Rebuilding a reference field is an offline job, run deliberately and never during a benchmark run:

```bash
uv run python benchmark/make_references.py --list
uv run python benchmark/make_references.py --only pde_kuramoto_sivashinsky
uv run python benchmark/make_references.py --check     # re-measure the stored disagreement
```

## Baselines — does the *structure* help?

The pipeline mixes Opus and Sonnet across several agents, so beating a weaker setup could just
mean "a bigger model." To isolate the **structure**, `oneshot.py` runs the same problems through
weaker structures on the *same* model, and `compare.py` renders the head-to-head.

| Rung | What | Isolates |
|---|---|---|
| **C0** | naked one-shot: a single tool-less completion, given only `problem.md`, returns `solver.py` | "vs pasting it into a chat" |
| **C1** | single agent + tools: one agent may write, run and iterate on `solver.py` (Bash/Read/Write), with none of the pipeline's decomposition | the multi-agent structure, beyond mere tool access + iteration |
| **C2** | the full pipeline (`run.py`) | — |

Each rung also runs **naked** (`problem.md` only) or **manual** (+ the same `sde_manual.md` /
`pde_manual.md` the pipeline's agents read), separating the structure's contribution from the
reference material's.

**Fairness is enforced mechanically.** Both sides read the identical `problem.md` (which does not
leak the ground-truth moments), and every `solver.py` — pipeline and baseline alike — is graded by
the *same* `verify.py` at the *same* `(num_paths, eval dt, seed)`, **in the same sandbox, under the
same wall-clock budget** (`verify.DEFAULT_SOLVER_TIMEOUT_S`, recorded in each run's results config).
Until 2026-10-17 the pipeline was graded by importing `solver.py` inline, which made a
pipeline-vs-baseline *time* comparison a comparison of two harnesses; routing both through
[`runner.py`](runner.py) costs 2.4% (184.7 s against 180.3 s over the 34 solved workspaces) and
reproduces the inline verdict on all 34. Getting there took three fixes to what the boundary
carried — `functionals`, both result shapes at once, and an absolute solver path — the first of
which had been failing *every* baseline on the one functional-scored problem for not returning a
number it had returned. `run.py --no-verify-sandbox` restores the inline path for debugging.
Solver code is untrusted, so it runs through [`runner.py`](runner.py): a fresh subprocess, a wall-clock timeout, a best-effort memory cap, and
the **same import rule the pipeline solvers follow** — numpy + stdlib for SDE (`solver-sde.md`),
numpy + scipy for PDE (`solver-pde.md`, where 38 of 55 pipeline solvers use `scipy.sparse`). A
baseline that hangs, OOMs, violates the contract or reaches for a forbidden library simply fails;
there is no self-score to be unsure about. The crash reasons are a closed set: `no_solve_fn`,
`bad_signature`, `blocked_import`, `oom`, `timeout`, `bad_schema`, `exception`.

**Two clocks come back from every sandboxed solve.** `wall_s` is the whole subprocess —
interpreter startup, the numpy/scipy import, the solver module's top-level code and the solve —
and it is what the timeout is applied to. `solve_s` is the `solve_pde` / `solve_sde` call alone,
started in the child *after* the import. Their difference is the harness's own overhead, which on
a cheap problem is nearly all of `wall_s`: measured 2026-10-17 on a Crank–Nicolson `pde_heat_1d`,
0.0053 s of solve inside a 0.200 s process at N=64 and 0.0228 s inside 0.220 s at N=128 — so the
4× between the two grids reads as 1.1× on the process clock. An error-vs-time curve has to be
drawn against `solve_s`. A PDE verdict carries one clock per graded grid under `verify.timing`;
an SDE verdict carries its one under `verify.run.solve_s`. The inline pipeline path
(`sandbox=False`) is **not** timed, because a warm in-process call and a fresh interpreter are
different measurement conditions — see step 2 of
[`docs/plan-benchop-slv.md`](../docs/plan-benchop-slv.md).

**Every sandboxed verdict carries an error-vs-time series** under `verify.work_precision`, because
"more accurate" is not a claim until "at what cost" is attached. The first points are free: the PDE
path already solves at `N` and `2N` for the order check and already times both, so the series is a
join of two records the harness was keeping separately — measured over the 34 solved workspaces,
turning it on costs **−0.4% (179.7 s against 180.5 s, i.e. noise)** and changes **0 of 34** verdicts,
while **34 of 34** gain a series. A problem can extend it with `work_precision_levels` —
`[32, 64, 128, 256]` for PDE, `[(20000, 0.02), (80000, 0.005)]` for SDE — which costs real solves
(2.2×–2.9× the verification time for four PDE levels) and is what `--no-work-precision` turns off.
Three rules hold it honest: it is **sandbox-only**, it is **measured rather than modelled** (no
interpolated points, no fitted slope), and **it cannot move a verdict** — a level that crashes,
times out, or is ignored by the solver is recorded under `skipped` with its reason while the graded
ladder's decision stands. It is worth more than a cost axis: run on this repo's own solvers,
`pde_advection_1d` passes a graded 2-grid order of **4.00** while the full ladder reads
**3.89, 4.00, 2.71** — the order collapses at a doubling the graded pair does not look at.

| One-shot verdict | Meaning |
|---|---|
| `PASS` / `STABLE_PASS` | independently confirmed (moments/L2 within gate; or stayed finite/in-domain) |
| `FAIL` / `BLOWUP` | ran, but out of tolerance / non-finite / escaped its domain |
| `CONSTRAINT_VIOLATION` | accurate-looking, but broke a hard structural gate |
| `CRASHED` | never produced a scorable run — one of the seven crash reasons. Counts as a failure |
| `NO_GT` | no independent check exists — excluded from the rate |

```bash
# ALWAYS smoke-test the plumbing cheaply first (haiku, 2 problems):
uv run python benchmark/oneshot.py --only sde_gbm,pde_heat_1d --condition c0 --model haiku

# the hard discriminator subset, both conditions + knowledge settings, Opus, 5 reps:
uv run python benchmark/oneshot.py --discriminators --condition c0,c1 --manuals both --model opus --reps 5

# a Sonnet-pinned pipeline sweep (for the cross-tier cell), saved aside so it does not
# clobber the Opus+Sonnet results.json, then the head-to-head:
uv run python benchmark/run.py --model sonnet \
    --results-json benchmark/results/pipeline/pipeline_sonnet.json
uv run python benchmark/compare.py \
    --pipeline results/results.json=C2a:opus+sonnet \
    --pipeline results/pipeline/pipeline_sonnet.json=C2b:sonnet
```

`--resume` and `--redo CRASHED` re-run selected verdicts without repeating the whole sweep, and
`--c0-max-turns` (default 2) raises the C0 turn cap while keeping it tool-less — a larger cap only
lets the model answer *after* its denied tool attempts, which matters because a model given a
one-shot harness reaches for Bash reflexively and the turn cap used to discard the finished solver
with it. [`tests/test_oneshot_salvage.py`](tests/test_oneshot_salvage.py) pins that recovery, and
pins that it does not become an extra retry or a silent pass.

### The matched baseline

[`baseline_select.py`](baseline_select.py) answers the question a matched comparison turns on:
given Opus as many independent attempts at a problem as the pipeline spent on it
(`results/matched_attempts.json`), **which** attempt is the baseline's answer? Grading the best one
by ground truth would hand the baseline an oracle the pipeline never had, so this runs the *same*
`verifylib` kernel over the attempts and ranks them with the *same* `verifylib.ranking.rank_key`
the conductor uses — score, provenance, quality index, solver time. Three cuts come out of it, and
a comparison should quote all three: `first` (rep 1 only — what one completion gets you), `kernel`
(matched attempts, matched selection rule, no oracle — the headline), and `pass_at_k` (any attempt
passes — the oracle-selection upper bound, which brackets how much of the gap is *selection*
rather than solving). It makes no model calls, so it is free to re-run.

One asymmetry is stated plainly rather than hidden: the kernel cannot evaluate a solver without a
`problem_spec.json`, and a C0 baseline produces none — that is the formulator's job, and the
formulator is exactly what C0 lacks. So each attempt is scored against *the pipeline's own spec*
for that problem, copied into a scratch directory beside the attempt's `solver.py`. The baseline's
solver never sees it; only the selector does. A run with no spec is reported as `spec_missing`
rather than silently scored.

> **A caveat for the PDE side:** a PDE `problem.md` that stated its closed-form solution would let
> a baseline (or the pipeline) pass by evaluating the given formula instead of solving. This is
> closed on two fronts: **(1) leakage** — no PDE `problem.md` states its solution (MMS problems
> ship the explicit source term instead, so they stay fully specified); **(2) the order check** —
> a solver must actually *converge* under refinement, so a hard-coded single-grid answer fails.
> Hard PDEs added later should stay closed-form-free in the statement the same way.

## results/

Generated and almost entirely gitignored — but an **evidence store, not scratch space**: several
of its files are cited by name from the design record, so a tidy-up that deleted them would break
the record. It has its own [results/README.md](results/README.md), which documents the layout, the
three committed files, and a name→location table for the flat paths recorded in older documents.

The short version: the live files the tools read and write by default stay at the top level
(`results.json`, `REPORT.md`, `COMPARISON.md`, `logs/`), and finished artifacts are grouped
beneath it — `pipeline/` (completed C2 sweeps), `baselines/` (C0/C1 sweeps and their selections),
`studies/` (targeted runs cited by a plan), `superseded/` (void, clobbered and pre-fix backups,
kept as evidence) and `snapshots/` (frozen read-only copies).

## Adding a problem

Append an entry to `PROBLEMS` in [`problems.py`](problems.py) with metadata, a `description` (the
`problem.md` body), and — per its `ground_truth_kind` (default `exact`):

- `exact`: a ground-truth callable (`moments(T) -> {mean, variance, ...}` for SDE,
  `analytic(t, *coords)` for PDE).
- `reference`: `ground_truth_kind="reference"` and `moments(T) -> {key: (value, se)}` from a
  discretization-free MC of the exact solution.
- `stability`: `ground_truth_kind="stability"`, `has_ground_truth=False`, and a `stability_check`
  (`{"type": "finite"}` or `{"type": "domain", "abs_bound": ...}`).
- a no-reference stress test: `has_ground_truth=False`, `analytic=None` (default `exact` kind →
  reported as `SELF_ONLY`).

`setup.py` automatically appends a **Solver contract** section to every `problem.md` — the
`solve_sde(num_paths, dt, T, seed)` / `solve_pde(N)` signature, the return schema, and the *exact*
arguments the verifier will call it with (SDE: `num_paths`, the evaluation
`dt = min(problem_dt, 0.04/T)`, `T`, `seed`; PDE: the two resolutions `N` and `2N` plus the
accuracy tolerance and the observed-order floor). This keeps the harness API out of the numerical
problem, and makes a one-shot LLM given only `problem.md` a fair comparison to the pipeline. PDE
tuning knobs: `grid_N` (base resolution, default 64), `min_order` (order floor, default 1.0),
`pde_order_check` (default True; False for shocks).

For a problem with **no closed form**, pick a route first — the four are catalogued in
[`design/NO_CLOSED_FORM_CANDIDATES.md`](design/NO_CLOSED_FORM_CANDIDATES.md), and if none applies
the problem is not ready — then set `closed_form=False` so it joins
`--suite no-closed-form` (or `suite="benchop"` for an externally anchored one):

- *harness-owned reference field*: add a generator to
  [`make_references.py`](make_references.py) that runs **two independent method families** and
  records their disagreement, build the artifact, then set `ground_truth_kind="reference"`,
  `reference=reference_evaluator(slug)` and `reference_error=<the measured number>`. Grids must
  nest or be periodic (a periodic axis is trig-interpolated exactly; a non-periodic one needs
  `grid_N` chosen so the solver's `N` and `2N` are subsets of the reference grid).
- *semi-analytic exact*: an ordinary `analytic` callable that evaluates a quadrature, series or
  eigen-expansion. No harness change.
- *functional*: `functional_truth={"name": value}` plus `functional_tol`; the solver returns a
  `functionals` dict. Compute the value yourself at a **de-memorized** parameter — a published
  constant can be recalled by a one-shot model that solved nothing, which corrupts the very
  comparison the benchmark exists to make. If the value you compute has a known absolute
  uncertainty, declare it as `reference_abs_err` rather than loosening `functional_tol`: the
  former floors each value separately and leaves the precise ones precise. On an SDE problem
  this is the whole ground truth if you omit `moments`, and the harness will solve twice to
  check the number moved with the path count.
- *structure*: `diagnostics` entries with `gate=True` (`_diag_mass_error`, `_diag_nontrivial`,
  `_diag_single_signed`, `_diag_divergence_free`, …). Never the only check.

Then re-run `uv run python benchmark/validate_ground_truth.py` before trusting it in a benchmark
run. For a no-closed-form problem it checks both directions: that the reference is far more
accurate than the gate it defines, that its declared `reference_error` matches the artifact, and —
the part that actually matters — that a **correct scheme clears every gate at the benchmark's own
resolutions**. Calibrate that before paying for a high-resolution reference: a problem every
resolution passes measures nothing, and is worse than no problem at all. On the SDE side it also
checks that a correct scheme clears both gates at the evaluation `dt` (the mean gate, 5%, is
tighter than the variance gate, 10%).

## tests/

464 tests on the harness itself (`--collect-only`, 2026-10-21; 15 of them `slow`). They never run
`claude` and never touch `workspace/`: they feed `verify.py` synthetic solver output built from the
harness's own ground truth — a reference field with noise added, an exact eigenfunction with a
wrong eigenvalue beside it — and check what the verdict says. Deterministic, and seconds apart
from the `slow`-marked reference rebuilds.

```bash
uv run --extra dev python -m pytest benchmark/tests -q        # 115 passed, 4 deselected, ~28 s
uv run --extra dev python -m pytest verifylib benchmark/tests -q   # both layers: 1023 passed
```

| File | What it pins |
|---|---|
| [tests/test_truth_routes.py](tests/test_truth_routes.py) | the four routes to truth without a closed form, as the harness grades them — including that a recalled eigenvalue with no mode behind it does not pass |
| [tests/test_every_plan.py](tests/test_every_plan.py) | `grade_every_plan` and `wrong_plan_won`; and that the solver hash the kernel wrote is what stops a drifted file from being graded as the plan |
| [tests/test_cross_layer.py](tests/test_cross_layer.py) | one solver through both layers — the calibration that keeps `OVERCLAIM` meaning "the kernel was wrong" |
| [tests/test_orchestration.py](tests/test_orchestration.py) | the runner, the suite split and the verdict logic — what gets graded and how the verdict is read |
| [tests/test_oneshot_salvage.py](tests/test_oneshot_salvage.py) | that a C0 completion which produced a working solver is not discarded by the turn cap |
| [tests/test_leakage_ncf.py](tests/test_leakage_ncf.py) | that truth stays in `problems.py` and the artifacts, and `problem.md` carries no formula, reference value or benchmark path |
| [tests/test_solve_timing.py](tests/test_solve_timing.py) | that the solve clock is the `solve_*` call alone, separated from interpreter startup and the import |
| [tests/test_execution_parity.py](tests/test_execution_parity.py) | that pipeline and baseline are graded through the same sandbox, so a comparison of either accuracy or time measures the solvers |
| [tests/test_work_precision.py](tests/test_work_precision.py) | the error-against-time series: the points that are free, the levels that are paid for, and that none of it can move a verdict |
| [tests/test_functional_tolerance.py](tests/test_functional_tolerance.py) | a quantity of interest gated relatively and floored per value by the reference number's own absolute uncertainty |
| [tests/test_pde_functional.py](tests/test_pde_functional.py) | that a PDE problem whose only truth is a number still fails a hardcoded constant — the gap §5's memorization argument rested on |
| [tests/test_sde_functional.py](tests/test_sde_functional.py) | the same route on the SDE side, where a price can be the whole grade and the defence is a second path count |
| [tests/test_benchop_truth.py](tests/test_benchop_truth.py) | that the 24 published reference values were re-derived rather than transcribed, and that each wrong scheme is wrong in the direction claimed |
| [tests/test_benchop_suite.py](tests/test_benchop_suite.py) | the ten entries: their numbers come from the ledger, their gates are calibrated both ways at the settings they ship with, and staging writes the call the verifier will make |
