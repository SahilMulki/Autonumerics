# Autonumerics Benchmark

A rigorous, ground-truth-verified benchmark for the Autonumerics pipeline.

**37 problems — 15 PDE + 22 SDE.** The SDE side is deliberately weighted toward hard
tier-3 cases drawn from a deep numerical-SDE literature review (see
[`hard_sde_candidates_from_lit.md`](hard_sde_candidates_from_lit.md)). Every problem
except the chaotic Kuramoto–Sivashinsky PDE is **independently checkable** — by exact
ground truth, a discretization-free Monte-Carlo reference, or a stability
(no-blow-up) criterion — so the benchmark can catch the pipeline believing it
succeeded when it did not.

## What makes it rigorous

The pipeline grades itself: the formulator writes down the analytic solution and the
evaluator scores against it. That is circular — a formulator that transcribes the
wrong solution can still be scored 10/10. This benchmark closes the loop:

1. **Curated, tiered coverage.** Tier 1 is textbook material the pipeline's manuals
   cover (expected to pass). Tier 2 needs genuine care (boundary layers, indefinite
   operators, anisotropy, multi-D, positivity/log guards). Tier 3 pushes *beyond* the
   manuals (shocks, a time-fractional operator, stiffness, heavy-tailed Monte Carlo,
   chaos) — where a benchmark earns its keep by exposing limits.
2. **Independent ground truth, three kinds.** Every reference is authored in
   [`problems.py`](problems.py) and cross-checked in `validate_ground_truth` — never
   the formulator's:
   - **exact** — closed-form / machine-precision moments or analytic solutions
     (SDE moments checked against moment-ODE integration, matrix-exponential / affine
     systems, and direct simulation; PDE solutions against their defining residuals
     and identities).
   - **reference** — for superlinear SDEs with no elementary moment formula, a
     *discretization-free* Monte-Carlo of the known exact solution (zero scheme bias),
     compared within a standard-error-aware tolerance and anchored to a literature
     value.
   - **stability** — for blow-up/domain stress tests with no ground truth at all, the
     independent check is only that a correct scheme stays finite / in-domain — which
     is exactly what a naive scheme gets wrong.
3. **Trust-but-verify scoring.** After the pipeline finishes a problem, the runner
   re-imports the winning `solver.py`, re-runs it, and compares the raw numerical
   output to the independent ground truth. Each problem gets a **verdict**:

   | Verdict | Meaning |
   |---|---|
   | `VERIFIED_PASS` | pipeline scored 10/10 **and** the independent check (exact or reference) confirms it |
   | `OVERCLAIM` | pipeline scored 10/10 but the independent check **fails** — the key finding |
   | `STABLE_PASS` | stability problem: a correct scheme stayed finite / in-domain |
   | `BLOWUP` | stability problem: the solution diverged or escaped its domain — a stability overclaim |
   | `UNDERCLAIM` | pipeline gave up below 10 but the solution actually passes |
   | `FAIL` | neither the pipeline nor the independent check passes |
   | `UNVERIFIED` | independent check could not run (error/inconclusive) |
   | `SELF_ONLY` | no ground truth exists; only the pipeline self-score is available |
   | `SPEC_BLOCKED` | the conductor's requirements gate halted the run before any plan was created — the formulator's ledger did not carry every constraint in `problem.md` into `problem_spec.json` |
   | `RUN_ERROR` | the pipeline run itself failed or timed out |

4. **Enforced isolation.** "Never the formulator's" above is a claim about what the
   pipeline can see, so it is enforced rather than assumed.
   [`pipeline-settings.json`](pipeline-settings.json) denies every pipeline
   invocation read access to `benchmark/`, and `run.py` and `oneshot.py` pass it via
   `claude --settings`. Deny rules are honored even under
   `--dangerously-skip-permissions`, and passing them per-invocation (rather than
   putting them in the repo's `.claude/settings.json`) leaves ordinary development
   sessions able to work on `benchmark/` normally.

   This closes the direct paths — `Read`, `Grep`/`Glob`, and the Bash commands whose
   file arguments the harness can see (`cat`, `head`, `ls`, `sed`, ...) — plus the
   inline-interpreter route (`python -c`, `uv run python -c`), which is denied
   outright because a solver or evaluator only ever needs to run *script files*. What
   it cannot catch is a file opened from inside a script the agent wrote itself; no
   pattern rule can. The remaining defense there is
   [`project_manual.md`](../references/project_manual.md#never-read-benchmark), which
   every agent reads and which states the rule and the reason.

   The leak this exists to prevent is real and was observed: before this was added, a
   `pde_heston_2d` spec cited `benchmark/problems.py, _heston_call` and told the
   evaluator to transcribe it verbatim, and a `sde_quintic_drift_noise` evaluator
   cited `benchmark/verify.py::verify_sde_stability` for its pass criterion. Both made
   the "independent" check a self-check for that problem.

## Layout

```
benchmark/
├── problems.py     # the 37-problem set + independently-authored ground truth
├── setup.py        # write workspace/{slug}/problem.md (+ auto-appended solver
│                   #   contract & exact eval config) and manifest.json
├── run.py          # sequential PIPELINE runner: setup → conductor → verify → results → report
├── verify.py       # independent verification (SDE + PDE); --sandbox for untrusted solvers
├── runner.py       # sandboxed subprocess solver execution (timeout, mem cap, import policy)
├── oneshot.py      # BASELINE runner: C0 (one-shot) / C1 (single agent+tools) on the same problems
├── compare.py      # head-to-head report: pipeline vs baselines (accuracy + cost)
├── report.py       # thorough REPORT.md generator (verdicts, discrepancies, tiers)
├── manifest.json   # generated: JSON summary of every problem (with ground-truth moments)
└── results/        # generated: results.json, REPORT.md, oneshot_results.json, COMPARISON.md, logs/
```

## Usage

All commands run from the repo root with `uv`.

```bash
# 1. stage the inputs (writes workspace/{slug}/problem.md for all 37, plus manifest.json)
uv run python benchmark/setup.py

# 2a. run the full benchmark — sequential, and EXPENSIVE:
#     each problem drives the multi-agent pipeline (Opus + Sonnet, up to 3 plans ×
#     5 iterations). Budget hours and real API/subscription usage.
uv run python benchmark/run.py

# 2b. or run a cheap subset / single smoke test
uv run python benchmark/run.py --only sde_gbm,pde_heat_1d
uv run python benchmark/run.py --only pde_heat_1d --timeout 1800

# resume after an interruption (skips problems already completed in results.json)
uv run python benchmark/run.py --resume

# re-verify + re-report from existing workspaces, WITHOUT calling the pipeline (free)
uv run python benchmark/run.py --skip-run
```

Useful flags: `--timeout <s>` (per-problem wall clock, default 2400), `--fresh`
(clear a workspace's prior pipeline outputs before running), `--list` (preview the
selection), `--no-skip-permissions --permission-mode acceptEdits` (if you prefer not
to bypass permission prompts; the default passes `--dangerously-skip-permissions`
for unattended batch runs).

Inspect a single solved problem independently at any time:

```bash
uv run python benchmark/verify.py --slug sde_gbm
uv run python benchmark/verify.py --slug pde_heat_1d --plan-dir workspace/pde_heat_1d/plans/2-crank-nicolson
```

Regenerate just the report from existing data:

```bash
uv run python benchmark/report.py            # results/results.json → results/REPORT.md
```

## Baseline comparison (does the *structure* help?)

The pipeline mixes Opus and Sonnet across several agents, so beating a weaker
setup could just mean "a bigger model." To isolate the **structure**, `oneshot.py`
runs the same problems through weaker structures on the *same* model, and
`compare.py` renders the head-to-head. Three rungs:

| Rung | What | Isolates |
|---|---|---|
| **C0** | naked one-shot: a single tool-less completion, given only `problem.md`, returns `solver.py` | "vs pasting it into a chat" |
| **C1** | single agent + tools: one agent may write, run, and iterate on `solver.py` (Bash/Read/Write), but with none of the pipeline's decomposition | the value of the multi-agent structure, beyond mere tool access + iteration |
| **C2** | the full pipeline (`run.py`) | — |

Each baseline is also run **naked** (`problem.md` only) or **manual** (+ the same
`sde_manual.md` / `pde_manual.md` the pipeline's agents read), to separate the
structure's contribution from the reference material's.

**Fairness is enforced mechanically.** Both sides read the identical `problem.md`
(which does *not* leak the ground-truth moments), and every baseline `solver.py`
is graded by the *same* `verify.py` at the *same* `(num_paths, eval dt, seed)`.
Baseline code is untrusted, so it runs through `runner.py`: a fresh subprocess with
a wall-clock timeout, a best-effort memory cap, and the **same import rule the
pipeline solvers follow** — numpy + stdlib for SDE (`solver-sde.md`), numpy +
`scipy.sparse` for PDE (`solver-pde.md`). A baseline that hangs, OOMs, violates
the contract, or reaches for a forbidden library simply **fails** (there is no
self-score to be unsure about):

| One-shot verdict | Meaning |
|---|---|
| `PASS` / `STABLE_PASS` | independently confirmed (moments/L2 within gate; or stayed finite/in-domain) |
| `FAIL` / `BLOWUP` | ran, but out of tolerance / non-finite / escaped its domain |
| `CRASHED` | never produced a scorable run: no solver, bad signature, exception, timeout, OOM, or a blocked import — counts as a failure |
| `NO_GT` | no independent check exists (only Kuramoto–Sivashinsky) — excluded from the rate |

```bash
# ALWAYS smoke-test the plumbing cheaply first (haiku, 2 problems):
uv run python benchmark/oneshot.py --only sde_gbm,pde_heat_1d --condition c0 --model haiku

# the hard discriminator subset, both conditions + knowledge settings, Opus, 5 reps:
uv run python benchmark/oneshot.py --discriminators --condition c0,c1 --manuals both --model opus --reps 5

# a Sonnet-pinned pipeline sweep (for the cross-tier cell), saved aside so it does
# not clobber the Opus+Sonnet results.json, then the head-to-head:
uv run python benchmark/run.py --model sonnet          # then copy results.json → results/pipeline_sonnet.json
uv run python benchmark/compare.py \
    --pipeline results/results.json=C2a:opus+sonnet \
    --pipeline results/pipeline_sonnet.json=C2b:sonnet
```

The **discriminator subset** (`problems.discriminator_problems()`, currently the 7
hard-literature SDEs) is where the thesis has to show up — run it at higher `--reps`.
Flag incoming hard PDEs with `"discriminator": True` to fold them in.

> **A caveat for the PDE side:** several PDE `problem.md`s state their closed-form
> solution, so a baseline (or the pipeline) *could* pass by evaluating the given
> formula instead of solving numerically. This is now closed on two fronts:
> **(1) leakage** — no PDE `problem.md` states its solution (MMS problems ship the
> explicit source term instead, so they stay fully specified); **(2) the order
> check** — a solver must actually *converge* under refinement, so hard-coding a
> single-grid answer fails. Hard PDEs added later should stay closed-form-free in
> the statement the same way.

## Pass criteria

Identical to the pipeline's own thresholds, applied independently:

- **PDE:** the solver is `solve_pde(N)`, run at `N` and `2N`. A pass requires **both**
  relative L2 error `< 1%` at the fine grid **and** an observed spatial convergence
  order `p = log2(err_N / err_2N)` at least the problem's floor (`min_order`, default
  1.0) — so a scheme accurate on one grid but not converging fails. Legacy no-argument
  `solve_pde()` solvers are still verified single-grid (relative L2 `< 1%`). A shock
  problem (fractional L2 order) can waive the order check via `pde_order_check=False`.
- **SDE (exact):** variance relative error `< 10%` **and** (mean relative error `< 5%`
  or the exact mean is within `0.01` of zero). Moments are recomputed at the fixed
  `(num_paths, dt, seed)` recorded in `manifest.json`.
- **SDE (reference):** the same thresholds, but widened to at least `3×` the reference
  Monte-Carlo standard error so a correct solution within reference noise is not
  failed.
- **SDE (stability):** every terminal state is finite (and, for a domain problem, all
  within `|X| < bound`). Any Inf/NaN or domain escape is a `BLOWUP`.

The independent re-run scales the step to the horizon (`dt = min(problem_dt, 0.04/T)`)
so a *correct* scheme's discretization error cannot masquerade as a failure.

## The 37 problems

**PDE (15)** — T1: `heat_1d`, `heat_2d`, `wave_1d`, `advection_1d`, `poisson_2d`;
T2: `laplace_2d`, `convection_diffusion_bl` (ε=1e-3 layer), `helmholtz_2d` (indefinite),
`anisotropic_diffusion` (100:1), `wave_2d`;
T3: `burgers_inviscid` (shock), `fokker_planck_ou`, `fractional_diffusion` (Caputo α=½),
`black_scholes_call`, `kuramoto_sivashinsky` (chaotic, no closed form).

**SDE (22)** — T1: `gbm`, `ornstein_uhlenbeck`, `bm_with_drift`, `linear_additive`,
`bm_standard`; T2: `cir`, `exponential_ou`, `black_scholes`, `gbm_2d_correlated`,
`stochastic_oscillator`; T3 (classical stress): `cir_feller_violated` (zero-hitting),
`gbm_high_vol` (heavy tail), `ou_stiff` (θ=50), `gbm_2d_high_corr` (ρ=0.95),
`oscillator_long_horizon`.

**SDE T3 — hard-literature additions (7):**

| Slug | Kind | What makes it hard |
|---|---|---|
| `log_heston_feller_violated` | exact | coupled 2-D affine (log-price + CIR variance), 2a=0.2 < σ²=1.0 ⇒ variance hits zero; naive √Y-Euler goes negative. Exact moments from a closed affine ODE system. |
| `multichannel_stiff_m13` | exact | 13 non-commuting multiplicative noise channels on a spiral drift; exact moments via matrix exponential / Lyapunov ODE. |
| `quintic_random_ic` | exact | superlinear `dX=-X⁵dt` with a Gaussian IC; Gauss–Hermite variance. Untamed Euler overshoots on the tail. |
| `ginzburg_landau_s4`, `ginzburg_landau_s6` | reference | superlinear cubic drift with multiplicative noise; untamed Euler biases E[X²] low (σ=6 also NaNs). Reference = MC of the exact solution. |
| `quintic_drift_noise` | stability | `dX=-X⁵dt + X dW`; plain Euler diverges (HJK), tamed stays finite. |
| `fene_blowup` | stability | singular spring `dX=-X/(1-X²)dt + dW` confined to (-1,1); a fixed step overshoots the boundary and escapes to NaN. |

## Adding a problem

Append an entry to `PROBLEMS` in [`problems.py`](problems.py) with metadata, a
`description` (the `problem.md` body), and — per its `ground_truth_kind` (default
`exact`):

- `exact`: a ground-truth callable (`moments(T) -> {mean, variance, ...}` for SDE,
  `analytic(t, *coords)` for PDE).
- `reference`: `ground_truth_kind="reference"` and `moments(T) -> {key: (value, se)}`
  from a discretization-free MC of the exact solution.
- `stability`: `ground_truth_kind="stability"`, `has_ground_truth=False`, and a
  `stability_check` (`{"type": "finite"}` or `{"type": "domain", "abs_bound": ...}`).
- a no-reference stress test: `has_ground_truth=False`, `analytic=None` (default
  `exact` kind → reported as `SELF_ONLY`).

`setup.py` automatically appends a **Solver contract** section to every `problem.md`
— the `solve_sde(num_paths, dt, T, seed)` / `solve_pde(N)` signature, the return
schema, and the *exact* arguments the verifier will call it with (SDE: `num_paths`,
the evaluation `dt = min(problem_dt, 0.04/T)` from `verify.sde_verify_dt`, `T`,
`seed`; PDE: the two resolutions `N` and `2N` plus the accuracy tolerance and the
observed-order floor). This keeps the harness API out of the numerical problem and
makes a one-shot LLM given only `problem.md` a fair comparison to the pipeline.
PDE tuning knobs on a problem: `grid_N` (base resolution, default 64), `min_order`
(order floor, default 1.0), `pde_order_check` (default True; False for shocks).

Then re-run `uv run python benchmark/validate_ground_truth.py` before trusting it in a
benchmark run. It also checks that a *correct* scheme actually clears both gates at
that evaluation `dt` (the mean gate, 5%, is tighter than the variance gate, 10%).
