# Autonumerics Benchmark Report

_Generated 2026-11-26 17:00_

Pipeline invocation: `claude --plugin-dir . -p "/conductor workspace/{slug}/problem.md"`

Active time, most recent run invocation: **10.1m**

## Executive summary

- **Problems:** 9 (0 PDE, 9 SDE); 9 with exact/reference ground truth.
- **Runs completed:** 9/9.
- **Pipeline self-reported pass (10/10):** 5/9 (56%).
- **Independently confirmed pass:** 9/9 (100% of independently-checkable problems).
- **OVERCLAIMS / BLOW-UPS:** none — every success the pipeline reported was independently confirmed.
- **Wrong plan won:** 0/9 — the winner was less accurate than another evaluated plan by more than the reference error (see *Every plan, graded*).

### Verdict distribution

| Verdict | Count | Share |
|---|---|---|
| PASS (verified) | 5 | 56% |
| UNDERCLAIM | 4 | 44% |

### Independently-verified pass rate by type and tier

| Type | Tier | N | Pipeline pass | Independently verified |
|---|---|---|---|---|
| PDE | Tier 1 | 0 | 0/0 | n/a |
| PDE | Tier 2 | 0 | 0/0 | n/a |
| PDE | Tier 3 | 0 | 0/0 | n/a |
| SDE | Tier 1 | 0 | 0/0 | n/a |
| SDE | Tier 2 | 0 | 0/0 | n/a |
| SDE | Tier 3 | 9 | 5/9 | 9/9 |

> Tier 1 is textbook material the reference manuals cover; Tier 2 needs care (boundary layers, indefinite operators, multi-D, guards); Tier 3 pushes beyond the manuals (shocks, fractional operators, stiffness, heavy-tailed sampling, chaos, and the hard-SDE additions: Feller-violated log-Heston, many-channel stiffness, superlinear/tamed drift, and blow-up/domain stability). Independently verified counts both exact/reference moment passes and stability (no blow-up) passes.

## Independent-verification discrepancies

### N01 sde_clark_cameron — UNDERCLAIM

Pipeline stopped at **9/10**, but the independent check **passes** (ok). The solution was actually good enough; the pipeline under-scored or ran out of iterations.

### N02 sde_kubo_oscillator — UNDERCLAIM

Pipeline stopped at **9/10**, but the independent check **passes** (ok). The solution was actually good enough; the pipeline under-scored or ran out of iterations.

### N07 sde_langevin_baoab — UNDERCLAIM

Pipeline stopped at **9/10**, but the independent check **passes** (varErr=0.7%, meanErr=0.7%). The solution was actually good enough; the pipeline under-scored or ran out of iterations.

### N08 sde_three_halves_vol — UNDERCLAIM

Pipeline stopped at **9/10**, but the independent check **passes** (funcErr[inv_var_mean:6.3e-04 inv_var_variance:2.7e-04]). The solution was actually good enough; the pipeline under-scored or ran out of iterations.

## Full results

| ID | Slug | Type | Tier | Pipe | Best plan | Iters | Independent metric | Verdict | Time | Cost |
|---|---|---|---|---|---|---|---|---|---|---|
| N01 | sde_clark_cameron | SDE | 3 | 9 | 1-levy-area-milstein-drift-corrected | 6 | ok | UNDERCLAIM | 18.3m | $6.69 |
| N02 | sde_kubo_oscillator | SDE | 3 | 9 | 2-stochastic-heun | 6 | ok | UNDERCLAIM | 13.3m | $5.89 |
| N03 | sde_aitsahalia_rate | SDE | 3 | 10 | 1-lamperti-trapezoidal | 3 | varErr=0.4%, meanErr=0.0% (ref) | PASS (verified) | 15.4m | $5.20 |
| N04 | sde_wright_fisher_mutation | SDE | 3 | 10 | 1-beta-moment-matched | 3 | varErr=0.2%, meanErr=0.0% | PASS (verified) | 9.6m | $4.38 |
| N05 | sde_merton_jump_call | SDE | 3 | 10 | 1-milstein-cv | 3 | funcErr[call_price:3.7e-03] | PASS (verified) | 26.7m | $5.80 |
| N06 | sde_sis_epidemic | SDE | 3 | 10 | 1-logit-strang-heun | 3 | varErr=0.7%, meanErr=0.4% (ref) | PASS (verified) | 10.9m | $3.95 |
| N07 | sde_langevin_baoab | SDE | 3 | 9 | 2-stochastic-trapezoidal | 6 | varErr=0.7%, meanErr=0.7% | UNDERCLAIM | 8.2m | $3.89 |
| N08 | sde_three_halves_vol | SDE | 3 | 9 | 3-log-implicit-euler | 6 | funcErr[inv_var_mean:6.3e-04 inv_var_variance:2.7e-04] | UNDERCLAIM | 12.0m | $5.32 |
| N09 | sde_geometric_asian | SDE | 3 | 10 | 1-milstein-bridge-cv | 3 | funcErr[call_price:2.9e-03 geo_average_variance:3.5e-03] | PASS (verified) | 8.3m | $3.21 |

## Tier-by-tier detail

### PDE

### SDE

**Tier 3**

- `N01 sde_clark_cameron` (clark_cameron) — **UNDERCLAIM**. The counterexample of the strong-convergence literature: no scheme built from the Brownian increments alone beats strong order 1/2 on U = int S dW1 (Clark & Cameron 1980). Fine sub-increments are supplied so a Milstein scheme that assembles the Levy area from them converges at order 1; Euler-Maruyama at the graded step does not. Independent: ok.
- `N02 sde_kubo_oscillator` (kubo_oscillator) — **UNDERCLAIM**. The textbook Stratonovich system (Kloeden & Platen; Milstein, Repin & Tretyakov 2002): the exact flow is a rotation and conserves |X|. Reading the coefficients as Ito solves a different equation whose amplitude grows by e^{sigma^2 T/2}. Independent: ok.
- `N03 sde_aitsahalia_rate` (aitsahalia) — **PASS (verified)**. Ait-Sahalia's (1996) strongly nonlinear short rate: a 1/X drift that repels the origin, a superlinear X^2 mean reversion and an X^{3/2} diffusion. Backward Euler-Maruyama preserves positivity (Szpruch, Mao, Higham & Pan 2011); the explicit step overshoots into X < 0 and the variance it reports is garbage. Independent: varErr=0.4%, meanErr=0.0% (ref).
- `N04 sde_wright_fisher_mutation` (wright_fisher) — **PASS (verified)**. Kimura's diffusion of population genetics (the Jacobi diffusion in finance), bounded in [0,1] with both boundaries attainable at these mutation rates. A boundary-preserving step is required; clipping to [0,1] stays finite and biases the mean by 8%. Independent: varErr=0.2%, meanErr=0.0%.
- `N05 sde_merton_jump_call` (merton_jump) — **PASS (verified)**. Merton's (1976) lognormal jump diffusion, the first jump-diffusion option model, at a published parameter set. The only non-Gaussian driver in the benchmark; the compensator is the decision -- dropping it moves the mean by 1.4% (inside the gate) and the price by 19%. Independent: funcErr[call_price:3.7e-03].
- `N06 sde_sis_epidemic` (stochastic_sis) — **PASS (verified)**. The SDE SIS model of Gray, Greenhalgh, Hu, Mao & Pan (2011) at its own Example 5.2: a superlinear sigma I (N - I) diffusion on (0, N) and a stiff logistic drift. Explicit Euler diverges at the graded step; the same splitting with the noise step clipped in I rather than taken in the logit chart biases the mean by 10%. Independent: varErr=0.7%, meanErr=0.4% (ref).
- `N07 sde_langevin_baoab` (underdamped_langevin) — **UNDERCLAIM**. The workhorse of molecular sampling, at moderate friction. Leimkuhler & Matthews (2012/13) compare integrators by the invariant measure they sample: BAOAB's bias on Var(q) is a hundredth of the gate here, Euler-Maruyama's is 2.8x the gate while linearly stable. Independent: varErr=0.7%, meanErr=0.7%.
- `N08 sde_three_halves_vol` (three_halves) — **UNDERCLAIM**. The 3/2 model of Ahn & Gao (1999) at Drimus's (2012) S&P 500 fit: a V^{3/2} diffusion with a vol-of-vol of 8.56. 1/V is a CIR process with exact moments; explicit Euler on V reflected at 0 biases E[1/V] by 18% and Var[1/V] by 39x. Independent: funcErr[inv_var_mean:6.3e-04 inv_var_variance:2.7e-04].
- `N09 sde_geometric_asian` (geometric_asian) — **PASS (verified)**. Kemna & Vorst's (1990) closed-form Asian option, the standard MLMC test problem (Giles 2008). The only entry grading a time integral along the path: an endpoint Riemann sum of int ln S dt misprices the call by 3.4% against a 1% gate; the trapezoid by 0.02%. Independent: funcErr[call_price:2.9e-03 geo_average_variance:3.5e-03].

## Scheme coverage and efficiency

Winning scheme among verified passes:

| Scheme (plan slug) | Times best |
|---|---|
| lamperti-trapezoidal | 1 |
| beta-moment-matched | 1 |
| milstein-cv | 1 |
| logit-strang-heun | 1 |
| milstein-bridge-cv | 1 |

- Mean solver-evaluator iterations per problem (summed over plans): 4.3.
- Mean wall time per completed run: 13.6m; max 26.7m.
- Pipeline cost, 9 of 9 completed runs priced: **$44.32** total, $4.92 mean, $6.69 max.

## Error vs work

Each row is one solve of the winning plan at a declared resolution. `evals = paths x steps x stages` is the language-free work axis: `stages` is the plan's *declared* scheme family's cost per step, so a family that is not declared counts as 1 and says so. The slope is the least-squares log-log fit of error against work over the series, reported only where three or more points resolve it.

**N01 sde_clark_cameron** — scheme milstein (declared); d log err / d log evals = -0.42; observed order in dt = 0.42

| dt | steps | paths | stages | evals | rel err | solve s | graded |
|---|---|---|---|---|---|---|---|
| 0.04 | 25 | 1000 | 1 | 25000 | 4.468e-03 | 0.042 | * |
| 0.02 | 50 | 1000 | 1 | 50000 | 3.128e-03 | 0.034 |  |
| 0.01 | 100 | 1000 | 1 | 100000 | 2.487e-03 | 0.036 |  |

**N05 sde_merton_jump_call** — scheme milstein (declared)

| dt | steps | paths | stages | evals | rel err | solve s | graded |
|---|---|---|---|---|---|---|---|
| 0.01 | 25 | 100000 | 1 | 2.5e+06 | 6.447e-03 | 0.047 |  |
| 0.01 | 25 | 400000 | 1 | 1e+07 | 3.718e-03 | 0.205 | * |

**N08 sde_three_halves_vol** — scheme implicit (declared)

| dt | steps | paths | stages | evals | rel err | solve s | graded |
|---|---|---|---|---|---|---|---|
| 0.004 | 25 | 50000 | 1 | 1.25e+06 | 5.827e-04 | 0.076 |  |
| 0.004 | 25 | 200000 | 1 | 5e+06 | 6.259e-04 | 0.312 | * |

**N09 sde_geometric_asian** — scheme milstein (declared)

| dt | steps | paths | stages | evals | rel err | solve s | graded |
|---|---|---|---|---|---|---|---|
| 0.04 | 25 | 100000 | 1 | 2.5e+06 | 7.151e-03 | 0.084 |  |
| 0.04 | 25 | 400000 | 1 | 1e+07 | 3.510e-03 | 0.354 | * |

## Failures and errors

No failures, overclaims, blow-ups, run errors, or unverifiable results.

## Kernel provenance vs the independent verdict

What the pipeline's own kernel says its score rests on (`provenance`, from `verifylib/kernel/score.py`), against what the harness's independent check found. The kernel never sees the ground truth these verdicts are computed from, so agreement here is evidence and disagreement is a finding.

| kernel provenance | VERIFIED_PASS | UNDERCLAIM |
|---|---|---|
| analytic | 3 | 3 |
| manufactured_partial |  | 1 |
| surrogate | 2 |  |

| problem | kernel score | provenance | tier | independent verdict |
|---|---|---|---|---|
| sde_aitsahalia_rate | 10 | surrogate | A_prime | PASS (verified) |
| sde_clark_cameron | 9 | analytic | A | UNDERCLAIM |
| sde_geometric_asian | 10 | analytic | A | PASS (verified) |
| sde_kubo_oscillator | 9 | analytic | A | UNDERCLAIM |
| sde_langevin_baoab | 9 | analytic | A | UNDERCLAIM |
| sde_merton_jump_call | 10 | analytic | A | PASS (verified) |
| sde_sis_epidemic | 10 | surrogate | A_prime | PASS (verified) |
| sde_three_halves_vol | 8 | manufactured_partial | C | UNDERCLAIM |
| sde_wright_fisher_mutation | 10 | analytic | A | PASS (verified) |

## Every plan, graded

The harness grades every plan directory with a `solver.py`, not only the winner. **`wrong_plan_won`: 0/9** -- the conductor's winner was less accurate than another evaluated plan by more than the reference's own error. This is the number the Path-B ranking rules exist to drive to zero, and it is invisible in a winner-only record.

| kernel provenance (per plan) | VERIFIED_PASS | UNDERCLAIM | FAIL |
|---|---|---|---|
| analytic | 7 | 8 | 3 |
| manufactured_partial |  | 3 |  |
| surrogate | 5 |  | 1 |

| problem | plan (* winner) | kernel score | provenance | harness rel err | harness err / gate | verdict |
|---|---|---|---|---|---|---|
| sde_aitsahalia_rate | 1-lamperti-trapezoidal * | 10.0 | surrogate | - | - | PASS (verified) |
| sde_aitsahalia_rate | 2-drift-implicit-milstein | 10.0 | surrogate | - | - | PASS (verified) |
| sde_aitsahalia_rate | 3-drift-implicit-euler | 10.0 | surrogate | - | - | PASS (verified) |
| sde_clark_cameron | 1-levy-area-milstein-drift-corrected * | 9.0 | analytic | - | - | UNDERCLAIM |
| sde_clark_cameron | 2-levy-area-milstein | 9.0 | analytic | - | - | UNDERCLAIM |
| sde_clark_cameron | 3-step-increment-conditional-expectation | 9.0 | analytic | - | - | FAIL |
| sde_geometric_asian | 1-milstein-bridge-cv * | 10.0 | analytic | - | 0.287 | PASS (verified) |
| sde_geometric_asian | 2-log-euler-bridge | 10.0 | analytic | - | 0.104 | PASS (verified) |
| sde_geometric_asian | 3-euler-maruyama-trapezoid | 8.0 | analytic | - | 0.132 | UNDERCLAIM |
| sde_kubo_oscillator | 1-cayley-midpoint | 9.0 | analytic | - | - | UNDERCLAIM |
| sde_kubo_oscillator | 2-stochastic-heun * | 9.0 | analytic | - | - | UNDERCLAIM |
| sde_kubo_oscillator | 3-milstein | 9.0 | analytic | - | - | UNDERCLAIM |
| sde_langevin_baoab | 1-baoab-splitting | 9.0 | analytic | - | - | UNDERCLAIM |
| sde_langevin_baoab | 2-stochastic-trapezoidal * | 9.0 | analytic | - | - | UNDERCLAIM |
| sde_langevin_baoab | 3-euler-maruyama | 4.0 | analytic | - | - | FAIL |
| sde_merton_jump_call | 1-milstein-cv * | 10.0 | analytic | - | 0.372 | PASS (verified) |
| sde_merton_jump_call | 2-weak-order2-taylor | 10.0 | analytic | - | 0.307 | PASS (verified) |
| sde_merton_jump_call | 3-euler-maruyama | 10.0 | analytic | - | 0.352 | PASS (verified) |
| sde_sis_epidemic | 1-logit-strang-heun * | 10.0 | surrogate | - | - | PASS (verified) |
| sde_sis_epidemic | 2-strang-milstein-in-i | 10.0 | surrogate | - | - | PASS (verified) |
| sde_sis_epidemic | 3-logit-drift-implicit-euler | 7.0 | surrogate | - | - | FAIL |
| sde_three_halves_vol | 1-reciprocal-implicit-milstein | 9.0 | manufactured_partial | - | 0.286 | UNDERCLAIM |
| sde_three_halves_vol | 2-lamperti-implicit-euler | 9.0 | manufactured_partial | - | 0.055 | UNDERCLAIM |
| sde_three_halves_vol | 3-log-implicit-euler * | 9.0 | manufactured_partial | - | 0.013 | UNDERCLAIM |
| sde_wright_fisher_mutation | 1-beta-moment-matched * | 10.0 | analytic | - | - | PASS (verified) |
| sde_wright_fisher_mutation | 2-milstein-reflected | 10.0 | analytic | - | - | PASS (verified) |
| sde_wright_fisher_mutation | 3-euler-maruyama-projected | 8.0 | analytic | - | - | FAIL |

## Methodology and reproducibility

- **Pipeline pass** = the conductor's best plan reached the terminal self-score of 10/10 (PDE: rel L2 < 1%; SDE: variance rel err < 10% and, unless the mean is near zero, mean rel err < 5%).
- **Independent verification** re-imports the best plan's `solver.py`, re-runs it, and compares the raw numerical output to ground truth authored and validated in `benchmark/problems.py` — never to the formulator's extracted solution. SDE moments use the fixed `(num_paths, dt, seed)` recorded in the manifest; PDE errors use the RMS-based relative L2 norm from `pde_manual.md`.
- **Ground-truth kinds:** *exact* (closed-form / machine-precision moments or analytic solution); *reference* (no elementary moment formula — the ground truth is a discretization-free Monte-Carlo of the known exact solution, compared within an SE-aware tolerance); *stability* (no ground truth — the independent check only confirms a correct scheme stays finite / in its domain, which is what a naive scheme gets wrong).
- **Five suites, five baseline denominators.** `main` (49) is the default; `no-closed-form` (7), `benchop` (10), `sciml` (9) and `notable` (9) each run only when asked for by `--suite`, because mixing them into the main suite would move the denominator every published pass rate is quoted against. `benchop` is graded against 24 published reference values; `sciml` is graded against truth this repo owns and *compared* against another library's schemes, which is why it carries the `scheme-matched` / `unrestricted` arm distinction and the `ANALYTIC_TRANSCRIPTION` exclusion; `notable` is nine recognisable, cited SDE problems each punishing one numerical decision, run in both arms, with the arm staged into the brief and the workspace path on every route. A run records its arm; two arms are never merged.
- **Problems with no closed form** (`closed_form: False`) run as their own suite (`--no-closed-form`) and are excluded by default, so the main benchmark's baseline denominators stay comparable. They are still independently checked, by one of four routes: a harness-owned high-resolution *reference field* (two independent method families agreeing far below the pass tolerance, built by `make_references.py` and recorded as the problem's `reference_error`, which widens the tolerance so the answer key's own error cannot fail a correct solver); *semi-analytic exact* truth from a quadrature, series or eigen-expansion; a scalar *functional* (`functional_truth`); and structural gates. See `benchmark/design/NO_CLOSED_FORM_CANDIDATES.md`.
- Ground truth for all 9 exact/reference problems is validated independently (SDE moments against moment-ODE integration, matrix-exponential / affine systems, and direct simulation; the Ginzburg-Landau reference anchored to the literature value E[X_1^2]=0.8114 at sigma=2; PDE solutions against their defining identities).
- Runs are **sequential**; each problem has a wall-clock timeout and its full transcript is saved under `benchmark/results/logs/`.
