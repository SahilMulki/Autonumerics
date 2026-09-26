# Autonumerics Benchmark Report

_Generated 2026-09-25 23:56_

Pipeline invocation: `claude --plugin-dir . -p "/conductor workspace/{slug}/problem.md"`

Active time, most recent run invocation: **54.8m**

## Executive summary

- **Problems:** 3 (2 PDE, 1 SDE); 3 with exact/reference ground truth.
- **Runs completed:** 3/3.
- **Pipeline self-reported pass (10/10):** 3/3 (100%).
- **Independently confirmed pass:** 3/3 (100% of independently-checkable problems).
- **OVERCLAIMS / BLOW-UPS:** none — every success the pipeline reported was independently confirmed.
- **Wrong plan won:** 0/2 — the winner was less accurate than another evaluated plan by more than the reference error (see *Every plan, graded*).

### Verdict distribution

| Verdict | Count | Share |
|---|---|---|
| PASS (verified) | 3 | 100% |

### Independently-verified pass rate by type and tier

| Type | Tier | N | Pipeline pass | Independently verified |
|---|---|---|---|---|
| PDE | Tier 1 | 0 | 0/0 | n/a |
| PDE | Tier 2 | 0 | 0/0 | n/a |
| PDE | Tier 3 | 2 | 2/2 | 2/2 |
| SDE | Tier 1 | 0 | 0/0 | n/a |
| SDE | Tier 2 | 0 | 0/0 | n/a |
| SDE | Tier 3 | 1 | 1/1 | 1/1 |

> Tier 1 is textbook material the reference manuals cover; Tier 2 needs care (boundary layers, indefinite operators, multi-D, guards); Tier 3 pushes beyond the manuals (shocks, fractional operators, stiffness, heavy-tailed sampling, chaos, and the hard-SDE additions: Feller-violated log-Heston, many-channel stiffness, superlinear/tamed drift, and blow-up/domain stability). Independently verified counts both exact/reference moment passes and stability (no blow-up) passes.

## Independent-verification discrepancies

None. Every pipeline self-score agreed with the independent ground-truth check.

## Full results

| ID | Slug | Type | Tier | Pipe | Best plan | Iters | Independent metric | Verdict | Time |
|---|---|---|---|---|---|---|---|---|---|
| P32 | pde_kdv_zabusky_kruskal | PDE | 3 | 10 | 1-spectral-etdrk4 | 3 | relL2=5.02e-09, p=5.47 | PASS (verified) | 39.1m |
| P33 | pde_gross_pitaevskii_1d | PDE | 3 | 10 | 3-fourier-spectral-yoshida4 | 2 | relL2=2.00e-06, p=5.71 [re:2.0e-06 im:2.0e-06] | PASS (verified) | 6.5m |
| S23 | sde_double_well_langevin | SDE | 3 | 10 | 2-drift-implicit-trapezoid | 3 | varErr=0.2%, meanErr=1.0% | PASS (verified) | 47.7m |

## Tier-by-tier detail

### PDE

**Tier 3**

- `P32 pde_kdv_zabusky_kruskal` (kdv) — **PASS (verified)**. Dispersive third-order operator; a cosine fissions into a train of solitons of width ~delta whose phases a second-order scheme gets wrong at any affordable grid. Three exact invariants; ground truth is a harness-owned spectral reference. Independent: relL2=5.02e-09, p=5.47.
- `P33 pde_gross_pitaevskii_1d` (gross-pitaevskii) — **PASS (verified)**. A trapped cubic Schrodinger equation as a real/imaginary two-field system; the error is a phase error that a second-order discretization cannot bring under 1% at the graded grid. Mass and energy are exact invariants; ground truth is a harness-owned sine-spectral reference on the Dirichlet box. Independent: relL2=2.00e-06, p=5.71 [re:2.0e-06 im:2.0e-06].

### SDE

**Tier 3**

- `S23 sde_double_well_langevin` (double_well) — **PASS (verified)**. dX = k(X - X^3)dt + sigma dW started in one well; the mean at T is the fraction of paths that have not crossed the barrier, and a first-order weak scheme misprices the escape rate by O(k dt) -- outside the gate at the step given. No closed form for the transient or its moments; truth is a Kolmogorov-equation solve. Independent: varErr=0.2%, meanErr=1.0%.

## Scheme coverage and efficiency

Winning scheme among verified passes:

| Scheme (plan slug) | Times best |
|---|---|
| spectral-etdrk4 | 1 |
| fourier-spectral-yoshida4 | 1 |
| drift-implicit-trapezoid | 1 |

- Mean solver-evaluator iterations per problem (summed over plans): 2.7.
- Mean wall time per completed run: 31.1m; max 47.7m.

## Failures and errors

No failures, overclaims, blow-ups, run errors, or unverifiable results.

## Kernel provenance vs the independent verdict

What the pipeline's own kernel says its score rests on (`provenance`, from `verifylib/kernel/score.py`), against what the harness's independent check found. The kernel never sees the ground truth these verdicts are computed from, so agreement here is evidence and disagreement is a finding.

| kernel provenance | VERIFIED_PASS |
|---|---|
| manufactured | 2 |
| surrogate | 1 |

| problem | kernel score | provenance | tier | independent verdict |
|---|---|---|---|---|
| pde_gross_pitaevskii_1d | 10 | manufactured | B | PASS (verified) |
| pde_kdv_zabusky_kruskal | 10 | manufactured | B | PASS (verified) |
| sde_double_well_langevin | 10 | surrogate | A_prime | PASS (verified) |

## Every plan, graded

The harness grades every plan directory with a `solver.py`, not only the winner. **`wrong_plan_won`: 0/2** -- the conductor's winner was less accurate than another evaluated plan by more than the reference's own error. This is the number the Path-B ranking rules exist to drive to zero, and it is invisible in a winner-only record.

| kernel provenance (per plan) | VERIFIED_PASS | UNDERCLAIM | FAIL |
|---|---|---|---|
| - |  | 1 | 1 |
| manufactured | 5 |  |  |
| surrogate | 1 | 2 |  |

| problem | plan (* winner) | kernel score | provenance | harness rel err | verdict |
|---|---|---|---|---|---|
| pde_gross_pitaevskii_1d | 1-fd4-strang-cn | 10.0 | manufactured | 6.188e-04 | PASS (verified) |
| pde_gross_pitaevskii_1d | 2-fd4-yoshida4 | - | - | 3.394e-03 | UNDERCLAIM |
| pde_gross_pitaevskii_1d | 3-fourier-spectral-yoshida4 * | 10.0 | manufactured | 1.997e-06 | PASS (verified) |
| pde_kdv_zabusky_kruskal | 1-spectral-etdrk4 * | 10.0 | manufactured | 5.021e-09 | PASS (verified) |
| pde_kdv_zabusky_kruskal | 2-spectral-ifrk4 | 10.0 | manufactured | 4.933e-08 | PASS (verified) |
| pde_kdv_zabusky_kruskal | 3-fd4-implicit-midpoint | 10.0 | manufactured | 2.052e-04 | PASS (verified) |
| pde_kdv_zabusky_kruskal | 4-fd2-explicit-rk4 | - | - | 1.856e-02 | FAIL |
| sde_double_well_langevin | 1-tamed-heun-drift-trapezoid | 3.0 | surrogate | - | UNDERCLAIM |
| sde_double_well_langevin | 2-drift-implicit-trapezoid * | 10.0 | surrogate | - | PASS (verified) |
| sde_double_well_langevin | 3-strang-splitting-exact-flows | 3.0 | surrogate | - | UNDERCLAIM |

## Methodology and reproducibility

- **Pipeline pass** = the conductor's best plan reached the terminal self-score of 10/10 (PDE: rel L2 < 1%; SDE: variance rel err < 10% and, unless the mean is near zero, mean rel err < 5%).
- **Independent verification** re-imports the best plan's `solver.py`, re-runs it, and compares the raw numerical output to ground truth authored and validated in `benchmark/problems.py` — never to the formulator's extracted solution. SDE moments use the fixed `(num_paths, dt, seed)` recorded in the manifest; PDE errors use the RMS-based relative L2 norm from `pde_manual.md`.
- **Ground-truth kinds:** *exact* (closed-form / machine-precision moments or analytic solution); *reference* (no elementary moment formula — the ground truth is a discretization-free Monte-Carlo of the known exact solution, compared within an SE-aware tolerance); *stability* (no ground truth — the independent check only confirms a correct scheme stays finite / in its domain, which is what a naive scheme gets wrong).
- **Problems with no closed form** (`closed_form: False`) run as their own suite (`--no-closed-form`) and are excluded by default, so the main benchmark's baseline denominators stay comparable. They are still independently checked, by one of four routes: a harness-owned high-resolution *reference field* (two independent method families agreeing far below the pass tolerance, built by `make_references.py` and recorded as the problem's `reference_error`, which widens the tolerance so the answer key's own error cannot fail a correct solver); *semi-analytic exact* truth from a quadrature, series or eigen-expansion; a scalar *functional* (`functional_truth`); and structural gates. See `benchmark/NO_CLOSED_FORM_CANDIDATES.md`.
- Ground truth for all 3 exact/reference problems is validated independently (SDE moments against moment-ODE integration, matrix-exponential / affine systems, and direct simulation; the Ginzburg-Landau reference anchored to the literature value E[X_1^2]=0.8114 at sigma=2; PDE solutions against their defining identities).
- Runs are **sequential**; each problem has a wall-clock timeout and its full transcript is saved under `benchmark/results/logs/`.
