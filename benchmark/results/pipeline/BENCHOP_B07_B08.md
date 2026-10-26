# Autonumerics Benchmark Report

_Generated 2026-10-26 18:38_

Pipeline invocation: `claude --plugin-dir . -p "/conductor workspace/{slug}/problem.md"`

Active time, most recent run invocation: **70.9m**

## Executive summary

- **Problems:** 1 (0 PDE, 1 SDE); 1 with exact/reference ground truth.
- **Runs completed:** 1/1.
- **Pipeline self-reported pass (10/10):** 0/1 (0%).
- **Independently confirmed pass:** 0/1 (0% of independently-checkable problems).
- **OVERCLAIMS / BLOW-UPS:** none — every success the pipeline reported was independently confirmed.
- **Wrong plan won:** 0/1 — the winner was less accurate than another evaluated plan by more than the reference error (see *Every plan, graded*).

### Verdict distribution

| Verdict | Count | Share |
|---|---|---|
| UNDERCLAIM | 1 | 100% |

### Independently-verified pass rate by type and tier

| Type | Tier | N | Pipeline pass | Independently verified |
|---|---|---|---|---|
| PDE | Tier 1 | 0 | 0/0 | n/a |
| PDE | Tier 2 | 0 | 0/0 | n/a |
| PDE | Tier 3 | 0 | 0/0 | n/a |
| SDE | Tier 1 | 0 | 0/0 | n/a |
| SDE | Tier 2 | 0 | 0/0 | n/a |
| SDE | Tier 3 | 1 | 0/1 | 0/1 |

> Tier 1 is textbook material the reference manuals cover; Tier 2 needs care (boundary layers, indefinite operators, multi-D, guards); Tier 3 pushes beyond the manuals (shocks, fractional operators, stiffness, heavy-tailed sampling, chaos, and the hard-SDE additions: Feller-violated log-Heston, many-channel stiffness, superlinear/tamed drift, and blow-up/domain stability). Independently verified counts both exact/reference moment passes and stability (no blow-up) passes.

## Independent-verification discrepancies

### B07 sde_hhw_call_zerocorr — UNDERCLAIM

Pipeline stopped at **3/10**, but the independent check **passes** (funcErr[price_S75:8.9e-04 price_S100:8.8e-04 price_S125:8.7e-04]). The solution was actually good enough; the pipeline under-scored or ran out of iterations.

## Full results

| ID | Slug | Type | Tier | Pipe | Best plan | Iters | Independent metric | Verdict | Time |
|---|---|---|---|---|---|---|---|---|---|
| B07 | sde_hhw_call_zerocorr | SDE | 3 | 3 | 1-euler-maruyama-full-truncation | 4 | funcErr[price_S75:8.9e-04 price_S100:8.8e-04 price_S125:8.7e-04] | UNDERCLAIM | 58.9m |

## Tier-by-tier detail

### PDE

### SDE

**Tier 3**

- `B07 sde_hhw_call_zerocorr` (heston_hull_white) — **UNDERCLAIM**. The same three factors and the same ten-year horizon, with the rate decorrelated. Removing two correlations removes none of the difficulty that matters -- the Feller ratio is still 0.04 and the discount factor is still path-dependent -- but it does admit an independent check, which is why it is here beside its correlated sibling. Independent: funcErr[price_S75:8.9e-04 price_S100:8.8e-04 price_S125:8.7e-04].

## Scheme coverage and efficiency

- Mean solver-evaluator iterations per problem (summed over plans): 4.0.
- Mean wall time per completed run: 58.9m; max 58.9m.

## Failures and errors

No failures, overclaims, blow-ups, run errors, or unverifiable results.

## Kernel provenance vs the independent verdict

What the pipeline's own kernel says its score rests on (`provenance`, from `verifylib/kernel/score.py`), against what the harness's independent check found. The kernel never sees the ground truth these verdicts are computed from, so agreement here is evidence and disagreement is a finding.

| kernel provenance | UNDERCLAIM |
|---|---|
| surrogate | 1 |

| problem | kernel score | provenance | tier | independent verdict |
|---|---|---|---|---|
| sde_hhw_call_zerocorr | 3 | surrogate | A_prime | UNDERCLAIM |

## Every plan, graded

The harness grades every plan directory with a `solver.py`, not only the winner. **`wrong_plan_won`: 0/1** -- the conductor's winner was less accurate than another evaluated plan by more than the reference's own error. This is the number the Path-B ranking rules exist to drive to zero, and it is invisible in a winner-only record.

| kernel provenance (per plan) | UNDERCLAIM | FAIL |
|---|---|---|
| surrogate | 1 | 1 |

| problem | plan (* winner) | kernel score | provenance | harness rel err | verdict |
|---|---|---|---|---|---|
| sde_hhw_call_zerocorr | 1-euler-maruyama-full-truncation * | 3.0 | surrogate | - | UNDERCLAIM |
| sde_hhw_call_zerocorr | 2-qe-log-euler | 3.0 | surrogate | - | FAIL |

## Methodology and reproducibility

- **Pipeline pass** = the conductor's best plan reached the terminal self-score of 10/10 (PDE: rel L2 < 1%; SDE: variance rel err < 10% and, unless the mean is near zero, mean rel err < 5%).
- **Independent verification** re-imports the best plan's `solver.py`, re-runs it, and compares the raw numerical output to ground truth authored and validated in `benchmark/problems.py` — never to the formulator's extracted solution. SDE moments use the fixed `(num_paths, dt, seed)` recorded in the manifest; PDE errors use the RMS-based relative L2 norm from `pde_manual.md`.
- **Ground-truth kinds:** *exact* (closed-form / machine-precision moments or analytic solution); *reference* (no elementary moment formula — the ground truth is a discretization-free Monte-Carlo of the known exact solution, compared within an SE-aware tolerance); *stability* (no ground truth — the independent check only confirms a correct scheme stays finite / in its domain, which is what a naive scheme gets wrong).
- **Problems with no closed form** (`closed_form: False`) run as their own suite (`--no-closed-form`) and are excluded by default, so the main benchmark's baseline denominators stay comparable. They are still independently checked, by one of four routes: a harness-owned high-resolution *reference field* (two independent method families agreeing far below the pass tolerance, built by `make_references.py` and recorded as the problem's `reference_error`, which widens the tolerance so the answer key's own error cannot fail a correct solver); *semi-analytic exact* truth from a quadrature, series or eigen-expansion; a scalar *functional* (`functional_truth`); and structural gates. See `benchmark/design/NO_CLOSED_FORM_CANDIDATES.md`.
- Ground truth for all 1 exact/reference problems is validated independently (SDE moments against moment-ODE integration, matrix-exponential / affine systems, and direct simulation; the Ginzburg-Landau reference anchored to the literature value E[X_1^2]=0.8114 at sigma=2; PDE solutions against their defining identities).
- Runs are **sequential**; each problem has a wall-clock timeout and its full transcript is saved under `benchmark/results/logs/`.
