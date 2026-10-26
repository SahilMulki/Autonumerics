# Autonumerics Benchmark Report

_Generated 2026-10-26 18:38_

Pipeline invocation: `claude --plugin-dir . -p "/conductor workspace/{slug}/problem.md"`

Active time, most recent run invocation: **101.0m**

## Executive summary

- **Problems:** 2 (0 PDE, 2 SDE); 2 with exact/reference ground truth.
- **Runs completed:** 2/2.
- **Pipeline self-reported pass (10/10):** 0/2 (0%).
- **Independently confirmed pass:** 0/2 (0% of independently-checkable problems).
- **OVERCLAIMS / BLOW-UPS:** none — every success the pipeline reported was independently confirmed.
- **Wrong plan won:** 1/2 — the winner was less accurate than another evaluated plan by more than the reference error (see *Every plan, graded*).

### Verdict distribution

| Verdict | Count | Share |
|---|---|---|
| UNDERCLAIM | 2 | 100% |

### Independently-verified pass rate by type and tier

| Type | Tier | N | Pipeline pass | Independently verified |
|---|---|---|---|---|
| PDE | Tier 1 | 0 | 0/0 | n/a |
| PDE | Tier 2 | 0 | 0/0 | n/a |
| PDE | Tier 3 | 0 | 0/0 | n/a |
| SDE | Tier 1 | 0 | 0/0 | n/a |
| SDE | Tier 2 | 0 | 0/0 | n/a |
| SDE | Tier 3 | 2 | 0/2 | 0/2 |

> Tier 1 is textbook material the reference manuals cover; Tier 2 needs care (boundary layers, indefinite operators, multi-D, guards); Tier 3 pushes beyond the manuals (shocks, fractional operators, stiffness, heavy-tailed sampling, chaos, and the hard-SDE additions: Feller-violated log-Heston, many-channel stiffness, superlinear/tamed drift, and blow-up/domain stability). Independently verified counts both exact/reference moment passes and stability (no blow-up) passes.

## Independent-verification discrepancies

### B09 sde_heston_dnt_hivar — UNDERCLAIM

Pipeline stopped at **9/10**, but the independent check **passes** (funcErr[no_touch_S75:5.0e-04 no_touch_S100:2.6e-04 no_touch_S125:1.8e-03]). The solution was actually good enough; the pipeline under-scored or ran out of iterations.

### B10 sde_sabr_cev_call_ii_lowvol — UNDERCLAIM

Pipeline stopped at **2/10**, but the independent check **passes** (funcErr[price_K1:1.8e-03 price_K2:2.6e-03 price_K3:4.3e-03]). The solution was actually good enough; the pipeline under-scored or ran out of iterations.

## Full results

| ID | Slug | Type | Tier | Pipe | Best plan | Iters | Independent metric | Verdict | Time |
|---|---|---|---|---|---|---|---|---|---|
| B09 | sde_heston_dnt_hivar | SDE | 3 | 9 | 2-euler-maruyama-cir-milstein | 4 | funcErr[no_touch_S75:5.0e-04 no_touch_S100:2.6e-04 no_touch_S125:1.8e-03] | UNDERCLAIM | 28.2m |
| B10 | sde_sabr_cev_call_ii_lowvol | SDE | 3 | 2 | 1-euler-maruyama-absorbing | 4 | funcErr[price_K1:1.8e-03 price_K2:2.6e-03 price_K3:4.3e-03] | UNDERCLAIM | 52.7m |

## Tier-by-tier detail

### PDE

### SDE

**Tier 3**

- `B09 sde_heston_dnt_hivar` (heston_barrier) — **UNDERCLAIM**. The same continuously monitored corridor started at a higher variance, so more paths reach the barriers and the per-step monitoring error a naive scheme makes is correspondingly larger. Independent: funcErr[no_touch_S75:5.0e-04 no_touch_S100:2.6e-04 no_touch_S125:1.8e-03].
- `B10 sde_sabr_cev_call_ii_lowvol` (sabr_cev) — **UNDERCLAIM**. The long-horizon SABR set started at a lower initial volatility, which moves every option value without changing the vol-of-vol, the correlation or the absorbing boundary that make it hard. Independent: funcErr[price_K1:1.8e-03 price_K2:2.6e-03 price_K3:4.3e-03].

## Scheme coverage and efficiency

- Mean solver-evaluator iterations per problem (summed over plans): 4.0.
- Mean wall time per completed run: 40.5m; max 52.7m.

## Failures and errors

No failures, overclaims, blow-ups, run errors, or unverifiable results.

## Kernel provenance vs the independent verdict

What the pipeline's own kernel says its score rests on (`provenance`, from `verifylib/kernel/score.py`), against what the harness's independent check found. The kernel never sees the ground truth these verdicts are computed from, so agreement here is evidence and disagreement is a finding.

| kernel provenance | UNDERCLAIM |
|---|---|
| analytic | 1 |
| none | 1 |

| problem | kernel score | provenance | tier | independent verdict |
|---|---|---|---|---|
| sde_heston_dnt_hivar | 9 | analytic | A | UNDERCLAIM |
| sde_sabr_cev_call_ii_lowvol | 2 | none | C | UNDERCLAIM |

## Every plan, graded

The harness grades every plan directory with a `solver.py`, not only the winner. **`wrong_plan_won`: 1/2** -- the conductor's winner was less accurate than another evaluated plan by more than the reference's own error. This is the number the Path-B ranking rules exist to drive to zero, and it is invisible in a winner-only record.

- `sde_sabr_cev_call_ii_lowvol`: winner `1-euler-maruyama-absorbing` at 4.250e-01, best `2-qe-martingale-absorbing` at 4.118e-03.

| kernel provenance (per plan) | UNDERCLAIM |
|---|---|
| analytic | 2 |
| none | 2 |

| problem | plan (* winner) | kernel score | provenance | harness rel err | verdict |
|---|---|---|---|---|---|
| sde_heston_dnt_hivar | 1-euler-maruyama | 9.0 | analytic | - | UNDERCLAIM |
| sde_heston_dnt_hivar | 2-euler-maruyama-cir-milstein * | 9.0 | analytic | - | UNDERCLAIM |
| sde_sabr_cev_call_ii_lowvol | 1-euler-maruyama-absorbing * | 2.0 | none | - | UNDERCLAIM |
| sde_sabr_cev_call_ii_lowvol | 2-qe-martingale-absorbing | 2.0 | none | - | UNDERCLAIM |

## Methodology and reproducibility

- **Pipeline pass** = the conductor's best plan reached the terminal self-score of 10/10 (PDE: rel L2 < 1%; SDE: variance rel err < 10% and, unless the mean is near zero, mean rel err < 5%).
- **Independent verification** re-imports the best plan's `solver.py`, re-runs it, and compares the raw numerical output to ground truth authored and validated in `benchmark/problems.py` — never to the formulator's extracted solution. SDE moments use the fixed `(num_paths, dt, seed)` recorded in the manifest; PDE errors use the RMS-based relative L2 norm from `pde_manual.md`.
- **Ground-truth kinds:** *exact* (closed-form / machine-precision moments or analytic solution); *reference* (no elementary moment formula — the ground truth is a discretization-free Monte-Carlo of the known exact solution, compared within an SE-aware tolerance); *stability* (no ground truth — the independent check only confirms a correct scheme stays finite / in its domain, which is what a naive scheme gets wrong).
- **Problems with no closed form** (`closed_form: False`) run as their own suite (`--no-closed-form`) and are excluded by default, so the main benchmark's baseline denominators stay comparable. They are still independently checked, by one of four routes: a harness-owned high-resolution *reference field* (two independent method families agreeing far below the pass tolerance, built by `make_references.py` and recorded as the problem's `reference_error`, which widens the tolerance so the answer key's own error cannot fail a correct solver); *semi-analytic exact* truth from a quadrature, series or eigen-expansion; a scalar *functional* (`functional_truth`); and structural gates. See `benchmark/design/NO_CLOSED_FORM_CANDIDATES.md`.
- Ground truth for all 2 exact/reference problems is validated independently (SDE moments against moment-ODE integration, matrix-exponential / affine systems, and direct simulation; the Ginzburg-Landau reference anchored to the literature value E[X_1^2]=0.8114 at sigma=2; PDE solutions against their defining identities).
- Runs are **sequential**; each problem has a wall-clock timeout and its full transcript is saved under `benchmark/results/logs/`.
