---
id: 2
plan_slug: stochastic-heun
scheme: stochastic-heun (Stratonovich predictor-corrector)
scheme_family: other
strategy: Integrate the Stratonovich form with the explicit stochastic Heun predictor-corrector, which converges to the Stratonovich solution at strong order 1.0 under commutative noise.
---

## SDE Reference

**Stratonovich form (as stated in problem.md):**

    dX1 = -a X2 dt - sigma X2 o dW
    dX2 =  a X1 dt + sigma X1 o dW,        X(0) = (X_0, Y_0) = (1, 0)

Matrix form: `dX = a J X dt + sigma J X o dW`, with `J = [[0, -1], [1, 0]]`. One Brownian motion
(single noise channel, hence commutative noise, no Levy area).

**Equivalent Ito form** (Stratonovich-to-Ito drift correction `0.5 (dg/dX) g = -0.5 sigma^2 X`):

    dX1 = (-a X2 - 0.5 sigma^2 X1) dt - sigma X2 dW
    dX2 = ( a X1 - 0.5 sigma^2 X2) dt + sigma X1 dW

- Stratonovich drift (array form): component 0 `-a*X[:, 1]`; component 1 `a*X[:, 0]`
- Ito drift (array form): component 0 `-a*X[:, 1] - 0.5*sigma**2*X[:, 0]`; component 1 `a*X[:, 0] - 0.5*sigma**2*X[:, 1]`
- Diffusion (single column): component 0 `-sigma*X[:, 1]`; component 1 `sigma*X[:, 0]`
- `dg/dX = sigma*J`, `(dg/dX) g = -sigma**2 * X`

**Parameters:** `a = 1.0`, `sigma = 1.0`, `X_0 = 1.0`, `Y_0 = 0.0`, `T = 1.0`.

**Exact pathwise solution (for reference only; must NOT be used to produce the answer):**
`X(t) = R(a t + sigma W(t)) X(0)`, a rotation, so `|X(t)| = |X(0)| = 1` on every path.

**Exact moments at T = 1** (Tier A reference):

| | mean | variance |
|---|---|---|
| X (component 0) | 0.32770991402245986 | 0.36444653725532805 |
| Y (component 1) | 0.5103779515445728 | 0.26767402157322967 |

Expressions: `mean_X = exp(-0.5 sigma^2 t)(X_0 cos(a t) - Y_0 sin(a t))`,
`mean_Y = exp(-0.5 sigma^2 t)(X_0 sin(a t) + Y_0 cos(a t))`,
`var_X = 0.5|X_0|^2 + 0.5 exp(-2 sigma^2 t)((X_0^2 - Y_0^2) cos(2 a t) - 2 X_0 Y_0 sin(2 a t)) - mean_X^2`,
`var_Y = 0.5|X_0|^2 - 0.5 exp(-2 sigma^2 t)((X_0^2 - Y_0^2) cos(2 a t) - 2 X_0 Y_0 sin(2 a t)) - mean_Y^2`.

**Solver contract (problem.md):** `solve_sde(num_paths=2000, dt=0.04, T=1, seed=42, dW=...)`. `dW`
is always supplied, shape `(num_paths, Nt)` = `(2000, 25)`, already scaled by `sqrt(dt)`; take
`Nt = dW.shape[1]`, `dt = T / Nt`. When `dW is None`: `Nt = max(1, round(T / dt))`, `dt = T / Nt`,
draw `dW = sqrt(dt) * np.random.default_rng(seed).standard_normal((num_paths, Nt))`. Return
`terminal_paths` of shape `(num_paths, 2)` (columns X, Y), `empirical_mean` (list of 2 floats),
`empirical_variance` (list of 2 floats, `ddof=1`). numpy + stdlib only; no work at import time.
Also accept an optional `observables: dict | None = None` keyword (accumulate the trapezoid path
integral of each `phi(X)` and return it under `path_integrals`) so the evaluator's Dynkin check can
run; it must default to `None` so the problem.md call is unaffected.

**Grading (problem.md):** path by path. Relative strong error
`sqrt(mean((X_num - X_exact)^2)) / sqrt(mean(X_exact^2)) < 20%` against the exact solution on the
same increments; amplitude conservation is checked (spec constraint: path-mean of
`| |X(T)| - 1 |` < 0.25); a solver with no discretization error (identical terminal states at
`dt`, `dt/2`, `dt/4`) is **excluded**. Spec moment thresholds: mean rel. err < 5%, variance rel. err
< 10%.

## Numerical Scheme

- **Method**: Stochastic Heun (explicit trapezoidal predictor-corrector) on the **Stratonovich** coefficients
- **dt**: 0.04 (use the `dt` given; with `dW` supplied, `dt = T / dW.shape[1]`)
- **num_paths**: 2000
- **seed**: 42
- **T**: 1.0
- **Update** (with `f_S(X) = a J X` the Stratonovich drift and `g(X) = sigma J X`):
  ```
  Xp      = X_n + f_S(X_n) dt + g(X_n) dW_n                       # predictor (Euler)
  X_{n+1} = X_n + 0.5 (f_S(X_n) + f_S(Xp)) dt
                + 0.5 (g(X_n) + g(Xp)) dW_n                       # corrector (trapezoid)
  ```
  Component-wise: `f_S(X) = (-a X2, a X1)`, `g(X) = (-sigma X2, sigma X1)`.
- **Milstein correction**: not applicable (Stratonovich scheme; the trapezoid in the diffusion
  generates the Stratonovich correction automatically)
- **Expected strong order**: 1.0 (single commutative noise channel; spec `strong_order_note`)
- **Expected weak order**: 1.0
- **Reference for scoring**: analytic moments (Tier A), with `verification.moment_ode` as the
  closed moment hierarchy; the harness additionally grades against the exact pathwise solution.

## Implementation Notes

- Use the **Stratonovich** drift `a J X` (no `-0.5 sigma^2 X` term). The Heun trapezoid is what
  makes the scheme Stratonovich-consistent; adding the Ito correction on top would double-count it.
- Consume the supplied `dW[:, n]` step by step; draw no randomness when `dW` is given. The same
  `dW_n` is used in predictor and corrector. All components are stepped with the supplied
  increments, so `exact_sampling_components` is **not** declared.
- Do not evaluate the closed-form rotation (from per-step or summed increments): it carries no
  discretization error and the harness excludes such a run.
- Amplitude is not conserved exactly (Heun is not norm-preserving); spec-measured path-mean
  deviation 0.016 at dt = 0.04, well under the 0.25 constraint. `amplitude_rms` is an ungated
  diagnostic.
- Expected accuracy (spec, 50 000 paths): relative strong error 0.037 / 0.018 / 0.009 at
  dt = 0.04 / 0.02 / 0.01 against the 20% gate.
- `scheme_family: other` selects no family-specific floor, so the order study is one-sided against
  `expected_strong_order: 0.5`; a measured ~1.0 passes (possibly with a non-fatal
  `strong_order_above_band` flag).
- Vectorise over paths: state `X` of shape `(num_paths, 2)`, update both columns from the old values
  (do not overwrite X1 before computing X2).

## Results

Run: `solve_sde(num_paths=2000, dt=0.04, T=1.0, seed=42)` (dW=None path, 25 steps).

| | empirical mean | exact mean | empirical variance | exact variance |
|---|---|---|---|---|
| X (comp 0) | 0.315646 | 0.327710 (rel err 3.7%) | 0.386155 | 0.364447 (rel err 6.0%) |
| Y (comp 1) | 0.505256 | 0.510378 (rel err 1.0%) | 0.292686 | 0.267674 (rel err 9.3%) |

- Path-mean amplitude deviation `| |X(T)| - 1 |` = 0.0165 (constraint < 0.25).
- Explicit `dW` hook (1000 paths, 25 steps): `dt = 0.04`; relative strong error vs the exact rotation on the same increments = 0.0370 (matches the spec's 0.037 at dt = 0.04).
- `observables` hook returns `path_integrals` of shape `(num_paths,)`.
- Note: the Y variance error (9.3%) sits close to the 10% gate; at 2000 paths this is dominated by Monte Carlo sampling noise (the variance estimator's relative s.e. is about 3-4%), not by scheme bias.
- **Unresolved-measurement retry (conductor cycle): no lever on the sampling error; scheme and `solver.py` left untouched.** The graded call is `solve_sde(num_paths=2000, dt=0.04, T=1, seed=42, dW=...)` with the harness supplying `dW` of shape `(2000, 25)` (ledger R12/R14; spec `num_paths = graded_num_paths = 2000`). When `dW` is given the number of paths is `dW.shape[0]`, and `terminal_paths` must be the path-by-path image of those increments, so the solver cannot raise `num_paths`. Variance reduction is also unavailable: antithetic or control-variate pairing would return terminal states not driven by the supplied increments and break the pathwise grading, and the plan describes no such device. Measured at the graded 2000 paths (dW=None, seed 42): variance standard error 0.00954 (comp 0, 2.5% of the exact variance) and 0.00994 (comp 1, 3.4%), consistent with the evaluator's `mc_se_rel` 0.034. For reference only, the same estimator would give comp 0 / comp 1 relative variance SE of 0.83% / 1.13% at 20000 paths and 0.26% / 0.36% at 200000 paths, so a larger graded path count in the spec (a formulator/conductor decision) is the only route to `resolved: true`. No code or scheme change was made; `solver.py` runs cleanly (mean [0.315646, 0.505256], variance [0.386155, 0.292686], dW and observables hooks both confirmed).

## Evaluation

<metrics>
kind: sde
score: 6
certified: false
provenance: analytic
estimated_rel_error: 0.0934
error_is_estimate: false
observed_order: 1.0012
observed_order_se: 0.0326
order_floor: 0.5000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 0.5000
scheme_family: other
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: unavailable
invariant_max_drift: 0.0000
invariant_worst_headroom: 6.0000
reference_outcome: validated
mean_rel_err: 0.0368
variance_rel_err: 0.0934
order_check: true
mc_se_rel: 0.0340
resolved: false
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 3c8b30ae82a378a16f66ecd180d61f718a5259b1f165a13c149cfa45c43ffef8
solve_time_s: 0.0940
wall_time_s: 1.2500
cost_source: cpu
</metrics>

<review score=6>

**Score: 6/10**

This is an MC-inconclusive 6, the same state as the previous pass. It is not a solver bug, and nothing in `solver.py` needs to change. The kernel run is deterministic at the spec's seed: every number below reproduces the previous pass to the printed digit, and `solver_sha256` is unchanged (`3c8b30ae...`).

### Numerical Accuracy
- provenance:              analytic  (reference route: analytic_moments, validated against the spec's own `moment_ode`; both components, mean and variance)
- mean_rel_err:            comp 0: 0.0368 (abs se 0.0139, 2 x se = 0.0278 abs, about 8.5% rel; tol 0.05, **unresolved**); comp 1: 0.0100 (abs se 0.0121, 2 x se about 4.7% rel; tol 0.05, pass). `mean_ok` is therefore `unavailable`, not a pass
- variance_rel_err:        comp 0: 0.0596 +/- 2 x 0.0095 abs (se_rel 2.6%, interval 5.2% wide); comp 1: 0.0934 +/- 2 x 0.0099 abs (se_rel 3.7%, interval 7.4% wide). tol 0.10. `variance_ok: true` is a point-estimate verdict; the comp-1 interval, roughly 1.9% to 16.8%, straddles the 10% line
- mc_se_rel:               0.0340   resolved: **false**. The `resolved` bar is 0.05 (half the variance tolerance); the 2 x se interval widths are 0.0523 (comp 0) and 0.0743 (comp 1), at 2000 paths
- functional_rel_error:    not applicable. The spec declares no gated functional, so the kernel wrote no `functional_*` fields and there is no bias/sampling split or `functional_limited_by` to report. The reading of the variance moment against its tolerance is the only accuracy claim here
- observed_order:          1.0012 +/- 0.0326 (median-based 0.939), expected_strong_order 0.5 (spec), one-sided below the declared floor. Per component: 1.005 (comp 0), 1.001 (comp 1), flagged `strong_order_above_band` and never scored on. This is Heun's real strong order on a single commutative channel and matches the spec's measured 1.0
- weak_order:              1.11 (comp 0), 1.21 (comp 1; median of x and x^2, tanh at the noise floor and dropped), against expected 1.0, within band. Richardson shift on the mean is 0.73% (comp 0) and 0.16% (comp 1), stable, so the dt = 0.04 bias is small beside the sampling noise
- dynkin:                  `not_run` (a skip, not a pass and not a failure). See the note below
- terminal_law:            not declared in the spec (wrapped-normal law, not an implemented family)
- shape:                   comp 0: skewness -0.676 +/- 0.037, excess kurtosis -0.778 +/- 0.068; comp 1: skewness -1.139 +/- 0.044, excess kurtosis +0.309 +/- 0.133. Descriptive only
- constraints:             `finite` pass (gate); `amplitude_conservation` pass (gate, headroom 6.0 decades); `amplitude_rms` non-gate, pass. `neg_fraction` not applicable. No gate violation
- reported_moments_agree:  true (worst relative difference 1.3e-15, so `ddof=1` is correct)
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. MC-inconclusive: the error bars are too wide to decide. Raise num_paths; this is not a solver bug. The Tier-A certification ceiling would have allowed 10 (`certified: false`, headroom_fraction +0.18)

### What the 6 does and does not say
- `resolved: false` is a claim about the error bars at 2000 paths. It is not a claim about the scheme.
- **The retry's conclusion is correct.** The solver reported no lever on the sampling error, and the source agrees. The graded call is `solve_sde(num_paths=2000, dt=0.04, T=1, seed=42, dW=...)` with the harness supplying `dW` of shape `(2000, 25)` (ledger R12 and R14; spec `num_paths = graded_num_paths = 2000`). With `dW` given, the path count is `dW.shape[0]`, and the terminal states must be the path-by-path image of those increments, so neither a larger `num_paths` nor an antithetic or control-variate device can be applied without breaking the pathwise grading. Leaving the scheme alone was the right call.
- **Where the width comes from.** The variance estimator's standard error is 2.6% to 3.7% of the reference at 2000 paths. That is the sampling variance of a sample variance on the wrapped-normal law, not scheme bias. A correct solver, including one that integrates the equation exactly, reads `resolved: false` on this spec. The `resolved` bar of 0.05 needs the interval width (ci_mult x se_rel) below 5%, i.e. se_rel below 2.5%, which for the binding component (comp 1, se_rel 3.7%) needs about (3.7/2.5)^2 = 2.2 times the paths, roughly 4500 or more instead of 2000.
- **What would lift it.** A larger `graded_num_paths` in the spec (formulator and conductor decision), or a rubric that does not demand 2 x se below half the tolerance at 2000 paths. Neither is the solver's to change. I record this as a structural limit of the spec, not a defect of this plan.
- **Dynkin note.** `dynkin_ok` is `not_run`, and the reason string the kernel prints says "a declared constraint gate failed, so the score is capped". The recorded evidence contradicts it: `constraints_ok: true`, `gate_failed: []`, `gates_clean: true`, and `config.run_dynkin` is the kernel default `false`. The score is transcribed as emitted and is bound by the MC-inconclusive row, not by Dynkin. The reason string looks wrong for this case and is worth a look by whoever maintains the kernel. Even with enough paths, a 10 here needs a Dynkin verdict of `True`; without one the ceiling is 9.

### Feedback for solver
- None required on the code. `solver.py` is consistent with `SOLUTION.md`: stochastic Heun on the Stratonovich coefficients, consuming the supplied `dW` step by step (accepting the `(M, Nt, 1)` shape), taking `Nt = dW.shape[1]` and recomputing `dt = T / Nt`. It has no state clip and no closed-form shortcut. No `agent_cap` was needed.
- Do not change the scheme in response to the 6. The comp-1 variance error of 9.3% against a 10% tolerance carries a standard error of 3.7%, so it is a sampling excursion and not Heun's bias.
- Keep `scheme_family: other`. The order study reads 1.00 and no `undeclared_exact_sampling` finding is present, because every component is stepped with the supplied increments. Do not declare `exact_sampling_components`: it would hide a real defect and buy nothing.
- The only route to a higher score is outside this plan: raise `graded_num_paths` in the spec.

</review>
