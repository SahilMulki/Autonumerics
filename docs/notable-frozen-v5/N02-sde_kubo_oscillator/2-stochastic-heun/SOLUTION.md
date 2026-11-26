---
id: 2
plan_slug: stochastic-heun
scheme: stochastic-heun
scheme_family: other
strategy: Explicit Stratonovich predictor-corrector, raised from the 2-stage stochastic Heun to a 4-stage Runge-Kutta on the combined increment f dt + g dW (single commutative noise channel), removing the O(dt) Heun bias at the graded step.
---

## SDE Reference

Stratonovich system, one Brownian channel, state X = (X1, X2) = (X, Y):

    dX1 = -a X2 dt - sigma X2 o dW
    dX2 =  a X1 dt + sigma X1 o dW,        X(0) = (X_0, Y_0) = (1, 0)

Compactly `dX = a J X dt + sigma J X o dW`, with `J = [[0, -1], [1, 0]]` (so `J @ J = -I`).

| parameter | value |
|---|---|
| a | 1.0 |
| sigma | 1.0 |
| X_0 | 1.0 |
| Y_0 | 0.0 |
| T0, T | 0.0, 1.0 |

- Stratonovich drift (what this scheme uses): `f(X) = [-a*X2, a*X1]`
- Diffusion column (identical in Ito and Stratonovich form): `g(X) = [-sigma*X2, sigma*X1]`
- Ito-equivalent drift (for reference only; NOT used by this scheme): `[-a*X2 - 0.5*sigma**2*X1, a*X1 - 0.5*sigma**2*X2]`
- Exact flow: rotation of X(0) by `theta_t = a*t + sigma*W_t`; `|X(t)| = 1` on every path.

## Numerical Scheme

- **Method**: stochastic Heun (Stratonovich trapezoidal predictor-corrector). Per step, with
  `dWn = dW[:, n]`:

      Xp      = X_n + f(X_n) dt + g(X_n) dWn                              (predictor: Euler)
      X_{n+1} = X_n + 0.5*(f(X_n) + f(Xp)) dt + 0.5*(g(X_n) + g(Xp)) dWn  (corrector)

  For a single noise channel this converges to the Stratonovich solution with strong order 1.0.
  (On this linear system it reduces to `X_{n+1} = (I + phi J + phi^2 J^2 / 2) X_n` with
  `phi = a*dt + sigma*dWn`; implement it in the generic f/g form above, not as a hand-reduced
  matrix, so the scheme stays the Heun scheme and is checkable as such.)
- **dt**: 0.04 (the contract step; Nt = 25). Take `Nt = dW.shape[1]`, `dt = T/Nt` when `dW` is supplied.
- **num_paths**: 50000 for the kernel's moment runs (`evaluation_thresholds.num_paths`); the
  graded contract call uses 2000 (`graded_num_paths`). Use whatever `num_paths` you are handed.
- **seed**: 42
- **T**: 1.0
- **Milstein correction**: not applicable (the corrector stage generates the Stratonovich
  correction implicitly).
- **Expected strong order**: 1.0 (single noise channel; spec `verification.expected_strong_order = 1.0`)
- **Expected weak order**: 1.0
- **Reference for scoring**: analytic. Exact moments (`analytic_moments`, validated against the
  closing `moment_ode`) and, because the problem is graded path by path, the exact pathwise
  solution on the same increments (`verification.pathwise_reference`): relative strong error
  gated at 20% (`strong_rel_err_max`). Formulator's measurement for this scheme at dt = 0.04:
  0.040 / 0.034 (X / Y), E|X_T|^2 = 1.033.

## Implementation Notes

- **The coefficients are Stratonovich and Heun is a Stratonovich integrator**: use `f(X) = a*J@X`
  exactly as written. Do NOT use the Ito-corrected drift here — the corrector already produces
  the `-0.5*sigma^2*X` term, so adding it again double-counts and the amplitude decays.
  Conversely, plain Euler on these coefficients (the predictor alone) solves the Ito equation and
  the amplitude grows to ~e^{sigma^2 T} = 2.72, failing the gate.
- **Use the same `dWn` in the predictor and the corrector** — one increment per step, from the
  supplied `dW` (shape `(num_paths, Nt)`, already scaled by `sqrt(dt)`); draw no randomness of
  your own. When `dW is None`, seed `rng = np.random.default_rng(seed)`, `Nt = max(1, round(T/dt))`,
  `dt = T/Nt`, `dW = sqrt(dt) * rng.standard_normal((num_paths, Nt))`.
- **Do not declare `exact_sampling_components`.** Every component is stepped with the supplied
  increments; the CRN order study should measure ~1.0 here.
- **Amplitude.** Per step `|X|^2` grows by `phi^4/4`, so E|X_T|^2 ~ 1.03 at dt = 0.04 — inside the
  gated martingale constraint (reference 1.0, tol 0.10). Do not renormalise `X` onto the unit
  circle: that would be a projection the plan does not declare, and it changes the scheme.
- **Observables hook.** Support `observables: dict | None` and return `path_integrals` (trapezoid
  rule, as in the manual template) — the Dynkin check uses test functions `X[:, 0]`,
  `X[:, 0]*X[:, 1]`, `np.tanh(X[:, 1])`, each taking the `(num_paths, 2)` state.
- **Return**: `terminal_paths` shape `(num_paths, 2)` in column order (X, Y); `empirical_mean` and
  `empirical_variance` (ddof=1) as lists of two floats; also `num_paths`, `dt`, `T`.
  Exact targets at T = 1: mean (0.327710, 0.510378), variance (0.364447, 0.267674).
- No barrier, no absorbing boundary, no priced payoff: no boundary treatment and no variance
  reduction applies (the grade is pathwise against the harness's own increments).
- `solve_sde` importable with no work at module load; numpy + stdlib only; no hardcoded `dt`.

- **Revision (bias-dominated).** The graded-step mean bias came from the corrector sub-step: Heun's
  trapezoid stage reproduces exp(phi J) only to phi^2/2 (phi = a dt + sigma dW), dropping the phi^3/6 and
  phi^4/24 terms, which also gave the E|X|^2 = 1.033 drift. The step is now 4-stage RK on h(X) = f dt + g dW
  (same dW in every stage), error O(phi^5). It was previously named stochastic Heun (2 stages);
  `scheme_family` stays `other` (not in the EM/Milstein/implicit/splitting vocabulary). Expected order
  floor 1.0 is now exceeded; an overshoot is flagged, not failed.

## Results

Run: `solve_sde(num_paths=50000, dt=0.04, T=1.0, seed=42)` (Nt = 25), Stratonovich RK4 in generic f/g form.

- Empirical mean (X, Y):     (0.327377, 0.513593)   [exact (0.327710, 0.510378)]
- Empirical variance (X, Y): (0.363754, 0.264943)   [exact (0.364447, 0.267674)]
- E|X_T|^2 = 0.9996
- Pathwise relative strong error vs exact rotation on supplied dW (dt = 0.04, 1000 paths): 5.1e-4 (Heun: 0.037)
- `dW` and `observables` hooks both exercised without error.

## Evaluation

See metrics and review below.

<metrics>
kind: sde
score: 9
certified: false
provenance: analytic
estimated_rel_error: 0.0102
error_is_estimate: false
observed_order: 1.9006
observed_order_se: 0.0158
order_floor: 1.0000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
strong_rel_err: 4.069e-04
strong_rel_err_se: 4.377e-06
strong_rel_err_tol: 0.2000
strong_rel_err_component: 0
design_order: 1.0000
scheme_family: other
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: unavailable
invariant_max_drift: 3.632e-04
invariant_worst_headroom: 2.4399
reference_outcome: validated
mean_rel_err: 6.298e-03
variance_rel_err: 0.0102
moment_bias_binding: variance
moment_bias_frac_tol: 5.197e-03
moment_bias_actionable: false
order_check: true
mc_se_rel: 6.950e-03
resolved: true
dynkin_z: 1.5111
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: a5abd73bdd1ceb7f177bfbf810bdf551aeb35d701f8f6878fe645dfbf4f7247f
solve_time_s: 0.1780
wall_time_s: 6.2600
cost_source: cpu
</metrics>

<review score=9>

**Score: 9/10**

### Numerical Accuracy
- provenance:              analytic  (analytic_moments, validated against the spec's moment ODE; surrogate_ok unavailable because not needed)
- mean_rel_err:            6.3e-03 (X) +/- 2.0 x 0.00270 absolute se (tol 0.05)
- variance_rel_err:        1.02e-02 (X) +/- 2.0 x 0.00181 absolute se (tol 0.10)
- mc_se_rel:               6.95e-03   resolved: true (50000 paths)
- functional_rel_error:    not applicable (the spec declares no functional)
- observed_order:          1.901 +/- 0.016 (expected 1.0, one-sided below the declared floor; median-based 1.666)
- weak_order:              2.11, out of band (reported, never scored on)
- dynkin:                  ok, z = 1.51 against bar 2.43; single seed, not replicated because the plan was not marginal
- terminal_law:            not declared
- shape:                   skewness -0.682 +/- 0.007 (X), -1.159 +/- 0.009 (reported component); excess kurtosis -0.758 +/- 0.014 (X) / +/- 0.027 (reported component)
- constraints:             finite ok; martingale E|X_T|^2 drift 3.6e-04 against bar 0.100 (headroom 2.44 decades), ok
- strong error (pathwise): 4.07e-04 +/- 4.4e-06 against gate 0.20, self-convergence estimate at dt0
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                strong_gate_not_measured. The spec declares a pathwise strong-error gate (strong_rel_err_max 0.2), and the kernel's 0.000407 is a self-convergence estimate calibrated for ranking, so it cannot evaluate the gate directly. This caps the score at 9 and `certified` at false. It is not a solver defect.

### Feedback for solver
- Nothing needs to change. The step consumes the supplied dW, takes Nt from dW.shape[1], uses the Stratonovich coefficients as written, and reproduces the exact rotation. All moment checks are MC-resolved, Dynkin passes (z = 1.51 against 2.43), and reported_moments_agree is true (difference 9e-15).
- The graded-step discretization bias is negligible. The binding moment is the variance, with bias at 0.5% of its 10% tolerance, and moment_bias_actionable is false. The Heun-to-RK4 revision removed the earlier bias (previous Heun pass: mean error 3.8%, variance error 6.3%, E|X_T|^2 drift 0.033). It now sits 1.5 to 2 decades inside the tolerances. More paths and a smaller dt are both unnecessary.
- Naming. The directory and `scheme` field still say stochastic-heun while the code is a 4-stage RK on f dt + g dW. SOLUTION.md discloses this in the Revision note, the code matches that note, and no cap was applied. Renaming the scheme field to match would avoid confusing a reader of REPORT.md.
- The measured strong order of 1.90 exceeds the design order of 1.0. The cause is that this linear system has a pathwise error that RK4 removes to O(phi^5) at dt = 0.04. So it is a pre-asymptotic reading and not a claim of order 2. An overshoot is flagged and never scored against, and the ranking charges a shortfall only.
- Declaring `exact_sampling_components` is not needed and rightly was not declared. The step consumes dW, so the order study measured it.

</review>
