---
id: 1
plan_slug: cayley-midpoint
scheme: stochastic-implicit-midpoint (Cayley rotation)
scheme_family: implicit
strategy: Integrate the Stratonovich form with the stochastic implicit midpoint rule, which for this linear skew-symmetric system is the explicit Cayley map and conserves |X| exactly at every step.
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

- **Method**: Stochastic implicit midpoint rule on the **Stratonovich** coefficients (a Stratonovich-consistent scheme; no Ito correction is added)
- **dt**: 0.04 (use the `dt` given; with `dW` supplied, `dt = T / dW.shape[1]`)
- **num_paths**: 2000
- **seed**: 42
- **T**: 1.0
- **Update**: midpoint rule `X_{n+1} = X_n + J (X_n + X_{n+1})/2 * (a dt + sigma dW_n)`. With
  `th_n = (a dt + sigma dW_n) / 2` this solves in closed form as the Cayley transform
  `X_{n+1} = (I - th_n J)^{-1} (I + th_n J) X_n`, i.e. per path
  ```
  c = (1 - th^2) / (1 + th^2),   s = 2 th / (1 + th^2)
  X1_new = c*X1 - s*X2
  X2_new = s*X1 + c*X2
  ```
  This is a rotation by `2 arctan(th_n)`, not by the exact angle `2 th_n`; the difference
  (`O(th^3)` per step) is the scheme's discretization error, and it is what keeps the plan a
  genuine discretised scheme that the harness's dt-ladder can see.
- **Milstein correction**: not applicable (Stratonovich midpoint scheme)
- **Expected strong order**: 1.0 (commutative single-channel noise; spec `strong_order_note`)
- **Expected weak order**: 1.0
- **Reference for scoring**: analytic moments (Tier A), with `verification.moment_ode` as the
  closed moment hierarchy; the harness additionally grades against the exact pathwise solution.

## Implementation Notes

- **Do not** rotate by the exact per-step angle `a dt + sigma dW_n` with `cos`/`sin` and do not
  evaluate the closed-form solution from summed increments. Both carry zero discretization error,
  so the terminal states match at `dt`, `dt/2` and `dt/4` to rounding and the harness **excludes**
  the run. The Cayley map above (rational in `th`, no trig) is the scheme.
- Use the Stratonovich drift `a J X` only. Adding the Ito correction `-0.5 sigma^2 X` here would
  double-count and shrink the amplitude.
- Consume the supplied `dW[:, n]` step by step (one channel, shape `(num_paths, Nt)`); draw no
  randomness when `dW` is given. All components are stepped with the supplied increments, so
  `exact_sampling_components` is **not** declared.
- Amplitude: `|X_n| = 1` to round-off on every path, so the amplitude constraint passes trivially.
  The spec deliberately omits `|X|^2` as a Dynkin test function for this reason.
- Expected accuracy (spec `strong_order_note`, 50 000 paths): relative strong error 0.016 / 0.008 /
  0.004 at dt = 0.04 / 0.02 / 0.01 against the 20% gate - the smallest error of the candidate
  schemes on every measured axis.
- `scheme_family: implicit` selects no family-specific floor in this spec, so the order study
  judges it one-sided against `expected_strong_order: 0.5`; a measured ~1.0 passes, and may be
  flagged `strong_order_above_band` as a non-fatal note.
- Moment check is subject to Monte Carlo error at 2000 paths (variance relative SE roughly 3%);
  the scheme bias itself is small.
- Return `empirical_mean` / `empirical_variance` as Python lists over the two columns.

## Results

Run: `solve_sde(num_paths=2000, dt=0.04, T=1.0, seed=42)` (Nt = 25, internal rng draw), Cayley/implicit-midpoint scheme.

| | empirical mean | exact mean | empirical variance | exact variance |
|---|---|---|---|---|
| X (comp 0) | 0.336287 | 0.327710 | 0.357387 | 0.364447 |
| Y (comp 1) | 0.503057 | 0.510378 | 0.276775 | 0.267674 |

- Amplitude: max over paths of `| |X(T)| - 1 |` = 1.6e-15 (conserved to round-off).
- Strong error against the exact pathwise rotation on shared increments (20 000 paths), relative: 0.0163 / 0.0081 / 0.0041 at Nt = 25 / 50 / 100, i.e. observed strong order about 1.0.
- `dW` hook (explicit (2000, 25) increments) and `observables` hook (trapezoid path integral, shape (2000,)) both confirmed working.

### Unresolved-measurement retry (resolved: false, mc_se_rel 0.0343)

No change to `solver.py`'s scheme, `dt`, or the 2000-path call. The graded path count is fixed from outside: `problem.md` and the ledger (R12, R14, `evaluation_thresholds.graded_num_paths = 2000`) require `solve_sde(num_paths=2000, dt=0.04, ...)` with harness-supplied `dW` of shape (2000, 25). When `dW` is given the solver must use exactly those increments and take no randomness of its own, so it has no lever on the sampling error. Antithetic pairs or a control variate cannot be formed from increments the solver does not draw. Raising `num_paths` is impossible without departing from the ledger.

Diagnostic scan (kept in `solver.py`'s `__main__`, own-draw runs, seed 42, kurtosis-form delta-method CI width = 2 x relative SE of the variance, bar 0.05):

| num_paths | CI width X | CI width Y | resolved |
|---|---|---|---|
| 2000 | 0.0512 | 0.0686 | no |
| 4000 | 0.0365 | 0.0480 | yes |
| 8000 | 0.0255 | 0.0344 | yes |
| 16000 | 0.0179 | 0.0245 | yes |

At the graded 2000 paths, variance relative SE is about 0.026 (X) and 0.034 (Y), matching the kernel's `mc_se_rel` 0.0343. The bar is met from about 4000 paths. The fix belongs to the spec (graded path count or `ci_mult`/bar), not the solver.

## Evaluation

<metrics>
kind: sde
score: 6
certified: false
provenance: analytic
estimated_rel_error: 0.0340
error_is_estimate: false
observed_order: 0.9587
observed_order_se: 0.0344
order_floor: 0.5000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 0.5000
scheme_family: implicit
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: unavailable
invariant_max_drift: 0.0000
invariant_worst_headroom: 6.0000
reference_outcome: validated
mean_rel_err: 0.0262
variance_rel_err: 0.0340
order_check: true
mc_se_rel: 0.0343
resolved: false
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 9aa89dcb5f133185777acd88007e4284325fbb242977f88c5f4dcb11933ce71d
solve_time_s: 0.0910
wall_time_s: 1.2300
cost_source: cpu
</metrics>

<review score=6>

**Score: 6/10**

Verdict: MC-inconclusive in moment space, not a solver bug. Every check that reached a verdict passed (variance within tolerance on both components, strong order 0.96, weak order 1.01, amplitude conserved to round-off, reported moments equal the kernel's to 1e-15). The score is bound by the rubric's `resolved` row. `certified` is false. This is the same outcome as the previous pass: the retry changed nothing in `solver.py`'s scheme, and nothing in the solver could have changed it.

### Numerical Accuracy
- provenance:              analytic  (reference route: analytic_moments; `reference_outcome: validated`, the closed form was checked against the spec's own moment ODE)
- mean_rel_err:            0.0262 +/- 2.0 x 0.01337 (component 0, tol 0.05, `unresolved`: CI wider than the tolerance band), 0.0143 +/- 2.0 x 0.01176 (component 1, tol 0.05, pass). `mean_ok` therefore reads `unavailable`, not false.
- variance_rel_err:        0.0194 +/- 2.0 x 0.00915 (component 0, tol 0.10, pass), 0.0340 +/- 2.0 x 0.00949 (component 1, tol 0.10, pass). `variance_ok: true`.
- mc_se_rel:               0.0343   resolved: false. Variance CI relative width 0.0502 on component 0 and 0.0709 on component 1 against a resolution bar of 0.05. Component 0 misses by a hair, component 1 by about 1.4x. Path count in this run: 2000.
- functional_rel_error:    not applicable. The spec declares no gated functional, so there is no bias/sampling split and `functional_limited_by` is not reported.
- observed_order:          0.959 +/- 0.034 (median-based 0.904; per component 0.994 and 0.959), expected_strong_order 0.5 from the spec, `order_band: one-sided`. `strong_order_above_band: true` is a non-fatal note and is not scored on. It agrees with the design strong order 1.0 of the midpoint rule on commutative single-channel noise. Richardson `stable: true`.
- weak_order:              1.011 (x: 1.022, tanh: 1.000; x2 at the noise floor and not used), in band.
- dynkin:                  not run (`dynkin_ok: not_run`). The kernel prints "a declared constraint gate failed", which does not match this run: `constraints_ok` is true and `gate_failed` is empty, and the config shows `run_dynkin: false` from the kernel default. Read it as skipped. It would still cost the plan the 10 row (9 at most), which needs a Dynkin verdict, even once `resolved` is true.
- terminal_law:            not declared (spec `terminal_law: null`; the wrapped-normal law on the circle is not an implemented law).
- shape:                   skewness -0.717 +/- 0.037 and excess kurtosis -0.688 +/- 0.073 (component 0); skewness -1.152 +/- 0.044 and excess kurtosis +0.354 +/- 0.135 (component 1). Diagnostic only.
- constraints:             `finite` pass; `amplitude_conservation` pass (drift 0.0); `amplitude_rms` pass (non-gate, 0.0). Worst headroom 6.0 decades. `gate_violation: false`.
- reported_moments_agree:  true (worst relative difference 1.2e-15, so `ddof=1` is right).
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. MC-inconclusive: the error bars are too wide to decide. The certification ceiling is Tier A (10) and is not binding; `gates_clean: true`.

### Feedback for solver
- Nothing in `solver.py` needs to change, and I agree with the retry's reading. The Cayley rotation `c = (1-th^2)/(1+th^2), s = 2 th/(1+th^2)` with `th = (a dt + sigma dW_n)/2` consumes `dW[:, n]` exactly, and every measurable check on it is clean.
- Do not raise `num_paths` in `solver.py` to chase the 6. The graded call is `solve_sde(num_paths=2000, dt=0.04, ...)` with a harness-supplied `dW` of shape (2000, 25) (ledger R12, R14), so the sample is fixed from outside. With `dW` supplied the solver draws no noise, so antithetic pairs or a control variate cannot be formed, and it has no lever on the sampling error. Your path-count scan (4000 paths clears the bar; X 0.0365, Y 0.0480) matches the kernel's measurement at 2000 (X 0.0502, Y 0.0709).
- This is MC-inconclusive, not a defect: variance errors 0.019 and 0.034 against 0.10 and a weak-order bias near 0.1% of the mean (Richardson shift 4.5e-4 on a mean of 0.504) mean the scheme's own bias is far inside tolerance. More paths would resolve it, and only a path count in the spec can supply them.
- Optional: `scheme_family: implicit` selects no family floor in this spec, so the order check is one-sided against 0.5 and a measured ~1.0 is flagged, not failed. Nothing to change.

### Formulator note (not a solver defect)
- `evaluation_thresholds.num_paths = 2000` with `ci_mult = 2` and a 0.05 resolution bar on the variance CI cannot be met on component Y by any correct scheme. The finite-sample relative standard error of the variance of a bounded rotated-angle law (excess kurtosis about +0.35 on Y, skewness about -1.15) is about 3.4%, so the CI width is about 7%. The rubric returns 6 for every correct plan on this spec. The exact reference was recoverable and was declared, so this is not a missing-surrogate defect. It is a mismatch between the declared sample size and the declared resolution bar, and it belongs to the formulator and conductor.

</review>
