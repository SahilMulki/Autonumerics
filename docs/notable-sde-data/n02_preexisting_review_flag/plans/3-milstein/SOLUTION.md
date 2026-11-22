---
id: 3
plan_slug: milstein
scheme: milstein (Ito form)
scheme_family: milstein
strategy: Convert to the Ito form by adding the drift correction -0.5 sigma^2 X and apply the Milstein scheme, which is well defined here because the single noise channel is commutative.
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

- **Method**: Milstein on the **Ito** form
- **dt**: 0.04 (use the `dt` given; with `dW` supplied, `dt = T / dW.shape[1]`)
- **num_paths**: 2000
- **seed**: 42
- **T**: 1.0
- **Update** (spec `milstein_correction`), all coefficients evaluated at `X_n`:
  ```
  X_{n+1} = X_n + (a J X_n - 0.5 sigma^2 X_n) dt
                + sigma J X_n dW_n
                + 0.5 * (-sigma^2 X_n) * (dW_n^2 - dt)
  ```
  Component-wise:
  ```
  X1_new = X1 + (-a*X2 - 0.5*sigma**2*X1)*dt - sigma*X2*dW - 0.5*sigma**2*X1*(dW**2 - dt)
  X2_new = X2 + ( a*X1 - 0.5*sigma**2*X2)*dt + sigma*X1*dW - 0.5*sigma**2*X2*(dW**2 - dt)
  ```
- **Milstein correction**: `0.5 (dg/dX) g (dW^2 - dt)` with `(dg/dX) g = sigma^2 J J X = -sigma^2 X`.
  Multi-D Milstein is admissible here (`milstein_eligible: true`) because there is a single noise
  channel, so no Levy area appears.
- **Expected strong order**: 1.0 (spec `expected_strong_order_milstein`)
- **Expected weak order**: 1.0
- **Reference for scoring**: analytic moments (Tier A), with `verification.moment_ode` as the
  closed moment hierarchy; the harness additionally grades against the exact pathwise solution.

## Implementation Notes

- The Ito drift correction `-0.5 sigma^2 X` is **mandatory**. Applying Euler/Milstein to the
  Stratonovich coefficients as if they were Ito solves a different SDE whose amplitude grows to
  `exp(sigma^2 T / 2) = 1.65` (spec: strong error ~0.69, amplitude deviation 0.65) and fails both
  the 20% strong-error gate and the amplitude constraint.
- Note the correction and the drift correction combine: `-0.5 sigma^2 X dt - 0.5 sigma^2 X (dW^2 - dt)
  = -0.5 sigma^2 X dW^2`. Either form is fine; a sign error in the Milstein term shows up as a
  measured strong order near 0.5 instead of 1.0.
- Evaluate every coefficient at `X_n` (keep both old columns before writing new ones).
- Consume the supplied `dW[:, n]` step by step; draw no randomness when `dW` is given. All
  components are stepped with the supplied increments, so `exact_sampling_components` is **not**
  declared. Do not evaluate the closed-form rotation; the harness excludes a zero-discretization-
  error run.
- Thinnest moment margin of the order-1 schemes: the spec measures Milstein's variance at +8.7%
  relative error at dt = 0.04 against the 10% tolerance (50 000 paths), and at 2000 paths the MC
  standard error on the variance is a few percent on top of that. The graded pathwise strong
  error is comfortable (spec: 0.063 / 0.032 / 0.016 at dt = 0.04 / 0.02 / 0.01 vs the 20% gate).
  The `dt` is fixed by problem.md, so do not refine it in the graded call.
- Amplitude: not conserved exactly; spec-measured path-mean deviation 0.044 at dt = 0.04, under the
  0.25 constraint.
- `scheme_family: milstein` makes the order study read `expected_strong_order_milstein: 1.0`; this
  plan is the one in the pool whose correction term the order study checks against a 1.0 floor.

## Results

Run: `solve_sde(num_paths=2000, dt=0.04, T=1.0, seed=42)` (Milstein on the Ito form, internal draw of dW).

| | empirical mean | exact mean | empirical variance | exact variance |
|---|---|---|---|---|
| X (comp 0) | 0.322129 | 0.327710 | 0.395571 | 0.364447 |
| Y (comp 1) | 0.515693 | 0.510378 | 0.307889 | 0.267674 |

- Relative errors: mean X 1.7%, mean Y 1.0%; variance X 8.5%, variance Y 15.0%.
- The Y variance is above the 10% tolerance at this seed and path count. The scheme itself is verified:
  with an explicit dW of shape (2000, 25) the relative strong error against the exact pathwise rotation is
  0.0643 (spec-measured 0.063), and the mean amplitude deviation is 0.0441 (spec 0.044). So the excess is the
  known +8.7% Milstein variance bias at dt = 0.04 plus Monte Carlo noise at 2000 paths, not a coding defect.
  `dt` is fixed by problem.md, so it was not refined.
- Hooks confirmed: explicit `dW` (dt = T / dW.shape[1] = 0.04) and `observables` (path integral shape (2000,)).

### Unresolved-measurement retry (num_paths)

The evaluator recorded `resolved: false`, `mc_se_rel: 0.0318`. `num_paths` is **not a lever here**, so it was
left at 2000 and the scheme, `dt` and `solver.py` are unchanged:

- `problem_spec.json` ledger R12 fixes the graded call at `solve_sde(num_paths=2000, dt=0.04, T=1, seed=42)`
  (`evaluation_thresholds.num_paths = graded_num_paths = 2000`, `seed = 42`), and R14 says `dW` is always
  supplied with shape `(2000, 25)`. The sampling error is therefore set by the harness, not by the solver. Raising
  `num_paths` in `solver.py` would have no effect on a call that passes `dW` (the solver must take its paths
  from `dW.shape[0]`), and would break R12 if forced.
- `dt = 0.04` is fixed by R13 and the Milstein correction is what the plan declares, so neither was touched.
- Measured at the graded 2000 paths (seed 42, internal draw): variance SE (fourth-moment estimator) is
  0.01067 abs for X (2.9% of the exact variance 0.3644) and 0.00979 abs for Y (3.7% of 0.2677); the
  evaluator's `mc_se_rel` = 0.0318 is consistent with this. A 2-SE interval on the Y variance error
  (15.0% observed) is roughly 7.7% to 22%, so it straddles the 10% bar. Resolving it to the bar would need roughly 4x
  fewer SE (about 30 000 to 50 000 paths), which the spec does not permit for the graded call.
- Antithetic pairing of the supplied increments was not added: it would change the estimator relative to the
  harness-supplied `dW` and the plan does not describe it.

## Evaluation

Evaluated by the verification kernel (`cli.py evaluate`); full output in `evaluation.json`.

<metrics>
kind: sde
score: 6
certified: false
provenance: analytic
estimated_rel_error: 0.1502
error_is_estimate: false
observed_order: 0.9682
observed_order_se: 0.0203
order_floor: 1.0000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 1.0000
scheme_family: milstein
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: unavailable
invariant_max_drift: 0.0000
invariant_worst_headroom: 6.0000
reference_outcome: validated
mean_rel_err: 0.0170
variance_rel_err: 0.1502
order_check: true
mc_se_rel: 0.0318
resolved: false
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 909f397c5d33a34d32cec562e62609689fbc331386e08df3b1ef2dbc681cbb08
solve_time_s: 0.0920
wall_time_s: 1.2400
cost_source: cpu
</metrics>

<review score=6>

**Score: 6/10**

### Numerical Accuracy
- provenance:              analytic  (analytic_moments, reference_outcome validated; the spec's moment ODE agrees to 1e-14 per the spec)
- mean_rel_err:            X 0.0170 +/- 2.0 x 0.0141 abs (outcome unresolved: CI wider than the 5% band); Y 0.0104 +/- 2.0 x 0.0124 abs (passes)
- variance_rel_err:        X 0.0854 +/- 2.0 x 0.0107 abs (passes, ref 0.3644); Y 0.1502 +/- 2.0 x 0.0098 abs (ref 0.2677; fails the 10% bar, and the 2-se interval on the relative error is roughly 7.7% to 22%, which straddles it)
- mc_se_rel:               0.0318   resolved: false (per-component CI width 5.85% for X and 7.32% for Y against the 5% bar, at 2000 paths)
- functional_rel_error:    not applicable, the spec declares no gated functional (the kernel emits no bias/sampling split here)
- observed_order:          0.968 +/- 0.020 strong (expected 1.0, one-sided below the declared floor); per component 1.023 (X), 0.968 (Y); median-based 1.008 / 0.884. Weak order 1.10, Richardson stable.
- weak_order:              1.10 (x), 1.05 (x^2); the tanh estimate is noise-limited on component 0 and not scored
- dynkin:                  not run (`dynkin_ok: not_run`, not a pass; `config.run_dynkin` is false). The reason string the kernel prints cites "a declared constraint gate failed", but `gate_failed` is empty and `constraints_ok` is true, so that string is stale and does not describe this run.
- terminal_law:            not declared (spec `terminal_law: null`, wrapped normal). Shape for context: skewness -0.80 +/- 0.04 (X), -1.04 +/- 0.04 (Y); excess kurtosis -0.54 +/- 0.08 (X), 0.03 +/- 0.12 (Y)
- constraints:             finite true; amplitude_conservation true (drift 0.0, headroom 6.0); amplitude_rms true; invariants_ok true; neg_fraction not applicable
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric, MC-inconclusive: the error bars are too wide to decide (certified: false)

### Feedback for solver
- **Same kernel result as the previous pass: score 6, certified false, resolved false, mc_se_rel 0.0318, estimated_rel_error 0.1502.** `solver.py` is byte-identical (same sha256), so nothing could move.
- **This is MC-inconclusive, not a solver bug, and your reading of the retry is correct on the path count.** The graded call is fixed at 2000 paths by ledger R12 and `dW` is supplied, so `num_paths` is not a lever, and declining to touch it was right. The kernel's text "raise num_paths" does not apply to this problem as specified.
- **The code is correct.** The Milstein correction has the right sign and uses `X_n` (strong order 0.968 +/- 0.020 against the floor of 1.0), `dW` is consumed step by step, `dt = T/dW.shape[1]` is recomputed, `ddof=1` is used (`reported_moments_agree` true to 3e-15), the Ito drift correction is present, and amplitude conservation passes. No cap was applied.
- **Where the retry's reasoning stops short.** It says there is no lever on the sampling error and leaves the plan unchanged. That is true of the path count and untrue of the result, because the bias term is a lever you declined to pull. The Y variance error is 15.0% against a 10% bar, and the kernel's weak-order study puts the dt = 0.04 gap in E[x^2] on Y at 0.0211, about 8% of the exact variance 0.2677. This matches the spec's own +8.7% Milstein figure. This is my reading of `weak_detail`, not a kernel field. Even with zero sampling noise this scheme sits only 1 to 2 points inside the bar, so the plan is thin-margin by construction, and no seed or path count changes that.
- **What would change the outcome is the scheme, not the sampling.** `dt` is fixed at 0.04, so a Stratonovich-consistent scheme is the lever. The spec measured strong error 0.037 for stochastic Heun and 0.016 for the Cayley midpoint rotation `X_{n+1} = (I - th J)^{-1}(I + th J) X_n`, `th = (a dt + sigma dW_n)/2`, which conserves |X| to round-off, against 0.063 for Milstein. Both consume `dW` exactly, so they need no `exact_sampling_components` declaration. If this plan is meant to stay the Milstein member of the pool, accept that it will not resolve at 2000 paths and that its score is bound by the rubric rather than by a defect. Antithetic pairing of the supplied increments does not apply, since the harness fixes the noise.
- **Spec-level observation, not a solver defect.** With `graded_num_paths = 2000` the variance CI half-width is about 6 to 7% of the variance, against a 10% bar. Any order-1 scheme carrying a bias above roughly 3% will be MC-inconclusive on this problem whatever the seed. That is a property of the problem's path budget. No formulator defect found otherwise: the reference is a validated closed form with a matching moment ODE, so no surrogate was left undeclared.

</review>
