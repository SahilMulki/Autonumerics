---
id: 1
plan_slug: stratonovich-midpoint
scheme: stratonovich-midpoint (implicit midpoint / Cayley realization of Stratonovich-Milstein)
scheme_family: milstein
levy_area_treatment: not-required (single Brownian channel, noise_dimension = 1)
strategy: Advance each step by the implicit midpoint rule on the Stratonovich coefficients as written, which for this linear rotation system solves in closed form to a Cayley transform — strong order 1.0 and exact pathwise amplitude conservation at every dt.
---

Do **not** declare `exact_sampling_components` for this plan, and the spec says so explicitly:
every component consumes the supplied `dW`, there is no auxiliary randomness anywhere, and the
step carries a genuine O(dt) strong error. The CRN ladder must see it shrink.

**This is a discretised scheme, not a transcription.** The exact flow is the rotation through
`theta(t) = a*t + sigma*W(t)`, and composing *exact* per-step rotations through
`a*dt + sigma*dW_n` reproduces it identically at every `dt` — because rotation angles add and
`sum_n (a*dt + sigma*dW_n) = a*T + sigma*W(T)`. Such a run is excluded from the comparison as a
transcription. **Do not do that.** The midpoint step below rotates through
`2*arctan(theta_n/2)`, not through `theta_n`; the angles no longer telescope to the exact total,
the terminal states differ between `dt`, `dt/2` and `dt/4`, and the pathwise difference shrinks
like `dt` — which is the order-1 evidence this plan is claiming.

## SDE Reference

Stratonovich form (coefficients exactly as the problem states them):

```
dX1 = -a*X2 dt - sigma*X2 o dW
dX2 =  a*X1 dt + sigma*X1 o dW,        X(0) = (1, 0)
```

Equivalent Ito form (Stratonovich-to-Ito correction `0.5*(dg/dX)g dt` with
`(dg/dX)g = (-sigma**2*X1, -sigma**2*X2)`):

```
dX1 = (-a*X2 - 0.5*sigma**2*X1) dt - sigma*X2 dW
dX2 = ( a*X1 - 0.5*sigma**2*X2) dt + sigma*X1 dW
```

Parameters: `a = 1.0`, `sigma = 1.0`, `X_0 = 1.0`, `Y_0 = 0.0`, `T0 = 0.0`, `T = 1.0`.
One Brownian channel (`noise_dimension = 1`), state dimension 2, multiplicative noise, not stiff.

Useful matrix form with `J = [[0, -1], [1, 0]]`: the Stratonovich system is `dX = J X (a dt + sigma o dW)`.

Exact pathwise solution (reference only — **do not evaluate it as the answer**):
`X1(t) = cos(a*t + sigma*W(t))`, `X2(t) = sin(a*t + sigma*W(t))`.

Exact moments at `T = 1` (from `analytic_moments`, validated against the closed moment ODE at rtol 1e-12):

```
mean_X     = np.cos(a*t) * np.exp(-0.5*sigma**2*t)                                  = 0.3277099140224599
mean_Y     = np.sin(a*t) * np.exp(-0.5*sigma**2*t)                                  = 0.5103779515445728
variance_X = 0.5 + 0.5*np.cos(2*a*t)*np.exp(-2*sigma**2*t) - np.cos(a*t)**2*np.exp(-sigma**2*t) = 0.364446537255328
variance_Y = 0.5 - 0.5*np.cos(2*a*t)*np.exp(-2*sigma**2*t) - np.sin(a*t)**2*np.exp(-sigma**2*t) = 0.2676740215732297
```

## Numerical Scheme

- **Method**: implicit midpoint rule on the Stratonovich coefficients (the midpoint / Stratonovich-Milstein
  scheme the spec's `milstein_note` names), solved exactly because the system is linear.

  With `theta_n = a*dt + sigma*dW_n` the midpoint rule reads

  ```
  X_{n+1} = X_n + theta_n * J * (X_n + X_{n+1}) / 2
  ```

  which for this 2x2 system inverts in closed form (a Cayley transform):

  ```
  X_{n+1} = (I - 0.5*theta_n*J)^{-1} (I + 0.5*theta_n*J) X_n
          = 1/(1 + theta_n**2/4) * [ (1 - theta_n**2/4) * I + theta_n * J ] X_n
  ```

  Componentwise, with `c = (1 - theta**2/4) / (1 + theta**2/4)` and `s = theta / (1 + theta**2/4)`:

  ```
  X1_{n+1} = c*X1_n - s*X2_n
  X2_{n+1} = s*X1_n + c*X2_n
  ```

  No linear solve and no inner loop over components are needed: `theta` is a `(num_paths,)` vector
  and the whole update vectorises. `c**2 + s**2 == 1` identically, so this matrix is an exact
  rotation through `2*arctan(theta_n/2)` and the amplitude `|X_n|` is conserved to roundoff on
  every path at every step. That is the problem's conserved-amplitude property satisfied *exactly*
  rather than to O(dt), which is why this plan is ranked first.
- **Stratonovich handling**: the midpoint rule is Stratonovich-consistent, so it is applied to the
  coefficients **as written** — do *not* additionally subtract the `-0.5*sigma**2*X` Ito correction
  here. Adding it on top would double-count the conversion and shrink the amplitude like
  `exp(-0.5*sigma**2*t)`; omitting the correction while using an *Ito* scheme is the opposite error
  (amplitude growing like `exp(sigma**2*t)`, 172% high in `E[|X_T|**2]`). Exactly one of the two
  treatments must be in the code. This plan uses the Stratonovich-consistent one.
- **dt**: `0.04` — the contract step (`evaluation_thresholds.dt`). Take it from the argument; do not
  hardcode a different value. Round as `Nt = max(1, round(T/dt))`, then `dt = T/Nt`; when `dW` is
  supplied take `Nt = dW.shape[1]` and `dt = T/Nt` instead (at the graded call `Nt = 25`).
- **num_paths**: `50000` (`evaluation_thresholds.num_paths`) for the kernel's moment checks. The
  contract's own graded call is `solve_sde(num_paths=2000, dt=0.04, T=1, seed=42)`; the solver must
  honour whatever `num_paths` it is passed. 2000 paths does not resolve the 10% variance tolerance
  (spec requirement R11), which is why the moment checks run at 50000.
- **T**: `1.0`
- **seed**: `42`, via `np.random.default_rng(seed)` only (never `np.random.seed`). Randomness is
  drawn only when `dW is None`; at the graded call `dW` is always supplied and must be used verbatim
  (shape `(2000, 25)`, **already scaled by sqrt(dt)**).
- **Milstein correction** (for reference, the equivalent explicit form on the Ito system):
  `0.5*(dg/dX)g*(dW**2 - dt) = -0.5*sigma**2*X_n*(dW**2 - dt)`. The midpoint step above agrees with
  it to O(dt**{3/2}) and attains the same strong order; it is preferred because it conserves the
  amplitude exactly.
- **Expected strong order**: **1.0** (`verification.expected_strong_order_milstein`)
- **Expected weak order**: **1.0** (`verification.expected_weak_order`)
- **Reference for scoring**: **analytic moments** — `analytic_moments.has_analytic_solution` is
  `true` and all four `*_value_at_T` constants are supplied and independently validated against the
  closed moment ODE in `verification.moment_ode`. Expected provenance `analytic`.

## Implementation Notes

- **Stratonovich vs Ito is the single decisive choice.** See the scheme section: use the midpoint
  rule on the coefficients as written, with no Ito drift correction added.
- **Use the supplied increments exactly.** `dW` has shape `(num_paths, Nt)` for one channel and is
  already scaled by `sqrt(dt)`. Index it as `dW[:, n]`; do not rescale by `sqrt(dt)` again and do
  not generate any Brownian randomness of your own when it is given. A solver that ignores `dW`
  reads a strong order near zero and is failed by the order study.
- **Return contract**: `terminal_paths` as a `(num_paths, 2)` float array with columns `(X1, X2)` in
  that order; `empirical_mean` as a list of 2 floats; `empirical_variance` as a list of 2 floats with
  `ddof=1`. Keep `solve_sde` importable — no module-load work outside `if __name__ == "__main__"`.
  numpy and the standard library only.
- **Also accept the optional `observables` argument** (`{name: phi(X)}`, accumulating the trapezoid
  path integral of `phi` per path into `out["path_integrals"]`). The problem's own contract lists only
  `dW`, but the kernel's Dynkin check uses `observables`, and the declared test functions are
  `X[:, 0]`, `X[:, 1]` and `X[:, 0]**2 + X[:, 1]**2`. Supporting it costs three lines and is the
  difference between `dynkin_ok: True` (score 10 available) and a Dynkin that never reached a
  verdict (capped at 9). Default it to `None` so the contract call is unaffected.
- **Amplitude diagnostic.** `verification.constraints` carries an ungated `martingale` entry on
  `X1**2 + X2**2` with reference value `1.0`. This scheme conserves it to roundoff pathwise
  (`c**2 + s**2 = 1`), so expect `E[|X_T|**2] = 1.0` to ~1e-15 and essentially zero pathwise
  amplitude spread. A value near `exp(sigma**2*T) = 2.718` means the Ito correction was dropped while
  using an Ito scheme; a value near `exp(-sigma**2*T) = 0.368` means the correction was applied on
  top of a Stratonovich-consistent scheme.
- **No positivity or support guard is needed** — the state lives on the unit circle and is bounded.
  The only gated constraint is `finite`, which a bounded rotation satisfies trivially.
- **No boundary and no barrier** are declared in `verification.structural_facts`, so no
  `boundary_treatment` applies.
- **No declared payoff functional** (`evaluation_thresholds` has no `functionals` entry) and the
  problem is graded path by path against its own exact solution, so there is no quantity of interest
  for a control variate or antithetic pair to reduce. Variance reduction is not an axis here: the
  binding error is the pathwise discretization error, not the sampling error.
- **Pathwise gate.** The harness gates `sqrt(mean((X_num - X_exact)**2)) / sqrt(mean(X_exact**2))`
  at `< 20%` against its own exact solution on the supplied increments. Spec requirement R20 records
  that `evaluation_thresholds` has no vocabulary slot for this statistic; it is carried by the exact
  moments (5% mean / 10% variance) plus the strong-order floor of 1.0. An order-1 scheme at
  `dt = 0.04` should land well inside 20%.

## Results

Run: `num_paths=50000, dt=0.04 (Nt=25), T=1, seed=42`, implicit midpoint (Cayley) step.

- Empirical mean (X1, X2): [0.335395, 0.515565]  (exact 0.327710, 0.510378)
- Empirical variance (X1, X2): [0.358565, 0.263150]  (exact 0.364447, 0.267674)
- Mean |X_T|^2 = 1.0 (amplitude conserved to roundoff)
- `dW` hook (explicit increments) and `observables` hook (trapezoid path integrals) both verified.

## Evaluation

(Filled by evaluator)

<metrics>
kind: sde
score: 3
certified: false
provenance: analytic
estimated_rel_error: 0.0234
error_is_estimate: false
observed_order: 0.9628
observed_order_se: 0.0111
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
invariants_not_reported: ['amplitude_conservation']
reference_outcome: validated
mean_rel_err: 0.0234
variance_rel_err: 0.0169
moment_bias_binding: mean
moment_bias_frac_tol: 0.0722
moment_bias_actionable: false
order_check: true
mc_se_rel: 6.970e-03
resolved: true
dynkin_z: 8.2415
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 76da8c07a42169161c1e9a63d7681173bf27c3d4db6bc871eeb2e56c080564ff
solve_time_s: 0.1310
wall_time_s: 14.9400
cost_source: cpu
</metrics>

<review score=3>

**Score: 3/10**

### Numerical Accuracy
- provenance:              analytic  (reference route: analytic_moments, validated)
- mean_rel_err:            0.0234 (X1), 0.0102 (X2) against tol 0.05; se 0.00268 / 0.00229, ci_mult 2.0 (family-wise 2.53). Pass.
- variance_rel_err:        0.0161 (X1), 0.0169 (X2) against tol 0.10; se 0.00181 / 0.00183. Pass.
- mc_se_rel:               0.00697   resolved: true (CI width 0.0099 / 0.0137 against bar 0.05, 50000 paths)
- functional_rel_error:    none declared (no gated functional)
- observed_order:          0.963 +/- 0.011 (expected 1.0, one-sided below the declared floor; median-based 0.879). Pass
- weak_order:              1.10 (x), 1.10 (tanh); x2 at the noise floor and not used. In band.
- dynkin:                  FAILED. z = 8.24 against bar 2.43 (family-wise), single seed, not replicated, so not marginal (|z - bar| = 5.8 against margin 0.7). Confirmed on a 4-level ladder after a 3-level failure. Per function: X[:,0] z = -1.00 ok, X[:,1] z = -0.74 ok, X[:,0]**2 + X[:,1]**2 z = -8.24 FAIL.
- terminal_law:            not declared
- shape:                   skewness -0.699 / -1.162 (se 0.0074 / 0.0088), excess kurtosis -0.721 / 0.429 (se 0.014 / 0.028)
- constraints:             finite ok, martingale ok (drift 0.0), amplitude_conservation not_reported
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. Dynkin fails (row 3), and the certification ceiling also lands on 3.

What the failure is and is not. The two moments match, the MC is resolved, the order study passes, the constraints hold, and the solver's reported moments agree with the kernel's to 7e-15. The scheme integrates the right equation: the first two test functions, which are sensitive to the drift and the Stratonovich correction, have z of -1.0 and -0.74. The failing function is the squared amplitude, and its extrapolated generator residual is -2.6e-11 with se 3.1e-12. That is roundoff scale in absolute terms. The Cayley step conserves X1^2 + X2^2 to machine precision on every path, so the true residual is zero and the measured one is accumulated roundoff. The finer residual difference (2.8e-12) was inside its own noise (2.2e-12), so no rate was read and the extrapolation assumed order 1 (`extrapolation_order_source: assumed_1`). A z of 8 on a residual of 1e-11 therefore says the standard error is also at the floor, not that the equation is wrong. By the manual's own scale a wrong equation gives z in the hundreds. I am not capping and not re-scoring; the score stays as the kernel computed it. I am flagging it for the conductor as a probable kernel false-positive on exactly conserved observables. `window_agreement` reports `at_roundoff: false` for this function although its own floor (2.7e-23) was computed, so the roundoff guard is not catching it. I did not edit anything outside this block.

### Feedback for solver
- Nothing in `solver.py` looks wrong, and I found no defect to fix. It consumes `dW` exactly with `Nt = dW.shape[1]` and `dt = T/Nt`. It uses `ddof=1`, applies the Stratonovich midpoint with no Ito double-counting, and `observables` is supported with the trapezoid integral.
- The score of 3 comes from a single test function, the exactly conserved `X1**2 + X2**2`, whose residual is about 1e-11. If the conductor judges this a real defect rather than a kernel artifact, the only knob on the solver side is the accumulation of the path integral for that observable (the trapezoid sum of phi(X_prev) and phi(X) times dt). The 25-term sum is already at roundoff, so I would not change it on this evidence.
- `SOLUTION.md` frontmatter says `scheme_family: milstein`. The scheme is the implicit midpoint (Cayley) step. It is the Stratonovich-Milstein-equivalent, and the measured strong order 0.963 is consistent with 1.0, so the claim holds. Nothing to change.
- `amplitude_conservation` was `not_reported`. It is non-gated, so it cost nothing, but a `constraints` return entry would make the exact amplitude conservation visible to the kernel instead of leaving it only in prose.
- Do not raise `num_paths`: `resolved` is true and the failing check is not MC-limited. A seed or path-count change will not move it.

</review>
