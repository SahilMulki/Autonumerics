---
id: 2
plan_slug: milstein
scheme: milstein
scheme_family: milstein
levy_area_treatment: not-required (single Brownian channel, noise_dimension = 1)
strategy: Explicit Milstein on the Ito-converted system, whose correction -0.5*sigma**2*X_n*(dW**2 - dt) is exactly computable because there is only one Brownian channel and therefore no Levy area.
---

Do **not** declare `exact_sampling_components` for this plan, and the spec says so explicitly:
every component consumes the supplied `dW`, there is no auxiliary randomness anywhere, and the
step carries a genuine O(dt) strong error. The CRN ladder must see it shrink.

**This is a discretised scheme, not a transcription.** The exact flow is the rotation through
`theta(t) = a*t + sigma*W(t)`, and composing *exact* per-step rotations through
`a*dt + sigma*dW_n` reproduces it identically at every `dt`, because rotation angles add. Such a
run is excluded from the comparison as a transcription. **Do not do that.** The Milstein step below
is the truncated expansion of that rotation — it is *not* a rotation (its matrix has determinant
`(1 - 0.5*sigma**2*dW**2)**2 + theta**2 != 1`), the terminal states differ between `dt`, `dt/2` and
`dt/4`, and the pathwise difference shrinks like `dt`.

## SDE Reference

Stratonovich form, as the problem states it:

```
dX1 = -a*X2 dt - sigma*X2 o dW
dX2 =  a*X1 dt + sigma*X1 o dW,        X(0) = (1, 0)
```

Equivalent **Ito** form, which is what this plan integrates (Stratonovich-to-Ito correction
`0.5*(dg/dX)g dt` with `g(X) = (-sigma*X2, sigma*X1)`, `dg/dX = sigma*[[0, -1], [1, 0]]`, hence
`(dg/dX)g = (-sigma**2*X1, -sigma**2*X2)`):

```
dX1 = (-a*X2 - 0.5*sigma**2*X1) dt - sigma*X2 dW
dX2 = ( a*X1 - 0.5*sigma**2*X2) dt + sigma*X1 dW
```

Parameters: `a = 1.0`, `sigma = 1.0`, `X_0 = 1.0`, `Y_0 = 0.0`, `T0 = 0.0`, `T = 1.0`.
One Brownian channel (`noise_dimension = 1`), state dimension 2, multiplicative noise, not stiff.

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

- **Method**: explicit Milstein on the **Ito** form above. With `J = [[0, -1], [1, 0]]` the step is

  ```
  X_{n+1} = X_n + f(X_n)*dt + g(X_n)*dW_n + 0.5*(dg/dX)g|_{X_n} * (dW_n**2 - dt)
          = X_n + (a*dt*J - 0.5*sigma**2*dt*I) X_n + sigma*dW_n*J X_n
                - 0.5*sigma**2*X_n*(dW_n**2 - dt)
  ```

  The two `+/- 0.5*sigma**2*dt*X_n` terms cancel, so it collapses to a single vectorised matrix
  action with `theta_n = a*dt + sigma*dW_n`:

  ```
  X_{n+1} = [ (1 - 0.5*sigma**2*dW_n**2) * I + theta_n * J ] X_n
  ```

  Componentwise, with `m = 1 - 0.5*sigma**2*dW_n**2`:

  ```
  X1_{n+1} = m*X1_n - theta_n*X2_n
  X2_{n+1} = theta_n*X1_n + m*X2_n
  ```

  `theta_n` and `m` are `(num_paths,)` vectors, so the whole path vectorises with no inner loop over
  components. **Use `X_n` in the correction, never `X_{n+1}`** — applying it at the updated state is
  the classic bug that silently drops the scheme back to strong order 0.5, which the kernel's CRN
  ladder detects.
- **Milstein correction**: `0.5*(dg/dX)g*(dW**2 - dt) = -0.5*sigma**2*X_n*(dW_n**2 - dt)`, applied
  componentwise, exactly as `milstein_correction` in the spec gives it. No Levy area is required:
  `noise_dimension = 1`, so the only iterated integral needed is `int dW dW = 0.5*(dW**2 - dt)`,
  which is computable in closed form from the supplied increment.
- **Stratonovich handling**: this plan integrates the **Ito-converted** drift, so the
  `-0.5*sigma**2*X` term **must** be present in `f`. Dropping it integrates a different equation
  whose amplitude grows like `exp(sigma**2*t)` — 172% high in `E[|X_T|**2]` at `T = 1`. Do *not*
  additionally use a midpoint/Heun corrector on top; exactly one of the two treatments belongs in
  the code and this plan's is the Ito one.
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
- **Expected strong order**: **1.0** (`verification.expected_strong_order_milstein`)
- **Expected weak order**: **1.0** (`verification.expected_weak_order`)
- **Reference for scoring**: **analytic moments** — `analytic_moments.has_analytic_solution` is
  `true` and all four `*_value_at_T` constants are supplied and independently validated against the
  closed moment ODE in `verification.moment_ode`. Expected provenance `analytic`.

## Implementation Notes

- **Stratonovich vs Ito is the single decisive choice.** This plan uses the Ito form, so the
  `-0.5*sigma**2*X` drift correction is mandatory.
- **Use the supplied increments exactly.** `dW` has shape `(num_paths, Nt)` for one channel and is
  already scaled by `sqrt(dt)`. Index it as `dW[:, n]`; do not rescale by `sqrt(dt)` again and do not
  generate any Brownian randomness of your own when it is given. A solver that ignores `dW` reads a
  strong order near zero and is failed by the order study.
- **Return contract**: `terminal_paths` as a `(num_paths, 2)` float array with columns `(X1, X2)` in
  that order; `empirical_mean` as a list of 2 floats; `empirical_variance` as a list of 2 floats with
  `ddof=1`. Keep `solve_sde` importable — no module-load work outside `if __name__ == "__main__"`.
  numpy and the standard library only.
- **Also accept the optional `observables` argument** (`{name: phi(X)}`, accumulating the trapezoid
  path integral of `phi` per path into `out["path_integrals"]`). The problem's own contract lists only
  `dW`, but the kernel's Dynkin check uses `observables` and the declared test functions are
  `X[:, 0]`, `X[:, 1]` and `X[:, 0]**2 + X[:, 1]**2`. Supporting it is the difference between
  `dynkin_ok: True` (score 10 available) and a Dynkin that never reached a verdict (capped at 9).
  Default it to `None` so the contract call is unaffected.
- **Amplitude diagnostic.** `verification.constraints` carries an ungated `martingale` entry on
  `X1**2 + X2**2` with reference value `1.0`. This step is not a rotation, so it conserves the
  amplitude only to O(dt**2) per step: at `dt = 0.04` expect `E[|X_T|**2]` within a few percent of
  1.0 with a modest pathwise spread. That is a correct order-1 scheme and the entry is ungated for
  exactly this reason. What it still separates: `~2.718` means the Ito correction was dropped,
  `~0.368` means it was applied twice.
- **No positivity or support guard is needed** — the state is a bounded near-rotation. The only gated
  constraint is `finite`.
- **No boundary and no barrier** are declared in `verification.structural_facts`, so no
  `boundary_treatment` applies.
- **No declared payoff functional** (`evaluation_thresholds` has no `functionals` entry) and the
  problem is graded path by path against its own exact solution, so there is no quantity of interest
  for a control variate or antithetic pair to reduce. The binding error is the pathwise
  discretization error, not the sampling error.
- **Pathwise gate.** The harness gates `sqrt(mean((X_num - X_exact)**2)) / sqrt(mean(X_exact**2))`
  at `< 20%` against its own exact solution on the supplied increments. Spec requirement R20 records
  that `evaluation_thresholds` has no vocabulary slot for this statistic; it is carried by the exact
  moments (5% mean / 10% variance) plus the strong-order floor of 1.0. An order-1 scheme at
  `dt = 0.04` should land well inside 20%.

## Results

Run: `solve_sde(num_paths=50000, dt=0.04, T=1.0, seed=42)` (Nt = 25), explicit Milstein on the Ito form.

| | X1 | X2 |
|---|---|---|
| empirical mean | 0.322273 (exact 0.327710) | 0.528361 (exact 0.510378) |
| empirical variance (ddof=1) | 0.396148 (exact 0.364447) | 0.293643 (exact 0.267674) |

E|X_T|^2 = 1.0728 (diagnostic, ungated). Variance errors are about 8.7% (X1) and 9.7% (X2) at dt = 0.04, near the 10% tolerance; this is discretization bias, which shrinks like dt. The `dW` and `observables` hooks both ran.


## Evaluation

<metrics>
kind: sde
score: 5
certified: false
provenance: analytic
estimated_rel_error: 0.0970
error_is_estimate: false
observed_order: 0.9922
observed_order_se: 7.364e-03
order_floor: 1.0000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 1.0000
scheme_family: milstein
invariants_ok: false
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: unavailable
invariant_max_drift: 0.0728
invariant_worst_headroom: -1.9656
invariants_not_reported: ['amplitude_conservation']
reference_outcome: validated
mean_rel_err: 0.0352
variance_rel_err: 0.0970
moment_bias_binding: variance
moment_bias_frac_tol: 0.9033
moment_bias_actionable: true
order_check: true
mc_se_rel: 6.441e-03
resolved: true
dynkin_z: 16.7354
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 08028ac211996d41295d702a8d16d2b2c8019c4ce80540c6d14a49bb77760986
solve_time_s: 0.1320
wall_time_s: 14.8600
cost_source: cpu
</metrics>

<review score=5>

**Score: 5/10**

### Numerical Accuracy
- provenance:              analytic  (analytic_moments, validated against the closed moment ODE)
- mean_rel_err:            0.0352 (X2, worst) +/- 2 x 0.0024 se; tol 0.05 -- passes
- variance_rel_err:        0.0970 (X2, worst; X1 is 0.0870) +/- 2 x 0.0019 se; tol 0.10 -- passes with 3% headroom
- mc_se_rel:               6.44e-03   resolved: true (CI width rel 0.0116 / 0.0141 against bar 0.05)
- functional_rel_error:    not applicable, the spec declares no gated functional
- observed_order:          0.992 +/- 0.007 (expected 1.0, one-sided band, scheme_family declared). Median-based estimate 0.954. Weak order 0.98
- weak_order:              0.98 (x: 0.92, x2: 1.04; tanh excluded at the noise floor on component 0)
- dynkin:                  unresolved (not a pass, not a failure). Test functions X1 and X2 are clean (z = -0.98, -0.75). The amplitude function X1^2 + X2^2 reads z = 16.7 against bar 2.43, but the kernel declined to rule on it. The extrapolated limit moves from 5.2e-4 to 1.5e-4 between its two windows, 14 paired se, so the ladder is not in one asymptotic regime. That is a property of the estimator on this function, not evidence about the generator. Not replicated: one seed, since "unresolved" is not a verdict more seeds can change
- terminal_law:            not declared (wrapped normal is outside the implemented laws)
- shape:                   skewness -0.78 +/- 0.008 (X1), -1.04 +/- 0.008 (X2); excess kurtosis -0.58 +/- 0.016 (X1), 0.07 +/- 0.024 (X2)
- constraints:             finite: pass. martingale on X1^2 + X2^2: FAIL (non-gate), E|X_T|^2 = 1.0728, drift 0.0728 against bar 7.9e-4, headroom -1.97 decades. amplitude_conservation: not_reported (free-form entry, no expr)
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric -- moments match the closed form and MC is resolved, but a declared non-gate invariant drifts materially

### Feedback for solver
- **The score is bound at 5 by the amplitude invariant, and it is a real discretization effect, not a code bug.** Your step matrix `[(1 - 0.5 s^2 dW^2) I + theta J]` has per-step squared norm `(1 - 0.5 w^2)^2 + (dt + w)^2`, whose expectation is `1 + 1.75 dt^2`. Over 25 steps that gives 1.0728, exactly what was measured (the plan's own notes expected this). Euler-Maruyama gives `1 + 1.25 dt^2`, about 1.05, so the Milstein correction makes this particular invariant slightly worse, not better. The bar is a 2-se band around 1.0 (7.9e-4), so no order-1 step with O(dt^2) per-step amplitude error can land inside it at dt = 0.04. Your SOLUTION.md frames it as "a correct order-1 scheme and the entry is ungated". It is ungated for the gate, but a non-gate invariant that drifts still caps the score at 5.
- **The fix is a scheme that conserves the amplitude by construction and still consumes dW.** Use the Stratonovich midpoint (Cayley) step on the coefficients as written: `X_{n+1} = (I - 0.5*theta*J)^{-1} (I + 0.5*theta*J) X_n` with `theta = a*dt + sigma*dW_n`. In 2-D this is a closed form: with `c = 0.25*theta**2`, `X_{n+1} = [(1 - c) X_n + theta J X_n] / (1 + c)`. It preserves `|X|` exactly, so the invariant drift goes to roundoff. It is still a discretisation, not a transcription: the step angle is `2*atan(theta/2)` rather than `theta`, so terminal states differ between dt, dt/2 and dt/4 and the CRN ladder will see a strong order of 1. It uses `dW[:, n]` exactly, so no `exact_sampling_components` declaration applies and none should be made. The spec's `milstein_note` names this midpoint scheme as strong order 1.
- **The same change should relieve the variance bias.** The variance error is 0.0970 of a 0.10 tolerance. `moment_bias_frac_tol` is 0.90 (graded value 0.3946 against extrapolated 0.3619 on X1, `shift_in_se` about 60), so the plan clears that gate only by 3%. More paths will not help, since `resolved` is true with se 0.0019. The rotation-angle error of the Cayley step is O(theta^3) rather than the O(theta^2) amplitude defect and O(dt) angle defect carried by the Milstein matrix, so expect the variance bias to fall well below 0.9 of tolerance. Re-measure it rather than assuming.
- **Dynkin unresolved is the second thing that stops this at under 10, and it is not a solver defect.** The generator identity holds for X1 and X2. The amplitude test function (`L phi = 0` exactly) has a residual that is entirely discretization error, and the extrapolation windows disagree by 14 se, so the kernel abstained. A step that conserves `|X|` exactly should give a residual near roundoff on this function, which the estimator treats as a clean pass (`at_roundoff`). That would be the route from 9 to 10, and would be a consequence of the same fix, not a separate one.
- Everything else is fine: `dW` is consumed verbatim with `Nt = dW.shape[1]`, `ddof=1` is used (`reported_moments_agree` 1.2e-14), the Ito correction has the right sign and is applied at `X_n`, and the strong order of 0.992 +/- 0.007 is on the design value. Do not touch those.
- **For the formulator, not the solver:** the spec's `martingale` bar is a 2-se Monte Carlo interval (7.9e-4) applied to a quantity the spec itself says every consistent order-1 scheme misses by about 0.05 at the contract step. The `note` on `verification.constraints[1]` states this, yet the kernel still scores the miss at 5. Either the entry should be dropped from `constraints` (and left in prose), or a `tol` that reflects the contract step's discretization error should be declared. As written, only an amplitude-preserving scheme can pass it.

</review>
