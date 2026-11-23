---
id: 3
plan_slug: euler-maruyama
scheme: euler-maruyama
scheme_family: euler-maruyama
strategy: Plain Euler-Maruyama on the Ito-converted system — the order-0.5 baseline that establishes what the Milstein correction is worth, and the plan most at risk of the 20% pathwise gate at dt = 0.04.
---

Do **not** declare `exact_sampling_components` for this plan, and the spec says so explicitly:
every component consumes the supplied `dW`, there is no auxiliary randomness anywhere, and the
step carries a genuine O(sqrt(dt)) strong error. The CRN ladder must see it shrink.

**This is the baseline, and it is ranked last on purpose.** Euler-Maruyama's strong order is 0.5,
so at the contract step `dt = 0.04` its pathwise error is roughly `C*sqrt(dt) = 0.2*C` — the same
order of magnitude as the harness's 20% relative strong-error gate. It may clear that gate and it
may not; what it does reliably is measure the baseline so the order-1 plans in this pool have
something to be compared against. Do not substitute a higher-order step to rescue it: a plan that
implements a different scheme than it declares is capped by the evaluator regardless of accuracy.

**Not a transcription.** Composing *exact* per-step rotations through `a*dt + sigma*dW_n` would
reproduce the exact solution identically at every `dt` (rotation angles add) and such a run is
excluded from the comparison. The Euler step below is only the first-order truncation of that
rotation; its terminal states differ between `dt`, `dt/2` and `dt/4` and its pathwise difference
shrinks like `sqrt(dt)`.

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

- **Method**: Euler-Maruyama on the **Ito** form above. With `J = [[0, -1], [1, 0]]` the step is the
  single vectorised matrix action the spec's `implementation_notes` give:

  ```
  X_{n+1} = [ (1 - 0.5*sigma**2*dt) * I + (a*dt + sigma*dW_n) * J ] X_n
  ```

  Componentwise, with `m = 1 - 0.5*sigma**2*dt` and `theta_n = a*dt + sigma*dW_n`:

  ```
  X1_{n+1} = m*X1_n - theta_n*X2_n
  X2_{n+1} = theta_n*X1_n + m*X2_n
  ```

  `theta_n` is a `(num_paths,)` vector, so the whole path vectorises with no inner loop over
  components.
- **Milstein correction**: **none** — this is the Euler-Maruyama baseline. Adding
  `-0.5*sigma**2*X_n*(dW_n**2 - dt)` would make it the Milstein scheme and contradict the declared
  `scheme_family`.
- **Stratonovich handling**: this plan integrates the **Ito-converted** drift, so the
  `-0.5*sigma**2*X*dt` term **must** be present. Dropping it integrates a different equation whose
  amplitude grows like `exp(sigma**2*t)` — 172% high in `E[|X_T|**2]` at `T = 1`, which is precisely
  the failure mode `problem.md` warns about.
- **dt**: `0.04` — the contract step (`evaluation_thresholds.dt`). Take it from the argument; do not
  hardcode a different value. Round as `Nt = max(1, round(T/dt))`, then `dt = T/Nt`; when `dW` is
  supplied take `Nt = dW.shape[1]` and `dt = T/Nt` instead (at the graded call `Nt = 25`). The step
  is fixed by the contract, so refining `dt` is **not** available as a response to an accuracy
  shortfall on this plan — that is the point of running it.
- **num_paths**: `50000` (`evaluation_thresholds.num_paths`) for the kernel's moment checks. The
  contract's own graded call is `solve_sde(num_paths=2000, dt=0.04, T=1, seed=42)`; the solver must
  honour whatever `num_paths` it is passed. 2000 paths does not resolve the 10% variance tolerance
  (spec requirement R11), which is why the moment checks run at 50000.
- **T**: `1.0`
- **seed**: `42`, via `np.random.default_rng(seed)` only (never `np.random.seed`). Randomness is
  drawn only when `dW is None`; at the graded call `dW` is always supplied and must be used verbatim
  (shape `(2000, 25)`, **already scaled by sqrt(dt)**).
- **Expected strong order**: **0.5** (`verification.expected_strong_order`)
- **Expected weak order**: **1.0** (`verification.expected_weak_order`)
- **Reference for scoring**: **analytic moments** — `analytic_moments.has_analytic_solution` is
  `true` and all four `*_value_at_T` constants are supplied and independently validated against the
  closed moment ODE in `verification.moment_ode`. Expected provenance `analytic`.

## Implementation Notes

- **Stratonovich vs Ito is the single decisive choice.** This plan uses the Ito form, so the
  `-0.5*sigma**2*X*dt` drift correction is mandatory.
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
  `X1**2 + X2**2` with reference value `1.0`. The spec records the measured value for a *correct*
  Euler-Maruyama run on the Ito form at `dt = 0.04`: `E[|X_T|**2]` about `1.05`, with a pathwise
  amplitude spread of about `0.14`. That is this scheme's own O(dt**2)-per-step amplitude bias, which
  is why the entry is ungated — a hard gate would reject a correct solver here. The Ito-misread
  failure mode (`2.718`) sits two orders of magnitude outside that margin, so the diagnostic still
  separates the two; a value near `0.368` means the drift correction was applied twice.
- **No positivity or support guard is needed** — the state is bounded. The only gated constraint is
  `finite`.
- **No boundary and no barrier** are declared in `verification.structural_facts`, so no
  `boundary_treatment` applies.
- **No declared payoff functional** (`evaluation_thresholds` has no `functionals` entry) and the
  problem is graded path by path against its own exact solution, so there is no quantity of interest
  for a control variate or antithetic pair to reduce. The binding error here is squarely the
  pathwise discretization error.
- **Refinement round (evaluator score 5, non-gate martingale drift 0.0513 vs bar 2.82e-3; variance bias 0.634 of tol): this sub-step is the floor for the family, so nothing was changed.**
  The sub-step that carries both is the one-stage drift/diffusion evaluation at `X_n`, i.e. the rotation `exp(theta J)` replaced by its first-order truncation `I + theta*J`. Per step, `E|X_{n+1}|^2 = |X_n|^2 * (1 + 1.25*dt^2)` exactly (the `dW` terms cancel in expectation: `(1-dt/2)^2 + dt^2 + dt = 1 + 1.25 dt^2`), so over `Nt = 25` the drift is `(1+1.25*0.0016)^25 - 1 ~ 0.051`, matching the measured 0.0513. It is O(dt) globally, scheme-intrinsic, independent of `num_paths`, and cannot be reduced at the fixed contract `dt = 0.04`. Raising its order means adding the Milstein term `-0.5*sigma**2*X_n*(dW**2 - dt)` (which would make this the Milstein scheme and require `scheme_family: milstein`) or an exact/Heun rotation (excluded by problem.md as transcription). That is a different plan's job; this plan stays the declared Euler-Maruyama baseline, `scheme_family: euler-maruyama` unchanged (it was and remains `euler-maruyama`). The variance bias at 0.634 of tolerance is the same first-order truncation seen in the second moment and likewise the family floor.
- **Pathwise gate — the known risk on this plan.** The harness gates
  `sqrt(mean((X_num - X_exact)**2)) / sqrt(mean(X_exact**2))` at `< 20%` against its own exact
  solution on the supplied increments. At strong order 0.5 and a fixed `dt = 0.04`, this plan sits
  near that bar with no step-refinement lever, so it is expected to be the weakest plan in the pool
  and possibly to miss. Report the measured value rather than tuning the scheme to beat it.

## Results

Run: `solve_sde(num_paths=50000, dt=0.04, T=1.0, seed=42)`, plain Euler-Maruyama on the Ito form (no Milstein term).

- Empirical mean (X1, X2): [0.321722, 0.528647] (exact 0.327710, 0.510378)
- Empirical variance (X1, X2): [0.386453, 0.281917] (exact 0.364447, 0.267674)
- Variance rel. error X1: 6.0% (within the 10% tolerance); mean rel. error X1: 1.8%
- E|X_T|^2 = 1.051 (the amplitude diagnostic the spec expects for EM at dt = 0.04; the Ito misread gives 2.718)
- Pathwise relative strong error on 2000 supplied increments (dt = 0.04, Nt = 25): 0.148, under the 20% gate
- dW and observables hooks both confirmed working.

## Evaluation

(Filled by evaluator)

<metrics>
kind: sde
score: 5
certified: false
provenance: analytic
estimated_rel_error: 0.0604
error_is_estimate: false
observed_order: 0.5306
observed_order_se: 0.0118
order_floor: 0.5000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 0.5000
scheme_family: euler-maruyama
invariants_ok: false
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: unavailable
invariant_max_drift: 0.0513
invariant_worst_headroom: -1.2607
invariants_not_reported: ['amplitude_conservation']
reference_outcome: validated
mean_rel_err: 0.0358
variance_rel_err: 0.0604
moment_bias_binding: variance
moment_bias_frac_tol: 0.6341
moment_bias_actionable: true
order_check: true
mc_se_rel: 6.977e-03
resolved: true
dynkin_z: 2.1165
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 848c5d6ccf96eb9945f5a456d329573552ead13e5b69ae97a4cf8f25d9031b12
solve_time_s: 0.1370
wall_time_s: 22.5900
cost_source: cpu
</metrics>

<review score=5>

**Score: 5/10**

### Numerical Accuracy
- provenance:              analytic  (reference route: analytic_moments, validated)
- mean_rel_err:            0.0358 (X2, worst) +/- 2.0 x 0.00237 abs se; tol 0.05; X1 0.0183. Passes.
- variance_rel_err:        0.0604 (X1, worst) +/- 2.0 x 0.00224 abs se; tol 0.10; X2 0.0532. Passes.
- mc_se_rel:               6.98e-03   resolved: true (CI width rel 0.0123 / 0.0147 against bar 0.05)
- functional_rel_error:    not applicable, no functional declared. Graded quantity is the moments; binding dt bias is the variance at 0.634 of its tolerance (moment_bias_actionable: true), mean at 0.583.
- observed_order:          0.531 +/- 0.012 (expected 0.5, one-sided below the declared floor). Consumes dW correctly.
- weak_order:              1.15 (X2), 0.90 (X1); inside band, reported only
- dynkin:                  ok. z = 2.12 against bar 2.43, **marginal**. Replicated on 3 seeds (median z 2.12, per-seed worst z 2.12 / 3.49 / 0.89, pooled z 3.75 and the pooled estimate did not agree). Pass rests on the median rule; the 3.49 seed is above the bar.
- terminal_law:            not declared
- shape:                   skewness -0.786 +/- 0.008 (X1), -1.036 +/- 0.009 (X2); excess kurtosis -0.315 +/- 0.019, 0.434 +/- 0.028
- constraints:             finite: pass (gate). martingale on X1^2+X2^2: **fail**, drift 0.0513 against bar 0.00282 (headroom -1.26 decades). amplitude_conservation: not_reported (free-form, declared but unchecked).
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. Moments match the closed form and MC is resolved, but a declared non-gate invariant drifts materially.

### Feedback for solver
- The score is 5 because of one thing only: `E[X1^2+X2^2]` at T is about 1.0513, not 1.0, against a bar of 0.0028. Moments, order, Dynkin and the gated constraint all pass. This is not MC noise (the check is resolved) and not a transcription error: the Ito drift correction is present and the Ito-misread value (2.718) is far away.
- The drift is the scheme's own amplitude bias. Each EM step multiplies |X|^2 by (1 - 0.5 dt)^2 + (dt + dW)^2, which exceeds 1 by O(dt^2) plus a dW^2 term that does not cancel. It is first order in dt and does not shrink with paths. The plan's own notes predict this (about 1.05) and say it is why the entry is ungated. The kernel still scores a non-gate invariant, so this plan will stay at 5 as written.
- Plain EM at the contract dt = 0.04 cannot clear the 0.0028 bar, since the dt is fixed. The only fix is a different step. Strong-order and amplitude behaviour both improve with the Milstein correction `-0.5*sigma**2*X_n*(dW**2 - dt)` (valid here, one channel, no Levy area). A step that applies the rotation through `a*dt + sigma*dW` to second order (the spec's Stratonovich midpoint/Heun form) cuts the amplitude drift much more. Either belongs in a different plan, not this one, because this plan declares itself the EM baseline. Do not add the correction here: it would contradict `scheme_family: euler-maruyama`.
- Solver code review: no defects found. Increments are consumed from `dW`, `Nt = dW.shape[1]`, `dt = T/Nt`, `ddof=1`, observables hook is trapezoidal, reported moments agree with the kernel to 5e-15. No cap applied.
- Dynkin was marginal (z 2.12 vs bar 2.43, one replicate seed at 3.49). That is the O(dt) residual of EM at 0.04 showing through, not a wrong equation (z of hundreds would be). Nothing to change in the code.
- The order study is consistent with EM: 0.531 +/- 0.012 against 0.5.

</review>
