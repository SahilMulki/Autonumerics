---
id: 2
plan_slug: stochastic-trapezoidal
scheme: stochastic-trapezoidal
scheme_family: implicit
strategy: Drift-implicit trapezoidal (Crank-Nicolson / implicit midpoint) step for the linear drift with the additive noise injected through the same resolvent, solved in closed form as a 2x2 linear map; A-stable, strong order 1, second order in the moments.
---

## SDE Reference

State `X = (q, p)`, columns in that order; one Brownian channel entering `p` only (additive noise).

```
dq = p dt
dp = (-omega**2 * q - gamma * p) dt + sqrt(2*gamma/beta) dW
(q, p)(0) = (q_0, p_0) = (1, 0)
```

| parameter | value |
|---|---|
| omega | 4.0 |
| gamma | 1.0 |
| beta  | 1.0 |
| q_0   | 1.0 |
| p_0   | 0.0 |
| T     | 1.0 |

In matrix form: `dZ = A Z dt + b dW`, with `A = [[0, 1], [-omega**2, -gamma]]`, `b = [0, sqrt(2*gamma/beta)]`.

Exact terminal moments (for the solver's own self-check only; `nu = sqrt(omega**2 - gamma**2/4)`):

```python
nu = np.sqrt(omega**2 - gamma**2/4)
mean_q = np.exp(-gamma*T/2)*((np.cos(nu*T) + gamma/(2*nu)*np.sin(nu*T))*q_0 + np.sin(nu*T)/nu*p_0)
mean_p = np.exp(-gamma*T/2)*(-omega**2*np.sin(nu*T)/nu*q_0 + (np.cos(nu*T) - gamma/(2*nu)*np.sin(nu*T))*p_0)
var_q  = (1 - np.exp(-gamma*T)*(np.cos(nu*T) + gamma/(2*nu)*np.sin(nu*T))**2)/(beta*omega**2) \
         - np.exp(-gamma*T)*np.sin(nu*T)**2/(beta*nu**2)
var_p  = (1 - np.exp(-gamma*T)*(np.cos(nu*T) - gamma/(2*nu)*np.sin(nu*T))**2)/beta \
         - omega**2*np.exp(-gamma*T)*np.sin(nu*T)**2/(beta*nu**2)
```

## Numerical Scheme

- **Method**: stochastic trapezoidal rule (theta = 1/2), drift-implicit, noise explicit:
  ```
  (I - h/2 A) Z_{n+1} = (I + h/2 A) Z_n + b dW_n
  =>  Z_{n+1} = R Z_n + M b dW_n,   M = (I - h/2 A)^{-1},  R = M (I + h/2 A)
  ```
  `R` and `M b` are a fixed 2x2 matrix and 2-vector, computed once per call (invert the 2x2 by hand or with `np.linalg.inv`); the per-step update is two fused multiply-adds per component, vectorised over paths. Note this is *not* the exact propagator `expm(A h)` — it is its (1,1) Padé / Cayley approximation.
- **dt**: 0.04 (the contract step; `Nt = max(1, round(T/dt))`, `dt = T/Nt` -> Nt = 25)
- **num_paths**: 100000
- **seed**: 42
- **T**: 1.0
- **Milstein correction**: not applicable (additive noise, `dG/dX = 0`)
- **Expected strong order**: 1.0 (additive noise)
- **Expected weak order**: 2.0 in the mean and variance for this linear system (the spec's generic `expected_weak_order` is 1.0)
- **Reference for scoring**: analytic moments (`analytic_moments`, `verification.terminal_law`, both components), cross-checked by the exact moment ODE (`verification.moment_ode`)

## Implementation Notes

- **A discretised scheme is required by `problem.md`.** Do not replace `R` with `expm(A*dt)` or `M b dW` with an exact-covariance Gaussian draw — that would be exact sampling, which the problem rules out. Step with the trapezoidal map above.
- **Consume the supplied increments; draw no randomness of your own when `dW` is given.** Signature:
  `solve_sde(num_paths, dt, T, seed=42, dW=None, observables=None)`. If `dW` is `None`, draw
  `dW = sqrt(dt) * rng.standard_normal((num_paths, Nt))` from `np.random.default_rng(seed)` in one block. If `dW` is given, take `Nt = dW.shape[1]`, `dt = T/Nt`, and recompute `R`, `M b` from that `dt`. Do not declare `exact_sampling_components`.
- **Observables hook**: accumulate `int_0^T phi(X_s) ds` with the trapezoid rule on consecutive states, returned as `out["path_integrals"]`. Dynkin test functions: `q*p`, `p**2`, `tanh(q)`.
- **Why it should pass at dt = 0.04**: the Cayley factor's modulus per step matches `exp(-gamma*h/2)` to about 1e-4 relative and its phase error is `O((nu*h)**3/12)` per step (under 1e-2 rad over T), so the mean bias is expected at or below about 1%; the stationary covariance of the trapezoidal rule is exact for linear SDEs, so the variance bias is also small. A-stability means there is no stability risk at any `dt`.
- **Output**: `terminal_paths` shape `(num_paths, 2)` columns `(q, p)`; `empirical_mean` and `empirical_variance` (`ddof=1`) as lists of two floats; also return `num_paths`, `dt`, `T`. numpy + stdlib only; no work at import time.
- **Variance reduction**: not applicable — moments are graded, not a priced payoff.
- **No boundary**: the spec declares no barrier or absorbing boundary.

## Results

Run: `solve_sde(num_paths=100000, dt=0.04, T=1.0, seed=42)` (Nt = 25), stochastic trapezoidal rule.

| component | empirical mean | exact mean | empirical variance | exact variance |
|---|---|---|---|---|
| q | -0.470047 | -0.466895 | 0.036050 | 0.036225 |
| p | 1.792015 | 1.799563 | 0.667216 | 0.671983 |

Relative errors: q mean 0.68%, q variance 0.48%; p mean 0.42%, p variance 0.71% (all inside the 5% / 10% gates). The `dW` hook (Nt taken from `dW.shape[1]`, dt = T/Nt) and the `observables` hook (trapezoid path integral, shape `(num_paths,)`) both ran.

## Evaluation

(See the metrics and review blocks below.)

<metrics>
kind: sde
score: 9
certified: false
provenance: analytic
estimated_rel_error: 7.094e-03
error_is_estimate: false
observed_order: 1.0173
observed_order_se: 0.0108
order_floor: 1.0000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
strong_rel_err: 0.0287
strong_rel_err_se: 1.527e-04
strong_rel_err_component: 1
design_order: 1.0000
scheme_family: implicit
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: unavailable
invariant_max_drift: 1.194e-03
invariant_worst_headroom: 0.9231
reference_outcome: validated
mean_rel_err: 6.752e-03
variance_rel_err: 7.094e-03
moment_bias_binding: mean
moment_bias_frac_tol: 0.0860
moment_bias_actionable: false
order_check: true
mc_se_rel: 4.493e-03
resolved: true
dynkin_z: 30.3178
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: eb1b609bde23d5731c0b451361896b205134b274c6ba60a8ea233e1e7d215679
solve_time_s: 0.1500
wall_time_s: 31.5500
cost_source: cpu
</metrics>

<review score=9>

**Score: 9/10**

### Numerical Accuracy
- provenance:              analytic  (reference route: analytic_moments, validated against the spec)
- mean_rel_err:            q 6.75e-03 (+/- 2.53 x 6.0e-04 abs se), p 4.19e-03 (+/- 2.53 x 2.58e-03 abs se); tol 5e-2
- variance_rel_err:        q 4.85e-03 (+/- 2.53 x 1.61e-04 abs se), p 7.09e-03 (+/- 2.53 x 3.00e-03 abs se); tol 1e-1 (2.53 is the family-wise multiplier on ci_mult 2.0)
- mc_se_rel:               4.49e-03   resolved: true (CI width rel 8.9e-03 against bar 5e-02, both components)
- functional_rel_error:    not applicable, the spec declares no gated functional
- observed_order:          1.017 +/- 0.011 (expected 1.0, one-sided below the declared floor; median-based 1.008). strong_rel_err 2.87e-02 +/- 1.5e-04, not gated
- weak_order:              1.87, out of band above the theoretical 1.0; reported, never scored on (second-order moments on a linear system, as the plan predicts)
- dynkin:                  unresolved (not a failure). Single seed, not replicated (not marginal). q*p: z = -0.59, p^2: z = 0.16, both pass. tanh(q): z = 30.3 against bar 2.43, extrapolation abstained
- terminal_law:            KS 1.73e-03 (q) and 1.45e-03 (p) against critical 2.83e-03, standardized, pass
- shape:                   q skewness 0.0098 +/- 0.0077, excess kurtosis -0.0010 +/- 0.0158; p skewness -0.0007 +/- 0.0078, excess kurtosis 0.0186 +/- 0.0157 (all consistent with Gaussian)
- constraints:             finite: pass (gate). estimator_stability invariant: pass, drift 1.19e-03 against 1e-02. cross_moment, surrogate, functionals: unavailable (nothing declared)
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. Everything measurable passed, but the generator check never reached a verdict, so the score is 9 and certified is false (uncertified, not suspect).

The solver ran cleanly and consumed the supplied `dW`. Both reported moments agree with the kernel's recomputation to 1e-14 (so `ddof=1` is right), and the plan's declared `scheme_family: implicit` matches the code. No exact-sampling declaration is needed or applicable. The strong order of 1.017 +/- 0.011 is the order expected for additive noise.

### Feedback for solver
- This is not a solver defect. The only thing between this plan and a 10 is `dynkin_ok: "unresolved"`, and it is a property of the test function, not of your scheme. For `tanh(q)` the extrapolated generator residual moves from 2.31e-07 to 1.15e-08 between the two ladder windows, which is 71 paired standard errors. The levels are not in one asymptotic regime, so the check declined to decide. The measured extrapolation orders are about 2.0 on all three functions and the raw residuals fall about 4x per halving of dt, so the scheme is converging as designed. The two polynomial functions pass cleanly (z = -0.59, 0.16).
- Do not read `dynkin_z = 30.3` as a failure. It is the abstained function's z, and the check was not scored as a Dynkin failure (that would be the 3 row). The residual is about 1e-8 on a bounded function, so the generator is not in doubt. No code change is indicated. Changing the scheme to chase this would be wrong.
- Moments are 4-15x inside tolerance and MC-resolved, so more paths are not needed. The kernel's bias study finds a small real mean bias (+0.43% relative, 8.6% of the mean tolerance), judged not actionable. Variance bias is 0.23%.
- There is no `expm(A h)` or exact-covariance draw in the step, which is correct given the problem requires a discretised scheme.
- `weak_order_out_of_band` (1.87 against a nominal 1.0) is informational only. Do not alter the declared orders for it.
- If you want to push toward 10, the only lever is on the verification side: a smoother or polynomial third test function in the spec (formulator side) so the Dynkin extrapolation has an asymptotic window. Nothing in `solver.py` needs to change.

</review>
