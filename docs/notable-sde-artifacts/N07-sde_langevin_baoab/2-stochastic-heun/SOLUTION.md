---
id: 2
plan_slug: stochastic-heun
scheme: stochastic-heun
scheme_family: other
strategy: Stochastic Heun (explicit trapezoidal predictor-corrector) on the full 2-D drift with the supplied additive increment; weak order 2 for additive noise, cross-checking a non-splitting second-order discretisation against the moment gates at dt = 0.04.
---

## SDE Reference

State `X = (q, p)`, in that column order (column 0 = q, column 1 = p). The noise is one Brownian channel.

```
dq = p dt
dp = ( -omega^2 q - gamma p ) dt + sqrt(2 gamma / beta) dW
(q, p)(0) = (q_0, p_0)
```

| parameter | value |
|---|---|
| q_0 | 1.0 |
| p_0 | 0.0 |
| omega | 4.0 |
| gamma | 1.0 |
| beta | 1.0 |
| T | 1.0 |

- Drift: `f(X) = [X[:, 1], -omega**2*X[:, 0] - gamma*X[:, 1]]`
- Diffusion matrix (d=2, m=1): `G = [[0], [sqrt(2*gamma/beta)]]`. It is constant (additive noise), so the Milstein correction is identically 0.

Exact terminal moments at T = 1 (Tier A reference, from `analytic_moments`; documentation values):

| moment | value |
|---|---|
| mean q | -0.46689475052370816 |
| var q | 0.03622547864892864 |
| mean p | 1.7995627626756276 |
| var p | 0.6719833571773481 |
| cov(q, p) | 0.01265010209690913 |

## Numerical Scheme

- **Method**: stochastic Heun, i.e. the explicit trapezoidal predictor-corrector with additive noise. With `h = dt` and `dW_n = dW[:, n]`:
  ```
  predictor:  X* = X_n + f(X_n) h + G dW_n
  corrector:  X_{n+1} = X_n + 0.5 (f(X_n) + f(X*)) h + G dW_n
  ```
  The **same** `dW_n` is used in both stages. `G dW_n` adds `sqrt(2 gamma/beta) * dW_n` to p only.
- **dt**: 0.04 (given by the contract; `Nt = max(1, round(T/dt)) = 25`, then `dt = T/Nt`)
- **num_paths**: 100000
- **seed**: 42 (`np.random.default_rng(seed)` only)
- **T**: 1.0
- **Milstein correction**: not applicable (it vanishes for additive noise)
- **Expected strong order**: 1.0 (additive noise)
- **Expected weak order**: 2.0 (Heun is weak order 2 under additive noise). The spec's floor is 1.0, and a reading that is indeterminate at the noise floor does not gate.
- **Reference for scoring**: analytic moments (`analytic_moments.mean_X/variance_X/mean_Y/variance_Y`; `verification.moment_ode` closes exactly)

## Implementation Notes

1. **Step, do not sample.** problem.md requires a discretised time-stepping scheme and forbids drawing the exact Gaussian transition or terminal law. Heun is a genuine discretisation and consumes only the supplied increments. Do **not** declare `exact_sampling_components`.
2. **Expected accuracy.**
   - The deterministic amplification of RK2 on the oscillator has modulus about `1 + (omega h)^4/8`, which is negligible growth.
   - The phase error is about `(omega h)^3/6` per step, about 0.017 rad over 25 steps.
   - So the mean of q should land at roughly 2–3% error. That is inside the 5% gate but with less margin than a symplectic splitting.
   - Variances should be well inside 10%.
   - If the evaluator reports a mean of q near the gate, the cause is the phase error of the non-symplectic deterministic part, not MC noise. MC relative s.e. on the mean of q is about 0.0004 at 100 000 paths.
   - **dt is fixed by the contract and must not be changed.**
3. **Hooks.**
   - `dW`: pre-generated increments of shape `(num_paths, Nt)` for the single channel, already scaled by `sqrt(dt)`. Take `Nt = dW.shape[1]` and `dt = T/Nt`. Accept a trailing singleton axis `(num_paths, Nt, 1)` defensively by reshaping.
   - Without `dW`: draw `dW = sqrt(dt) * rng.standard_normal((num_paths, Nt))` from `rng = np.random.default_rng(seed)`.
   - `observables`: `{name: phi}`, where `phi` takes the `(num_paths, 2)` state array. Accumulate `0.5*(phi(X_prev) + phi(X))*dt` per step and return it under `path_integrals`.
   - The Dynkin test functions are `X[:, 0]`, `X[:, 1]**2` and `tanh(X[:, 0] + X[:, 1])`.
4. **Return contract.**
   - `terminal_paths`: shape `(num_paths, 2)`, columns `(q, p)`.
   - `empirical_mean`: list of 2 floats.
   - `empirical_variance`: list of 2 floats with `ddof=1`.
   - Also return `num_paths`, `dt` and `T`.
   - Keep the signature `solve_sde(num_paths, dt, T, seed=42, dW=None, observables=None)`.
   - No work at import. Use numpy and the standard library only.
5. **No variance reduction, no boundary.** The problem grades moments, not a payoff. Antithetic pairing would conflict with consuming the supplied `dW`. No barrier or absorbing boundary is declared. No component is conditionally tractable.
6. **Stability.** omega*dt = 0.16 is well inside RK2's imaginary-axis stability region. The `finite` and `shape` constraints are gated.

## Results

Run: `solve_sde(num_paths=100000, dt=0.04, T=1.0, seed=42)` (Nt = 25, stochastic Heun, same `dW_n` in predictor and corrector, consuming the supplied increments).

Revision after the evaluator's Dynkin finding: the q update now takes the trapezoid of the **reported** momenta, `q_{n+1} = q_n + (h/2)(p_n + p_{n+1})`, with `p_{n+1}` the unchanged Heun momentum update (predictor `X*` and the p corrector are as written under Numerical Scheme). Previously q used the predictor momentum `p*`, so `q_{n+1} - q_n` was not the trapezoid of the p values the scheme reports, a deterministic O(h^2) inconsistency in the one test function that is linear in q. This is a sequential (Gauss-Seidel) form of the corrector and differs from the Numerical Scheme text, which writes the q row of the corrector with `f(X*)`. The Numerical Scheme section is not mine to edit and should be brought in line.

| moment | empirical | exact | rel. error |
|---|---|---|---|
| mean q | -0.463108 | -0.466895 | 0.81% |
| mean p | 1.805591 | 1.799563 | 0.33% |
| var q | 0.036108 | 0.036225 | 0.32% |
| var p | 0.671209 | 0.671983 | 0.11% |
| cov(q, p) | 0.012766 | 0.012650 | (diagnostic) |

All four graded moments are inside the 5% mean and 10% variance gates, with the mean of q error down from 2.17% to 0.81%. The `dW` hook (Nt taken from `dW.shape[1]`) and the `observables` hook (`path_integrals` of shape `(num_paths,)`) were both exercised and work.

Local replay of the kernel's Dynkin check (`dynkin.check` on the dt, dt/2, dt/4 CRN ladder at 100 000 paths, the kernel's own central-difference generators): seed 42 passes with z = (-0.43, -0.25, 0.87) against the 2.43 bar, where the previous version read z = -384.1 on `X[:, 0]` in the same replay. Two other seeds: seed 7 passes (z = -0.13, 1.28, 1.43); seed 1 fails on `X[:, 1]**2` at z = -3.08 with the other two small, which is the seed-sensitive martingale term section 20 describes, not the q residual. The strong order is unchanged in structure (same noise coefficient, same shared increment).

## Evaluation

(Filled by evaluator)

<metrics>
kind: sde
score: 10
certified: true
provenance: analytic
estimated_rel_error: 8.111e-03
error_is_estimate: false
observed_order: 1.0170
observed_order_se: 0.0108
order_floor: 1.0000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 1.0000
scheme_family: other
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: unavailable
invariant_max_drift: 1.215e-03
invariant_worst_headroom: 1.2166
invariants_not_reported: ['cross_covariance']
reference_outcome: validated
mean_rel_err: 8.111e-03
variance_rel_err: 3.237e-03
order_check: true
mc_se_rel: 4.493e-03
resolved: true
dynkin_z: 0.8721
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 2d6953c04519c664d146cb5c72bcb2def82837e33cdc448da297531d935ed55f
solve_time_s: 0.1540
wall_time_s: 15.3500
cost_source: cpu
</metrics>

<review score=10>

**Score: 10/10**

Bound by the rubric: every checked moment is within tolerance of the exact moments, the estimates are MC-resolved, and `dynkin_ok` reached a verdict and passed. Tier A, certified. Nothing was capped.

### Numerical Accuracy
- provenance:              analytic  (route analytic_moments, reference_outcome validated; `surrogate_ok` is unavailable because Path A needed no surrogate, which counts neither for nor against the plan)
- mean_rel_err:            q 8.11e-03, p 3.35e-03, against a 5e-02 bar. The MC se of the mean is 6.0e-04 (q) and 2.6e-03 (p), so q sits about 6.3 se (2 x se = 1.2e-03 is the interval) off the exact value and p about 2.3 se. The q offset is a real but small discretization bias, 6x inside the gate.
- variance_rel_err:        q 3.24e-03, p 1.15e-03, against 1e-01. se of the variance is 1.6e-04 (q) and 3.0e-03 (p), so the errors are 0.7 se and 0.3 se: indistinguishable from MC noise. Family-wise multiplier 2.53 over 4 tests.
- mc_se_rel:               4.49e-03   resolved: true (CI width 8.91e-03 on q and 8.98e-03 on p, against the 5e-02 bar, ci_mult 2.0). MC noise is not the limit.
- functional_rel_error:    not applicable. The spec declares no gated functional (`functionals_ok: unavailable`), so there is no bias/sampling split to report.
- observed_order:          1.017 +/- 0.011 on p (median-based 1.008) and 1.022 +/- 0.011 on q, expected 1.0, one-sided below the floor. The scheme consumes the supplied `dW`, and the strong order clears its floor.
- weak_order:              p 2.24, q 1.85. Above the declared floor of 1.0, so `weak_order_out_of_band` is flagged and never scored. It matches Heun's design weak order 2 on additive noise.
- dynkin:                  ok. z = (-0.43, -0.25, 0.87) on (`X[:, 0]`, `X[:, 1]**2`, `tanh(q+p)`) against a bar of 2.43. Not replicated and not marginal: |z - bar| = 1.56 is outside the 0.7 margin, so one seed settled it (seeds: 42 only). On `X[:, 0]` the residuals are at roundoff (about 1e-16 at every level), so the finer difference sat inside its own noise and the extrapolation used the assumed order 1 (`assumed_1`). That is a clean zero, not a degenerate fit. The other two functions measured orders 2.07 and 2.02, ratios 4.2 and 4.05.
- terminal_law:            KS q 1.91e-03 and p 1.62e-03, against a critical value of 2.83e-03 (parametric bootstrap, standardized). Passes on both.
- shape:                   q skewness 0.0099 +/- 0.0077, excess kurtosis -0.0007 +/- 0.0158. p skewness -0.0007 +/- 0.0079, excess kurtosis 0.0188 +/- 0.0157. All within about 1.3 se of Gaussian.
- constraints:             finite true, shape true (both gates), estimator_stability true (drift 1.21e-03 against 2e-02, 1.22 decades of headroom). `cross_covariance` is `not_reported`: a non-gate diagnostic the kernel does not evaluate, which is not a failure.
- reported_moments_agree:  true, worst relative difference 9.4e-15, so `ddof=1` and the arithmetic are right.
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric: every checked moment is within tolerance of the exact moments, and the estimates are MC-resolved. Headroom 1.05 decades, quality 0.72.

Skip vocabulary: `surrogate_ok`, `cross_plan_agreement`, `functionals_ok`, `cross_check_ok` are `unavailable` (no surrogate needed on Path A, no sibling comparison on file, no declared functional or ordering). None of these is a pass or a failure.

### What changed since the previous pass
The previous pass of this plan scored 3 on `dynkin_ok: false`, with z = -384 on `X[:, 0]` alone. The revision fixed it, and the cause was the one that review pointed to as structural, not the Richardson-remainder hypothesis it also offered. The old q update took the trapezoid of the predictor momentum p*, so q_{n+1} - q_n was not the trapezoid of the reported p values. The generator of phi = q is p, so Dynkin integrates exactly that quadrature. The new sequential form removes the inconsistency, and the `X[:, 0]` residual is now at roundoff. The error on the mean of q fell from 2.17% to 0.81% as a side effect. That is the Dynkin check doing its job. It found a real O(dt^2) defect in the scheme that the moment checks tolerated.

What a Dynkin pass does not establish: the generator is built from the spec's own drift and diffusion, so a drift mis-transcribed from `problem.md` would fool it. Here the drift and the sqrt(2 gamma/beta) coefficient match requirements R1 and R2 and the closed form, so that risk is covered by the moment comparison against the validated analytic reference.

### Feedback for solver
- None required for the score. All checks that reached a verdict passed and none is marginal except the mean of q, below.
- The mean of q is the one place a bias is visible, at 8.1e-03 relative and 6.3 se, 6x inside the 5% bar. It is the non-symplectic phase error of an RK2-type step, as your Implementation Note 2 predicted. It is no risk at the graded dt = 0.04.
- Housekeeping, which does not affect the score: `solver.py` advances q by the trapezoid of the reported momenta (a sequential, Gauss-Seidel corrector), while the Numerical Scheme section of `SOLUTION.md` still writes the q row of the corrector with `f(X*)`. Your Results section already says so. The text and the code should be brought in line so the plan describes the scheme it runs. I judged this a documentation lag, not a mislabelled scheme: both are second-order Heun-type steps that consume the supplied increment, so I applied no cap.
- Declarations are right as they stand. `scheme_family: other` with no `exact_sampling_components` is correct, because the step consumes the supplied increments and the strong order reads 1.02. Naming a more specific family would only turn the one-sided order band two-sided, which is not what binds.

</review>
