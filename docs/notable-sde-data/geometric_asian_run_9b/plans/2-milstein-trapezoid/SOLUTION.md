---
id: 2
plan_slug: milstein-trapezoid
scheme: milstein
scheme_family: milstein
levy_area_treatment: not applicable (scalar state, single noise — no iterated integral needed)
strategy: Step S itself with the scalar Milstein scheme (strong order 1.0 via the 0.5*sigma^2*S_n*(dW^2 - dt) correction) at the supplied dt, and discretise the geometric average with the trapezoidal rule, whose O(dt^2) quadrature error is the only remaining bias at Nt = 25.
---

This plan is the pool's **discretised cross-check in the asset coordinate**: it steps `S` directly,
in the variable the problem is posed in, with the highest-order pathwise scheme the problem admits.
Agreement between it and a plan that steps `ln S` exactly is a genuine cross-family circularity
break, because the two schemes make different errors — a Milstein step in `S` carries a real
`O(dt**(3/2))` pathwise remainder and a weak-order-one bias, while a log step carries none.

`exact_sampling_components` is **not** declared and must not be: every step consumes the supplied
Brownian increment `dW_n`, and the Milstein correction is built from `dW_n**2` — the same increment,
not fresh randomness. Declaring it would forfeit the order check that is this plan's main piece of
evidence. No `boundary_treatment` is declared: the spec declares no barrier and no absorbing
boundary.

## SDE Reference

Scalar Black–Scholes asset (GBM), copied from `problem_spec.json`:

```
dS(t) = r * S(t) * dt + sigma * S(t) * dW(t),    S(0) = S_0
```

| parameter | value |
|---|---|
| `S_0` | 100.0 |
| `r` | 0.05 |
| `sigma` | 0.2 |
| `K` | 100.0 |
| `T` | 1.0 (`T0 = 0.0`) |

Noise structure: multiplicative. `state_dimension = 1`, `noise_dimension = 1`, `milstein_eligible:
true`. Drift `f(S) = r*S`; diffusion `g(S) = sigma*S`; `dg/dS = sigma`.

**Graded path functionals:**

```
A                    = exp( (1/T) * int_0^T ln S(t) dt )
call_price           = exp(-r*T) * E[ (A - K)^+ ]
geo_average_variance = Var[A]        (sample variance, ddof=1)
```

Closed forms, from `problem_spec.json → path_functionals`:

```
m  = np.log(S_0) + (r - 0.5*sigma**2)*T/2   = 4.6201701859880915
s2 = sigma**2 * T / 3                        = 0.013333333333333336
ln A ~ Normal(m, s2)   exactly
call_price           = 5.546818633789223
geo_average_variance = 140.17051781155084
E[A]                 = 102.19030933272934
```

Terminal moments of `S(T)` (exact, GBM):

```
exact_mean     = S_0 * np.exp(r*T)                                    = 105.1271096376
exact_variance = S_0**2 * np.exp(2*r*T) * (np.exp(sigma**2*T) - 1)    = 451.0288078158
```

## Numerical Scheme

- **Method**: scalar Milstein on `S`, trapezoidal quadrature for the log-average
- **Update** (`dW_n = sqrt(dt)*Z_n`, and note the correction uses `S_n`, never `S_{n+1}`):

  ```
  S_{n+1} = S_n + r*S_n*dt + sigma*S_n*dW_n + 0.5*sigma**2*S_n*(dW_n**2 - dt)
  ```

- **Milstein correction**: `0.5 * sigma**2 * S_n * (dW**2 - dt)` — taken verbatim from
  `problem_spec.json → milstein_correction`, with `g(S) = sigma*S`, `dg/dS = sigma`
- **dt**: `0.04` — the contract step. `Nt = max(1, round(T/dt)) = 25`, then `dt = T/Nt = 0.04`.
  Use the supplied `dt`; do not hardcode another
- **num_paths**: `400000` (and again at `100000`, same `dt`, `T`, `seed`)
- **T**: `1.0`
- **seed**: `42`, via `np.random.default_rng(seed)` only
- **Expected strong order**: **1.0** (`verification.expected_strong_order_milstein`). This is the
  sharpest check available on this plan: the kernel runs the scheme at three step sizes on a shared
  Brownian path and reads the pathwise rate. A measured ~0.5 means the correction term is broken —
  almost always a sign error, or applying it with `S_{n+1}` instead of `S_n`
- **Expected weak order**: `1.0` (`evaluation_thresholds.expected_weak_order`)
- **Reference for scoring**: **analytic moments** for `S(T)` (`analytic_moments`, validated by the
  exactly-closing moment ODE in `verification.moment_ode`) **and** the closed-form functional values
  above

## Implementation Notes

**The quadrature, not the step, is what decides the price.** `problem_spec.json → verification.note`
is explicit: the discretization error of this problem lives almost entirely in the quadrature of
`int_0^T ln S(t) dt`. Use the **trapezoidal rule**, which is `O(dt**2)`:

```
log_avg = (1/T) * sum_{n=0}^{Nt-1} 0.5*dt*( np.log(S_n) + np.log(S_{n+1}) )
A       = np.exp(log_avg)
```

A left-endpoint Riemann sum is `O(dt)`: it omits half of each step's drift and the whole of the last
increment's contribution, and at `Nt = 25` that is a percent-scale error on `call_price` — outside
the 1% gate. `problem.md` warns about this in as many words. Accumulate `log_avg` inside the
stepping loop rather than storing the whole `(num_paths, Nt+1)` path matrix: at 400000 paths and 25
steps a full matrix is ~80 MB of float64, which is survivable but pointless. Add
`0.5*dt*(log(S_prev) + log(S_new))` each step.

**Estimators:**

```
payoff               = np.exp(-r*T) * np.maximum(A - K, 0.0)
call_price           = float(np.mean(payoff))
geo_average_variance = float(np.var(A, ddof=1))
functional_se        = {"call_price": float(np.std(payoff, ddof=1)/np.sqrt(num_paths)), ...}
```

Report `functional_se` for both functionals — for `geo_average_variance` use a jackknife or the
delta-method standard error of the sample variance. The kernel reads these.

**Positivity.** A Milstein step on `S` itself can in principle produce `S <= 0` at large `dt`; at
`dt = 0.04` with `sigma = 0.2` that is a ~1e-30 event (`verification.constraints.positivity`), so a
breach signals a wrong step, not bad luck. Do **not** silently clamp: `np.log` of a non-positive `S`
would produce NaN and poison `log_avg` and the whole estimate. Assert finiteness and strict
positivity of the stepped path and let a breach surface as the bug it is. `positivity` and `finite`
are both **gated** constraints — a violation caps the score at 4 whatever the prices read.

**Expected accuracy.** The Milstein step's weak error on the asset path at `dt = 0.04` is small, and
the trapezoidal quadrature's `O(dt**2)` bias on `ln A` is of order `sigma**2*dt**2/(12*T) = 5.33e-6`
in variance terms — about 0.04% of `s2 = 0.01333`, which propagates to well under 0.1% on
`call_price` and on `geo_average_variance`. Both comfortably inside the 1% and 3% gates, with the
0.22% relative sampling standard error at 400000 paths the binding term. That is the expected
posture here: **sampling-limited, not bias-limited.** If the evaluator reports
`functional_limited_by: bias`, the quadrature is the first thing to look at, not `num_paths`.

**Sampling budget.** `problem_spec.json` records the discounted payoff's relative standard deviation
as about 1.39 → roughly 0.22% relative standard error on `call_price` at 400000 paths, 0.44% at
100000. Both inside the 1% gate, but with no slack for a biased quadrature on top. Do not reduce the
path count.

**Two-path-count check (R26).** `solve_sde` is called at 400000 and at 100000 with the same seed.
Draw the increments in a single `rng.standard_normal((num_paths, Nt))` call so the two runs see
genuinely different samples; the two reported values must differ and must agree within the implied
sampling noise. Never cache or hardcode a functional value.

**Contract details that are hard-gated** (`problem_spec.json → requirements`, ledger-audited):
`solve_sde(num_paths, dt, T, seed=42)` signature exactly; `Nt = max(1, round(T/dt))` then
`dt = T/Nt`; seed only through `np.random.default_rng(seed)`; return `terminal_paths` as a 1-D array
of shape `(num_paths,)`, `empirical_mean`, `empirical_variance` with `ddof=1`, and a `functionals`
dict carrying the floats `"call_price"` and `"geo_average_variance"`; nothing executed at module
load outside `if __name__ == "__main__"`; standard library and numpy only.

**Evaluator hooks.** Accept `dW` and `observables` keyword arguments with `None` defaults. When `dW`
is supplied, take `Nt = dW.shape[1]`, set `dt = T/Nt`, consume the increments exactly and draw no
randomness of your own — the strong-order measurement depends on it, and a scheme that ignores `dW`
reads a strong order near zero and fails the order study. Accumulate `observables` path integrals
with the trapezoid rule for the Dynkin check; the spec's test functions are `X`, `X**2`,
`np.tanh(X/100.0)`.

**Invariants the kernel will check.** `martingale`: `mean(exp(-r*T)*S(T))` must equal `S_0 = 100`
within sampling error (ungated, `ci_mult: 2.0`) — the cheapest falsifier of a mis-signed or
mis-scaled drift. `estimator_stability` is an 8-batch batch-means scatter, so compute the
functionals from the same paths that are returned.

## Results

Run: `solve_sde(num_paths=400000, dt=0.04, T=1.0, seed=42)` (Nt=25, scalar Milstein on S, trapezoid on ln S).

- empirical_mean: 105.124795 (exact 105.127110)
- empirical_variance: 448.626880 (exact 451.028808)
- call_price: 5.530083, SE 0.007365 (closed form 5.546819); now uses an S_T control variate (known scheme-exact mean S_0(1+R dt)^Nt, beta fitted in-sample), SE down from 0.01216. Milstein step, trapezoid quadrature and dW handling unchanged.
- geo_average_variance: 139.620550, SE 0.3288 (closed form 140.170518)
- At 100000 paths, same seed: call_price 5.505915, geo_average_variance 139.114539
- dW and observables hooks both exercised without error.

## Evaluation

(Filled by evaluator)

<metrics>
kind: sde
score: 10
certified: true
provenance: analytic
estimated_rel_error: 5.325e-03
error_is_estimate: false
observed_order: 0.9882
observed_order_se: 7.836e-03
order_floor: 1.0000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 1.0000
scheme_family: milstein
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: true
invariant_max_drift: 2.202e-03
invariant_worst_headroom: 1.4429
reference_outcome: validated
mean_rel_err: 2.202e-05
variance_rel_err: 5.325e-03
functional_name: call_price
functional_rel_tol: 0.0100
functional_rel_error: 1.666e-03
functional_bias_rel: 3.339e-04
functional_se_rel: 1.332e-03
functional_limited_by: sampling
moment_bias_binding: variance
moment_bias_frac_tol: 0.0418
moment_bias_actionable: false
order_check: true
mc_se_rel: 2.581e-03
resolved: true
dynkin_z: 0.3346
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 13e5c77be47c3cdba7125fbcc7047a96baa9eb825c54fabce1befaffc7ff2e38
solve_time_s: 0.3800
wall_time_s: 40.2000
cost_source: cpu
</metrics>

<review score=10>

**Score: 10/10**

### Numerical Accuracy
- provenance:              analytic  (analytic_moments, validated by the moment ODE; Tier A)
- mean_rel_err:            2.20e-05 +/- 2.0 x 0.0335 abs (SE; 105.1248 vs exact 105.1271), inside the 5% tolerance
- variance_rel_err:        5.33e-03 +/- 2.0 x 1.158 abs (SE; 448.63 vs exact 451.03), inside the 10% tolerance
- mc_se_rel:               2.58e-03   resolved: true
- functional_rel_error:    1.67e-03 against functional_rel_tol 1.0e-02  (bias 3.34e-04 + sampling 1.33e-03 -> limited by sampling, binding functional call_price). geo_average_variance also clean (ladder bias 8.9e-04 against its 3% gate)
- observed_order:          0.988 +/- 0.0078 (expected 1.0, one-sided below the declared floor); median-based 0.969
- weak_order:              1.004 (x and x2 test functions; tanh at the noise floor and not used)
- dynkin:                  ok, z = 0.33 against bar 2.43 (single seed, not replicated: |z - bar| = 2.09 is outside the 0.7 margin, so the verdict was not seed-sensitive). Residual extrapolation order 0.98 on all three test functions
- terminal_law:            KS 8.78e-04 vs critical 1.45e-03, standardized, passes
- shape:                   skewness 0.610 +/- 0.0053 (lognormal 0.614), excess kurtosis 0.664 +/- 0.0228 (lognormal 0.678)
- constraints:             finite pass, positivity pass, martingale pass (drift 2.2e-03 vs bar 6.4e-02), estimator_stability pass (3.6e-04 vs 1e-02). No gate violation.
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric -- every checked moment is within tolerance of the exact moments, and the estimates are MC-resolved

### Notes
- Dynkin now reaches a verdict (the earlier "unavailable" was the spec not declaring an initial state; the spec now carries `initial_condition.value`). It still does not establish the drift was transcribed correctly, since the generator is built from the spec's own coefficients. The martingale invariant and the exact moments cover that gap.
- `surrogate_ok: unavailable` is expected on Path A; the analytic moments were the reference. `cross_check_ok` and `cross_plan_agreement` are unavailable (no declared ordering, no sibling agreement), which is a skip and not a failure.
- `moment_bias_actionable: false`. The graded-step variance bias is 4.2% of its tolerance. It is real (72 SE) but far inside the gate.
- Source check: solver.py matches SOLUTION.md (Milstein on S with the correction on S_n, trapezoid on ln S, no clip, ddof=1, dW consumed). `exact_sampling_components` is correctly not declared. No agent cap.

### Feedback for solver
- None. Every graded check passed, and the strong order of 0.988 +/- 0.008 confirms the Milstein correction and the use of the supplied dW.

</review>
