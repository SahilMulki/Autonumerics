---
id: 3
plan_slug: euler-maruyama-trapezoid
scheme: euler-maruyama
scheme_family: euler-maruyama
levy_area_treatment: not applicable (scalar state, single noise)
strategy: Plain Euler-Maruyama on S at the supplied dt with trapezoidal quadrature of the log-average — the lowest-order baseline in the pool, carried to establish how much of the measured accuracy comes from the step scheme and how much from the quadrature rule.
---

This is the pool's **baseline**, ranked last on purpose. It is the plan a reader would write first,
and it is here to answer one question no other plan in the pool can: how much of the accuracy comes
from the quadrature rule and how much from the step scheme. It shares its trapezoidal quadrature
with plan 2 and differs only in the step, so the difference between the two prices isolates the
Euler step's weak bias at `dt = 0.04`.

It is expected to **pass** both functional gates — Euler–Maruyama's weak order is 1.0 and at
`Nt = 25` on GBM with `sigma = 0.2` its weak bias on the log-average is modest — but it is expected
to be the **least accurate** of the three and to spend more of the 1% `call_price` budget on bias
than either other plan. If anything in this pool misses a gate, the prior is that it is this plan.

`exact_sampling_components` is **not** declared and must not be: every step consumes the supplied
`dW_n`. No `boundary_treatment` is declared: the spec declares no barrier and no absorbing boundary.

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

Noise structure: multiplicative. `state_dimension = 1`, `noise_dimension = 1`.
Drift `f(S) = r*S`; diffusion `g(S) = sigma*S`; `dg/dS = sigma`.

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

- **Method**: plain Euler–Maruyama on `S`, trapezoidal quadrature for the log-average
- **Update** (`dW_n = sqrt(dt)*Z_n`):

  ```
  S_{n+1} = S_n + r*S_n*dt + sigma*S_n*dW_n
  ```

- **Milstein correction**: **not applied** — this plan is deliberately the uncorrected baseline.
  (For reference, the correction this plan omits is `0.5*sigma**2*S_n*(dW**2 - dt)`.)
- **dt**: `0.04` — the contract step. `Nt = max(1, round(T/dt)) = 25`, then `dt = T/Nt = 0.04`.
  Use the supplied `dt`; do not hardcode another
- **num_paths**: `400000` (and again at `100000`, same `dt`, `T`, `seed`)
- **T**: `1.0`
- **seed**: `42`, via `np.random.default_rng(seed)` only
- **Expected strong order**: **0.5** (`verification.expected_strong_order`). The kernel measures
  this with common random numbers across three step sizes; on GBM the reference measurement is 0.488.
  A reading near **zero** would mean the solver is not consuming the supplied `dW` at all
- **Expected weak order**: `1.0` (`evaluation_thresholds.expected_weak_order`)
- **Reference for scoring**: **analytic moments** for `S(T)` (`analytic_moments`, validated by the
  exactly-closing moment ODE in `verification.moment_ode`) **and** the closed-form functional values
  above

## Implementation Notes

**The quadrature is still the dominant error, even here.** Use the **trapezoidal rule**, `O(dt**2)`:

```
log_avg = (1/T) * sum_{n=0}^{Nt-1} 0.5*dt*( np.log(S_n) + np.log(S_{n+1}) )
A       = np.exp(log_avg)
```

A left-endpoint Riemann sum is `O(dt)` and moves `call_price` by percent at `Nt = 25` — outside the
1% gate — which is the trap `problem.md` is warning about. Pairing a first-order quadrature with a
first-order step would compound two biases of the same sign. Accumulate `log_avg` inside the loop
rather than storing the full `(num_paths, Nt+1)` path matrix.

**Positivity is a real risk in this scheme and it is gated.** Unlike a log step, a plain
Euler–Maruyama update on `S` can return `S <= 0` when `sigma*dW_n < -(1 + r*dt)`; at `dt = 0.04`
with `sigma = 0.2` that needs a ~25-sigma increment, a ~1e-30 event per step, so across 400000×25
steps it should never fire (`verification.constraints.positivity` says exactly this: a breach
signals a wrong step, not bad luck). It must still be **checked**, because a single non-positive `S`
makes `np.log(S)` NaN and poisons that path's `log_avg`, the mean, and both functionals. Assert
`np.all(S > 0)` and `np.all(np.isfinite(S))` each step or at the end; surface a breach rather than
clamping it away. `positivity` and `finite` are both gated — a violation caps the score at 4 no
matter how good the prices look.

**Estimators:**

```
payoff               = np.exp(-r*T) * np.maximum(A - K, 0.0)
call_price           = float(np.mean(payoff))
geo_average_variance = float(np.var(A, ddof=1))
functional_se        = {"call_price": float(np.std(payoff, ddof=1)/np.sqrt(num_paths)), ...}
```

Report `functional_se` for both functionals (jackknife or delta-method for the variance). No
variance reduction is applied in this plan — that is part of what makes it the baseline, and the
`functional_se` it reports is the unreduced reference the pool's variance-reduced plan is measured
against.

**Expected accuracy and where the budget goes.** Sampling gives roughly 0.22% relative standard
error on `call_price` at 400000 paths (the discounted payoff's relative standard deviation is about
1.39, per `problem_spec.json`) and 0.44% at 100000. The Euler step adds a weak bias on top of that
which the other plans do not carry, so this plan has the least headroom inside the 1% gate. If the
evaluator reports `functional_limited_by: bias` or a `call_price` error above ~0.5%, the diagnosis
is the **step scheme**, not the path count — and the fix is the Milstein correction or a log-exact
step, which are what the other plans in this pool are. Raising `num_paths` would not help and in any
case the contract fixes it at 400000; do not reduce it.

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
randomness of your own — the strong-order measurement depends on it. Accumulate `observables` path
integrals with the trapezoid rule for the Dynkin check; the spec's test functions are `X`, `X**2`,
`np.tanh(X/100.0)`.

**Invariants the kernel will check.** `martingale`: `mean(exp(-r*T)*S(T))` must equal `S_0 = 100`
within sampling error (ungated, `ci_mult: 2.0`). Euler–Maruyama on GBM is in fact exactly
martingale-preserving in the mean (`E[S_{n+1}] = (1 + r*dt)*E[S_n]`, so
`E[S_T] = S_0*(1 + r*dt)**Nt = 105.12186` against the exact `105.12711` — a 0.005% shortfall, inside
the sampling error at 400000 paths and inside the 5% mean tolerance). `estimator_stability` is an
8-batch batch-means scatter, so compute the functionals from the same paths that are returned.

## Implementation Notes (refine cycle 2)

**Decision: no scheme change; the Euler baseline's terminal-law bias at the contract dt is irreducible without changing the scheme.**
The evaluator's terminal-law KS test on S(T) fails (D = 2.08e-3 against 1.45e-3 critical) because additive Euler on a
multiplicative equation at dt = 0.04 gives slightly under-skewed paths (skewness 0.585 against 0.614, excess kurtosis 0.603 against 0.678).
That is a dt bias, so more paths cannot remove it. The two fixes the evaluator named, the Milstein correction and a log-exact step, are
exactly plans 2 and 1, so adopting either would make this plan a copy of one of them and remove the baseline's job. The contract fixes dt = 0.04,
so the ceiling stays. `scheme_family` stays `euler-maruyama` (unchanged); no `design_order` is declared and none was added.
`solver.py` is unchanged and still consumes the supplied `dW` exactly (observed strong order 0.496 against 0.5). The graded functionals
are sampling-limited and clean (call_price error 2.2e-03 against a 1e-02 tolerance), and the shape-test failure is the documented price of the baseline.

## Results

Run: `solve_sde(num_paths=400000, dt=0.04, T=1.0, seed=42)` (Nt = 25), plain Euler-Maruyama on S, trapezoid quadrature of ln S.

| quantity | value | reference |
|---|---|---|
| empirical_mean | 105.125757 | 105.127110 |
| empirical_variance | 448.264126 | 451.028808 |
| call_price | 5.539498 (se 0.01210) | 5.546819 |
| geo_average_variance | 139.672441 (se 0.3258) | 140.170518 |

At 100000 paths (same dt, T, seed): call_price 5.530073, geo_average_variance 139.169767.
Terminal paths all positive and finite. The `dW` and `observables` hooks both ran.

## Evaluation

See the metrics and review blocks below.

<metrics>
kind: sde
score: 8
certified: false
provenance: analytic
estimated_rel_error: 6.130e-03
error_is_estimate: false
observed_order: 0.4964
observed_order_se: 0.0112
order_floor: 0.5000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 0.5000
scheme_family: euler-maruyama
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: true
invariant_max_drift: 1.287e-03
invariant_worst_headroom: 1.4436
reference_outcome: validated
mean_rel_err: 1.287e-05
variance_rel_err: 6.130e-03
functional_name: call_price
functional_rel_tol: 0.0100
functional_rel_error: 2.222e-03
functional_bias_rel: 3.829e-05
functional_se_rel: 2.184e-03
functional_limited_by: sampling
moment_bias_binding: variance
moment_bias_frac_tol: 0.0773
moment_bias_actionable: false
order_check: true
mc_se_rel: 2.551e-03
resolved: true
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: f44c6d8bb0440c300dba857a6e08f16883ca721f0e57d389cf1abe27c27b49d0
solve_time_s: 0.3590
wall_time_s: 30.9400
cost_source: cpu
</metrics>

<review score=8>

**Score: 8/10**

### Numerical Accuracy
- provenance:              analytic  (analytic_moments, validated by the moment ODE; certified: false)
- mean_rel_err:            1.29e-05 +/- 2.0 x 3.35e-02 absolute se (0.032% relative); tolerance 5%; passes
- variance_rel_err:        6.13e-03 +/- 2.0 x 1.14 absolute se (0.25% relative); tolerance 10%; passes
- mc_se_rel:               2.55e-03   resolved: true (CI width 0.51% against a 5% bar)
- functional_rel_error:    2.22e-03 against functional_rel_tol 1.0e-02  (bias 3.83e-05 + sampling 2.18e-03 -> limited by sampling, binding functional call_price). geo_average_variance also clean (ladder error 6.8e-04 against 3e-02)
- observed_order:          0.496 +/- 0.011 (expected 0.5, one-sided below the declared floor); median-based 0.503
- weak_order:              indeterminate, at the noise floor (all three test functions rejected as not shrinking above noise)
- dynkin:                  not run. The moments check failed on the terminal-law shape, so the score is capped on that finding and the generator check cannot lift it (cost rule). It never reached a verdict here
- terminal_law:            KS D = 2.08e-03 against critical 1.45e-03 (standardized, parametric bootstrap, 1.44x critical): FAIL
- shape:                   skewness 0.585 +/- 0.0052 (reference 0.614, 5.7 se below), excess kurtosis 0.603 +/- 0.0214 (reference 0.678, 3.5 se below); the two shape moments are reported but not graded
- constraints:             finite true, positivity true (gate, no breach, neg_fraction 0), martingale true (drift 1.29e-03 against bar 6.4e-02), estimator_stability true (3.6e-04 against 1e-02)
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. The variance passes, and the terminal-law shape test did not (KS D = 0.00208 against 0.00145). The shape test is distributional and says nothing about pathwise accuracy

### Feedback for solver
- Headline: the score is bound by `terminal_law_ok: false`, not by anything about the graded functionals. At 400000 paths the KS test is sharp enough to see Euler-Maruyama's own step error in the *shape* of S(T). The paths are slightly less skewed than lognormal (skewness 0.585 against 0.614, excess kurtosis 0.603 against 0.678), which is the signature of an additive-Euler step on a multiplicative equation. The first two moments are still within 0.01% (mean) and 0.25% (variance), so this is a genuine but small dt = 0.04 bias, and the estimates are MC-resolved, so this is not a path-count problem.
- More paths will not help, since the failure is a bias. The variance moment confirms it: the ladder reads the graded variance 0.77% below its extrapolated limit, 8.5 se away (`moment_bias.variance.real: true`, 7.7% of the tolerance, so the kernel flags it not actionable). To pass the shape test, step with something that carries less of this bias: the Milstein correction `0.5*SIGMA**2*S_prev*(dW[:, n]**2 - dt)` (which still consumes `dW`), or step `ln S` with `(R - 0.5*SIGMA**2)*dt + SIGMA*dW[:, n]`, which is exact in law and keeps the same `exp`-free trapezoid on `ln S`. Either is a scheme change. This plan was written as the deliberate baseline, so whether to change it or leave it as the pool's reference is a pool-level choice.
- What is already right: the plan consumes `dW` exactly (observed strong order 0.496 +/- 0.011 against the declared 0.5, so no undeclared-exact-sampling issue and no ignore-`dW` defect), `exact_sampling_components` correctly undeclared, the trapezoid quadrature keeps the functional bias to 3.8e-05 relative on call_price, and both functionals are well inside their gates. Functional error is sampling-dominated (2.18e-03 of 2.22e-03), so the plan's own prior that Euler would spend more of the call_price budget on bias was not borne out. `functional_se` for call_price matches the sampling term.
- SOLUTION.md Results table quotes values that match the kernel's recomputation (reported_moments_agree: true, worst relative difference 0).
- Not run, not failed: the Dynkin check (capped before it could run) and the weak-order estimate (at the noise floor). Neither counts against the plan. A Milstein or log-exact variant that passes the KS test would let Dynkin run and open the 9-10 band.

</review>
