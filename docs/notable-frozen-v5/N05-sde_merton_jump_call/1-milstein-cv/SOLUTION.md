---
id: 1
plan_slug: milstein-cv
scheme: milstein
scheme_family: milstein
strategy: Scalar Milstein on the compensated GBM diffusion part, with dt-independent multiplicative compound-Poisson jumps applied at the end of the step they fall in, and the call price estimated with the terminal state X_T as a control variate whose discrete-scheme mean is known exactly.
---

## SDE Reference

Scalar Ito SDE with multiplicative compound-Poisson jumps (Merton 1976):

    dX(t) = (r - lam*kappa) * X(t-) dt + sigma * X(t-) dW(t) + X(t-) dJ(t),   X(0) = X_0

- `J(t) = sum_{i=1}^{N(t)} (Y_i - 1)`, where `N` is a Poisson process of intensity `lam`, `Y_i = exp(Z_i)`
  and `Z_i ~ N(m, delta^2)` iid, independent of `W` and `N`. At a jump time tau, `X(tau) = Y * X(tau-)`.
- `kappa = E[Y] - 1 = exp(m + delta^2/2) - 1` (derived, not given).
- The diffusion-part coefficients are `drift f(X) = (r - lam*kappa)*X` and `diffusion g(X) = sigma*X`, with `g'(X) = sigma`.
  The jump term comes on top of these.

| parameter | value |
|---|---|
| X_0 | 100.0 |
| r | 0.05 |
| sigma | 0.15 |
| lam (the problem's lambda) | 0.1 |
| m | -0.90 |
| delta | 0.45 |
| kappa | -0.550109023493067 |
| mu_c = r - lam*kappa | 0.1050109023493067 |
| K (strike) | 100.0 |
| T | 0.25 |

Graded quantity: `call_price = exp(-r*T) * E[(X_T - K)^+]`, gated at 1% relative.

Exact supporting moments at T (Tier A): `E[X_T] = X_0 exp(rT) = 101.25784515406345`, and
`Var[X_T] = 147.94671758296317`.

## Numerical Scheme

- **Method**: Milstein (scalar, multiplicative noise) for the diffusion part. Jumps are applied multiplicatively.
- **dt**: 0.01. Use the `dt` passed in: `Nt = max(1, round(T/dt))`, then `dt = T/Nt`, which gives Nt = 25 at the graded call.
- **num_paths**: 400000 (graded). A second graded call uses 100000 with the same dt, T and seed.
- **seed**: 42
- **T**: 0.25
- **Milstein correction**: `0.5 * sigma**2 * X_n * (dW**2 - dt)`. It uses `X_n`, the state at the start of the step.
- **Per-step update** (diffusion part):

      A_n     = 1 + mu_c*dt + sigma*dW_n + 0.5*sigma**2*(dW_n**2 - dt)
      X_{n+1} = X_n * A_n * prod_{jumps i in step n} Y_i

  `A_n > 0` for every `dW_n`, because the quadratic in `dW_n` has negative discriminant (`sigma^2 dt < 2`). So the scheme preserves positivity exactly.
- **Expected strong order**: 1.0
- **Expected weak order**: 1.0
- **Reference for scoring**: analytic moments (mean `X_0 e^{rT}` and the closed-form variance, both checked against
  `verification.moment_ode`) as supporting Tier-A evidence. The graded `call_price` is gated through
  `evaluation_thresholds.functionals.call_price`, and its documented reference is the Merton Poisson-weighted
  Black-Scholes series in `verification.functional_reference`. Use that value only as a sanity check and never
  hard-code it. Dynkin uses `verification.generator`, a moment-equivalent local generator, on `X` and `X**2`.

## Variance Reduction: control variate on X_T

The payoff is strongly correlated with the terminal state, and the discrete scheme's mean of `X_T` is known **exactly**:

    E[X_T^disc] = X_0 * (1 + mu_c*dt)**Nt * exp(lam*T*kappa)

This holds because `E[A_n] = 1 + mu_c*dt` (the `dW` and `dW^2 - dt` terms have mean zero), the steps are independent,
and `E[prod_{i<=N} Y_i] = E[(1+kappa)^N] = exp(lam*T*kappa)` for `N ~ Poisson(lam*T)`. Use this discrete mean, not
`X_0 e^{rT}`. The two differ by about 1.4e-5 relative at dt = 0.01, and using the discrete one keeps the control
variate unbiased **for the scheme**. The CV estimator then has the same expectation as the plain Monte Carlo mean,
so the evaluator's dt ladder measures the same discretization bias.

    P   = exp(-r*T) * max(X_T - K, 0)
    b   = cov(P, X_T) / var(X_T)            # estimated from the same sample (O(1/M) bias, negligible)
    C   = P - b * (X_T - E[X_T^disc])
    call_price    = mean(C)
    functional_se = {"call_price": std(C, ddof=1) / sqrt(M)}

The estimator is a function of `terminal_paths` only, as `problem.md` requires ("estimated from your terminal paths").
Apply it the same way whether `dW` is supplied or not. With `dW` supplied, compute `Nt` from `dW.shape[1]` in the
`E[X_T^disc]` formula. Also record the plain estimate `mean(P)` and its s.e. in **Results** as a diagnostic, but do
**not** put it in the `functionals` dict, which should carry only `call_price`. The expected variance reduction is
roughly `1 - rho^2`, where `rho = corr(P, X_T)`. Report the measured ratio of the s.e.s in Results.

## Implementation Notes

- **RNG layout (required for the order study).** Use `rng = np.random.default_rng(seed)` and `rng_w, rng_j = rng.spawn(2)`.
  The Brownian increments come from `rng_w` (`dW = sqrt(dt) * rng_w.standard_normal((num_paths, Nt))`) and all jump
  randomness from `rng_j`. Supplying `dW` then removes only the Brownian draws and leaves the jump stream unchanged.
- **Draw the jumps independently of dt.** Per path, draw `N ~ Poisson(lam*T)`, then `N` jump times uniform on `[0, T]`
  and `N` log-sizes `Z ~ N(m, delta^2)`, all from `rng_j`. Vectorise with `N = rng_j.poisson(lam*T, num_paths)`,
  `tot = N.sum()`, `times = rng_j.uniform(0, T, tot)`, `Z = rng_j.normal(m, delta, tot)` and
  `pid = np.repeat(np.arange(num_paths), N)`. Then map each time to its step with
  `k = np.minimum((times/dt).astype(int), Nt-1)`. Done this way, the jump path is identical at dt, dt/2 and dt/4.
  A per-step Poisson draw would change the jump path with dt and make the strong-order study read about 0 on a
  correct scheme.
- **Apply jumps sparsely.** About `lam*T*M` = 10,000 jumps over 400k paths. Either accumulate the log-jumps into a
  sparse per-step structure (sort by `k`, then apply `X[pid_k] *= exp(Z_k)` after step `k`) or use `np.add.at` into a
  `(num_paths, Nt)` log-jump array. Both are cheap. The continuous dynamics is linear, so multiplying by `Y` commutes
  with the step and its position inside the step does not change `X_T`. The step index still matters for the
  `observables` path integrals.
- **Do not double-count the compensator.** The drift is already compensated: `(r - lam*kappa)*X` with `kappa < 0`.
  Do not add `-lam*kappa` again and do not drop it. If either happens, the martingale invariant `E[X_T] = 101.2578`
  misses by about 1.4%.
- **`dW` hook**: when `dW` is given, use it exactly, take `Nt = dW.shape[1]`, `dt = T/Nt` and `num_paths = dW.shape[0]`,
  and draw no Brownian randomness of your own. Still draw the jumps from `rng_j`.
- **`observables` hook**: accumulate `int_0^T phi(X_s) ds` by the trapezoid rule per step. Take the end-of-step value
  after any jumps in that step. The kernel's Dynkin check uses the moment-equivalent generator (`f_eff = r X`,
  `g_eff = 0.239384 X`) on `phi in {X, X**2}` only. If a Dynkin failure on `phi = X` reads about 100 s.e. on a correct
  solver, that is the R36 kernel gap described in the spec, not a solver defect.
- **Kink**: the payoff is non-differentiable at the money (`X_0 = K = 100`). The GBM part's smoothing keeps the weak
  order near 1 for this payoff, but expect the price's dt-bias estimate to be noisy relative to the moments.
- **Budget**: 400k x 25 steps is cheap (about 80 MB for `dW`). The plain price's relative s.e. is about 0.2% at 400k
  and 0.4% at 100k, against the 1% gate. The control variate should shrink both.
- **Two path counts**: the 400k and 100k calls must give different `call_price` values that agree within their implied
  noise. A seeded estimator that genuinely uses `num_paths` satisfies this automatically. Never return a constant.
- **No exact sampling.** `problem.md` requires a discretised time-stepping scheme and explicitly says that sampling the
  exactly samplable law (the per-step lognormal or the terminal Poisson-lognormal mixture) is not what is measured. So
  this plan steps the diffusion with Milstein and declares no `exact_sampling_components`. The jump draws are
  auxiliary randomness *in addition to* the consumed `dW`, not a replacement for it. The state still consumes every
  supplied increment.
- Return `terminal_paths` (shape `(num_paths,)`), `empirical_mean`, `empirical_variance` (`ddof=1`),
  `functionals = {"call_price": ...}` and `functional_se = {"call_price": ...}`. Python stdlib and numpy only, with no
  work at import.

## Results

Run: `solve_sde(num_paths, dt=0.01, T=0.25, seed=42)` (Nt = 25).

| num_paths | empirical_mean | empirical_variance | call_price (CV) | functional_se (CV) | plain price | plain s.e. | s.e. ratio CV/plain |
|---|---|---|---|---|---|---|---|
| 400000 | 101.253466 | 147.108858 | 4.374919 | 0.006464 | 4.374019 | 0.008674 | 0.745 |
| 100000 | 101.239946 | 146.236169 | 4.362936 | 0.012919 | 4.357932 | 0.017359 | 0.744 |

Exact moments: mean 101.257845, variance 147.946718 (mean error 0.004%, variance error 0.57%).
The CV price is about 0.4% below the documented reference 4.3912 (about 2.6 s.e.), consistent with dt bias at the kink
plus noise; the two counts agree within noise. The `dW` and `observables` hooks both run.

## Evaluation

(Filled by evaluator)

<metrics>
kind: sde
score: 10
certified: true
provenance: analytic
estimated_rel_error: 5.66e-03
error_is_estimate: false
observed_order: 0.9970
observed_order_se: 1.15e-03
order_floor: 1.00
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
strong_rel_err: 6.45e-05
strong_rel_err_se: 1.65e-07
strong_rel_err_component: 0
design_order: 1.0
scheme_family: milstein
invariants_ok: true
constraints_ok: true
functionals_ok: true
invariant_max_drift: 4.38e-03
invariant_worst_headroom: 1.03
reference_outcome: validated
mean_rel_err: 4.32e-05
variance_rel_err: 5.66e-03
functional_name: call_price
functional_rel_tol: 1.0e-02
functional_rel_error: 1.69e-03
functional_bias_rel: 2.15e-04
functional_se_rel: 1.48e-03
functional_limited_by: sampling
moment_bias_binding: variance
moment_bias_frac_tol: 8.69e-03
moment_bias_actionable: false
order_check: true
mc_se_rel: 6.79e-03
resolved: true
dynkin_ok: true
dynkin_z: 0.655
dynkin_bar: 2.28
dynkin_seeds: 1
dynkin_replicated: false
ci_mult: 2.0
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 4f9e457f715fcdaea46b681901f24b809869262753e8e6ad8725bff315f2ee42
solve_time_s: 0.286
wall_time_s: 9.34
cost_source: cpu
</metrics>

<review score=10>

**Score: 10/10**

### Numerical Accuracy
- provenance:              analytic  (reference route: analytic_moments, validated; Tier A, certified)
- mean_rel_err:            4.32e-05 +/- 2.0 x 1.9e-04 (se_mean 0.0192 absolute on 101.258; family-wise multiplier 2.43)
- variance_rel_err:        5.66e-03 +/- 2.0 x 6.7e-03 (se_variance 0.998 absolute on 147.947); within the 10% tolerance
- mc_se_rel:               6.79e-03   resolved: true (CI width 1.35e-02 against bar 5.0e-02)
- functional_rel_error:    1.69e-03 against functional_rel_tol 1.0e-02  (bias 2.15e-04 + sampling 1.48e-03 -> limited by sampling, binding functional call_price). Ladder value 4.3907
- observed_order:          0.9970 +/- 0.0012 (expected 1.0, one-sided below the declared floor). Median-based estimate 0.9925
- weak_order:              0.9998 (x: 0.99991, x^2: 0.99961; tanh excluded as at the noise floor)
- dynkin:                  ok. z = 0.655 and 0.648 (X and X^2) against bar 2.28, extrapolation orders 0.990 and 0.983 measured. Not replicated, and not marginal: |z - bar| = 1.62 is outside the 0.7 margin, so one seed settled it
- terminal_law:            not declared
- shape:                   skewness -3.066 +/- 0.011, excess kurtosis 16.42 +/- 0.085 (descriptive only)
- constraints:             finite pass; positivity pass; martingale pass (drift 4.38e-03 against bar 4.66e-02); estimator_stability pass (1.6e-04 against bar 1.0e-02)
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. Every checked moment is within tolerance of the exact moments and the estimates are MC-resolved. Tier A.

### Feedback for solver
- None. The earlier Dynkin `unavailable` was a spec defect (test functions broadcasting to (M,M) inside the kernel generator), fixed in problem_spec.json. The solver is unchanged.
- For the record, what the 10 does not establish: the Dynkin generator is built from the spec's own moment-equivalent drift and diffusion, so it cannot detect a mis-transcribed coefficient. The exact moments and the strong order 0.997 for a declared Milstein cover that.

</review>
