---
id: 1
plan_slug: euler-maruyama
scheme: euler-maruyama
strategy: Euler-Maruyama Monte Carlo for additive-noise scalar Brownian motion; Milstein adds no value since dg/dX = 0.
---

## SDE Reference

Standard Ito Brownian motion:

    dX(t) = dW(t),   X(0) = 0.0

- **Family**: bm_standard
- **State dimension**: 1 (scalar)
- **Noise structure**: additive (g(X) = 1, constant unit diffusion)
- **Parameters**: X_0 = 0.0
- **Time interval**: [0.0, 1.0], T = 1.0

Analytic moments at T:
- exact_mean = 0.0
- exact_variance = t  (= 1.0 at T = 1)

## Numerical Scheme

- **Method**: Euler-Maruyama
- **dt**: 0.01
- **num_paths**: 50000
- **T**: 1.0
- **Milstein correction**: not applicable (additive noise, dg/dX = 0, so Milstein == EM)

Scalar update:

    dW_n    = sqrt(dt) * Z_n,   Z_n ~ N(0,1)
    X_{n+1} = X_n + f(X_n) * dt + g(X_n) * dW_n
            = X_n + 0 * dt + 1 * dW_n
            = X_n + dW_n

## Implementation Notes

- Drift f(X) = 0, diffusion g(X) = 1. Each step reduces to accumulating a Brownian increment, so EM is exact in distribution here (no discretization bias in the moments).
- No positivity guard needed; X ranges over all of R.
- **Near-zero mean**: exact_mean = 0.0 < near_zero_mean_threshold (0.01), so the evaluator skips the mean relative-error check. Score is driven by variance_rel_err < 10% against exact_variance = 1.0.
- Use `np.random.default_rng(seed=42)`; return terminal paths of shape (num_paths,), empirical_mean, and empirical_variance (ddof=1).

## Results

- **Empirical mean**: 0.008598
- **Empirical variance**: 1.014938
- **num_paths**: 50000
- **dt**: 0.01
- **T**: 1.0

Exact mean = 0.0 (mean check skipped, |exact_mean| < 0.01). Exact variance = 1.0. Variance relative error ≈ 1.5%, well within the 10% threshold.

## Evaluation

(Filled by evaluator)

<review score=10>

**Score: 10/10 — Done**

### Numerical Accuracy
- mean_relative_error:     skipped  (PASS / skipped — |exact_mean| = 0.0 < 0.01)
- variance_relative_error: 0.0149  (PASS)
- Overall: PASS ✓

### Feedback for solver
- None

</review>
