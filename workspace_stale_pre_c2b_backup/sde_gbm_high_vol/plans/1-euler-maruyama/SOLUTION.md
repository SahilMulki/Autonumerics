---
id: 1
plan_slug: euler-maruyama
scheme: euler-maruyama
strategy: Baseline Euler-Maruyama Monte Carlo with small dt and a large path count to tame the heavy-tailed variance estimator at sigma = 1.0.
---

## SDE Reference

Ito SDE (Geometric Brownian Motion, high volatility):

    dX(t) = mu * X(t) * dt + sigma * X(t) * dW(t),   X(0) = 1.0

Parameters:
- X_0 = 1.0
- mu  = 0.05
- sigma = 1.0

Time interval: [T0, T] = [0.0, 1.0]

Analytic moments at t = T:
- exact_mean     = X_0 * np.exp(mu * t)
- exact_variance = X_0**2 * np.exp(2*mu*t) * (np.exp(sigma**2 * t) - 1)

At T = 1: exact_mean ~= 1.0513, exact_variance = exp(0.1)*(exp(1)-1) ~= 1.899.

## Numerical Scheme

- **Method**: Euler-Maruyama (strong order 0.5, weak order 1.0)
- **dt**: 0.005
- **num_paths**: 200000
- **T**: 1.0
- **seed**: 42

Scalar update:
```
dW_n    = sqrt(dt) * Z_n,   Z_n ~ N(0,1)
X_{n+1} = X_n + mu * X_n * dt + sigma * X_n * dW_n
```

## Implementation Notes

- High-volatility multiplicative scalar GBM (sigma = 1.0). X(T) is lognormal with
  log-variance sigma^2 * T = 1.0, making it strongly heavy-tailed.
- The empirical variance estimator has large sampling error because the fourth moment
  is very large. The 10% variance tolerance therefore demands many paths; num_paths is
  set to 200000 (well above the 50000 default) for this reason. Increase further if the
  variance tolerance is not met.
- No positivity guard needed: X stays positive given X_0 > 0 (multiplicative structure).
- EM has weak order 1.0, so the mean is captured well, but the strong order 0.5 leaves
  a larger large-sigma bias in the variance than Milstein; if the variance error is
  persistently high, reduce dt (e.g. 0.002 or 0.001) before adding more paths.

## Results

- num_paths: 200000
- dt: 0.001
- T: 1.0
- seed: 42
- Empirical mean:     1.050268
- Empirical variance: 1.863881

Analytic reference (T=1): exact_mean ~= 1.0513, exact_variance ~= 1.899.

## Evaluation

(Filled by evaluator)

<review score=10>

**Score: 10/10 — Done**

### Numerical Accuracy
- mean_relative_error:     0.0010  (PASS)
- variance_relative_error: 0.0185  (PASS)
- Overall: PASS ✓

### Feedback for solver
- None.

</review>
