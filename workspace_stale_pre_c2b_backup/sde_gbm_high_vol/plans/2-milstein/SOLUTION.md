---
id: 2
plan_slug: milstein
scheme: milstein
strategy: Milstein scheme (strong order 1.0) to reduce the large-sigma discretization bias in the variance, paired with a large path count for the heavy-tailed estimator.
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

- **Method**: Milstein (strong order 1.0, weak order 1.0)
- **dt**: 0.005
- **num_paths**: 200000
- **T**: 1.0
- **seed**: 42

Scalar update:
```
dW_n    = sqrt(dt) * Z_n,   Z_n ~ N(0,1)
X_{n+1} = X_n + mu * X_n * dt + sigma * X_n * dW_n
                + 0.5 * sigma**2 * X_n * (dW_n**2 - dt)
```

- **Milstein correction**: `0.5 * sigma**2 * X_n * (dW**2 - dt)`
  (g(X) = sigma*X, dg/dX = sigma, so 0.5 * g * g' = 0.5 * sigma**2 * X)

## Implementation Notes

- High-volatility multiplicative scalar GBM (sigma = 1.0). X(T) is lognormal with
  log-variance sigma^2 * T = 1.0, making it strongly heavy-tailed.
- Milstein's strong order 1.0 (vs EM's 0.5) helps control the large-sigma pathwise
  bias, which is the main advantage here; the variance should be more accurate than EM
  at the same dt.
- The empirical variance estimator still has large sampling error from the heavy tail,
  so num_paths is set to 200000 (well above the 50000 default). Increase further if the
  variance tolerance is not met.
- No positivity guard needed: X stays positive given X_0 > 0 (multiplicative structure).
- Reuse the SAME dW for the base step and the correction term; do not resample.

## Results

Run with num_paths=500000, dt=0.005, T=1.0, seed=42:
- Empirical mean:     1.052093
- Empirical variance: 1.905449

Analytic reference: exact_mean ~= 1.0513, exact_variance ~= 1.899.

## Evaluation

(Filled by evaluator)

<review score=10>

**Score: 10/10 — Done**

### Numerical Accuracy
- mean_relative_error:     0.0008  (PASS)
- variance_relative_error: 0.0034  (PASS)
- Overall: PASS

### Feedback for solver
- None

</review>
