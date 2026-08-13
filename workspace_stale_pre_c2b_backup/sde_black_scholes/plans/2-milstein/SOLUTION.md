---
id: 2
plan_slug: milstein
scheme: milstein
strategy: Milstein scheme with the GBM diffusion correction for higher strong-order accuracy on the Black-Scholes asset price.
---

## SDE Reference

    dX(t) = r * X(t) * dt + sigma * X(t) * dW(t)

Risk-neutral Black-Scholes asset price (mathematically identical to GBM with drift mu = r).

Parameters:
- X_0   = 100.0
- r     = 0.05
- sigma = 0.20

Time interval: [0, 1], so T = 1.0.

Drift: f(X) = r * X
Diffusion: g(X) = sigma * X,  dg/dX = sigma

Analytic moments at t = T:
- exact_mean     = X_0 * np.exp(r * t)                                    (~= 105.13)
- exact_variance = X_0**2 * np.exp(2*r*t) * (np.exp(sigma**2 * t) - 1)

## Numerical Scheme

- **Method**: Milstein (strong order 1.0, weak order 1.0)
- **dt**: 0.01
- **num_paths**: 50000
- **T**: 1.0
- **seed**: 42

Scalar update:
```
dW_n    = sqrt(dt) * Z_n,   Z_n ~ N(0,1)
X_{n+1} = X_n + r * X_n * dt + sigma * X_n * dW_n
         + 0.5 * sigma**2 * X_n * (dW_n**2 - dt)
```

- **Milstein correction**: 0.5 * sigma * X_n * sigma * (dW**2 - dt) = 0.5 * sigma**2 * X_n * (dW**2 - dt)

## Implementation Notes

- Scalar multiplicative noise; state shape (num_paths,). Milstein eligible (state_dimension == 1, multiplicative noise).
- X stays positive given X_0 = 100 > 0; no positivity guard strictly required.
- Mean is large (~105) relative to the near-zero threshold (0.01), so the mean relative-error check applies (mean_rel_err_max = 0.05).
- Variance is large (order 1e2); variance relative-error check applies (variance_rel_err_max = 0.10).
- The Milstein correction 0.5 * g * g' * (dW**2 - dt) improves strong order to 1.0; weak-order accuracy on terminal moments should be at least as good as EM at dt = 0.01.

## Results

- Empirical mean:     105.338254
- Empirical variance: 459.532331
- num_paths: 50000
- dt: 0.01
- T: 1.0

## Evaluation

(Filled by evaluator)

<review score=10>

**Score: 10/10 — Done**

### Numerical Accuracy
- mean_relative_error:     0.0020  (PASS)
- variance_relative_error: 0.0189  (PASS)
- Overall: PASS ✓

### Feedback for solver
- None

</review>
