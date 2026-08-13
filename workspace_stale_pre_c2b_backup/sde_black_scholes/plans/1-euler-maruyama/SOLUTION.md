---
id: 1
plan_slug: euler-maruyama
scheme: euler-maruyama
strategy: Baseline Euler-Maruyama Monte Carlo for the Black-Scholes asset price with scalar multiplicative noise.
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
Diffusion: g(X) = sigma * X

Analytic moments at t = T:
- exact_mean     = X_0 * np.exp(r * t)                                    (~= 105.13)
- exact_variance = X_0**2 * np.exp(2*r*t) * (np.exp(sigma**2 * t) - 1)

## Numerical Scheme

- **Method**: Euler-Maruyama (strong order 0.5, weak order 1.0)
- **dt**: 0.01
- **num_paths**: 50000
- **T**: 1.0
- **seed**: 42

Scalar update:
```
dW_n    = sqrt(dt) * Z_n,   Z_n ~ N(0,1)
X_{n+1} = X_n + r * X_n * dt + sigma * X_n * dW_n
```

## Implementation Notes

- Scalar multiplicative noise; state shape (num_paths,).
- X stays positive given X_0 = 100 > 0; no positivity guard strictly required.
- Mean is large (~105) relative to the near-zero threshold (0.01), so the mean relative-error check applies (mean_rel_err_max = 0.05).
- Variance is large (order 1e2); variance relative-error check applies (variance_rel_err_max = 0.10).
- EM has weak order 1.0, so terminal moments should be accurate at dt = 0.01. Refine dt (e.g. 0.005) if the evaluator reports variance error above threshold.

## Results

Empirical mean:     105.339525
Empirical variance: 459.465404

Run with num_paths=50000, dt=0.01, T=1.0, seed=42.

## Evaluation

(Filled by evaluator)

<review score=10>

**Score: 10/10 — Done**

### Numerical Accuracy
- mean_relative_error:     0.0020  (PASS)
- variance_relative_error: 0.0187  (PASS)
- Overall: PASS ✓

### Feedback for solver
- None

</review>
