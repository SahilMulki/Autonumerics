---
id: 1
plan_slug: euler-maruyama
scheme: euler-maruyama
strategy: Vanilla Euler-Maruyama Monte Carlo for scalar multiplicative GBM, strong order 0.5.
---

## SDE Reference

Geometric Brownian Motion (Ito):

    dX(t) = mu * X(t) * dt + sigma * X(t) * dW(t),   X(0) = 1.0

over t in [0, 1].

Parameters:
- X_0   = 1.0
- mu    = 0.1
- sigma = 0.2

Drift:     f(X) = mu * X   = 0.1 * X
Diffusion: g(X) = sigma * X = 0.2 * X

Analytic moments at t = T:
- exact_mean     = X_0 * np.exp(mu * t)                                  ~= exp(0.1)   ~= 1.10517
- exact_variance = X_0**2 * np.exp(2*mu*t) * (np.exp(sigma**2 * t) - 1)  ~= 0.04987

## Numerical Scheme

- **Method**: Euler-Maruyama
- **dt**: 0.01
- **num_paths**: 50000
- **T**: 1.0
- **seed**: 42

Scalar EM update:

    dW_n     = sqrt(dt) * Z_n,   Z_n ~ N(0,1)
    X_{n+1}  = X_n + mu * X_n * dt + sigma * X_n * dW_n

## Implementation Notes

- Multiplicative scalar noise, Ito interpretation. Standard GBM.
- X stays positive for X_0 > 0; no positivity guard strictly required. EM can in principle produce a negative value under a very large negative increment, but this is extremely unlikely at dt = 0.01, sigma = 0.2. No guard needed.
- Vectorize over all 50000 paths with NumPy broadcasting; state array shape (num_paths,).
- EM has strong order 0.5, weak order 1.0. Weak order 1.0 means terminal moments (mean, variance) should be accurate — this is what the evaluator checks. dt = 0.01 should comfortably meet the variance_rel_err < 10% and mean_rel_err < 5% thresholds.
- If the evaluator reports poor accuracy, refine dt (e.g. 0.005 or 0.001).

## Results

- Empirical mean:     1.107361
- Empirical variance: 0.050723

Analytic values at T=1: mean ~= 1.10517, variance ~= 0.04987.
Mean relative error ~= 0.20%, variance relative error ~= 1.7%. Both well within thresholds.

## Evaluation

(Filled by evaluator)

<review score=10>

**Score: 10/10 — Done**

### Numerical Accuracy
- mean_relative_error:     0.0020  (PASS)
- variance_relative_error: 0.0176  (PASS)
- Overall: PASS ✓

### Feedback for solver
None

</review>
