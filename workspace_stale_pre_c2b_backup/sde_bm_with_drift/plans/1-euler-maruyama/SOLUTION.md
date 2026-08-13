---
id: 1
plan_slug: euler-maruyama
scheme: euler-maruyama
strategy: Monte Carlo Euler-Maruyama for additive-noise scalar BM with drift; EM is exact in distribution here.
---

## SDE Reference

Ito SDE (scalar, additive noise):

```
dX(t) = mu * dt + sigma * dW(t),   X(0) = X_0
```

Parameters:
- X_0 = 1.0
- mu = 0.5
- sigma = 0.3

Time interval: [T0, T] = [0.0, 1.0]

Drift f(X) = mu = 0.5 (state-independent).
Diffusion g(X) = sigma = 0.3 (state-independent, additive noise).

Analytic moments at t = T:
- exact_mean = X_0 + mu * t  → at t=1: 1.5
- exact_variance = sigma**2 * t  → at t=1: 0.09

## Numerical Scheme

- **Method**: Euler-Maruyama
- **dt**: 0.01
- **num_paths**: 50000
- **T**: 1.0
- **seed**: 42
- **Milstein correction**: 0 (additive noise, dg/dX = 0, so Milstein reduces exactly to Euler-Maruyama). Only EM is proposed.

Scalar EM update:
```
dW_n    = sqrt(dt) * Z_n,   Z_n ~ N(0,1)
X_{n+1} = X_n + mu * dt + sigma * dW_n
```

## Implementation Notes

- Additive noise with constant coefficients: no positivity guards, no state-dependent diffusion, no correlation structure needed.
- X(T) is exactly Gaussian, so Euler-Maruyama is exact in distribution at any dt; discretization error affects only Monte Carlo sampling noise, not scheme bias. dt = 0.01 is comfortably sufficient.
- exact_mean = 1.5 is well above the near-zero threshold (0.01), so the mean relative error check applies.
- Use single-array state of shape (num_paths,); step all paths simultaneously with NumPy broadcasting.

## Results

Ran with num_paths=50000, dt=0.01, T=1.0, seed=42.

- Empirical mean:     1.502579  (exact: 1.5, rel err ≈ 0.17%)
- Empirical variance: 0.091344  (exact: 0.09, rel err ≈ 1.49%)

## Evaluation

(Filled by evaluator)

<review score=10>

**Score: 10/10 — Done**

### Numerical Accuracy
- mean_relative_error:     0.0017  (PASS)
- variance_relative_error: 0.0149  (PASS)
- Overall: PASS ✓

### Feedback for solver
- None

</review>
