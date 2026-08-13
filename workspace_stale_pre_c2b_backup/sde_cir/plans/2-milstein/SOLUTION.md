---
id: 2
plan_slug: milstein
scheme: milstein
strategy: Milstein scheme for the CIR process, adding the (sigma^2/4)*(dW^2 - dt) correction to reach strong order 1.0, with a mandatory positivity guard before the sqrt.
---

## SDE Reference

Cox-Ingersoll-Ross (Ito):

    dX(t) = kappa * (theta - X(t)) * dt + sigma * sqrt(X(t)) * dW(t)

Parameters:
- X_0   = 0.5
- kappa = 2.0
- theta = 1.0
- sigma = 0.5

Time interval: [T0, T] = [0.0, 1.0]

Drift:     f(X) = kappa * (theta - X)
Diffusion: g(X) = sigma * sqrt(X)   (multiplicative)
dg/dX:     sigma / (2*sqrt(X))

Analytic moments at t = T:
- mean     = theta + (X_0 - theta) * np.exp(-kappa * t)
- variance = (sigma**2 / kappa) * (X_0 * (np.exp(-kappa*t) - np.exp(-2*kappa*t)) + 0.5 * theta * (1 - np.exp(-kappa*t))**2)

## Numerical Scheme

- **Method**: Milstein (strong order 1.0, weak order 1.0)
- **dt**: 0.01
- **num_paths**: 50000
- **T**: 1.0
- **Milstein correction**: `0.5 * g * g' * (dW**2 - dt)` which for CIR simplifies to `(sigma**2 / 4) * (dW**2 - dt)`.

Scalar update:
```
dW = sqrt(dt) * Z,   Z ~ N(0,1)
X_pos = max(X, 0.0)
X = X + kappa*(theta - X)*dt + sigma*sqrt(X_pos)*dW
      + (sigma**2 / 4) * (dW**2 - dt)
```

## Implementation Notes

- **Positivity guard is MANDATORY**: apply `X_pos = np.maximum(X, 0.0)` before the sqrt in the diffusion term. The Milstein correction term `(sigma**2/4)*(dW**2 - dt)` uses no sqrt and needs no guard.
- The correction simplifies nicely because g * g' = sigma*sqrt(X) * sigma/(2*sqrt(X)) = sigma^2/2, so the term is state-independent: `(sigma**2/4)*(dW**2 - dt)`.
- Mean at T is well away from zero (~0.93), so the mean relative-error check applies (threshold 5%).
- Milstein's strong order 1.0 should reduce discretization bias versus EM at the same dt; expect equal or better accuracy at dt = 0.01.

## Results

Empirical mean:     0.936078
Empirical variance: 0.055548

## Evaluation

(Filled by evaluator)

<review score=10>

**Score: 10/10 — Done**

### Numerical Accuracy
- mean_relative_error:     0.0040  (PASS)
- variance_relative_error: 0.0279  (PASS)
- Overall: PASS ✓

### Feedback for solver
- None

</review>
