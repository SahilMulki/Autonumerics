---
id: 1
plan_slug: euler-maruyama
scheme: euler-maruyama
strategy: Baseline Euler-Maruyama Monte Carlo for the CIR square-root process with a mandatory positivity guard before the sqrt.
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

Analytic moments at t = T:
- mean     = theta + (X_0 - theta) * np.exp(-kappa * t)
- variance = (sigma**2 / kappa) * (X_0 * (np.exp(-kappa*t) - np.exp(-2*kappa*t)) + 0.5 * theta * (1 - np.exp(-kappa*t))**2)

## Numerical Scheme

- **Method**: Euler-Maruyama (strong order 0.5, weak order 1.0)
- **dt**: 0.01
- **num_paths**: 50000
- **T**: 1.0
- **Milstein correction**: not applicable (EM baseline)

Scalar update:
```
dW = sqrt(dt) * Z,   Z ~ N(0,1)
X_pos = max(X, 0.0)
X = X + kappa*(theta - X)*dt + sigma*sqrt(X_pos)*dW
```

## Implementation Notes

- **Positivity guard is MANDATORY**: g(X) = sigma*sqrt(X) is undefined for X < 0. Apply `X_pos = np.maximum(X, 0.0)` before the sqrt. The Feller condition 2*kappa*theta >= sigma^2 holds (4.0 >= 0.25) so X stays positive mathematically, but floating-point rounding can produce tiny negative values near zero.
- Mean at T is well away from zero (~0.93), so the mean relative-error check applies (threshold 5%).
- If the evaluator reports poor accuracy, the solver may refine `dt` (e.g. 0.005 or 0.001). EM only reaches strong order 0.5, so the CIR sqrt diffusion is the main source of discretization bias here.

## Results

- Empirical mean:     0.936103
- Empirical variance: 0.055532

## Evaluation

(Filled by evaluator)

<review score=10>

**Score: 10/10 — Done**

### Numerical Accuracy
- mean_relative_error:     0.0040  (PASS)
- variance_relative_error: 0.0276  (PASS)
- Overall: PASS ✓

### Feedback for solver
- None

</review>
