---
id: 1
plan_slug: euler-maruyama
scheme: euler-maruyama
strategy: Euler-Maruyama with full-truncation positivity guard for the Feller-violated CIR process.
---

## SDE Reference

Ito SDE (Cox-Ingersoll-Ross):

    dX(t) = kappa * (theta - X(t)) * dt + sigma * sqrt(X(t)) * dW(t)

Parameters:
- X_0   = 0.5
- kappa = 1.0
- theta = 0.5
- sigma = 2.0
- T0 = 0.0, T = 1.0

Feller condition VIOLATED: 2*kappa*theta = 1.0 < sigma^2 = 4.0, so X hits zero frequently.

Drift:      f(X) = kappa * (theta - X)
Diffusion:  g(X) = sigma * sqrt(X)   (undefined for X < 0)

Analytic moments at t = T:
- exact_mean     = theta + (X_0 - theta) * np.exp(-kappa * t)
- exact_variance = (sigma**2 / kappa) * (X_0 * (np.exp(-kappa*t) - np.exp(-2*kappa*t)) + 0.5 * theta * (1 - np.exp(-kappa*t))**2)

At T = 1: exact_mean = 0.5 (well above the near-zero threshold, so the mean check applies).

## Numerical Scheme

- **Method**: Euler-Maruyama (strong order 0.5, weak order 1.0)
- **dt**: 0.01
- **num_paths**: 50000
- **T**: 1.0
- **seed**: 42

Scalar update:
```
dW_n    = sqrt(dt) * Z_n,   Z_n ~ N(0,1)
X_pos   = max(X_n, 0.0)                      # full truncation guard
X_{n+1} = X_n + kappa*(theta - X_n)*dt + sigma*sqrt(X_pos)*dW_n
```

## Implementation Notes

- **Positivity guard is MANDATORY** (full truncation scheme). The Feller condition is violated, so X reaches zero frequently and floating-point steps push it negative. Apply `X_pos = np.maximum(X_n, 0.0)` before every `sqrt`. The guard is applied only inside the diffusion term (full truncation); the drift may use the raw X_n.
- Diffusion g(X) = sigma*sqrt(X) is undefined for X < 0; without the guard the run produces NaNs and the evaluator scores 1-3.
- The mean E[X(T)] = 0.5 is well above the near-zero threshold (0.01), so both the mean and variance checks apply.
- If the evaluator reports poor variance accuracy, refine dt downward (e.g. 0.005 or 0.001) — the strong bias near the zero boundary is the likely culprit for a Feller-violated CIR.

## Results

- Empirical mean:     0.512951
- Empirical variance: 0.919528

## Evaluation

(Filled by evaluator)

<review score=10>

**Score: 10/10 — Done**

### Numerical Accuracy
- mean_relative_error:     0.0259  (PASS)
- variance_relative_error: 0.0635  (PASS)
- Overall: PASS ✓

### Feedback for solver
- None

</review>
