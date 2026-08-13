---
id: 2
plan_slug: milstein
scheme: milstein
strategy: Milstein scheme with constant diffusion correction and full-truncation guard for higher strong order on the Feller-violated CIR process.
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
dg/dX     = sigma / (2*sqrt(X))

Analytic moments at t = T:
- exact_mean     = theta + (X_0 - theta) * np.exp(-kappa * t)
- exact_variance = (sigma**2 / kappa) * (X_0 * (np.exp(-kappa*t) - np.exp(-2*kappa*t)) + 0.5 * theta * (1 - np.exp(-kappa*t))**2)

At T = 1: exact_mean = 0.5 (well above the near-zero threshold, so the mean check applies).

## Numerical Scheme

- **Method**: Milstein (strong order 1.0, weak order 1.0)
- **dt**: 0.01
- **num_paths**: 50000
- **T**: 1.0
- **seed**: 42
- **Milstein correction**: `0.5 * g * (dg/dX) * (dW**2 - dt) = (sigma**2 / 4) * (dW**2 - dt)`

Scalar update:
```
dW_n    = sqrt(dt) * Z_n,   Z_n ~ N(0,1)
X_pos   = max(X_n, 0.0)                      # full truncation guard
X_{n+1} = X_n + kappa*(theta - X_n)*dt + sigma*sqrt(X_pos)*dW_n
                 + (sigma**2 / 4) * (dW_n**2 - dt)
```

## Implementation Notes

- The Milstein correction term `(sigma**2/4)*(dW**2 - dt)` is **constant** (independent of X) because for CIR `g*g' = sigma*sqrt(X) * sigma/(2*sqrt(X)) = sigma**2/2` cancels the sqrt. It can be applied even at the zero boundary without any special handling — no division by sqrt(X).
- **Positivity guard is MANDATORY** (full truncation scheme). Apply `X_pos = np.maximum(X_n, 0.0)` before every `sqrt` in the diffusion term. The Feller condition is violated, so X reaches zero frequently.
- Do NOT compute dg/dX = sigma/(2*sqrt(X)) directly at runtime — it blows up at X = 0. Use the simplified constant form `(sigma**2 / 4) * (dW**2 - dt)`.
- The mean E[X(T)] = 0.5 is well above the near-zero threshold (0.01), so both the mean and variance checks apply.
- If the evaluator reports poor accuracy, refine dt downward (e.g. 0.005 or 0.001).

## Results

Solver run with num_paths=50000, dt=0.01, T=1.0, seed=42:

- Empirical mean:     0.511178
- Empirical variance: 0.924684

Analytic reference (T=1.0):
- exact_mean:     0.500000
- exact_variance: 0.864665

Relative errors:
- mean_rel_err:   2.24%  (threshold: 5%)
- var_rel_err:    6.94%  (threshold: 10%)

## Evaluation

(Filled by evaluator)

<review score=10>

**Score: 10/10 — Done**

### Numerical Accuracy
- mean_relative_error:     0.0224  (PASS)
- variance_relative_error: 0.0694  (PASS)
- Overall: PASS ✓

### Feedback for solver
- None

</review>
