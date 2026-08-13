---
id: 2
plan_slug: milstein
scheme: milstein
strategy: Milstein scheme with the GBM diffusion-derivative correction term, strong order 1.0.
---

## SDE Reference

Geometric Brownian Motion (Ito):

    dX(t) = mu * X(t) * dt + sigma * X(t) * dW(t),   X(0) = 1.0

over t in [0, 1].

Parameters:
- X_0   = 1.0
- mu    = 0.1
- sigma = 0.2

Drift:     f(X)  = mu * X    = 0.1 * X
Diffusion: g(X)  = sigma * X = 0.2 * X
Diffusion derivative: dg/dX = sigma = 0.2

Analytic moments at t = T:
- exact_mean     = X_0 * np.exp(mu * t)                                  ~= exp(0.1)   ~= 1.10517
- exact_variance = X_0**2 * np.exp(2*mu*t) * (np.exp(sigma**2 * t) - 1)  ~= 0.04987

## Numerical Scheme

- **Method**: Milstein
- **dt**: 0.01
- **num_paths**: 50000
- **T**: 1.0
- **seed**: 42

Scalar Milstein update:

    dW_n     = sqrt(dt) * Z_n,   Z_n ~ N(0,1)
    X_{n+1}  = X_n + mu * X_n * dt + sigma * X_n * dW_n
                   + 0.5 * sigma**2 * X_n * (dW_n**2 - dt)

- **Milstein correction**: `0.5 * sigma * X_n * sigma * (dW**2 - dt)  =  0.5 * sigma**2 * X_n * (dW**2 - dt)`

## Implementation Notes

- Multiplicative scalar noise, Ito interpretation. GBM is a textbook Milstein-eligible case (scalar, multiplicative, dg/dX = sigma in closed form).
- The correction term `0.5 * g * g' * (dW**2 - dt)` = `0.5 * sigma**2 * X_n * (dW**2 - dt)` upgrades strong order from 0.5 to 1.0.
- X stays positive for X_0 > 0; no positivity guard strictly required.
- Vectorize over all 50000 paths with NumPy broadcasting; state array shape (num_paths,).
- Milstein improves strong (pathwise) accuracy over EM. Weak-order terminal moments are already 1.0 for both schemes, so both should pass the moment thresholds; Milstein is the differentiated higher-strong-order plan.
- If the evaluator reports poor accuracy, refine dt (e.g. 0.005 or 0.001).

## Results

Solver run with num_paths=50000, dt=0.01, T=1.0, seed=42:

- Empirical mean:     1.107348
- Empirical variance: 0.050730

Analytic reference:
- exact_mean     = exp(0.1) ~= 1.105171
- exact_variance ~= 0.049867

Mean relative error:     ~0.20%  (threshold 5%)
Variance relative error: ~1.73%  (threshold 10%)

## Evaluation

(Filled by evaluator)

<review score=10>

**Score: 10/10 — Done**

### Numerical Accuracy
- mean_relative_error:     0.0020  (PASS)
- variance_relative_error: 0.0177  (PASS)
- Overall: PASS ✓

### Feedback for solver
- None

</review>
