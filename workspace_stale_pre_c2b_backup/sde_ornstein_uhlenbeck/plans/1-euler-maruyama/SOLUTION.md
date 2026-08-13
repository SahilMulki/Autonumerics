---
id: 1
plan_slug: euler-maruyama
scheme: euler-maruyama
strategy: Euler-Maruyama Monte Carlo for the additive-noise OU process; Milstein offers no gain since dg/dX = 0.
---

## SDE Reference

Ito SDE (Ornstein-Uhlenbeck, mean-reverting, additive noise):

    dX(t) = theta * (mu - X(t)) * dt + sigma * dW(t),   X(0) = 2.0

Parameters:
- X_0   = 2.0
- theta = 1.5
- mu    = 0.0
- sigma = 0.5

Time interval: T0 = 0.0, T = 1.0

Drift:     f(X) = theta * (mu - X)   (linear in X)
Diffusion: g(X) = sigma             (constant → additive noise)

Analytic moments at time t:
- exact_mean     = mu + (X_0 - mu) * np.exp(-theta * t)
- exact_variance = sigma**2 / (2*theta) * (1 - np.exp(-2*theta*t))

At T = 1: exact_mean ≈ 2.0*exp(-1.5) ≈ 0.4463 (above near-zero threshold, so mean check applies);
exact_variance ≈ 0.0791.

## Numerical Scheme

- **Method**: Euler-Maruyama (strong order 0.5, weak order 1.0)
- **dt**: 0.01
- **num_paths**: 50000
- **T**: 1.0
- **Milstein correction**: 0 (additive noise, dg/dX = 0, so Milstein reduces to Euler-Maruyama; not eligible)

Scalar EM update:

    dW_n    = sqrt(dt) * Z_n,   Z_n ~ N(0,1)
    X_{n+1} = X_n + theta*(mu - X_n)*dt + sigma*dW_n

## Implementation Notes

- Single scalar state, additive constant noise — no positivity guard or matrix diffusion needed.
- Mean is well above the near-zero threshold (≈ 0.446), so the mean relative-error check (< 5%) applies in addition to the variance check (< 10%).
- Vectorize across all num_paths with NumPy broadcasting; shape (num_paths,).
- Use rng = np.random.default_rng(seed) with seed = 42 for reproducibility.
- dt = 0.01 over T = 1.0 gives 100 steps; EM weak error is O(dt), which comfortably meets weak-moment thresholds. Solver may refine dt if accuracy is reported poor.

## Results

Empirical mean:     0.443810
Empirical variance: 0.081047

(num_paths=50000, dt=0.01, T=1.0, seed=42)

## Evaluation

(Filled by evaluator)

<review score=10>

**Score: 10/10 — Done**

### Numerical Accuracy
- mean_relative_error:     0.0055  (PASS)
- variance_relative_error: 0.0235  (PASS)
- Overall: PASS ✓

### Feedback for solver
- None

</review>
