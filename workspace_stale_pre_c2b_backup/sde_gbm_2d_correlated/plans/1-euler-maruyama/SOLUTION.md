---
id: 1
plan_slug: euler-maruyama
scheme: euler-maruyama
strategy: Euler-Maruyama Monte Carlo on the coupled 2D correlated GBM system, with correlated Brownian increments generated via a Cholesky factor.
---

## SDE Reference

Coupled 2D correlated GBM (Ito):

```
dX = mu1 * X * dt + sigma1 * X * dW1;   X(0) = X_0
dY = mu2 * Y * dt + sigma2 * Y * dW2;   Y(0) = Y_0
corr(dW1, dW2) = rho
```

Parameters:
- X_0 = 1.0
- Y_0 = 1.0
- mu1 = 0.10
- sigma1 = 0.20
- mu2 = 0.15
- sigma2 = 0.25
- rho = 0.60

Time interval: [0.0, 1.0], T = 1.0

Analytic per-component moments (correlation does not affect marginal means/variances):
```
exact_mean_X     = X_0 * np.exp(mu1 * t)
exact_variance_X = X_0**2 * np.exp(2*mu1*t) * (np.exp(sigma1**2 * t) - 1)
exact_mean_Y     = Y_0 * np.exp(mu2 * t)
exact_variance_Y = Y_0**2 * np.exp(2*mu2*t) * (np.exp(sigma2**2 * t) - 1)
```

## Numerical Scheme

- **Method**: Euler-Maruyama (strong order 0.5, weak order 1.0)
- **dt**: 0.01
- **num_paths**: 50000
- **T**: 1.0
- **Milstein correction**: N/A — multi-dimensional system; Milstein requires the Lévy area (iterated integral ∫∫dW_i dW_j) which is not computable without approximation. EM only.

Vector EM update, state layout shape `(num_paths, 2)`:
```
X_{n+1} = X_n + mu1 * X_n * dt + sigma1 * X_n * dW1_n
Y_{n+1} = Y_n + mu2 * Y_n * dt + sigma2 * Y_n * dW2_n
```

## Implementation Notes

- **Correlation via Cholesky**: draw independent Z1, Z2 ~ N(0,1). Set
  ```
  dW1 = sqrt(dt) * Z1
  dW2 = sqrt(dt) * (rho * Z1 + sqrt(1 - rho**2) * Z2)
  ```
  This gives corr(dW1, dW2) = rho with correct marginal variances dt.
- X and Y are each standard multiplicative GBM, updated independently after the correlated increments are generated.
- The correlation rho affects only the cross-correlation between X and Y, not the per-component marginal means or variances — the evaluator checks marginals.
- X_0, Y_0 > 0 and GBM stays positive, so no positivity guard is required.
- Use a single seeded `np.random.default_rng(seed)` with seed = 42; draw both Z1 and Z2 each step.

## Results

Ran with num_paths=50000, dt=0.01, T=1.0, seed=42.

- Empirical mean:     [1.1048456591465634, 1.1623332254940246]
- Empirical variance: [0.049634915346008224, 0.08756522920187612]

## Evaluation

(Filled by evaluator)

<review score=10>

**Score: 10/10 — Done**

### Numerical Accuracy

**Component X:**
- mean_relative_error:     0.0003  (PASS)
- variance_relative_error: 0.0042  (PASS)

**Component Y:**
- mean_relative_error:     0.0004  (PASS)
- variance_relative_error: 0.0058  (PASS)

- Overall: PASS

### Feedback for solver
- None

</review>
