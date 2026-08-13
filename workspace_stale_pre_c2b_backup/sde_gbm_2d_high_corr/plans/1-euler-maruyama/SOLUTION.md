---
id: 1
plan_slug: euler-maruyama
scheme: euler-maruyama
strategy: Vector Euler-Maruyama Monte Carlo with Cholesky-correlated increments for the two GBM components at rho=0.95.
---

## SDE Reference

Coupled 2D Ito system (correlated GBMs):

```
dX = mu1 * X * dt + sigma1 * X * dW1,   X(0) = X_0
dY = mu2 * Y * dt + sigma2 * Y * dW2,   Y(0) = Y_0
corr(dW1, dW2) = rho
```

Parameters:

| symbol | value |
|---|---|
| X_0 | 1.0 |
| Y_0 | 1.0 |
| mu1 | 0.10 |
| sigma1 | 0.30 |
| mu2 | 0.10 |
| sigma2 | 0.30 |
| rho | 0.95 |

Time interval: T0 = 0.0, T = 1.0.

Analytic per-component moments (marginals are exact GBM, unaffected by rho) at t = T:

```python
mean_X     = X_0 * np.exp(mu1 * t)
variance_X = X_0**2 * np.exp(2*mu1*t) * (np.exp(sigma1**2 * t) - 1)
mean_Y     = Y_0 * np.exp(mu2 * t)
variance_Y = Y_0**2 * np.exp(2*mu2*t) * (np.exp(sigma2**2 * t) - 1)
```

## Numerical Scheme

- **Method**: Euler-Maruyama (vector, multi-D). Milstein is not used — multi-D Milstein requires the Levy area for the cross terms and `milstein_eligible == false`.
- **dt**: 0.01
- **num_paths**: 50000
- **T**: 1.0
- **Milstein correction**: N/A (multi-D).

Vector EM update with state shape `(num_paths, 2)`:

```
Z1, Z2 ~ N(0,1) independent, shape (num_paths,)
dW1     = sqrt(dt) * Z1
dW2     = sqrt(dt) * (rho * Z1 + sqrt(1 - rho**2) * Z2)   # correlated increment
X_{n+1} = X_n + mu1 * X_n * dt + sigma1 * X_n * dW1
Y_{n+1} = Y_n + mu2 * Y_n * dt + sigma2 * Y_n * dW2
```

## Implementation Notes

- Introduce correlation via Cholesky on the increments: generate two independent
  standard normals per step, then form `dW2 = rho*dW1 + sqrt(1 - rho**2)*dW2_indep`.
  X and Y are stepped independently after generating the correlated increments.
- rho = 0.95 is near-singular, so the Cholesky factor is ill-conditioned. This does
  NOT change the per-component marginal moments (each component is still exact GBM),
  so the analytic targets above are the standard GBM moments and are independent of rho.
- Elevated volatility sigma = 0.30 widens the terminal lognormal distribution, which
  increases Monte Carlo variance error. 50000 paths are needed to hit the 10% variance
  tolerance; the solver may increase num_paths or refine dt if the evaluator reports the
  variance error is above threshold.
- GBM components are positive by construction; no positivity guard is strictly required,
  but avoid any log/sqrt of the state (none needed for GBM).

## Results

Ran with num_paths=50000, dt=0.01, T=1.0, seed=42.

Empirical mean:     [1.1047040849473568, 1.1051725040012792]
Empirical variance: [0.11436079244694165, 0.11497423981989724]

## Evaluation

(Filled by evaluator)

<review score=10>

**Score: 10/10 — Done**

### Numerical Accuracy

Component X:
- mean_relative_error:     0.0004  (PASS)
- variance_relative_error: 0.0058  (PASS)

Component Y:
- mean_relative_error:     0.0000  (PASS)
- variance_relative_error: 0.0004  (PASS)

- Overall: PASS ✓

### Feedback for solver
- None

</review>
