---
id: 1
plan_slug: euler-maruyama
scheme: euler-maruyama
strategy: Full-truncation Euler-Maruyama on the 2D log-Heston system with a mandatory positivity guard on the variance Y before every square root, and small dt to control the bias from frequent Y=0 excursions.
---

## SDE Reference

Two-dimensional log-Heston system (Ito), state (X, Y) with X = log-price, Y = instantaneous variance:

```
dX = (r - Y/2) dt + sqrt(Y) * (rho*dW1 + sqrt(1 - rho**2)*dW2),   X(0) = 0.0
dY = (a - b*Y) dt + sigma*sqrt(Y)*dW1,                            Y(0) = 0.1
```

W1 drives the variance Y and is correlated (through rho) with the log-price X.
W2 is the independent second Brownian factor of the log-price.

**Parameters**:
- X_0 = 0.0
- Y_0 = 0.1
- r = 0.05
- a = 0.1
- b = 1.0
- sigma = 1.0
- rho = -0.9

**Time interval**: T0 = 0.0, T = 1.0

**Feller condition**: 2a = 0.2 < sigma**2 = 1.0 → VIOLATED. Y reaches 0 frequently.

**Exact terminal moments (at T = 1.0)**:
- mean_X     = 0.0        (near-zero → mean check on X is skipped)
- variance_X = 0.13731143072354426
- mean_Y     = 0.1
- variance_Y = 0.04323323583816936

## Numerical Scheme

- **Method**: Euler-Maruyama (full-truncation variant for the CIR variance component)
- **dt**: 0.001
- **num_paths**: 50000
- **T**: 1.0
- **Milstein correction**: N/A (state_dimension == 2; multi-D Milstein requires Lévy areas and is not used)

**Vector layout**: state array shape `(num_paths, 2)`, column 0 = X, column 1 = Y.
All paths stepped simultaneously with NumPy broadcasting.

**Per-step update** (with Y_pos = np.maximum(Y, 0.0)):
```
Z1, Z2 ~ N(0,1)                     # two independent standard normals per path
dW1 = sqrt(dt) * Z1
dW2 = sqrt(dt) * Z2
sqrtY = sqrt(Y_pos)                 # positivity guard BEFORE every sqrt

X_{n+1} = X_n + (r - Y_pos/2) * dt + sqrtY * (rho*dW1 + sqrt(1 - rho**2)*dW2)
Y_{n+1} = Y_n + (a - b*Y_pos) * dt + sigma * sqrtY * dW1
```
Note the SAME dW1 enters both the Y update and the correlated part of the X update — this is what couples X to the variance factor. Do not draw a fresh normal for the X drift's variance term.

## Implementation Notes

- **Positivity guard is MANDATORY.** Compute `Y_pos = np.maximum(Y, 0.0)` and use `sqrt(Y_pos)` everywhere (both the X diffusion and the Y diffusion). A naive Euler step on `sqrt(Y)` drives Y negative on the frequent Y=0 excursions and produces NaNs. This is the full-truncation scheme: the drift term `(a - b*Y_pos)` also uses the truncated value so that when Y is at/near 0 the drift is the pure mean-reversion push `a` back into the positive region.
- **Feller violated → small dt required.** Because Y hits 0 frequently, the biased-Euler variance of Y converges slowly; dt = 0.001 is chosen (per spec) to keep the discretization bias in variance_Y and variance_X inside the 10% threshold. If the evaluator reports variance_rel_err above threshold, the solver may refine dt downward (e.g. 0.0005).
- **Ninomiya-Victoir splitting is INVALID here** (it needs sigma**2 <= 4a = 0.4, but sigma**2 = 1.0). Do not use it.
- **Alternative (bias-free) treatment if EM struggles**: exact noncentral chi-square sampling of the CIR variance Y, combined with a broadened/exact conditional scheme for X. This removes the discretization bias entirely but is more involved; keep it as a fallback only if full-truncation EM cannot reach the variance threshold even at small dt.
- **Mean checks**: E[X](T) = 0 exactly, so |exact_mean_X| < 0.01 → the mean check on X is skipped (variance-only). E[Y](T) = 0.1 is a genuine nonzero mean → mean_Y must satisfy the 5% relative-error check.
- **Return**: `terminal_paths` shape `(num_paths, 2)` — column 0 = X, column 1 = Y. Also report empirical means and variances of each column.

## Results

Ran `solve_sde(num_paths=50000, dt=0.001, T=1.0, seed=42)` (full-truncation Euler-Maruyama, Nt = 1000 steps).

| Quantity | Empirical | Exact | Relative error |
|---|---|---|---|
| mean_X     | -0.002425 | 0.0 (near-zero, check skipped) | n/a |
| variance_X | 0.137350  | 0.13731143072354426 | 0.028% |
| mean_Y     | 0.100354  | 0.1 | 0.354% |
| variance_Y | 0.042205  | 0.04323323583816936 | 2.38% |

All checks well within the evaluation thresholds (variance_rel_err < 10%, mean_rel_err < 5% for Y). No positivity-guard failures or non-finite values observed. Runtime ~0.9s.

## Evaluation

(Filled by evaluator)

<review score=10>

Score: 10/10 — Done

**Score: 10/10**

### Numerical Accuracy

Component X:
- mean_relative_error:     n/a (skipped, |exact_mean_X| = 0.0 < 0.01)
- variance_relative_error: 0.0003  (PASS)

Component Y:
- mean_relative_error:     0.0035  (PASS)
- variance_relative_error: 0.0238  (PASS)

- Overall: PASS ✓

Ran `solve_sde(num_paths=50000, dt=0.001, T=1.0, seed=42)`. All outputs finite, no NaNs despite frequent Y=0 excursions (Feller condition violated: 2a=0.2 < sigma^2=1.0). The full-truncation positivity guard (Y_pos = max(Y,0) before every sqrt, applied consistently in both the X diffusion term and the Y drift/diffusion) is working correctly. Both variance errors are well under the 10% threshold and mean_Y error is well under the 5% threshold, with comfortable margin — no need for smaller dt.

### Feedback for solver
None.

</review>
