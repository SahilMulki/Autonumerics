# Report: sde_gbm_2d_correlated

## Problem

- **Type:** SDE
- **Family:** 2D correlated Geometric Brownian Motion (multiplicative noise)
- **System:** `dX = mu1·X·dt + sigma1·X·dW1`, `dY = mu2·Y·dt + sigma2·Y·dW2`, with `corr(dW1, dW2) = rho`
- **Parameters:** mu1=0.10, sigma1=0.20, mu2=0.15, sigma2=0.25, rho=0.60, X_0=Y_0=1.0, T=1.0
- **Analytic reference:** standard 1D GBM marginal moments (independent of rho — correlation only affects the X–Y cross-correlation, not the per-component marginals the evaluator checks).

## Plans

### 1-euler-maruyama — **score 10/10, iter 1** ✅

- **Scheme:** Euler-Maruyama Monte Carlo, dt=0.01, 50,000 paths, seed=42, T=1.0.
- **Correlation handling:** correlated Brownian increments via Cholesky factor — `dW1 = sqrt(dt)·Z1`, `dW2 = sqrt(dt)·(rho·Z1 + sqrt(1-rho²)·Z2)`, Z1,Z2 ~ i.i.d. N(0,1).
- **Milstein:** not applicable — multi-dimensional multiplicative noise requires the Lévy area; EM only (`milstein_eligible = false`).

**Final metrics (empirical vs analytic):**

| Component | Empirical mean | Empirical variance | Mean rel. err | Var rel. err |
|---|---|---|---|---|
| X | 1.1048 | 0.04963 | 0.0003 | 0.0042 |
| Y | 1.1623 | 0.08757 | 0.0004 | 0.0058 |

All errors well within the thresholds (mean < 5%, variance < 10%).

## Recommendation

**Best plan: `1-euler-maruyama`** — the only and terminal plan. Achieved score 10/10 on the first solver↔evaluator cycle. The Cholesky-based correlated-increment construction is correct, and both marginal distributions match their analytic GBM moments with relative errors below 1%. No refinement needed.

## Outstanding issues

None. The single plan reached score 10; no plans failed.
