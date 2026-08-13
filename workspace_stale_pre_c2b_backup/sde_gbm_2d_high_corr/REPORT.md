# Report: sde_gbm_2d_high_corr

## Problem

- **Type:** SDE
- **Family:** 2D correlated geometric Brownian motion (`gbm_2d_correlated`), state dimension 2, multiplicative noise, linear, non-stiff.
- **Parameters:** mu1 = mu2 = 0.10, sigma1 = sigma2 = 0.30, rho = 0.95, X_0 = Y_0 = 1.0, T = 1.0.
- **Note:** The "near-singular correlation" / "ill-conditioned Cholesky" framing is a numerical-difficulty flag. The per-component marginals are standard GBM and their moments are independent of rho, so accuracy is judged against the exact marginal GBM mean/variance.

## Plans

### 1-euler-maruyama — score 10 (1 iteration)

Vector Euler-Maruyama Monte Carlo, dt=0.01, 50000 paths, seed=42, T=1.0. Correlation introduced via Cholesky of the increments: `dW2 = sqrt(dt) * (rho*Z1 + sqrt(1-rho^2)*Z2)`; the two GBM components are then stepped independently.

Last-run metrics:
- Empirical mean: [1.10470, 1.10517] vs analytic exp(0.10) ≈ 1.10517
- Empirical variance: [0.11436, 0.11497] vs analytic exp(0.20)*(exp(0.09)-1) ≈ 0.11462
- Component X: mean_rel_err 0.0004, variance_rel_err 0.0058 (both PASS)
- Component Y: mean_rel_err 0.0000, variance_rel_err 0.0004 (both PASS)

Errors are at least an order of magnitude below their thresholds (mean < 5%, variance < 10%).

## Best Plan

**1-euler-maruyama** — the only plan (Milstein is not well-defined for multi-D noise without Lévy areas) and it reached score 10 on the first cycle. Vector EM with Cholesky-correlated increments reproduces both marginal GBM moments to well within tolerance; the high correlation rho=0.95 does not degrade marginal accuracy.

## Remaining Issues

None. The single plan reached the terminal score (10). No plans failed.
