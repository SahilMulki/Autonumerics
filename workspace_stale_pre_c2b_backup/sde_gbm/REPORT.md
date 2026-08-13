# Autonumerics Report — sde_gbm

## Problem

- **Type:** SDE
- **Family:** Geometric Brownian Motion (scalar, multiplicative noise, linear, non-stiff)
- **Equation:** `dX = mu*X*dt + sigma*X*dW`, `X(0) = 1.0`, over `t ∈ [0, 1]`
- **Parameters:** X_0 = 1.0, mu = 0.1, sigma = 0.2
- **Analytic moments at T=1:** mean = exp(0.1) ≈ 1.105171, variance ≈ 0.049867
- **Thresholds:** mean_rel_err < 5%, variance_rel_err < 10%

## Plans

### 1-euler-maruyama — Score 10/10 (iter 1)
- **Scheme:** Euler-Maruyama, `X_{n+1} = X_n + mu*X_n*dt + sigma*X_n*dW_n` (strong order 0.5, weak order 1.0)
- **Hyperparameters:** dt = 0.01, 50000 paths, seed 42
- **Results:** empirical mean = 1.107361, empirical variance = 0.050723
- **Errors:** mean_rel_err = 0.20% (PASS), variance_rel_err = 1.76% (PASS)

### 2-milstein — Score 10/10 (iter 1)
- **Scheme:** Milstein, adds correction `0.5*sigma^2*X_n*(dW_n^2 - dt)` (strong order 1.0)
- **Hyperparameters:** dt = 0.01, 50000 paths, seed 42
- **Results:** empirical mean = 1.107348, empirical variance = 0.050730
- **Errors:** mean_rel_err = 0.20% (PASS), variance_rel_err = 1.77% (PASS)

## Recommendation

**Best plan: `1-euler-maruyama`**

Both schemes reached the terminal score of 10 on the first cycle with effectively identical moment accuracy (mean error 0.20%, variance error ~1.77% for both). Since the evaluator checks weak-sense terminal moments — where both Euler-Maruyama and Milstein share weak order 1.0 — the Milstein strong-order-1.0 correction confers no measurable benefit here while adding computation. Euler-Maruyama is therefore recommended as the simplest scheme that meets all thresholds. Milstein remains the preferred choice if pathwise (strong) accuracy were the objective rather than terminal moments.

## Failures

None. Both plans passed all thresholds on iteration 1.
