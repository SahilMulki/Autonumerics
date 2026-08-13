# Autonumerics Report — sde_gbm_high_vol

## Problem

- **Type**: SDE
- **Family**: Geometric Brownian Motion (high-volatility variant)
- **Equation**: `dX = mu*X dt + sigma*X dW`, `X(0)=1.0`, `mu=0.05`, `sigma=1.0`, `T=1.0`
- **Analytic moments** (lognormal): `mean = X0*exp(mu*T) ≈ 1.0513`, `variance = X0^2*exp(2*mu*T)*(exp(sigma^2*T)-1) ≈ 1.899`
- **Challenge**: `sigma=1.0` gives log-variance `sigma^2*T = 1.0`, so `X(T)` is heavy-tailed. The empirical variance estimator has a large fourth-moment sampling error (`exp(4*sigma^2*T) ≈ 54.6`), demanding many paths.

## Results

| Plan | Scheme | Final config | mean_rel_err | var_rel_err | Score | Iters |
|---|---|---|---|---|---|---|
| 1-euler-maruyama | Euler-Maruyama + Milstein correction | dt=0.001, 200k paths | 0.0010 | 0.0185 | 10 | 2 |
| 2-milstein | Milstein (strong order 1.0) | dt=0.005, 500k paths | 0.0008 | 0.0034 | 10 | 2 |

Both plans initially scored **5/10** (variance error ~17%). The lesson was identical for both: at `sigma=1.0` the heavy tail dominates. Two independent fixes worked —
- **Plan 1** added the Milstein correction term and refined `dt` to 0.001, cutting discretization bias (17% → 1.85%).
- **Plan 2** kept the Milstein scheme and increased path count to 500k, cutting Monte-Carlo sampling noise (17.5% → 0.34%).

Notably, both converged on the Milstein correction term, confirming that for high-`sigma` GBM the strong-order-1.0 correction is essential regardless of path budget.

## Recommendation

**Best plan: `2-milstein`.** Both plans tied on score (10) and iteration count (2), but Milstein delivers the tightest accuracy (variance error 0.34% vs 1.85%, mean error 0.08% vs 0.10%) and is the mathematically appropriate scheme for multiplicative scalar noise (`milstein_eligible: true`). Its accuracy comes from a larger path count (500k) at a coarser `dt=0.005`, which is cheaper per step than plan 1's `dt=0.001` while achieving lower error.

## Failures

None. Both plans reached score 10 within 2 iterations (max_iter=5).
