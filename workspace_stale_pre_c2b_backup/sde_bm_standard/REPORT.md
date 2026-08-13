# Report: sde_bm_standard

## Problem

- **Type**: SDE (stochastic differential equation)
- **Family**: `bm_standard` — standard Ito Brownian motion, `dX = dW`, `X(0) = 0` on `[0, 1]`
- **State dimension**: 1 (scalar), additive unit noise, non-stiff
- **Analytic moments at T = 1**: exact_mean = 0.0, exact_variance = t = 1.0
- **Scoring**: mean check skipped (|exact_mean| < 0.01); evaluation driven by `variance_rel_err < 10%`

## Plans

### 1-euler-maruyama — Score 10/10, 1 iteration ✓

- **Scheme**: Euler-Maruyama Monte Carlo, dt = 0.01, 50,000 paths, seed = 42, T = 1.0
- **Rationale**: Additive-noise scalar case, so EM is exact-in-distribution (f(X)=0, g(X)=1, each step is `X += sqrt(dt)·Z`). Milstein skipped — `dg/dX = 0` makes it identical to EM.
- **Key metrics**:
  - Empirical mean: 0.008598 (mean check skipped)
  - Empirical variance: 1.014938 vs exact 1.0 → variance relative error ≈ 1.49% (PASS, well within 10%)
- The residual variance deviation is pure Monte Carlo sampling noise, not discretization bias.

## Best Plan

**1-euler-maruyama** — the only plan and a clean pass. Reached score 10 on the first iteration with variance relative error of 1.49%. Euler-Maruyama is the natural and exact-in-distribution choice for standard Brownian motion; no refinement was needed.

## Failures

None. All plans reached score 10.
