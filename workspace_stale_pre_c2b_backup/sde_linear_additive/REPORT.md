# Report: sde_linear_additive

## Problem

- **Type**: SDE — linear SDE with additive noise (`linear_sde_additive`)
- **Equation**: `dX = (a + b·X) dt + c dW` with `a=2.0, b=-1.0, c=0.5`, i.e. `dX = (2 - X) dt + 0.5 dW`
- **Initial condition**: `X(0) = 0.0`, horizon `T = 1.0`
- **Analytic moments at T=1**:
  - mean = `exp(b·t)·(X_0 + a/b) - a/b` ≈ **1.264241**
  - variance = `c²/(2b)·(exp(2b·t) - 1)` ≈ **0.108083**

Because the diffusion coefficient `g = c` is constant (`dg/dX = 0`), the Milstein correction vanishes and reduces to Euler-Maruyama. A single scheme was therefore proposed.

## Plans

| Plan | Scheme | Final score | Iters | Mean rel. err | Var rel. err |
|---|---|---|---|---|---|
| 1-euler-maruyama | Euler-Maruyama MC (dt=0.01, 50k paths, seed=42) | **10** | 1 | 0.53% | 2.22% |

**Last Results (1-euler-maruyama):**
- Empirical mean 1.270954 vs exact 1.264241 → rel. err 0.53% (threshold 5%) — PASS
- Empirical variance 0.110484 vs exact 0.108083 → rel. err 2.22% (threshold 10%) — PASS

## Best Plan

**1-euler-maruyama** — the only plan, reaching score 10 on the first iteration. Both moment errors are comfortably within thresholds. For additive-noise linear SDEs, Euler-Maruyama is strongly convergent order 1.0 (equal to Milstein), so no higher-order scheme is warranted.

## Failures / Remaining Errors

None. All plans reached score 10.
