# Report: CIR Process with Feller Condition Violated

## Problem

- **Type:** SDE — Cox-Ingersoll-Ross (CIR), scalar (1D), multiplicative nonlinear diffusion.
- **Equation:** `dX = kappa*(theta - X) dt + sigma*sqrt(X) dW`, with `kappa=1.0, theta=0.5, sigma=2.0, X_0=0.5`, `T=1.0`.
- **Key challenge:** The Feller condition is **violated** (`2*kappa*theta = 1.0 < sigma^2 = 4.0`), so `X` hits zero frequently. A positivity guard (full truncation, `np.maximum(X, 0.0)` before `sqrt`) is mandatory to avoid NaNs. Analytic moments remain valid: exact mean = 0.5 at T=1.
- **Thresholds:** variance_rel_err < 10%, mean_rel_err < 5%.

## Plans

Both plans used dt=0.01, num_paths=50000, seed=42, with the mandatory full-truncation positivity guard.

| Plan | Scheme | Mean rel err | Variance rel err | Score | Iters |
|---|---|---|---|---|---|
| 1-euler-maruyama | Euler-Maruyama (strong order 0.5) | 2.59% | 6.35% | **10** | 1 |
| 2-milstein | Milstein (strong order 1.0), constant CIR correction | 2.24% | 6.94% | **10** | 1 |

Both schemes passed both thresholds on the first solver↔evaluator cycle. Milstein uses the simplified constant correction `(sigma^2/4)*(dW^2 - dt)`, which avoids division by `sqrt(X)` and is therefore stable at the frequently-touched zero boundary.

## Recommendation

**Best plan: `2-milstein`.**

Both plans tie on score (10) and iteration count (1). Milstein is recommended because it is the higher strong-order scheme (1.0 vs 0.5) and its constant correction term is well-defined at the zero boundary — a meaningful advantage precisely because the Feller condition is violated and zero is hit often. Its mean error (2.24%) is slightly lower than EM's (2.59%). Euler-Maruyama is an equally valid, slightly cheaper fallback that also achieves score 10 at these settings.

## Failures

None. No plan failed to reach score 10.
