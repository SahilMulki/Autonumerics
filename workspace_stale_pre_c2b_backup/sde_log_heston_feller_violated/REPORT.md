# Report: sde_log_heston_feller_violated

## Problem Type

SDE — 2D log-Heston stochastic volatility system (log-price X, instantaneous variance Y), with the Feller condition violated (2a = 0.2 < sigma^2 = 1.0), so Y hits 0 frequently over [0, 1].

## Plans

### 1-euler-maruyama — Score: 10/10 (1 iteration)

Full-truncation Euler-Maruyama, dt = 0.001 (1000 steps), 50,000 paths, seed = 42. Positivity guard `Y_pos = max(Y, 0)` applied before every `sqrt(Y)` in both the X diffusion and Y drift/diffusion, with the same `dW1` shared between the X and Y updates to preserve the rho correlation.

Key metrics (terminal moments at T = 1.0):

| Quantity | Empirical | Exact | Relative error |
|---|---|---|---|
| mean_X     | -0.002425 | 0.0 (skipped, near-zero) | n/a |
| variance_X | 0.137350  | 0.137311 | 0.03% |
| mean_Y     | 0.100354  | 0.1 | 0.35% |
| variance_Y | 0.042205  | 0.043233 | 2.38% |

All errors well within thresholds (variance_rel_err < 10%, mean_rel_err < 5%). No NaNs or non-finite values despite frequent Y=0 excursions — the full-truncation guard handled the Feller-violated regime cleanly on the first attempt.

## Best Plan

**1-euler-maruyama** — the only plan proposed, and it reached score 10 on the first solver↔evaluator cycle with comfortable margin below every threshold. No refinement (e.g., smaller dt or exact CIR sampling) was needed.

## Outstanding Issues

None. No plans failed to reach score 10.
